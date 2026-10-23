"""Evaluate base, LoRA or GGUF invoice extraction with auditable outputs."""

import argparse
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from evaluate_baseline import FIELDS, TOLERANCE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from schema import Invoice, safe_invoice_payload

TEST_PATH = ROOT / "data" / "test.jsonl"
LOCALE_CURRENCIES = {"fr-FR": "EUR", "en-GB": "GBP", "en-US": "USD"}
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"

# Conserve l'instruction d'entraînement et ajoute les garde-fous d'inférence.
SYSTEM_PROMPT = (
    "Tu es un extracteur de données. Réponds uniquement par un objet JSON valide.\n"
    "Ne jamais confondre les en-têtes de colonnes ou les libellés de sous-totaux "
    "(ex: 'Sous-total') avec la description d'un article. "
    "Copier littéralement le texte exact des descriptions sans omettre de lettres. "
    "Préserve les quantités décimales et associe chaque quantité à son prix unitaire "
    "sur la même ligne, sans mélanger les colonnes. "
    "7,2 devient 7.2 et 5,1 devient 5.1, jamais 72 ou 51. "
    "Conserve le nom du fournisseur ; normalise seulement les espaces et la typographie "
    "certaine de Ltd, sans deviner un suffixe OCR ambigu. "
    "Pour un champ illisible ou irrécupérable, renvoie null, jamais un chiffre inventé "
    "ou une valeur par défaut. "
    "Exemple fictif de lecture :\n"
    "Description | Quantité | Prix unitaire HT | Montant HT\n"
    "Maintenance préventive | 7,2 | 95,07 | 684,50\n"
    "Sous-total | 684,50\n"
    'line_items attendu : [{"description": "Maintenance préventive", "quantity": 7.2, '
    '"unit_price": 95.07, "total_ht": 684.50}]. '
    "Ni l'en-tête ni 'Sous-total' ne sont des articles. "
    "Ne recopie aucune valeur de cet exemple fictif dans la facture réelle."
)
INSTRUCTION = (
    "Extrais les champs de la facture suivante au format JSON : supplier_name, "
    "invoice_date (AAAA-MM-JJ), total_ht, total_ttc, "
    "line_items (description, quantity, unit_price, total_ht)."
)


def build_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\n{row['input']}"},
    ]


def load_model(model_path: Path, n_ctx: int = 4096):
    from llama_cpp import Llama

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_batch=128,
        n_ubatch=128,
        n_gpu_layers=-1,
        verbose=False,
    )


def generate(llm, row: dict, max_tokens: int = 2048) -> dict:
    result = llm.create_chat_completion(messages=build_messages(row), temperature=0.0, max_tokens=max_tokens)
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("GGUF returned no text content")
    completion_tokens = (result.get("usage") or {}).get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(llm.tokenize(content.encode("utf-8"), add_bos=False, special=True))
    return {
        "raw": content, "completion_tokens": completion_tokens,
        "finish_reason": result["choices"][0].get("finish_reason"),
        "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
    }


def check_adapter(adapter_path: Path) -> dict:
    for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        path = adapter_path / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty adapter artifact: {path}")
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or not config.get("base_model_name_or_path"):
        raise ValueError("Expected a LoRA adapter with base_model_name_or_path")
    return config


class BaseInference:
    def __init__(self, device: str, load_in_4bit: bool, n_ctx: int, max_tokens: int,
                 *, model_name: str = BASE_MODEL, revision: str | None = None,
                 tokenizer_path: str | None = None):
        if load_in_4bit and device != "cuda":
            raise ValueError("4-bit inference requires CUDA in this pipeline")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a GPU runtime; CPU requires enough RAM for the base model.")
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32
        kwargs = {"device_map": {"": device}, "torch_dtype": dtype, "trust_remote_code": False}
        if revision:
            kwargs["revision"] = revision
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if getattr(self.model, "peft_config", None):
            raise ValueError("Base-model loading unexpectedly attached PEFT adapters")
        self.model.eval()
        tokenizer_kwargs = {"trust_remote_code": False}
        if tokenizer_path is None and revision:
            tokenizer_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_name, **tokenizer_kwargs)
        if not self.tokenizer.chat_template:
            raise ValueError("The tokenizer has no chat template")
        self.torch = torch
        self.device = device
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.metadata = {
            "model": "qwen_base", "adapter_applied": False,
            "base_model": model_name, "base_model_revision": revision,
            "resolved_base_revision": getattr(self.model.config, "_commit_hash", None),
            "device": device, "load_in_4bit": load_in_4bit,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        }

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def predict(self, row: dict, *, messages: list[dict] | None = None) -> dict:
        inputs = self.tokenizer.apply_chat_template(
            build_messages(row) if messages is None else messages,
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        model_context = getattr(self.model.config, "max_position_embeddings", self.n_ctx)
        context = min(self.n_ctx, model_context)
        if prompt_tokens + self.max_tokens > context:
            raise ValueError(
                f"{row.get('id', 'invoice')}: prompt ({prompt_tokens}) + generation budget "
                f"({self.max_tokens}) exceeds context ({context}); adjust explicit limits, no truncation applied"
            )
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False, use_cache=True,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
            )
        generated = output[0][prompt_tokens:]
        count = len(generated)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        stopped = bool(count and int(generated[-1]) in eos_ids)
        return {
            "raw": self.tokenizer.decode(generated, skip_special_tokens=True),
            "completion_tokens": count, "prompt_tokens": prompt_tokens,
            "finish_reason": "stop" if stopped else "length" if count >= self.max_tokens else "other",
        }


class LoRAInference(BaseInference):
    def __init__(self, adapter_path: Path, device: str, load_in_4bit: bool,
                 n_ctx: int, max_tokens: int):
        config = check_adapter(adapter_path)
        manifest_path = adapter_path.parent / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        trained_context = manifest.get("training_args", {}).get("max_length")
        if trained_context and n_ctx > trained_context:
            raise ValueError(f"Requested context {n_ctx} exceeds trained context {trained_context}")
        same_base = manifest.get("base_model") == config["base_model_name_or_path"]
        if manifest.get("base_model") and not same_base:
            logging.warning(
                "Training model name %s differs from PEFT's saved base %s (Unsloth may resolve aliases). "
                "Loading the PEFT-declared base; verify the training provenance.",
                manifest["base_model"], config["base_model_name_or_path"],
            )
        revision = config.get("revision") or (manifest.get("base_model_revision") if same_base else None)
        super().__init__(
            device, load_in_4bit, n_ctx, max_tokens,
            model_name=config["base_model_name_or_path"], revision=revision,
            tokenizer_path=str(adapter_path),
        )
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, str(adapter_path), is_trainable=False)
        self.model.eval()
        self.metadata.update({
            "model": "finetuned_lora", "adapter_applied": True,
            "adapter_path": str(adapter_path),
            "adapter_config_sha256": file_hash(adapter_path / "adapter_config.json"),
            "adapter_weights_sha256": file_hash(adapter_path / "adapter_model.safetensors"),
            "training_manifest": manifest,
        })


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_training_provenance(adapter_path: Path, test_path: Path) -> list[str]:
    manifest_path = adapter_path.parent / "training_manifest.json"
    if not manifest_path.is_file():
        logging.warning("No training manifest beside %s; training dataset provenance cannot be verified", adapter_path)
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "international-v3":
        raise ValueError("Training manifest must identify international-v3")
    if manifest.get("status") != "completed":
        raise ValueError("Training manifest does not identify a completed run")
    checked = []
    for name in ("train", "val"):
        path = test_path.parent / f"{name}.jsonl"
        expected = manifest.get("dataset_sha256", {}).get(name)
        if path.is_file() and expected:
            if file_hash(path) != expected:
                raise ValueError(f"{name}.jsonl differs from the data used to train this adapter")
            checked.append(name)
        else:
            logging.warning("Cannot verify %s dataset hash: local split or manifest hash missing", name)
    return checked


def parse_object(text: str) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique_keys(pairs: list[tuple]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def validate_invoice_json(raw: str, *, arithmetic_warnings: list[str] | None = None) -> dict:
    invoice = Invoice.model_validate(
        parse_object(raw),
        context={"validate_arithmetic": arithmetic_warnings is None},
    )
    payload = invoice.model_dump(mode="json")
    json.dumps(payload, allow_nan=False)
    if arithmetic_warnings is not None:
        arithmetic_warnings.extend(invoice.arithmetic_warnings())
    return payload


def finalize_invoice_json(raw: str, *, warnings: list[str],
                          initial_error: str | None = None) -> dict:
    source = parse_object(raw)
    if initial_error is not None:
        try:
            Invoice.model_validate(source)
            state = "corrigée lors de la seconde passe"
        except ValidationError:
            state = "détectée, non résolue ou non vérifiable après la seconde passe"
        warnings.append(f"Passe 1 — anomalie {state} :\n{initial_error}")
    payload = safe_invoice_payload(source, warnings)
    json.dumps(payload, allow_nan=False)
    return payload


def correction_messages(messages: list[dict], raw: str, error: Exception,
                        warnings: list[str]) -> list[dict]:
    arithmetic_focus = ""
    if warnings:
        arithmetic_focus = (
            "Erreur arithmétique détectée.\n"
            "Priorité de correction arithmétique : relis d'abord les champs quantity et "
            "unit_price des lignes concernées dans le texte de la facture réelle. "
            "Corrige en priorité les quantités et prix unitaires mal transcrits, uniquement "
            "si les valeurs sources le confirment. Respecte exactement la virgule décimale : "
            "7,2 devient 7.2 (jamais 142 ou 72) et 4,2 devient 4.2 (jamais 42). "
            "Ne déduis pas ces champs par division des totaux. Vérifie ensuite les montants "
            "de ligne et le total HT global en les relisant dans la facture ; ne les recalcule "
            "pas pour masquer une erreur de quantité ou de prix.\n"
            "Corrige la quantité si elle est mal transcrite et ré-extrais la ligne exacte, "
            "avec sa description complète. Le montant HT d'une ligne se compare au produit "
            "quantity * unit_price ; seul le cumul des lignes se compare au sous-total "
            "HT de la facture.\n"
            "Anomalies arithmétiques à examiner :\n"
            + "\n".join(warnings)
            + "\n\n"
        )
    return [
        *messages,
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "La précédente extraction a échoué à la validation du schéma Pydantic "
                f"pour les raisons suivantes :\n{error}\n\n"
                f"{arithmetic_focus}"
                "Respecte strictement les valeurs de la facture réelle ci-dessus. Ne modifie pas les "
                "montants pour simplement faire passer la validation et n'utilise pas "
                "les valeurs de l'exemple fictif. Si un champ reste illisible, ambigu ou "
                "irrécupérable, utilise null : ne conserve pas un chiffre ou un texte inventé. "
                "Préserve 5,1 en 5.1 et les décimales, sans supprimer le séparateur. "
                "Pour le fournisseur, ne remplace pas un suffixe OCR incertain par Ltd "
                "sans confirmation dans la source. "
                "Réponds uniquement par le JSON complet corrigé.\n\n"
                "Relis attentivement la facture fournie (en faisant particulièrement attention "
                "aux décimales des prix unitaires, aux quantités et aux totaux de lignes), "
                "corrige l'erreur de calcul ou d'extraction, et renvoie un JSON rigoureusement valide."
            ),
        },
    ]


def predict_with_correction(inference: BaseInference, row: dict) -> dict:
    messages = build_messages(row)
    initial = inference.predict(row, messages=messages)
    attempts = [initial]
    selected = initial
    errors = []
    operational_error = None
    correction_attempted = False
    initial_error = None
    if initial["finish_reason"] == "stop":
        try:
            validate_invoice_json(initial["raw"])
        except (ValidationError, ValueError) as exc:
            initial_error = str(exc)
            correction_attempted = True
            logging.warning("%s: validation failed; attempting one correction: %s", row["id"], exc)
            errors.append(str(exc))
            warnings = []
            try:
                validate_invoice_json(initial["raw"], arithmetic_warnings=warnings)
                initial_object_valid = True
            except (ValidationError, ValueError) as structure_error:
                logging.debug("%s: no structurally valid fallback: %s", row["id"], structure_error)
                try:
                    parse_object(initial["raw"])
                    initial_object_valid = True
                except ValueError:
                    initial_object_valid = False
            initial_error = "\n".join([initial_error, *warnings])
            messages = correction_messages(messages, initial["raw"], exc, warnings)
            try:
                corrected = inference.predict(row, messages=messages)
                attempts.append(corrected)
                if corrected["finish_reason"] == "stop":
                    parse_object(corrected["raw"])
                    selected = corrected
                else:
                    operational_error = f"Correction incomplete: {corrected['finish_reason']}"
                    logging.warning("%s: %s", row["id"], operational_error)
                    if not initial_object_valid:
                        selected = corrected
            except (OSError, ValueError, RuntimeError, ImportError) as correction_error:
                logging.warning("%s: correction failed: %s", row["id"], correction_error)
                errors.append(str(correction_error))
                operational_error = str(correction_error)
                if len(attempts) > 1 and not initial_object_valid:
                    selected = attempts[-1]
    warnings = []
    result_warnings = []
    payload = None
    try:
        validate_invoice_json(selected["raw"], arithmetic_warnings=warnings)
    except (ValidationError, ValueError) as final_error:
        errors.append(str(final_error))
    if selected["finish_reason"] == "stop":
        try:
            payload = finalize_invoice_json(
                selected["raw"], warnings=result_warnings, initial_error=initial_error,
            )
        except ValueError as final_error:
            if initial_error:
                result_warnings.append(f"Passe 1 — anomalie non résolue :\n{initial_error}")
            result_warnings.append(f"JSON inutilisable : {final_error}")
    if operational_error:
        result_warnings.append(f"Auto-correction impossible : {operational_error}")
    return {
        **selected, "attempts": attempts, "attempt_count": 2 if correction_attempted else 1,
        "correction_attempted": correction_attempted,
        "validation_errors": errors, "arithmetic_warnings": warnings,
        "payload": payload, "warnings": result_warnings,
        "operational_error": operational_error,
        "completion_tokens": sum(attempt["completion_tokens"] for attempt in attempts),
        "prompt_tokens": sum(attempt["prompt_tokens"] for attempt in attempts),
    }


def extract_json(raw: str) -> dict | None:
    """Return the first top-level JSON object found in the model output, or None."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return parse_object(raw[start:end + 1])
    except ValueError:
        return None


def score(raw: str, gold: dict) -> dict:
    parse_error = None
    try:
        payload = parse_object(raw)
        strict_json = True
    except ValueError as exc:
        strict_json = False
        parse_error = str(exc)
        payload = extract_json(raw)
    valid = False
    schema_error = None
    if payload is not None:
        try:
            invoice = Invoice.model_validate(payload)
            numbers = [invoice.total_ht, invoice.total_ttc]
            numbers.extend(value for item in invoice.line_items for value in (item.quantity, item.unit_price, item.total_ht))
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError("Non-finite invoice amounts")
            valid = True
        except (ValidationError, ValueError) as exc:
            schema_error = str(exc)

    def numeric_match(pred, expected) -> bool:
        return (
            isinstance(pred, (int, float)) and not isinstance(pred, bool)
            and (isinstance(pred, int) or math.isfinite(pred))
            and abs(Decimal(str(pred)) - Decimal(str(expected))) <= Decimal(str(TOLERANCE))
        )

    def matches(field: str, pred, expected) -> bool:
        if field in ("total_ht", "total_ttc"):
            return numeric_match(pred, expected)
        if field == "line_items":
            return isinstance(pred, list) and len(pred) == len(expected) and all(
                isinstance(item, dict) and item.get("description") == target["description"]
                and all(numeric_match(item.get(key), target[key]) for key in ("quantity", "unit_price", "total_ht"))
                for item, target in zip(pred, expected)
            )
        return pred == expected

    fields = {field: matches(field, (payload or {}).get(field), gold[field]) for field in FIELDS}

    recovered_exact = valid and all(fields.values())
    return {
        "valid": valid, "strict_json": strict_json,
        "recovered_json": payload is not None, "fields": fields,
        "exact": strict_json and recovered_exact, "recovered_exact": recovered_exact,
        "parse_error": parse_error, "schema_error": schema_error,
    }


def metric_rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def build_report(results: list[dict], model_file: str) -> dict:
    if not results:
        raise ValueError("Cannot evaluate an empty result set")

    def summary(subset: list[dict]) -> dict:
        accuracy = {field: metric_rate([r["fields"][field] for r in subset]) for field in FIELDS}
        return {
            "num_examples": len(subset),
            "json_validity_rate": metric_rate([r["strict_json"] for r in subset]),
            "schema_validity_rate": metric_rate([r["valid"] for r in subset]),
            "recovered_json_rate": metric_rate([r["recovered_json"] for r in subset]),
            "field_accuracy": accuracy,
            "mean_field_accuracy": sum(accuracy.values()) / len(FIELDS),
            "exact_match_rate": metric_rate([r["exact"] for r in subset]),
            "recovered_exact_match_rate": metric_rate([r["recovered_exact"] for r in subset]),
            "truncation_rate": metric_rate([r["finish_reason"] == "length" for r in subset]),
            "truncated_attempt_rate": metric_rate([
                any(attempt["finish_reason"] == "length" for attempt in r.get("attempts", [r]))
                for r in subset
            ]),
            "correction_rate": metric_rate([r.get("correction_attempted", False) for r in subset]),
            "initial_json_validity_rate": metric_rate([
                r.get("initial_score", r)["strict_json"] for r in subset
            ]),
            "initial_exact_match_rate": metric_rate([r.get("initial_score", r)["exact"] for r in subset]),
        }

    groups = {}
    for key in ("format", "locale", "currency", "template_id"):
        groups[f"per_{key.removesuffix('_id')}"] = {
            str(value): summary([r for r in results if r[key] == value])
            for value in sorted({r[key] for r in results}, key=str)
        }
    latencies_ms = [r["latency_ms"] for r in results]
    total_latency_seconds = sum(latencies_ms) / 1000
    completion_tokens_total = sum(r["completion_tokens"] for r in results)
    return {
        "model": "finetuned",
        "model_file": model_file,
        "split": "test",
        **summary(results),
        "latency_ms_mean": statistics.mean(latencies_ms),
        "latency_ms_median": statistics.median(latencies_ms),
        "completion_tokens_total": completion_tokens_total,
        "tokens_per_second": completion_tokens_total / total_latency_seconds if total_latency_seconds else 0.0,
        **groups,
    }


def load_test_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 60 or Counter(row["locale"] for row in rows) != dict.fromkeys(LOCALE_CURRENCIES, 20):
        raise ValueError("Expected the full v3 test split: 60 invoices, 20 per locale")
    if len({row["id"] for row in rows}) != len(rows) or len({row["input"] for row in rows}) != len(rows):
        raise ValueError("Duplicate test invoice IDs or inputs")
    for row in rows:
        if row["dataset_version"] != "international-v3" or row["currency"] != LOCALE_CURRENCIES[row["locale"]]:
            raise ValueError(f"Invalid version or currency for {row['id']}")
        if not isinstance(row["input"], str) or not row["input"].strip():
            raise ValueError(f"Empty input for {row['id']}")
        if set(row["output"]) != set(FIELDS) or not score(
            json.dumps(row["output"], allow_nan=False), row["output"],
        )["exact"]:
            raise ValueError(f"Invalid target schema or numeric types for {row['id']}")
    for name in ("train", "val"):
        sibling = path.parent / f"{name}.jsonl"
        if sibling.is_file():
            other = [json.loads(line) for line in sibling.read_text(encoding="utf-8").splitlines() if line.strip()]
            if ({row["id"] for row in rows} & {row["id"] for row in other}
                    or {row["input"] for row in rows} & {row["input"] for row in other}):
                raise ValueError(f"Test leakage into {name}")
    return rows


def evaluate_rows(rows: list[dict], predict: Callable[[dict], dict],
                  synchronize: Callable[[], None] | None = None) -> list[dict]:
    results = []
    for row in rows:
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        prediction = predict({"input": row["input"], "id": row["id"]})
        if synchronize is not None:
            synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        results.append({
            **{key: row[key] for key in ("id", "format", "locale", "currency", "template_id")},
            **prediction, "latency_ms": latency_ms,
            **score(prediction["raw"], row["output"]),
        })
        if prediction.get("attempts"):
            results[-1]["initial_score"] = score(prediction["attempts"][0]["raw"], row["output"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model-path", type=Path, help="GGUF model file (legacy backend)")
    models.add_argument("--adapter-path", type=Path, help="Extracted results/<RUN_NAME>/lora_adapter directory")
    models.add_argument("--base-only", action="store_true", help="Raw Qwen baseline, always CUDA NF4, no adapter")
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path, help="New directory for metrics and per-invoice predictions")
    parser.add_argument("--check-data-only", action="store_true", help="Validate inputs/artifacts without loading a model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use CUDA bitsandbytes NF4 for the LoRA base model")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--self-correct", action="store_true", help="One correction using the demo's shared validation/prompt (Transformers only)")
    parser.add_argument("--limit", type=int, help="Smoke test on the first N examples, not a full evaluation")
    args = parser.parse_args()

    if args.n_ctx <= 0 or not 0 < args.max_new_tokens < args.n_ctx or args.warmup < 0:
        parser.error("Require n_ctx > max_new_tokens > 0 and warmup >= 0")
    if args.limit is not None and not 1 <= args.limit <= 60:
        parser.error("--limit must be between 1 and 60")
    if not args.check_data_only and not (args.adapter_path or args.model_path or args.base_only):
        parser.error("--adapter-path, --base-only or --model-path required for inference")
    if args.base_only and args.device != "cuda":
        parser.error("--base-only requires CUDA for the requested 4-bit baseline")
    if args.self_correct and args.model_path:
        parser.error("--self-correct is supported only for Transformers base/LoRA models")
    rows = load_test_rows(args.test_path)
    provenance_checked = []
    if args.adapter_path:
        args.adapter_path = args.adapter_path.expanduser().resolve()
        check_adapter(args.adapter_path)
        provenance_checked = check_training_provenance(args.adapter_path, args.test_path)
    if args.model_path:
        args.model_path = args.model_path.expanduser().resolve()
        if not args.model_path.is_file():
            parser.error(f"GGUF model file not found: {args.model_path}")
    if args.check_data_only:
        print(json.dumps({
            "status": "preflight_only", "num_examples": len(rows),
            "per_locale": dict(Counter(row["locale"] for row in rows)),
            "test_sha256": file_hash(args.test_path), "adapter_checked": bool(args.adapter_path),
            "model_loaded": False,
            "training_hashes_verified": provenance_checked,
            "leakage_checked_against": [name for name in ("train", "val")
                                       if (args.test_path.parent / f"{name}.jsonl").is_file()],
        }, indent=2))
        return
    output_dir = args.output_dir or ROOT / "results" / ("evaluation-v3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")
    synchronize = None
    if args.adapter_path:
        inference = LoRAInference(args.adapter_path, args.device, args.load_in_4bit, args.n_ctx, args.max_new_tokens)
        predict, synchronize = inference.predict, inference.synchronize
        metadata = inference.metadata
        model_file = str(args.adapter_path)
    elif args.base_only:
        inference = BaseInference(args.device, True, args.n_ctx, args.max_new_tokens)
        predict, synchronize = inference.predict, inference.synchronize
        metadata = inference.metadata
        model_file = BASE_MODEL
    else:
        llm = load_model(args.model_path, args.n_ctx)
        predict = lambda row: generate(llm, row, args.max_new_tokens)
        metadata = {"model": "finetuned_gguf", "model_sha256": file_hash(args.model_path)}
        model_file = str(args.model_path)
    if args.self_correct:
        predict = lambda row: predict_with_correction(inference, row)
    if args.limit:
        rows = rows[:args.limit]
    for _ in range(args.warmup):
        predict({"input": rows[0]["input"], "id": rows[0]["id"]})
    results = evaluate_rows(rows, predict, synchronize)
    report = {
        **build_report(results, model_file), **metadata,
        "dataset_version": "international-v3", "test_path": str(args.test_path.resolve()),
        "test_sha256": file_hash(args.test_path), "full_test": len(rows) == 60,
        "training_hashes_verified": provenance_checked,
        "generation": {"n_ctx": args.n_ctx, "max_new_tokens": args.max_new_tokens,
                       "do_sample": False, "warmup_requests_excluded": args.warmup,
                       "self_correction": args.self_correct, "max_correction_attempts": 1 if args.self_correct else 0},
        "timing": "Synchronized request latency including tokenization, generation and decoding; excludes model loading and warmup.",
        "tolerance": TOLERANCE,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
