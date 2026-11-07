"""Real-model benchmark for the production extraction path.

The harness drives exactly what the application does - `document_text` then `extract_invoice` - and
only substitutes the model when `--simulate` is given. It scores the model's own proposals apart
from what grounding finally keeps, so an OCR miss, a gate refusal, a model error and a grounding
error are never reported as the same thing. Nothing here is imported by the application.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from app import main as application  # noqa: E402
from diagnose_independent_benchmark import fold, gold_seen  # noqa: E402
from evaluate_baseline import TOLERANCE  # noqa: E402
from evaluate_grounding import flatten  # noqa: E402
from evidence_capture import SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION, serialize_evidence  # noqa: E402
from invoice_evidence import reading_issue_category  # noqa: E402
from invoice_grounding import read_model  # noqa: E402
from ocr_config import OCR_CONFIG  # noqa: E402

VERSION = 1
TOP_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")
LINE_FIELDS = ("description", "quantity", "unit_price", "total_ht")
VERDICTS = (
    "gate_blocked", "model_unavailable", "ocr_not_recognized", "model_missed",
    "model_correct_retained", "model_correct_discarded",
    "model_wrong_rejected", "model_wrong_recovered", "model_wrong_retained",
    "model_wrong_replaced", "model_extra_rejected", "model_extra_retained",
    "correctly_absent", "final_spurious",
)
SIMILARITY_FLOOR = 0.75


def as_decimal(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.replace("\u00a0", "").replace(" ", "").replace(",", "."))
        except InvalidOperation:
            return None
    return None


def same_value(left, right):
    """Field equality: numbers within the frozen tolerance, text strictly."""
    if left is None or right is None:
        return left is None and right is None
    left_number, right_number = as_decimal(left), as_decimal(right)
    if isinstance(left, (int, float, Decimal)) or isinstance(right, (int, float, Decimal)):
        return (
            left_number is not None and right_number is not None
            and abs(left_number - right_number) <= Decimal(str(TOLERANCE))
        )
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= Decimal(str(TOLERANCE))
    return str(left).strip() == str(right).strip()


def present_paths(payload):
    """Every path the model actually emitted, null included: a missing key is not a null."""
    paths = set()
    for key in TOP_FIELDS:
        if key in payload:
            paths.add(key)
    items = payload.get("line_items")
    if isinstance(items, list):
        for index, item in enumerate(items):
            if isinstance(item, dict):
                for key in LINE_FIELDS:
                    if key in item:
                        paths.add("line_items.%d.%s" % (index, key))
    return paths


def parse_proposal(raw):
    if not raw:
        return "absent", {}, None
    try:
        return "ok", read_model(raw), None
    except (ValueError, TypeError) as exc:
        return "invalid", {}, str(exc)[:200]


def classify(*, gold, model, final, gate_blocked, unavailable, ocr_seen):
    """One field, one verdict. OCR and gate outcomes never count as model errors."""
    if gate_blocked:
        return "gate_blocked"
    if unavailable:
        return "model_unavailable"
    if gold is None:
        if model is None:
            return "correctly_absent" if final is None else "final_spurious"
        return "model_extra_rejected" if final is None or not same_value(final, model) else "model_extra_retained"
    if model is None:
        return "model_missed" if ocr_seen else "ocr_not_recognized"
    if same_value(model, gold):
        return "model_correct_retained" if same_value(final, model) else "model_correct_discarded"
    if same_value(final, gold):
        return "model_wrong_recovered"
    if final is None:
        return "model_wrong_rejected"
    return "model_wrong_retained" if same_value(final, model) else "model_wrong_replaced"


def align_rows(gold_rows, actual_rows):
    """Pair printed rows with proposed rows: same description first, then similarity, then order."""
    pairs, used = [], set()
    for gold_index, row in enumerate(gold_rows):
        want = fold(str((row or {}).get("description") or ""))
        if not want:
            continue
        for actual_index, candidate in enumerate(actual_rows):
            if actual_index in used:
                continue
            if fold(str((candidate or {}).get("description") or "")) == want:
                pairs.append((gold_index, actual_index))
                used.add(actual_index)
                break
    try:
        from difflib import SequenceMatcher
    except ImportError:  # pragma: no cover
        SequenceMatcher = None
    if SequenceMatcher is not None:
        for gold_index, row in enumerate(gold_rows):
            if any(pair[0] == gold_index for pair in pairs):
                continue
            want = fold(str((row or {}).get("description") or ""))
            best, score = None, 0.0
            for actual_index, candidate in enumerate(actual_rows):
                if actual_index in used:
                    continue
                got = fold(str((candidate or {}).get("description") or ""))
                ratio = SequenceMatcher(None, want, got).ratio() if want and got else 0.0
                if ratio > score:
                    best, score = actual_index, ratio
            if best is not None and score >= SIMILARITY_FLOOR:
                pairs.append((gold_index, best))
                used.add(best)
    remaining_actual = [index for index in range(len(actual_rows)) if index not in used]
    for gold_index in range(len(gold_rows)):
        if any(pair[0] == gold_index for pair in pairs) or not remaining_actual:
            continue
        pairs.append((gold_index, remaining_actual.pop(0)))
    pairs.sort()
    order_preserved = [actual for _, actual in pairs] == sorted(actual for _, actual in pairs)
    matched_actual = {actual for _, actual in pairs}
    return {
        "pairs": pairs,
        "order_preserved": order_preserved,
        "unmatched_gold": [index for index in range(len(gold_rows)) if index not in {g for g, _ in pairs}],
        "unmatched_actual": [index for index in range(len(actual_rows)) if index not in matched_actual],
    }


def duplicates(rows):
    counter = Counter(fold(str((row or {}).get("description") or "")) for row in rows)
    return sum(count - 1 for count in counter.values() if count > 1)


def row_accuracy(gold_rows, actual_rows):
    alignment = align_rows(gold_rows, actual_rows)
    if not gold_rows:
        return {"matched": 0, "gold_rows": 0, "accuracy": None, **alignment}
    complete = 0
    for gold_index, actual_index in alignment["pairs"]:
        gold_row, actual_row = gold_rows[gold_index], actual_rows[actual_index]
        if all(same_value(actual_row.get(key), gold_row.get(key)) for key in LINE_FIELDS):
            complete += 1
    return {"matched": complete, "gold_rows": len(gold_rows),
            "accuracy": complete / len(gold_rows), **alignment}


def field_records(*, truth, proposal, final, gate_blocked, unavailable, ocr_text, statuses, suggestions):
    records = []
    truth_paths = flatten(truth)
    proposal_paths = flatten(proposal)
    final_paths = flatten(final)
    emitted = present_paths(proposal)
    for path in sorted(set(truth_paths) | set(proposal_paths) | set(final_paths)):
        parts = path.split(".")
        gold = truth_paths.get(path)
        model = proposal_paths.get(path)
        value = final_paths.get(path)
        ocr_seen = gold is None or gold_seen(parts[-1], gold, ocr_text)
        verdict = classify(gold=gold, model=model, final=value, gate_blocked=gate_blocked,
                           unavailable=unavailable, ocr_seen=ocr_seen)
        records.append({
            "path": path,
            "gold": str(gold) if isinstance(gold, Decimal) else gold,
            "model": str(model) if isinstance(model, Decimal) else model,
            "final": str(value) if isinstance(value, Decimal) else value,
            "verdict": verdict,
            "ocr_seen": ocr_seen,
            "model_emitted_key": path in emitted,
            "grounded_status": statuses.get(path),
            "model_suggestion_withheld": suggestions.get(path),
        })
    return records


def line_item_section(*, truth, proposal, final):
    gold_rows = [row or {} for row in (truth.get("line_items") or [])]
    model_rows = [row for row in (proposal.get("line_items") or []) if isinstance(row, dict)]
    final_rows = [row or {} for row in (final.get("line_items") or [])]
    model_alignment = align_rows(gold_rows, model_rows)
    final_alignment = align_rows(gold_rows, final_rows)
    return {
        "gold_rows": len(gold_rows),
        "model_rows": len(model_rows),
        "final_rows": len(final_rows),
        "duplicates_in_gold": duplicates(gold_rows),
        "duplicates_in_model": duplicates(model_rows),
        "model": {
            "row_count_match": len(model_rows) == len(gold_rows),
            "order_preserved": model_alignment["order_preserved"],
            "unmatched_actual": model_alignment["unmatched_actual"],
            **{key: value for key, value in row_accuracy(gold_rows, model_rows).items()
               if key in {"matched", "gold_rows", "accuracy"}},
        },
        "final": {
            "row_count_match": len(final_rows) == len(gold_rows),
            "order_preserved": final_alignment["order_preserved"],
            "unmatched_actual": final_alignment["unmatched_actual"],
            **{key: value for key, value in row_accuracy(gold_rows, final_rows).items()
               if key in {"matched", "gold_rows", "accuracy"}},
        },
    }


def arithmetic_substitutions(truth, proposal):
    """Printed rows the model 'corrected' to their arithmetic product: never acceptable."""
    found = []
    for index, gold_row in enumerate(truth.get("line_items") or []):
        item = (proposal.get("line_items") or [])
        if index >= len(item) or not isinstance(item[index], dict):
            continue
        quantity = as_decimal(item[index].get("quantity"))
        unit_price = as_decimal(item[index].get("unit_price"))
        amount = as_decimal(item[index].get("total_ht"))
        printed = as_decimal((gold_row or {}).get("total_ht"))
        if None in (quantity, unit_price, amount, printed):
            continue
        product = (quantity * unit_price).quantize(Decimal("0.01"))
        if abs(amount - product) <= Decimal(str(TOLERANCE)) and abs(printed - product) > Decimal(str(TOLERANCE)):
            found.append({"line": index, "printed": str(printed), "model": str(amount),
                          "arithmetic_product": str(product)})
    return found


def expected_flags_present(validation, expected):
    codes = {flag.get("code") for flag in (validation or {}).get("flags") or []}
    return {code: code in codes for code in expected or []}


def evidence_facts(evidence):
    confidence = (sum(token.confidence for token in evidence.tokens) / len(evidence.tokens)
                  if evidence.tokens else 0.0)
    return {
        "available": evidence.available,
        "source_kind": evidence.source_kind,
        "lines": len(evidence.lines),
        "tokens": len(evidence.tokens),
        "rows": len(evidence.rows),
        "mean_confidence": round(confidence, 2),
        "quality_issues": list(evidence.quality_issues),
        "issue_category": reading_issue_category(evidence),
        "ocr_calls": evidence.ocr_calls,
        "ocr_elapsed_seconds": round(evidence.ocr_elapsed_seconds, 3),
    }


class SimulatedModel:
    """Deterministic stand-in: one queued response per request, in production call order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, row, messages):
        self.calls.append({"id": row.get("id"), "messages": len(messages)})
        if not self.responses:
            raise RuntimeError("simulated model has no response left for this request")
        response = self.responses.pop(0)
        return {
            "raw": response.get("raw", ""),
            "finish_reason": response.get("finish_reason", "stop"),
            "completion_tokens": response.get("completion_tokens", 0),
            "prompt_tokens": response.get("prompt_tokens", 0),
            "generation_seconds": response.get("generation_seconds", 0.0),
        }


@contextlib.contextmanager
def injected_model(predict):
    if predict is None:
        yield
        return
    original = application.predict_invoice
    application.predict_invoice = predict
    try:
        yield
    finally:
        application.predict_invoice = original


def run_case(case, *, predict=None, capture_evidence: bool = False):
    """Production path, one document. `predict` replaces the model only."""
    path = (ROOT / case["path"]).resolve()
    truth = load_truth(case["truth"])
    started = time.perf_counter()
    ocr_started = time.perf_counter()
    text = application.document_text(str(path))
    ocr_seconds = time.perf_counter() - ocr_started
    diagnostics = {}
    with injected_model(predict):
        payload = application.extract_invoice(text, warnings=[], diagnostics=diagnostics)
    total_seconds = time.perf_counter() - started
    evidence = text.evidence
    model_info = diagnostics.get("model_info") or {}
    gate_blocked = not model_info.get("used") and model_info.get("reason") == "preuves OCR insuffisantes"
    unavailable = not model_info.get("used") and model_info.get("reason") in {
        "indisponibilité technique", "quota"}
    raw = diagnostics.get("first_pass_raw") or ""
    parse_status, proposal, parse_error = parse_proposal(raw)
    validation = diagnostics.get("validation") or {}
    records = field_records(
        truth=truth, proposal=proposal, final=payload, gate_blocked=gate_blocked,
        unavailable=unavailable, ocr_text=evidence.text, statuses=diagnostics.get("field_statuses") or {},
        suggestions=diagnostics.get("unverified_model_readings") or {},
    )
    model_seconds = round(diagnostics.get("request_elapsed_seconds") or 0.0, 3)
    result = {
        "id": case["id"],
        "path": str(case["path"]),
        "status": case.get("status", "diagnostic"),
        "gate": {
            "reached_model": bool(model_info.get("used")),
            "reason": model_info.get("reason"),
            "passes_requested": model_info.get("passes_requested"),
            "passes_succeeded": model_info.get("passes_succeeded"),
            "correction_attempted": diagnostics.get("correction_attempted"),
            **evidence_facts(evidence),
        },
        "model": {
            "raw": raw,
            "raw_parse": parse_status,
            "parse_error": parse_error,
            "finish_reason": diagnostics.get("finish_reason"),
            "completion_tokens": diagnostics.get("completion_tokens"),
            "prompt_tokens": diagnostics.get("prompt_tokens"),
            "generation_seconds": model_info.get("generation_seconds"),
            "proposed_paths": sorted(present_paths(proposal)),
            "missing_truth_paths": sorted(set(flatten(truth)) - present_paths(proposal)),
        },
        "fields": records,
        "line_items": line_item_section(truth=truth, proposal=proposal, final=payload),
        "arithmetic_substitutions": arithmetic_substitutions(truth, proposal),
        "validation": {
            "flags": validation.get("flags") or [],
            "expected_flags": expected_flags_present(validation, case.get("expected_flags")),
        },
        "final_payload": payload,
        "model_info": model_info,
        "runtime": {
            "ocr_seconds": round(ocr_seconds, 3),
            "model_seconds": model_seconds,
            "generation_seconds": round(model_info.get("generation_seconds") or 0.0, 3),
            "grounding_seconds": round(max(0.0, total_seconds - ocr_seconds - model_seconds), 3),
            "total_seconds": round(total_seconds, 3),
        },
    }
    if capture_evidence:
        result["evidence_capture"] = serialize_evidence(evidence, diagnostics=diagnostics)
    return result


def aggregate(cases):
    verdicts = Counter()
    by_family = defaultdict_counter()
    final_stats = Counter()
    model_stats = Counter()
    for case in cases:
        for record in case["fields"]:
            verdicts[record["verdict"]] += 1
            family = "line_items" if record["path"].startswith("line_items.") else "header"
            by_family[family][record["verdict"]] += 1
            gold, value = record["gold"], record["final"]
            if gold is None:
                final_stats["gold_null" if value is None else "spurious_value"] += 1
            elif value is None:
                final_stats["false_null"] += 1
            elif same_value(value, gold):
                final_stats["correct"] += 1
            else:
                final_stats["wrong"] += 1
            if record["model"] is None:
                model_stats["not_proposed"] += 1
            elif gold is not None and same_value(record["model"], gold):
                model_stats["correct"] += 1
            elif gold is None:
                model_stats["extra"] += 1
            else:
                model_stats["wrong"] += 1
    decided = final_stats["correct"] + final_stats["wrong"]
    coverage_base = final_stats["correct"] + final_stats["wrong"] + final_stats["false_null"]
    return {
        "fields": len([record for case in cases for record in case["fields"]]),
        "verdicts": {name: verdicts.get(name, 0) for name in VERDICTS},
        "by_family": {name: dict(sorted(counts.items())) for name, counts in by_family.items()},
        "model": {key: model_stats.get(key, 0)
                  for key in ("correct", "wrong", "extra", "not_proposed")},
        "final": {
            "correct": final_stats.get("correct", 0),
            "wrong": final_stats.get("wrong", 0),
            "false_null": final_stats.get("false_null", 0),
            "gold_null": final_stats.get("gold_null", 0),
            "spurious_value": final_stats.get("spurious_value", 0),
            "precision": round(final_stats["correct"] / decided, 4) if decided else None,
            "coverage": round(final_stats["correct"] / coverage_base, 4) if coverage_base else None,
        },
        "grounding": {
            "kept_correct": verdicts.get("model_correct_retained", 0),
            "discarded_correct": verdicts.get("model_correct_discarded", 0),
            "rejected_wrong": verdicts.get("model_wrong_rejected", 0),
            "recovered_to_truth": verdicts.get("model_wrong_recovered", 0),
            "retained_wrong": verdicts.get("model_wrong_retained", 0)
                              + verdicts.get("model_extra_retained", 0),
        },
        "gate": {
            "documents": len(cases),
            "reached": sum(1 for case in cases if case["gate"]["reached_model"]),
            "blocked": [{"id": case["id"], "reason": case["gate"]["reason"]}
                        for case in cases if not case["gate"]["reached_model"]],
        },
        "runtime": {
            "ocr_seconds": round(sum(case["runtime"]["ocr_seconds"] for case in cases), 3),
            "model_seconds": round(sum(case["runtime"]["model_seconds"] for case in cases), 3),
            "grounding_seconds": round(sum(case["runtime"]["grounding_seconds"] for case in cases), 3),
            "total_seconds": round(sum(case["runtime"]["total_seconds"] for case in cases), 3),
        },
        "arithmetic_substitutions": [
            {"id": case["id"], **entry} for case in cases for entry in case["arithmetic_substitutions"]
        ],
        "expected_flags": {case["id"]: case["validation"]["expected_flags"]
                           for case in cases if case["validation"]["expected_flags"]},
    }


class defaultdict_counter(dict):
    def __missing__(self, key):
        self[key] = Counter()
        return self[key]


def load_truth(spec):
    if "inline" in spec:
        return spec["inline"]
    document = json.loads((ROOT / spec["file"]).read_text(encoding="utf-8"))
    for part in spec["pointer"].split("/"):
        document = document[part]
    return document


def environment():
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "accelerate": version("accelerate"),
        "bitsandbytes": version("bitsandbytes"),
        "cuda_available": _cuda_available(),
        "inference_device": os.environ.get("INVOICE_DEVICE", "cuda"),
        "adapter_path": os.environ.get("INVOICE_ADAPTER_PATH") or str(getattr(application, "DEFAULT_ADAPTER", "")),
        "offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def _cuda_available():
    try:
        import torch
    except ImportError:
        return None
    return bool(torch.cuda.is_available())


def reproducibility():
    """Frozen identity of the extraction logic, so a run can be tied to this prompt and OCR policy."""
    prompt = json.dumps({
        "system": application.LIVE_SYSTEM_PROMPT,
        "instruction": application.INSTRUCTION,
        "few_shot_invoice": application.FEW_SHOT_INVOICE,
        "few_shot_result": application.FEW_SHOT_RESULT,
    }, ensure_ascii=False, sort_keys=True)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = None
    return {
        "git_head": head,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "ocr_configuration": OCR_CONFIG.fingerprint(),
        "ocr_configuration_version": OCR_CONFIG.version,
        "n_ctx": 4096, "max_new_tokens": 2048, "do_sample": False,
        "grounding": "invoice_grounding.run_grounded_inference",
    }


def adapter_artifacts(adapter_path):
    if not adapter_path:
        return {"path": None, "present": False}
    directory = Path(adapter_path)
    if not directory.is_dir():
        return {"path": str(directory), "present": False}
    facts = {"path": str(directory), "present": True, "files": {}}
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        path = directory / name
        if path.is_file():
            facts["files"][name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            }
    manifest = directory.parent / "training_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        facts["training_manifest"] = {key: data.get(key) for key in
                                      ("base_model", "status", "dataset_version", "created_utc")}
    else:
        facts["training_manifest"] = None
    config = directory / "adapter_config.json"
    if config.is_file():
        data = json.loads(config.read_text(encoding="utf-8"))
        facts["base_model"] = data.get("base_model_name_or_path")
    return facts


def manifest_fingerprint(cases):
    """Identity of the evaluation: documents, printed truth and the flags it expects."""
    identity = [
        {"id": case.get("id"), "path": case.get("path"),
         "truth": load_truth(case.get("truth") or {}), "expected_flags": case.get("expected_flags") or [],
         "status": case.get("status", "diagnostic")}
        for case in cases
    ]
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_manifest(cases):
    problems = []
    for case in cases:
        if not isinstance(case.get("id"), str) or not case["id"]:
            problems.append({"case": case.get("id"), "problem": "missing id"})
        path = ROOT / str(case.get("path", ""))
        if not path.is_file():
            problems.append({"case": case.get("id"), "problem": "document not found: %s" % path})
        try:
            truth = load_truth(case.get("truth") or {})
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            problems.append({"case": case.get("id"), "problem": "truth not readable: %s" % exc})
            continue
        if not isinstance(truth, dict) or not isinstance(truth.get("line_items"), list):
            problems.append({"case": case.get("id"), "problem": "truth must contain line_items"})
    return problems


def simulated_from_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("simulation file must be an object keyed by case id")
    responses = {key: list(value) for key, value in payload.items()}

    def factory(case_id):
        return SimulatedModel(responses.get(case_id, payload.get("default", [])))

    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New directory for bench.json and cases.jsonl")
    parser.add_argument("--simulate", type=Path, help="JSON: case id -> list of canned model responses")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", help="comma-separated case ids to run")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--capture-evidence", action="store_true",
                        help="Add the read-only documentary evidence capture to each case.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        parser.error("manifest must contain a non-empty cases list")
    declared = len(cases)
    if args.only:
        wanted = [name.strip() for name in args.only.split(",") if name.strip()]
        known = {case["id"] for case in cases}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            parser.error("unknown case ids: " + ", ".join(unknown))
        cases = [case for case in cases if case["id"] in wanted]
    problems = validate_manifest(cases)
    if problems:
        parser.error("invalid manifest: " + json.dumps(problems, ensure_ascii=False))
    if args.check_data_only:
        print(json.dumps({
            "status": "preflight_only", "cases": len(cases),
            "documents": [case["id"] for case in cases],
            "manifest_sha256": manifest_fingerprint(cases),
            "model_loaded": False, "ocr_executed": False,
        }, ensure_ascii=False, indent=1))
        return 0

    os.environ["INVOICE_DEVICE"] = args.device
    if args.adapter_path:
        os.environ["INVOICE_ADAPTER_PATH"] = str(args.adapter_path.resolve())
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.simulate:
        factory = simulated_from_file(args.simulate)
        predict_for = factory
    else:
        predict_for = lambda case_id: None  # noqa: E731 - the production model stays in place

    results = []
    for case in cases[: args.limit] if args.limit else cases:
        predictor = predict_for(case["id"]) if args.simulate else None
        results.append(run_case(case, predict=predictor, capture_evidence=args.capture_evidence))

    report = {
        "harness": {"version": VERSION, "mode": "simulated" if args.simulate else "production",
                    "limit": args.limit, "cases": len(results),
                    "generation": {"n_ctx": 4096, "max_new_tokens": 2048, "source": "app.get_extractor"}},
        "environment": environment(),
        "reproducibility": reproducibility(),
        "adapter": adapter_artifacts(os.environ.get("INVOICE_ADAPTER_PATH", ""))
        if os.environ.get("INVOICE_ADAPTER_PATH") else None,
        "manifest": {"path": str(args.manifest), "sha256": manifest_fingerprint(cases),
                     "declared_cases": declared, "selected_cases": len(cases)},
        "aggregate": aggregate(results),
        "cases": results,
    }
    if args.capture_evidence:
        report["evidence_capture"] = {
            "enabled": True,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "files": [f"evidence/{case['id']}.json" for case in results],
        }
    if args.output is None:
        print(json.dumps(report["aggregate"], ensure_ascii=False, indent=1))
        return 0
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with (args.output / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in results:
            handle.write(json.dumps(case, ensure_ascii=False, default=str) + "\n")
    if args.capture_evidence:
        evidence_dir = args.output / "evidence"
        evidence_dir.mkdir()
        for case in results:
            (evidence_dir / f"{case['id']}.json").write_text(
                json.dumps(case["evidence_capture"], ensure_ascii=False, indent=1, default=str) + "\n",
                encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
