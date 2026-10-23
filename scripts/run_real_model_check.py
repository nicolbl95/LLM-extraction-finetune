#!/usr/bin/env python3
"""Opt-in actual CUDA model checks. Never substitute OCR-only fallback for success."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def environment_errors():
    missing = [name for name in ("torch", "transformers", "peft", "gradio", "pytesseract", "pymupdf")
               if importlib.util.find_spec(name) is None]
    return ["Missing dependencies: " + ", ".join(missing)] if missing else []


def compare_runs(runs, expected):
    actual_model = all(
        run.get("attempts") and run["attempts"][0].get("finish_reason") == "stop"
        and bool(run["attempts"][0].get("raw")) and not run.get("technical_errors")
        for run in runs
    )
    return {
        "actual_model_completed": actual_model,
        "final_equals_reference": [run.get("final") == expected for run in runs],
        "raw_deterministic": actual_model and all(
            [attempt["raw"] for attempt in run["attempts"]] ==
            [attempt["raw"] for attempt in runs[0]["attempts"]] for run in runs[1:]
        ),
        "final_deterministic": actual_model and all(run["final"] == runs[0]["final"] for run in runs[1:]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-environment", action="store_true", help="Probe imports only; never load/download weights")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit base model downloads on the target GPU machine")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--stress", action="store_true", help="Also run all 24 deterministic degraded images, with true generations where evidence allows")
    args = parser.parse_args()
    errors = environment_errors()
    if errors or args.check_environment:
        print(json.dumps({"dependencies_available": not errors, "errors": errors, "model_loaded": False}))
        return 1 if errors else 0
    if args.adapter_path is None or args.output is None:
        parser.error("--adapter-path and --output are required for real runs")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available():
        parser.error("A working CUDA GPU is required; no CPU or simulated fallback")
    from app import main as application
    from diagnose_pipeline import source_hash
    from evaluate_finetuned import LoRAInference, build_messages
    from cases import REFERENCES, image_cases
    from evaluate_grounding import flatten, equal
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = LoRAInference(args.adapter_path.resolve(), "cuda", args.load_in_4bit, 4096, 2048)
    report = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hash(), "model_metadata": model.metadata,
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "downloads_permitted": args.allow_download, "repetitions": 3, "images": [],
    }
    experiments = [(name, ROOT / "app/examples" / f"{name}.png", None, REFERENCES[name], False)
                   for name in ("france", "uk", "us")]
    if args.stress:
        experiments += [(case.name, None, case.image, case.expected, True) for case in image_cases(ROOT)]
    for name, image_path, image, expected, is_stress in experiments:
        image_bytes = image_path.read_bytes() if image_path else image.tobytes()
        entry = {"image": name, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                 "stress": is_stress, "runs": []}
        for repetition in range(3):
            attempts = []

            def predict(row, messages):
                result = model.predict(row, messages=messages)
                attempts.append(result)
                return result

            torch.cuda.synchronize()
            start = time.perf_counter()
            diagnostics = {}
            try:
                if image_path:
                    text = application.document_text(str(image_path))
                else:
                    try:
                        text = application.ocr_image(image)
                    except application.UnreadableOCR as error:
                        from invoice_evidence import DocumentEvidence, EvidenceText
                        text = EvidenceText(DocumentEvidence(
                            "", None, [], [], source_kind="ocr", quality_issues=[str(error)],
                        ))
                # Replace only the ZeroGPU transport: the actual model remains real.
                with patch.object(application, "predict_invoice", side_effect=predict):
                    final = application.extract_invoice(text, diagnostics=diagnostics)
                measurement_only = False
                if is_stress and not attempts:
                    # Measure the true LLM even when the production quality gate
                    # skips it; this extra raw response never enters the final JSON.
                    measurement_row = {"id": name + "-measurement-only", "input": str(text)}
                    predict(measurement_row, build_messages(measurement_row))
                    measurement_only = True
                torch.cuda.synchronize()
                run = {"repetition": repetition + 1, "attempts": attempts, "final": final,
                       "measurement_only_generation": measurement_only,
                       "technical_errors": diagnostics.get("validation_errors", []),
                       "validation": diagnostics.get("validation"), "field_statuses": diagnostics.get("field_statuses")}
                actual, gold = flatten(final), flatten(expected)
                run["reference_fields"] = len(gold)
                run["false_nulls"] = sum(value is not None and actual.get(path) is None for path, value in gold.items())
                run["wrong_values_retained"] = [
                    {"field": path, "actual": value, "expected": gold.get(path)}
                    for path, value in actual.items() if value is not None and not equal(value, gold.get(path))
                ]
                run["phantom_anomalies"] = [
                    flag for flag in diagnostics.get("validation", {}).get("flags", [])
                    if flag["severity"] != "information"
                ]
            except Exception as error:
                # Experiment runner must persist a failed run, never relabel it as OCR success.
                import traceback
                run = {"repetition": repetition + 1, "attempts": attempts, "final": None,
                       "technical_errors": [f"{type(error).__name__}: {error}"], "trace": traceback.format_exc()}
            run["duration_seconds"] = time.perf_counter() - start
            entry["runs"].append(run)
        entry.update(compare_runs(entry["runs"], expected))
        report["images"].append(entry)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    successful = all(
        image["actual_model_completed"] and all(image["final_equals_reference"])
        and image["raw_deterministic"] and image["final_deterministic"] for image in report["images"]
    )
    actual_model_called = any(run["attempts"] for image in report["images"] for run in image["runs"])
    print(json.dumps({"output": str(args.output), "acceptance_passed": successful,
                      "actual_model_called": actual_model_called}))
    return 0 if successful else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "tests/stress"))
    raise SystemExit(main())
