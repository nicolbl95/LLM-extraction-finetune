#!/usr/bin/env python3
"""Local real-document review: OCR/constraints only, never a model download."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evaluate_grounding import equal, flatten
from invoice_evidence import DocumentEvidence, UnreadableOCR
from invoice_grounding import reconcile
from invoice_validation import render_report
from ocr_config import OCR_CONFIG


def load_truth(path):
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("documents"), dict):
        raise ValueError("Truth must be an object with 'documents': {relative_filename: expected_invoice}")
    for name, payload in data["documents"].items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            raise ValueError("Each truth document must map a relative filename to an invoice object")
    return data


def evaluate_document(path, relative_name, expected=None):
    from app import main
    actual_ocr = main.pytesseract.image_to_data
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual_ocr(*args, **kwargs)

    start = time.perf_counter()
    operational_error = None
    with patch.object(main.pytesseract, "image_to_data", side_effect=counted):
        try:
            document = main.document_text(str(path))
            evidence = document.evidence
        except UnreadableOCR as error:
            operational_error = f"{type(error).__name__}: {error}"
            evidence = DocumentEvidence("", None, [], [], source_kind="ocr", quality_issues=[str(error)])
        except (OSError, ValueError, RuntimeError, main.pytesseract.TesseractError) as error:
            main.LOGGER.exception("Real document evaluation could not read %s", relative_name)
            operational_error = f"{type(error).__name__}: {error}"
            evidence = DocumentEvidence("", None, [], [], available=False)
        result = reconcile(evidence, [{}])
    actual = flatten(result.payload)
    record = {
        "file": relative_name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": "actual OCR/native PDF plus constraints; no model",
        "seconds": time.perf_counter() - start, "tesseract_calls": calls,
        "neutralized_fields": [key for key, value in actual.items() if value is None],
        "field_statuses": result.statuses, "validation": result.validation.export(),
        "final": result.payload, "ocr_lines": len(evidence.lines),
        "technical_error": operational_error,
        "report": render_report(evidence, result.validation, result.statuses, [], available=evidence.available),
        "manual_review_required": True,
    }
    if expected is not None:
        gold = flatten(expected)
        record["reference_fields"] = len(gold)
        record["false_nulls"] = [key for key, value in gold.items() if value is not None and actual.get(key) is None]
        record["wrong_values_retained"] = [
            {"field": key, "actual": value, "expected": gold.get(key)}
            for key, value in actual.items() if value is not None and not equal(value, gold.get(key))
        ]
    return record


def run(folder, truth_path=None):
    truth = load_truth(truth_path)
    files = sorted(path for path in folder.rglob("*") if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"} and path.is_file())
    if not files:
        raise ValueError("No supported image/PDF documents in the folder")
    names = [path.relative_to(folder).as_posix() for path in files]
    if truth is not None and set(names) != set(truth["documents"]):
        raise ValueError("Ground-truth filenames must exactly match the evaluated files; partial scoring is refused")
    records = [
        evaluate_document(path, name, truth["documents"][name] if truth is not None else None)
        for path, name in zip(files, names)
    ]
    report = {
        "model_called": False, "truth_provided": truth is not None,
        "ocr_configuration": OCR_CONFIG.export(), "ocr_configuration_sha256": OCR_CONFIG.fingerprint(),
        "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest() if truth_path else None,
        "documents": records, "document_count": len(records),
        "warning": "No truth: nulls and flags are review candidates, not measured mistakes. No thresholds are fitted here.",
    }
    if truth is not None:
        report["summary"] = {
            "reference_fields": sum(record["reference_fields"] for record in records),
            "false_nulls": sum(len(record["false_nulls"]) for record in records),
            "wrong_values_retained": sum(len(record["wrong_values_retained"]) for record in records),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error("The scans folder does not exist")
    if args.output.exists():
        parser.error("Refusing to overwrite an earlier experiment")
    report = run(args.folder, args.truth)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "documents"}, ensure_ascii=False))
    if report.get("summary", {}).get("wrong_values_retained", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
