#!/usr/bin/env python3
"""Test-only shadow evaluator for a four-call OCR fast path.

This module never changes the production OCR decision path.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pytesseract
from pytesseract import Output

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/stress")]

from cases import error_profile, image_cases
from evaluate_grounding import equal, flatten
from invoice_evidence import from_tsv
from invoice_grounding import read_model, reconcile
from ocr_config import OCR_CONFIG
from ocr_fast_path import attempt_fast_path
from ocr_preprocessing import deskew, transform


def _read(reader, image, psm, label, calls):
    started = time.perf_counter()
    data = reader(image, psm)
    calls.append({"label": label, "psm": psm, "seconds": time.perf_counter() - started})
    return from_tsv(data)


def evaluate_fast(image):
    """Evaluate the proposed early gate; return a diagnostic record."""
    calls = []
    started = time.perf_counter()

    def reader(prepared, psm):
        return pytesseract.image_to_data(
            prepared,
            lang="fra+eng",
            config=f"--psm {psm}",
            output_type=Output.DICT,
            timeout=OCR_CONFIG.call_seconds,
        )

    rotated, angle = deskew(image)
    prepared = {
        "deskew": transform(rotated, angle, "deskew")[0],
        "clahe": transform(rotated, angle, "clahe")[0],
    }
    first = []
    for family, psm in zip(OCR_CONFIG.fast_families, OCR_CONFIG.segmentations):
        first.append(_read(reader, prepared[family], psm, f"fast/{family}", calls))

    def read(prepared, psm, label):
        return _read(reader, prepared, psm, label, calls)

    outcome = attempt_fast_path(first, prepared, read)
    return {
        "accepted": outcome.accepted,
        "reason": outcome.reason,
        "calls": calls,
        "seconds": time.perf_counter() - started,
        "evidence": outcome.evidence,
        "result": outcome.result,
        "invoice": outcome.invoice,
    }


def _score(result, expected):
    actual = flatten(result.payload)
    gold = flatten(expected)
    false_nulls = [
        path for path, value in gold.items() if value is not None and actual.get(path) is None
    ]
    wrong = [
        {"field": path, "actual": value, "expected": gold.get(path)}
        for path, value in actual.items()
        if value is not None and not equal(value, gold.get(path))
    ]
    anomalies = [
        flag.code for flag in result.validation.flags if flag.severity != "information"
    ]
    return false_nulls, wrong, anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_report = json.loads(args.report.read_text(encoding="utf-8"))
    full = {case["case"]: case for case in full_report["cases"]}
    image_names = {case.name for case in image_cases(ROOT)}
    records = []
    for case in image_cases(ROOT):
        shadow = evaluate_fast(case.image)
        record = {
            "case": case.name,
            "applicable": True,
            "accepted": shadow["accepted"],
            "reason": shadow["reason"],
            "fast_calls": len(shadow["calls"]),
            "fast_seconds": round(shadow["seconds"], 3),
            "full_calls": full[case.name]["tesseract_calls"],
            "full_seconds": round(full[case.name]["seconds"], 3),
            "call_breakdown": shadow["calls"],
            "invoice": shadow["invoice"],
        }
        if shadow["accepted"]:
            full_result = full[case.name]
            fast_false, fast_wrong, fast_anomalies = _score(shadow["result"], case.expected)
            record.update(
                fast_false_nulls=fast_false,
                fast_wrong_values=fast_wrong,
                fast_anomalies=fast_anomalies,
                full_false_nulls=full_result["false_nulls"],
                full_wrong_values=full_result["wrong_values"],
                full_anomalies=full_result["phantom_anomalies"],
                full_matches=shadow["result"].payload == full_result["final"],
            )
            profile_mismatches = []
            for profile in ("reported", "coherent", "permuted"):
                model = read_model(json.dumps(error_profile(case.expected, profile)))
                profiled = reconcile(shadow["evidence"], [model])
                false, wrong, anomalies = _score(profiled, case.expected)
                profile_mismatches.append({
                    "profile": profile,
                    "false_nulls": false,
                    "wrong_values": wrong,
                    "anomalies": anomalies,
                })
            record["fast_model_profiles"] = profile_mismatches
        records.append(record)

    # Existing stress cases with no image OCR (text/native layouts).
    records.extend({
        "case": name,
        "applicable": False,
        "accepted": False,
        "reason": "no image OCR fast path",
        "fast_calls": 0,
        "full_calls": full[name]["tesseract_calls"],
        "full_seconds": round(full[name]["seconds"], 3),
    } for name in full if name not in image_names)

    output = {
        "scope": "shadow only; production path unchanged",
        "cases": records,
        "image_cases": len(image_names),
        "not_applicable": len(records) - len(image_names),
        "accepted": sum(record["accepted"] for record in records),
        "rejected": sum(record["applicable"] and not record["accepted"] for record in records),
        "stress_baseline": full_report["summary"],
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "cases"}, ensure_ascii=False))
    for record in records:
        print(json.dumps({
            key: record.get(key) for key in (
                "case", "applicable", "accepted", "reason", "fast_calls", "fast_seconds",
                "full_calls", "full_seconds", "invoice", "full_matches",
            ) if key in record
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
