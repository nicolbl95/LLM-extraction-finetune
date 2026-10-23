#!/usr/bin/env python3
"""Real OCR / simulated LLM stress checks, separate from model benchmark data."""

import argparse
import json
import sys
import tempfile
import os
import time
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/stress")]

from cases import image_cases, layout_cases, error_profile
from invoice_evidence import DocumentEvidence, UnreadableOCR, from_text
from invoice_grounding import read_model, reconcile
from invoice_validation import render_report
from evaluate_grounding import equal, flatten
from ocr_config import OCR_CONFIG


def run_case(case):
    from app import main
    started = time.perf_counter()
    calls = 0
    original = main.pytesseract.image_to_data

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    try:
        with patch.object(main.pytesseract, "image_to_data", side_effect=counted):
            if case.image is not None:
                try:
                    evidence = main.ocr_image(case.image).evidence
                except UnreadableOCR as error:
                    evidence = DocumentEvidence("", None, [], [], source_kind="ocr", quality_issues=[str(error)])
            elif case.pdf:
                evidence = main.document_text(str(case.pdf)).evidence
            else:
                evidence = from_text(case.text)
        result = reconcile(evidence, [read_model(json.dumps(case.model))])
        actual, expected = flatten(result.payload), flatten(case.expected)
        wrong = [
            {"field": field, "actual": value, "expected": expected.get(field)}
            for field, value in actual.items() if value is not None and not equal(value, expected.get(field))
        ]
        nulls = [field for field, value in expected.items() if value is not None and actual.get(field) is None]
        flags = [flag for flag in result.validation.export()["flags"] if flag["severity"] != "information"]
        profile_results = []
        for profile in ("reported", "coherent", "permuted"):
            profiled = reconcile(evidence, [read_model(json.dumps(error_profile(case.expected, profile)))])
            values = flatten(profiled.payload)
            profile_results.append({
                "profile": profile,
                "wrong_values": [
                    {"field": field, "actual": value, "expected": expected.get(field)}
                    for field, value in values.items() if value is not None and not equal(value, expected.get(field))
                ],
                "false_nulls": [field for field, value in expected.items() if value is not None and values.get(field) is None],
                "phantom_anomalies": [
                    flag for flag in profiled.validation.export()["flags"] if flag["severity"] != "information"
                ],
            })
        return {
            "case": case.name, "wrong_values": wrong, "false_nulls": nulls, "phantom_anomalies": flags,
            "crash": None, "ocr_lines": len(evidence.lines), "table_rows": len(evidence.rows),
            "ocr_mean_confidence": sum(token.confidence for token in evidence.tokens) / max(1, len(evidence.tokens)),
            "quality_issues": getattr(evidence, "quality_issues", []),
            "final": result.payload, "statuses": result.statuses,
            "report": render_report(evidence, result.validation, result.statuses, []),
            "model_error_profiles": profile_results,
            "expected_fields": len(expected),
            "seconds": time.perf_counter() - started, "tesseract_calls": calls,
        }
    except Exception as error:
        # Test runner records failures rather than hiding or returning a success-shaped result.
        import traceback
        return {"case": case.name, "wrong_values": [], "false_nulls": [], "phantom_anomalies": [],
                "seconds": time.perf_counter() - started, "tesseract_calls": calls,
                "crash": {"type": type(error).__name__, "message": str(error), "trace": traceback.format_exc()}}


def run():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests/stress") as directory:
        cases = [*image_cases(ROOT), *layout_cases(Path(directory))]
        results = [run_case(case) for case in cases]
    totals = {
        "cases": len(results), "wrong_values": sum(len(r["wrong_values"]) for r in results),
        "false_nulls": sum(len(r["false_nulls"]) for r in results),
        "phantom_anomalies": sum(len(r["phantom_anomalies"]) for r in results),
        "crashes": sum(r["crash"] is not None for r in results),
        "expected_fields": sum(r.get("expected_fields", 0) for r in results),
    }
    profiles = [profile for record in results for profile in record.get("model_error_profiles", [])]
    return {"scope": "real Tesseract on degraded images; native PDF/text layouts; simulated wrong model",
            "seed": 713, "summary": totals, "cases": results,
            "ocr_configuration": OCR_CONFIG.export(), "ocr_configuration_sha256": OCR_CONFIG.fingerprint(),
            "additional_model_errors": {
                "cases": len(profiles), "wrong_values": sum(len(r["wrong_values"]) for r in profiles),
                "phantom_anomalies": sum(len(r["phantom_anomalies"]) for r in profiles),
                "provenance": "User-reported Qwen failures reconstructed for replay, plus coherent/permuted errors; not locally captured generations",
            }}


def compare_latency_candidate(report, baseline, label):
    if len(report["cases"]) != len(baseline["cases"]) or (
        {r["case"] for r in report["cases"]} != {r["case"] for r in baseline["cases"]}
    ):
        raise ValueError("Latency acceptance requires the exact same stress cases")
    denominator = baseline["summary"]["expected_fields"]
    if denominator <= 0 or report["summary"]["expected_fields"] != denominator:
        raise ValueError("Latency acceptance requires the same positive field denominator")
    delta = 100 * (report["summary"]["false_nulls"] - baseline["summary"]["false_nulls"]) / denominator
    profiles = report["additional_model_errors"]
    return {
        "baseline": label, "false_null_delta_percentage_points": delta,
        "allowed_increase_percentage_points": 2.0,
        "passed": delta <= 2 and all(report["summary"][key] == 0 for key in
                                   ("wrong_values", "phantom_anomalies", "crashes"))
        and profiles["wrong_values"] == profiles["phantom_anomalies"] == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Refuse >2 percentage points more false nulls than this fixed baseline")
    args = parser.parse_args()
    report = run()
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        report["latency_acceptance"] = compare_latency_candidate(report, baseline, str(args.compare))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    for case in report["cases"]:
        if case["wrong_values"] or case["phantom_anomalies"] or case["crash"]:
            print(json.dumps({key: case[key] for key in ("case", "wrong_values", "phantom_anomalies", "crash")}, ensure_ascii=False))
    if any(report["summary"][key] for key in ("wrong_values", "phantom_anomalies", "crashes")):
        raise SystemExit(1)
    if report["additional_model_errors"]["wrong_values"] or report["additional_model_errors"]["phantom_anomalies"]:
        raise SystemExit(1)
    if report.get("latency_acceptance", {}).get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
