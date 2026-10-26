#!/usr/bin/env python3
"""Re-measure the nine tracked degraded fixtures with real Tesseract.

The nine images are the degraded twins of the three distributable examples. The model
response is the documented reference, so every remaining difference is a reading limit:
the script reports null fields, false values, anomalies and crashes, and writes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/stress")]

from cases import REFERENCES  # noqa: E402
from evaluate_grounding import equal, flatten  # noqa: E402
from invoice_grounding import read_model, reconcile  # noqa: E402

FIXTURES = (
    ("france", "rotate-2", "france-rotate-2.png"),
    ("france", "blur-light", "france-blur-light.png"),
    ("france", "jpeg-q40", "france-jpeg-q40.jpg"),
    ("uk", "rotate-2", "uk-rotate-2.png"),
    ("uk", "blur-light", "uk-blur-light.png"),
    ("uk", "jpeg-q40", "uk-jpeg-q40.jpg"),
    ("us", "rotate-2", "us-rotate-2.png"),
    ("us", "blur-light", "us-blur-light.png"),
    ("us", "jpeg-q40", "us-jpeg-q40.jpg"),
)


def run_case(country: str, name: str, filename: str) -> dict:
    from app import main

    expected = REFERENCES[country]
    path = ROOT / "tests/fixtures" / filename
    record = {"case": f"{country}/{name}", "expected_fields": len(flatten(expected))}
    try:
        evidence = main.document_text(str(path)).evidence
        result = reconcile(evidence, [read_model(json.dumps(expected))])
        actual = flatten(result.payload)
        expected_values = flatten(expected)
        wrong = [
            {"field": field, "actual": value, "expected": expected_values.get(field)}
            for field, value in actual.items()
            if value is not None and not equal(value, expected_values.get(field))
        ]
        null_paths = [
            field for field, value in expected_values.items()
            if value is not None and actual.get(field) is None
        ]
        anomalies = [
            flag for flag in result.validation.export()["flags"] if flag["severity"] != "information"
        ]
        record.update(
            null_fields=len(null_paths), null_paths=null_paths, wrong_values=wrong,
            anomalies=anomalies, crash=None, ocr_calls=evidence.ocr_calls,
            source_kind=evidence.source_kind,
        )
    except Exception as error:  # noqa: BLE001 - a crash is a measured result here
        import traceback
        record.update(
            null_fields=len(flatten(expected)), null_paths=[], wrong_values=[],
            anomalies=[], crash={"type": type(error).__name__, "message": str(error)},
            trace=traceback.format_exc(), ocr_calls=0, source_kind="unavailable",
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    import shutil
    if not shutil.which("tesseract"):
        raise SystemExit("Real Tesseract is required: this check must not be simulated.")
    cases = [run_case(country, name, filename) for country, name, filename in FIXTURES]
    summary = {
        "cases": len(cases),
        "expected_fields": sum(case["expected_fields"] for case in cases),
        "null_fields": sum(case["null_fields"] for case in cases),
        "wrong_values": sum(len(case["wrong_values"]) for case in cases),
        "anomalies": sum(len(case["anomalies"]) for case in cases),
        "crashes": sum(case["crash"] is not None for case in cases),
    }
    print(json.dumps({"degraded_9": summary, "degraded_cases": cases},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
