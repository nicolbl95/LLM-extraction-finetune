"""Run actual application calls, never substitute simulated model responses."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_hash() -> str:
    files = [ROOT / "app/main.py", *sorted((ROOT / "src").glob("*.py")), ROOT / "data/schema.py"]
    return hashlib.sha256(b"".join(
        str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() for path in files
    )).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-real-model", action="store_true", help="Fail if any image has no completed actual generation")
    args = parser.parse_args()
    from app import main as application
    from invoice_evidence import EvidenceText
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fingerprint = source_hash()
    actual_generations = 0
    records = []
    for label, path in application.EXAMPLES:
        text = application.document_text(str(path))
        analysis = {}
        payload = application.extract_invoice(text, diagnostics=analysis)
        raw_responses = [attempt["raw"] for attempt in analysis["attempts"] if attempt.get("raw")]
        actual_generations += int(bool(raw_responses))
        records.append({
            "invoice": label, "head": revision, "source_sha256": fingerprint,
            "ocr_available": isinstance(text, EvidenceText) and text.evidence.available,
            "ocr_lines": analysis["ocr_lines"],
            "inference_requested": analysis["attempt_count"], "completed_generations": len(raw_responses),
            "raw_model_json": raw_responses, "final_json": payload, "validation": analysis["validation"],
            "technical_errors": analysis["validation_errors"], "report": analysis["report"],
        })
    print(json.dumps(records, ensure_ascii=False, indent=2))
    if args.require_real_model and actual_generations != len(application.EXAMPLES):
        raise SystemExit("Real-model requirement not satisfied: inspect technical_errors; OCR-only outputs are not model results.")


if __name__ == "__main__":
    main()
