#!/usr/bin/env python3
"""Actual CPU OCR timings and Tesseract calls; no model or GPU is invoked."""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/stress")]

from cases import REFERENCES, degrade
from invoice_evidence import DocumentEvidence, UnreadableOCR
from invoice_grounding import reconcile
from ocr_config import OCR_CONFIG


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        raise ValueError("No timing samples")
    position = (len(values) - 1) * fraction
    left = int(position)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)


def sample(name, category, read):
    from app import main
    calls = []
    actual = main.pytesseract.image_to_data

    def counted(*args, **kwargs):
        started = time.perf_counter()
        try:
            return actual(*args, **kwargs)
        finally:
            calls.append({"seconds": time.perf_counter() - started,
                          "psm": kwargs.get("config"), "timeout": kwargs.get("timeout")})

    started = time.perf_counter()
    error = None
    with patch.object(main.pytesseract, "image_to_data", side_effect=counted):
        try:
            text = read()
            evidence = text.evidence
        except UnreadableOCR as exc:
            error = f"{type(exc).__name__}: {exc}"
            evidence = DocumentEvidence("", None, [], [], quality_issues=[str(exc)])
        output = reconcile(evidence, [{}])
    return {
        "sample": name, "category": category, "seconds": time.perf_counter() - started,
        "tesseract_calls": len(calls), "calls": calls, "technical_error": error,
        "quality_issues": evidence.quality_issues, "recognized_rows": len(evidence.rows),
        "final": output.payload, "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scanned-pdf-only", action="store_true", help="Measure three actual 10-page scanned PDFs under the shared deadline")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing measurements")
    from app import main
    from PIL import Image
    import pymupdf
    records = []
    with tempfile.TemporaryDirectory(dir=ROOT / "tests/stress") as directory:
        pdf_path = Path(directory) / "ten_pages.pdf"
        with pymupdf.open() as pdf:
            for page_index in range(10):
                pdf.new_page().insert_text((40, 40), (
                    f"Invoice page {page_index + 1}\nSupplier: Latency Ltd\nCurrency: GBP\n"
                    "Description | Quantity | Net unit price | Net amount\nService | 2 | 10 | 20"
                ), fontsize=10)
            pdf.save(pdf_path)
        if args.scanned_pdf_only:
            scanned_path = Path(directory) / "ten_scanned_pages.pdf"
            image_bytes = (ROOT / "app/examples/us.png").read_bytes()
            with pymupdf.open() as pdf:
                for _ in range(10):
                    page = pdf.new_page()
                    page.insert_image(page.rect, stream=image_bytes)
                pdf.save(scanned_path)
            for index in range(3):
                records.append(sample(f"ten_scanned_pages/{index + 1}", "scanned_pdf_10_pages",
                                      lambda: main.document_text(str(scanned_path))))
                persist(args.output, records, main)
            return
        for index, name in enumerate(("france", "uk", "us")):
            with Image.open(ROOT / "app/examples" / f"{name}.png") as source:
                image = source.convert("RGB")
            for category, operation in (
                ("clean_image", None), ("degraded_image", ("rotate:1.5", "jpeg40", "noise")[index]),
                ("destroyed_image", "destroyed"),
            ):
                prepared = image if operation is None else degrade(image, operation)
                record = sample(name + "/" + str(operation), category, lambda: main.ocr_image(prepared))
                records.append(record)
                print(category, name, round(record["seconds"], 3), record["tesseract_calls"], flush=True)
                persist(args.output, records, main)
            record = sample(f"ten_native_pages/{index + 1}", "native_pdf_10_pages",
                            lambda: main.document_text(str(pdf_path)))
            records.append(record)
            persist(args.output, records, main)


def persist(path, records, application):
    files = [ROOT / "app/main.py", *sorted((ROOT / "src").glob("*.py"))]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    summaries = {}
    for category in sorted({record["category"] for record in records}):
        subset = [record for record in records if record["category"] == category]
        values = [record["seconds"] for record in subset]
        summaries[category] = {
            "samples": len(values), "median_seconds": statistics.median(values),
            "p95_seconds_linear": percentile(values, .95), "worst_seconds": max(values),
            "tesseract_calls": [record["tesseract_calls"] for record in subset],
        }
    path.write_text(json.dumps({
        "model_called": False, "source_sha256": digest, "platform": platform.platform(),
        "python": sys.version, "tesseract": str(application.pytesseract.get_tesseract_version()),
        "thread_limit": os.environ.get("OMP_THREAD_LIMIT"), "samples": records, "summary": summaries,
        "ocr_configuration": OCR_CONFIG.export(), "ocr_configuration_sha256": OCR_CONFIG.fingerprint(),
        "limits": "Three samples/category only; p95 is interpolation, not a production SLO. RSS is process cumulative maximum. PDF is native text, not ten scanned pages.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
