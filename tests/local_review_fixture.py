"""Synthetic-only server fixture for persistence restart/browser tests."""

import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from PIL import Image, ImageDraw
from invoice_review_store import DraftStore
from app.local_review import launch_local
from test_invoice_review_store import MODEL, REVIEW, STRICT


def configure(main):
    def extraction(filename, diagnostics=None):
        if diagnostics is not None:
            diagnostics.update(raw=json.dumps(MODEL, default=lambda value: float(value)))
        yield STRICT, "SYNTHETIC FIXTURE - no OCR/model calls", "", "", "", {}, None

    original_build = main.build_review

    def review(diagnostics, payload, **kwargs):
        if payload is None:
            return original_build(diagnostics, payload, **kwargs)
        contract = {**STRICT, **REVIEW}
        return {
            "summary": "SYNTHETIC fixture evidence",
            "table": {"headers": list(main.invoice_review.COLUMN_HEADERS),
                      "data": main.invoice_review.review_rows(contract)},
            "json": contract,
        }

    main.run_extraction_ui = extraction
    main.build_review = review
    main.document_text = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("OCR must not run"),
    )
    main.predict_invoice = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Model must not run"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    folder = Path(args.directory).resolve()
    if not folder.is_relative_to(ROOT / ".pytest_cache"):
        raise ValueError("The synthetic fixture must use an isolated pytest cache directory.")
    folder.mkdir(exist_ok=True)
    source = folder / "synthetic.png"
    if not source.exists():
        image = Image.new("RGB", (480, 600), "white")
        ImageDraw.Draw(image).text((20, 20), "SYNTHETIC INVOICE - NOT REAL DATA", fill="black")
        image.save(source)
    demo, boundary = launch_local(
        port=args.port, store=DraftStore(folder / "storage"), configure=configure,
    )
    print(f"SYNTHETIC_READY {demo.local_url} SOURCE={source}", flush=True)
    try:
        threading.Event().wait()
    finally:
        boundary.active = False
        demo.close()
