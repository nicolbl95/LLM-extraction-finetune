import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_real_scans import load_truth, run


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
        yield Path(directory)


def native_invoice(folder):
    folder.mkdir()
    path = folder / "invoice.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 40), (
            "INVOICE GB-2026-00001\nSupplier: Independent Example Ltd\n"
            "Invoice date: 2026-01-01\nCurrency: GBP\n"
            "Description | Quantity | Net unit price | Net amount\n"
            "Service | 2 | 10 | 20\nSubtotal: 20\nVAT (20%): 4\nInvoice total: 24"
        ), fontsize=10)
        pdf.save(path)
    return path


def truth():
    return {
        "supplier_name": "Independent Example Ltd", "invoice_date": "2026-01-01",
        "total_ht": 20, "total_ttc": 24,
        "line_items": [{"description": "Service", "quantity": 2, "unit_price": 10, "total_ht": 20}],
    }


def test_without_truth_real_scan_runner_reports_only_review_candidates(tmp_path):
    native_invoice(tmp_path / "scans")
    from app import main
    with patch.object(main, "get_extractor", side_effect=AssertionError("Never load a model")):
        report = run(tmp_path / "scans")
    assert not report["model_called"]
    assert not report["truth_provided"]
    assert "summary" not in report
    record = report["documents"][0]
    assert record["final"] == truth()
    assert "wrong_values_retained" not in record
    assert "false_nulls" not in record
    assert record["tesseract_calls"] == 0
    assert record["seconds"] >= 0
    assert record["manual_review_required"]


def test_manual_reference_is_scored_and_never_used_to_extract(tmp_path):
    native_invoice(tmp_path / "scans")
    expected = truth()
    expected["line_items"][0]["quantity"] = 3
    reference = tmp_path / "truth.json"
    reference.write_text(json.dumps({"documents": {"invoice.pdf": expected}}))
    report = run(tmp_path / "scans", reference)
    assert report["truth_provided"]
    assert report["summary"]["wrong_values_retained"] == 1
    assert report["summary"]["false_nulls"] == 0
    assert report["documents"][0]["final"]["line_items"][0]["quantity"] == 2
    assert report["truth_sha256"]


def test_partial_filename_truth_is_rejected_before_scoring(tmp_path):
    native_invoice(tmp_path / "scans")
    reference = tmp_path / "truth.json"
    reference.write_text(json.dumps({"documents": {"another.pdf": truth()}}))
    with pytest.raises(ValueError, match="exactly match"):
        run(tmp_path / "scans", reference)


@pytest.mark.parametrize("payload", [[], {}, {"documents": []}, {"documents": {"invoice.pdf": None}}])
def test_invalid_truth_shapes_are_explicit_errors(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Truth|Each truth"):
        load_truth(path)


def test_empty_folder_is_not_a_successful_real_validation(tmp_path):
    with pytest.raises(ValueError, match="No supported"):
        run(tmp_path)
