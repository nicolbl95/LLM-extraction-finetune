import inspect
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import EvidenceText, combine_pages, from_text
from invoice_grounding import reconcile
from ocr_budget import OCRBudgetExceeded, ProcessingBudget
from ocr_config import OCR_CONFIG
from ocr_consensus import build_consensus
from ocr_recovery import CONFIDENCE, FAMILIES, recover_image


TEXT = (
    "Supplier: Budget Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
    "Description | Quantity | Net unit price | Net amount\nService | 2 | 10 | 20\n"
    "Subtotal: 20\nVAT (0%): 0\nInvoice total: 20"
)


def test_policy_is_unique_frozen_versioned_and_fingerprinted():
    assert OCR_CONFIG.version
    assert CONFIDENCE == OCR_CONFIG.confidence == 40
    assert FAMILIES is OCR_CONFIG.families
    assert FAMILIES == ("deskew", "clahe", "crop", "denoise", "scale3")
    assert len(set(OCR_CONFIG.fast_families)) == 2
    assert OCR_CONFIG.fast_families == ("deskew", "clahe")
    assert inspect.signature(build_consensus).parameters["confidence"].default == OCR_CONFIG.confidence
    assert len(OCR_CONFIG.fingerprint()) == 64
    assert json.dumps(OCR_CONFIG.export(), sort_keys=True)
    with pytest.raises(AttributeError):
        OCR_CONFIG.confidence = 0


def test_budget_is_shared_and_never_resets_at_remaining_calls():
    clock = [0.0]
    budget = ProcessingBudget(45, clock=lambda: clock[0])
    clock[0] = 20
    assert budget.remaining() == 25
    clock[0] = 44.5
    assert budget.remaining() == .5
    clock[0] = 45
    with pytest.raises(OCRBudgetExceeded, match="45 s"):
        budget.remaining()


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected_explicitly(seconds):
    with pytest.raises(ValueError, match="positive"):
        ProcessingBudget(seconds)


def test_expired_budget_does_not_start_preprocessing():
    clock = [0]
    budget = ProcessingBudget(1, clock=lambda: clock[0])
    clock[0] = 2
    with patch("ocr_recovery.deskew", side_effect=AssertionError("Must not preprocess")):
        with pytest.raises(OCRBudgetExceeded):
            recover_image(Image.new("L", (20, 20)), lambda *args: {}, budget=budget)


def test_deadline_after_first_read_stops_all_additional_work():
    clock = [0]
    budget = ProcessingBudget(2, clock=lambda: clock[0])
    calls = []

    def read(image, psm):
        calls.append(psm)
        clock[0] = 3
        return {"text": []}

    with pytest.raises(OCRBudgetExceeded):
        recover_image(Image.new("L", (20, 20)), read, budget=budget)
    assert calls == [4]


def test_clear_first_pair_never_runs_extra_families():
    def parsed(data):
        return from_text(TEXT)
    with patch("ocr_recovery.from_tsv", side_effect=parsed), \
            patch("ocr_recovery.transform", wraps=__import__("ocr_recovery").transform) as transform:
        calls = []
        result = recover_image(Image.new("L", (20, 20)), lambda image, psm: calls.append(psm) or {})
    assert calls == [4, 6]
    assert [call.args[2] for call in transform.call_args_list] == ["deskew", "clahe"]
    assert not result.quality_issues
    assert reconcile(result, [{}]).payload["line_items"][0]["quantity"] == 2


def test_tesseract_timeout_is_bounded_by_remaining_document_budget():
    from app import main
    clock = [0]
    budget = ProcessingBudget(45, clock=lambda: clock[0])
    timeouts = []

    def actual_stub(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        clock[0] = 46
        raise RuntimeError("Tesseract process timeout")

    with patch.object(main.pytesseract, "image_to_data", side_effect=actual_stub):
        with pytest.raises(OCRBudgetExceeded):
            main.ocr_image(Image.new("RGB", (20, 20)), budget=budget)
    assert len(timeouts) == 1
    assert 0 < timeouts[0] <= 45


def test_scanned_pdf_pages_share_the_same_budget_without_reparse_of_rejected_numbers():
    from app import main
    clock = [0]
    budget = ProcessingBudget(45, clock=lambda: clock[0])
    budgets = []

    def scan(image, *, budget):
        budgets.append(budget)
        clock[0] += 25
        proof = from_text(TEXT)
        proof.source_kind = "consensus"
        proof.ocr_cross_checked = True
        proof.rows[0].cells["quantity"] = []
        return EvidenceText(proof)

    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
        path = Path(directory) / "scanned.pdf"
        with pymupdf.open() as pdf:
            for _ in range(3):
                pdf.new_page()
            pdf.save(path)
        with patch.object(main, "ocr_image", side_effect=scan):
            with pytest.raises(OCRBudgetExceeded):
                main.document_text(str(path), budget=budget)
    assert len(budgets) == 2
    assert all(seen is budget for seen in budgets)


def test_merge_pages_does_not_recreate_a_number_that_ocr_rejected():
    evidence = from_text(TEXT)
    evidence.source_kind = "consensus"
    evidence.ocr_cross_checked = True
    evidence.rows[0].cells["quantity"] = []
    evidence.numbers = [number for number in evidence.numbers if number.value != 2]
    merged = combine_pages([evidence], contains_ocr=True)
    assert merged.rows[0].cells["quantity"] == []
    result = reconcile(merged, [{}])
    assert result.payload["line_items"][0]["quantity"] is None
    assert result.payload["line_items"][0]["unit_price"] == 10
    assert result.payload["line_items"][0]["total_ht"] == 20


def test_budget_failure_ui_is_insufficient_and_never_loads_model():
    from app import main
    with patch.object(main, "document_text", side_effect=OCRBudgetExceeded("Budget documentaire OCR dépassé")), \
            patch.object(main, "predict_invoice", side_effect=AssertionError("No GPU")):
        payload, report, technical, validation = list(main.run_extraction_ui(str(main.EXAMPLES[0][1])))[-1]
    assert "Preuves OCR insuffisantes" in report
    assert payload["total_ht"] is None
    assert not validation["flags"]
    assert "Budget" in technical


def test_measured_latency_candidate_passes_only_the_explicit_safety_and_null_gate():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_stress_check import compare_latency_candidate
    baseline = {"cases": [{"case": "x"}], "summary": {"expected_fields": 612, "false_nulls": 155}}

    def candidate(nulls, wrong=0):
        return {"cases": [{"case": "x"}], "summary": {
            "expected_fields": 612, "false_nulls": nulls, "wrong_values": wrong,
            "phantom_anomalies": 0, "crashes": 0,
        }, "additional_model_errors": {"wrong_values": 0, "phantom_anomalies": 0}}

    assert compare_latency_candidate(candidate(153), baseline, "reference")["passed"]
    assert compare_latency_candidate(candidate(167), baseline, "reference")["passed"]
    assert not compare_latency_candidate(candidate(168), baseline, "reference")["passed"]
    assert not compare_latency_candidate(candidate(100, wrong=1), baseline, "reference")["passed"]
    mismatched = candidate(153)
    mismatched["cases"] = [{"case": "another"}]
    with pytest.raises(ValueError, match="exact same"):
        compare_latency_candidate(mismatched, baseline, "reference")
