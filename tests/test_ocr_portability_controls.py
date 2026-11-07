"""Negative controls for uncalibrated OCR confidence and exact signature comparison."""

import copy
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from invoice_evidence import from_text, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus
from ocr_quality import signature
from test_registered_office_supplier import tokens


def test_repetition_cannot_rescue_low_confidence_issuer_attribution():
    """Repeatable synthetic hallucinations are not independent documentary facts."""
    source = [
        replace(token, confidence=30) if token.box[1] == 200 else token
        for token in tokens()
    ]
    readings = {
        "gray": [from_tokens(source, locale="fr-FR")],
        "contrast": [from_tokens(copy.deepcopy(source), locale="fr-FR")],
        "crop": [from_tokens(copy.deepcopy(source), locale="fr-FR")],
    }
    result = build_consensus(readings)
    assert result.supplier is None
    assert reconcile(result, [{"supplier_name": "Actual Studio"}]).payload["supplier_name"] is None


def test_similar_issuer_and_customer_names_do_not_establish_supplier_agreement():
    first = from_tokens(tokens(), locale="fr-FR")
    second = from_tokens([
        replace(token, text="Actuals") if token.text == "Actual" else token
        for token in tokens()
    ], locale="fr-FR")
    result = build_consensus({"gray": [first], "contrast": [second]})
    assert result.supplier is None
    assert reconcile(result, [{"supplier_name": "Actual Studio"}]).payload["supplier_name"] is None


def _diagnostics():
    file = ROOT / ".venv/product-review37/candidate/app/private_staging_validation.py"
    if not file.is_file():
        pytest.skip("Isolated private diagnostic candidate is not present in this checkout")
    spec = importlib.util.spec_from_file_location("portability_diagnostics", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document():
    return from_text(
        "Supplier: Control Studio Ltd\nInvoice date: 2025-05-18\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Service | 7.8 | 10 | 78\nSubtotal: 78\nVAT (0%): 0\nInvoice total: 78"
    )


def test_signature_ignores_unrelated_ocr_footer_noise_but_not_real_cell_disagreement():
    left = _document()
    right = copy.deepcopy(left)
    right.text += "\nUnrelated noisy payment instructions"
    assert signature(left) == signature(right)
    right.rows[0].cells["quantity"] = [
        replace(right.rows[0].cells["quantity"][0], value=78)
    ]
    assert signature(left) != signature(right)
    comparison = _diagnostics()._signature_comparison([left, right])
    assert comparison["mismatching_components"] == ["rows"]
    assert comparison["rows"][0]["cells_equal"]["quantity"] is False
    assert comparison["rows"][0]["cells_equal"]["unit_price"] is True
    assert comparison["early_path_signature_equal"] is False
    assert "Control Studio" not in str(comparison)
    assert "payment" not in str(comparison)


def test_invoice_number_disagreement_is_already_excluded_from_early_comparison():
    left, right = _document(), _document()
    left.invoice_number = "FIRST"
    right.invoice_number = None
    comparison = _diagnostics()._signature_comparison([left, right])
    assert comparison["mismatching_components"] == ["invoice_number"]
    assert comparison["full_signature_equal"] is False
    assert comparison["early_path_signature_equal"] is True


def test_diagnostic_reports_each_signature_component_without_value_disclosure():
    left, right = _document(), _document()
    right.supplier = None
    right.invoice_date = None
    right.totals["total_ht"] = []
    right.taxes = []
    comparison = _diagnostics()._signature_comparison([left, right])
    assert set(comparison["mismatching_components"]) == {
        "supplier", "invoice_date", "totals", "taxes",
    }
    assert comparison["invoice_totals_equal"]["total_ht"] is False
    assert comparison["invoice_totals_equal"]["total_ttc"] is True
    assert comparison["no_signature_values_returned"]


def test_oak_observed_mismatch_shape_requires_more_documentary_quantity_evidence():
    """Synthetic candidate values; staging proved the field disagreement, not these raw values."""
    text = (
        "Supplier: Example Studio Ltd\nInvoice date: 2025-05-18\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Design | 7.8 | 757.97 | 5912.17\n"
        "Subtotal: 5912.17\nVAT (0%): 0\nInvoice total: 5912.17"
    )
    correct = from_text(text)
    conflicting = from_text(text.replace("Design | 7.8 |", "Design | 78 |"))
    correct.invoice_number = "READABLE"
    conflicting.invoice_number = None
    comparison = _diagnostics()._signature_comparison([correct, conflicting])
    assert set(comparison["mismatching_components"]) == {"invoice_number", "rows"}
    assert comparison["early_path_signature_equal"] is False
    assert comparison["rows"][0]["cells_equal"] == {
        "quantity": False, "unit_price": True, "total_ht": True,
    }
    first_pair = build_consensus({"deskew": [correct], "clahe": [conflicting]})
    assert first_pair.rows[0].cells["quantity"] == []
    guessed = {"line_items": [{
        "description": "Design", "quantity": 7.8,
        "unit_price": 757.97, "total_ht": 5912.17,
    }]}
    assert reconcile(first_pair, [guessed]).payload["line_items"][0]["quantity"] is None
    confirmed = build_consensus({
        "deskew": [correct], "clahe": [conflicting],
        "crop": [copy.deepcopy(correct)], "denoise": [copy.deepcopy(correct)],
    })
    assert reconcile(confirmed, [guessed]).payload["line_items"][0]["quantity"] == 7.8
