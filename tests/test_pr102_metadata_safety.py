"""PR102 Priority 5: metadata safety. Cross-engine agreement is not metadata verification.

Synthetic and deterministic: no OCR, no model, no invoice data. Every scenario asserts on the exported
strict JSON (through ``reconcile``) and on the arbitration record, never on tokens alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pytest

import dual_ocr_experiment as experiment
import evaluation_metrics as metrics
from invoice_evidence import Token, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus

HEADER_Y = 100
COLUMN_X = (60, 200, 380, 520, 660)
LABELS = ["Description", "Quantité", "Prix unitaire", "", "Montant HT"]
ROW = [["Prestation", "2", "40,00", "", "80,00"]]


def token(text, x, y, confidence=95.0):
    return Token(text, 0, (x, y, x + 12 + 5 * len(text), y + 14), confidence)


def reading(extra_lines=(), supplier=None):
    tokens = [token(label, x, HEADER_Y) for label, x in zip(LABELS, COLUMN_X) if label]
    for value, x in zip(ROW[0], COLUMN_X):
        if value:
            tokens.append(token(value, x, HEADER_Y + 30))
    for index, text in enumerate(extra_lines):
        tokens.append(token(text, 60, 20 + index * 16))
    evidence = from_tokens(tokens, literal_columns=False)
    if supplier is not None:
        evidence.supplier = supplier
    return evidence


def consensus(evidence, supplier=None):
    combined = build_consensus({"gray": [evidence], "contrast": [evidence]})
    # build_consensus constructs fresh evidence and does not carry a supplier across, so the fixture sets
    # it on the consensus output — the same place the production read path leaves it.
    if supplier is not None:
        combined.supplier = supplier
    return combined


def exported(combined):
    return reconcile(combined, [{}]).payload


def supplier_entry(record):
    return [item for item in record["document"] if item["field"] == "supplier"]


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_both_engines_agreeing_on_an_incorrect_supplier_never_exports_it(strategy):
    primary = consensus(reading(supplier=None))
    other = reading(supplier="Photohgraphy")
    combined, record = experiment.arbitrate(primary, other, strategy)
    assert combined.supplier is None
    assert exported(combined)["supplier_name"] is None
    assert supplier_entry(record)[0]["provenance"] == experiment.DROPPED


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_one_engine_correct_and_the_other_incorrect_does_not_overwrite(strategy):
    primary = consensus(reading(), supplier="Correct Supplier")
    other = reading(supplier="Wrong Supplier")
    combined, _ = experiment.arbitrate(primary, other, strategy)
    assert combined.supplier == "Correct Supplier"
    assert exported(combined)["supplier_name"] == "Correct Supplier"


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_a_correct_supplier_read_only_by_the_second_engine_is_still_not_adopted(strategy):
    primary = consensus(reading(supplier=None))
    other = reading(supplier="Correct Supplier")
    combined, record = experiment.arbitrate(primary, other, strategy)
    assert combined.supplier is None
    assert exported(combined)["supplier_name"] is None
    assert supplier_entry(record)[0]["provenance"] == experiment.DROPPED


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_a_stylized_logo_read_as_garbage_by_both_engines_exports_no_supplier(strategy):
    primary = consensus(reading(extra_lines=["~~~ LOGO ~~~"], supplier=None))
    other = reading(extra_lines=["~~~ L0G0 ~~~"], supplier="~~~ L0G0 ~~~")
    combined, _ = experiment.arbitrate(primary, other, strategy)
    assert combined.supplier is None
    assert exported(combined)["supplier_name"] is None


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_the_billing_recipient_is_not_promoted_to_supplier(strategy):
    primary = consensus(reading(extra_lines=["Billed to App Revolution"], supplier=None))
    other = reading(extra_lines=["Billed to App Revolution"], supplier="App Revolution")
    combined, _ = experiment.arbitrate(primary, other, strategy)
    assert combined.supplier is None
    assert exported(combined)["supplier_name"] is None


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_a_name_in_payment_instructions_is_not_promoted_to_supplier(strategy):
    primary = consensus(reading(extra_lines=["Please make payment to Future Graphics"], supplier=None))
    other = reading(extra_lines=["Please make payment to Future Graphics"], supplier="Future Graphics")
    combined, _ = experiment.arbitrate(primary, other, strategy)
    assert combined.supplier is None
    assert exported(combined)["supplier_name"] is None


def test_the_evaluator_separates_an_incorrect_supplier_from_a_missing_one():
    reference = {"line_items": [], "supplier_name": "Real Supplier"}
    wrong = metrics.evaluate({"line_items": [], "supplier_name": "Wrong"}, reference)
    absent = metrics.evaluate({"line_items": [], "supplier_name": None}, reference)
    assert wrong["header"]["supplier_name"] == metrics.INCORRECT
    assert absent["header"]["supplier_name"] == metrics.MISSING
    assert wrong["header_summary"]["incorrect_supplier"] is True
    assert absent["header_summary"]["incorrect_supplier"] is False


def test_the_evaluator_separates_an_incorrect_date_from_a_missing_one():
    reference = {"line_items": [], "invoice_date": "2025-09-01"}
    wrong = metrics.evaluate({"line_items": [], "invoice_date": "2025-01-09"}, reference)
    absent = metrics.evaluate({"line_items": [], "invoice_date": None}, reference)
    assert wrong["header"]["invoice_date"] == metrics.INCORRECT
    assert absent["header"]["invoice_date"] == metrics.MISSING
    assert wrong["header_summary"]["incorrect_date"] is True
    assert absent["header_summary"]["incorrect_date"] is False


def test_an_unambiguous_sibling_date_resolves_the_order_without_a_global_assumption():
    from invoice_evidence import document_date_locale, parse_document_date

    # 09.14.2025 can only be month-first: no month is 14. That single printed date settles the document.
    readings = [("invoice", "09.01.2025", None), ("due", "09.14.2025", None)]
    assert document_date_locale(readings, None) == "en-US"
    assert parse_document_date("09.01.2025", "en-US")[0].isoformat() == "2025-09-01"


def test_two_ambiguous_dates_alone_do_not_establish_an_order():
    from invoice_evidence import document_date_locale, parse_document_date

    readings = [("invoice", "09.01.2025", None), ("due", "09.05.2025", None)]
    assert document_date_locale(readings, None) is None
    # With no order established the parser must not silently pick one.
    assert parse_document_date("09.01.2025", None)[0] is None


def test_a_day_first_sibling_date_resolves_the_other_order():
    from invoice_evidence import document_date_locale, parse_document_date

    readings = [("invoice", "01.09.2025", None), ("due", "14.09.2025", None)]
    assert document_date_locale(readings, None) == "fr-FR"
    assert parse_document_date("01.09.2025", "fr-FR")[0].isoformat() == "2025-09-01"
