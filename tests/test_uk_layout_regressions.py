import copy
import shutil
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from app import main
from invoice_evidence import from_text
from invoice_grounding import NULL, RECOVERED, UNVERIFIED, VALID, read_model, reconcile
from ocr_consensus import build_consensus

EMPTY_MODEL = [read_model('{"line_items":[]}')]


def invoice_text(header, rows, extra=""):
    return (
        "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        f"{header}\n{rows}\n{extra}"
    )


def supplier_text(prefix="", footer=""):
    return (
        "INVOICE FR-001\n"
        f"{prefix}"
        "Invoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Total\n"
        "Service | 1 | 100 | 100\n"
        "Total Due: 100\n"
        f"{footer}"
    )


def require_real_tesseract():
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is not installed")
    if not {"fra", "eng"} <= set(pytesseract.get_languages()):
        pytest.skip("Tesseract fra/eng languages are missing")


def diagnostic(name):
    path = ROOT / "diagnostics" / f"{name}.png"
    if not path.is_file():
        pytest.skip(f"Diagnostic image missing: {path}")
    return path


def ocr_result(monkeypatch, name, fast_path):
    require_real_tesseract()
    if fast_path:
        monkeypatch.setenv("INVOICE_OCR_FAST_PATH", "1")
    else:
        monkeypatch.delenv("INVOICE_OCR_FAST_PATH", raising=False)
    evidence = main.document_text(str(diagnostic(name))).evidence
    return evidence, reconcile(evidence, EMPTY_MODEL)


def test_table_header_aliases_accept_amount_and_optional_columns():
    text = invoice_text(
        "No. | Description | Quantity | Unit | Unit price | Total",
        "1 | Product A | 15 | Piece | 10.00 | 150.00",
    )
    evidence = from_text(text)
    assert len(evidence.rows) == 1
    row = evidence.rows[0]
    assert row.description == "Product A"
    assert [n.value for n in row.cells["quantity"]] == [Decimal("15")]
    assert [n.value for n in row.cells["unit_price"]] == [Decimal("10.00")]
    assert [n.value for n in row.cells["total_ht"]] == [Decimal("150.00")]


def test_table_header_aliases_accept_currency_annotated_amount():
    text = invoice_text(
        "DESCRIPTION | QUANTITY | UNIT PRICE (£) | AMOUNT (£)",
        "Sample service | 1 | 400.00 | 400.00",
    )
    evidence = from_text(text)
    assert len(evidence.rows) == 1
    assert evidence.rows[0].description == "Sample service"
    assert [n.value for n in evidence.rows[0].cells["total_ht"]] == [Decimal("400.00")]


def test_isolated_total_label_is_not_a_table_header():
    text = invoice_text(
        "Description | Quantity | Unit price",
        "Total 100",
    )
    evidence = from_text(text)
    assert not evidence.rows
    assert evidence.totals.get("total_candidate")


def test_date_labels_exclude_due_and_delivery_dates():
    text = (
        "Supplier: Acme Ltd\nDue date: 30/04/2022\nDelivery date: 30/04/2022\n"
        "Currency: GBP\nDescription | Quantity | Unit price | Total\n"
        "Service | 1 | 100 | 100\nTotal Due: 100"
    )
    evidence = from_text(text)
    assert evidence.invoice_date is None


def test_total_label_prefix_and_currency_annotation_are_normalized():
    text = (
        "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Total\n"
        "Service | 1 | 275 | 275\n"
        "‘Subtotal £275.00\nVAT (20%) £55.00\nTotal Due £330.00"
    )
    evidence = from_text(text)
    assert [n.value for n in evidence.totals["total_ht"]] == [Decimal("275.00")]
    assert [n.value for n in evidence.totals["total_ttc"]] == [Decimal("330.00")]


def test_explicit_total_without_vat_maps_to_ttc_only():
    text = (
        "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Sample service | 1 | 400.00 | 400.00\n"
        "Sample service 1 | 1 | 200.00 | 200.00\n"
        "TOTAL (GBP): £600.00"
    )
    evidence = from_text(text)
    result = reconcile(evidence, EMPTY_MODEL)
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] == 600.00


def test_conflicting_unqualified_totals_do_not_map_to_ttc():
    text = (
        "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Sample service | 1 | 400.00 | 400.00\n"
        "TOTAL (GBP): £600.00\nTOTAL (GBP): £700.00"
    )
    evidence = from_text(text)
    result = reconcile(evidence, EMPTY_MODEL)
    assert result.payload["total_ttc"] is None
    assert result.payload["total_ht"] is None


def test_supplier_letterhead_with_address_is_extracted():
    evidence = from_text(supplier_text("Acme Ltd\n12 Main Street\n"))
    assert evidence.supplier == "Acme Ltd"


def test_supplier_footer_with_legal_suffix_is_extracted():
    evidence = from_text(supplier_text(footer="Acme Ltd.\n"))
    assert evidence.supplier == "Acme Ltd"


def test_supplier_is_not_taken_from_bill_to():
    evidence = from_text(supplier_text("Bill to:\nCustomer LLC\n"))
    assert evidence.supplier is None


def test_supplier_is_not_taken_from_contact_person():
    evidence = from_text(supplier_text("Contact: Jane Doe\nEmail: jane@example.com\n"))
    assert evidence.supplier is None


def test_supplier_requires_one_unambiguous_candidate():
    evidence = from_text(supplier_text("Acme Ltd\nBeta LLC\n"))
    assert evidence.supplier is None


def test_supplier_comes_from_letterhead_not_bill_to():
    evidence = from_text(supplier_text("Acme Ltd\nBill to:\nCustomer LLC\n"))
    assert evidence.supplier == "Acme Ltd"


def _multiline_evidence():
    return from_text(
        "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Total\n"
        "Sample service | 1 | 400 | 400\n"
        "Sample service 1 | 1 | 200 | 200\n"
        "Total Due: 600"
    )


def test_primary_description_survives_when_only_one_family_has_continuation():
    base = _multiline_evidence()
    with_secondary = copy.deepcopy(base)
    with_secondary.rows[0].secondary_description = "Sample wood decoration service"
    consensus = build_consensus(
        {"deskew": [base], "clahe": [with_secondary]}, confidence=40,
    )
    assert consensus.rows[0].description == "Sample service"
    assert consensus.rows[1].description == "Sample service 1"


def test_corroborated_secondary_description_is_appended():
    first = _multiline_evidence()
    second = copy.deepcopy(first)
    first.rows[0].secondary_description = "Sample wood decoration service"
    second.rows[0].secondary_description = "Sample wood decoration service"
    consensus = build_consensus(
        {"deskew": [first], "clahe": [second]}, confidence=40,
    )
    assert consensus.rows[0].description == (
        "Sample service Sample wood decoration service"
    )
    assert consensus.rows[1].description == "Sample service 1"


def test_conflicting_secondary_descriptions_do_not_merge():
    first = _multiline_evidence()
    second = copy.deepcopy(first)
    first.rows[0].secondary_description = "Sample wood decoration service"
    second.rows[0].secondary_description = "Different decoration service"
    consensus = build_consensus(
        {"deskew": [first], "clahe": [second]}, confidence=40,
    )
    assert consensus.rows[0].description == "Sample service"
    assert consensus.rows[1].description == "Sample service 1"


def test_corrupted_secondary_description_is_not_merged():
    first = _multiline_evidence()
    second = copy.deepcopy(first)
    first.rows[0].secondary_description = "Sample wood decoration service"
    second.rows[0].secondary_description = "Cr\ufffdation service"
    consensus = build_consensus(
        {"deskew": [first], "clahe": [second]}, confidence=40,
    )
    assert consensus.rows[0].description == "Sample service"


@pytest.mark.parametrize("fast_path", [False, True])
def test_example_company_recovers_date_and_totals(monkeypatch, fast_path):
    evidence, result = ocr_result(monkeypatch, "uk-example-company", fast_path)
    assert evidence.invoice_date == "2025-05-11"
    assert result.payload["total_ht"] == 275.00
    assert result.payload["total_ttc"] == 330.00
    assert result.payload["supplier_name"] == "Example Company Ltd"
    assert result.payload["line_items"] == [
        {"description": "Product A", "quantity": 15, "unit_price": 10, "total_ht": 150},
        {"description": "Product B", "quantity": 10, "unit_price": 5, "total_ht": 50},
        {"description": "Product C", "quantity": 3, "unit_price": 25, "total_ht": 75},
    ]


@pytest.mark.parametrize("fast_path", [False, True])
def test_ellington_recovers_date_total_and_individual_table_cells(monkeypatch, fast_path):
    evidence, result = ocr_result(monkeypatch, "uk-ellington-wood", fast_path)
    assert evidence.invoice_date == "2022-04-30"
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] == 600.00
    assert result.payload["supplier_name"] == "Ellington Wood Decor"
    items = result.payload["line_items"]
    assert len(items) == 2
    assert [item["quantity"] for item in items] == [1.0, 1.0]
    assert [item["unit_price"] for item in items] == [400.0, 200.0]
    assert [item["total_ht"] for item in items] == [400.0, 200.0]
    assert items[0]["description"].startswith("Sample service")
    assert items[1]["description"] == "Sample service 1 Sample wood decoration service 1"
    for index, item in enumerate(items):
        if item["description"] is None:
            assert result.statuses[f"line_items.{index}.description"] in {NULL, UNVERIFIED}
        else:
            assert "\ufffd" not in item["description"]
            assert result.statuses[f"line_items.{index}.description"] in {VALID, RECOVERED}
    assert all(
        value is None or "\ufffd" not in value
        for item in items for value in item.values() if isinstance(value, str)
    )
