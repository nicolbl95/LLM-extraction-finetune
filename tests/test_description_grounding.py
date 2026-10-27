import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text
from invoice_grounding import NULL, RECOVERED, read_model, reconcile
from ocr_consensus import build_consensus


BASE = (
    "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Clavier sans fil | 2 | 10 | 20\n"
    "Subtotal: 20\nVAT (0%): 0.00\nInvoice total: 20"
)

UK_ROW = (
    "Supplier: Oak Studio Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Logo design | {quantity} | 757.97 | 5912.17\n"
    "Subtotal: 5912.17\nVAT (0%): 0.00\nInvoice total: 5912.17"
)


def _uk(quantity):
    return from_text(UK_ROW.format(quantity=quantity), locale="en-GB")


def test_confident_ocr_description_survives_rejected_quantity_and_price():
    evidence = build_consensus({
        "deskew": [_uk("78")],
        "clahe": [_uk("78")],
        "crop": [_uk("78")],
        "denoise": [_uk("78")],
        "scale3": [_uk("7.8")],
    }, confidence=40)
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    item = result.payload["line_items"][0]
    assert item["description"] == "Logo design"
    assert item["quantity"] is None
    assert item["unit_price"] is None
    assert item["total_ht"] == 5912.17
    assert result.statuses["line_items.0.description"] == RECOVERED
    assert 78 not in item.values()


def test_shifted_numeric_description_is_not_exported():
    evidence = from_text(
        "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Net unit price | Quantity\n"
        "1 234,56 EUR | 2 | 10 | 20\n"
        "Subtotal: 20\nVAT (0%): 0.00\nInvoice total: 20",
        locale="en-GB",
    )
    assert evidence.rows == []
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    assert all(item["description"] is None for item in result.payload["line_items"])


def test_monetary_value_in_description_column_is_not_exported():
    evidence = from_text(BASE, locale="en-GB")
    assert len(evidence.rows) == 1
    evidence.rows[0].description = "1 234,56 EUR"
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    item = result.payload["line_items"][0]
    assert item["description"] is None
    assert item["quantity"] == 2
    assert item["unit_price"] == 10
    assert item["total_ht"] == 20
    assert result.statuses["line_items.0.description"] in {NULL, "non vérifié"}


def test_conflicting_ocr_descriptions_are_not_exported():
    wrong = BASE.replace("Clavier sans fil", "Ciavier sans fil")
    evidence = build_consensus({
        "deskew": [from_text(BASE, locale="en-GB")],
        "clahe": [from_text(wrong, locale="en-GB")],
    }, confidence=40)
    assert evidence.rows[0].description == ""
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    assert result.payload["line_items"][0]["description"] is None


def test_missing_description_is_not_filled_from_the_model():
    evidence = from_text(
        "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Net unit price | Net amount\n"
        " | 2 | 10 | 20\nSubtotal: 20\nVAT (0%): 0.00\nInvoice total: 20",
        locale="en-GB",
    )
    model = read_model(
        '{"line_items":[{"description":"Inventé","quantity":2,'
        '"unit_price":10,"total_ht":20}]}'
    )
    result = reconcile(evidence, [model])
    assert result.payload["line_items"][0]["description"] is None


def test_verified_description_with_numeric_fields_is_exported():
    evidence = build_consensus({
        "deskew": [from_text(BASE, locale="en-GB")],
        "clahe": [from_text(BASE, locale="en-GB")],
    }, confidence=40)
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    item = result.payload["line_items"][0]
    assert item["description"] == "Clavier sans fil"
    assert item["quantity"] == 2
    assert item["unit_price"] == 10
    assert item["total_ht"] == 20
