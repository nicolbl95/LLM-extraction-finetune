"""Geometry fallback: documentary-only rows, OFF by default, never ordinal-bearing."""

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import table_geometry  # noqa: E402
from invoice_evidence import DocumentEvidence, Number, TextLine, Token  # noqa: E402

WIDTH, HEIGHT = 1000, 1400
FLAG = table_geometry.FLAG_NAME


class Reading:
    def __init__(self, lines):
        self.lines = lines


def tok(text, x0, x1, y, page=0, conf=95):
    return Token(text, page, (x0, float(y), float(x1), float(y + 18)), conf)


def line(text, tokens, page=0):
    x0 = min(t.box[0] for t in tokens)
    x1 = max(t.box[2] for t in tokens)
    y0 = min(t.box[1] for t in tokens)
    y1 = max(t.box[3] for t in tokens)
    return TextLine(text, tokens, page, (x0, y0, x1, y1))


def prepared(width=WIDTH, height=HEIGHT):
    return {"deskew": Image.new("RGB", (width, height)),
            "clahe": Image.new("RGB", (width, height))}


def reading(lines):
    return {"deskew": [Reading(lines)]}


def evidence(totals=None, discounts=None):
    return DocumentEvidence("", "fr-FR", [], [], totals=totals or {}, discounts=discounts or [])


def number(raw):
    return Number(raw=raw, value=Decimal(raw), page=0, box=(0, 0, 1, 1))


def row(description, qty, price, amount, y, *, qty_x=500, price_x=700, amount_x=880, amount_split=False):
    tokens = [tok(description, 120, 380, y), tok(qty, qty_x, qty_x + 40, y),
              tok(price, price_x, price_x + 70, y)]
    if amount:
        if amount_split:
            whole, cents = amount.split(" ")
            tokens.append(tok(whole, amount_x - 10, amount_x, y))
            tokens.append(tok(cents, amount_x + 2, amount_x + 80, y))
        else:
            tokens.append(tok(amount, amount_x, amount_x + 80, y))
    return line(" ".join(t.text for t in tokens), tokens)


def test_geometry_is_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    document = evidence()
    result = table_geometry.recover(
        document, reading([row("Alpha", "1", "10.00", "10.00", 100)]), prepared=prepared())
    assert result is document
    assert document.rows == []


def test_merged_rows_with_unknown_header_vocabulary(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    lines = [
        line("Description Heures Taux horaire TOTAL", [tok("Description", 120, 300, 60),
                                                       tok("Heures", 500, 560, 60),
                                                       tok("TOTAL", 880, 950, 60)]),
        row("Lorem ipsum", "2", "50,00", "100,00", 120),
        row("Dolor sit", "10", "40,00", "400,00", 170),
    ]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert len(document.rows) == 2
    assert document.rows_provenance == "recovered"
    assert document.rows[0].cells["quantity"][0].value == Decimal("2")
    assert document.rows[0].cells["unit_price"][0].value == Decimal("50.00")
    assert document.rows[0].cells["total_ht"][0].value == Decimal("100.00")
    assert document.rows[0].row_index is None and document.rows[1].row_index is None
    assert document.rows[0].row_evidence["method"] == "geometry"
    assert document.rows[0].row_evidence["corroboration"]["rule"] == "quantity x unit_price = amount"
    assert document.rows[0].row_evidence["positions"]["quantity"]["pixel"]


def test_single_row_with_discount(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence(totals={"total_ht": [number("140.00")]}, discounts=[number("10.00")])
    table_geometry.recover(document, reading([row("Sample service", "1", "150.00", "150.00", 100)]),
                           prepared=prepared())
    assert len(document.rows) == 1
    assert document.rows[0].cells["total_ht"][0].value == Decimal("150.00")


def test_single_row_without_total_corroboration_is_refused(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence(totals={"total_ht": [number("999.00")]}, discounts=[])
    table_geometry.recover(document, reading([row("Sample service", "1", "150.00", "150.00", 100)]),
                           prepared=prepared())
    assert document.rows == []


def test_decimal_comma_and_split_thousands(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    lines = [
        row("Maintenance", "7,2", "95,07", "684,50", 120),
        row("Création", "4,2", "863,80", "3 627,96", 170, amount_split=True),
    ]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert len(document.rows) == 2
    assert document.rows[0].cells["quantity"][0].value == Decimal("7.2")
    assert document.rows[0].cells["total_ht"][0].value == Decimal("684.50")
    assert document.rows[1].cells["quantity"][0].value == Decimal("4.2")
    assert document.rows[1].cells["total_ht"][0].value == Decimal("3627.96")


def test_ambiguous_quantity_and_price_stay_unassigned(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence(totals={"total_ht": [number("500.00")]})
    lines = [row("Alpha", "10", "10", "100", 120), row("Beta", "20", "20", "400", 170)]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert len(document.rows) == 2
    for item in document.rows:
        assert "quantity" not in item.cells and "unit_price" not in item.cells
        assert item.cells["total_ht"][0].value in {Decimal("100"), Decimal("400")}
        assert {"quantity", "unit_price"} <= set(item.row_evidence["refused_cells"])


def test_missing_numeric_columns_produce_no_rows(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    lines = [line("Lorem ipsum dolor 300,00", [tok("Lorem", 120, 200, 100), tok("300,00", 800, 880, 100)]),
             line("Dolor sit amet 50,00", [tok("Dolor", 120, 200, 150), tok("50,00", 800, 880, 150)])]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert document.rows == []


def test_summary_lines_never_become_items(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence(totals={"total_ht": [number("500.00")]})
    lines = [
        row("Alpha", "1", "100.00", "100.00", 120),
        row("Beta", "2", "200.00", "400.00", 170),
        line("Sous-total HT 5 716,77", [tok("Sous-total", 120, 260, 220), tok("5", 700, 720, 220),
                                       tok("716,77", 725, 800, 220)]),
        line("TVA 20 % 1 143,35", [tok("TVA", 120, 180, 260), tok("20", 700, 730, 260),
                                   tok("%", 735, 755, 260), tok("1", 800, 820, 260),
                                   tok("143,35", 825, 900, 260)]),
    ]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert len(document.rows) == 2
    assert [r.description for r in document.rows] == ["Alpha", "Beta"]


def test_existing_rows_are_never_overwritten(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    document.rows = ["sentinel"]
    table_geometry.recover(document, reading([row("Alpha", "1", "10.00", "10.00", 100)]),
                           prepared=prepared())
    assert document.rows == ["sentinel"]


def test_no_table_leaves_evidence_untouched(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    table_geometry.recover(document, reading([line("FACTURE", [tok("FACTURE", 120, 260, 80)])]),
                           prepared=prepared())
    assert document.rows == []
    assert document.rows_provenance is None


def test_tax_layout_keeps_only_quantity_and_price(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()

    def tax_row(description, qty, price, tax, gross, y):
        tokens = [
            tok(description, 120, 300, y), tok(qty, 500, 540, y), tok(price, 650, 720, y),
            tok("20", 760, 785, y), tok("%", 790, 805, y), tok(tax, 850, 900, y),
            tok(gross, 940, 990, y),
        ]
        return line(" ".join(t.text for t in tokens), tokens)

    lines = [tax_row("Alpha", "1", "100.00", "20.00", "120.00", 120),
             tax_row("Beta", "2", "50.00", "20.00", "120.00", 170)]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert len(document.rows) == 2
    for item in document.rows:
        assert set(item.cells) == {"quantity", "unit_price"}
        assert item.row_evidence["corroboration"]["rule"] == "tax = quantity x unit_price x rate"


def test_geometry_rows_never_drive_document_totals(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    from invoice_grounding import reconcile

    document = evidence(totals={"total_candidate": [number("950.00")]})
    lines = [row("Alpha", "1", "100.00", "100.00", 120), row("Beta", "2", "200.00", "400.00", 170)]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert document.rows
    assert document.rows_ordinal_proof is False
    document.available = True
    document.source_kind = "consensus"
    document.ocr_cross_checked = True
    result = reconcile(document, [{}])
    assert result.payload["total_ttc"] is None
    assert result.payload["total_ht"] is None


def test_rows_follow_printed_order_without_duplication(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    lines = [row("Alpha", "1", "10.00", "10.00", 120), row("Beta", "2", "20.00", "40.00", 170),
             row("Gamma", "3", "30.00", "90.00", 220)]
    table_geometry.recover(document, reading(lines), prepared=prepared())
    assert [r.description for r in document.rows] == ["Alpha", "Beta", "Gamma"]
    assert len(document.rows) == 3
