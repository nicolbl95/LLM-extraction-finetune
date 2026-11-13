"""Observed book-table structure, independently corroborated cells and summary boundaries."""

import copy
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from invoice_evidence import Token, from_text, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus
from invoice_validation import document_validation


def book_tokens():
    rows = [
        (0, [(0, "Facture"), (250, "Date: 12 juillet 2023")]),
        (30, [(0, "Référence"), (95, "|"), (110, "Quantité"), (210, "Titre du livre"),
              (430, "Prix"), (540, "Montant")]),
        # A centred continuation may begin left of the first word.
        (44, [(425, "unitaire")]),
        (64, [(0, "5497621584"), (110, "12"), (210, "La cuisine de tous les jours"),
              (430, "15,95 $"), (540, "191,40 $")]),
        (84, [(0, "5498721685"), (110, "3"), (210, "Ismène et la ville"),
              (430, "8,75 $"), (540, "26,25 $")]),
        (104, [(0, "5498825741"), (110, "10"), (210, "La pensée positive"),
               (430, "11,45 $"), (540, "114,50 $")]),
        (140, [(210, "Frais de livraison"), (540, "12,50 $")]),
        (160, [(210, "Montant total"), (540, "344,65 $")]),
        (180, [(210, "TPS"), (540, "17,23 $")]),
        (200, [(210, "Total général"), (540, "361,88 $")]),
    ]
    return [Token(text, 0, (x, y, x + max(20, len(text) * 4), y + 10), 95)
            for y, cells in rows for x, text in cells]


def test_split_book_header_and_border_glyphs_keep_three_printed_rows():
    evidence = from_tokens(book_tokens(), literal_columns=False)
    assert len(evidence.rows) == 3
    assert [row.description for row in evidence.rows] == [
        "La cuisine de tous les jours", "Ismène et la ville", "La pensée positive",
    ]
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    proposal = {"line_items": [
        {"description": row.description, "quantity": quantity, "unit_price": price, "total_ht": amount}
        for row, quantity, price, amount in zip(evidence.rows, [12, 3, 10], [15.95, 8.75, 11.45], [191.4, 26.25, 114.5])
    ] + [{"description": "Frais de livraison", "quantity": 1, "unit_price": 12.5, "total_ht": 12.5}]}
    result = reconcile(consensus, [proposal])
    assert len(result.payload["line_items"]) == 3
    assert result.payload["line_items"][0]["quantity"] == 12
    assert result.payload["line_items"][0]["unit_price"] == 15.95
    assert result.payload["line_items"][0]["total_ht"] == 191.4
    assert result.payload["total_ht"] is None


def test_shipping_summary_never_merges_with_last_book_description():
    evidence = from_tokens(book_tokens(), literal_columns=False)
    assert len(evidence.rows) == 3
    assert all(not row.secondary_description for row in evidence.rows)
    assert all("livraison" not in row.description for row in evidence.rows)


def test_text_pipe_table_still_preserves_literal_columns():
    evidence = from_text(
        "Facture\nTitre du livre | Quantité | Prix unitaire | Montant\n"
        "Volume A | 2 | 10,00 $ | 20,00 $\n"
        "Frais de livraison: 12,50 $\nMontant total: 32,50 $\n"
        "TPS: 1,63 $\nTotal général: 34,13 $"
    )
    assert len(evidence.rows) == 1
    assert evidence.rows[0].description == "Volume A"
    assert evidence.rows[0].cells["quantity"][0].value == 2
    assert not evidence.rows[0].secondary_description
    assert evidence.totals["total_ttc"][0].value == Decimal("34.13")
    assert not evidence.totals.get("total_ht")


def test_unrelated_next_line_cannot_be_used_to_complete_a_header():
    source = [token for token in book_tokens() if token.box[1] != 44]
    evidence = from_tokens(source, literal_columns=False)
    assert evidence.rows == []


def test_shipping_is_a_valid_item_only_when_printed_item_operands_exist():
    evidence = from_text(
        "Facture\nDescription | Quantité | Prix unitaire | Montant\n"
        "Frais de livraison | 2 | 10 | 20\n"
    )
    assert len(evidence.rows) == 1
    assert evidence.rows[0].cells["quantity"][0].value == 2


def test_centred_title_header_does_not_drop_text_after_printed_quantity():
    source = [
        replace(token, box=(280, token.box[1], 380, token.box[3]))
        if token.text == "Titre du livre" else token for token in book_tokens()
    ]
    evidence = from_tokens(source, literal_columns=False)
    assert evidence.rows[0].description == "La cuisine de tous les jours"


def amount_consensus(*, reference_confidence=10, amount_confidence=95, families=2):
    source = [
        replace(token, confidence=reference_confidence)
        if token.text == "5497621584" else
        replace(token, confidence=amount_confidence)
        if token.text == "191,40 $" else token
        for token in book_tokens()
    ]
    evidence = from_tokens(source, literal_columns=False)
    evidence.rows = evidence.rows[:1]
    evidence.rows[0].cells["quantity"] = []
    evidence.rows[0].cells["unit_price"] = []
    readings = {"gray": [evidence]}
    if families == 2:
        readings["contrast"] = [copy.deepcopy(evidence)]
    return build_consensus(readings)


def test_low_confidence_reference_does_not_erase_independently_legible_printed_amount():
    result = reconcile(amount_consensus(), [{
        "line_items": [{"description": "Invented", "quantity": 1,
                        "unit_price": 191.4, "total_ht": 999}],
    }])
    assert result.payload["line_items"][0]["total_ht"] == 191.4
    assert result.payload["line_items"][0]["quantity"] is None
    assert result.payload["line_items"][0]["unit_price"] is None
    assert result.payload["line_items"][0]["description"] is None


def test_legible_amount_still_requires_two_families_and_unchanged_cell_confidence():
    for evidence in (
        amount_consensus(amount_confidence=59), amount_consensus(families=1),
    ):
        assert reconcile(evidence, [{}]).payload["line_items"][0]["total_ht"] is None


def test_conflicting_cell_amounts_cannot_be_chosen_by_model_or_arithmetic():
    evidence = amount_consensus()
    amount = evidence.rows[0].cells["total_ht"][0]
    evidence.rows[0].cells["total_ht"].append(replace(amount, value=Decimal("999")))
    result = reconcile(evidence, [{"line_items": [{"total_ht": 999}]}])
    assert result.payload["line_items"][0]["total_ht"] is None


def test_french_long_invoice_date_requires_independent_confident_source_lines():
    text = "Facture\nDate de facturation: 12 juillet 2023\n"
    first = from_text(text)
    result = build_consensus({"gray": [first], "contrast": [copy.deepcopy(first)]})
    assert result.invoice_date == "2023-07-12"
    second = from_text(text)
    date_line = next(line for line in second.lines if "juillet" in line.text)
    date_line.tokens = [replace(token, confidence=39) for token in date_line.tokens]
    rejected = build_consensus({"gray": [first], "contrast": [second]})
    assert rejected.invoice_date is None


def test_french_numeric_locale_and_dollar_symbol_do_not_assert_euro_or_usd():
    evidence = from_tokens(book_tokens(), literal_columns=False)
    validation = document_validation(evidence)
    assert not any("Devise lue : EUR" in reason or "Devise lue : USD" in reason
                   for reason in validation.passed)
    assert any(reason.startswith("Devise : non vérifiable") for reason in validation.not_verified)
    explicit = from_text(
        "Facture\nCurrency: CAD\nDescription | Quantité | Prix unitaire | Montant\n"
        "Livre | 2 | 10,00 $ | 20,00 $\nTotal général: 20,00 $"
    )
    assert "Devise lue : CAD." in document_validation(explicit).passed


def table_blocks():
    return [
        "Facture",
        "Désignation | Quantité | Prix unitaire | Montant",
        "Livre A | 2 | 10,00 $ | 20,00 $",
        "Livre B | 3 | 8,50 $ | 25,50 $",
        "Montant total: 45,50 $", "TPS: 2,28 $", "Total général: 47,78 $",
    ]


def cells_of(evidence):
    return [(row.description, row.secondary_description,
             tuple(str(cell.value) for key in ("quantity", "unit_price", "total_ht")
                   for cell in row.cells.get(key, [])))
            for row in evidence.rows]


def test_explicit_semicolon_consensus_text_keeps_the_same_boundaries_as_lines():
    """A pasted one-line table keeps its rows: ` ; ` ends a line and `|` never joins two cells.

    `from_text` reads a separator convention instead of a newline count, so this is the only
    guard that a text arriving without line breaks still yields one row per printed line.
    """
    blocks = table_blocks()
    lined = from_text("\n".join(blocks))
    explicit = from_text(" ; ".join(blocks))
    assert [line.text for line in explicit.lines] == [line.text for line in lined.lines]
    assert cells_of(explicit) == cells_of(lined)
    assert [row[0] for row in cells_of(explicit)] == ["Livre A", "Livre B"]
    assert any(token.text == "|" for token in explicit.tokens), (
        "le séparateur de colonne n'est plus un token à lui seul : les colonnes littérales "
        "ne peuvent plus être dérivées de sa position"
    )
    assert not [token.text for token in explicit.tokens if "|" in token.text and token.text != "|"]
    assert explicit.totals["total_ttc"][0].value == lined.totals["total_ttc"][0].value


def test_two_rows_on_one_line_collapse_without_moving_an_amount_onto_the_wrong_item():
    """Mixed separators lose a row rather than export it: the guard is the surviving cell values."""
    blocks = table_blocks()
    collapsed = from_text("\n".join([blocks[0], blocks[1], " ; ".join(blocks[2:4]), *blocks[4:]]))
    assert [row[0] for row in cells_of(collapsed)] == ["Livre A"]
    assert cells_of(collapsed)[0][2] == ("2", "10.00", "20.00")
    assert collapsed.totals["total_ttc"][0].value == Decimal("47.78")
