"""PR85 hourly-header integration: the M2 vocabulary and the boundaries it must not cross.

Offline and deterministic — synthetic tokens only, no Tesseract, no model, no private fixture.
The positive cases are the three header shapes PR84 recorded on the consulting layout plus the
two other currency/per-hour forms the tested predicate covers. The negative cases pin what the
integration deliberately does NOT do: invent a price column, open a table from ordinary prose,
or accept the bare French `Prix` header that PR84 measured as NO-GO.
"""

import copy
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import Token, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus

HEADER_Y = 100
COLUMN_X = (60, 120, 300, 460)


def token(text, x, y, width=None):
    return Token(text, 0, (x, y, x + (width or 12 + 5 * len(text)), y + 14), 95.0)


def table(labels, rows, xs=COLUMN_X, header_y=HEADER_Y):
    """One header line plus printed data lines, each cell under its own column start."""
    tokens = [token(label, x, header_y) for label, x in zip(labels, xs) if label]
    for index, row in enumerate(rows):
        y = header_y + 22 + index * 20
        tokens += [
            token(value, x, y, width)
            for value, x, width in zip(row, xs, (12, None, 40, 44)) if value
        ]
    return tokens


def cells(row):
    return {key: [number.value for number in row.cells.get(key, [])]
            for key in ("quantity", "unit_price", "total_ht")}


def test_printed_heures_euro_per_hour_header_recovers_the_printed_cells():
    evidence = from_tokens(table(
        ["HEURES", "DESCRIPTION", "€/HEURE", "MONTANT"],
        [["2", "Prestation conseil", "35,00", "70,00"],
         ["5", "Atelier equipe", "25,00", "125,00"]],
    ), literal_columns=False)
    assert [row.description for row in evidence.rows] == ["Prestation conseil", "Atelier equipe"]
    assert [cells(row) for row in evidence.rows] == [
        {"quantity": [Decimal(2)], "unit_price": [Decimal("35.00")], "total_ht": [Decimal("70.00")]},
        {"quantity": [Decimal(5)], "unit_price": [Decimal("25.00")], "total_ht": [Decimal("125.00")]},
    ]


def test_merged_issuer_prefix_on_the_header_line_still_locates_the_columns():
    """PR84 recorded `ENTREPRSE CLIENT HEURES DESCRIPTION €/HEURE MONTANT` in one denoise read."""
    xs = (200, 300, 480, 640)
    tokens = [token("ENTREPRSE", 0, HEADER_Y), token("CLIENT", 90, HEADER_Y)]
    tokens += table(["HEURES", "DESCRIPTION", "€/HEURE", "MONTANT"],
                    [["2", "Prestation conseil", "35,00", "70,00"]], xs=xs)
    evidence = from_tokens(tokens, literal_columns=False)
    assert len(evidence.rows) == 1
    assert evidence.rows[0].description == "Prestation conseil"
    assert cells(evidence.rows[0]) == {
        "quantity": [Decimal(2)], "unit_price": [Decimal("35.00")], "total_ht": [Decimal("70.00")],
    }


def test_degraded_hourly_header_without_a_rate_column_builds_no_rows():
    """PR84's `QTÉ DESCRIPTION MONTANT` read: the rate column was lost, so no price may be invented."""
    evidence = from_tokens(table(
        ["QTÉ", "DESCRIPTION", "", "MONTANT"],
        [["2", "Prestation conseil", "35,00", "70,00"]],
    ), literal_columns=False)
    assert evidence.rows == []


def test_other_tested_currency_per_hour_forms_open_the_same_columns():
    for labels, amount in ((["HRS", "DESCRIPTION", "$/HR", "AMOUNT"], "70.00"),
                           (["HEURES", "DESCRIPTION", "PER HEURE", "MONTANT"], "70,00")):
        evidence = from_tokens(table(labels, [["2", "Consulting", "35.00", amount]]),
                               literal_columns=False)
        assert len(evidence.rows) == 1, labels
        assert cells(evidence.rows[0]) == {
            "quantity": [Decimal(2)], "unit_price": [Decimal("35.00")],
            "total_ht": [Decimal("70.00")],
        }


def test_unrelated_heures_prose_does_not_open_an_invoice_table():
    tokens = [
        token("Heures", 60, 100), token("d'ouverture", 120, 100),
        token("du", 230, 100), token("lundi", 260, 100), token("au", 320, 100),
        token("vendredi", 350, 100),
        token("Nombre", 60, 140), token("d'heures", 130, 140), token("supplementaires", 220, 140),
        token("12", 400, 140, 12),
        token("Montant", 60, 180), token("total", 140, 180), token("540,00", 400, 180, 44),
    ]
    assert from_tokens(tokens, literal_columns=False).rows == []


def test_currency_per_hour_expression_outside_a_valid_header_creates_no_rows():
    """A rate quoted in prose is not a column: description and rate alone cannot open the gate."""
    tokens = [
        token("Prestation", 60, 100), token("de", 160, 100), token("conseil", 190, 100),
        token("35,00", 300, 100, 40), token("€/heure", 360, 100),
        token("2", 60, 130, 12), token("70,00", 300, 130, 44),
    ]
    assert from_tokens(tokens, literal_columns=False).rows == []


def test_bare_french_prix_header_still_builds_no_rows():
    """PR84 M1 is NO-GO and stays out: layout a's `Prix TVA Montant` must remain unrecovered."""
    xs = (60, 200, 340, 430, 520)
    tokens = [token(label, x, HEADER_Y) for label, x in
              zip(["Description", "Quantité", "Prix", "TVA", "Montant"], xs)]
    tokens += [token(value, x, HEADER_Y + 22, width) for value, x, width in
               zip(["Mini portrait", "2", "35,00", "20%", "70,00"], xs, (None, 12, 40, 24, 44))]
    assert from_tokens(tokens, literal_columns=False).rows == []


def test_book_table_continuation_invariant_is_preserved():
    """The invariant M1 broke: a lone `Prix` line may not be completed by an unrelated next line."""
    tokens = [
        token("Référence", 0, 30), token("Quantité", 110, 30), token("Titre", 210, 30),
        token("du", 250, 30), token("livre", 275, 30), token("Prix", 430, 30),
        token("Montant", 540, 30),
        token("5497621584", 0, 64), token("12", 110, 64, 12),
        token("La cuisine de tous les jours", 210, 64),
        token("15,95 $", 430, 64, 40), token("191,40 $", 540, 64, 44),
    ]
    assert from_tokens(tokens, literal_columns=False).rows == []


def test_single_family_hourly_cells_are_not_corroborated():
    evidence = from_tokens(table(["HEURES", "DESCRIPTION", "€/HEURE", "MONTANT"],
                                 [["2", "Prestation conseil", "35,00", "70,00"]]),
                           literal_columns=False)
    assert len(evidence.rows) == 1
    consensus = build_consensus({"gray": [evidence]})
    assert all(not consensus.rows[0].cells.get(key) for key in ("quantity", "unit_price", "total_ht"))
    exported = reconcile(consensus, [{}]).payload["line_items"]
    assert exported == [{"description": None, "quantity": None, "unit_price": None, "total_ht": None}]


def test_two_families_export_the_printed_hourly_cells():
    evidence = from_tokens(table(["HEURES", "DESCRIPTION", "€/HEURE", "MONTANT"],
                                 [["2", "Prestation conseil", "35,00", "70,00"]]),
                           literal_columns=False)
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert reconcile(consensus, [{}]).payload["line_items"] == [
        {"description": "Prestation conseil", "quantity": 2.0, "unit_price": 35.0, "total_ht": 70.0},
    ]


def test_vat_inclusive_hourly_amount_still_triggers_the_product_guard():
    """3 x 25,00 = 75,00 but the printed amount is 90,00: the guard must clear quantity and price.

    The surviving unqualified amount is the pre-existing TTC/HT gap PR84 measured under M1; this
    test pins the safeguard that fires, not the correctness of that amount.
    """
    evidence = from_tokens(table(["HEURES", "DESCRIPTION", "€/HEURE", "MONTANT"],
                                 [["3", "Prestation conseil", "25,00", "90,00"]]),
                           literal_columns=False)
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert not consensus.rows[0].cells.get("quantity")
    assert not consensus.rows[0].cells.get("unit_price")
    assert any("produit" in issue for issue in consensus.issues)
    exported = reconcile(consensus, [{}]).payload["line_items"][0]
    assert exported["quantity"] is None
    assert exported["unit_price"] is None
