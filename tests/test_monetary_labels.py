"""Monetary label semantics: explicit HT/TTC wording only; ambiguous kinds stay unresolved."""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text, total_label_field  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Total HT", "total_ht"),
        ("Total hors taxe", "total_ht"),
        ("Total hors taxes", "total_ht"),
        ("Total avant taxes", "total_ht"),
        ("Total (HT)", "total_ht"),
        ("Sous-total HT", "total_ht"),
        ("Subtotal", "total_ht"),
        ("Total TTC", "total_ttc"),
        ("Total TVA comprise", "total_ttc"),
        ("Total toutes taxes comprises", "total_ttc"),
        ("Total due", "total_ttc"),
        ("Amount payable", "total_ttc"),
        ("Total (TTC)", "total_ttc"),
        ("VAT", None),
        ("TVA", None),
        ("Total TVA", None),
        ("Unit price", None),
        ("Prix unitaire HT", None),
        ("Line amount", None),
        ("Montant HT", None),
        ("Montant TTC", None),
        ("TOTAL 605,00", "total_candidate"),
        ("Total (GBP) 147.00", "total_candidate"),
    ],
)
def test_total_label_matrix(text, expected):
    assert total_label_field(text) == expected


def _totals(text):
    evidence = from_text(text, locale="fr-FR")
    return evidence.totals, evidence.taxes, evidence.discounts


def test_total_hors_taxe_is_explicit_ht():
    totals, taxes, _ = _totals("TOTAL HORS TAXE 950,00 €")
    assert [n.value for n in totals.get("total_ht", [])] == [Decimal("950.00")]
    assert not totals.get("total_candidate")


def test_total_hors_taxes_plural_is_explicit_ht():
    totals, _, _ = _totals("Total hors taxes 1 000,00 €")
    assert [n.value for n in totals.get("total_ht", [])] == [Decimal("1000.00")]


def test_total_tva_is_never_a_document_total():
    totals, taxes, _ = _totals("Total TVA 27,00 €")
    assert not totals
    assert not taxes


def test_total_ttc_and_inclusive_variants():
    for text in ("Total TTC 1 045,00 €", "Total TVA comprise 1 045,00 €",
                 "Total toutes taxes comprises 1 045,00 €", "Amount payable 1 045,00"):
        totals, _, _ = _totals(text)
        assert [n.value for n in totals.get("total_ttc", [])] == [Decimal("1045.00")], text


def test_montant_labels_stay_unqualified():
    totals, _, _ = _totals("Montant HT 135,00 €")
    assert not totals
    totals, _, _ = _totals("Montant TTC 162,00 €")
    assert not totals


def test_unqualified_total_stays_a_candidate():
    totals, _, _ = _totals("TOTAL 605,00 €")
    assert [n.value for n in totals.get("total_candidate", [])] == [Decimal("605.00")]
    assert not totals.get("total_ttc")


def test_multiple_totals_keep_the_duplicate_ht_labels():
    totals, taxes, _ = _totals("Total HT 135,00 €\nTotal TVA 27,00 €\nTotal HT 162,00 €")
    assert [n.value for n in totals.get("total_ht", [])] == [Decimal("135.00"), Decimal("162.00")]
    assert not totals.get("total_ttc")
    assert not taxes


def test_discount_line_is_recorded_separately():
    totals, _, discounts = _totals("Discount: 10.00")
    assert [n.value for n in discounts] == [Decimal("10.00")]
    assert not totals


def test_tax_lines_keep_their_rate_and_amount():
    _, taxes, _ = _totals("TVA 20% 234,00 €")
    assert len(taxes) == 1
    assert taxes[0].rate.value == Decimal("20")
    assert taxes[0].amount.value == Decimal("234.00")
