"""PR131 Phase 7 — the two accepted corrections, each pinned by its own failing example.

Every assertion here was read off the pre-change tree, not off what the code was expected to do: the
before/after table is in `results/product-review131/pr131-phase7-fixture-probe.json`. The fixtures are
synthetic and use no OCR, so a fresh clone runs them.
"""

import sys
from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text, total_label_field  # noqa: E402
from invoice_validation import document_validation  # noqa: E402
from ocr_consensus import build_consensus  # noqa: E402

TODAY = date(2026, 6, 1)
HEAD = "FACTURE FR-2026-00099\nFournisseur: Acme SARL\n"
TABLE = (
    "Description | Quantité | Prix unitaire HT | Montant HT\n"
    "Prestation | 2 | 100,00 | 200,00\n"
)
TOTALS = "Total HT: 200,00 EUR\nTotal TVA: 40,00 EUR\nTotal TTC: 240,00 EUR\n"

# A date label printed with an impossible calendar date: the validation describes that specifically, and the
# consensus filter used to add a second, vaguer sentence about the same field.
IMPOSSIBLE_DATE = HEAD + "Date de facture: 13/45/2026\n" + TABLE + TOTALS
# A date that reads but cannot be plausible: nothing specific is said about it, so the generic sentence is
# the only information the report has and must survive.
UNPLAUSIBLE_DATE = HEAD + "Date de facture: 01/01/1900\n" + TABLE + TOTALS
SPECIFIC_DATE = (
    "Date de facture : non vérifiable, le libellé de date de facture est lu mais le texte qui le suit ne "
    "forme pas une date lisible ; aucune interprétation supposée."
)
GENERIC_DATE = (
    "Date de facture : non vérifiable, les lectures OCR ne permettent pas de confirmer ce contrôle."
)


def consensus(text: str):
    evidence = from_text(text, locale="fr-FR")
    return build_consensus({"gray": [evidence], "contrast": [deepcopy(evidence)]})


def test_spaced_sub_total_is_read_as_the_net_total():
    assert total_label_field("Sub Total 61.97") == "total_ht"
    totals = from_text("FACTURE FR-1\nFournisseur: Acme\nSub Total 61.97\n", locale="fr-FR").totals
    assert [node.value for node in totals["total_ht"]] == [Decimal("61.97")]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sub Total 61.97", "total_ht"),
        ("Sub Total: 61,97 €", "total_ht"),
        ("Sub Total HT 61,97", "total_ht"),
        ("Sub Total (net) 61.97", "total_ht"),
        ("SUBTOTAL 61.97", "total_ht"),
        ("Sous-total 61,97", "total_ht"),
        ("Total 605,00", "total_candidate"),
        ("Montant HT 135,00", None),
        ("Prix unitaire HT 10,00", None),
        ("Total TVA 27,00", None),
        ("VAT 20.00", None),
    ],
)
def test_only_the_spacing_of_an_established_label_changed(text, expected):
    assert total_label_field(text) == expected


@pytest.mark.parametrize(
    "spaced,established",
    [
        ("Sub Total 61.97", "Subtotal 61.97"),
        ("Sub Total HT 61.97", "Subtotal HT 61.97"),
        ("Sub Total (net) 61.97", "Subtotal (net) 61.97"),
        ("Sub Total TTC 100,00", "Subtotal TTC 100,00"),
        ("Sub Total TVA 20,00", "Subtotal TVA 20,00"),
    ],
)
def test_a_spaced_label_reads_exactly_like_the_unspaced_one(spaced, established):
    """The spaced print may add a spelling, never a different verdict.

    `Subtotal TTC` and `Subtotal TVA` have always been read as the net total, so the spaced variants inherit
    that scope: a qualifier after the label is a separate, pre-existing gap, not something this change fixes
    or widens.
    """
    assert total_label_field(spaced) == total_label_field(established)


def test_an_unreadable_invoice_date_is_described_once():
    result = document_validation(consensus(IMPOSSIBLE_DATE), today=TODAY)
    assert [s for s in result.not_verified if s.startswith("Date de facture :")] == [SPECIFIC_DATE]
    assert GENERIC_DATE not in result.not_verified
    assert result.flags == []


def test_the_generic_sentence_survives_when_nothing_specific_was_said():
    result = document_validation(consensus(UNPLAUSIBLE_DATE), today=TODAY)
    assert [s for s in result.not_verified if s.startswith("Date de facture :")] == [GENERIC_DATE]
    assert result.flags == []
