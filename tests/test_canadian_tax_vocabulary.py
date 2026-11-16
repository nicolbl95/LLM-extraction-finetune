"""PR177 — Canadian TPS/TVQ labels reach the same tax machinery as TVA/VAT, with no invented value.

Today a line printed `TPS 13,42 $` is not a tax line at all: `invoice_evidence.py:1149` starts a fiscal label
with `tva|vat|tax|sales tax`, and `invoice_validation.py:88` repeats that vocabulary for the report and for the
PR176 gross-up. Nothing Canadian ever becomes a `Tax`, so every arithmetic route in `invoice_grounding.py`
abstains and the verification report says "ventilation absente".

These fixtures are synthetic, use no OCR and no reference file, so a fresh clone runs them. They state two
things at once: a Canadian tax line must be *represented* (`rate=None` when the document prints no percentage —
no rate is ever assumed from the Canadian or Quebec averages), and the *totals* may only move through a route
that already exists for TVA/VAT documents. The last group is the safety side: a registration number printed as
`TVQ : 2222222222` is not a tax amount, a supplier called "TPS Studio" is not a tax, a garbled label is not a
tax, and a tax amount is never exported as the payable total.
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text  # noqa: E402
from invoice_grounding import reconcile  # noqa: E402

FRENCH_TABLE = "Description | Quantite | Prix unitaire HT | Montant HT\n"
UK_TABLE = "Description | Quantity | Unit Price | Amount\n"


def ground(text: str, locale: str):
    evidence = from_text(text, locale=locale)
    return evidence, reconcile(evidence, [{}])


def taxes(evidence) -> list[tuple]:
    """(rate, amount) as printed, with rate=None exactly when the document shows no percentage."""
    return [
        (None if tax.rate is None else Decimal(str(tax.rate.value)), Decimal(str(tax.amount.value)))
        for tax in evidence.taxes
    ]


def warning(result, needle: str) -> str | None:
    return next((text for text in result.warnings if needle in text), None)


# A Canadian document that prints a tax and no percentage anywhere.
TPS_ONLY = (
    "FACTURE CA-1\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 268,40 | 268,40\n"
    "Montant total : 268,40 $\nTPS 13,42 $\nTotal general : 281,82 $\n"
)
TVQ_ONLY = (
    "FACTURE CA-2\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 268,40 | 268,40\n"
    "Montant total : 268,40 $\nTVQ 26,84 $\nTotal general : 295,24 $\n"
)
BOTH_TAXES = (
    "FACTURE CA-3\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 268,40 | 268,40\n"
    "Montant total : 268,40 $\nTPS 13,42 $\nTVQ 26,84 $\nTotal general : 308,66 $\n"
)
AMOUNT_NO_RATE = (
    "FACTURE CA-4\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 10,00 $\nTotal : 210,00 $\n"
)
RATE_AND_AMOUNT = (
    "FACTURE CA-5\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 10,00 $ (5 %)\nTotal : 210,00 $\n"
)
RATE_NO_AMOUNT = (
    "FACTURE CA-6\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 5 %\nTotal : 210,00 $\n"
)
CONTRADICTORY_AMOUNT = (
    "FACTURE CA-7\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 30,00 $\nTotal : 210,00 $\n"
)
# Quebec prints both registration numbers in the `LABEL : value` syntax a tax amount would use.
REGISTRATION_NUMBERS = (
    "FACTURE CA-8\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 268,40 | 268,40\n"
    "Montant total : 268,40 $\nTPS 13,42 $\nNumeros d'enregistrement TPS : 1111111111\n"
    "TVQ : 2222222222\nTotal general : 281,82 $\n"
)
NO_TAX = (
    "FACTURE CA-9\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTotal : 200,00 $\n"
)
ZERO_RATE = (
    "FACTURE CA-10\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 0,00 $ (0 %)\nTotal : 200,00 $\n"
)
TWO_RATES = (
    "FACTURE CA-11\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 5,00 $ (5 %)\nTVQ 10,00 $ (10 %)\nTotal : 215,00 $\n"
)
COHERENT = (
    "FACTURE CA-12\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 20,00 $\nTotal TTC : 220,00 $\n"
)
# The same print one cent short of the identity, and the French document that already answers it.
INCOHERENT = (
    "FACTURE CA-13\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTPS 20,00 $\nTotal TTC : 225,00 $\n"
)
INCOHERENT_FRENCH = (
    "FACTURE FR-2\nFournisseur: Acme SARL\n" + FRENCH_TABLE +
    "Prestation | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 EUR\nTVA 40,00 EUR\nTotal TTC : 265,00 EUR\n"
)
MISREAD_LABEL = (
    "FACTURE CA-14\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\n1PS 20,00 $\nTotal : 220,00 $\n"
)
INSUFFICIENT_PROOF = (
    "FACTURE CA-15\nFournisseur: Atelier Nord\n"
    "Total HT : 200,00 $\nTPS 5 %\n"
)
TOTAL_PREFIXED_TAX = (
    "FACTURE CA-19\nFournisseur: Atelier Nord\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 268,40 | 268,40\n"
    "Montant total : 268,40 $\nTotal TVQ 26,84 $\nTotal general : 295,24 $\n"
)
SUPPLIER_CONTAINS_TPS = (
    "FACTURE CA-21\nFournisseur: TPS Studio Enregistre\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 $\nTotal : 200,00 $\n"
)
CONTROL_FRENCH = (
    "FACTURE FR-1\nFournisseur: Acme SARL\n" + FRENCH_TABLE +
    "Prestation | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 EUR\nTVA 40,00 EUR (20 %)\nTotal TTC : 240,00 EUR\n"
)
CONTROL_UK = (
    "INVOICE UK-1\nSupplier: Acme Ltd\n" + UK_TABLE +
    "Design work | 1 | 200.00 | 200.00\n"
    "Net 200.00\nVAT 40.00 (20%)\nTotal 240.00\n"
)


# --- 1, 2: each Canadian label with an explicit printed amount becomes a tax ----------------------------

@pytest.mark.parametrize("text,amount", [(TPS_ONLY, "13.42"), (TVQ_ONLY, "26.84")])
def test_a_canadian_label_with_a_printed_amount_is_a_tax(text, amount):
    evidence, _ = ground(text, "fr-CA")
    assert taxes(evidence) == [(None, Decimal(amount))]


# --- 3: two Canadian taxes are both represented, and never resolved into an assumed rate -----------------

def test_two_amounts_without_percentages_stay_unresolved_rather_than_guessed():
    evidence, result = ground(BOTH_TAXES, "fr-CA")
    assert taxes(evidence) == [(None, Decimal("13.42")), (None, Decimal("26.84"))]
    assert warning(result, "plusieurs montants incompatibles") is not None
    # Neither total is inferred from a pair of amounts that cannot be tied to a rate.
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] == 308.66


# --- 4, 5: amount without percentage, and percentage with amount ---------------------------------------

@pytest.mark.parametrize("text,tax", [
    (AMOUNT_NO_RATE, (None, Decimal("10.00"))),
    (RATE_AND_AMOUNT, (Decimal("5"), Decimal("10.00"))),
])
def test_a_recognised_canadian_tax_lets_the_existing_additive_route_export_the_printed_total(text, tax):
    evidence, result = ground(text, "fr-CA")
    assert taxes(evidence) == [tax]
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] == 210.0


# --- 6: a percentage with no amount is a clue, never a tax, and only corroborates the printed total ------

def test_a_percentage_without_amount_never_becomes_a_tax_amount():
    evidence, result = ground(RATE_NO_AMOUNT, "fr-CA")
    assert evidence.taxes == []
    # The PR176 gross-up may now use the printed 5 % on the TPS line, and exports the print, not 210.00001.
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] == 210.0


# --- 7: contradictory tax amount keeps the contradiction protection --------------------------------------

def test_a_canadian_amount_that_contradicts_the_totals_is_not_used_to_reach_one():
    evidence, result = ground(CONTRADICTORY_AMOUNT, "fr-CA")
    assert taxes(evidence) == [(None, Decimal("30.00"))]
    assert result.payload["total_ttc"] is None


# --- 8: registration numbers are identity data, not tax amounts ------------------------------------------

def test_a_registration_number_is_never_read_as_a_tax_amount():
    evidence, result = ground(REGISTRATION_NUMBERS, "fr-CA")
    assert taxes(evidence) == [(None, Decimal("13.42"))]
    assert {str(tax.amount.value) for tax in evidence.taxes}.isdisjoint(
        {"1111111111", "2222222222", "3333333350.23"}
    )
    assert result.payload["total_ttc"] == 281.82


# --- 9, 10: no tax at all, and a printed zero rate ------------------------------------------------------

def test_a_document_without_any_tax_line_is_untouched():
    evidence, result = ground(NO_TAX, "fr-CA")
    assert evidence.taxes == []
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] is None


def test_a_printed_zero_rate_is_a_rate_and_not_a_missing_one():
    evidence, result = ground(ZERO_RATE, "fr-CA")
    assert taxes(evidence) == [(Decimal("0"), Decimal("0.00"))]
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] == 200.0


# --- 11: multiple rates on one document -----------------------------------------------------------------

def test_two_canadian_rates_stay_distinct_and_add_up_to_the_printed_total():
    evidence, result = ground(TWO_RATES, "fr-CA")
    assert taxes(evidence) == [(Decimal("5"), Decimal("5.00")), (Decimal("10"), Decimal("10.00"))]
    assert result.payload["total_ttc"] == 215.0
    assert warning(result, "TVA : HT lu") is None
    assert warning(result, "plusieurs montants incompatibles") is None


# --- 12, 13: coherent and incoherent HT/tax/TTC chains ---------------------------------------------------

def test_a_coherent_canadian_chain_is_verified_rather_than_reported_absent():
    evidence, result = ground(COHERENT, "fr-CA")
    assert taxes(evidence) == [(None, Decimal("20.00"))]
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] == 220.0
    assert warning(result, "ventilation absente") is None


def test_an_incoherent_canadian_chain_abstains_like_the_french_one():
    _, french = ground(INCOHERENT_FRENCH, "fr-FR")
    assert french.payload["total_ttc"] is None
    assert warning(french, "TVA : HT lu") is not None

    evidence, canadian = ground(INCOHERENT, "fr-CA")
    assert taxes(evidence) == [(None, Decimal("20.00"))]
    assert canadian.payload["total_ttc"] is None
    assert warning(canadian, "TVA : HT lu") is not None


# --- 14, 15: a garbled label and a document that proves nothing -----------------------------------------

def test_a_misread_tax_label_does_not_create_a_tax():
    evidence, result = ground(MISREAD_LABEL, "fr-CA")
    assert evidence.taxes == []
    assert result.payload["total_ttc"] is None


def test_insufficient_proof_still_exports_nothing_beyond_the_printed_net_total():
    evidence, result = ground(INSUFFICIENT_PROOF, "fr-CA")
    assert evidence.taxes == []
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] is None


# --- safety side: labels that must never move a total ---------------------------------------------------

def test_a_tax_line_amount_is_never_exported_as_the_payable_total():
    evidence, result = ground(TOTAL_PREFIXED_TAX, "fr-CA")
    assert result.payload["total_ttc"] != 26.84
    assert result.payload["total_ttc"] == 295.24


def test_a_supplier_whose_name_contains_the_acronym_supplies_no_tax():
    evidence, _ = ground(SUPPLIER_CONTAINS_TPS, "fr-CA")
    assert evidence.taxes == []


@pytest.mark.parametrize("text,locale,expected_tax", [
    (CONTROL_FRENCH, "fr-FR", (Decimal("20"), Decimal("40.00"))),
    (CONTROL_UK, "en-GB", (Decimal("20"), Decimal("40.00"))),
])
def test_the_french_and_english_controls_keep_their_own_tax_handling(text, locale, expected_tax):
    evidence, result = ground(text, locale)
    assert taxes(evidence) == [expected_tax]
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] == 240.0
