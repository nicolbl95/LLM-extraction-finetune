"""Field-selective retention of labelled totals in a document rejected on average confidence."""

import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import from_text
from invoice_grounding import (
    RECOVERED, UNVERIFIED, reconcile, run_grounded_inference, selective_total_recovery,
)
from ocr_config import OCR_CONFIG

BLUR_ISSUE = "Confiance OCR moyenne insuffisante : 12.4/100 (minimum 85)."
FRENCH_TEXT = """FACTURE FR-2019-014
Date : 29/01/2019
Fournisseur : Mon Entreprise
QTE DESIGNATION PRIX UNITAIRE HT MONTANT HT
0,00 0,00 0,00 0,00 0,00 0,00 0,00 0,00
Sous-total HT : 1 350,00 €
TVA 20,00 % : 270,00 €
Total TTC : 1 620,00 €
"""


def consensus(text, *, legible=(), locale="fr-FR", clear_rows=True):
    """Consensus evidence: labelled totals imply two agreeing preprocessing families.

    ``ocr_consensus`` only fills ``totals`` for a value that at least two families read under
    the same label, so the fixture marks the evidence as consensus instead of replaying OCR.
    """
    evidence = from_text(text, locale=locale)
    lines = text.splitlines()

    def rescale(token):
        index = int(token.box[1] // 10)
        keep = any(lines[index].startswith(prefix) for prefix in legible)
        return token if keep else replace(token, confidence=5)

    evidence.tokens = [rescale(token) for token in evidence.tokens]
    for line in evidence.lines:
        line.tokens = [rescale(token) for token in line.tokens]
    if clear_rows:
        evidence.rows = []
    evidence.source_kind = "consensus"
    evidence.ocr_cross_checked = True
    evidence.quality_issues = [BLUR_ISSUE]
    return evidence


def french(**kwargs):
    return consensus(
        FRENCH_TEXT,
        legible=("Sous-total HT", "Total TTC"),
        **kwargs,
    )


def blurred_rejection(evidence):
    result = reconcile(evidence, [{}])
    assert BLUR_ISSUE in result.warnings
    return result


def test_blurred_document_keeps_its_legible_labelled_totals():
    result = blurred_rejection(french())
    assert result.payload["total_ht"] == 1350.0
    assert result.payload["total_ttc"] == 1620.0
    assert result.statuses["total_ht"] == RECOVERED
    assert result.statuses["total_ttc"] == RECOVERED
    assert result.provenance["total_ht"][0]["raw"] == "1 350,00 €"
    assert result.provenance["total_ttc"][0]["value"] == "1620.00"


def test_other_fields_of_a_rejected_document_stay_unverified():
    result = blurred_rejection(french())
    assert result.payload["supplier_name"] is None
    assert result.payload["invoice_date"] is None
    assert result.statuses["supplier_name"] == UNVERIFIED
    assert result.statuses["invoice_date"] == UNVERIFIED
    assert not result.validation.flags
    assert not any("aucun champ présenté comme vérifié" in warning for warning in result.warnings)
    assert any("seuls les totaux libellés" in warning for warning in result.warnings)


def test_retained_totals_do_not_release_the_model():
    predict = MagicMock(side_effect=AssertionError("Le modèle reste refusé tant que le document est rejeté"))
    result = run_grounded_inference(french(), {"input": FRENCH_TEXT}, [], predict)
    predict.assert_not_called()
    assert result["payload"]["total_ht"] == 1350.0
    assert result["payload"]["total_ttc"] == 1620.0
    assert result["model_info"]["used"] is False
    assert result["model_info"]["passes_requested"] == 0


def test_conflicting_labelled_totals_are_not_chosen_between():
    text = FRENCH_TEXT.replace(
        "Sous-total HT : 1 350,00 €",
        "Sous-total HT : 1 350,00 €\nSous-total HT : 1 450,00 €",
    )
    result = blurred_rejection(consensus(text, legible=("Sous-total HT", "Total TTC")))
    assert result.payload["total_ht"] is None
    assert result.statuses["total_ht"] == UNVERIFIED
    # The conflict is local to the HT label: the cleanly read TTC line keeps its own evidence.
    assert result.payload["total_ttc"] == 1620.0


def test_one_confidence_dropped_digit_keeps_only_that_total_out():
    evidence = french()
    for line in evidence.lines:
        if "350" in line.text:
            line.tokens = [
                replace(token, confidence=OCR_CONFIG.fast_token_confidence - 1)
                if "350" in token.text else token
                for token in line.tokens
            ]
    result = blurred_rejection(evidence)
    assert result.payload["total_ht"] is None
    assert result.statuses["total_ht"] == UNVERIFIED
    assert result.payload["total_ttc"] == 1620.0


def test_same_amount_under_both_labels_is_semantically_ambiguous():
    text = FRENCH_TEXT.replace("Total TTC : 1 620,00 €", "Total TTC : 1 350,00 €")
    result = blurred_rejection(consensus(text, legible=("Sous-total HT", "Total TTC")))
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] is None


def test_corrupted_label_line_is_not_retained():
    evidence = french()
    for line in evidence.lines:
        if "620" in line.text:
            line.tokens = [
                replace(token, text="\ufffd") if token.text == "€" else token
                for token in line.tokens
            ]
            line.text = line.text.replace("€", "\ufffd")
    result = blurred_rejection(evidence)
    assert result.payload["total_ttc"] is None
    assert result.payload["total_ht"] == 1350.0


def test_amount_without_any_printed_currency_is_not_retained():
    text = (
        "Invoice INV-9\nDate: 2026-01-01\nSupplier: Acme Ltd\nJunk unreadable noise row\n"
        "Subtotal: 1350.00\nInvoice total: 1620.00\n"
    )
    assert from_text(text).locale is None
    result = blurred_rejection(consensus(text, legible=("Subtotal", "Invoice total"), locale=None))
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] is None
    with_currency = text.replace("Subtotal: 1350.00", "Subtotal: $1350.00")
    retained = blurred_rejection(consensus(with_currency, legible=("Subtotal", "Invoice total"), locale=None))
    assert retained.payload["total_ht"] == 1350.0


def test_several_amounts_on_the_total_line_stay_ambiguous():
    text = FRENCH_TEXT.replace("Sous-total HT : 1 350,00 €", "Sous-total HT : 1 350,00 € 12,50 €")
    result = blurred_rejection(consensus(text, legible=("Sous-total HT", "Total TTC")))
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] == 1620.0


def test_single_family_reading_never_qualifies():
    evidence = french()
    evidence.source_kind = "ocr"
    evidence.ocr_cross_checked = False
    assert selective_total_recovery(evidence) == {}
    result = blurred_rejection(evidence)
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] is None


def test_reconstructed_table_keeps_going_through_the_row_constraints():
    text = (
        "FACTURE FR-1\nCurrency €\nDescription | Quantity | Unit price | Line total\n"
        "Prestation | 2 | 10 | 20\nSous-total HT : 1 350,00 €\nTotal TTC : 1 620,00 €\n"
    )
    evidence = consensus(text, legible=("Sous-total HT", "Total TTC"), clear_rows=False)
    assert evidence.rows
    assert selective_total_recovery(evidence) == {}
    result = blurred_rejection(evidence)
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] is None


def test_unlabelled_total_is_not_promoted():
    text = FRENCH_TEXT.replace("Total TTC : 1 620,00 €", "Total : 1 620,00 €")
    evidence = consensus(text, legible=("Sous-total HT", "Total :"))
    assert {number.value for number in evidence.totals.get("total_candidate", [])} == {Decimal("1620.00")}
    result = blurred_rejection(evidence)
    assert result.payload["total_ttc"] is None
    assert result.payload["total_ht"] == 1350.0
