"""PR126 — the verification report must not call a read value absent.

These tests pin the four report sentences that PR126 changed, in both directions: each new sentence has to
appear for the evidence shape that motivates it, and the original sentence has to survive byte-for-byte when
the document really does not carry the information. Nothing here touches the strict JSON payload: every
assertion reads `document_validation`, whose result travels in `diagnostics`.
"""
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text
from invoice_validation import document_validation
from ocr_consensus import build_consensus


def consensus(evidence):
    """`document_validation` only reaches these lines on cross-checked consensus evidence."""
    evidence.source_kind = "consensus"
    evidence.ocr_cross_checked = True
    return evidence


def unverified(validation):
    return validation.not_verified


# --- a rate printed on an item line, while the labelled tax line carries only an amount --------------

VAT_LINE_WITHOUT_RATE = (
    "TAX INVOICE\nSupplier: Example Supplier\nCurrency: USD\nInvoice date: 03/11/2024\n"
    "Description | Quantity | Unit price | Rate | Amount\n"
    "Logo design | 5 | 290 | 10% | 1450.00\n"
    "Subtotal: 1450.00\nTAX 145.00\nTotal: 1595.00\n"
)


def test_rate_read_on_an_item_line_is_reported_instead_of_a_missing_rate():
    evidence = consensus(from_text(VAT_LINE_WITHOUT_RATE, locale="en-US"))
    validation = document_validation(evidence)
    assert [tax.rate for tax in evidence.taxes] == [None], "the fixture must leave the tax rate unread"
    messages = unverified(validation)
    assert any(message.startswith("TVA : taux absent de la ligne de ventilation ; « 10 % »")
               for message in messages), messages
    assert "TVA : taux non lu ou ambigu." not in messages


def test_passed_total_line_never_prints_a_none_rate():
    validation = document_validation(consensus(from_text(VAT_LINE_WITHOUT_RATE, locale="en-US")))
    assert any(message.startswith("HT + taxes lues (145.00) = TTC ; les montants de taxe lus ne portent aucun taux")
               for message in validation.passed), validation.passed
    assert not any("None" in message for message in validation.passed), validation.passed


# --- the original sentences stay when the information is genuinely missing ---------------------------

NO_RATE_ANYWHERE = (
    "TAX INVOICE\nSupplier: Example Supplier\nCurrency: USD\nInvoice date: 03/11/2024\n"
    "Description | Quantity | Unit price | Amount\n"
    "Logo design | 5 | 290 | 1450.00\n"
    "Subtotal: 1450.00\nTAX 145.00\nTotal: 1595.00\n"
)


def test_a_document_without_any_percentage_keeps_the_plain_unread_rate_message():
    validation = document_validation(consensus(from_text(NO_RATE_ANYWHERE, locale="en-US")))
    assert "TVA : taux non lu ou ambigu." in unverified(validation)
    assert not any("taux absent de la ligne de ventilation" in message for message in unverified(validation))


PERCENT_IN_PAYMENT_TERMS = (
    "FACTURE\nFournisseur: Exemple SARL\nDevise: EUR\nDate de facture: 2024-03-11\n"
    "Description | Quantité | Prix unitaire | Montant\n"
    "Prestation | 2 | 10 | 20\nSous-total: 20,00\nTotal: 20,00\n"
    "Modalités de paiement: pour éviter des frais supplémentaires (8 % par année), veuillez acquitter.\n"
)


def test_a_percentage_in_payment_terms_is_never_reported_as_a_read_tax_rate():
    evidence = consensus(from_text(PERCENT_IN_PAYMENT_TERMS, locale="fr-FR"))
    messages = unverified(document_validation(evidence))
    ventilation = [message for message in messages if message.startswith("TVA par taux :")]
    assert ventilation == ["TVA par taux : non vérifiable, aucune ventilation n’a été lue."], ventilation


# --- a rate read in the line detail, with no summary line to attach it to ----------------------------

HOURLY_RATE_COLUMN = (
    "FACTURE\nFournisseur: Atelier Example\nDevise: EUR\nDate de facture: 2022-08-19\n"
    "Description | Quantité | Unité | Prix unitaire HT | % TVA | Total TVA | Total TTC\n"
    "Main d’œuvre | 5 | h | 60,00 € | 20 % | 60,00 € | 360,00 €\n"
    "Sous-total: 300,00\nTotal TVA: 60,00\nTotal général: 360,00\n"
)


def test_a_rate_confirmed_by_two_pretraitments_says_so():
    evidence = build_consensus({family: [from_text(HOURLY_RATE_COLUMN, locale="fr-FR")]
                                for family in ("deskew", "clahe")})
    evidence.source_kind = "consensus"
    evidence.ocr_cross_checked = True
    assert evidence.rows[0].family_votes["tax_rate"] == {Decimal(20): ("clahe", "deskew")}
    messages = unverified(document_validation(evidence))
    assert ("TVA par taux : non vérifiable, aucune ventilation n’a été lue ; « 20 % » est lu dans le détail "
            "des lignes par plusieurs prétraitements.") in messages, messages


def test_a_rate_seen_by_a_single_pretraitment_does_not_claim_corroboration():
    evidence = consensus(from_text(HOURLY_RATE_COLUMN, locale="fr-FR"))
    assert evidence.rows[0].cells["tax_rate"], "the fixture must read the rate into the line detail"
    messages = unverified(document_validation(evidence))
    clause = [message for message in messages if message.startswith("TVA par taux :")]
    assert len(clause) == 1, messages
    assert "« 20 % »" in clause[0] and "prétraitement" in clause[0], clause
    assert "plusieurs" not in clause[0], clause


# --- dates ------------------------------------------------------------------------------------------

def test_an_unreadable_date_under_a_label_is_not_reported_as_no_date_read():
    text = ("INVOICE\nSupplier: Adventure Club\nCurrency: USD\nInvoice date: march 10. 2021\n"
            "Description | Quantity | Unit price | Amount\nBus | 2 | 100 | 200\nSubtotal: 200\nTotal: 200\n")
    validation = document_validation(consensus(from_text(text, locale="en-US")))
    messages = unverified(validation)
    assert any(message.startswith("Date de facture : non vérifiable, le libellé de date de facture est lu")
               for message in messages), messages
    assert "Date de facture : non vérifiable, date non lue." not in messages


def test_a_date_exported_from_elsewhere_is_reported_as_not_recovered_by_this_check():
    text = ("INVOICE\nSupplier: Adventure Club\nCurrency: USD\n"
            "Description | Quantity | Unit price | Amount\nBus | 2 | 100 | 200\nSubtotal: 200\nTotal: 200\n")
    evidence = consensus(from_text(text, locale="en-US"))
    evidence.invoice_date = "2024-11-05"
    messages = unverified(document_validation(evidence))
    assert any(message.startswith(
        "Date de facture : non vérifiable, le JSON exporte une date que ce contrôle ne retrouve pas")
        for message in messages), messages
    assert "Date de facture : non vérifiable, date non lue." not in messages


def test_a_document_without_any_date_keeps_the_plain_date_non_lue_message():
    text = ("INVOICE\nSupplier: Adventure Club\nCurrency: USD\n"
            "Description | Quantity | Unit price | Amount\nBus | 2 | 100 | 200\nSubtotal: 200\nTotal: 200\n")
    messages = unverified(document_validation(consensus(from_text(text, locale="en-US"))))
    assert "Date de facture : non vérifiable, date non lue." in messages
    assert "Échéance : non vérifiable, date d’échéance non lue." in messages


def test_an_unreadable_due_label_is_not_reported_as_no_due_date_read():
    text = ("INVOICE\nSupplier: Zylker\nCurrency: USD\nInvoice date: 18 May 2023\n"
            "Description | Quantity | Unit price | Amount\nJeans | 2 | 100 | 200\nSubtotal: 200\n"
            "Balance Due on receipt of this invoice\n")
    messages = unverified(document_validation(consensus(from_text(text, locale="en-US"))))
    assert any(message.startswith("Échéance : non vérifiable, le libellé d’échéance est lu")
               for message in messages), messages
    assert "Échéance : non vérifiable, date d’échéance non lue." not in messages


# --- currency ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "printed,quoted",
    [
        ("Amount: 20.00 $\n", ["$"]),
        ("Amount: 20.00 €\nAmount: 5.00 £\n", ["€", "£"]),
    ],
)
def test_currency_clause_only_quotes_the_indications_the_document_prints(printed, quoted):
    text = ("INVOICE\nSupplier: Acme\nInvoice date: 2024-03-11\n"
            "Description | Quantity | Unit price | Amount\nService | 2 | 10 | 20\nSubtotal: 20\n"
            f"Total: 20\n{printed}")
    evidence = consensus(from_text(text, locale="en-US"))
    messages = unverified(document_validation(evidence))
    clause = [message for message in messages if message.startswith("Devise :")]
    assert len(clause) == 1, messages
    assert "aucune devise unique n’a été lue ;" in clause[0], clause
    assert all(f"« {symbol} »" in clause[0] for symbol in quoted), clause
    assert all(symbol in evidence.text for symbol in quoted), clause


def test_an_unambiguously_printed_currency_produces_no_devise_message():
    text = ("INVOICE\nSupplier: Acme\nCurrency: EUR\nInvoice date: 2024-03-11\n"
            "Description | Quantity | Unit price | Amount\nService | 2 | 10 | 20\nSous-total: 20\nTotal: 20\n")
    messages = unverified(document_validation(consensus(from_text(text, locale="fr-FR"))))
    assert not any(message.startswith("Devise : non vérifiable") for message in messages), messages
