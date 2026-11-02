"""Supplier identity must rest on issuer evidence, never on body prose or form labels."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text  # noqa: E402
from invoice_grounding import RECOVERED, VALID  # noqa: E402
from invoice_validation import Validation, render_report  # noqa: E402


def supplier_of(text: str, locale: str = "fr-FR"):
    return from_text(text, locale=locale).supplier


def test_shipping_prose_never_becomes_a_supplier():
    text = (
        "Conditions de livraison et colisage\n"
        "Conditions de mise à disposition et répartition des responsabilités.\n"
        "Nombre de colis, taille, poids unitaire, etc.\n"
        "Description Quantité Unité Prix unitaire HT\n"
        "Thé vert 5 Kg 15,00 75,00\n"
    )
    assert supplier_of(text) is None


def test_ordinary_comma_sentence_never_becomes_a_supplier():
    assert supplier_of("Veuillez noter que les frais, taxes et remises sont inclus.") is None


@pytest.mark.parametrize(
    "text",
    [
        "Notre produit, 3 tailles disponibles",
        "Produit standard, 75002 exemplaires",
        "Gamme complète, 12 coloris disponibles",
    ],
)
def test_comma_prose_with_numbers_never_becomes_a_supplier(text):
    assert supplier_of(text) is None


def test_form_labels_never_become_a_supplier():
    text = (
        "Adresse de livraison\n"
        "Pays d'origine : France\n"
        "Incoterm FCA\n"
        "Description Quantité Prix\n"
        "Service 1 100,00\n"
    )
    assert supplier_of(text) is None


def test_supplier_followed_by_a_real_address_is_accepted():
    assert supplier_of("Acme Industries, 12 rue de la Paix, 75002 Paris") == "Acme Industries"


def test_address_rule_requires_address_shaped_evidence():
    assert supplier_of("Acme Industries, qualité et service") is None


def test_garbled_street_type_with_place_name_is_still_an_address():
    assert supplier_of("Ellington Wood Decor, 36 Terrick gt 2N") == "Ellington Wood Decor"


def test_supplier_in_issuer_block_is_accepted():
    assert supplier_of(
        "Fast Photography\n2132 Duncan Street\nSan Francisco, CA 94131", locale="en-US"
    ) == "Fast Photography"


def test_ambiguous_candidates_are_refused():
    assert supplier_of("Alpha Studio\nBeta Studio\nGamma Studio") is None


def test_document_without_supplier_stays_empty():
    assert supplier_of(
        "FACTURE\nDate : 14/05/2021\nDescription Quantité Prix\nService 1 100,00"
    ) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("PURE CLEANING\n36 Terrick Rd\nEllington", "PURE CLEANING"),
        ("Lawn maintenance\n25 Kendell Street\nSheffield", "Lawn maintenance"),
        ("Mon Entreprise\n22 Avenue Voltaire\n13000 Marseille, France", "Mon Entreprise"),
        ("Éditions Mirada\n3230, avenue Duchesne\nLongueuil", "Éditions Mirada"),
    ],
)
def test_existing_supplier_extractions_are_preserved(text, expected):
    assert supplier_of(text) == expected


def test_display_labels_never_promote_recovery():
    from app.main import display_status_label

    assert display_status_label(VALID) == "vérifié"
    assert display_status_label(RECOVERED) == "récupéré (document)"
    assert display_status_label(RECOVERED) != display_status_label(VALID)
    assert display_status_label("non vérifié") == "non vérifié"
    assert display_status_label("neutralisé (non fiable)") == "incertain"


def test_report_counts_recovered_separately_from_verified():
    evidence = from_text("FACTURE\nDate : 14/05/2021\nService 1 100,00")
    statuses = {"supplier_name": VALID, "invoice_date": RECOVERED}
    report = render_report(evidence, Validation(), statuses, [])
    assert "1 champs vérifiés, 1 récupéré" in report
    assert "2 champs vérifiés" not in report
