import shutil
import sys
from pathlib import Path

import pytest
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from app import main
from invoice_evidence import from_text
from invoice_grounding import read_model, reconcile
from invoice_validation import Flag, Validation, render_report

TABLE_TEXT = (
    "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
    "Description | Quantity | Unit price | Total\n"
    "Product A | 15 | 10 | 150\n"
    "Total Due: 150"
)
STATUSES = {
    "supplier_name": "récupéré (corroboré par OCR)",
    "invoice_date": "récupéré (corroboré par OCR)",
    "total_ht": "récupéré (corroboré par OCR)",
    "total_ttc": "récupéré (corroboré par OCR)",
}


def incomplete_evidence():
    evidence = from_text(TABLE_TEXT)
    evidence.rows = []
    evidence.source_kind = "consensus"
    evidence.ocr_cross_checked = True
    evidence.quality_issues = ["Tableau non reconstruit malgré les prétraitements."]
    return evidence


def complete_evidence():
    return from_text(TABLE_TEXT)


def test_small_source_missing_table_gets_low_quality_warning():
    report = render_report(
        incomplete_evidence(), Validation(), STATUSES, [], source_size=(461, 670),
    )
    assert "⚠️ Lecture incomplète" in report
    assert "Tableau attendu mais illisible" in report
    assert "4 récupérés" in report
    assert "4 champs vérifiés" not in report
    assert "✅ Facture cohérente" not in report


def test_normal_readable_table_has_no_resolution_warning():
    report = render_report(
        complete_evidence(), Validation(), {"total_ttc": "valide"}, [],
        source_size=(400, 400),
    )
    assert "Tableau attendu mais illisible" not in report
    assert "Mise en page non reconnue" not in report


def test_blank_template_has_no_missing_table_warning():
    blank = from_text(
        "FACTURE\nDescription | Quantity | Unit price | Total\n"
        "Sous-total HT :\nTVA :\nTotal TTC :"
    )
    blank.rows = []
    blank.quality_issues = ["Tableau non reconstruit malgré les prétraitements."]
    report = render_report(blank, Validation(), {}, [], source_size=(893, 1263))
    assert "Tableau attendu mais illisible" not in report
    assert "Mise en page non reconnue" not in report
    assert "Lecture incomplète" in report


def test_unfamiliar_readable_layout_gets_layout_not_resolution_warning():
    layout = from_text(
        "FACTURE FR-001\nDate 29/01/2019\n"
        "QTE DESIGNATION PRIX UNIT HT MONTANT HT\n"
        "1 Grand brun escargot pour manger 100.00 100.00",
        locale="fr-FR",
    )
    layout.rows = []
    layout.source_kind = "consensus"
    layout.ocr_cross_checked = True
    layout.quality_issues = ["Tableau non reconstruit malgré les prétraitements."]
    report = render_report(
        layout, Validation(), {}, [], source_size=(1200, 1600),
    )
    assert "Mise en page non reconnue" in report
    assert "Tableau attendu mais illisible" not in report


def test_half_resolution_missing_table_gets_quality_warning():
    report = render_report(
        incomplete_evidence(), Validation(), STATUSES, [], source_size=(446, 631),
    )
    assert "Tableau attendu mais illisible" in report


def test_timeout_report_does_not_claim_resolution_and_keeps_verdict_honest():
    evidence = from_text("FACTURE FR-001 texte illisible")
    evidence.quality_issues = ["Budget documentaire OCR de 45 s dépassé : preuves insuffisantes."]
    report = render_report(
        evidence, Validation(), {}, [], available=False, source_size=(893, 1263),
    )
    assert "Délai de lecture dépassé" in report
    assert "✅ Facture cohérente" not in report


def test_quality_warning_coexists_with_genuine_arithmetic_anomaly():
    validation = Validation(flags=[Flag(
        code="line_arithmetic",
        severity="erreur",
        location="Ligne 1",
        printed="150",
        expected="149",
        difference="1",
        reason="quantité × prix ne correspond pas au montant imprimé.",
    )])
    report = render_report(
        incomplete_evidence(), validation, STATUSES, [], source_size=(461, 670),
    )
    assert "anomalie" in report
    assert "Tableau attendu mais illisible" in report


def require_real_tesseract():
    if not shutil.which("tesseract"):
        pytest.skip("Tesseract is not installed")
    if not {"fra", "eng"} <= set(pytesseract.get_languages()):
        pytest.skip("Tesseract fra/eng languages are missing")


def test_example_company_real_report_is_not_marked_coherent():
    require_real_tesseract()
    path = ROOT / "diagnostics" / "uk-example-company.png"
    if not path.is_file():
        pytest.skip("Diagnostic image missing")
    image = main.open_image(path)
    evidence = main.ocr_image(image).evidence
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    report = render_report(
        evidence, result.validation, result.statuses, [], source_size=image.size,
    )
    assert "Tableau attendu mais illisible" in report
    assert "✅ Facture cohérente" not in report


def test_ellington_real_report_has_no_resolution_warning():
    require_real_tesseract()
    path = ROOT / "diagnostics" / "uk-ellington-wood.png"
    if not path.is_file():
        pytest.skip("Diagnostic image missing")
    image = main.open_image(path)
    evidence = main.ocr_image(image).evidence
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    report = render_report(
        evidence, result.validation, result.statuses, [], source_size=image.size,
    )
    assert "Tableau attendu mais illisible" not in report
