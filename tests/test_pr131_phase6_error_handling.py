"""PR131 Phase 6: error handling must report what it actually observed.

Each test here fails on the pre-Phase-6 baseline: a failed language listing cached as a
verdict, information-severity findings dropped from the only text the demo shows, and a
cleaner claiming it replaced a value that was already null.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src", ROOT / "data"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def listing_failing_once():
    attempts: list[str] = []

    def get_languages(config=""):
        attempts.append(config)
        if len(attempts) == 1:
            raise OSError("[WinError 5] Accès refusé")
        return ["eng", "fra", "osd"]

    return attempts, get_languages


def test_a_transient_language_listing_failure_is_not_cached_as_a_verdict(monkeypatch):
    from app import main

    attempts, get_languages = listing_failing_once()
    monkeypatch.setattr(main, "_OCR_LANGUAGES_CACHE", {})
    monkeypatch.setattr(main, "_OCR_LISTING_FAILURES", {}, raising=False)
    with patch.object(main.pytesseract, "get_languages", side_effect=get_languages):
        assert main.installed_ocr_languages() == frozenset()
        assert main.installed_ocr_languages() == frozenset({"eng", "fra", "osd"})
    assert len(attempts) == 2, "the next document must retry a listing that was never answered"


def test_a_document_is_extracted_again_after_a_transient_listing_failure(monkeypatch):
    from app import main

    attempts, get_languages = listing_failing_once()
    monkeypatch.setattr(main, "_OCR_LANGUAGES_CACHE", {})
    monkeypatch.setattr(main, "_OCR_LISTING_FAILURES", {}, raising=False)
    with patch.object(main.pytesseract, "get_languages", side_effect=get_languages):
        with pytest.raises(RuntimeError, match="liste des langues"):
            main.require_ocr_languages()
        assert main.require_ocr_languages()["effective"] == ["fra", "eng"], (
            "the upload after a transient listing error must be read, not refused"
        )
    assert len(attempts) == 2


def test_a_persistent_listing_failure_names_the_listing_instead_of_the_installation(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "_OCR_LANGUAGES_CACHE", {})
    monkeypatch.setattr(main, "_OCR_LISTING_FAILURES", {}, raising=False)
    with patch.object(main.pytesseract, "get_languages", side_effect=OSError("boom")):
        with pytest.raises(RuntimeError) as refused:
            main.require_ocr_languages()
    message = str(refused.value)
    assert "liste des langues Tesseract n'a pas pu être lue" in message
    assert "OSError: boom" in message, "the refusal must carry the reason it actually saw"
    assert "installez Tesseract et les langues" not in message


def test_a_listing_that_succeeded_without_the_languages_still_blames_the_installation(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "_OCR_LANGUAGES_CACHE", {})
    monkeypatch.setattr(main, "_OCR_LISTING_FAILURES", {}, raising=False)
    with patch.object(main.pytesseract, "get_languages", return_value=["osd"]):
        with pytest.raises(RuntimeError) as refused:
            main.require_ocr_languages()
    message = str(refused.value)
    assert "installez Tesseract et les langues fra/eng" in message
    assert "liste des langues" not in message, (
        "a readable listing that lacks the languages is an installation gap, not a read error"
    )


def information_flag(reason: str) -> dict:
    return {
        "code": "date_formats", "severity": "information", "location": "Dates",
        "printed": "iso, fra_long", "expected": "format uniforme conseillé",
        "difference": "", "reason": reason,
    }


def coherent_diagnostics(reason: str) -> tuple[dict, dict]:
    payload = {
        "supplier_name": "Acme SARL", "invoice_date": "2026-05-12",
        "total_ht": 200.0, "total_ttc": 240.0,
        "line_items": [{"description": "Prestation", "quantity": 2.0, "unit_price": 100.0, "total_ht": 200.0}],
    }
    statuses = {
        "supplier_name": "valide", "invoice_date": "valide", "total_ht": "valide",
        "total_ttc": "valide", "line_items.0.description": "valide",
        "line_items.0.quantity": "valide", "line_items.0.unit_price": "valide",
        "line_items.0.total_ht": "valide",
    }
    diagnostics = {
        "field_statuses": statuses,
        "validation": {"flags": [information_flag(reason)], "passed": [], "not_verified": []},
        "evidence_available": True, "document_coverage": {}, "model_info": {},
    }
    return payload, diagnostics


def test_an_information_flag_reaches_the_summary_without_changing_the_verdict():
    from app import main

    payload, diagnostics = coherent_diagnostics("Plusieurs formats imprimés dans le même document.")
    summary = main._compact_validation_summary(payload, diagnostics, "")
    assert "Extraction cohérente" in summary, "information is not an anomaly and must not become one"
    assert "Anomalies à vérifier" not in summary
    assert "Plusieurs formats imprimés dans le même document." in summary


def test_information_flags_are_stated_alongside_real_anomalies():
    from app import main

    payload, diagnostics = coherent_diagnostics("Plusieurs formats imprimés dans le même document.")
    diagnostics["validation"]["flags"].append({
        "code": "subtotal", "severity": "erreur", "location": "Sous-total",
        "printed": 210.0, "expected": 200.0, "difference": 10.0,
        "reason": "La somme des lignes ne correspond pas au sous-total imprimé.",
    })
    diagnostics["field_statuses"]["total_ht"] = "non vérifié"
    summary = main._compact_validation_summary(payload, diagnostics, "")
    assert "Anomalies à vérifier" in summary
    assert "Plusieurs formats imprimés dans le même document." in summary


def test_information_flags_of_a_real_document_appear_in_both_surfaces():
    from app import main
    from invoice_evidence import from_text
    from invoice_grounding import reconcile
    from invoice_validation import render_report

    evidence = from_text(
        "FACTURE FR-2026-00021\nFournisseur: Acme SARL\nDate de facture: 12/05/2026\n"
        "Date d'échéance: 2026-05-30\nDescription | Quantité | Prix unitaire HT | Montant HT\n"
        "Prestation | 2 | 100,00 | 200,00\nTotal HT: 200,00 EUR\nTotal TVA: 40,00 EUR\n"
        "Total TTC: 240,00 EUR\n"
    )
    result = reconcile(evidence, [{}])
    information = [flag for flag in result.validation.export()["flags"] if flag["severity"] == "information"]
    assert [flag["code"] for flag in information] == ["date_formats"], information
    report = render_report(
        evidence, result.validation, result.statuses, [], available=evidence.available,
        insufficient=False, suggestions={}, model_info={"used": False, "reason": "quota"}, source_size=None,
    )
    summary = main._compact_validation_summary(result.payload, {
        "field_statuses": result.statuses, "validation": result.validation.export(),
        "evidence": evidence, "evidence_available": evidence.available,
        "document_coverage": {}, "model_info": {"used": False, "reason": "quota"},
    }, report)
    for flag in information:
        assert flag["reason"] in report, "the full report is what the summary must not contradict"
        assert flag["reason"] in summary, (
            "the compact summary is the only verification text the published demo shows"
        )


def test_the_cleaner_stops_claiming_repairs_it_never_made():
    from schema import safe_invoice_payload

    payload = {
        "supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": 240.0,
        "line_items": [{"description": None, "quantity": None, "unit_price": None, "total_ht": None}],
    }
    warnings: list[str] = []
    cleaned = safe_invoice_payload(payload, warnings)
    assert cleaned["supplier_name"] is None and cleaned["line_items"][0]["quantity"] is None
    assert not [warning for warning in warnings if "remplacée par null" in warning], warnings
    assert not any("None" in warning for warning in warnings), warnings


def test_the_cleaner_still_reports_a_value_it_actually_replaced():
    from schema import safe_invoice_payload

    payload = {
        "supplier_name": "Acme SARL", "invoice_date": "2026-05-12",
        "total_ht": 200.0, "total_ttc": 240.0,
        "line_items": [{"description": "Prestation", "quantity": "deux", "unit_price": 100.0, "total_ht": 200.0}],
    }
    warnings: list[str] = []
    cleaned = safe_invoice_payload(payload, warnings)
    assert cleaned["line_items"][0]["quantity"] is None
    replacements = [warning for warning in warnings if "Valeur" in warning and "remplacée par null." in warning]
    assert [warning for warning in replacements if "None" in warning] == [], replacements
    assert any(
        warning.startswith("line_items.0.quantity") and "'deux'" in warning for warning in replacements
    ), warnings
