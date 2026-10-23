import json
import sys
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import DocumentEvidence, EvidenceText, from_text
from invoice_grounding import UNVERIFIED, read_model, reconcile, run_grounded_inference
from invoice_validation import (
    IDENTIFIER_CALIBRATION, IDENTIFIER_SEVERITIES, canonical_name, document_validation,
    identifier_checks, luhn, parse_date, render_report, valid_ein, valid_fr_vat, valid_iban,
    valid_siret, valid_uk_vat,
)


def document(*, amount="20", subtotal="20", tax="4", ttc="24", suffix="", qty="2", price="10"):
    return from_text(
        "INVOICE GB-2026-00001\nSupplier: Acme Ltd\nCurrency: GBP\nInvoice date: 2026-01-01\n"
        "Description | Quantity | Net unit price | Net amount\n"
        f"Service | {qty} | {price} | {amount}\n"
        f"Subtotal (net): GBP {subtotal}\nVAT (20%): GBP {tax}\nInvoice total: GBP {ttc}\n"
        f"Due date: 2026-02-15\nPayment terms: NET 45\n{suffix}",
    )


def model():
    return read_model(
        '{"supplier_name":"Acme Ltd","invoice_date":"2026-01-01","total_ht":20,"total_ttc":24,'
        '"line_items":[{"description":"Service","quantity":2,"unit_price":10,"total_ht":20}]}'
    )


def test_capture_replay_is_recovered_without_a_document_anomaly():
    row = next(row for row in map(json.loads, (ROOT / "data/test.jsonl").read_text().splitlines()) if row["id"] == "invoice-0435")
    payload = read_model(json.dumps(row["output"]))
    payload["line_items"][2].update(quantity=Decimal(54), unit_price=Decimal("77.68"), total_ht=Decimal("4210.32"))
    evidence = from_text(row["input"])
    result = reconcile(evidence, [payload])
    assert result.payload == row["output"]
    assert not result.validation.flags
    assert not any(flag.severity == "erreur" for flag in result.validation.flags)
    report = render_report(evidence, result.validation, result.statuses, [])
    assert "Aucune anomalie" in report
    assert "4210" not in report
    assert "Pydantic" not in report
    assert "récupérée" in report


def test_printed_arithmetic_error_is_retained_and_flagged_outside_scored_json():
    evidence = document(amount="19")
    result = reconcile(evidence, [model()])
    assert result.payload["line_items"][0] == {"description": "Service", "quantity": 2, "unit_price": 10, "total_ht": 19}
    assert result.payload["total_ht"] == 20
    assert set(result.payload) == {"supplier_name", "invoice_date", "total_ht", "total_ttc", "line_items"}
    flags = result.validation.export()["flags"]
    assert next(flag for flag in flags if flag["code"] == "line_arithmetic")["severity"] == "erreur"
    subtotal = next(flag for flag in flags if flag["code"] == "subtotal")
    assert "cohérent avec" in subtotal["reason"]
    assert "1" in subtotal["reason"]
    assert result.failed_rows == []


@pytest.mark.parametrize("left,right,equivalent", [
    ("Acme Ltd", "Acme Limited", True), ("ACMÉ SAS", "Acme S.A.S.", True),
    ("Acme Ltd", "Acmes Ltd", False),
])
def test_supplier_variants_have_graphies_and_not_model_generated_errors(left, right, equivalent):
    evidence = from_text(document(suffix=f"Supplier: {right}").text.replace("Supplier: Acme Ltd", f"Supplier: {left}", 1))
    validation = document_validation(evidence)
    flags = [flag for flag in validation.flags if flag.code.startswith("name_")]
    assert len(flags) == 1
    assert left in flags[0].printed and right in flags[0].printed
    assert flags[0].severity == ("information" if equivalent else "avertissement")
    assert (canonical_name(left) == canonical_name(right)) is equivalent


def test_equivalent_supplier_forms_are_harmonized_without_an_error_or_false_null():
    evidence = document(suffix="Supplier: Acme Limited")
    result = reconcile(evidence, [model()])
    assert result.payload["supplier_name"] == "Acme Ltd"
    assert not any(flag.severity == "erreur" for flag in result.validation.flags)


def test_client_variants_are_checked_separately_from_supplier():
    evidence = document(suffix="Bill to: Summit LLC\nBill to: Summits LLC")
    flags = [flag for flag in document_validation(evidence).flags if flag.code == "name_variant"]
    assert len(flags) == 1
    assert flags[0].location == "Nom du client"
    assert "Summit LLC" in flags[0].printed and "Summits LLC" in flags[0].printed


def test_due_date_net_45_variants_and_calendar_controls():
    evidence = from_text(document().text.replace("2026-02-15", "2026-02-16"))
    flags = document_validation(evidence).flags
    flag = next(flag for flag in flags if flag.code == "payment_terms")
    assert flag.severity == "avertissement"
    assert flag.expected == "2026-02-15"
    assert flag.difference == "1"
    assert not document_validation(document()).flags
    assert any(flag.code == "due_before_invoice" for flag in document_validation(
        from_text(document().text.replace("2026-02-15", "2025-12-31")),
    ).flags)
    assert parse_date("Date : 31/02/2026", "fr-FR")[0] is None
    assert any(flag.code == "date_invalid" for flag in document_validation(
        from_text(document().text.replace("2026-01-01", "2026-02-31")),
    ).flags)
    mixed = from_text(document().text.replace("2026-02-15", "15.02.2026"))
    assert any(flag.code == "date_formats" for flag in document_validation(mixed).flags)
    future = from_text(document().text.replace("2026-01-01", "2099-01-01"))
    assert any(flag.code == "date_plausibility" for flag in document_validation(future, today=date(2026, 10, 23)).flags)


@pytest.mark.parametrize("validator,valid,invalid", [
    (valid_siret, "73282932000074", "73282932000075"),
    (valid_iban, "FR1420041010050500013M02606", "FR1420041010050500013M02607"),
    (valid_uk_vat, "GB123456789", "GB1234"),
    (valid_ein, "12-3456789", "123-456789"),
])
def test_identifier_checks_valid_and_invalid(validator, valid, invalid):
    assert validator(valid)
    assert not validator(invalid)


def test_fr_vat_key_and_siren_match():
    assert luhn("73282932000074")
    assert valid_fr_vat("FR44732829320", "73282932000074")
    assert not valid_fr_vat("FR44732829321", "73282932000074")
    assert not valid_fr_vat("FR00732829320", "73282932000074")


def test_sixty_reference_documents_have_no_error_level_false_positives_and_reproduce_calibration():
    counts = Counter()
    for row in map(json.loads, (ROOT / "data/test.jsonl").read_text().splitlines()):
        for kind, value, ok in identifier_checks(row["input"]):
            counts[kind, "checked"] += 1
            counts[kind, "failed"] += not ok
        validation = document_validation(from_text(row["input"], locale=row["locale"]), today=date(2026, 10, 23))
        assert not [flag for flag in validation.flags if flag.severity == "erreur"], row["id"]
    for kind, expected in IDENTIFIER_CALIBRATION.items():
        assert counts[kind, "checked"] == expected["checked"]
        assert counts[kind, "failed"] == expected["failed"]
    assert IDENTIFIER_SEVERITIES["siret"] == "information"


def test_identifier_failures_are_information_or_error_according_to_calibration():
    evidence = document(suffix=(
        "SIRET : 73282932000075\nTVA : FR00732829320\n"
        "IBAN FR1420041010050500013M02607\nVAT ID GB1234\nEIN 123-456789"
    ))
    flags = {flag.code: flag for flag in document_validation(evidence).flags}
    assert flags["identifier_siret"].severity == "information"
    for code in ("fr_vat", "iban", "uk_vat", "ein"):
        assert flags[f"identifier_{code}"].severity == "erreur"


def test_duplicate_and_tax_and_country_checks_are_document_only():
    evidence = document(suffix="Service | 2 | 10 | 20")
    # Explicit duplicate table row before totals.
    evidence = from_text(document().text.replace("Subtotal", "Service | 2 | 10 | 20\nSubtotal", 1))
    assert any(flag.code == "duplicate_line" for flag in document_validation(evidence).flags)
    assert any(flag.code == "tax_total" for flag in document_validation(document(ttc="25")).flags)
    assert any(flag.code == "tax_rate" for flag in document_validation(document(tax="5", ttc="25")).flags)
    country = from_text(document().text.replace("GBP", "USD"))
    assert any(flag.code == "country_currency" for flag in document_validation(country).flags)
    unusual = from_text(document().text.replace("20%", "99%"))
    assert any(flag.code == "country_tax_rate" for flag in document_validation(unusual).flags)


def test_printed_tax_base_is_checked_without_assuming_all_lines_have_the_same_rate():
    evidence = document()
    text = evidence.text.replace("VAT (20%): GBP 4", "VAT (20%) on GBP 15: GBP 4\nVAT (0%) on GBP 5: GBP 0")
    flags = document_validation(from_text(text)).flags
    tax_flag = next(flag for flag in flags if flag.code == "tax_printed_basis")
    assert tax_flag.severity == "erreur"
    assert tax_flag.expected == "3.00"
    correct = from_text(text.replace("GBP 15", "GBP 20").replace("GBP 5: GBP 0", "GBP 0: GBP 0"))
    assert not [flag for flag in document_validation(correct).flags if flag.severity == "erreur"]


def test_printed_discount_and_line_tax_valid_and_invalid():
    text = (
        "Currency: GBP\nDescription | Quantity | List price | Discount | Net unit price | Net amount | VAT % | VAT\n"
        "Service | 2 | 12.50 | 20% | 10 | 20 | 20% | 4\n"
        "Subtotal: 20\nVAT (20%): 4\nInvoice total: 24"
    )
    assert not [flag for flag in document_validation(from_text(text)).flags if flag.severity == "erreur"]
    bad_discount = from_text(text.replace("12.50", "15"))
    assert any(flag.code == "net_discount" for flag in document_validation(bad_discount).flags)
    bad_tax = from_text(text.replace("20% | 4", "20% | 5"))
    assert any(flag.code == "line_tax" for flag in document_validation(bad_tax).flags)


def test_low_confidence_ocr_is_not_evidence_of_a_document_error():
    from dataclasses import replace
    evidence = document(amount="19", suffix="EIN 123-456789")
    for line in evidence.lines:
        if "Service" in line.text or "EIN" in line.text:
            line.tokens = [replace(token, confidence=20) for token in line.tokens]
    flags = document_validation(evidence).flags
    assert not [flag for flag in flags if flag.code in {"line_arithmetic", "identifier_ein", "subtotal"}]


def test_ocr_unavailable_never_presents_model_values_as_verified():
    evidence = DocumentEvidence("", None, [], [], available=False)
    result = reconcile(evidence, [model()])
    assert set(result.statuses.values()) == {UNVERIFIED}
    assert all(value is None for value in result.payload["line_items"][0].values())
    report = render_report(evidence, result.validation, result.statuses, [], available=False)
    assert "Preuves OCR indisponibles : extraction non recoupée" in report
    assert "0 champs" in report
    assert not result.validation.flags


def test_ui_has_report_first_and_closed_technical_accordion():
    from app import main
    demo = main.build_app()
    components = demo.config["components"]
    accordion = next(component for component in components if component["type"] == "accordion")
    assert accordion["props"]["label"] == "Détails techniques"
    assert not accordion["props"]["open"]
    footer = next(component for component in components if component["type"] == "html")
    assert "OCR :" in footer["props"]["value"]
    assert main.preview_ui(None) == (None, None, "", "", {"flags": [], "passed": []})


def test_app_ocr_failure_is_reported_without_loading_model(caplog):
    from app import main
    with patch.object(main, "document_text", side_effect=RuntimeError("OCR absent")), \
            patch.object(main, "predict_invoice", side_effect=AssertionError("No evidence, no GPU")):
        outputs = list(main.run_extraction_ui(str(main.EXAMPLES[0][1])))
    payload, report, technical, validation = outputs[-1]
    assert "Preuves OCR indisponibles" in report
    assert "Pydantic" not in report
    assert "indisponibles" in technical
    assert not validation["flags"]
    assert all(payload[key] is None for key in ("supplier_name", "invoice_date", "total_ht", "total_ttc"))


def test_repeated_grounded_outputs_are_identical_and_model_errors_remain_technical():
    evidence = document()
    wrong = model()
    wrong["line_items"][0]["quantity"] = Decimal(54)
    raw = json.dumps(wrong, default=str)
    calls = []
    for _ in range(2):
        backend = MagicMock(return_value={"raw": raw, "finish_reason": "stop"})
        calls.append(run_grounded_inference(evidence, {"id": "test", "input": evidence.text}, [], backend))
    assert calls[0]["payload"] == calls[1]["payload"]
    assert calls[0]["report"] == calls[1]["report"]
    assert not calls[0]["validation"]["flags"]
    assert "54" not in calls[0]["report"]
    assert "54" in "\n".join(calls[0]["warnings"])


def test_second_pass_exception_has_french_cause_and_only_technical_trace(caplog):
    evidence = from_text(document().text.replace("Service | 2 |", "Service |  |"))
    backend = MagicMock(side_effect=[
        {"raw": json.dumps(model(), default=str), "finish_reason": "stop"},
        ValueError("ValueError"),
    ])
    result = run_grounded_inference(evidence, {"id": "test", "input": evidence.text}, [], backend)
    assert "La seconde passe a échoué" in result["report"]
    assert "service n'a pas communiqué" in result["report"]
    assert "ValueError" not in result["report"]
    assert "Pydantic" not in result["report"]
    assert "Traceback" in caplog.text
    assert result["payload"]["line_items"][0]["quantity"] is None
    assert result["payload"]["line_items"][0]["unit_price"] == 10
    assert not result["validation"]["flags"]


def test_deployment_ocr_packages_and_startup_status_are_explicit(caplog):
    from app import main
    packages = set((ROOT / "packages.txt").read_text().splitlines())
    assert {"tesseract-ocr", "tesseract-ocr-fra", "tesseract-ocr-eng"} <= packages
    with caplog.at_level("INFO"), patch.object(main, "ocr_availability", return_value=(False, "test absent")):
        demo = main.build_app()
    assert "Startup OCR: available=False" in caplog.text
    footer = next(component for component in demo.config["components"] if component["type"] == "html")
    assert "OCR : indisponible" in footer["props"]["value"]
