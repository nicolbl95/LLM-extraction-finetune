import json
import re
import sys
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import DocumentEvidence, EvidenceText, from_text, invoice_number_from_text
from invoice_grounding import UNVERIFIED, read_model, reconcile, run_grounded_inference
from ocr_consensus import build_consensus
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
    assert f"{sum(value in {'valide', 'récupéré (corroboré par OCR)'} for value in result.statuses.values())} champs vérifiés" in report


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
    report = render_report(evidence, result.validation, result.statuses, [])
    assert "⚠️ 2 anomalie(s) dans la facture" in report
    assert "Ligne 1" in report


UK_MULTI_RATE = (
    "INVOICE GB-2025-00110\nSupplier: Example Creative Ltd\nCurrency: GBP\n"
    "Invoice date: 2025-02-01\n"
    "Description | Quantity | Net unit price | Net amount | VAT % | VAT\n"
    "Logo design | 7.8 | 757.97 | 5912.17 | 20% | 1182.43\n"
    "Translation | 1 | 500 | 500 | 0% | 0\n"
    "Subtotal: 6412.17\nVAT (0%): 0\nVAT (20%): 1182.43\n"
    "Invoice total: 7594.60\nDue date: 2025-03-18\nPayment terms: NET 45"
)


def test_uk_multirate_document_has_complete_controls_in_ocr_fallback():
    evidence = from_text(UK_MULTI_RATE, locale="en-GB")
    backend = MagicMock(side_effect=RuntimeError("ZeroGPU quota exceeded; try again in 36 min"))
    result = run_grounded_inference(
        evidence, {"id": "uk-test", "input": evidence.text}, [], backend,
    )
    report = result["report"]
    assert evidence.invoice_number == "GB-2025-00110"
    assert "Analyse de la facture GB-2025-00110" in report
    assert "Lecture effectuée par reconnaissance de texte (OCR)" in report
    assert "Nouvel essai possible dans 36 min." in report
    assert "⚠️ 1 anomalie(s)" not in report
    for control in (
        "Ligne 1", "Sous-total", "TVA 0 %", "TVA 20 %",
        "HT + taxes lues", "Devise GBP cohérente", "Échéance :",
    ):
        assert control in report
    assert result["passes_requested"] == 1
    assert result["passes_succeeded"] == 0
    assert not result["validation"]["flags"]
    assert report.count("Devise") == 1
    assert all(result["payload"]["line_items"][0][key] is not None
               for key in ("quantity", "unit_price", "total_ht"))


def test_successful_model_generation_is_distinguished_from_ocr_fallback():
    evidence = document()
    backend = MagicMock(return_value={
        "raw": json.dumps(model(), default=str),
        "finish_reason": "stop",
        "generation_seconds": 1.25,
    })
    result = run_grounded_inference(
        evidence, {"id": "model-test", "input": evidence.text}, [], backend,
    )
    assert result["model_info"]["used"]
    assert result["model_info"]["passes_requested"] == 1
    assert result["model_info"]["passes_succeeded"] == 1
    assert result["model_info"]["generation_seconds"] == 1.25
    assert "Lecture effectuée par reconnaissance de texte (OCR)" not in result["report"]


@pytest.mark.parametrize("invoice_id", ["invoice-0406", "invoice-0435"])
def test_fr_and_us_fallback_reports_are_human_readable_without_duplicates(invoice_id):
    row = next(
        row for row in map(json.loads, (ROOT / "data/test.jsonl").read_text().splitlines())
        if row["id"] == invoice_id
    )
    evidence = from_text(row["input"], locale=row["locale"])
    backend = MagicMock(side_effect=RuntimeError("ZeroGPU quota exceeded; try again in 36 min"))
    report = run_grounded_inference(
        evidence, {"id": invoice_id, "input": evidence.text}, [], backend,
    )["report"]
    for forbidden in ("None", "non ancré", "description_non_verifiee", "ancrage", "Pydantic"):
        assert forbidden not in report
    assert report.count("### Anomalies dans la facture") == 1
    assert report.count("Lecture effectuée par reconnaissance de texte (OCR)") == 1
    assert "line_items." not in report


def test_consensus_ocr_arithmetic_disagreement_does_not_create_a_ghost_anomaly():
    bad_text = UK_MULTI_RATE.replace("5912.17", "5900.00", 1)
    evidence = build_consensus({
        "deskew": [from_text(bad_text, locale="en-GB")],
        "clahe": [from_text(bad_text, locale="en-GB")],
    })
    validation = document_validation(evidence)
    assert not any(flag.code == "line_arithmetic" for flag in validation.flags)
    assert evidence.rows[0].cells["quantity"] == []
    assert evidence.rows[0].cells["unit_price"] == []
    report = render_report(evidence, validation, {}, [])
    assert "Aucune anomalie n’est affirmée avec les éléments disponibles." in report


def _reference_report(invoice_id, *, invoice_number_readable=True, siret_readable=True):
    row = next(
        row for row in map(json.loads, (ROOT / "data/test.jsonl").read_text().splitlines())
        if row["id"] == invoice_id
    )
    text = row["input"]
    if not invoice_number_readable:
        text = re.sub(r"\b(?:FR|GB|US)-\d{4}-\d{5}\b", "NUMERO ILLISIBLE", text)
    if invoice_id == "invoice-0406" and not siret_readable:
        text = re.sub(r"(?<=SIRET : )\d{14}", "illisible", text, count=1)
    evidence = build_consensus({
        "deskew": [from_text(text, locale=row["locale"])],
        "clahe": [from_text(text, locale=row["locale"])],
    })
    result = reconcile(evidence, [read_model(json.dumps(row["output"]))])
    report = render_report(
        evidence, result.validation, result.statuses, [],
        model_info={"used": True},
    )
    return evidence, result, report


def test_invoice_number_parser_requires_the_expected_country_year_and_five_digits():
    assert invoice_number_from_text("FACTURE FR-2025-00406") == "FR-2025-00406"
    assert invoice_number_from_text("INVOICE GB-2025-00110") == "GB-2025-00110"
    assert invoice_number_from_text("INVOICE US-2023-00435") == "US-2023-00435"
    assert invoice_number_from_text("FACTURE FR-2025-0406") is None
    assert invoice_number_from_text("FACTURE FR-2025-000406") is None


def test_consensus_requires_two_independent_reads_for_invoice_number_and_siret():
    text = (
        "FACTURE FR-2025-00406\nSupplier: Services Horizon SAS\n"
        "SIRET : 40023297980955\nCurrency: EUR\n"
        "Description | Quantity | Unit price | Line total\nService | 1 | 10 | 10\n"
        "Subtotal: 10\nInvoice total: 10"
    )
    single_family = build_consensus({"deskew": [from_text(text, locale="fr-FR")]})
    assert single_family.invoice_number is None
    assert single_family.corroborated_identifiers == []

    paired = build_consensus({
        "deskew": [from_text(text, locale="fr-FR")],
        "clahe": [from_text(text, locale="fr-FR")],
    })
    assert paired.invoice_number == "FR-2025-00406"
    assert paired.corroborated_identifiers == [("siret", "40023297980955")]
    validation = document_validation(paired)
    assert any(
        flag.code == "identifier_siret"
        and flag.reason == "SIRET : la clé de contrôle ne correspond pas. À vérifier si ce numéro est réel."
        for flag in validation.flags
    )
    assert not any("SIRET" in message for message in validation.not_verified)


@pytest.mark.parametrize(
    "invoice_id,number_readable",
    [
        ("invoice-0406", True), ("invoice-0406", False),
        ("invoice-0110", True), ("invoice-0110", False),
        ("invoice-0435", True), ("invoice-0435", False),
    ],
)
def test_country_report_verdict_depends_on_json_fields_not_invoice_number(invoice_id, number_readable):
    evidence, result, report = _reference_report(
        invoice_id, invoice_number_readable=number_readable,
    )
    expected_count = len(result.statuses)
    assert all(status in {"valide", "récupéré (corroboré par OCR)"} for status in result.statuses.values())
    assert report.startswith(f"**✅ Facture cohérente : {expected_count} champs vérifiés sur {expected_count}**")
    if number_readable:
        assert evidence.invoice_number is not None
        assert "### Points non vérifiés" not in report
        assert "Numéro de facture non lu." not in report
    else:
        assert evidence.invoice_number is None
        assert "### Analyse de la facture\n" in report
        assert "Numéro de facture non lu." in report
        if invoice_id == "invoice-0406":
            assert "### Informations" in report
            assert "SIRET : la clé de contrôle ne correspond pas." in report
        else:
            assert "### Informations" not in report
        assert "### Points non vérifiés" in report
    for forbidden in ("None", "non ancré", "ancrage", "description_non_verifiee"):
        assert forbidden not in report


@pytest.mark.parametrize(
    "number_readable,siret_readable,information,point",
    [
        (True, True, True, False),
        (False, True, True, True),
        (True, False, False, True),
        (False, False, False, True),
    ],
)
def test_fr_siret_readability_is_reported_separately_from_verified_json(
    number_readable, siret_readable, information, point,
):
    evidence, result, report = _reference_report(
        "invoice-0406",
        invoice_number_readable=number_readable,
        siret_readable=siret_readable,
    )
    if number_readable:
        assert evidence.invoice_number is not None
    else:
        assert evidence.invoice_number is None
    assert all(status in {"valide", "récupéré (corroboré par OCR)"} for status in result.statuses.values())
    assert report.startswith("**✅ Facture cohérente : 20 champs vérifiés sur 20**")
    if information:
        assert "### Informations" in report
        assert "SIRET : la clé de contrôle ne correspond pas. À vérifier si ce numéro est réel." in report
        assert "SIRET : numéro illisible avec certitude, contrôle non effectué." not in report
    else:
        assert "### Informations" not in report
    if point:
        assert "### Points non vérifiés" in report
    else:
        assert "### Points non vérifiés" not in report
    if not siret_readable:
        assert "SIRET : numéro illisible avec certitude, contrôle non effectué." in report
    for forbidden in ("None", "non ancré", "ancrage", "description_non_verifiee"):
        assert forbidden not in report


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
    assert "0 champs vérifiés" in report
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
    assert main.preview_ui(None) == (
        None, None, "", "", "", {"flags": [], "passed": [], "not_verified": []}, None,
    )
    assert any(component["type"] == "downloadbutton" for component in components)
    assert any(
        "copy" in component["props"].get("buttons", [])
        for component in components if component["type"] == "markdown"
    )


def test_app_ocr_failure_is_reported_without_loading_model(caplog):
    from app import main
    with patch.object(main, "document_text", side_effect=RuntimeError("OCR absent")), \
            patch.object(main, "predict_invoice", side_effect=AssertionError("No evidence, no GPU")):
        outputs = list(main.run_extraction_ui(str(main.EXAMPLES[0][1])))
    payload, report, technical, advanced, validation, download = outputs[-1]
    assert "Preuves OCR indisponibles" in report
    assert "Pydantic" not in report
    assert "indisponible pour ce document" in technical
    assert "Empreinte de configuration OCR" in advanced
    assert download.endswith(".json")
    assert not validation["flags"]
    assert all(payload[key] is None for key in ("supplier_name", "invoice_date", "total_ht", "total_ttc"))


def test_ui_technical_details_are_compact_and_json_download_matches_result():
    from app import main
    evidence = EvidenceText(from_text(UK_MULTI_RATE, locale="en-GB"))
    with patch.object(main, "document_text", return_value=evidence), patch.object(
        main, "predict_invoice",
        side_effect=main.gr.Error("ZeroGPU quota exceeded; try again in 36 min"),
    ):
        payload, report, technical, advanced, validation, download = list(
            main.run_extraction_ui(str(main.EXAMPLES[1][1]))
        )[-1]
    try:
        for heading in ("### Lecture du document", "### Modèle", "### Champs", "### Version"):
            assert technical.count(heading) == 1
        assert "non utilisé — quota ZeroGPU indisponible" in technical
        assert "passes demandées : 1 ; réussies : 0" in technical
        assert "| Champ | Valeur | Statut | Source |" in technical
        assert "| Ligne 1 — Quantité | 7.8 | vérifié | OCR seul |" in technical
        assert len(re.search(r"[a-f0-9]{64}", advanced).group()) == 64
        assert json.loads(Path(download).read_text()) == payload
        for forbidden in ("None", "non ancré", "description_non_verifiee", "ancrage", "Pydantic"):
            assert forbidden not in report + technical
        assert not validation["flags"]
    finally:
        Path(download).unlink(missing_ok=True)


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
    assert "Une passe d’extraction n’a pas abouti (indisponibilité technique)" in result["report"]
    assert "La seconde passe a échoué" not in result["report"]
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
