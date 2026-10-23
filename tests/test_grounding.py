import copy
import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import EvidenceText, from_text, from_tsv, parse_number
from invoice_grounding import (
    NULL, RECOVERED, VALID, inference_error, read_model, reconcile, run_grounded_inference,
)


def replay_cases():
    cases = json.loads((ROOT / "tests/fixtures/invoice_model_errors.json").read_text())
    rows = {row["id"]: row for row in map(json.loads, (ROOT / "data/test.jsonl").read_text().splitlines())}
    for case in cases:
        row = rows[case["id"]]
        raw = copy.deepcopy(row["output"])
        for path, value in case["mutations"].items():
            keys = path.split(".")
            target = raw
            for key in keys[:-1]:
                target = target[int(key)] if isinstance(target, list) else target[key]
            target[keys[-1]] = value
        yield case, row, json.dumps(raw, ensure_ascii=False)


@pytest.mark.parametrize("raw,locale,expected", [
    ("3 627,96 €", "fr-FR", "3627.96"),
    ("3\u00a0627,96 €", "fr-FR", "3627.96"),
    ("1\u202f143,35", "fr-FR", "1143.35"),
    ("7,222.74", "en-US", "7222.74"),
    ("GBP 5,912.17", "en-GB", "5912.17"),
    ("7,2", "fr-FR", "7.2"), ("5,1", "fr-FR", "5.1"),
    ("7,8", "fr-FR", "7.8"), ("757,97", "fr-FR", "757.97"),
    ("1,234", "en-US", "1234"), ("1,234", "fr-FR", "1.234"),
    ("1.234,56", "en-US", "1234.56"),
    ("-12,50", "fr-FR", "-12.50"), ("($12.50)", "en-US", "-12.50"),
])
def test_locale_decimal_parser(raw, locale, expected):
    assert parse_number(raw, locale) == Decimal(expected)


def test_document_preserves_raw_thin_space_number_and_supplier_typography_provenance():
    text = (
        "Currency: EUR\nFournisseur : Acme  LTD.\n"
        "Description | Quantité | PU net HT | Montant HT\n"
        "Service | 4,2 | 863,80 | 3\u202f627,96 €\n"
        "Sous-total HT : 3\u202f627,96 €"
    )
    evidence = from_text(text)
    assert evidence.rows[0].cells["total_ht"][0].raw == "3\u202f627,96 €"
    assert evidence.rows[0].cells["total_ht"][0].value == Decimal("3627.96")
    assert evidence.supplier == "Acme Ltd"
    assert evidence.supplier_raw == "Acme  LTD."


@pytest.mark.parametrize("raw,locale", [
    ("1,234", None), ("7 2", "fr-FR"), ("7_2", "en-US"),
    ("1.23,56", "en-US"), ("7,22.74", "en-GB"), ("NaN", None),
    ("Infinity", None), (True, None), ("5,91217", "en-GB"),
])
def test_numbers_are_not_guessed(raw, locale):
    with pytest.raises(ValueError):
        parse_number(raw, locale)


@pytest.mark.parametrize("case,row,raw", list(replay_cases()), ids=lambda value: (
    value["invoice"] if isinstance(value, dict) and "invoice" in value else None
))
def test_replay_recovers_all_documented_values_without_false_nulls(case, row, raw):
    evidence = from_text(row["input"], locale=row["locale"])
    result = reconcile(evidence, [read_model(raw)])
    assert result.payload == row["output"]
    assert result.failed_rows == []
    assert RECOVERED in result.statuses.values()
    assert "corrigé" not in "\n".join(result.warnings)
    document_values = {number.value for number in evidence.numbers}
    for item in result.payload["line_items"]:
        for key in ("quantity", "unit_price", "total_ht"):
            assert Decimal(str(item[key])) in document_values
    for key in ("total_ht", "total_ttc"):
        assert Decimal(str(result.payload[key])) in document_values
    if row["locale"] == "fr-FR":
        assert result.payload["line_items"][1]["unit_price"] == 95.07
        assert result.payload["line_items"][1]["total_ht"] == 684.50
        assert result.payload["line_items"][0]["description"] == "Clavier sans fil"
        assert result.payload["line_items"][3]["description"] == "Création de logo"
    if row["locale"] == "en-GB":
        assert {tax.rate.value for tax in evidence.taxes} == {Decimal(0), Decimal(20)}
        assert sum(tax.amount.value for tax in evidence.taxes) == Decimal("1216.06")


def simple_evidence(quantity="", price="10", amount="20", total="20", extra=""):
    return from_text(
        "Supplier: Acme Ltd\nInvoice date: 2026-10-22\nCurrency: GBP\n"
        "Description | Quantity | Net unit price | Net amount\n"
        f"Service | {quantity} | {price} | {amount}\n"
        f"Subtotal (net): GBP {total}\nVAT (0%): GBP 0.00\nInvoice total: GBP {total}\n{extra}",
        locale="en-GB",
    )


def simple_model(quantity="2", price="10", total="20"):
    return read_model(json.dumps({
        "supplier_name": "Acme Ltd", "invoice_date": "2026-10-22",
        "total_ht": total, "total_ttc": total,
        "line_items": [{"description": "Service", "quantity": quantity, "unit_price": price, "total_ht": total}],
    }))


def test_two_anchored_fields_survive_when_quantity_absent_from_document():
    evidence = simple_evidence()
    result = reconcile(evidence, [simple_model(quantity="142")])
    assert result.payload["line_items"][0] == {
        "description": "Service", "quantity": None, "unit_price": 10, "total_ht": 20,
    }
    assert result.payload["total_ht"] == 20
    assert result.statuses["line_items.0.quantity"] == NULL
    assert result.statuses["line_items.0.unit_price"] == VALID


def test_no_division_can_create_a_quantity_absent_from_document():
    result = reconcile(simple_evidence(amount="30", total="30"), [simple_model(quantity="3", total="30")])
    assert result.payload["line_items"][0]["quantity"] is None
    assert result.payload["line_items"][0]["unit_price"] == 10
    assert result.payload["line_items"][0]["total_ht"] == 30


@pytest.mark.parametrize("missing", ["quantity", "unit_price", "total_ht"])
def test_any_two_of_three_anchored_fields_survive_without_creating_the_third(missing):
    fields = {"quantity": "2", "price": "10", "amount": "20"}
    fields[{"quantity": "quantity", "unit_price": "price", "total_ht": "amount"}[missing]] = ""
    evidence = simple_evidence(**fields)
    result = reconcile(evidence, [simple_model()])
    item = result.payload["line_items"][0]
    assert item[missing] is None
    for key, value in {"quantity": 2, "unit_price": 10, "total_ht": 20}.items():
        if key != missing:
            assert item[key] == value
    assert result.payload["total_ht"] == 20


@pytest.mark.parametrize("description", ["", "ERROR", "INVALID"])
def test_unreadable_description_does_not_erase_corroborated_numbers(description):
    evidence = from_text(simple_evidence(quantity="2").text.replace("Service |", f"{description} |"))
    result = reconcile(evidence, [simple_model()])
    assert result.payload["line_items"][0]["description"] is None
    assert result.payload["line_items"][0]["quantity"] == 2
    assert result.payload["line_items"][0]["unit_price"] == 10
    assert result.payload["line_items"][0]["total_ht"] == 20


def test_single_pass_mode_keeps_grounding_without_requesting_a_correction():
    evidence = simple_evidence()
    backend = MagicMock(return_value={
        "raw": json.dumps(simple_model(quantity="142"), default=str), "finish_reason": "stop",
    })
    result = run_grounded_inference(
        evidence, {"id": "test", "input": evidence.text}, [], backend, allow_correction=False,
    )
    backend.assert_called_once()
    assert result["payload"]["line_items"][0]["quantity"] is None
    assert result["payload"]["line_items"][0]["unit_price"] == 10
    assert not result["correction_attempted"]


def test_multiple_solutions_are_not_repaired_even_if_model_prefers_one():
    evidence = simple_evidence(quantity="1 / 2", price="10 / 20")
    result = reconcile(evidence, [simple_model(quantity="3")])
    assert result.payload["line_items"][0]["quantity"] is None
    assert result.payload["line_items"][0]["unit_price"] is None
    assert result.payload["line_items"][0]["total_ht"] == 20
    assert RECOVERED not in {result.statuses["line_items.0.quantity"], result.statuses["line_items.0.unit_price"]}


def test_unique_row_and_fields_are_not_lost_because_another_row_is_ambiguous():
    evidence = from_text(
        "Currency: GBP\nDescription | Quantity | Net unit price | Net amount\n"
        "First | 1 / 2 | 10 | 10 / 20\nSecond | 1 / 2 | 10 / 20 | 20\n"
        "Subtotal (net): GBP 40\nVAT (0%): GBP 0\nInvoice total: GBP 40"
    )
    model = read_model(
        '{"total_ht":40,"total_ttc":40,"line_items":['
        '{"description":"First","quantity":3,"unit_price":10,"total_ht":20},'
        '{"description":"Second","quantity":3,"unit_price":10,"total_ht":20}]}'
    )
    result = reconcile(evidence, [model])
    assert result.payload["line_items"][0]["quantity"] == 2
    assert result.payload["line_items"][0]["unit_price"] == 10
    assert result.payload["line_items"][0]["total_ht"] == 20
    assert result.statuses["line_items.0.quantity"] == RECOVERED
    assert result.payload["line_items"][1]["quantity"] is None
    assert result.payload["line_items"][1]["unit_price"] is None
    assert result.payload["line_items"][1]["total_ht"] == 20


def test_quota_failure_uses_ocr_only_and_keeps_error_type_and_trace(caplog):
    evidence = simple_evidence()
    backend = MagicMock(side_effect=[
        {"raw": json.dumps(simple_model(), default=str), "finish_reason": "stop"},
        RuntimeError("ZeroGPU quota exceeded; try again in 00:12:34"),
    ])
    result = run_grounded_inference(evidence, {"id": "test", "input": evidence.text}, [], backend)
    assert backend.call_count == 2
    assert result["payload"]["line_items"][0]["unit_price"] == 10
    assert result["payload"]["line_items"][0]["total_ht"] == 20
    assert result["payload"]["line_items"][0]["quantity"] is None
    assert "Quota ZeroGPU épuisé, réessayer dans 00:12:34" in "\n".join(result["warnings"])
    assert "corrigé" not in "\n".join(result["warnings"])
    assert "RuntimeError" in caplog.text
    assert "Traceback" in caplog.text
    focused_row, focused_messages = backend.call_args.args
    assert focused_row["max_new_tokens"] == 512
    assert "Supplier:" not in focused_messages[-1]["content"]
    assert "Invoice date" not in focused_messages[-1]["content"]
    assert "Service" in focused_messages[-1]["content"]


def test_targeted_second_pass_can_recover_an_anchored_field_but_not_invent_a_quantity():
    evidence = simple_evidence()
    initial = simple_model(quantity="142", price="11")
    corrected = {
        "line_items": [
            {"index": 0, "description": "Service", "quantity": 2, "unit_price": 10, "total_ht": 20},
        ],
    }
    backend = MagicMock(side_effect=[
        {"raw": json.dumps(initial, default=str), "finish_reason": "stop"},
        {"raw": json.dumps(corrected), "finish_reason": "stop"},
    ])
    result = run_grounded_inference(evidence, {"id": "test", "input": evidence.text}, [], backend)
    assert backend.call_count == 2
    assert result["payload"]["line_items"][0]["quantity"] is None
    assert result["payload"]["line_items"][0]["unit_price"] == 10
    assert result["payload"]["line_items"][0]["total_ht"] == 20
    assert result["field_statuses"]["line_items.0.unit_price"] == RECOVERED
    assert len(result["provenance"]["line_items.0.unit_price"][0]["model_proposals"]) == 2


def test_initial_gpu_failure_can_recover_a_complete_invoice_without_model(caplog):
    evidence = simple_evidence(quantity="2")
    backend = MagicMock(side_effect=RuntimeError("GPU quota exceeded; retry after 40 s"))
    result = run_grounded_inference(evidence, {"id": "test", "input": evidence.text}, [], backend)
    backend.assert_called_once()
    assert result["payload"]["line_items"][0]["quantity"] == 2
    assert result["payload"]["total_ht"] == 20
    assert "Repli OCR sans GPU" in "\n".join(result["warnings"])


def test_gradio_quota_exception_is_handled_by_the_application_without_losing_two_anchors(caplog):
    from app import main
    evidence = simple_evidence()
    warnings = []
    with patch.object(main, "predict_invoice", side_effect=[
        {"raw": json.dumps(simple_model(quantity="142"), default=str), "finish_reason": "stop"},
        main.gr.Error("ZeroGPU quota exceeded; try again in 00:01:00"),
    ]) as backend:
        result = main.extract_invoice(EvidenceText(evidence), warnings=warnings)
    assert backend.call_count == 2
    assert result["line_items"][0]["quantity"] is None
    assert result["line_items"][0]["unit_price"] == 10
    assert result["line_items"][0]["total_ht"] == 20
    assert any("Quota ZeroGPU épuisé, réessayer dans 00:01:00." in warning for warning in warnings)
    assert "Traceback" in caplog.text
    assert "corrigé" not in "\n".join(warnings)


def test_empty_value_error_has_readable_ui_message():
    message = inference_error(ValueError("ValueError"))
    assert "message non communiqué" in message
    assert "ValueError:" in message


def test_tsv_keeps_positions_raw_numbers_and_table_rows():
    words = [
        ("Description", 0, 0), ("Quantity", 200, 0), ("Net", 300, 0), ("unit", 330, 0),
        ("price", 360, 0), ("Net", 450, 0), ("amount", 480, 0),
        ("Service", 0, 20), ("7,2", 200, 20), ("95,07", 300, 20), ("684,50", 450, 20),
    ]
    data = {"text": [word[0] for word in words], "left": [word[1] for word in words],
            "top": [word[2] for word in words], "width": [25] * len(words),
            "height": [10] * len(words), "conf": [95] * len(words)}
    evidence = from_tsv(data, locale="fr-FR")
    assert len(evidence.rows) == 1
    assert evidence.rows[0].cells["quantity"][0].raw == "7,2"
    assert evidence.rows[0].cells["quantity"][0].value == Decimal("7.2")
    assert evidence.rows[0].cells["quantity"][0].box[0] == 200
    assert evidence.tokens[0].confidence == 95


def test_image_ocr_is_x2_grayscale_tsv():
    from app import main
    from PIL import Image
    data = {"text": ["FACTURE"], "left": [0], "top": [0], "width": [40], "height": [10], "conf": [99]}
    with patch.object(main.pytesseract, "image_to_data", return_value=data) as ocr:
        text = main.ocr_image(Image.new("RGB", (20, 30)))
    images = [call.args[0] for call in ocr.call_args_list]
    assert (40, 60) in {image.size for image in images}
    assert (60, 90) in {image.size for image in images}
    assert all(image.mode == "L" for image in images)
    assert ocr.call_args.kwargs["lang"] == "fra+eng"
    assert ocr.call_args.kwargs["output_type"] == main.pytesseract.Output.DICT
    assert isinstance(text, EvidenceText)


def test_grounded_app_replay_avoids_a_second_gpu_request_when_ocr_solves_everything():
    from app import main
    for _, row, raw in replay_cases():
        evidence = from_text(row["input"], locale=row["locale"])
        warnings = []
        with patch.object(main, "predict_invoice", return_value={"raw": raw, "finish_reason": "stop"}) as backend:
            result = main.extract_invoice(EvidenceText(evidence), warnings=warnings)
        backend.assert_called_once()
        assert result == row["output"]
        assert "récupéré (corroboré par OCR)" in "\n".join(warnings)


def test_duration_is_dynamic_bounded_and_smaller_for_corrections():
    from app import main
    small = [{"role": "user", "content": "Service 7,2 95,07 684,50"}]
    large = [{"role": "user", "content": "x" * 100000}]
    assert 8 <= main.gpu_duration({"correction": True}, small) <= 30
    assert main.gpu_duration({"correction": True}, small) < main.gpu_duration({}, large)
    assert main.gpu_duration({"correction": True}, large) == 30
    assert main.gpu_duration({}, large) == 60


def test_deployed_hash_footer_has_no_model_load():
    from app import main
    revision = "a" * 40
    with patch.dict(main.os.environ, {"SPACE_COMMIT_SHA": revision}), patch.object(main, "get_extractor") as backend:
        demo = main.build_app()
    backend.assert_not_called()
    footers = [component["props"]["value"] for component in demo.config["components"] if component["type"] == "html"]
    assert any(revision[:7] in footer for footer in footers)


def test_real_tesseract_evidence_with_simulated_wrong_model_outputs_on_all_three_images():
    import shutil
    import pytesseract
    from app import main
    if not shutil.which("tesseract") or not {"fra", "eng"} <= set(pytesseract.get_languages()):
        pytest.skip("Real OCR needs Tesseract fra+eng")
    images = {"invoice-0406": "france.png", "invoice-0110": "uk.png", "invoice-0435": "us.png"}
    for case, row, raw in replay_cases():
        text = main.document_text(str(ROOT / "app/examples" / images[case["id"]]))
        result = reconcile(text.evidence, [read_model(raw)])
        assert len(text.evidence.rows) == len(row["output"]["line_items"])
        assert result.payload == row["output"]
        assert result.failed_rows == []
        assert all(
            "%" not in n.raw for row in text.evidence.rows
            for key in ("quantity", "unit_price", "total_ht") for n in row.cells[key]
        )


def test_metrics_count_false_nulls_and_wrong_values_instead_of_only_valid_json():
    from evaluate_grounding import evaluate
    expected = {
        "supplier_name": "Acme Ltd", "invoice_date": "2026-10-22",
        "total_ht": 20, "total_ttc": 20,
        "line_items": [{"description": "Service", "quantity": 3, "unit_price": 10, "total_ht": 20}],
    }
    records = [
        {"id": "null", "ocr_text": simple_evidence().text, "locale": "en-GB",
         "model_responses": [json.dumps(simple_model(), default=str)], "expected": expected},
        {"id": "wrong", "ocr_text": simple_evidence(quantity="2").text, "locale": "en-GB",
         "model_responses": [json.dumps(simple_model(), default=str)], "expected": expected},
    ]
    metrics = evaluate(records)["fields"]["line_items.quantity"]
    assert metrics["count"] == 2
    assert metrics["exact_accuracy"] == 0
    assert metrics["false_null_rate"] == .5
    assert metrics["wrong_value_retained_rate"] == .5


def test_tax_failure_does_not_erase_lines_or_subtotal_supported_by_two_proofs():
    evidence = from_text(
        "Supplier: Acme Ltd\nInvoice date: 2026-10-22\nCurrency: GBP\n"
        "Description | Quantity | Net unit price | Net amount\nService | 2 | 10 | 20\n"
        "Subtotal (net): GBP 20\nVAT (20%): GBP 4\nInvoice total: GBP 99"
    )
    model = simple_model()
    model["total_ttc"] = Decimal(99)
    result = reconcile(evidence, [model])
    assert result.payload["line_items"][0]["quantity"] == 2
    assert result.payload["line_items"][0]["unit_price"] == 10
    assert result.payload["line_items"][0]["total_ht"] == 20
    assert result.payload["total_ht"] == 20
    assert result.payload["total_ttc"] is None
    assert any("TVA" in warning and "écart" in warning for warning in result.warnings)


def test_native_pdf_text_keeps_positions_without_ocr():
    import tempfile
    import pymupdf
    from app import main
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
        path = Path(directory) / "proof.pdf"
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((40, 40), simple_evidence(quantity="2").text, fontsize=9)
            pdf.save(path)
        with patch.object(main, "ocr_image", side_effect=AssertionError("Native PDF must not use OCR")):
            text = main.document_text(str(path))
        assert isinstance(text, EvidenceText)
        assert text.evidence.tokens
        assert text.evidence.tokens[0].page == 0
        result = reconcile(text.evidence, [simple_model()])
        assert result.payload["line_items"][0]["quantity"] == 2
        assert result.payload["total_ht"] == 20


def test_ambiguous_supplier_and_invalid_date_do_not_become_successful_metadata():
    evidence = from_text(
        simple_evidence(quantity="2").text.replace("Acme Ltd", "Acme LId").replace("2026-10-22", "2026-02-31")
    )
    model = simple_model()
    model["supplier_name"] = "Acme LId"
    model["invoice_date"] = "2026-02-31"
    result = reconcile(evidence, [model])
    assert result.payload["supplier_name"] is None
    assert result.payload["invoice_date"] is None
    assert result.payload["total_ht"] == 20
    assert any("suffixe OCR ambigu" in warning for warning in result.warnings)


def test_accounting_negatives_do_not_change_tax_percent_sign_or_lose_credit_sign():
    evidence = from_text(
        "Currency: USD\nDescription | Quantity | Net unit price | Net amount\n"
        "Credit | -2 | $10.00 | ($20.00)\n"
        "Subtotal (net): ($20.00)\nSales tax (0%): $0.00\nInvoice total: ($20.00)"
    )
    assert evidence.rows[0].cells["total_ht"][0].value == Decimal("-20")
    assert evidence.taxes[0].rate.value == 0
    assert evidence.totals["total_ht"][0].value == -20
    source = read_model('{"total_ht":-20,"total_ttc":-20,"line_items":[{"description":"Credit","quantity":-2,"unit_price":10,"total_ht":-20}]}')
    result = reconcile(evidence, [source])
    assert result.payload["line_items"][0]["total_ht"] == -20
    assert result.payload["total_ht"] == -20


def test_sixty_reference_texts_keep_only_documented_values():
    from evaluate_grounding import flatten
    for line in (ROOT / "data/test.jsonl").read_text().splitlines():
        row = json.loads(line)
        evidence = from_text(row["input"], locale=row["locale"])
        result = reconcile(evidence, [read_model(json.dumps(row["output"]))])
        expected = flatten(row["output"])
        for path, value in flatten(result.payload).items():
            if value is not None:
                assert value == expected[path], (row["id"], path)
                if isinstance(value, (int, float)):
                    assert Decimal(str(value)) in {n.value for n in evidence.numbers}
        if evidence.rows:
            assert result.payload == row["output"], row["id"]
            assert not result.failed_rows, row["id"]
            assert not result.global_failure, row["id"]
