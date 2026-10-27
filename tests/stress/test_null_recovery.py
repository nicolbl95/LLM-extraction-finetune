import copy
import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests/stress")]

from cases import error_profile, REFERENCES
from experiment_ocr import split_cases
from invoice_evidence import from_text
from invoice_grounding import NULL, UNVERIFIED, read_model, reconcile, run_grounded_inference
from invoice_validation import render_report
from ocr_consensus import build_consensus
from ocr_preprocessing import estimate_skew


TEXT = (
    "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: EUR\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Clavier sans fil | 5,1 | 77,66 | 396,07\n"
    "Subtotal: 396,07\nVAT (0%): 0\nInvoice total: 396,07"
)

ACCENT_TEXT = (
    "Supplier: Joanna Binet\nInvoice date: 2019-01-29\nCurrency: EUR\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Facile {accent} jouer accordéon | 3 | 5.00 | 15.00\n"
    "Subtotal: 15.00\nVAT (0%): 0.00\nInvoice total: 15.00"
)


def _reading(quantity, price, total, description="Logo design"):
    return from_text(
        "Supplier: Oak Studio Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
        "Description | Quantity | Net unit price | Net amount\n"
        f"{description} | {quantity} | {price} | {total}\n"
        f"Subtotal: {total}\nVAT (0%): 0.00\nInvoice total: {total}",
        locale="en-GB",
    )


def _uk_reading(quantity, *, total="5912.17"):
    return _reading(quantity, "757.97", total)


def test_two_segmentations_of_one_preprocessing_family_are_not_independent_votes():
    result = build_consensus({"gray": [from_text(TEXT), from_text(TEXT)]}, confidence=40)
    assert result.rows[0].cells["quantity"] == []
    assert result.supplier is None
    assert not result.totals["total_ht"]


def test_two_preprocessing_families_recover_without_inventing_a_number():
    result = build_consensus({"gray": [from_text(TEXT)], "contrast": [from_text(TEXT)]}, confidence=40)
    model = read_model('{"line_items":[{"description":"Clavier sans il","quantity":54,"unit_price":77.68,"total_ht":4210.32}]}')
    final = reconcile(result, [model])
    assert final.payload["line_items"][0] == {
        "description": "Clavier sans fil", "quantity": 5.1, "unit_price": 77.66, "total_ht": 396.07,
    }
    assert not final.validation.flags
    assert final.payload["total_ht"] == 396.07


def test_single_family_decimal_is_not_recovered_even_when_arithmetic_confirms_it():
    result = build_consensus({
        "deskew": [_uk_reading("78")],
        "clahe": [_uk_reading("78")],
        "crop": [_uk_reading("78")],
        "denoise": [_uk_reading("78")],
        "scale3": [_uk_reading("7.8")],
    }, confidence=40)
    line = result.rows[0]
    assert line.cells["quantity"] == []
    assert line.cells["unit_price"] == []
    assert [number.value for number in line.cells["total_ht"]] == [Decimal("5912.17")]

    model = read_model(json.dumps({
        "line_items": [{
            "description": "Logo design", "quantity": 7.8,
            "unit_price": 757.97, "total_ht": 5912.17,
        }],
    }))
    final = reconcile(result, [model])
    item = final.payload["line_items"][0]
    assert item["description"] == "Logo design"
    assert item["quantity"] is None
    assert item["unit_price"] is None
    assert item["total_ht"] == 5912.17
    assert final.statuses["line_items.0.quantity"] in {NULL, UNVERIFIED}
    assert final.statuses["line_items.0.unit_price"] in {NULL, UNVERIFIED}
    assert 7.8 not in item.values()


def test_two_family_decimal_is_recovered_with_corroborated_price_and_total():
    result = build_consensus({
        "deskew": [_uk_reading("7.8")],
        "clahe": [_uk_reading("7.8")],
    }, confidence=40)
    line = result.rows[0]
    assert [number.value for number in line.cells["quantity"]] == [Decimal("7.8")]
    assert [number.value for number in line.cells["unit_price"]] == [Decimal("757.97")]
    assert [number.value for number in line.cells["total_ht"]] == [Decimal("5912.17")]

    model = read_model(json.dumps({
        "line_items": [{
            "description": "Logo design", "quantity": 7.8,
            "unit_price": 757.97, "total_ht": 5912.17,
        }],
    }))
    item = reconcile(result, [model]).payload["line_items"][0]
    assert item["quantity"] == 7.8
    assert item["unit_price"] == 757.97
    assert item["total_ht"] == 5912.17


def test_multiple_conflicting_quantity_candidates_are_not_recovered():
    result = build_consensus({
        "deskew": [_uk_reading("78")],
        "clahe": [_uk_reading("78")],
        "crop": [_uk_reading("78")],
        "denoise": [_uk_reading("78")],
        "scale3": [_uk_reading("7.8")],
        "extra": [_uk_reading("8.7")],
    }, confidence=40)
    line = result.rows[0]
    assert line.cells["quantity"] == []
    assert line.cells["unit_price"] == []

    model = read_model(json.dumps({
        "line_items": [{
            "description": "Logo design", "quantity": 7.8,
            "unit_price": 757.97, "total_ht": 5912.17,
        }],
    }))
    item = reconcile(result, [model]).payload["line_items"][0]
    assert item["quantity"] is None
    assert item["unit_price"] is None
    assert 7.8 not in item.values()
    assert 8.7 not in item.values()


def test_multiple_quantities_within_arithmetic_tolerance_are_not_recovered():
    result = build_consensus({
        "deskew": [_reading("10.00", "100", "1000")],
        "clahe": [_reading("10.00", "100", "1000")],
        "crop": [_reading("10.0001", "100", "1000")],
        "denoise": [_reading("10.0001", "100", "1000")],
    }, confidence=40)
    line = result.rows[0]
    assert {number.value for number in line.cells["quantity"]} == {
        Decimal("10.00"), Decimal("10.0001"),
    }

    model = read_model(json.dumps({
        "line_items": [{
            "description": "Logo design", "quantity": 10.0,
            "unit_price": 100.0, "total_ht": 1000.0,
        }],
    }))
    item = reconcile(result, [model]).payload["line_items"][0]
    assert item["quantity"] is None
    assert item["unit_price"] == 100.0
    assert item["total_ht"] == 1000.0


def test_wrong_mutually_consistent_price_and_total_do_not_justify_single_family_quantity():
    result = build_consensus({
        "deskew": [_reading("78", "100", "1000")],
        "clahe": [_reading("78", "100", "1000")],
        "scale3": [_reading("10", "100", "1000")],
    }, confidence=40)
    line = result.rows[0]
    assert line.cells["quantity"] == []
    assert line.cells["unit_price"] == []

    model = read_model(json.dumps({
        "line_items": [{
            "description": "Logo design", "quantity": 78,
            "unit_price": 12.82, "total_ht": 1000.0,
        }],
    }))
    item = reconcile(result, [model]).payload["line_items"][0]
    assert item["quantity"] is None
    assert item["unit_price"] is None


def test_single_family_decimal_without_arithmetic_support_is_not_exported():
    result = build_consensus({
        "deskew": [_uk_reading("78", total="6000.00")],
        "clahe": [_uk_reading("78", total="6000.00")],
        "scale3": [_uk_reading("7.8", total="6000.00")],
    }, confidence=40)
    line = result.rows[0]
    assert line.cells["quantity"] == []
    assert line.cells["unit_price"] == []

    model = read_model(json.dumps({
        "line_items": [{
            "description": "Logo design", "quantity": 7.8,
            "unit_price": 757.97, "total_ht": 6000.0,
        }],
    }))
    item = reconcile(result, [model]).payload["line_items"][0]
    assert item["quantity"] is None
    assert item["unit_price"] is None


def test_majority_numeric_reading_is_never_exported_without_a_decimal_candidate():
    result = build_consensus({
        "deskew": [_uk_reading("78")],
        "clahe": [_uk_reading("78")],
    }, confidence=40)
    line = result.rows[0]
    assert line.cells["quantity"] == []
    assert line.cells["unit_price"] == []

    model = read_model(json.dumps({
        "line_items": [{
            "description": "Logo design", "quantity": 7.8,
            "unit_price": 757.97, "total_ht": 5912.17,
        }],
    }))
    item = reconcile(result, [model]).payload["line_items"][0]
    assert item["quantity"] is None
    assert item["unit_price"] is None
    assert 78 not in item.values()


def test_consensus_preserves_accented_description_when_folded_readings_agree():
    plain = from_text(ACCENT_TEXT.format(accent="a"), locale="fr-FR")
    accented = from_text(ACCENT_TEXT.format(accent="à"), locale="fr-FR")
    for readings in (
        {"deskew": [plain], "clahe": [accented]},
        {"deskew": [accented], "clahe": [plain]},
    ):
        result = build_consensus(readings, confidence=40)
        assert result.rows[0].description == "Facile à jouer accordéon"


def test_consensus_keeps_plain_description_when_no_read_has_an_accent():
    plain = from_text(ACCENT_TEXT.format(accent="a"), locale="fr-FR")
    result = build_consensus({"deskew": [plain], "clahe": [plain]}, confidence=40)
    assert result.rows[0].description == "Facile a jouer accordéon"


def test_consensus_does_not_merge_different_description_words_for_an_accent():
    accented = from_text(ACCENT_TEXT.format(accent="à"), locale="fr-FR")
    different_word = from_text(ACCENT_TEXT.format(accent="e"), locale="fr-FR")
    result = build_consensus({"deskew": [accented], "clahe": [different_word]}, confidence=40)
    assert result.rows[0].description == ""
    assert not result.rows[0].description_confident


def test_repeated_wrong_description_is_vetoed_by_a_contradictory_preprocessing_family():
    wrong = TEXT.replace("Clavier", "Ciavier")
    evidence = build_consensus({
        "gray": [from_text(wrong)], "contrast": [from_text(wrong)], "denoise": [from_text(TEXT)],
    }, confidence=40)
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    assert result.payload["line_items"][0]["description"] is None
    assert result.payload["line_items"][0]["quantity"] == 5.1
    assert not result.validation.flags


def test_coherent_but_unsupported_numbers_are_refused():
    evidence = build_consensus({"gray": [from_text(TEXT)], "contrast": [from_text(TEXT)]}, confidence=40)
    candidate = read_model(
        '{"line_items":[{"description":"Clavier sans fil","quantity":15,"unit_price":29.68,"total_ht":445.20}]}'
    )
    result = reconcile(evidence, [candidate])
    assert result.payload["line_items"][0]["quantity"] == 5.1
    assert result.payload["line_items"][0]["unit_price"] == 77.66
    assert result.payload["line_items"][0]["total_ht"] == 396.07


def test_model_reading_is_visible_separately_and_never_added_to_the_json():
    evidence = from_text(TEXT.replace("5,1", ""))
    candidate = json.dumps({
        "supplier_name": "Acme Ltd", "invoice_date": "2026-01-01", "total_ht": 396.07, "total_ttc": 396.07,
        "line_items": [{"description": "Clavier sans fil", "quantity": 54, "unit_price": 77.66, "total_ht": 396.07}],
    })
    backend = MagicMock(return_value={"raw": candidate, "finish_reason": "stop"})
    result = run_grounded_inference(evidence, {"id": "test", "input": evidence.text}, [], backend, allow_correction=False)
    assert result["payload"]["line_items"][0]["quantity"] is None
    assert result["unverified_model_readings"]["line_items.0.quantity"] == "54"
    assert "Valeur non recoupée : à vérifier manuellement" in result["report"]
    assert "non intégrée au résultat" in result["report"]
    assert "54" in result["report"]
    assert "unverified_model_readings" not in result["payload"]


def test_more_than_half_doubtful_fields_adds_scan_instruction_at_the_top():
    evidence = from_text("Texte incomplet")
    statuses = {"supplier_name": "non vérifié", "total_ht": "non vérifié", "total_ttc": "valide"}
    from invoice_validation import Validation
    report = render_report(evidence, Validation(), statuses, [], insufficient=True)
    assert "Lecture partielle" in report.split("\n\n")[1]


def test_partition_is_exact_frozen_and_has_no_overlap():
    split = json.loads((ROOT / "tests/stress/split.json").read_text())["cases"]
    assert split_cases(list(split)) == split
    assert sum(part == "tuning" for part in split.values()) == 21
    assert sum(part == "validation" for part in split.values()) == 14
    assert set(split.values()) == {"tuning", "validation"}


def test_deskew_estimates_geometry_not_invoice_values():
    image = Image.new("L", (800, 500), 255)
    drawing = ImageDraw.Draw(image)
    for top in range(60, 440, 30):
        drawing.line((50, top, 750, top), fill=0, width=2)
    rotated = image.rotate(2, expand=True, fillcolor=255)
    assert abs(estimate_skew(rotated) + 2) < .4


def test_reported_errors_are_reconstructions_not_actual_model_captures():
    for expected in REFERENCES.values():
        for profile in ("reported", "coherent", "permuted"):
            before = copy.deepcopy(expected)
            candidate = error_profile(expected, profile)
            assert candidate != expected
            assert before == expected
    assert error_profile(REFERENCES["france"], "reported")["line_items"][1]["quantity"] == 142
    assert error_profile(REFERENCES["uk"], "reported")["line_items"][0]["unit_price"] == "75,797"


def test_colab_notebook_cells_compile_without_loading_model():
    notebook = json.loads((ROOT / "notebooks/real_model_check.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), f"real_model_check.ipynb:{index}", "exec")


def test_tesseract_timeout_is_explicitly_unverified_not_a_crash(caplog):
    from unittest.mock import patch
    from app import main
    from invoice_evidence import UnreadableOCR
    with patch.object(main.pytesseract, "image_to_data", side_effect=RuntimeError("Tesseract process timeout")):
        with pytest.raises(UnreadableOCR, match="Délai OCR dépassé"):
            main.ocr_image(Image.new("RGB", (20, 20)))
        result = list(main.run_extraction_ui(str(main.EXAMPLES[0][1])))[-1]
    payload, report, technical, advanced, first_pass_raw, validation, download = result
    # No false value: a timed-out read never confirms a field.
    assert payload["total_ht"] is None
    assert payload["supplier_name"] is None
    assert all(value is None for item in payload["line_items"] for value in item.values())
    assert not validation["flags"]
    # The exact cause is stated, both in the report and in the technical panel.
    assert "Délai de lecture dépassé" in report
    assert "Délai de lecture dépassé" in technical
    # No internal jargon is exposed to the reader.
    for jargon in ("Traceback", "None", "Pydantic", "UnreadableOCR", "Tesseract process timeout"):
        assert jargon not in report
        assert jargon not in technical
    # The traceback stays available to operators, in the logs only.
    assert "Traceback" in caplog.text


def test_unrelated_ocr_runtime_errors_are_not_silently_swallowed():
    from unittest.mock import patch
    from app import main
    with patch.object(main.pytesseract, "image_to_data", side_effect=RuntimeError("unexpected OCR error")):
        with pytest.raises(RuntimeError, match="unexpected OCR error"):
            main.ocr_image(Image.new("RGB", (20, 20)))
