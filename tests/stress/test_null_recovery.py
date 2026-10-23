import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests/stress")]

from cases import error_profile, REFERENCES
from experiment_ocr import split_cases
from invoice_evidence import from_text
from invoice_grounding import read_model, reconcile, run_grounded_inference
from invoice_validation import render_report
from ocr_consensus import build_consensus
from ocr_preprocessing import estimate_skew


TEXT = (
    "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: EUR\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Clavier sans fil | 5,1 | 77,66 | 396,07\n"
    "Subtotal: 396,07\nVAT (0%): 0\nInvoice total: 396,07"
)


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
    assert "Image de qualité insuffisante : renvoyez un scan plus net (300 dpi, bien droit, sans flou)" in report.split("\n\n")[1]


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
    payload, report, technical, advanced, validation, download = result
    assert payload["total_ht"] is None
    assert "Preuves OCR insuffisantes" in report
    assert not validation["flags"]
    assert "Temps OCR dépassé" in technical
    assert "Traceback" in caplog.text


def test_unrelated_ocr_runtime_errors_are_not_silently_swallowed():
    from unittest.mock import patch
    from app import main
    with patch.object(main.pytesseract, "image_to_data", side_effect=RuntimeError("unexpected OCR error")):
        with pytest.raises(RuntimeError, match="unexpected OCR error"):
            main.ocr_image(Image.new("RGB", (20, 20)))
