import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/stress"), str(ROOT / "scripts")]

from invoice_evidence import DocumentEvidence, from_text, from_tsv
from invoice_grounding import UNVERIFIED, read_model, reconcile, run_grounded_inference
from invoice_validation import document_validation, render_report
from ocr_quality import assess_ocr
from run_stress_check import run


@pytest.fixture(scope="module")
def stress_report():
    import shutil
    import pytesseract
    if not shutil.which("tesseract") or not {"fra", "eng"} <= set(pytesseract.get_languages()):
        pytest.fail("Stress acceptance requires actual Tesseract fra+eng, not a simulated OCR or silent skip")
    return run()


@pytest.mark.parametrize("index", range(35))
def test_stress_no_wrong_value_no_phantom_no_crash(stress_report, index):
    case = stress_report["cases"][index]
    assert not case["crash"], case
    assert not case["wrong_values"], case
    assert not case["phantom_anomalies"], case
    for path in case["false_nulls"]:
        assert case["statuses"].get(path) in {UNVERIFIED, "neutralisé (non fiable)"}, case


def test_layout_variants_are_independent_and_fully_recovered(stress_report):
    layouts = [case for case in stress_report["cases"] if case["case"].startswith("layout/")]
    assert len(layouts) == 11
    assert not any(case["false_nulls"] for case in layouts)
    assert not any(case["crash"] or case["wrong_values"] or case["phantom_anomalies"] for case in layouts)


def test_destroyed_ocr_with_false_model_values_is_not_verified(stress_report):
    destroyed = [case for case in stress_report["cases"] if case["case"].endswith("/destroyed")]
    assert len(destroyed) == 3
    for case in destroyed:
        assert not case["wrong_values"]
        assert set(case["statuses"].values()) == {UNVERIFIED}
        assert all(value is None for item in case["final"]["line_items"] for value in item.values())
        assert not case["phantom_anomalies"]
        assert "Preuves OCR insuffisantes : extraction non recoupée" in case["report"]


def test_average_confidence_and_unrecognized_table_produce_insufficient_report():
    data = {"text": ["FACTURE"], "left": [0], "top": [0], "width": [50], "height": [10], "conf": [15]}
    evidence = assess_ocr(from_tsv(data), from_tsv(data))
    assert "Confiance OCR" in " ".join(evidence.quality_issues)
    model = read_model('{"line_items":[{"description":"Inventé","quantity":54,"unit_price":77.68,"total_ht":4210.32}]}')
    result = reconcile(evidence, [model])
    report = render_report(evidence, result.validation, result.statuses, [], insufficient=True)
    assert "Preuves OCR insuffisantes : extraction non recoupée" in report
    assert set(result.statuses.values()) == {UNVERIFIED}
    assert not result.validation.flags
    assert all(value is None for item in result.payload["line_items"] for value in item.values())


def test_model_ocr_line_count_disagreement_is_not_a_document_error():
    text = (
        "Currency: EUR\nDescription | Quantity | Unit price | Line total\n"
        "Service | 2 | 10 | 20\nSubtotal: 20\nVAT (0%): 0\nInvoice total: 20"
    )
    evidence = from_text(text)
    evidence.source_kind = "ocr"
    model = read_model('{"line_items":[{"quantity":54},{"quantity":142}]}')
    result = reconcile(evidence, [model])
    assert set(result.statuses.values()) == {UNVERIFIED}
    assert not result.validation.flags
    assert result.payload["total_ht"] is None
    assert any("Nombre d'articles" in warning for warning in result.warnings)


def test_bad_quality_does_not_allocate_gpu():
    from unittest.mock import MagicMock
    evidence = DocumentEvidence("bruit", None, [], [], quality_issues=["Tableau OCR non reconnu."])
    predict = MagicMock(side_effect=AssertionError("No GPU on insufficient evidence"))
    result = run_grounded_inference(evidence, {"input": "bruit"}, [], predict)
    predict.assert_not_called()
    assert "Preuves OCR insuffisantes" in result["report"]
    assert not result["validation"]["flags"]


def test_app_empty_ocr_is_insufficient_not_a_success_or_crash():
    from unittest.mock import patch
    from app import main
    from invoice_evidence import UnreadableOCR
    with patch.object(main, "document_text", side_effect=UnreadableOCR("Aucun texte détecté")), \
            patch.object(main, "predict_invoice", side_effect=AssertionError("No GPU without evidence")):
        payload, report, technical, advanced, first_pass_raw, validation, download = list(
            main.run_extraction_ui(str(main.EXAMPLES[0][1]))
        )[-1]
    assert "Preuves OCR insuffisantes : extraction non recoupée" in report
    assert payload["total_ht"] is None
    assert not validation["flags"]
    assert "0 lignes lues" in technical


def test_single_ocr_read_cannot_bypass_the_quality_gate():
    evidence = from_text(
        "Currency: EUR\nDescription | Quantity | Unit price | Line total\n"
        "Service | 2 | 10 | 20\nSubtotal: 20\nInvoice total: 20"
    )
    evidence.source_kind = "ocr"
    result = reconcile(evidence, [{}])
    assert set(result.statuses.values()) == {UNVERIFIED}
    assert result.payload["total_ht"] is None
    assert not result.validation.flags
    assert any("seconde lecture" in warning for warning in result.warnings)


def test_real_model_checker_rejects_ocr_only_success():
    from run_real_model_check import compare_runs
    expected = {"total_ht": 20}
    fallback = [{"attempts": [], "final": expected, "technical_errors": ["GPU failed"]}] * 3
    result = compare_runs(fallback, expected)
    assert result["final_equals_reference"] == [True] * 3
    assert not result["actual_model_completed"]
    assert not result["raw_deterministic"]
    assert not result["final_deterministic"]
    real = [{"attempts": [{"raw": '{"total_ht":20}', "finish_reason": "stop"}],
             "final": expected, "technical_errors": []} for _ in range(3)]
    assert all(compare_runs(real, expected).values())
    real[2]["attempts"][0]["raw"] = '{"total_ht": 20}'
    assert not compare_runs(real, expected)["raw_deterministic"]
