"""Observed selector/worker-failure paths; no hosted inference is performed."""

import hashlib
import json
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from app import main
from invoice_evidence import EvidenceText, from_text
from invoice_grounding import inference_failure_category


def reference(identifier):
    return next(row for row in map(
        json.loads, (ROOT / "data/test.jsonl").read_text(encoding="utf-8").splitlines(),
    ) if row["id"] == identifier)


@pytest.mark.parametrize("altered", [False, True])
def test_us_example_selection_and_altered_upload_have_distinct_grounded_results(altered):
    if not {"fra", "eng"} <= set(main.pytesseract.get_languages()):
        pytest.skip("This real-image/captured-OCR test needs Tesseract fra+eng")
    row = reference("invoice-0435")
    with tempfile.TemporaryDirectory(dir=ROOT / ".venv") as directory, \
            patch.dict(os.environ, {"GRADIO_TEMP_DIR": directory}):
        app = main.build_app()
        events = {event["api_name"]: event for event in app.config["dependencies"]}
        select = app.fns[events["load_example"]["id"]].fn
        extract = app.fns[events["run_extraction_ui"]["id"]].fn
        if altered:
            path = ROOT / "tests/fixtures/us-line3-amount-error.png"
            expected_hash = "627eca6531248fb876b6d97d8feca96649fafcff4081723e6c539c2ee03a6a6b"
        else:
            # Exercise the actual user-reported regional selector, not a fabricated report.
            selected = select((2, [str(main.EXAMPLES[2][1])]))
            path = Path(selected["value"]["path"])
            expected_hash = "5a7b9bc7e1faef3f9570e3bfccd6626d583f66da189609b9ca00482466b27ae4"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        with patch.object(main, "predict_invoice", return_value={
            "raw": json.dumps(row["output"]), "finish_reason": "stop",
            "generation_seconds": 1.0,
        }) as predict, patch.object(
            main, "_compact_validation_summary", wraps=main._compact_validation_summary,
        ) as summary:
            frames = list(extract(str(path)))
        payload, status, performance, download = frames[-1]
        diagnostics = summary.call_args.args[1]
        evidence = diagnostics["evidence"]
        third = evidence.rows[2]
        if altered:
            assert {number.value for number in third.cells["total_ht"]} == {Decimal("369.07")}
            assert payload["line_items"][2]["total_ht"] == 369.07
            assert payload["line_items"][2]["quantity"] is None
            assert payload["line_items"][2]["unit_price"] is None
            assert payload["total_ht"] == 8972.8
            assert "Facture cohérente" not in status
            assert "369.07" in status and "396.07" in status
            assert {"corroborated_line_amount_mismatch", "subtotal"} <= {
                flag["code"] for flag in diagnostics["validation"]["flags"]
            }
        else:
            assert payload == row["output"]
            assert "Extraction cohérente — 24/24" in status
            assert not diagnostics["validation"]["flags"]
        assert json.loads(Path(download).read_text(encoding="utf-8")) == payload
        assert predict.call_count == 1


def test_observed_zerogpu_worker_abort_preserves_twenty_ocr_fields_without_claiming_generation():
    row = reference("invoice-0406")
    captured = {}
    original = main.extract_invoice
    def extraction(text, **kwargs):
        result = original(text, **kwargs)
        captured.update(kwargs["diagnostics"])
        return result
    with tempfile.TemporaryDirectory(dir=ROOT / ".venv") as directory, \
            patch.dict(os.environ, {"GRADIO_TEMP_DIR": directory}), \
            patch.object(main, "document_text", return_value=EvidenceText(from_text(
                row["input"], locale=row["locale"],
            ))), patch.object(main, "predict_invoice", side_effect=main.gr.Error(
                "GPU task aborted",
            )) as predict, patch.object(main, "extract_invoice", side_effect=extraction):
        result = list(main.run_recruiter_extraction_ui(str(main.EXAMPLES[0][1])))[-1]
    assert result[0] == row["output"]
    assert "OCR seul" in result[2]
    assert "0/1 réussies" in result[2]
    assert "Génération Qwen+LoRA : non effectuée" in result[2]
    assert "GPU task aborted" not in result[1] + result[2]
    assert captured["model_info"]["failures"] == [{
        "phase": "initial", "category": "zerogpu_worker_aborted",
        "stage": "zerogpu_worker", "exception_type": "Error",
    }]
    assert captured["model_info"]["reason"] == "indisponibilité technique"
    assert captured["model_info"]["used"] is False
    predict.assert_called_once()


@pytest.mark.parametrize("error,category,stage", [
    (RuntimeError("GPU task aborted"), "zerogpu_worker_aborted", "zerogpu_worker"),
    (RuntimeError("ZeroGPU quota exceeded"), "zerogpu_quota", "zerogpu_allocation"),
    (TimeoutError("request timed out"), "timeout", "unknown"),
    (RuntimeError("CUDA out of memory"), "out_of_memory", "unknown"),
    (ImportError("private dependency value"), "dependency_unavailable", "dependency_import"),
    (FileNotFoundError("private model path"), "artifact_missing", "artifact_resolution"),
    (ValueError("loading configuration failed"), "unclassified_runtime_error", "unknown"),
])
def test_internal_categories_are_content_free_and_do_not_guess_hidden_failure_stage(error, category, stage):
    result = inference_failure_category(error, phase="initial")
    assert result == {
        "phase": "initial", "category": category,
        "stage": stage, "exception_type": type(error).__name__,
    }


def test_successful_uk_model_pass_keeps_exact_payload_and_no_failure_category():
    row = reference("invoice-0110")
    diagnostics = {}
    with patch.object(main, "predict_invoice", return_value={
        "raw": json.dumps(row["output"]), "finish_reason": "stop",
        "generation_seconds": 11.64,
    }) as predict:
        payload = main.extract_invoice(EvidenceText(from_text(
            row["input"], locale=row["locale"],
        )), diagnostics=diagnostics)
    assert payload == row["output"]
    assert payload["line_items"][0]["quantity"] == 7.8
    assert len(diagnostics["field_statuses"]) == 16
    assert diagnostics["model_info"]["failures"] == []
    assert diagnostics["model_info"]["passes_succeeded"] == 1
    assert diagnostics["model_info"]["generation_seconds"] == 11.64
    assert not diagnostics["validation"]["flags"]
    predict.assert_called_once()
