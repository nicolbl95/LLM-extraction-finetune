"""The benchmark harness must score honestly before it is ever pointed at a real model.

Every case here runs the production `document_text -> extract_invoice` path with a deterministic
stand-in for the model, so the verdicts, the gate handling and the arithmetic rules are exercised
without OCR cost and without GPU. Nothing in this file asserts production behaviour changes.
"""

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import bench_real_model as bench  # noqa: E402
from app import main as application  # noqa: E402
from invoice_evidence import DocumentEvidence, EvidenceText, from_text  # noqa: E402

TEXT = (
    "Supplier: Stress Services Ltd\nInvoice date: 2026-01-01\nCurrency: EUR\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Installation | 2 | 617,28 | 1 234,56\n"
    "Manuel | 2 | 25,00 | 50,00\n"
    "Sous-total HT : 1 284,56 €\n"
    "TVA (20%) sur 1 234,56 € : 246,91 €\n"
    "Total TTC : 1 531,47 €"
)
TRUTH = {
    "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
    "total_ht": 1284.56, "total_ttc": 1531.47,
    "line_items": [
        {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
        {"description": "Manuel", "quantity": 2, "unit_price": 25.0, "total_ht": 50.0},
    ],
}
FIXTURE_PATH = "tests/fixtures/layout-joanna-binet.png"


def model_json(**overrides):
    payload = {
        "supplier_name": "Stress Services Ltd",
        "invoice_date": "2026-01-01",
        "total_ht": 1284.56,
        "total_ttc": 1531.47,
        "line_items": [
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
            {"description": "Manuel", "quantity": 2, "unit_price": 25.0, "total_ht": 50.0},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def text_evidence(*, with_ttc=True, misprint=False):
    text = TEXT
    if not with_ttc:
        text = text.replace("\nTotal TTC : 1 531,47 €", "")
    if misprint:
        text = text.replace("Manuel | 2 | 25,00 | 50,00", "Manuel | 2 | 25,00 | 50,00").replace(
            "Installation | 2 | 617,28 | 1 234,56", "Installation | 5,1 | 77,66 | 369,07")
    return from_text(text)


@pytest.fixture
def documents(monkeypatch):
    def install(evidence):
        monkeypatch.setattr(application, "document_text", lambda filename: EvidenceText(evidence))
    return install


def case(truth=None, *, case_id="case", expected_flags=()):
    return {"id": case_id, "path": FIXTURE_PATH, "status": "diagnostic",
            "truth": {"inline": truth or TRUTH}, "expected_flags": list(expected_flags)}


def run(truth=None, *, responses, case_id="case", expected_flags=()):
    return bench.run_case(case(truth, case_id=case_id, expected_flags=expected_flags),
                          predict=bench.SimulatedModel(responses))


def verdicts(result):
    return [record["verdict"] for record in result["fields"]]


def test_correct_and_incorrect_proposals_are_separated(documents):
    documents(text_evidence())
    raw = model_json(
        total_ttc=1531.40,
        line_items=[
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
            {"description": "Manuel", "quantity": 3, "unit_price": 25.0, "total_ht": 50.0},
        ],
    )
    result = run(responses=[{"raw": raw}])
    aggregate = bench.aggregate([result])
    assert result["model"]["raw_parse"] == "ok"
    assert aggregate["model"]["correct"] >= 6
    assert aggregate["model"]["wrong"] >= 2
    assert "model_correct_retained" in verdicts(result)
    assert aggregate["final"]["wrong"] == 0, "grounding must not keep a number the document does not print"
    assert aggregate["final"]["precision"] == 1.0
    assert result["runtime"]["model_seconds"] >= 0 and result["runtime"]["ocr_seconds"] >= 0


def test_gate_blocked_document_is_not_a_model_failure(documents, monkeypatch):
    evidence = text_evidence()
    evidence.source_kind = "ocr"
    evidence.quality_issues.append("Confiance OCR moyenne insuffisante : 70.0/100 (minimum 85).")
    documents(evidence)
    simulator = bench.SimulatedModel([{"raw": model_json()}])
    result = bench.run_case(case(), predict=simulator)
    aggregate = bench.aggregate([result])
    assert result["gate"]["reached_model"] is False
    assert result["gate"]["reason"] == "preuves OCR insuffisantes"
    assert simulator.calls == [], "a refused document must never reach the model"
    assert set(verdicts(result)) == {"gate_blocked"}
    assert aggregate["gate"]["blocked"] == [{"id": "case", "reason": "preuves OCR insuffisantes"}]
    assert aggregate["model"]["wrong"] == 0 and aggregate["final"]["wrong"] == 0


def test_production_path_matches_a_direct_call_and_restores_predict(documents):
    documents(text_evidence())
    original = application.predict_invoice
    stub = bench.SimulatedModel([{"raw": model_json()}])
    result = bench.run_case(case(), predict=stub)
    assert application.predict_invoice is original, "the harness must never leave a patched model"
    with bench.injected_model(bench.SimulatedModel([{"raw": model_json()}])):
        direct = application.extract_invoice(EvidenceText(text_evidence()))
    assert result["final_payload"] == direct, "instrumentation must not change the final payload"


def test_numeric_comparison_uses_the_frozen_tolerance():
    assert bench.same_value(617.28, "617.28000001")
    assert bench.same_value(617.28, 617.29), "the frozen tolerance is one cent, inclusive"
    assert not bench.same_value(617.28, 617.30)
    assert bench.same_value("Stress Services Ltd", "Stress Services Ltd")
    assert not bench.same_value(None, Decimal("0"))


def test_printed_error_is_preserved_and_the_model_correction_is_rejected(documents):
    truth = {
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": 1531.47,
        "line_items": [
            {"description": "Installation", "quantity": 5.1, "unit_price": 77.66, "total_ht": 369.07},
        ],
    }
    single_row = (
        "Supplier: Stress Services Ltd\nInvoice date: 2026-01-01\nCurrency: EUR\n"
        "Description | Quantity | Net unit price | Net amount\n"
        "Installation | 5,1 | 77,66 | 369,07\n"
        "Sous-total HT : 369,07 €\nTotal TTC : 369,07 €"
    )
    documents(from_text(single_row))
    raw = json.dumps({
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": 1531.47,
        "line_items": [
            {"description": "Installation", "quantity": 5.1, "unit_price": 77.66, "total_ht": 396.07},
        ],
    })
    result = run(truth, responses=[{"raw": raw}])
    printed = [record for record in result["fields"] if record["path"] == "line_items.0.total_ht"][0]
    assert printed["model"] == "396.07"
    assert bench.same_value(printed["final"], 369.07), "the printed amount is the document's, not the product"
    assert not bench.same_value(printed["final"], 396.07)
    assert printed["verdict"] == "model_wrong_recovered"
    assert result["arithmetic_substitutions"] == [
        {"line": 0, "printed": "369.07", "model": "396.07", "arithmetic_product": "396.07"}]


def test_gold_null_is_not_a_false_null(documents):
    truth = {
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": None,
        "line_items": [
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
        ],
    }
    documents(text_evidence(with_ttc=False))
    raw = json.dumps({
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": None,
        "line_items": [
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
        ],
    })
    result = run(truth, responses=[{"raw": raw}])
    aggregate = bench.aggregate([result])
    ttc = [record for record in result["fields"] if record["path"] == "total_ttc"][0]
    assert ttc["gold"] is None and ttc["final"] is None
    assert ttc["verdict"] == "correctly_absent"
    assert aggregate["final"]["false_null"] == 0, "an absent printed field is not a false null"
    assert aggregate["final"]["gold_null"] == 1


def test_malformed_json_is_recorded_and_never_crashes(documents):
    documents(text_evidence())
    result = run(responses=[{"raw": "Voici le résultat : {supplier: Stress}"}])
    assert result["model"]["raw_parse"] == "invalid"
    assert result["model"]["parse_error"]
    assert result["gate"]["reached_model"] is True
    assert set(verdicts(result)) <= {"model_missed", "ocr_not_recognized"}
    assert result["line_items"]["model_rows"] == 0


def test_missing_fields_are_distinguished_from_nulls(documents):
    documents(text_evidence())
    result = run(responses=[{"raw": '{"supplier_name": "Stress Services Ltd"}'}])
    supplier = [record for record in result["fields"] if record["path"] == "supplier_name"][0]
    total = [record for record in result["fields"] if record["path"] == "total_ht"][0]
    assert result["model"]["raw_parse"] == "ok"
    assert supplier["model_emitted_key"] is True and supplier["verdict"] == "model_correct_retained"
    assert total["model_emitted_key"] is False and total["verdict"] == "model_missed"
    assert "total_ht" in result["model"]["missing_truth_paths"]
    assert any(path.startswith("line_items.0.") for path in result["model"]["missing_truth_paths"])


def test_repeated_lines_are_counted_not_hidden(documents):
    truth = {
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": 1531.47,
        "line_items": [
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
        ],
    }
    documents(text_evidence())
    raw = json.dumps({
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": 1531.47,
        "line_items": [
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
        ],
    })
    result = run(truth, responses=[{"raw": raw}])
    section = result["line_items"]["model"]
    assert section["row_count_match"] is False
    assert result["line_items"]["duplicates_in_gold"] == 1
    assert section["accuracy"] == 0.5
    assert result["line_items"]["model_rows"] == 1


def test_row_order_is_reported_but_not_scored_as_a_value_error(documents):
    truth = {
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": 1531.47,
        "line_items": [
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
            {"description": "Manuel", "quantity": 2, "unit_price": 25.0, "total_ht": 50.0},
        ],
    }
    documents(text_evidence())
    raw = json.dumps({
        "supplier_name": "Stress Services Ltd", "invoice_date": "2026-01-01",
        "total_ht": 1284.56, "total_ttc": 1531.47,
        "line_items": [
            {"description": "Manuel", "quantity": 2, "unit_price": 25.0, "total_ht": 50.0},
            {"description": "Installation", "quantity": 2, "unit_price": 617.28, "total_ht": 1234.56},
        ],
    })
    result = run(truth, responses=[{"raw": raw}])
    assert result["line_items"]["model"]["order_preserved"] is False
    assert result["line_items"]["model"]["accuracy"] == 1.0
    assert result["line_items"]["model"]["unmatched_actual"] == []


def test_expected_flags_are_reported_by_code():
    export = {"flags": [{"code": "corroborated_line_amount_mismatch", "severity": "avertissement"}]}
    assert bench.expected_flags_present(export, ["corroborated_line_amount_mismatch", "absent_code"]) == {
        "corroborated_line_amount_mismatch": True, "absent_code": False}
    assert bench.expected_flags_present({}, []) == {}


def test_manifest_validation_reports_missing_documents_and_truth():
    problems = bench.validate_manifest([
        {"id": "missing", "path": "does/not/exist.png", "truth": {"inline": {"line_items": []}}},
        {"id": "bad-truth", "path": FIXTURE_PATH, "truth": {"file": "nope.json", "pointer": "x"}},
        {"id": "no-rows", "path": FIXTURE_PATH, "truth": {"inline": {"supplier_name": "x"}}},
    ])
    messages = " ".join(problem["problem"] for problem in problems)
    assert "document not found" in messages and "truth not readable" in messages and "line_items" in messages


def test_check_data_only_touches_neither_ocr_nor_model(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "cases": [case()]}), encoding="utf-8")
    monkeypatch.setattr(application, "document_text",
                        lambda filename: pytest.fail("preflight must not run OCR"))
    monkeypatch.setattr(sys, "argv", [
        "bench_real_model.py", "--manifest", str(manifest), "--check-data-only"])
    assert bench.main() == 0


def test_only_filter_runs_a_subset(monkeypatch, tmp_path, documents):
    documents(text_evidence())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "cases": [case(case_id="a"), case(case_id="b")]}),
                        encoding="utf-8")
    simulation = tmp_path / "sim.json"
    simulation.write_text(json.dumps({"a": [{"raw": model_json()}], "b": [{"raw": model_json()}]}),
                          encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "bench_real_model.py", "--manifest", str(manifest), "--only", "b",
        "--simulate", str(simulation), "--output", str(output)])
    assert bench.main() == 0
    report = json.loads((output / "bench.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in report["cases"]] == ["b"]
    assert report["manifest"]["declared_cases"] == 2


def test_manifest_fingerprint_is_stable_and_covers_truth_and_flags():
    first = bench.manifest_fingerprint([case()])
    assert first == bench.manifest_fingerprint([case()])
    assert first != bench.manifest_fingerprint([case(expected_flags=["corroborated_line_amount_mismatch"])])
    assert first != bench.manifest_fingerprint([case({"total_ht": 1.0, "line_items": []})])
