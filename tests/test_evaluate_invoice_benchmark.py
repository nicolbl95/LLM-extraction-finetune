import contextlib
import io
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_invoice_benchmark as benchmark


def present(value):
    return {"state": "present", "value": value}


def null_state(state):
    return {"state": state, "value": None}


def truth(*, supplier="Acme", rows=None, total_ht=100, total_ttc=120, supplier_state=None):
    top = {
        "supplier_name": present(supplier) if supplier_state is None else supplier_state,
        "invoice_date": present("2025-01-02"),
        "total_ht": present(total_ht),
        "total_ttc": present(total_ttc),
    }
    return {
        "status": "complete",
        "top_level": top,
        "line_items": rows or [],
    }


def row(description, quantity=1, unit_price=10, amount=10, *, amount_state=None):
    return {
        "description": present(description),
        "quantity": present(quantity),
        "unit_price": present(unit_price),
        "total_ht": present(amount) if amount_state is None else amount_state,
    }


def output(*, supplier="Acme", date_value="2025-01-02", total_ht=100, total_ttc=120, rows=None):
    return {
        "supplier_name": supplier,
        "invoice_date": date_value,
        "total_ht": total_ht,
        "total_ttc": total_ttc,
        "line_items": [] if rows is None else rows,
    }


def response(document_id, payload, **overrides):
    result = {
        "document_id": document_id,
        "model_variant": "qwen_lora",
        "software_revision": "test-revision",
        "execution_mode": "recorded",
        "inference_status": "succeeded",
        "model_used": True,
        "ocr_fallback": False,
        "input_sha256": None,
        "ocr_evidence_sha256": "ocr-hash",
        "prompt_sha256": "prompt-hash",
        "generation_settings_sha256": "settings-hash",
        "latency": {
            "ocr_seconds": 1.0,
            "generation_seconds": 2.0,
            "end_to_end_seconds": 3.0,
        },
        "output": payload,
    }
    result.update(overrides)
    return result


class InvoiceBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_document(self, document_truth, *, document_id="doc-1"):
        path = self.root / f"{document_id}.png"
        path.write_bytes(b"synthetic test invoice")
        return {
            "document_id": document_id,
            "path": path.name,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ground_truth": {
                "scoreable": True,
                **document_truth,
            },
        }

    def manifest(self, documents, *, systems=None):
        return {
            "documents": documents,
            "evaluation_sets": {
                "test": {
                    "description": "synthetic metric-test set",
                    "independence_status": "synthetic",
                    "document_ids": [document["document_id"] for document in documents],
                    "systems": ["qwen_lora"] if systems is None else systems,
                }
            },
        }

    def test_strict_and_substantive_metrics_normalize_text_numbers_and_dates(self):
        document = self.make_document(truth(rows=[row("Café   crème", 1.25, 20, 25)]))
        payload = output(
            supplier="ACME",
            date_value="2025-01-02",
            rows=[{
                "description": "Cafe creme",
                "quantity": 1.2500001,
                "unit_price": 20.001,
                "total_ht": 25.004,
            }],
        )
        result = benchmark.score_document(
            document, response("doc-1", payload),
        )
        self.assertEqual(result["metrics"]["strict_correct"], 6)
        self.assertEqual(result["metrics"]["correct_substantive_only"], 2)
        self.assertEqual(result["metrics"]["substantive_accuracy"], 1.0)
        self.assertTrue(result["schema_valid"])

    def test_wrong_non_null_and_avoidable_nulls_are_penalized(self):
        document = self.make_document(truth(rows=[row("Widget")]))
        payload = output(supplier="Unrelated supplier", total_ht=99, total_ttc=None)
        result = benchmark.score_document(document, response("doc-1", payload))
        self.assertEqual(result["metrics"]["incorrect_non_null"], 2)
        self.assertEqual(result["metrics"]["false_null"], 5)
        self.assertEqual(result["metrics"]["field_recall"], 0.125)
        self.assertEqual(result["metrics"]["false_null_rate"], 0.625)
        self.assertLess(result["metrics"]["exact_field_accuracy"], 1.0)

    def test_partial_invoice_extraction_reports_coverage_without_claiming_completeness(self):
        document = self.make_document(truth())
        payload = output(total_ht=None, total_ttc=None)
        result = benchmark.score_document(document, response("doc-1", payload))
        self.assertEqual(result["metrics"]["field_recall"], 0.5)
        self.assertEqual(result["document_coverage"], 0.5)
        self.assertFalse(result["complete_invoice_correct"])

    def test_correct_nulls_are_distinguished_from_false_nulls(self):
        gold = truth()
        gold["top_level"]["total_ttc"] = null_state("absent")
        document = self.make_document(gold)
        result = benchmark.score_document(
            document, response("doc-1", output(total_ttc=None)),
        )
        self.assertEqual(result["metrics"]["correct_null"], 1)
        self.assertEqual(result["metrics"]["false_null"], 0)
        self.assertEqual(result["metrics"]["correct_null_rate"], 1.0)

    def test_all_null_output_cannot_look_accurate(self):
        document = self.make_document(truth())
        result = benchmark.score_document(
            document,
            response("doc-1", output(supplier=None, date_value=None, total_ht=None, total_ttc=None)),
        )
        self.assertEqual(result["metrics"]["field_recall"], 0.0)
        self.assertEqual(result["metrics"]["false_null_rate"], 1.0)
        self.assertEqual(result["metrics"]["strict_correct"], 0)
        self.assertEqual(result["complete_invoice_correct"], False)

    def test_swapped_rows_align_by_description_and_fail_order_accuracy(self):
        document = self.make_document(truth(rows=[
            row("Alpha", 1, 10, 10),
            row("Beta", 2, 20, 40),
        ]))
        payload = output(rows=[
            {"description": "Beta", "quantity": 2, "unit_price": 20, "total_ht": 40},
            {"description": "Alpha", "quantity": 1, "unit_price": 10, "total_ht": 10},
        ])
        result = benchmark.score_document(document, response("doc-1", payload))
        self.assertEqual(result["line_items"]["matched_rows"], 2)
        self.assertEqual(result["line_items"]["missing_rows"], [])
        self.assertFalse(result["line_items"]["order_preserved"])
        self.assertEqual(result["field_metrics"]["line_items.total_ht"]["field_recall"], 1.0)
        self.assertFalse(result["complete_invoice_correct"])

    def test_missing_and_extra_rows_are_reported_and_extra_values_count_as_wrong(self):
        document = self.make_document(truth(rows=[
            row("Alpha", 1, 10, 10),
            row("Beta", 2, 20, 40),
        ]))
        payload = output(rows=[
            {"description": "Alpha", "quantity": 1, "unit_price": 10, "total_ht": 10},
            {"description": "Extra", "quantity": 1, "unit_price": 1, "total_ht": 1},
        ])
        result = benchmark.score_document(document, response("doc-1", payload))
        self.assertEqual(result["line_items"]["missing_rows"], [1])
        self.assertEqual(result["line_items"]["extra_rows"], [1])
        self.assertEqual(result["line_items"]["alignment_accuracy"], 0.5)
        self.assertGreater(result["metrics"]["incorrect_non_null"], 0)

    def test_no_positional_fallback_for_unmatched_descriptions(self):
        document = self.make_document(truth(rows=[row("Printed service", 1, 50, 50)]))
        payload = output(rows=[{
            "description": "Invented service",
            "quantity": 1,
            "unit_price": 50,
            "total_ht": 50,
        }])
        result = benchmark.score_document(document, response("doc-1", payload))
        self.assertEqual(result["line_items"]["matched_rows"], 0)
        self.assertEqual(result["line_items"]["missing_rows"], [0])
        self.assertEqual(result["line_items"]["extra_rows"], [0])

    def test_derived_only_printed_amount_is_not_scored_as_correct(self):
        document = self.make_document(truth())
        document["ground_truth"]["top_level"]["total_ht"] = null_state("derived_only")
        result = benchmark.score_document(
            document, response("doc-1", output(total_ht=100)),
        )
        self.assertEqual(result["metrics"]["strict_correct"], 3)
        self.assertEqual(result["metrics"]["incorrect_non_null"], 1)
        self.assertEqual(result["metrics"]["accepted_value_precision"], 0.75)

    def test_calculated_line_total_cannot_substitute_for_printed_amount(self):
        document = self.make_document(truth(rows=[row("Printed line", 3, 10, 29.99)]))
        payload = output(rows=[{
            "description": "Printed line",
            "quantity": 3,
            "unit_price": 10,
            "total_ht": 30,
        }])
        result = benchmark.score_document(document, response("doc-1", payload))
        line_amount = result["field_metrics"]["line_items.total_ht"]
        self.assertEqual(line_amount["expected_present"], 1)
        self.assertEqual(line_amount["strict_correct"], 0)
        self.assertEqual(line_amount["incorrect_non_null"], 1)

    def test_ambiguous_ground_truth_is_excluded_and_reported(self):
        document = self.make_document(truth(supplier_state=null_state("ambiguous")))
        result = benchmark.score_document(
            document, response("doc-1", output(supplier="Candidate A")),
        )
        self.assertEqual(result["metrics"]["unscored_fields"], 1)
        self.assertEqual(result["metrics"]["unscored_non_null_predictions"], 1)
        self.assertEqual(result["metrics"]["expected_present"], 3)
        self.assertEqual(result["metrics"]["non_null_predictions"], 3)

    def test_malformed_json_is_a_schema_failure_not_a_successful_empty_invoice(self):
        document = self.make_document(truth())
        result = benchmark.score_document(
            document, response("doc-1", '{"supplier_name":'),
        )
        self.assertFalse(result["schema_valid"])
        self.assertEqual(result["schema_errors"], ["output is not valid JSON"])
        self.assertEqual(result["metrics"]["field_recall"], 0.0)

    def test_missing_response_counts_as_failure_and_not_as_correct_nulls(self):
        document = self.make_document(truth())
        report = benchmark.evaluate(self.manifest([document]), [], "test", root=self.root)
        variant = report["variants"]["qwen_lora"]
        self.assertEqual(variant["responses_received"], 0)
        self.assertEqual(variant["model_inference_success_rate"], 0.0)
        self.assertEqual(variant["missing_response_ids"], ["doc-1"])
        self.assertEqual(variant["metrics"]["correct_null"], 0)
        self.assertEqual(variant["metrics"]["field_recall"], 0.0)

    def test_inference_fallback_and_latency_are_reported(self):
        document = self.make_document(truth())
        run = response(
            "doc-1",
            output(),
            model_used=False,
            ocr_fallback=True,
            inference_status="not_run",
            execution_mode="simulated",
            input_sha256=document["file_sha256"],
            latency={"ocr_seconds": 2, "generation_seconds": None, "end_to_end_seconds": 3},
        )
        report = benchmark.evaluate(
            self.manifest([document]), [run], "test", root=self.root,
        )["variants"]["qwen_lora"]
        self.assertEqual(report["model_inference_success_rate"], 0.0)
        self.assertEqual(report["ocr_only_fallback_frequency"], 1.0)
        self.assertEqual(report["execution_modes"], {"simulated": 1})
        self.assertEqual(report["latency"]["ocr_seconds"]["median_seconds"], 2.0)
        self.assertEqual(report["latency"]["generation_seconds"]["count"], 0)

    def test_base_lora_comparison_requires_matching_evidence_and_settings_hashes(self):
        document = self.make_document(truth())
        base = response(
            "doc-1", output(), model_variant="base_qwen",
            input_sha256=document["file_sha256"],
            model_revision="a" * 40,
        )
        lora = response(
            "doc-1", output(), model_variant="qwen_lora",
            input_sha256=document["file_sha256"],
            model_revision="a" * 40,
            adapter_revision="b" * 40,
        )
        manifest = self.manifest([document], systems=["base_qwen", "qwen_lora"])
        report = benchmark.evaluate(manifest, [base, lora], "test", root=self.root)
        self.assertEqual(report["comparison"]["status"], "paired")
        lora["ocr_evidence_sha256"] = "different-ocr"
        report = benchmark.evaluate(manifest, [base, lora], "test", root=self.root)
        self.assertEqual(report["comparison"]["status"], "not_proven")
        self.assertTrue(any("mismatched ocr_evidence_sha256" in item for item in report["comparison"]["problems"]))
        lora["ocr_evidence_sha256"] = "ocr-hash"
        lora["execution_mode"] = "simulated"
        report = benchmark.evaluate(manifest, [base, lora], "test", root=self.root)
        self.assertEqual(report["comparison"]["status"], "not_proven")
        self.assertTrue(any("simulated output" in item for item in report["comparison"]["problems"]))

    def test_duplicate_responses_and_wrong_input_hash_are_rejected(self):
        document = self.make_document(truth())
        run = response("doc-1", output(), input_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "input SHA-256 mismatch"):
            benchmark.evaluate(self.manifest([document]), [run], "test", root=self.root)
        good = response("doc-1", output())
        good["input_sha256"] = document["file_sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate response"):
            benchmark.evaluate(self.manifest([document]), [good, good], "test", root=self.root)

    def test_exact_duplicate_documents_are_not_double_counted(self):
        first = self.make_document(truth(), document_id="first")
        second = self.make_document(truth(), document_id="second")
        self.assertEqual(first["file_sha256"], second["file_sha256"])
        manifest = self.manifest([first, second])
        with self.assertRaisesRegex(ValueError, "exact duplicate documents"):
            benchmark.evaluate(manifest, [], "test", root=self.root)

    def test_ambiguous_line_description_does_not_force_positional_alignment(self):
        ambiguous_row = row("Unknown row")
        ambiguous_row["description"] = null_state("ambiguous")
        document = self.make_document(truth(rows=[ambiguous_row]))
        payload = output(rows=[{
            "description": "Possible row",
            "quantity": 1,
            "unit_price": 10,
            "total_ht": 10,
        }])
        result = benchmark.score_document(document, response("doc-1", payload))
        self.assertEqual(result["line_items"]["matched_rows"], 0)
        self.assertEqual(result["line_items"]["unscored_predicted_rows"], [0])
        self.assertEqual(result["metrics"]["unscored_fields"], 4)
        self.assertEqual(result["metrics"]["unscored_non_null_predictions"], 4)
        self.assertEqual(result["metrics"]["incorrect_non_null"], 0)

    def test_manifest_path_cannot_escape_root_or_change_hash(self):
        document = self.make_document(truth())
        document["path"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "path escapes"):
            benchmark.evaluate(self.manifest([document]), [], "test", root=self.root)
        document = self.make_document(truth(), document_id="changed")
        document["file_sha256"] = "0" * 64
        manifest = self.manifest([document])
        manifest["evaluation_sets"]["test"]["document_ids"] = ["changed"]
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            benchmark.evaluate(manifest, [], "test", root=self.root)

    def test_ground_truth_source_and_embedded_annotation_hashes_are_verified(self):
        document = self.make_document(truth())
        source = self.root / "truth-source.json"
        source.write_text('{"label":"frozen"}', encoding="utf-8")
        ground_truth = document["ground_truth"]
        ground_truth["source_file"] = source.name
        ground_truth["source_sha256"] = benchmark.file_sha256(source)
        ground_truth["annotation_sha256"] = benchmark.canonical_sha256({
            key: ground_truth[key] for key in ("status", "top_level", "line_items")
        })
        manifest = self.manifest([document])
        benchmark.evaluate(manifest, [], "test", root=self.root)

        ground_truth["top_level"]["supplier_name"] = present("edited label")
        with self.assertRaisesRegex(ValueError, "ground-truth annotation SHA-256 mismatch"):
            benchmark.evaluate(manifest, [], "test", root=self.root)

        ground_truth["top_level"]["supplier_name"] = present("Acme")
        source.write_text('{"label":"changed"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ground-truth source SHA-256 mismatch"):
            benchmark.evaluate(manifest, [], "test", root=self.root)

    def test_strict_json_schema_rejects_extra_keys_and_non_finite_numbers(self):
        payload = output()
        payload["currency"] = "EUR"
        self.assertIsNone(benchmark.validate_output(payload)[0])
        payload = output(total_ht=float("nan"))
        parsed, errors = benchmark.validate_output(payload)
        self.assertIsNone(parsed)
        self.assertTrue(any("finite JSON number" in error for error in errors))

    def test_cli_writes_machine_readable_and_human_readable_reports_from_recorded_output(self):
        document = self.make_document(truth())
        manifest_path = self.root / "manifest.json"
        responses_path = self.root / "responses.jsonl"
        output_path = self.root / "result.json"
        summary_path = self.root / "summary.txt"
        manifest_path.write_text(
            json.dumps(self.manifest([document])), encoding="utf-8",
        )
        recorded = response(
            "doc-1", output(), input_sha256=document["file_sha256"],
        )
        responses_path.write_text(json.dumps(recorded) + "\n", encoding="utf-8")
        arguments = [
            "evaluate_invoice_benchmark",
            "--manifest", str(manifest_path),
            "--responses", str(responses_path),
            "--set", "test",
            "--output", str(output_path),
            "--summary", str(summary_path),
            "--root", str(self.root),
        ]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark.main(), 0)
        report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["variants"]["qwen_lora"]["metrics"]["field_recall"], 1.0)
        self.assertIn("Invoice benchmark: test", summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
