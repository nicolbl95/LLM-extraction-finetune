"""Synthetic mechanics tests only; fixtures are not invoice references."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import annotation_tools as tools  # noqa: E402


def _field(state: str, text: str | None, value: str | None, *, note=None, candidates=None):
    evidence = (
        [{"page_index": 0, "bbox": [0.1, 0.1, 0.2, 0.2], "label_text": None}]
        if state in {"present", "unreadable", "ambiguous"}
        else []
    )
    return {
        "state": state,
        "visible_text": text,
        "normalized_value": value,
        "evidence": evidence,
        "candidates": candidates or [],
        "derivation": None,
        "note": note,
    }


def _annotation(kind="synthetic_test", reviewer_id="fixture-a"):
    role = "test_fixture" if kind == "synthetic_test" else "independent_annotator"
    source_kind = "synthetic_test" if kind == "synthetic_test" else "manual_visual"
    independent = kind == "independent"
    return {
        "schema_version": tools.ANNOTATION_VERSION,
        "annotation_kind": kind,
        "annotation_id": f"annotation-{reviewer_id}",
        "protocol_version": "invoice-blind-annotation-v1",
        "created_at": "2026-01-01T00:00:00Z",
        "document": {
            "document_id": "synthetic-test-only-not-a-source",
            "relative_path": "test-fixtures/not-an-invoice.png",
            "sha256": "a" * 64,
            "file_type": "png",
            "country": None,
            "language": None,
            "currency": _field("absent", None, None, note="No source image in unit test."),
        },
        "reviewer": {
            "reviewer_id": reviewer_id,
            "role": role,
            "attestation": {
                "source_image_viewed": independent,
                "predictions_viewed": False,
                "prior_annotations_viewed": False,
                "peer_annotation_viewed": False,
                "pr64_to_pr69_viewed": False,
            },
        },
        "provenance": {
            "source_kind": source_kind,
            "derived_from_existing_reference": False,
            "permission_status": "not_applicable" if not independent else "approved",
            "prior_use_status": "not_applicable" if not independent else "none_confirmed",
        },
        "invoice": {
            "supplier_name": _field("absent", None, None, note="Synthetic fixture."),
            "invoice_date": _field("absent", None, None, note="Synthetic fixture."),
            "total_ht": _field("present", "7.00", "7.00"),
            "total_ttc": _field("absent", None, None, note="Synthetic fixture."),
        },
        "line_items": [
            {
                "row_id": "p001-r001",
                "row_state": "present",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.3],
                "fields": {
                    "description": _field("present", "Synthetic service", "Synthetic service"),
                    "quantity": _field("present", "2", "2"),
                    "unit_price": _field("present", "3.00", "3.00"),
                    "total_ht": _field("present", "7.00", "7.00"),
                },
            }
        ],
        "taxes": [],
        "discounts": [],
        "financial_summary": [],
        "notes": ["Synthetic test fixture; not a document or benchmark reference."],
    }


def _arbitration(a, b, final_annotation):
    comparison = tools.compare_annotations(a, b)
    final_annotation["reviewer"]["reviewer_id"] = "fixture-adjudicator"
    return {
        "schema_version": tools.ARBITRATION_VERSION,
        "document_id": a["document"]["document_id"],
        "document_sha256": a["document"]["sha256"],
        "completed_at": "2026-01-02T00:00:00Z",
        "reviewer": {
            "reviewer_id": "fixture-adjudicator",
            "role": "adjudicator",
            "source_image_viewed": True,
            "predictions_viewed": False,
            "prior_annotations_viewed": False,
            "peer_annotations_viewed": True,
            "pr64_pr69_viewed": False,
        },
        "resolutions": [
            {
                "path": path,
                "decision": "new_manual",
                "rationale": "Synthetic test of required resolution bookkeeping.",
            }
            for path in comparison["review_paths"]
        ],
        "final_annotation": final_annotation,
    }


class AnnotationToolsTests(unittest.TestCase):
    def test_synthetic_annotation_matches_schema_contract(self):
        annotation = _annotation()
        self.assertEqual(tools.validate_annotation(annotation), [])
        schema = json.loads((ROOT / "annotation.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], tools.ANNOTATION_VERSION)
        self.assertIn("supplier_name", schema["$defs"]["invoiceFields"]["required"])
        run_schema = json.loads(
            (ROOT / "engine_run.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_schema["$id"], "invoice-engine-run-v1")
        self.assertEqual(
            set(run_schema["properties"]["system"]["enum"]),
            {"tesseract_current", "rapidocr_independent", "combined_proposals"},
        )

    def test_present_requires_printed_text_and_visual_box(self):
        annotation = _annotation()
        field = annotation["invoice"]["total_ht"]
        field["visible_text"] = None
        field["evidence"] = []
        messages = [item["message"] for item in tools.validate_annotation(annotation)]
        self.assertIn("present requires printed text", messages)
        self.assertIn("present requires visual evidence", messages)

    def test_derived_only_does_not_become_a_printed_value(self):
        annotation = _annotation()
        annotation["invoice"]["total_ttc"] = {
            "state": "derived_only",
            "visible_text": None,
            "normalized_value": "7.00",
            "evidence": [],
            "candidates": [],
            "derivation": {
                "value": "7.00",
                "formula": "synthetic test only",
                "source_paths": ["invoice.total_ht"],
            },
            "note": "Test that derived values cannot be scored as printed.",
        }
        messages = [item["message"] for item in tools.validate_annotation(annotation)]
        self.assertIn("derived_only must not assert a printed value", messages)

    def test_compare_reports_cell_divergence_without_selecting_a_winner(self):
        a = _annotation()
        b = copy.deepcopy(a)
        b["annotation_id"] = "annotation-fixture-b"
        b["reviewer"]["reviewer_id"] = "fixture-b"
        b["line_items"][0]["fields"]["quantity"]["visible_text"] = "3"
        b["line_items"][0]["fields"]["quantity"]["normalized_value"] = "3"
        result = tools.compare_annotations(a, b)
        self.assertEqual(
            [item["path"] for item in result["value_divergences"]],
            ["line_items[p001-r001].quantity"],
        )
        self.assertNotIn("selected_value", result)

    def test_arithmetic_discrepancy_is_warning_and_does_not_change_annotation(self):
        annotation = _annotation()
        annotation["taxes"] = [
            {
                "tax_id": "tax-1",
                "page_index": 0,
                "bbox": [0.1, 0.4, 0.8, 0.5],
                "fields": {
                    "label": _field("present", "VAT 20%", "VAT 20%"),
                    "rate": _field("present", "20%", "0.20"),
                    "base": _field("present", "100.00", "100.00"),
                    "amount": _field("present", "19.00", "19.00"),
                },
            }
        ]
        before = copy.deepcopy(annotation)
        warnings = tools.arithmetic_warnings(annotation)
        self.assertEqual(
            [item["code"] for item in warnings],
            ["line_product_mismatch", "tax_product_mismatch"],
        )
        self.assertEqual(annotation, before)
        self.assertEqual(warnings[0]["printed"], "7.00")
        self.assertEqual(warnings[0]["calculated"], "6.00")

    def test_sha256_verification_and_path_traversal_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            asset_root = Path(temp_dir)
            (asset_root / "doc.png").write_bytes(b"synthetic test bytes")
            annotation = _annotation()
            annotation["document"]["relative_path"] = "doc.png"
            annotation["document"]["sha256"] = hashlib.sha256(b"synthetic test bytes").hexdigest()
            result = tools.verify_document_hash(annotation, asset_root)
            self.assertTrue(result["ok"])
            annotation["document"]["relative_path"] = "..\\outside.png"
            escaped = tools.verify_document_hash(annotation, asset_root)
            self.assertFalse(escaped["ok"])

    def test_finalize_requires_a_resolution_for_every_divergence(self):
        a = _annotation()
        b = copy.deepcopy(a)
        b["annotation_id"] = "annotation-fixture-b"
        b["reviewer"]["reviewer_id"] = "fixture-b"
        b["line_items"][0]["fields"]["quantity"]["visible_text"] = "3"
        b["line_items"][0]["fields"]["quantity"]["normalized_value"] = "3"
        final = copy.deepcopy(a)
        final["annotation_kind"] = "synthetic_test"
        arbitration = _arbitration(a, b, final)
        arbitration["resolutions"] = []
        with self.assertRaisesRegex(ValueError, "resolution paths mismatch"):
            tools.finalize_reference(a, b, arbitration, allow_synthetic_test=True)

    def test_finalize_emits_strict_json_and_state_sidecar_after_arbitration(self):
        a = _annotation()
        b = copy.deepcopy(a)
        b["annotation_id"] = "annotation-fixture-b"
        b["reviewer"]["reviewer_id"] = "fixture-b"
        b["line_items"][0]["fields"]["quantity"]["visible_text"] = "3"
        b["line_items"][0]["fields"]["quantity"]["normalized_value"] = "3"
        final = copy.deepcopy(a)
        arbitration = _arbitration(a, b, final)
        output, metadata = tools.finalize_reference(
            a, b, arbitration, allow_synthetic_test=True
        )
        self.assertEqual(
            set(output),
            {"supplier_name", "invoice_date", "total_ht", "total_ttc", "line_items"},
        )
        self.assertEqual(
            set(output["line_items"][0]),
            {"description", "quantity", "unit_price", "total_ht"},
        )
        self.assertEqual(output["line_items"][0]["quantity"], 2)
        self.assertEqual(metadata["benchmark_eligible"], False)
        self.assertEqual(
            metadata["field_states"]["line_items[p001-r001].quantity"], "present"
        )

    def test_empty_selection_manifest_reports_missing_slots_without_fake_documents(self):
        manifest = json.loads(
            (ROOT / "holdout_selection_manifest.json").read_text(encoding="utf-8")
        )
        result = tools.validate_selection(manifest)
        self.assertTrue(result["valid"])
        self.assertFalse(result["ready"])
        self.assertEqual(result["selected_document_count"], 0)
        self.assertEqual(result["missing"]["development"]["fr-FR"], 2)
        self.assertEqual(result["missing"]["validation"]["en-GB"], 3)
        self.assertEqual(result["missing"]["final_test"]["en-US"], 5)

    def test_selection_rejects_reuse_of_same_hash_across_splits(self):
        manifest = json.loads(
            (ROOT / "holdout_selection_manifest.json").read_text(encoding="utf-8")
        )
        base = {
            "document_id": "future-id",
            "relative_path": "private/future.png",
            "sha256": "b" * 64,
            "split": "development",
            "locale": "fr-FR",
            "country": "France",
            "language": "French",
            "file_type": "png",
            "genuine_invoice_verified": True,
            "permission_status": "approved",
            "prior_use_status": "none_confirmed",
            "supplier_group_id": "supplier-a",
            "layout_family_id": "layout-a",
            "visual_quality_band": "unassessed",
            "edge_cases": [],
            "annotation_status": "not_started",
            "annotation_a_path": None,
            "annotation_a_sha256": None,
            "annotation_b_path": None,
            "annotation_b_sha256": None,
            "adjudication_path": None,
            "adjudication_sha256": None,
            "scored_field_count": None,
            "line_item_count": None,
            "expected_proposal_count": None,
        }
        second = dict(base, document_id="future-id-2", split="final_test", locale="en-GB")
        manifest["selected_documents"] = [base, second]
        result = tools.validate_selection(manifest)
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("duplicate or previously inventoried content" in d["message"]
                for d in result["diagnostics"])
        )

    def test_malformed_state_type_is_reported_instead_of_raising(self):
        annotation = _annotation()
        annotation["invoice"]["total_ht"]["state"] = []
        errors = tools.validate_annotation(annotation)
        self.assertTrue(any(item["path"] == "$.invoice.total_ht.state" for item in errors))

    def test_selection_rejects_supplier_and_layout_leakage_across_splits(self):
        manifest = json.loads(
            (ROOT / "holdout_selection_manifest.json").read_text(encoding="utf-8")
        )
        base = {
            "document_id": "doc-dev",
            "relative_path": "private/dev.png",
            "sha256": "c" * 64,
            "split": "development",
            "locale": "fr-FR",
            "country": "France",
            "language": "French",
            "file_type": "png",
            "genuine_invoice_verified": True,
            "permission_status": "approved",
            "prior_use_status": "none_confirmed",
            "supplier_group_id": "same-supplier",
            "layout_family_id": "same-layout",
            "visual_quality_band": "clear",
            "edge_cases": [],
            "annotation_status": "not_started",
            "annotation_a_path": None,
            "annotation_a_sha256": None,
            "annotation_b_path": None,
            "annotation_b_sha256": None,
            "adjudication_path": None,
            "adjudication_sha256": None,
            "scored_field_count": None,
            "line_item_count": None,
            "expected_proposal_count": None,
        }
        other = dict(
            base,
            document_id="doc-val",
            relative_path="private/val.png",
            sha256="d" * 64,
            split="validation",
        )
        manifest["selected_documents"] = [base, other]
        result = tools.validate_selection(manifest)
        self.assertTrue(
            any("supplier_group_id overlaps" in item["message"]
                for item in result["diagnostics"])
        )
        self.assertTrue(
            any("layout_family_id overlaps" in item["message"]
                for item in result["diagnostics"])
        )


if __name__ == "__main__":
    unittest.main()
