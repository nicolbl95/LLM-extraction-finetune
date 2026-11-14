"""PR103 Phase 3: the reference validator must refuse an untrustworthy annotation, not repair it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

spec = importlib.util.spec_from_file_location(
    "pr103_validate_reference", ROOT / "scripts" / "pr103_validate_reference.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def field(value, provenance="printed"):
    return {"value": value, "provenance": provenance}


def good_reference():
    return {
        "document": {
            "id": "fr-001", "source_file": "fr-001.png", "sha256": "x", "language": "fr",
            "region": "FR", "input_kind": "scanned_image", "annotation_date": "2026-11-08",
            "annotators": ["annotator-a"], "ocr_output_seen_before_annotation": False,
            "owner_permission": {"authorized_for_local_evaluation": True, "evidence": "written email"},
            "disagreements": [],
        },
        "reference": {
            "supplier_name": field(" Atelier Dupont"), "invoice_date": field("2025-03-04"),
            "currency": field("EUR"), "total_ht": field(100), "total_ttc": field(120),
            "line_items": [
                {"description": field("Prestation"), "quantity": field(2), "unit_price": field(50),
                 "total_ht": field(100), "is_product_or_service": True},
            ],
            "document_inconsistencies": [],
        },
        "freeze": {"reference_sha256": "abc", "frozen_before_pipeline_run": True},
    }


def problems(reference):
    return validator.validate(reference)


def test_a_complete_blind_reference_is_accepted():
    assert problems(good_reference()) == []


def test_seeing_ocr_output_before_annotating_is_refused():
    reference = good_reference()
    reference["document"]["ocr_output_seen_before_annotation"] = True
    assert any("blind" in problem for problem in problems(reference))


def test_missing_owner_permission_is_refused():
    reference = good_reference()
    reference["document"]["owner_permission"]["authorized_for_local_evaluation"] = False
    assert any("owner_permission" in problem for problem in problems(reference))


def test_a_printed_field_without_a_value_is_refused():
    reference = good_reference()
    reference["reference"]["total_ht"] = field(None, "printed")
    assert any("marked printed but has no value" in problem for problem in problems(reference))


def test_absent_and_illegible_are_distinct_and_carry_no_value():
    reference = good_reference()
    reference["reference"]["total_ht"] = field(None, "absent")
    reference["reference"]["total_ttc"] = field(None, "illegible")
    assert problems(reference) == []
    reference["reference"]["total_ttc"] = field(120, "illegible")
    assert any("marked illegible but carries a value" in problem for problem in problems(reference))


def test_an_unknown_provenance_label_is_refused():
    reference = good_reference()
    reference["reference"]["currency"] = field("EUR", "derived")
    assert any("provenance must be one of" in problem for problem in problems(reference))


def test_a_computed_total_presented_as_printed_is_refused():
    reference = good_reference()
    row = reference["reference"]["line_items"][0]
    row["quantity"], row["unit_price"], row["total_ht"] = field(3), field(50), field(150)
    row["total_ht"]["provenance"] = "derived"
    assert any("provenance must be one of" in problem for problem in problems(reference))


def test_a_printed_arithmetic_inconsistency_must_be_recorded_not_reconciled():
    reference = good_reference()
    row = reference["reference"]["line_items"][0]
    row["quantity"], row["unit_price"], row["total_ht"] = field(3), field(50), field(160)
    assert any("printed inconsistency" in problem for problem in problems(reference))
    reference["reference"]["document_inconsistencies"] = ["line 0: 3 x 50 = 150 but 160 is printed"]
    assert problems(reference) == []


def test_shipping_or_tax_may_not_be_a_product_row():
    reference = good_reference()
    reference["reference"]["line_items"].append(
        {"description": field("Livraison"), "quantity": field(None, "absent"),
         "unit_price": field(None, "absent"), "total_ht": field(10), "is_product_or_service": False})
    assert any("not a product/service row" in problem for problem in problems(reference))


def test_an_unresolved_disagreement_is_refused():
    reference = good_reference()
    reference["document"]["disagreements"] = [{"field": "total_ttc", "values": [120, 121]}]
    assert any("no resolution" in problem for problem in problems(reference))
    reference["document"]["disagreements"][0]["resolution"] = "re-read the print: 120"
    assert problems(reference) == []


def test_freezing_after_the_pipeline_run_is_refused():
    reference = good_reference()
    reference["freeze"]["frozen_before_pipeline_run"] = False
    assert any("frozen_before_pipeline_run" in problem for problem in problems(reference))


def test_an_empty_row_list_is_refused():
    reference = good_reference()
    reference["reference"]["line_items"] = []
    assert any("non-empty list" in problem for problem in problems(reference))


def test_a_mismatched_image_hash_is_refused(tmp_path):
    image = tmp_path / "invoice.png"
    image.write_bytes(b"not the recorded hash")
    reference = good_reference()
    reference["document"]["sha256"] = "0" * 64
    assert any("does not match the image" in problem for problem in problems_with_image(reference, image))


def problems_with_image(reference, image):
    return validator.validate(reference, image)


def test_the_template_itself_is_valid_json_and_declares_the_rules():
    import json

    template = json.loads((ROOT / "benchmark_prep" / "pr103-reference-template.json").read_text(encoding="utf-8"))
    assert template["allowed_provenance"] == ["printed", "absent", "illegible"]
    assert any("Never compute" in rule for rule in template["rules"])
    assert template["document"]["ocr_output_seen_before_annotation"] is False
