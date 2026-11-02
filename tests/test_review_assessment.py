import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.review_assessment import (  # noqa: E402
    POLICY_ID, POLICY_VERSION, ROW_LINE_MIN_MATCHES, assess, serialize_review, serialize_strict,
)


def evidence(lines, *, supplier=None, invoice_date=None, totals=None, rows=None):
    return {
        "text": "\n".join(lines),
        "lines": [{"text": line, "confidence": 95.0, "page": 0, "box": [10, 20, 300, 40]}
                  for line in lines],
        "supplier": supplier,
        "supplier_is_explicit": bool(supplier),
        "invoice_date": invoice_date,
        "totals": totals or {"total_ht": [], "total_ttc": [], "total_candidate": []},
        "rows": rows or [],
        "quality_issues": [],
    }


def raw(supplier=None, invoice_date=None, total_ht=None, total_ttc=None, line_items=None):
    return {
        "supplier_name": supplier,
        "invoice_date": invoice_date,
        "total_ht": total_ht,
        "total_ttc": total_ttc,
        "line_items": line_items or [],
    }


def row(description=None, quantity=None, unit_price=None, total_ht=None):
    return {"description": description, "quantity": quantity,
            "unit_price": unit_price, "total_ht": total_ht}


def strict(payload, statuses=None):
    return {"payload": payload, "statuses": statuses or {}}


def empty_strict(rows=0):
    return {
        "supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": None,
        "line_items": [row() for _ in range(rows)],
    }


def field(assessments, path):
    return assessments[path]


def test_policy_file_hash_and_constants_are_frozen():
    policy_path = ROOT / "experiments/review_assessment_policy.json"
    lock_path = ROOT / "experiments/review_assessment_policy.sha256"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    lock = lock_path.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == lock
    assert policy["policy_id"] == POLICY_ID
    assert policy["version"] == POLICY_VERSION
    assert policy["constants"]["row_line_min_matching_numbers"] == ROW_LINE_MIN_MATCHES
    assert policy["frozen_results"]["overall"]["proposed_correct"] == 64
    assert policy["frozen_results"]["overall"]["visible_correct"] == 72


def test_correct_row_blocked_by_zero_row_grounding_is_proposed_not_verified():
    ev = evidence(
        ["Prestige Services", "Date: December 11, 2026",
         "Rear Bumper Painting 1 USD 250 unit USD 250"],
        supplier="Prestige Services",
        totals={"total_ht": [], "total_ttc": [], "total_candidate": ["2214.50"]},
        rows=[],
    )
    raw_payload = raw("Prestige Services", "2026-12-11", 2214.5, 2535.95,
                      [row("Rear Bumper Painting", 1, 250, 250)])
    strict_payload = empty_strict(1)
    result = assess(ev, raw_payload, strict(strict_payload, {
        "supplier_name": "non lu", "invoice_date": "non lu",
        "total_ht": "non lu", "total_ttc": "non lu",
        "line_items[0].description": "non lu", "line_items[0].quantity": "non lu",
        "line_items[0].unit_price": "non lu", "line_items[0].total_ht": "non lu",
    }))
    assert result["supplier_name"].status == "proposed"
    assert result["invoice_date"].status == "proposed"
    assert all(result[f"line_items[0].{key}"].status == "proposed"
               for key in ("description", "quantity", "unit_price", "total_ht"))
    assert result["total_ht"].status == "rejected"
    assert result["total_ht"].reason == "ttc_as_ht"
    assert result["total_ttc"].status == "rejected"
    review = serialize_review(strict_payload, result)
    assert review["supplier_name"] is None
    assert review["line_items"] == strict_payload["line_items"]
    assert review["field_assessments"]["line_items[0].quantity"]["review_required"] is True
    assert "proposed" not in json.dumps(review["line_items"], ensure_ascii=False)


def test_quantity_unit_price_role_confusion_is_rejected():
    ev = evidence(["2 Lorem ipsum dolor sit amet 35,00 € 70,00 €"])
    raw_payload = raw(line_items=[row("Lorem ipsum dolor sit amet", 35, 35, 1225)])
    result = assess(ev, raw_payload, strict(empty_strict(1)))
    assert result["line_items[0].description"].status == "proposed"
    assert result["line_items[0].quantity"].status == "rejected"
    assert result["line_items[0].quantity"].reason == "quantity_equals_unit_price"
    assert result["line_items[0].unit_price"].status == "proposed"
    assert result["line_items[0].total_ht"].status == "rejected"
    assert result["line_items[0].total_ht"].reason in {"not_on_row_line", "row_line_not_found"}


def test_ttc_or_unqualified_total_is_not_proposed_as_ht():
    ev = evidence(["Total 100,00"], totals={"total_ht": [], "total_ttc": ["100.00"],
                                            "total_candidate": []})
    result = assess(ev, raw(total_ht=100, total_ttc=100), strict(empty_strict()))
    assert result["total_ht"].status == "rejected"
    assert result["total_ht"].reason == "ttc_as_ht"
    assert result["total_ttc"].status == "proposed"
    assert result["total_ttc"].reason == "labelled_ttc"

    unqualified = evidence(["Total 100,00"], totals={"total_ht": [], "total_ttc": [],
                                                     "total_candidate": ["100.00"]})
    result = assess(unqualified, raw(total_ht=100, total_ttc=100), strict(empty_strict()))
    assert result["total_ht"].status == "rejected"
    assert result["total_ht"].reason == "ttc_as_ht"
    assert result["total_ttc"].status == "proposed"
    assert result["total_ttc"].reason == "unqualified_total"
    assert result["total_ttc"].checks["label_conflict"] is True


def test_repeated_rows_keep_distinct_indices_and_order():
    ev = evidence(["Service A 1 10.00 10.00", "Service A 1 10.00 10.00"])
    raw_payload = raw(line_items=[row("Service A", 1, 10, 10), row("Service A", 1, 10, 10)])
    result = assess(ev, raw_payload, strict(empty_strict(2)))
    assert result["line_items[0].quantity"].status == "proposed"
    assert result["line_items[1].quantity"].status == "proposed"
    assert result["line_items[0].quantity"].row["index"] == 0
    assert result["line_items[1].quantity"].row["index"] == 1
    assert result["line_items[0].quantity"].row["order"] == 0
    assert result["line_items[1].quantity"].row["order"] == 1


def test_incorrect_row_attribution_is_rejected():
    ev = evidence([
        "Séance photo reportage mariage (6h) - Lyon 1,00 350,00 350,00",
        "Retouche et livraison de 150 images HD 1,00 180,00 180,00",
    ])
    raw_payload = raw(line_items=[row(), row("Séance photo reportage mariage (6h) - Lyon", 1, 180, 180)])
    result = assess(ev, raw_payload, strict(empty_strict(2)))
    assert result["line_items[1].description"].status == "rejected"
    assert result["line_items[1].description"].reason == "row_attribution_conflict"
    assert result["line_items[1].unit_price"].status == "proposed"
    assert result["line_items[1].total_ht"].status == "proposed"


def test_incomplete_description_is_rejected():
    ev = evidence(["Pepe Jeans Tapered fit Mid rise - Blue 1.00 24.99 24.99"])
    raw_payload = raw(line_items=[row("Pepe Jeans", 1, 24.99, 24.99)])
    result = assess(ev, raw_payload, strict(empty_strict(1)))
    assert result["line_items[0].description"].status == "rejected"
    assert result["line_items[0].description"].reason == "incomplete_description"
    assert result["line_items[0].quantity"].status == "proposed"
    assert result["line_items[0].unit_price"].status == "proposed"
    assert result["line_items[0].total_ht"].status == "proposed"


def test_missing_printed_versus_absent_is_evidence_based():
    ev = evidence(["Acme Ltd"], supplier="Acme Ltd", invoice_date=None, rows=[])
    result = assess(ev, raw(), strict(empty_strict(1)))
    assert result["supplier_name"].status == "missing_printed"
    assert result["invoice_date"].status == "absent"
    assert result["line_items[0].description"].status == "absent"
    assert result["line_items[0].quantity"].status == "absent"

    with_row = evidence(["Acme Ltd"], supplier="Acme Ltd", rows=[
        {"description": "Service", "cells": {"quantity": [1], "unit_price": [10], "total_ht": [10]}}
    ])
    result = assess(with_row, raw(line_items=[row()]), strict(empty_strict(1)))
    assert result["line_items[0].description"].status == "missing_printed"
    assert result["line_items[0].quantity"].status == "missing_printed"
    assert result["line_items[0].unit_price"].status == "missing_printed"
    assert result["line_items[0].total_ht"].status == "missing_printed"


def test_legitimate_quantity_equal_unit_price_follows_frozen_conservative_rule():
    ev = evidence(["Item 10 10.00 100.00"])
    raw_payload = raw(line_items=[row("Item", 10, 10, 100)])
    result = assess(ev, raw_payload, strict(empty_strict(1)))
    assert result["line_items[0].quantity"].status == "rejected"
    assert result["line_items[0].quantity"].reason == "quantity_equals_unit_price"
    assert result["line_items[0].unit_price"].status == "proposed"
    assert result["line_items[0].total_ht"].status == "proposed"


def test_unsupported_and_contradicted_supplier_are_rejected():
    ev = evidence(["FACTURE", "ENTREPRISE DU CLIENT"])
    result = assess(ev, raw("SERVICES"), strict(empty_strict()))
    assert result["supplier_name"].status == "rejected"
    assert result["supplier_name"].reason == "supplier_not_found"

    ev = evidence(["Acme Ltd"], supplier="Acme Ltd")
    result = assess(ev, raw("Other Company"), strict(empty_strict()))
    assert result["supplier_name"].status == "rejected"
    assert result["supplier_name"].reason == "contradicted_supplier"


def test_strict_parity_primary_immutability_and_review_conflict():
    strict_payload = {
        "supplier_name": "Mon Entreprise", "invoice_date": "2020-08-26",
        "total_ht": 135.0, "total_ttc": None,
        "line_items": [row("Thé vert biologique", 5, 15, None)],
    }
    statuses = {
        "supplier_name": "vérifié", "invoice_date": "vérifié",
        "total_ht": "vérifié", "total_ttc": "non lu",
        "line_items[0].description": "non lu", "line_items[0].quantity": "non lu",
        "line_items[0].unit_price": "non lu", "line_items[0].total_ht": "non lu",
    }
    raw_payload = raw("Nombre de colis", "2020-08-26", 135.0, 162.0,
                      [row("Thé vert biologique", 5, 15, 75)])
    ev = evidence(["Mon Entreprise", "26/08/2020"], supplier="Mon Entreprise",
                  invoice_date="2020-08-26")
    strict_copy = copy.deepcopy(strict_payload)
    raw_copy = copy.deepcopy(raw_payload)
    ev_copy = copy.deepcopy(ev)
    result = assess(ev, raw_payload, strict(strict_payload, statuses))
    assert result["supplier_name"].status == "verified"
    assert result["supplier_name"].verification_conflict is True
    assert result["supplier_name"].review_required is True
    assert result["supplier_name"].value == "Mon Entreprise"
    assert serialize_strict(strict_payload) == strict_copy
    review = serialize_review(strict_payload, result)
    assert review["supplier_name"] == strict_payload["supplier_name"]
    assert review["line_items"] == strict_payload["line_items"]
    assert review["mode"] == "review_assisted"
    assert review["schema_version"] == "2.1"
    assert review["review_summary"]["verification_conflicts"] == 1
    assert strict_payload == strict_copy
    assert raw_payload == raw_copy
    assert ev == ev_copy


def test_strict_output_has_no_new_keys():
    payload = empty_strict(1)
    result = assess(evidence([]), raw(), strict(payload))
    assert serialize_strict(payload) == payload
    assert set(serialize_strict(payload)) == set(payload)
    review = serialize_review(payload, result)
    assert review["mode"] == "review_assisted"
    assert "field_assessments" in review
    assert review["supplier_name"] is None
    assert review["line_items"] == payload["line_items"]
