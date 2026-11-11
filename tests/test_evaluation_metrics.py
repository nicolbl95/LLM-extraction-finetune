"""PR102 Priority 2: one deterministic test per metric of the unified evaluator.

Every assertion is about a verdict the evaluator must reach, not about a value appearing in text. The
fixtures are synthetic; no invoice data is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import evaluation_metrics as metrics  # noqa: E402


def row(description="Widget", quantity=2, unit_price=10, total_ht=20):
    return {"description": description, "quantity": quantity, "unit_price": unit_price, "total_ht": total_ht}


def payload(*rows, **header):
    return {"line_items": list(rows), **header}


def reference(*rows, **header):
    return {"line_items": list(rows), **header}


def test_expected_rows_come_from_the_reference():
    result = metrics.evaluate(payload(row()), reference(row(), row("Gadget", 1, 5, 5)))
    assert result["rows"]["expected"] == 2
    assert result["rows"]["exported"] == 1


def test_a_description_identified_row_is_correctly_matched():
    result = metrics.evaluate(payload(row()), reference(row()))
    assert result["rows"]["correctly_matched"] == 1
    assert result["rows"]["matched_with_field_errors"] == 0
    assert result["pairs"][0]["identity"] == metrics.IDENTITY_DESCRIPTION


def test_a_missing_row_is_counted_and_its_fields_are_missing_not_incorrect():
    result = metrics.evaluate(payload(row()), reference(row(), row("Gadget", 1, 5, 5)))
    assert result["rows"]["missing"] == 1
    assert result["rows"]["missing_indices"] == [1]
    assert result["fields"]["missing_in_unrecovered_rows"] == 4
    assert result["fields"]["incorrect"] == 0


def test_a_duplicate_exported_row_is_not_matched_twice():
    result = metrics.evaluate(payload(row(), row()), reference(row()))
    assert result["rows"]["matched_total"] == 1
    assert result["rows"]["duplicate"] == 1
    assert result["rows"]["duplicate_indices"] == [1]
    assert result["fields"]["correct"] == 4


def test_an_exported_row_with_no_reference_counterpart_is_unsupported():
    result = metrics.evaluate(payload(row("Invented", 9, 9, 81)), reference(row()))
    assert result["rows"]["unsupported"] == 1
    assert result["rows"]["matched_total"] == 0
    assert result["fields"]["correct"] == 0


def test_correct_incorrect_and_missing_field_values_are_separate():
    gold = row("Widget", 2, 10, 20)
    export = {"description": "Widget", "quantity": 3, "unit_price": None, "total_ht": 20}
    result = metrics.evaluate(payload(export), reference(gold))
    assert result["fields"]["by_field"]["description"]["correct"] == 1
    assert result["fields"]["by_field"]["quantity"]["incorrect"] == 1
    assert result["fields"]["by_field"]["unit_price"]["missing"] == 1
    assert result["fields"]["by_field"]["total_ht"]["correct"] == 1
    assert result["fields"]["correct"] == 2
    assert result["fields"]["incorrect"] == 1
    assert result["fields"]["missing"] == 1


def test_a_null_reference_field_is_unverified_and_never_counted():
    gold = {"description": "Widget", "quantity": None, "unit_price": 10, "total_ht": 20}
    result = metrics.evaluate(payload(row()), reference(gold))
    assert result["fields"]["by_field"]["quantity"]["unverified"] == 1
    assert result["fields"]["unverified"] == 1
    assert result["fields"]["incorrect"] == 0


def test_incorrect_monetary_exports_are_counted_separately():
    result = metrics.evaluate(payload(row("Widget", 2, 10, 99)), reference(row()))
    assert result["fields"]["incorrect_monetary"] == 1
    assert result["fields"]["incorrect_monetary_in_matched_rows"] == 1
    assert result["fields"]["incorrect_association"] == 0


def test_a_correct_amount_on_the_wrong_product_is_an_incorrect_association():
    gold_a, gold_b = row("Widget", 2, 10, 20), row("Gadget", 1, 5, 5)
    # The exported row carries Gadget's amounts but claims to be Widget.
    export = {"description": "Widget", "quantity": 1, "unit_price": 5, "total_ht": 5}
    result = metrics.evaluate(payload(export), reference(gold_a, gold_b))
    assert result["rows"]["misassociated"] == 1
    assert result["rows"]["correctly_matched"] == 0
    assert result["fields"]["incorrect_association"] == 3
    assert result["fields"]["incorrect_monetary"] == 3
    assert result["fields"]["correct"] == 0
    assert result["rejected_amount_agreements"]


def test_a_blank_description_with_a_unique_triple_is_matched_on_amounts_and_flagged():
    gold_a, gold_b = row("Widget", 2, 10, 20), row("Gadget", 1, 5, 5)
    export = {"description": None, "quantity": 1, "unit_price": 5, "total_ht": 5}
    result = metrics.evaluate(payload(export), reference(gold_a, gold_b))
    assert result["rows"]["matched_total"] == 1
    assert result["rows"]["identity_by_amounts_only"] == 1
    assert result["pairs"][0]["gold_index"] == 1
    assert result["fields"]["by_field"]["description"]["missing"] == 1
    assert result["fields"]["incorrect_association"] == 0


def test_an_ambiguous_triple_with_a_blank_description_is_not_matched():
    gold_a, gold_b = row("Widget", 2, 10, 20), row("Gadget", 2, 10, 20)
    export = {"description": None, "quantity": 2, "unit_price": 10, "total_ht": 20}
    result = metrics.evaluate(payload(export), reference(gold_a, gold_b))
    assert result["rows"]["matched_total"] == 0
    assert result["rows"]["misassociated"] == 1
    assert result["rows"]["misassociated_detail"][0]["amounts_belong_to_gold_rows"] == [0, 1]


def test_an_incorrect_supplier_is_never_scored_as_a_missing_one():
    gold = reference(row(), supplier_name="Real Supplier")
    wrong = metrics.evaluate(payload(row(), supplier_name="Wrong Supplier"), gold)
    absent = metrics.evaluate(payload(row(), supplier_name=None), gold)
    assert wrong["header"]["supplier_name"] == metrics.INCORRECT
    assert absent["header"]["supplier_name"] == metrics.MISSING
    assert wrong["header_summary"]["incorrect_supplier"] is True
    assert absent["header_summary"]["incorrect_supplier"] is False
    assert wrong["header_summary"]["incorrect"] == 1
    assert absent["header_summary"]["missing"] == 1


def test_an_incorrect_date_is_never_scored_as_a_missing_one():
    gold = reference(row(), invoice_date="2025-09-01")
    wrong = metrics.evaluate(payload(row(), invoice_date="2025-01-09"), gold)
    absent = metrics.evaluate(payload(row(), invoice_date=None), gold)
    assert wrong["header"]["invoice_date"] == metrics.INCORRECT
    assert absent["header"]["invoice_date"] == metrics.MISSING
    assert wrong["header_summary"]["incorrect_date"] is True
    assert absent["header_summary"]["incorrect_date"] is False


def test_the_positional_view_is_preserved_and_differs_from_the_association_view():
    gold = [row("First", 5, 40, 200), row("Second", 2, 40, 80)]
    export = [{"description": None, "quantity": 2, "unit_price": 40, "total_ht": 80}]
    result = metrics.evaluate(payload(*export), reference(*gold))
    # Association-aware: the amounts belong to the second reference row, uniquely.
    assert result["rows"]["matched_total"] == 1
    assert result["pairs"][0]["gold_index"] == 1
    assert result["fields"]["incorrect_monetary"] == 0
    # Positional: the same export sits in the first row's slot and looks wrong.
    assert result["positional"]["incorrect_monetary_cells"] == 2
    assert result["rows"]["missing"] == 1


def test_numeric_formatting_does_not_create_a_false_incorrect():
    gold = row("Widget", "2", "10.00", "20,00")
    result = metrics.evaluate(payload(row("Widget", 2, 10.0, 20)), reference(gold))
    assert result["fields"]["incorrect"] == 0
    assert result["fields"]["correct"] == 4


def test_the_evaluator_is_deterministic():
    gold = reference(row("Widget", 2, 10, 20), row("Gadget", 1, 5, 5), supplier_name="S")
    export = payload(row("Widget", 2, 10, 20), row("Gadget", 1, 5, 5), supplier_name="S")
    first = metrics.evaluate(export, gold)
    second = metrics.evaluate(export, gold)
    assert first == second


def test_aggregate_sums_documents_without_merging_header_verdicts():
    gold = reference(row(), supplier_name="S", invoice_date="2025-09-01")
    good = metrics.evaluate(payload(row(), supplier_name="S", invoice_date="2025-09-01"), gold)
    bad = metrics.evaluate(payload(row(), supplier_name="X", invoice_date=None), gold)
    total = metrics.aggregate([good, bad])
    assert total["documents"] == 2
    assert total["rows"]["expected"] == 2
    assert total["fields"]["correct"] == 8
    assert total["header"]["correct"] == 2
    assert total["header"]["incorrect"] == 1
    assert total["header"]["missing"] == 1
    assert total["incorrect_supplier_documents"] == 1
    assert total["incorrect_date_documents"] == 0


def test_two_unique_printed_values_identify_a_row_with_a_blank_description():
    gold = [row("First", 5, 40, 200), row("Second", 6.5, 40, 260), row("Third", 2, 40, 80)]
    export = {"description": None, "quantity": 6.5, "unit_price": None, "total_ht": 260}
    result = metrics.evaluate(payload(export), reference(*gold))
    assert result["rows"]["matched_total"] == 1
    assert result["pairs"][0]["gold_index"] == 1
    assert result["pairs"][0]["identity"] == metrics.IDENTITY_PARTIAL
    assert result["fields"]["correct"] == 2
    assert result["fields"]["missing_in_matched_rows"] == 2
    assert result["fields"]["incorrect"] == 0


def test_a_lone_total_cannot_identify_a_product_row():
    gold = [row("First", 5, 40, 200), row("Second", 6.5, 40, 260)]
    export = {"description": None, "quantity": None, "unit_price": None, "total_ht": 260}
    result = metrics.evaluate(payload(export), reference(*gold))
    assert result["rows"]["matched_total"] == 0
    assert result["rows"]["unsupported"] == 1
    assert result["fields"]["correct"] == 0


def test_two_values_shared_by_two_reference_rows_are_ambiguous_and_not_matched():
    gold = [row("First", 2, 40, 80), row("Second", 2, 25, 80)]
    export = {"description": None, "quantity": 2, "unit_price": None, "total_ht": 80}
    result = metrics.evaluate(payload(export), reference(*gold))
    assert result["rows"]["matched_total"] == 0
    assert result["rows"]["unsupported"] == 1


def test_empty_export_reports_every_reference_row_as_missing():
    result = metrics.evaluate(payload(), reference(row(), row("Gadget", 1, 5, 5)))
    assert result["rows"]["expected"] == 2
    assert result["rows"]["missing"] == 2
    assert result["fields"]["missing"] == 8
    assert result["fields"]["incorrect"] == 0
    assert result["fields"]["incorrect_monetary"] == 0
