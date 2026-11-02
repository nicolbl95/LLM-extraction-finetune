import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import invoice_review_edit as edit  # noqa: E402


STRICT = {
    "supplier_name": "Atelier Exemple",
    "invoice_date": "2026-03-12",
    "total_ht": 12.0,
    "total_ttc": 14.4,
    "line_items": [
        {"description": "Pose", "quantity": 2, "unit_price": 3, "total_ht": 6},
        {"description": "Pose", "quantity": 2, "unit_price": 3, "total_ht": 6},
    ],
}
MODEL = {
    **STRICT,
    "supplier_name": "Atelier Exemple SARL",
    "line_items": [
        {"description": "Pose", "quantity": 2, "unit_price": 3, "total_ht": 6},
        {"description": "Pose", "quantity": 2, "unit_price": 3, "total_ht": 6},
    ],
}
REVIEW = {
    "field_assessments": {
        "supplier_name": {
            "status": "proposed", "value": "Atelier Exemple SARL",
            "evidence": {"kind": "ocr_supplier", "text": "Atelier Exemple SARL"},
        },
        "line_items[0].quantity": {
            "status": "verified", "evidence": {"kind": "ocr_line", "text": "2"},
        },
        "line_items[1].quantity": {
            "status": "verified", "evidence": {"kind": "ocr_line", "text": "2"},
        },
    },
    "line_items": [{}, {}],
}


class InvoiceReviewEditTests(unittest.TestCase):
    def setUp(self):
        self.strict = copy.deepcopy(STRICT)
        self.review = copy.deepcopy(REVIEW)
        self.model = copy.deepcopy(MODEL)
        self.state = edit.initialize_edit_state(self.strict, self.review, self.model)

    def row_field(self, row_index, field):
        row_id = self.state["row_order"][row_index]
        return f"row:{row_id}:{field}"

    def test_initialization_preserves_all_original_snapshots(self):
        before = copy.deepcopy((self.strict, self.review, self.model))
        edit.initialize_edit_state(self.strict, self.review, self.model)
        self.assertEqual((self.strict, self.review, self.model), before)
        self.assertEqual(self.state["original_strict"], self.strict)
        self.assertEqual(self.state["original_review"], self.review)
        self.assertEqual(self.state["original_model"], self.model)
        self.assertNotIn("_review_row_id", self.state["original_model"]["line_items"][0])

    def test_accept_requires_separate_confirmation_and_keeps_raw_proposal(self):
        accepted = edit.accept_proposal(self.state, "supplier_name")
        self.assertEqual(accepted["draft"]["supplier_name"], "Atelier Exemple SARL")
        self.assertEqual(accepted["decisions"]["supplier_name"], "accepted")
        self.assertFalse(accepted["confirmed"]["supplier_name"])
        self.assertEqual(accepted["original_model"], self.model)
        confirmed = edit.confirm_field(accepted, "supplier_name")
        self.assertTrue(confirmed["confirmed"]["supplier_name"])
        view = edit.confirmed_view(confirmed)
        self.assertEqual(view["fields"]["supplier_name"]["status"], "human_confirmed")
        self.assertFalse(view["finalized"])

    def test_rejected_proposal_stays_rejected_after_manual_correction(self):
        rejected = edit.reject_proposal(self.state, "supplier_name")
        self.assertEqual(rejected["draft"]["supplier_name"], "Atelier Exemple")
        self.assertEqual(rejected["decisions"]["supplier_name"], "rejected")
        with self.assertRaises(edit.EditStateError):
            edit.confirm_field(rejected, "supplier_name")
        corrected = edit.apply_correction(rejected, "supplier_name", "Atelier corrigé")
        confirmed = edit.confirm_field(corrected, "supplier_name")
        item = edit.confirmed_view(confirmed)["fields"]["supplier_name"]
        self.assertEqual(item["proposal_decision"], "rejected")
        self.assertEqual(item["value_origin"], "human_correction")
        self.assertEqual(item["value"], "Atelier corrigé")

    def test_manual_edit_validates_types_dates_and_formats(self):
        with self.assertRaises(edit.EditStateError):
            edit.apply_correction(self.state, "total_ht", True)
        with self.assertRaises(edit.EditStateError):
            edit.apply_correction(self.state, "invoice_date", "12/03/2026")
        with self.assertRaises(edit.EditStateError):
            edit.apply_correction(self.state, "invoice_date", 20260312)
        with self.assertRaises(edit.EditStateError):
            edit.apply_correction(self.state, "supplier_name", 123)
        with self.assertRaises(edit.EditStateError):
            edit.apply_correction(self.state, "total_ht", "1,234.50")
        changed = edit.apply_correction(self.state, "invoice_date", "2026-03-13")
        self.assertEqual(changed["draft"]["invoice_date"], "2026-03-13")

    def test_duplicate_rows_have_distinct_ids_and_keep_original_evidence_association(self):
        first_id, second_id = self.state["row_order"]
        self.assertNotEqual(first_id, second_id)
        first = edit.field_details(self.state, f"row:{first_id}:quantity")
        second = edit.field_details(self.state, f"row:{second_id}:quantity")
        self.assertEqual(first["assessment"]["evidence"]["text"], "2")
        self.assertEqual(second["assessment"]["evidence"]["text"], "2")
        self.assertEqual(self.state["rows"][first_id]["original_index"], 0)
        self.assertEqual(self.state["rows"][second_id]["original_index"], 1)

    def test_duplicate_rows_can_be_edited_independently(self):
        first_id, second_id = self.state["row_order"]
        first_key = f"row:{first_id}:description"
        second_key = f"row:{second_id}:description"
        changed = edit.apply_correction(self.state, first_key, "Pose principale")
        self.assertEqual(changed["draft"][first_key], "Pose principale")
        self.assertEqual(changed["draft"][second_key], "Pose")
        self.assertEqual(self.state["draft"][first_key], "Pose")

    def test_missing_fields_and_human_added_rows_remain_incomplete(self):
        partial = edit.initialize_edit_state(
            {"supplier_name": None, "invoice_date": None, "total_ht": None,
             "total_ttc": None, "line_items": []},
            {"field_assessments": {}, "line_items": []},
            {},
        )
        self.assertFalse(edit.confirmed_view(partial)["finalized"])
        self.assertIn("supplier_name", edit.validate_draft(partial)["incomplete"])
        expanded = edit.add_line_item(partial)
        row_id = expanded["row_order"][0]
        self.assertEqual(expanded["rows"][row_id]["origin"], "human")
        self.assertIn(f"row:{row_id}:description", edit.validate_draft(expanded)["incomplete"])
        self.assertFalse(edit.confirmed_view(expanded)["finalized"])

    def test_arithmetic_validation_rejects_line_and_invoice_inconsistency(self):
        row_total = self.row_field(0, "total_ht")
        bad_line = edit.apply_correction(self.state, row_total, 5)
        self.assertIn(row_total, edit.validate_draft(bad_line)["errors"])
        with self.assertRaises(edit.EditStateError):
            edit.confirm_field(bad_line, row_total)

        subtotal = edit.apply_correction(self.state, "total_ht", 7)
        errors = edit.validate_draft(subtotal)["errors"]
        self.assertIn("total_ht", errors)
        self.assertIn(row_total, errors)

    def test_changing_quantity_invalidates_dependent_confirmations(self):
        quantity = self.row_field(0, "quantity")
        unit_price = self.row_field(0, "unit_price")
        row_total = self.row_field(0, "total_ht")
        state = self.state
        for field_id in ("supplier_name", quantity, unit_price, row_total, "total_ht"):
            state = edit.confirm_field(state, field_id)
        self.assertTrue(state["confirmed"][row_total])
        self.assertTrue(state["confirmed"]["total_ht"])
        changed = edit.apply_correction(state, quantity, 3)
        self.assertFalse(changed["confirmed"][quantity])
        self.assertFalse(changed["confirmed"][row_total])
        self.assertFalse(changed["confirmed"]["total_ht"])
        self.assertFalse(edit.confirmed_view(changed)["finalized"])

    def test_adding_row_invalidates_document_subtotal_confirmation(self):
        state = edit.confirm_field(self.state, "total_ht")
        expanded = edit.add_line_item(state)
        self.assertFalse(expanded["confirmed"]["total_ht"])

    def test_invalid_identifiers_and_malformed_states_are_rejected(self):
        with self.assertRaises(edit.EditStateError):
            edit.field_details(self.state, "row:someone-elses-row:quantity")
        malformed = copy.deepcopy(self.state)
        malformed["confirmed"]["supplier_name"] = "yes"
        with self.assertRaises(edit.EditStateError):
            edit.confirmed_view(malformed)
        malformed = copy.deepcopy(self.state)
        malformed["fields"]["unknown"] = {}
        with self.assertRaises(edit.EditStateError):
            edit.validate_draft(malformed)

    def test_session_states_are_independent(self):
        other = edit.initialize_edit_state(self.strict, self.review, self.model)
        changed = edit.apply_correction(self.state, "supplier_name", "Session A")
        self.assertNotEqual(changed["draft"]["supplier_name"], other["draft"]["supplier_name"])
        self.assertNotEqual(changed["row_order"], other["row_order"])

    def test_human_confirmed_status_is_bound_to_the_confirmed_value(self):
        confirmed = edit.confirm_field(self.state, "supplier_name")
        self.assertEqual(
            edit.confirmed_view(confirmed)["fields"]["supplier_name"]["status"],
            "human_confirmed",
        )
        forged = copy.deepcopy(confirmed)
        forged["draft"]["supplier_name"] = "Client forgery"
        with self.assertRaises(edit.EditStateError):
            edit.confirmed_view(forged)
        forged = copy.deepcopy(self.state)
        forged["confirmed"]["supplier_name"] = True
        with self.assertRaises(edit.EditStateError):
            edit.confirmed_view(forged)

    def test_flag_is_opt_in(self):
        self.assertFalse(edit.edit_ui_enabled({}))
        self.assertFalse(edit.edit_ui_enabled({"INVOICE_REVIEW_EDIT": "0"}))
        self.assertTrue(edit.edit_ui_enabled({"INVOICE_REVIEW_EDIT": "1"}))


if __name__ == "__main__":
    unittest.main()
