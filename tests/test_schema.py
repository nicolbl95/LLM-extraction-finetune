import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from schema import Invoice, LineItem


class SchemaTests(unittest.TestCase):
    def line(self, **overrides):
        return {
            "description": "Service", "quantity": 2.5, "unit_price": 5,
            "total_ht": 12.5, **overrides,
        }

    def invoice(self, **overrides):
        return {
            "supplier_name": "Test supplier", "invoice_date": "2026-10-21",
            "total_ht": 12.5, "total_ttc": 15,
            "line_items": [self.line()], **overrides,
        }

    def test_decimal_quantities_and_json_number_types_are_preserved(self):
        payload = Invoice.model_validate(self.invoice()).model_dump(mode="json")
        self.assertEqual(payload["line_items"][0]["quantity"], 2.5)
        self.assertIsInstance(payload["line_items"][0]["quantity"], float)
        self.assertEqual(payload["invoice_date"], "2026-10-21")
        line = LineItem.model_validate(self.line(quantity=0.3, unit_price=0.1, total_ht=0.03))
        self.assertEqual(line.total_ht, 0.03)

    def test_decimal_point_hallucination_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, r"quantity \(25\.0\) \* unit_price \(5\.0\)"):
            Invoice.model_validate(self.invoice(line_items=[self.line(quantity=25)]))

    def test_hallucinated_quantity_error_preserves_original_operands(self):
        with self.assertRaises(ValidationError) as validation:
            LineItem.model_validate(self.line(quantity=142, unit_price=95.07, total_ht=684.50))
        self.assertEqual(
            validation.exception.errors()[0]["msg"],
            "Value error, Incohérence sur la ligne 'Service' : "
            "quantity (142.0) * unit_price (95.07) = 13499.94, "
            "mais total_ht indiqué = 684.5. Écart absolu : 12815.44. Tolérance : 0.02.",
        )

    def test_invoice_error_reports_calculated_sum_expected_total_and_difference(self):
        with self.assertRaises(ValidationError) as validation:
            Invoice.model_validate(self.invoice(total_ht=10))
        self.assertEqual(
            validation.exception.errors()[0]["msg"],
            "Value error, Incohérence de facture : somme des total_ht des lignes = 12.5, "
            "mais total_ht global attendu = 10.0. Écart absolu : 2.5. Tolérance : 0.02.",
        )

    def test_line_tolerance_accepts_exactly_two_cents_without_correction(self):
        for total in (12.48, 12.5, 12.52):
            with self.subTest(total=total):
                self.assertEqual(LineItem.model_validate(self.line(total_ht=total)).total_ht, total)
        for total in (12.479999, 12.520001, 12.47, 12.53):
            with self.subTest(total=total):
                with self.assertRaisesRegex(ValidationError, "Incohérence sur la ligne"):
                    LineItem.model_validate(self.line(total_ht=total))

    def test_half_up_rounding_matches_dataset_at_tolerance_boundary(self):
        line = LineItem.model_validate(self.line(quantity=0.5, unit_price=0.05, total_ht=0.05))
        self.assertEqual(line.total_ht, 0.05)
        with self.assertRaises(ValidationError):
            LineItem.model_validate(self.line(quantity=0.5, unit_price=0.05, total_ht=0.050001))

    def test_invoice_sum_uses_all_lines_and_exact_decimal_tolerance(self):
        items = [self.line(), self.line(quantity=0.3, unit_price=0.1, total_ht=0.03)]
        for total in (12.51, 12.53, 12.55):
            with self.subTest(total=total):
                invoice = Invoice.model_validate(self.invoice(line_items=items, total_ht=total))
                self.assertEqual(invoice.total_ht, total)
        for total in (12.509999, 12.550001, 12.5, 12.56):
            with self.subTest(total=total):
                with self.assertRaisesRegex(ValidationError, "somme des total_ht"):
                    Invoice.model_validate(self.invoice(line_items=items, total_ht=total))

    def test_nonzero_invoice_without_lines_is_rejected(self):
        payload = self.invoice(line_items=[])
        with self.assertRaisesRegex(ValidationError, "somme des total_ht"):
            Invoice.model_validate(payload)
        del payload["line_items"]
        with self.assertRaises(ValidationError):
            Invoice.model_validate(payload)
        payload["total_ht"] = 0
        self.assertEqual(Invoice.model_validate(payload).line_items, [])

    def test_negative_credit_note_amounts_remain_supported(self):
        invoice = Invoice.model_validate(self.invoice(
            total_ht=-12.5, total_ttc=-15, line_items=[self.line(quantity=-2.5, total_ht=-12.5)],
        ))
        self.assertEqual(invoice.total_ht, -12.5)

    def test_nonfinite_values_are_rejected_before_arithmetic(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for field in ("quantity", "unit_price", "total_ht"):
                with self.subTest(value=value, field=field):
                    with self.assertRaises(ValidationError):
                        LineItem.model_validate(self.line(**{field: value}))
            for field in ("total_ht", "total_ttc"):
                with self.subTest(value=value, field=field):
                    with self.assertRaises(ValidationError):
                        Invoice.model_validate(self.invoice(**{field: value}))

    def test_unrepresentable_product_is_a_validation_error(self):
        with self.assertRaisesRegex(ValidationError, "non representable au centime"):
            LineItem.model_validate(self.line(quantity=1e308, unit_price=1e308))


if __name__ == "__main__":
    unittest.main()
