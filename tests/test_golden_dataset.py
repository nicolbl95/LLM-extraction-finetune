import copy
import json
import unittest
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from unittest.mock import MagicMock, patch

from app import main
from evaluate_finetuned import parse_object

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_dataset"
CENT = Decimal("0.01")


class GoldenDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = sorted(GOLDEN_DIR.glob("*.json"))
        if not paths:
            raise AssertionError("Golden dataset fixtures are missing")
        cls.cases = [parse_object(path.read_text(encoding="utf-8")) for path in paths]

    def test_fixture_coverage_and_unique_ids(self):
        self.assertGreaterEqual(len(self.cases), 4)
        self.assertEqual(len({case["id"] for case in self.cases}), len(self.cases))
        self.assertEqual({case["locale"] for case in self.cases}, {"fr-FR", "en-GB", "en-US"})
        tags = {tag for case in self.cases for tag in case["tags"]}
        self.assertTrue({
            "decimal_quantities", "special_characters", "discounts", "multiple_taxes",
        } <= tags)
        quantities = {
            item["quantity"] for case in self.cases for item in case["expected"]["line_items"]
        }
        self.assertTrue({7.2, 4.2} <= quantities)
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["invoice_text"].strip())
                self.assertIsInstance(case["response_json"], str)

    def test_frozen_responses_match_golden_outputs_exactly(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                model = MagicMock()
                model.predict.return_value = {"raw": case["response_json"], "finish_reason": "stop"}
                with patch.object(main, "get_extractor", return_value=model):
                    actual = main.extract_invoice(case["invoice_text"])
                self.assertEqual(actual, case["expected"])
                model.predict.assert_called_once()
                self.assertEqual(model.predict.call_args.args[0]["input"], case["invoice_text"])
                messages = model.predict.call_args.kwargs["messages"]
                self.assertIn(case["invoice_text"], messages[-1]["content"])

    def test_expected_tax_and_discount_arithmetic_is_exact(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                expected = case["expected"]
                base_sum = sum((Decimal(tax["base"]) for tax in case["taxes"]), Decimal(0))
                tax_sum = sum((Decimal(tax["amount"]) for tax in case["taxes"]), Decimal(0))
                self.assertEqual(base_sum, Decimal(str(expected["total_ht"])))
                self.assertEqual(base_sum + tax_sum, Decimal(str(expected["total_ttc"])))
                for tax in case["taxes"]:
                    calculated = (Decimal(tax["base"]) * Decimal(tax["rate"]) / 100).quantize(
                        CENT, rounding=ROUND_HALF_UP,
                    )
                    self.assertEqual(calculated, Decimal(tax["amount"]))
                for discount in case["discounts"]:
                    net_price = (
                        Decimal(discount["list_unit_price"])
                        * (100 - Decimal(discount["percent"])) / 100
                    ).quantize(CENT, rounding=ROUND_HALF_UP)
                    item = expected["line_items"][discount["line_index"]]
                    self.assertEqual(net_price, Decimal(str(item["unit_price"])))

    def test_decimal_quantity_hallucinations_are_rejected(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                invalid = copy.deepcopy(case["expected"])
                item = invalid["line_items"][0]
                item["quantity"] = 142.0 if item["quantity"] == 7.2 else item["quantity"] * 10
                with self.assertRaises(main.ValidationError):
                    main.validate_invoice_json(json.dumps(invalid, ensure_ascii=False))

    def test_self_correction_recovers_exact_golden_outputs(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                invalid = copy.deepcopy(case["expected"])
                invalid["line_items"][0]["quantity"] *= 10
                model = MagicMock()
                model.predict.side_effect = [
                    {"raw": json.dumps(invalid, ensure_ascii=False), "finish_reason": "stop"},
                    {"raw": case["response_json"], "finish_reason": "stop"},
                ]
                with patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(main.LOGGER, level="WARNING"):
                    actual = main.extract_invoice(case["invoice_text"])
                self.assertEqual(actual, case["expected"])
                self.assertEqual(model.predict.call_count, 2)

    def test_exact_golden_comparison_detects_errors_even_when_schema_accepts_them(self):
        for case in self.cases:
            for mutation in ("coherent_wrong_quantity", "one_cent_difference"):
                with self.subTest(case=case["id"], mutation=mutation):
                    changed = copy.deepcopy(case["expected"])
                    item = changed["line_items"][0]
                    if mutation == "coherent_wrong_quantity":
                        item["quantity"] *= 10
                        item["total_ht"] = float(
                            (Decimal(str(item["quantity"])) * Decimal(str(item["unit_price"]))).quantize(
                                CENT, rounding=ROUND_HALF_UP,
                            )
                        )
                    else:
                        item["total_ht"] = float(Decimal(str(item["total_ht"])) + CENT)
                    changed["total_ht"] = float(sum(
                        (Decimal(str(line["total_ht"])) for line in changed["line_items"]), Decimal(0),
                    ))
                    validated = main.validate_invoice_json(json.dumps(changed, ensure_ascii=False))
                    self.assertNotEqual(validated, case["expected"])


if __name__ == "__main__":
    unittest.main()
