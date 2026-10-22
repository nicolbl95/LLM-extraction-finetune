import copy
import json
import random
import struct
import sys
import tempfile
import unittest
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))

from generate_dataset import (
    CURRENCIES, FORMATS, LOCALES, build_document, build_invoice,
    format_date, generate_records, money, validate_record,
)
from render_invoice import render_png
from split_data import split_records


class InvoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = generate_records()
        cls.records = [record for record, _ in cls.generated]

    def test_balanced_diverse_dataset(self):
        self.assertEqual(len(self.records), 600)
        self.assertEqual(Counter(r["locale"] for r in self.records), dict.fromkeys(LOCALES, 200))
        self.assertEqual(Counter(r["format"] for r in self.records), dict.fromkeys(FORMATS, 200))
        self.assertEqual(len({r["input"] for r in self.records}), 600)
        self.assertEqual(len({r["image_path"] for r in self.records}), 600)
        self.assertEqual(
            len({(r["template_id"], json.dumps(r["visual"], sort_keys=True)) for r in self.records}), 600,
        )
        self.assertEqual(set(r["template_id"] for r in self.records), set(range(1, 10)))
        self.assertEqual(set(r["date_style"] for r in self.records), {"iso", "slash", "long", "dots"})
        self.assertEqual(set(r["visual"]["font"] for r in self.records), {"Helvetica", "Times-Roman", "Courier"})
        for locale in LOCALES:
            subset = [r for r in self.records if r["locale"] == locale]
            self.assertEqual(set(r["currency"] for r in subset), {CURRENCIES[locale]})
            self.assertEqual(set(r["template_id"] for r in subset), set(range(1, 10)))
            for template in range(1, 10):
                self.assertEqual(
                    set(r["format"] for r in subset if r["template_id"] == template), set(FORMATS),
                )
        for record in self.records:
            validate_record(record)
            self.assertTrue((ROOT / record["template_reference"]).is_file())
            self.assertEqual(set(record["output"]), {"supplier_name", "invoice_date", "total_ht", "total_ttc", "line_items"})

    def test_reproducible(self):
        self.assertEqual(self.generated, generate_records())
        self.assertNotEqual(self.generated, generate_records(seed=43))

    def test_localization(self):
        value = date(2025, 9, 23)
        self.assertEqual(format_date(value, "fr-FR", "slash"), "23/09/2025")
        self.assertEqual(format_date(value, "en-GB", "slash"), "23/09/2025")
        self.assertEqual(format_date(value, "en-US", "slash"), "09/23/2025")
        self.assertEqual(money(1234.56, "fr-FR"), "1 234,56 €")
        self.assertEqual(money(1234.56, "en-GB"), "£1,234.56")
        self.assertEqual(money(1234.56, "en-US"), "$1,234.56")
        for record in self.records:
            if record["locale"] != "fr-FR":
                self.assertNotIn("SIRET", record["input"])
                self.assertNotIn("FACTURE", record["input"])
                self.assertNotIn("Quantité", record["input"])
                self.assertNotIn("TVA", record["input"])

    def test_text_and_visual_share_all_content(self):
        for record, document in self.generated:
            for lines in (document.seller, document.customer, document.metadata, document.payment, document.legal):
                for line in lines:
                    self.assertIn(line, record["input"])
            for item in record["output"]["line_items"]:
                self.assertIn(item["description"], record["input"])
            for row in document.rows:
                for cell in row:
                    self.assertIn(cell, record["input"])
            for label, value in document.totals:
                self.assertIn(f"{label} : {value}", record["input"])

    def test_discounts_and_mixed_taxes(self):
        self.assertTrue(any(d["discount_percent"] for r in self.records for d in r["line_details"]))
        self.assertTrue(any(len({d["tax_rate"] for d in r["line_details"]}) > 1 for r in self.records))
        self.assertTrue(any(d["tax_rate"] == 0 for r in self.records for d in r["line_details"]))
        damaged = copy.deepcopy(self.records[0])
        damaged["output"]["total_ttc"] += 0.01
        with self.assertRaisesRegex(ValueError, "totals"):
            validate_record(damaged)
        damaged = copy.deepcopy(self.records[0])
        damaged["line_details"][0]["tax"] += 0.01
        with self.assertRaisesRegex(ValueError, "tax"):
            validate_record(damaged)

    def test_disjoint_reproducible_splits(self):
        splits = split_records(self.records)
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 480, "val": 60, "test": 60})
        self.assertEqual(splits, split_records(self.records))
        all_ids = []
        for name, rows in splits.items():
            per_locale = 160 if name == "train" else 20
            self.assertEqual(Counter(r["locale"] for r in rows), dict.fromkeys(LOCALES, per_locale))
            self.assertEqual(set(r["format"] for r in rows), set(FORMATS))
            for locale in LOCALES:
                self.assertEqual(set(r["template_id"] for r in rows if r["locale"] == locale), set(range(1, 10)))
            all_ids.extend(r["id"] for r in rows)
        self.assertEqual(len(set(all_ids)), 600)

    def test_legacy_split_without_metadata(self):
        rows = [{"format": fmt, "input": str(i)} for fmt in FORMATS for i in range(200)]
        self.assertEqual(
            {name: len(rows) for name, rows in split_records(rows).items()},
            {"train": 480, "val": 60, "test": 60},
        )

    def test_render_all_templates_and_locales(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            rng = random.Random(7)
            for locale in LOCALES:
                invoice, ctx = build_invoice(rng, locale, 1)
                for template in range(9):
                    document = build_document(invoice, ctx, template, rng)
                    path = Path(directory) / f"{locale}-{template}.png"
                    render_png(document, path)
                    image = path.read_bytes()
                    self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertEqual(struct.unpack(">II", image[16:24]), (893, 1263))

    def test_persisted_dataset_and_images(self):
        path = ROOT / "data" / "dataset.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, self.records)
        for row in rows:
            image = (ROOT / row["image_path"]).read_bytes()
            self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        splits = split_records(rows)
        for name, expected in splits.items():
            actual = [json.loads(line) for line in (ROOT / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
