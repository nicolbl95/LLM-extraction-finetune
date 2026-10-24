import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytesseract
from PIL import Image

from app import main
from invoice_evidence import EvidenceText, from_text


class AppTests(unittest.TestCase):
    def setUp(self):
        main.resolve_adapter_path.cache_clear()
        self.addCleanup(main.resolve_adapter_path.cache_clear)
        self.directory = tempfile.TemporaryDirectory(dir=main.ROOT / "app")
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.gold = {
            "supplier_name": "Demo supplier",
            "invoice_date": "2026-10-21",
            "total_ht": 100,
            "total_ttc": 120,
            "line_items": [
                {"description": "Service", "quantity": 1, "unit_price": 100, "total_ht": 100}
            ],
        }

    def make_pdf(self, texts):
        path = self.folder / "invoice.pdf"
        with pymupdf.open() as document:
            for text in texts:
                document.new_page().insert_text((72, 72), text)
            document.save(path)
        return path

    def evidence_text(self, payload=None):
        payload = payload or self.gold
        rows = "\n".join(
            f"{item['description']} | {item['quantity']} | {item['unit_price']} | {item['total_ht']}"
            for item in payload["line_items"]
        )
        return (
            f"Supplier: {payload['supplier_name']}\nInvoice date: {payload.get('invoice_date', '')}\nCurrency: GBP\n"
            "Description | Quantity | Net unit price | Net amount\n" + rows
            + f"\nSubtotal: {payload['total_ht']}\nInvoice total: {payload['total_ttc']}"
        )

    def test_three_distributable_examples_match_existing_regional_invoices(self):
        rows = {
            row["id"]: row
            for row in map(json.loads, (main.ROOT / "data/test.jsonl").read_text().splitlines())
        }
        sources = [
            ("invoice-0406", "fr-FR"),
            ("invoice-0110", "en-GB"),
            ("invoice-0435", "en-US"),
        ]
        self.assertEqual(len(main.EXAMPLES), 3)
        for (_, path), (identifier, locale) in zip(main.EXAMPLES, sources):
            row = rows[identifier]
            self.assertEqual(row["locale"], locale)
            self.assertEqual(path.read_bytes(), (main.ROOT / row["image_path"]).read_bytes())
            image, result, status = main.preview_document(str(path))
            self.assertIsInstance(image, Image.Image)
            self.assertIsNone(result)
            self.assertEqual(status, "")

    def test_pdf_reads_every_page_without_ocr_and_previews_first(self):
        path = self.make_pdf(["First invoice page", "Second invoice page"])
        with patch.object(main, "ocr_image") as ocr:
            text = main.document_text(str(path))
        ocr.assert_not_called()
        self.assertIn("First invoice page", text)
        self.assertIn("Second invoice page", text)
        image, result, status = main.preview_document(str(path))
        self.assertIsInstance(image, Image.Image)
        self.assertIsNone(result)
        self.assertIn("1 sur 2", status)

    def test_mixed_pdf_uses_ocr_only_for_scanned_page(self):
        path = self.make_pdf(["Native text", ""])
        with patch.object(main, "ocr_image", return_value="Scanned invoice") as ocr:
            self.assertEqual(main.document_text(str(path)), "Native text\n\nScanned invoice")
        ocr.assert_called_once()

    def test_input_limits_and_errors(self):
        with self.assertRaisesRegex(ValueError, "Sélectionnez une facture"):
            main.checked_path(None)
        unsupported = self.folder / "invoice.txt"
        unsupported.write_text("invoice")
        with self.assertRaisesRegex(ValueError, "Formats"):
            main.checked_path(str(unsupported))
        empty = self.folder / "empty.png"
        empty.touch()
        with self.assertRaisesRegex(ValueError, "non vide"):
            main.checked_path(str(empty))
        with patch.object(main, "MAX_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "20 Mo"):
                main.checked_path(str(main.EXAMPLES[0][1]))
        with self.assertRaisesRegex(ValueError, "mégapixels"):
            main.check_pixels(5001, 5000)
        too_long = self.make_pdf(["page"] * 11)
        with self.assertRaisesRegex(ValueError, "10 pages"):
            main.document_text(str(too_long))
        corrupt = self.folder / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with self.assertLogs(main.LOGGER, level="ERROR"):
            image, result, status = main.preview_document(str(corrupt))
        self.assertIsNone(image)
        self.assertIsNone(result)
        self.assertIn("Impossible", status)
        self.assertEqual(main.preview_document(None), (None, None, ""))

    def test_encrypted_pdf_is_rejected(self):
        path = self.folder / "protected.pdf"
        with pymupdf.open() as document:
            document.new_page()
            document.save(
                path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner", user_pw="secret",
            )
        with self.assertRaisesRegex(ValueError, "mot de passe"):
            main.document_text(str(path))

    def test_ocr_failure_is_explicit(self):
        image = Image.new("RGB", (10, 10), "white")
        with patch.object(pytesseract, "image_to_data",
                          side_effect=pytesseract.TesseractNotFoundError()):
            with self.assertRaisesRegex(RuntimeError, "OCR indisponible"):
                main.ocr_image(image)
        with patch.object(pytesseract, "image_to_data", return_value={"text": []}):
            with self.assertRaisesRegex(ValueError, "Aucun texte"):
                main.ocr_image(image)

    def test_real_ocr_on_three_examples(self):
        if not shutil.which("tesseract"):
            self.skipTest("Tesseract is not installed")
        if not {"fra", "eng"} <= set(pytesseract.get_languages()):
            self.skipTest("Tesseract fra/eng languages are missing")
        references = {
            "France · EUR": ("invoice-0406", "FR-2025-00406"),
            "Royaume-Uni · GBP": ("invoice-0110", "GB-2025-00110"),
            "États-Unis · USD": ("invoice-0435", "US-2023-00435"),
        }
        rows = {
            row["id"]: row
            for row in map(json.loads, (main.ROOT / "data/test.jsonl").read_text().splitlines())
        }
        for label, path in main.EXAMPLES:
            with self.subTest(example=label):
                text = main.document_text(str(path))
                self.assertGreater(len(text), 100)
                self.assertTrue("invoice" in text.lower() or "facture" in text.lower())
                reference_id, invoice_number = references[label]
                self.assertEqual(text.evidence.invoice_number, invoice_number)
                if reference_id == "invoice-0406":
                    self.assertIn(("siret", "40023297980955"), text.evidence.corroborated_identifiers)
                    diagnostics = {}
                    with patch.object(
                        main, "predict_invoice",
                        side_effect=main.gr.Error("ZeroGPU quota exceeded; try again in 36 min"),
                    ):
                        payload = main.extract_invoice(text, diagnostics=diagnostics)
                    self.assertEqual(payload, rows[reference_id]["output"])
                    report = diagnostics["report"]
                    self.assertTrue(report.startswith("**✅ Facture cohérente : 20 champs vérifiés sur 20**"))
                    self.assertIn(
                        "SIRET : la clé de contrôle ne correspond pas. À vérifier si ce numéro est réel.",
                        report,
                    )
                    for forbidden in ("None", "non ancré", "ancrage", "description_non_verifiee"):
                        self.assertNotIn(forbidden, report)

    def test_gradio_example_and_extract_handlers_return_declared_outputs_for_all_examples(self):
        app = main.build_app()
        dependencies = {
            dependency["api_name"]: dependency
            for dependency in app.config["dependencies"]
        }
        load_event = dependencies["load_example"]
        preview_event = dependencies["preview_ui"]
        extract_event = dependencies["run_extraction_ui"]
        load_handler = app.fns[load_event["id"]].fn
        preview_handler = app.fns[preview_event["id"]].fn
        extract_handler = app.fns[extract_event["id"]].fn
        rows = {
            row["id"]: row
            for row in map(json.loads, (main.ROOT / "data/test.jsonl").read_text().splitlines())
        }
        reference_ids = ("invoice-0406", "invoice-0110", "invoice-0435")
        for index, ((_, path), reference_id) in enumerate(zip(main.EXAMPLES, reference_ids, strict=True)):
            with self.subTest(example=path.name):
                loaded = load_handler((index, [str(path)]))
                loaded_outputs = loaded if isinstance(loaded, (list, tuple)) else [loaded]
                self.assertEqual(len(loaded_outputs), len(load_event["outputs"]))
                selected_path = loaded["value"]["path"]

                preview_outputs = preview_handler(selected_path)
                self.assertEqual(len(preview_outputs), len(preview_event["outputs"]))

                text = EvidenceText(from_text(
                    rows[reference_id]["input"],
                    locale=rows[reference_id]["locale"],
                ))
                with patch.object(main, "document_text", return_value=text), patch.object(
                    main, "predict_invoice",
                    return_value={
                        "raw": json.dumps(rows[reference_id]["output"]),
                        "finish_reason": "stop",
                        "generation_seconds": 0.01,
                    },
                ):
                    frames = list(extract_handler(selected_path))
                self.assertTrue(frames)
                self.assertTrue(all(
                    len(frame) == len(extract_event["outputs"]) for frame in frames
                ))
                self.assertEqual(frames[-1][0], rows[reference_id]["output"])

    def test_extraction_errors_show_a_readable_cause_and_keep_traceback(self):
        errors = (
            (OSError("cannot decode uploaded image"), "Fichier illisible ou inaccessible"),
            (main.pytesseract.TesseractNotFoundError(), "Tesseract indisponible"),
            (RuntimeError("Tesseract process timeout"), "Délai de lecture dépassé"),
        )
        filename = str(main.EXAMPLES[2][1])
        for error, expected in errors:
            with self.subTest(error=type(error).__name__), \
                    patch.object(main, "document_text", side_effect=error), \
                    self.assertLogs(main.LOGGER, level="ERROR") as captured:
                result = list(main.run_extraction(filename))[-1]
            self.assertIn(expected, result[1])
            self.assertIn("Traceback", "\n".join(captured.output))

    def test_uk_example_quota_fallback_reports_number_and_multirate_controls(self):
        if not shutil.which("tesseract"):
            self.skipTest("Tesseract is not installed")
        text = main.document_text(str(main.EXAMPLES[1][1]))
        diagnostics = {}
        with patch.object(
            main, "predict_invoice",
            side_effect=main.gr.Error("ZeroGPU quota exceeded; try again in 36 min"),
        ):
            payload = main.extract_invoice(text, diagnostics=diagnostics)
        reference = next(
            row["output"] for row in map(json.loads, (main.ROOT / "data/test.jsonl").read_text().splitlines())
            if row["id"] == "invoice-0110"
        )
        self.assertEqual(payload, reference)
        report = diagnostics["report"]
        self.assertIn("GB-2025-00110", report)
        self.assertIn("Ligne 1", report)
        self.assertIn("Sous-total", report)
        self.assertIn("TVA 0 %", report)
        self.assertIn("TVA 20 %", report)
        self.assertIn("HT + taxes lues", report)
        self.assertIn("Échéance", report)
        self.assertIn("Devise lue : GBP", report)
        self.assertEqual(report.count("Devise"), 1)
        self.assertIn("Nouvel essai possible dans 36 min.", report)
        self.assertTrue(report.startswith("**✅ Facture cohérente"))
        for forbidden in ("None", "non ancré", "description_non_verifiee", "ancrage", "Pydantic"):
            self.assertNotIn(forbidden, report)
        self.assertTrue(payload["line_items"])
        self.assertEqual(diagnostics["model_info"]["passes_requested"], 1)
        self.assertEqual(diagnostics["model_info"]["passes_succeeded"], 0)

    def test_validated_result_uses_model_not_dataset(self):
        model = MagicMock()
        model.predict.return_value = {"raw": json.dumps(self.gold), "finish_reason": "stop"}
        with patch.object(main, "get_extractor", return_value=model):
            payload = main.extract_invoice(self.evidence_text())
        model.predict.assert_called_once()
        self.assertEqual(
            model.predict.call_args.args[0],
            {"id": "live-upload", "input": self.evidence_text()},
        )
        messages = model.predict.call_args.kwargs["messages"]
        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "user"])
        self.assertIn("7,2", messages[0]["content"])
        self.assertIn("jamais 142", messages[0]["content"])
        self.assertIn("4,2 devient 4.2", messages[0]["content"])
        self.assertIn("tous les chiffres après la virgule", messages[0]["content"])
        self.assertIn("Ne déduis jamais une quantité ou un prix", messages[0]["content"])
        self.assertIn("7,2", messages[1]["content"])
        example = main.Invoice.model_validate_json(messages[2]["content"])
        self.assertEqual(example.line_items[0].quantity, 7.2)
        self.assertEqual(example.line_items[0].unit_price, 12.5)
        self.assertEqual(example.line_items[0].total_ht, 90.0)
        self.assertIn(self.evidence_text(), messages[-1]["content"])
        self.assertEqual(payload, self.gold)
        self.assertEqual(payload["invoice_date"], "2026-10-21")
        json.dumps(payload, allow_nan=False)

    def test_self_correction_preserves_decimal_quantity_and_injects_exact_pydantic_error(self):
        corrected = {
            **self.gold, "total_ht": 90.0, "total_ttc": 108.0,
            "line_items": [{
                "description": "Service", "quantity": 7.2, "unit_price": 12.5, "total_ht": 90.0,
            }],
        }
        invalid = {
            **corrected, "line_items": [{**corrected["line_items"][0], "quantity": 142}],
        }
        raw = json.dumps(invalid)
        with self.assertRaises(main.ValidationError) as validation:
            main.Invoice.model_validate(invalid)
        model = MagicMock()
        model.predict.side_effect = [
            {"raw": raw, "finish_reason": "stop"},
            {"raw": json.dumps(corrected), "finish_reason": "stop"},
        ]
        text = self.evidence_text(corrected)
        warnings = []
        with patch.object(main, "get_extractor", return_value=model), \
                self.assertLogs(level="WARNING"):
            result = main.extract_invoice(text, warnings=warnings)
        self.assertEqual(result, corrected)
        self.assertTrue(any("récupéré (corroboré par OCR)" in warning for warning in warnings))
        self.assertTrue(any("142" in warning for warning in warnings))
        self.assertEqual(model.predict.call_count, 1)
        first = model.predict.call_args
        initial_messages = first.kwargs["messages"]
        self.assertEqual(len(initial_messages), 4)
        self.assertIn(text, initial_messages[-1]["content"])
        self.assertTrue(any("142" in warning and "non ancré" in warning for warning in warnings))

    def test_arithmetic_retry_prioritizes_unit_price_for_a_french_decimal_quantity(self):
        corrected = {
            **self.gold, "total_ht": 84.0, "total_ttc": 100.8,
            "line_items": [
                {"description": "Service", "quantity": 4.2, "unit_price": 20.0, "total_ht": 84.0}
            ],
        }
        invalid = {
            **corrected, "line_items": [{**corrected["line_items"][0], "unit_price": 200.0}],
        }
        model = MagicMock()
        model.predict.side_effect = [
            {"raw": json.dumps(invalid), "finish_reason": "stop"},
            {"raw": json.dumps(corrected), "finish_reason": "stop"},
        ]
        text = self.evidence_text(corrected)
        warnings = []
        with patch.object(main, "get_extractor", return_value=model), \
                self.assertLogs(level="WARNING"):
            self.assertEqual(main.extract_invoice(text, warnings=warnings), corrected)
        self.assertEqual(model.predict.call_count, 1)
        self.assertTrue(any("récupéré (corroboré par OCR)" in warning for warning in warnings))
        self.assertTrue(any("200" in warning for warning in warnings))
        messages = model.predict.call_args.kwargs["messages"]
        self.assertIn(text, messages[-1]["content"])
        self.assertTrue(any("unit_price" in warning for warning in warnings))

    def test_self_correction_handles_missing_fields_and_malformed_json(self):
        missing_date = {key: value for key, value in self.gold.items() if key != "invoice_date"}
        for raw in (json.dumps(missing_date), '{"supplier_name":'):
            with self.subTest(raw=raw):
                model = MagicMock()
                model.predict.side_effect = [
                    {"raw": raw, "finish_reason": "stop"},
                    {"raw": json.dumps(self.gold), "finish_reason": "stop"},
                ]
                with self.assertRaises(ValueError) as validation:
                    main.validate_invoice_json(raw)
                with patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(level="WARNING"):
                    self.assertEqual(main.extract_invoice(self.evidence_text()), self.gold)
                self.assertEqual(model.predict.call_count, 1)

    def test_no_retry_for_model_failures_or_truncated_responses(self):
        model = MagicMock()
        model.predict.side_effect = RuntimeError("Model unavailable")
        with patch.object(main, "get_extractor", return_value=model), self.assertLogs(level="ERROR"):
            self.assertEqual(main.extract_invoice(self.evidence_text()), self.gold)
        self.assertEqual(model.predict.call_count, 1)
        model.reset_mock(side_effect=True)
        model.predict.return_value = {"raw": json.dumps(self.gold), "finish_reason": "length"}
        with patch.object(main, "get_extractor", return_value=model), self.assertLogs(level="ERROR"):
            self.assertEqual(main.extract_invoice(self.evidence_text()), self.gold)
        self.assertEqual(model.predict.call_count, 1)

    def test_second_arithmetic_inconsistency_is_displayed_without_a_third_attempt(self):
        first = {**self.gold, "total_ht": 999}
        last = {**self.gold, "total_ht": 888}
        model = MagicMock()
        model.predict.side_effect = [
            {"raw": json.dumps(first), "finish_reason": "stop"},
            {"raw": json.dumps(last), "finish_reason": "stop"},
        ]
        warnings = []
        with patch.object(main, "get_extractor", return_value=model), \
                self.assertLogs(level="WARNING"):
            actual = main.extract_invoice(self.evidence_text(), warnings=warnings)
        self.assertEqual(actual["total_ht"], self.gold["total_ht"])
        self.assertEqual(actual["line_items"][0]["total_ht"], self.gold["line_items"][0]["total_ht"])
        self.assertEqual(actual["line_items"][0]["quantity"], 1)
        self.assertTrue(any("total_ht" in warning for warning in warnings))
        self.assertEqual(model.predict.call_count, 1)

    def test_failed_correction_preserves_only_recoverable_fields_of_first_json(self):
        first = {**self.gold, "total_ht": 999}
        failures = [
            RuntimeError("GPU unavailable"),
            main.gr.Error("ZeroGPU worker unavailable"),
            {"raw": "not JSON", "finish_reason": "stop"},
            {"raw": json.dumps({**self.gold, "total_ht": float("nan")}), "finish_reason": "stop"},
            {"raw": json.dumps(self.gold), "finish_reason": "length"},
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                model = MagicMock()
                model.predict.side_effect = [
                    {"raw": json.dumps(first), "finish_reason": "stop"}, failure,
                ]
                warnings = []
                with patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(level="WARNING"):
                    actual = main.extract_invoice(self.evidence_text(), warnings=warnings)
                self.assertEqual(actual["total_ht"], self.gold["total_ht"])
                self.assertEqual(actual["line_items"][0]["total_ht"], self.gold["line_items"][0]["total_ht"])
                self.assertEqual(actual["supplier_name"], first["supplier_name"])
                self.assertEqual(model.predict.call_count, 1)
                self.assertTrue(any("total_ht" in message for message in warnings))

    def test_model_error_is_not_hidden_without_a_valid_fallback(self):
        model = MagicMock()
        model.predict.side_effect = [
            {"raw": "not JSON", "finish_reason": "stop"},
            RuntimeError("GPU unavailable"),
        ]
        with patch.object(main, "get_extractor", return_value=model), \
                self.assertLogs(level="WARNING"):
            result = main.extract_invoice("invoice")
        self.assertTrue(all(value is None for item in result["line_items"] for value in item.values()))
        self.assertEqual(model.predict.call_count, 1)

    def test_warning_mode_collects_every_line_and_global_discrepancy(self):
        payload = {
            **self.gold, "total_ht": 200,
            "line_items": [
                {**self.gold["line_items"][0], "quantity": 10},
                {"description": "Second service", "quantity": 1, "unit_price": 20, "total_ht": 19},
            ],
        }
        warnings = []
        actual = main.validate_invoice_json(json.dumps(payload), arithmetic_warnings=warnings)
        self.assertEqual(actual, payload)
        self.assertEqual(len(warnings), 3)
        self.assertIn("Ligne 1", warnings[0])
        self.assertIn("Écart absolu : 900", warnings[0])
        self.assertIn("Ligne 2", warnings[1])
        self.assertIn("Écart absolu : 1", warnings[1])
        self.assertIn("Total HT de la facture", warnings[2])
        self.assertIn("Écart absolu : 81", warnings[2])
        with self.assertRaises(main.ValidationError):
            main.validate_invoice_json(json.dumps(payload))

    def test_rejects_invalid_json_constants_and_truncated_responses(self):
        model = MagicMock()
        cases = [
            ("not JSON", "stop"),
            (json.dumps(self.gold), "length"),
            (f"```json\n{json.dumps(self.gold)}\n```", "stop"),
            (json.dumps({**self.gold, "total_ht": float("nan")}), "stop"),
        ]
        with patch.object(main, "get_extractor", return_value=model):
            for raw, finish in cases:
                with self.subTest(raw=raw, finish=finish):
                    model.predict.return_value = {"raw": raw, "finish_reason": finish}
                    warnings = []
                    with self.assertLogs(level="WARNING"):
                        result = main.extract_invoice("invoice text", warnings=warnings)
                    self.assertTrue(all(value is None for item in result["line_items"] for value in item.values()))
                    self.assertTrue(warnings)

    def test_arithmetic_inconsistencies_display_json_with_manual_review_warning(self):
        invalid_payloads = [
            {**self.gold, "line_items": [{**self.gold["line_items"][0], "quantity": 10}]},
            {**self.gold, "total_ht": 101},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                model = MagicMock()
                model.predict.return_value = {"raw": json.dumps(payload), "finish_reason": "stop"}
                with patch.object(main, "document_text", return_value=self.evidence_text(payload)), \
                        patch.object(main, "resolve_adapter_path"), \
                        patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(level="WARNING"):
                    updates = list(main.run_extraction(str(main.EXAMPLES[0][1])))
                self.assertEqual(updates[-1][0]["total_ht"], payload["total_ht"])
                self.assertEqual(updates[-1][0]["line_items"][0]["total_ht"], payload["line_items"][0]["total_ht"])
                self.assertEqual(updates[-1][0]["supplier_name"], payload["supplier_name"])
                self.assertIn("anomalie", updates[-1][1])
                self.assertIn("Erreur", updates[-1][1])
                self.assertIn("Écart", updates[-1][1])
                self.assertNotIn("cohérence des montants HT contrôlée", updates[-1][1])
                self.assertEqual(model.predict.call_count, 1)

    def test_model_reading_error_is_not_reported_as_invoice_anomaly(self):
                invalid = {**self.gold, "total_ht": 999}
                model = MagicMock()
                model.predict.side_effect = [
                    {"raw": json.dumps(invalid), "finish_reason": "stop"},
                    {"raw": json.dumps(self.gold), "finish_reason": "stop"},
                ]
                with patch.object(main, "document_text", return_value=self.evidence_text()), \
                        patch.object(main, "resolve_adapter_path"), \
                        patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(level="WARNING"):
                    updates = list(main.run_extraction(str(main.EXAMPLES[0][1])))
                payload, status = updates[-1]
                self.assertEqual(payload, self.gold)
                self.assertIn("Facture cohérente : 8 champs vérifiés sur 8", status)
                self.assertNotIn("999", status)
                self.assertIn("Aucune anomalie n’est affirmée", status)
                demo = main.build_app()
                components = demo.config["components"]
                status_index = next(
                    index for index, component in enumerate(components)
                    if component["type"] == "markdown" and component["props"]["value"] == ""
                )
                json_index = next(index for index, component in enumerate(components) if component["type"] == "json")
                self.assertLess(status_index, json_index)

    def test_supplier_and_decimal_normalizations_are_visible_without_retry(self):
                payload = {
                    **self.gold, "supplier_name": "  Acme  LTD. ", "total_ht": "51,00",
                    "line_items": [
                        {"description": "Service", "quantity": "5,1", "unit_price": "10,00", "total_ht": "51,00"},
                    ],
                }
                model = MagicMock()
                model.predict.return_value = {"raw": json.dumps(payload), "finish_reason": "stop"}
                warnings = []
                with patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(level="WARNING"):
                    text = self.evidence_text(payload).replace("Currency: GBP", "Currency: EUR")
                    result = main.extract_invoice(text, warnings=warnings)
                model.predict.assert_called_once()
                self.assertEqual(result["supplier_name"], "Acme Ltd")
                self.assertEqual(result["line_items"][0]["quantity"], 5.1)
                self.assertTrue(any("supplier_name" in warning for warning in warnings))
                self.assertTrue(any("line_items.0.quantity" in warning for warning in warnings))

    def test_irrecoverable_fields_are_null_in_partial_json_display(self):
        invalid = [
            ({key: value for key, value in self.gold.items() if key != "invoice_date"}, "invoice_date"),
            ({**self.gold, "total_ht": "ERROR"}, "total_ht"),
            ({**self.gold, "supplier_name": "Acme LId"}, "supplier_name"),
        ]
        for payload, field in invalid:
            with self.subTest(payload=payload):
                model = MagicMock()
                model.predict.return_value = {"raw": json.dumps(payload), "finish_reason": "stop"}
                with patch.object(main, "document_text", return_value=self.evidence_text(payload)), \
                        patch.object(main, "resolve_adapter_path"), \
                        patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(level="WARNING"):
                    updates = list(main.run_extraction(str(main.EXAMPLES[0][1])))
                self.assertIsNone(updates[-1][0][field])
                self.assertEqual(updates[-1][0]["total_ttc"], self.gold["total_ttc"])
                self.assertIn("Valeur non recoupée : à vérifier manuellement.", updates[-1][1])
                json.dumps(updates[-1][0], allow_nan=False)
                self.assertLessEqual(model.predict.call_count, 2)

    def test_errors_clear_previous_json_and_success_is_validated(self):
        filename = str(main.EXAMPLES[0][1])
        with patch.object(main, "document_text", return_value="OCR text"), \
                patch.object(main, "extract_invoice", return_value=self.gold):
            updates = list(main.run_extraction(filename))
        self.assertIsNone(updates[0][0])
        self.assertEqual(updates[-1][0], self.gold)
        self.assertIn("terminée", updates[-1][1])
        with patch.object(main, "document_text", side_effect=RuntimeError("OCR timeout")), \
                self.assertLogs(main.LOGGER, level="ERROR"):
            updates = list(main.run_extraction(filename))
        self.assertIsNone(updates[-1][0]["total_ht"])
        self.assertIn("Preuves OCR indisponibles", updates[-1][1])

    def test_model_is_lazy_cached_and_configurable(self):
        main.get_extractor.cache_clear()
        self.addCleanup(main.get_extractor.cache_clear)
        with patch.dict(os.environ, {
            "INVOICE_ADAPTER_PATH": "custom/adapter", "INVOICE_DEVICE": "cpu",
        }), patch.object(main, "resolve_adapter_path", return_value=main.ROOT / "custom/adapter"), \
                patch.object(main, "LoRAInference") as backend:
            main.get_extractor()
            main.get_extractor()
        backend.assert_called_once_with(
            adapter_path=main.ROOT / "custom/adapter", device="cpu",
            load_in_4bit=False, n_ctx=4096, max_tokens=2048,
        )

    def make_adapter(self):
        folder = self.folder / "downloaded-adapter"
        folder.mkdir()
        (folder / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": "test/base-model",
        }))
        (folder / "adapter_model.safetensors").write_bytes(b"test weights")
        (folder / "tokenizer_config.json").write_text("{}")
        return folder

    def test_complete_local_adapter_does_not_download(self):
        folder = self.make_adapter()
        with patch.dict(os.environ, {"INVOICE_ADAPTER_PATH": str(folder)}), \
                patch.object(main, "snapshot_download") as download:
            self.assertEqual(main.resolve_adapter_path(), folder)
        download.assert_not_called()

    def test_missing_local_weights_download_to_resolved_cache_path_once(self):
        folder = self.make_adapter()
        with patch.dict(os.environ, {
            "INVOICE_ADAPTER_PATH": str(self.folder / "missing"),
            "INVOICE_ADAPTER_REPO": "test/invoice-adapter",
            "INVOICE_ADAPTER_REVISION": "pinned-revision",
        }), patch.object(main, "snapshot_download", return_value=str(folder)) as download:
            self.assertEqual(main.resolve_adapter_path(), folder)
            self.assertEqual(main.resolve_adapter_path(), folder)
        download.assert_called_once_with(
            repo_id="test/invoice-adapter", revision="pinned-revision",
            allow_patterns=main.ADAPTER_FILES,
        )

    def test_incomplete_local_adapter_downloads_remote_replacement(self):
        folder = self.make_adapter()
        partial = self.folder / "partial"
        partial.mkdir()
        (partial / "adapter_config.json").write_text("{}")
        with patch.dict(os.environ, {"INVOICE_ADAPTER_PATH": str(partial)}), \
                patch.object(main, "snapshot_download", return_value=str(folder)) as download:
            self.assertEqual(main.resolve_adapter_path(), folder)
        download.assert_called_once()

    def test_invalid_local_adapter_is_not_silently_replaced(self):
        folder = self.make_adapter()
        (folder / "adapter_config.json").write_text('{"peft_type":"OTHER"}')
        with patch.dict(os.environ, {"INVOICE_ADAPTER_PATH": str(folder)}), \
                patch.object(main, "snapshot_download") as download:
            with self.assertRaisesRegex(ValueError, "LoRA"):
                main.resolve_adapter_path()
        download.assert_not_called()

    def test_missing_repo_incomplete_repo_and_download_errors_are_explicit(self):
        with patch.dict(os.environ, {
            "INVOICE_ADAPTER_PATH": str(self.folder / "missing"),
            "INVOICE_ADAPTER_REPO": "",
        }), patch.object(main, "snapshot_download") as download:
            with self.assertRaisesRegex(RuntimeError, "Configurez INVOICE_ADAPTER_REPO"):
                main.resolve_adapter_path()
        download.assert_not_called()
        with patch.dict(os.environ, {
            "INVOICE_ADAPTER_PATH": str(self.folder / "missing"),
            "INVOICE_ADAPTER_REPO": "test/empty-repository",
        }), patch.object(main, "snapshot_download", return_value=str(self.folder)):
            with self.assertRaisesRegex(RuntimeError, "fichiers requis"):
                main.resolve_adapter_path()
        with patch.dict(os.environ, {
            "INVOICE_ADAPTER_PATH": str(self.folder / "missing"),
            "INVOICE_ADAPTER_REPO": "test/private-repository",
        }), patch.object(
            main, "snapshot_download",
            side_effect=main.HfHubHTTPError("Forbidden", response=MagicMock()),
        ):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                main.resolve_adapter_path()

    def test_document_and_adapter_preparation_precede_gpu_inference(self):
        sequence = MagicMock()
        with patch.object(main, "document_text", sequence.document_text), \
                patch.object(main, "resolve_adapter_path", sequence.resolve_adapter_path), \
                patch.object(main, "extract_invoice", sequence.extract_invoice):
            sequence.document_text.return_value = "OCR text"
            sequence.extract_invoice.return_value = self.gold
            updates = main.run_extraction(str(main.EXAMPLES[0][1]))
            self.assertEqual(next(updates), (None, "Analyse des données, détection des erreurs"))
            sequence.document_text.assert_not_called()
            self.assertEqual(next(updates), (None, "Analyse des données, détection des erreurs"))
            sequence.resolve_adapter_path.assert_not_called()
            sequence.extract_invoice.assert_not_called()
            last_update = next(updates)
            with self.assertRaises(StopIteration):
                next(updates)
        self.assertEqual(
            [entry[0] for entry in sequence.mock_calls],
            ["document_text", "extract_invoice"],
        )
        self.assertEqual(last_update[0], self.gold)
        self.assertIn("terminée", last_update[1])

    def test_gradio_interface_builds_without_loading_model(self):
        with patch.object(main, "get_extractor") as backend:
            demo = main.build_app()
        backend.assert_not_called()
        types = [component["type"] for component in demo.config["components"]]
        self.assertIn("file", types)
        self.assertIn("image", types)
        self.assertIn("json", types)
        self.assertEqual(types.count("row"), 1)
        self.assertEqual(types.count("column"), 2)
        self.assertEqual(demo.config["title"], "Extraction standardisée de données de factures")
        examples = [
            component for component in demo.config["components"] if component["type"] == "dataset"
        ]
        self.assertEqual(examples[0]["props"]["label"], "")
        markdown = [
            component["props"]["value"] for component in demo.config["components"]
            if component["type"] == "markdown"
        ]
        self.assertEqual(
            markdown[0],
            "# Extraction standardisée de données de factures\n"
            "Transformez vos factures françaises, britanniques et américaines "
            "en données JSON structurées avec un modèle spécialisé pour obtenir un output clair "
            "des données importantes standardisées.",
        )
        self.assertEqual(
            markdown[1:3],
            [
                "PDF de 10 pages maximum · 20 Mo maximum · OCR français/anglais",
                "Essayer une facture d'exemple : France / Royaume-Uni / États-Unis 🔽",
            ],
        )
        help_components = [
            component["props"] for component in demo.config["components"]
            if component["type"] == "markdown" and component["props"]["value"] in markdown[1:3]
        ]
        self.assertEqual(
            {key: value for key, value in help_components[0].items() if key != "value"},
            {key: value for key, value in help_components[1].items() if key != "value"},
        )
        self.assertNotIn("Importez une facture ou choisissez un exemple.", markdown)
        self.assertEqual(markdown[-1], "")
        self.assertFalse(any("À vérifier avant utilisation" in text for text in markdown))
        self.assertFalse(any("Confidentialité" in text for text in markdown))

    def test_zerogpu_registers_inference_at_import(self):
        import runpy
        import sys
        import types

        gpu = MagicMock(side_effect=lambda *, duration: lambda function: function)
        with patch.dict(sys.modules, {"spaces": types.SimpleNamespace(GPU=gpu)}), \
                patch("evaluate_finetuned.LoRAInference") as backend:
            namespace = runpy.run_path(str(main.ROOT / "app/main.py"))
        gpu.assert_called_once_with(duration=namespace["gpu_duration"])
        backend.assert_not_called()

    def test_gpu_callback_returns_raw_predictions_without_pydantic_validation(self):
        prediction = {"raw": '{"supplier_name":"Incomplete invoice"}', "finish_reason": "stop"}
        model = MagicMock()
        model.predict.return_value = prediction
        row = {"id": "test", "input": "invoice"}
        messages = [{"role": "user", "content": "invoice"}]
        with patch.object(main, "get_extractor", return_value=model), \
                patch.object(main, "validate_invoice_json") as validate:
            self.assertEqual(main.predict_invoice(row, messages), prediction)
        model.predict.assert_called_once_with(row, messages=messages)
        validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
