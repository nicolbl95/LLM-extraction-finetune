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
            self.assertIn("prête", status)

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
        with self.assertRaisesRegex(ValueError, "Importez"):
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
        self.assertEqual(main.preview_document(None)[:2], (None, None))

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
        with patch.object(pytesseract, "image_to_string",
                          side_effect=pytesseract.TesseractNotFoundError()):
            with self.assertRaisesRegex(RuntimeError, "OCR indisponible"):
                main.ocr_image(image)
        with patch.object(pytesseract, "image_to_string", return_value=" "):
            with self.assertRaisesRegex(ValueError, "Aucun texte"):
                main.ocr_image(image)

    def test_real_ocr_on_three_examples(self):
        if not shutil.which("tesseract"):
            self.skipTest("Tesseract is not installed")
        if not {"fra", "eng"} <= set(pytesseract.get_languages()):
            self.skipTest("Tesseract fra/eng languages are missing")
        for label, path in main.EXAMPLES:
            with self.subTest(example=label):
                text = main.document_text(str(path))
                self.assertGreater(len(text), 100)
                self.assertTrue("invoice" in text.lower() or "facture" in text.lower())

    def test_validated_result_uses_model_not_dataset(self):
        model = MagicMock()
        model.predict.return_value = {"raw": json.dumps(self.gold), "finish_reason": "stop"}
        with patch.object(main, "get_extractor", return_value=model):
            payload = main.extract_invoice("Text from the uploaded invoice")
        model.predict.assert_called_once_with(
            {"id": "live-upload", "input": "Text from the uploaded invoice"}
        )
        self.assertEqual(payload, self.gold)
        self.assertEqual(payload["invoice_date"], "2026-10-21")
        json.dumps(payload, allow_nan=False)

    def test_rejects_invalid_schema_json_nonfinite_and_truncated_responses(self):
        model = MagicMock()
        cases = [
            ("not JSON", "stop"),
            ('{"supplier_name":"Only one field"}', "stop"),
            (json.dumps(self.gold), "length"),
            (f"```json\n{json.dumps(self.gold)}\n```", "stop"),
            (json.dumps({**self.gold, "total_ht": float("nan")}), "stop"),
            (json.dumps(self.gold).replace('"total_ht": 100', '"total_ht": 1e999'), "stop"),
        ]
        with patch.object(main, "get_extractor", return_value=model):
            for raw, finish in cases:
                with self.subTest(raw=raw, finish=finish):
                    model.predict.return_value = {"raw": raw, "finish_reason": finish}
                    with self.assertRaises(ValueError):
                        main.extract_invoice("invoice text")

    def test_arithmetic_inconsistencies_warn_and_do_not_display_invalid_json(self):
        invalid_payloads = [
            {**self.gold, "line_items": [{**self.gold["line_items"][0], "quantity": 10}]},
            {**self.gold, "total_ht": 101},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                model = MagicMock()
                model.predict.return_value = {"raw": json.dumps(payload), "finish_reason": "stop"}
                with patch.object(main, "document_text", return_value="OCR text"), \
                        patch.object(main, "resolve_adapter_path"), \
                        patch.object(main, "get_extractor", return_value=model), \
                        self.assertLogs(main.LOGGER, level="WARNING"):
                    updates = list(main.run_extraction(str(main.EXAMPLES[0][1])))
                self.assertIsNone(updates[-1][0])
                self.assertIn("Avertissement", updates[-1][1])
                self.assertIn("Incoherence", updates[-1][1])
                self.assertIn("Aucune valeur", updates[-1][1])

    def test_errors_clear_previous_json_and_success_is_validated(self):
        filename = str(main.EXAMPLES[0][1])
        with patch.object(main, "document_text", return_value="OCR text"), \
                patch.object(main, "extract_invoice", return_value=self.gold):
            updates = list(main.run_extraction(filename))
        self.assertIsNone(updates[0][0])
        self.assertEqual(updates[-1][0], self.gold)
        self.assertIn("validé", updates[-1][1])
        with patch.object(main, "document_text", side_effect=RuntimeError("OCR timeout")), \
                self.assertLogs(main.LOGGER, level="ERROR"):
            updates = list(main.run_extraction(filename))
        self.assertIsNone(updates[-1][0])
        self.assertIn("OCR timeout", updates[-1][1])

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
            updates = list(main.run_extraction(str(main.EXAMPLES[0][1])))
        self.assertEqual(
            [entry[0] for entry in sequence.mock_calls],
            ["document_text", "resolve_adapter_path", "extract_invoice"],
        )
        self.assertEqual(updates[-1][0], self.gold)

    def test_gradio_interface_builds_without_loading_model(self):
        with patch.object(main, "get_extractor") as backend:
            demo = main.build_app()
        backend.assert_not_called()
        types = [component["type"] for component in demo.config["components"]]
        self.assertIn("file", types)
        self.assertIn("image", types)
        self.assertIn("json", types)

    def test_zerogpu_registers_inference_at_import(self):
        import runpy
        import sys
        import types

        gpu = MagicMock(side_effect=lambda function: function)
        with patch.dict(sys.modules, {"spaces": types.SimpleNamespace(GPU=gpu)}), \
                patch("evaluate_finetuned.LoRAInference") as backend:
            namespace = runpy.run_path(str(main.ROOT / "app/main.py"))
        gpu.assert_called_once_with(namespace["extract_invoice"])
        backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
