import copy
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pymupdf
import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import main  # noqa: E402
from invoice_evidence import DocumentEvidence, EvidenceText, from_text  # noqa: E402
import invoice_review  # noqa: E402
import invoice_review_edit  # noqa: E402

INVOICE_TEXT = (
    "Fournisseur : Batisolite SARL\n"
    "Date de facture : 12/03/2026\n"
    "Désignation | Quantité | PU HT | Montant HT\n"
    "Pose isolation 7,2 12,50 90,00\n"
    "Total HT : 90,00\n"
    "Total TTC : 108,00\n"
)
MODEL_READING = {
    "supplier_name": "Batisolite SARL",
    "invoice_date": "2026-03-12",
    "total_ht": 90.0,
    "total_ttc": 108.0,
    "line_items": [
        {"description": "Pose isolation", "quantity": 7.2, "unit_price": 12.5, "total_ht": 90.0}
    ],
}
PREDICTION = {"raw": json.dumps(MODEL_READING), "finish_reason": "stop",
              "generation_seconds": 0.01}
KNOWN_LABELS = set(invoice_review.STATUS_LABELS.values()) | set(
    invoice_review.STRICT_STATUS_LABELS.values()
) | {"récupéré (document) — divergence avec le modèle"}


class ReviewUiTests(unittest.TestCase):
    def setUp(self):
        main.resolve_adapter_path.cache_clear()
        self.addCleanup(main.resolve_adapter_path.cache_clear)
        self.directory = tempfile.TemporaryDirectory(dir=main.ROOT / "app")
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.example = str(main.REGIONAL_EXAMPLES[0][1])

    @contextmanager
    def environment(self, enabled, edit_enabled=False):
        saved = os.environ.pop("INVOICE_REVIEW_UI", None)
        saved_edit = os.environ.pop("INVOICE_REVIEW_EDIT", None)
        if enabled:
            os.environ["INVOICE_REVIEW_UI"] = "1"
        if edit_enabled:
            os.environ["INVOICE_REVIEW_EDIT"] = "1"
        try:
            yield
        finally:
            os.environ.pop("INVOICE_REVIEW_UI", None)
            os.environ.pop("INVOICE_REVIEW_EDIT", None)
            if saved is not None:
                os.environ["INVOICE_REVIEW_UI"] = saved
            if saved_edit is not None:
                os.environ["INVOICE_REVIEW_EDIT"] = saved_edit

    def handlers(self, app):
        dependencies = {
            dependency["api_name"]: dependency
            for dependency in app.config["dependencies"]
            if dependency.get("api_name")
        }
        calls = {name: app.fns[dependencies[name]["id"]].fn
                 for name in ("preview_ui", "run_extraction_ui")}
        return dependencies, calls

    def review_handlers(self):
        return {
            "preview_ui": main.preview_ui_with_review,
            "run_extraction_ui": main.run_extraction_ui_with_review,
        }

    def extract(self, handler, filename, state=None):
        text = EvidenceText(from_text(INVOICE_TEXT, locale="fr-FR"))
        cache_dir = self.folder / "gradio-cache"
        cache_dir.mkdir(exist_ok=True)
        with patch.dict(os.environ, {"GRADIO_TEMP_DIR": str(cache_dir)}), \
                patch.object(main, "document_text", return_value=text), \
                patch.object(main, "predict_invoice", return_value=PREDICTION) as predict:
            if handler.__name__ in {"run_extraction_ui", "run_recruiter_extraction_ui"}:
                frames = list(handler(filename))
            else:
                frames = list(handler(filename, state))
        return frames, predict

    def make_pdf(self, pages):
        path = self.folder / f"invoice-{pages}-pages.pdf"
        with pymupdf.open() as document:
            for index in range(pages):
                document.new_page().insert_text((72, 72), f"Page {index + 1} invoice")
            document.save(path)
        return str(path)

    def test_flag_off_keeps_the_declared_outputs_and_the_existing_frames(self):
        with self.environment(False):
            app = main.build_app()
            events, calls = self.handlers(app)
            self.assertEqual(len(events["preview_ui"]["outputs"]), 5)
            self.assertEqual(len(events["run_extraction_ui"]["outputs"]), 4)
            self.assertEqual(len(events["run_extraction_ui"]["inputs"]), 1)
            configuration = json.dumps(app.config, ensure_ascii=False, default=str)
            self.assertNotIn("Révision — lecture seule", configuration)
            frames, predict = self.extract(calls["run_extraction_ui"], self.example)
            self.assertTrue(all(len(frame) == 4 for frame in frames))
            self.assertIsInstance(frames[-1][0], dict)
            self.assertEqual(predict.call_count, 1)

    def test_legacy_flags_do_not_restore_review_ui_or_change_strict_output(self):
        baseline, _ = self.extract(main.run_recruiter_extraction_ui, self.example)
        for review, edit in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(review=review, edit=edit), \
                    self.environment(review, edit_enabled=edit), \
                    patch.object(main, "get_extractor") as backend, \
                    patch.object(main, "build_review") as review_builder:
                app = main.build_app()
                events, calls = self.handlers(app)
                self.assertEqual(len(events["run_extraction_ui"]["outputs"]), 4)
                self.assertEqual(len(events["preview_ui"]["outputs"]), 5)
                self.assertEqual(len(events["run_extraction_ui"]["inputs"]), 1)
                configuration = json.dumps(app.config, ensure_ascii=False, default=str)
                for removed in (
                    "Révision — lecture seule", "Document source",
                    "Champs évalués par le contrôle P1", "Contrat de révision",
                    "Révision humaine", "Accepter la proposition", "Rejeter la proposition",
                    "Appliquer la correction", "Confirmer ce champ", "Brouillons locaux",
                    "Réponse brute du modèle — première passe",
                    "Détails des contrôles — hors du résultat JSON",
                    "Version déployée", "Empreinte de configuration OCR",
                ):
                    self.assertNotIn(removed, configuration)
                types = {c["type"] for c in app.config["components"]}
                self.assertTrue(types.isdisjoint({"gallery", "dataframe", "dropdown", "state"}))
                self.assertFalse(any(
                    str(d.get("api_name", "")).startswith(("review_", "draft_"))
                    for d in app.config["dependencies"]
                ))
                frames, predict = self.extract(calls["run_extraction_ui"], self.example)
                self.assertTrue(all(len(frame) == 4 for frame in frames))
                # The recruiter performance block now carries the measured duration of the finished
                # request, so it is not comparable for equality across runs; the strict output stays
                # byte-for-byte parallel on payload, verification summary and downloaded JSON, and the
                # block is checked against its own duration-only contract on every frame instead.
                self.assertEqual(
                    [frame[:2] for frame in frames], [frame[:2] for frame in baseline],
                )
                for side in (frames, baseline):
                    for frame in side[:-1]:
                        self.assertEqual(frame[2], "### Performances")
                    self.assertRegex(
                        side[-1][2],
                        r"^### Performances\n- \*\*Temps de traitement : \d+,\d{2} s\*\*$",
                    )
                for frame in frames + baseline:
                    for failure_word in ("Qwen", "LoRA", "GPU", "Modèle", "Passes", "Génération"):
                        self.assertNotIn(failure_word, frame[2])
                self.assertIsNone(frames[-1][0]["line_items"][0]["quantity"])
                self.assertIsNone(frames[-1][0]["line_items"][0]["unit_price"])
                self.assertEqual(
                    Path(frames[-1][-1]).read_bytes(), Path(baseline[-1][-1]).read_bytes(),
                )
                self.assertEqual(predict.call_count, 1)
                backend.assert_not_called()
                review_builder.assert_not_called()

    def test_retained_review_helpers_do_not_change_any_strict_frame(self):
        with self.environment(False):
            off_app = main.build_app()
            self.handlers(off_app)
            off_frames, off_predict = self.extract(main.run_extraction_ui, self.example)
        with self.environment(True):
            on_calls = self.review_handlers()
            on_frames, on_predict = self.extract(on_calls["run_extraction_ui"], self.example)
            self.assertTrue(all(len(frame) == 12 for frame in on_frames))
        self.assertEqual(
            [frame[:6] for frame in off_frames], [frame[:6] for frame in on_frames],
            "result, report, details, raw answer and validation stay identical",
        )
        self.assertEqual(off_predict.call_count, on_predict.call_count, "no extra GPU call")
        pages, table, summary, review_json, state = on_frames[-1][7:]
        self.assertTrue(pages and isinstance(pages[0], Image.Image))
        self.assertEqual(table["headers"], list(invoice_review.COLUMN_HEADERS))
        self.assertTrue(table["data"])
        self.assertIn("Révision — lecture seule", summary)
        self.assertEqual(state["document"], Path(self.example).name)
        self.assertEqual(len(state["pages"]), len(pages))
        self.assertEqual(review_json["mode"], "review_assisted")
        self.assertTrue({row[3] for row in table["data"]} <= KNOWN_LABELS)
        rendered = gr.Dataframe(headers=list(invoice_review.COLUMN_HEADERS), datatype="str",
                                interactive=False, row_count=0).postprocess(table).model_dump()
        self.assertEqual(rendered["headers"], list(invoice_review.COLUMN_HEADERS))
        self.assertEqual(len(rendered["data"]), len(table["data"]))
        self.assertEqual(len(gr.Gallery(columns=1).postprocess(pages).model_dump()), len(pages))

    def test_proposals_stay_out_of_the_primary_fields_of_both_payloads(self):
        with self.environment(True):
            calls = self.review_handlers()
            frames, _ = self.extract(calls["run_extraction_ui"], self.example)
        payload, table, review_json = frames[-1][0], frames[-1][8], frames[-1][10]
        proposed = {path: item for path, item in review_json["field_assessments"].items()
                    if item["status"] == "proposed"}
        for path, item in proposed.items():
            row_index, field = invoice_review._path_parts(path)
            primary = (payload["line_items"][row_index].get(field) if row_index is not None
                       else payload.get(field))
            self.assertIsNone(primary, f"{path} proposed {item['value']!r} must not be exported")
            retained = (review_json["line_items"][row_index].get(field) if row_index is not None
                        else review_json.get(field))
            self.assertIsNone(retained)
            display = next(row for row in table["data"] if row[0] == invoice_review.field_label(path))
            self.assertEqual(display[1], "—")
            self.assertNotEqual(display[2], "—")
        self.assertEqual(review_json["review_summary"]["proposed"], len(proposed))

    def test_status_labels_keep_verified_and_recovered_distinct(self):
        with self.environment(True):
            calls = self.review_handlers()
            frames, _ = self.extract(calls["run_extraction_ui"], self.example)
        rows = frames[-1][8]["data"]
        labels = {row[3] for row in rows}
        self.assertTrue(labels <= KNOWN_LABELS, labels)
        for row in rows:
            if row[3].startswith("récupéré"):
                self.assertNotEqual(row[1], "—")
        self.assertIn(invoice_review.PROPOSAL_DISCLAIMER, frames[-1][9])
        self.assertIn("Phase 1 : affichage seul", frames[-1][9])

    def test_unavailable_evidence_is_reported_as_a_limitation(self):
        unreadable = EvidenceText(DocumentEvidence(
            "", None, [], [], available=False, issues=["Preuves OCR indisponibles"],
        ))
        with self.environment(True):
            calls = self.review_handlers()
            with patch.object(main, "document_text", return_value=unreadable), \
                    patch.object(main, "predict_invoice",
                                 side_effect=AssertionError("inference was called")):
                frames = list(calls["run_extraction_ui"](self.example, None))
        self.assertIn("Aucune preuve documentaire disponible", frames[-1][9])
        self.assertTrue(frames[-1][8]["data"])
        self.assertTrue(all(row[3] == "absent du document" or row[1] != "—"
                            for row in frames[-1][8]["data"]))

    def test_review_pages_come_from_the_state_after_the_upload_cache_is_lost(self):
        with self.environment(True):
            pages, _, name = main.review_pages(self.example)
            copy_path = self.folder / "cache-perdu.png"
            copy_path.write_bytes(Path(self.example).read_bytes())
            copy_path.unlink()
            fresh, limitations, fresh_name = main.review_pages(str(copy_path))
            self.assertEqual(fresh, [])
            self.assertIn("cache Gradio", limitations[0])
            replayed, replayed_limitations, replayed_name = main.review_pages(
                str(copy_path), {"pages": pages, "limitations": [], "document": name})
            self.assertEqual([image.size for image in replayed], [image.size for image in pages])
            self.assertEqual(replayed_name, name)
            self.assertEqual(replayed_limitations, ())
            self.assertEqual(fresh_name, copy_path.name)

    def test_multi_page_documents_render_every_source_page(self):
        path = self.make_pdf(3)
        with self.environment(True):
            calls = self.review_handlers()
            preview = calls["preview_ui"](path)
            self.assertEqual(len(preview), 13)
            self.assertEqual(len(preview[8]), 3)
            self.assertEqual(len(preview[12]["pages"]), 3)
            self.assertEqual(preview[12]["limitations"], [])
            self.assertEqual(preview[11]["display"]["limitations"], [main.REVIEW_PENDING_REASON])
            frames, predict = self.extract(
                calls["run_extraction_ui"], path,
                {"pages": preview[8], "limitations": [], "document": Path(path).name})
            self.assertEqual(len(frames[-1][7]), 3)
            self.assertEqual(predict.call_count, 1)

    def test_pending_preview_never_invents_review_status(self):
        with self.environment(True):
            calls = self.review_handlers()
            preview = calls["preview_ui"](self.example)
        table, summary, review_json = preview[9], preview[10], preview[11]
        self.assertEqual(table["data"], [])
        self.assertIn(main.REVIEW_PENDING_REASON, summary)
        self.assertEqual(review_json["review_summary"]["verified"], 0)
        self.assertEqual(review_json["field_assessments"], {})
        self.assertEqual(len(preview[8]), 1)

    def test_states_of_two_documents_do_not_leak_into_each_other(self):
        first, second = self.make_pdf(1), self.make_pdf(2)
        with self.environment(True):
            first_pages, _, first_name = main.review_pages(first)
            second_pages, _, second_name = main.review_pages(second)
            self.assertEqual((len(first_pages), len(second_pages)), (1, 2))
            self.assertNotEqual(first_name, second_name)
            reused, _, reused_name = main.review_pages(
                second, {"pages": first_pages, "limitations": [], "document": first_name})
            self.assertEqual(reused_name, first_name)
            self.assertEqual(len(reused), 1)
            self.assertIsNot(reused, first_pages)

    def test_the_review_assessment_does_not_mutate_the_strict_result(self):
        diagnostics = {
            "field_statuses": {"supplier_name": "valide"}, "provenance": {},
            "evidence": from_text(INVOICE_TEXT), "evidence_available": True,
            "raw": json.dumps(MODEL_READING), "report": "x",
        }
        snapshot = json.dumps({key: value for key, value in diagnostics.items()
                               if key != "evidence"}, ensure_ascii=False, default=str)
        payload = json.loads(json.dumps(MODEL_READING))
        built = main.build_review(diagnostics, payload)
        self.assertEqual(payload, MODEL_READING, "the strict payload handed to the UI is unchanged")
        self.assertEqual(json.dumps({key: value for key, value in diagnostics.items()
                                     if key != "evidence"}, ensure_ascii=False, default=str),
                         snapshot)
        self.assertEqual(set(built), {"summary", "table", "json"})
        self.assertEqual(built["json"]["total_ttc"], 108.0)

    def test_retained_edit_helpers_do_not_require_extra_inference(self):
        with self.environment(False):
            baseline_app = main.build_app()
            _, baseline_calls = self.handlers(baseline_app)
            baseline_frames, _ = self.extract(
                baseline_calls["run_extraction_ui"], self.example,
            )
        with self.environment(True, edit_enabled=True):
            pending_preview = main.preview_ui_with_review(self.example)
            self.assertEqual(len(pending_preview), 24)
            self.assertFalse(pending_preview[-1]["interactive"])
            handler = main.run_extraction_ui_with_review
            text = EvidenceText(from_text(INVOICE_TEXT, locale="fr-FR"))
            with patch.object(main, "document_text", return_value=text) as ocr, \
                    patch.object(main, "predict_invoice", return_value=PREDICTION) as predict:
                frames = list(handler(self.example, {}))
            self.assertTrue(all(len(frame) == 23 for frame in frames))
            strict_payload = frames[-1][0]
            session = frames[-1][11]
            self.assertEqual(strict_payload, baseline_frames[-1][0])
            self.assertTrue(frames[-1][-1]["interactive"])
            self.assertIn("human_edit", session)
            self.assertEqual(session["human_edit"]["original_strict"], strict_payload)
            original_snapshot = json.dumps(strict_payload, sort_keys=True)

            edited = main._edit_action(
                session, "supplier_name", "Correction de session", "edit",
            )
            edited_state = edited[0]
            confirmed = main._edit_action(
                edited_state, "supplier_name", "", "confirm",
            )
            self.assertTrue(
                confirmed[0]["human_edit"]["confirmed"]["supplier_name"],
            )
            self.assertEqual(json.dumps(strict_payload, sort_keys=True), original_snapshot)
            self.assertEqual(predict.call_count, 1, "editing must not invoke inference again")
            self.assertEqual(ocr.call_count, 1, "editing must not invoke OCR again")
            self.assertEqual(confirmed[7]["fields"]["supplier_name"]["value"],
                             "Correction de session")
            self.assertFalse(confirmed[7]["finalized"])
            invalid = main._edit_action(confirmed[0], "supplier_name", "", "unexpected")
            self.assertEqual(invalid[6], "Action de révision invalide.")
            malformed_action = main._edit_action(
                confirmed[0], "supplier_name", "", ["confirm"],
            )
            self.assertEqual(malformed_action[6], "Action de révision invalide.")
            invalid_field = main._edit_action(
                confirmed[0], "supplier_name;total_ht", "forged", "edit",
            )
            self.assertIn("Champ invalide", invalid_field[6])
            forged = copy.deepcopy(confirmed[0])
            forged["human_edit"]["draft"]["supplier_name"] = "Client forgery"
            with self.assertRaises(invoice_review_edit.EditStateError):
                invoice_review_edit.confirmed_view(forged["human_edit"])

    def test_retained_helpers_block_invalid_arithmetic_and_invalidate_confirmation(self):
        with self.environment(True, edit_enabled=True):
            calls = self.review_handlers()
            text = EvidenceText(from_text(INVOICE_TEXT, locale="fr-FR"))
            with patch.object(main, "document_text", return_value=text) as ocr, \
                    patch.object(main, "predict_invoice", return_value=PREDICTION) as predict:
                session = list(calls["run_extraction_ui"](self.example, {}))[-1][11]
                row_id = session["human_edit"]["row_order"][0]
                quantity = f"row:{row_id}:quantity"
                unit_price = f"row:{row_id}:unit_price"
                row_total = f"row:{row_id}:total_ht"
                session = main._edit_action(session, quantity, "7.2", "edit")[0]
                session = main._edit_action(session, unit_price, "12.5", "edit")[0]
                session = main._edit_action(session, row_total, "90", "edit")[0]
                changed = main._edit_action(session, quantity, "9", "edit")
                self.assertFalse(changed[-1]["interactive"])
                blocked = main._edit_action(changed[0], quantity, "", "confirm")
                self.assertFalse(blocked[0]["human_edit"]["confirmed"][quantity])
                self.assertIn("ne correspond pas", blocked[6])

                corrected = main._edit_action(blocked[0], quantity, "7.2", "edit")
                self.assertTrue(corrected[-1]["interactive"])
                confirmed = main._edit_action(corrected[0], quantity, "", "confirm")
                self.assertTrue(confirmed[0]["human_edit"]["confirmed"][quantity])
                self.assertEqual(
                    confirmed[7]["line_items"][0]["fields"]["quantity"]["status"],
                    "human_confirmed",
                )
                edited_again = main._edit_action(confirmed[0], quantity, "6", "edit")
                self.assertFalse(edited_again[0]["human_edit"]["confirmed"][quantity])
                self.assertFalse(edited_again[-1]["interactive"])
            self.assertEqual(ocr.call_count, 1)
            self.assertEqual(predict.call_count, 1)

    def test_edit_flag_alone_does_not_enable_controls_without_phase_one(self):
        with self.environment(False, edit_enabled=True):
            app = main.build_app()
            configuration = json.dumps(app.config, ensure_ascii=False, default=str)
            self.assertNotIn("Révision humaine — brouillon de session", configuration)
            self.assertNotIn("review_confirm", configuration)

    def test_public_space_shows_upload_privacy_warning(self):
        with self.environment(False), patch.dict(os.environ, {"SPACE_ID": "public-test-space"}):
            public_app = main.build_app()
        public_configuration = json.dumps(public_app.config, ensure_ascii=False, default=str)
        self.assertIn("Confidentialité — Espace Hugging Face", public_configuration)
        self.assertIn("N'importez pas de facture confidentielle", public_configuration)
        for removed in (
            "Les fichiers temporaires peuvent rester disponibles",
            "une URL Gradio jusqu'au nettoyage périodique",
            "suppression immédiate n'est pas garantie",
            "Les brouillons humains ne sont pas sauvegardés",
        ):
            self.assertNotIn(removed, public_configuration)
        self.assertEqual(public_app.delete_cache, (600, 600))

        with self.environment(False), patch.dict(os.environ, {"SPACE_ID": ""}):
            local_app = main.build_app()
        local_configuration = json.dumps(local_app.config, ensure_ascii=False, default=str)
        self.assertNotIn("Confidentialité — Espace Hugging Face", local_configuration)

    def test_new_extraction_replaces_session_edit_state(self):
        with self.environment(True, edit_enabled=True):
            calls = self.review_handlers()
            text = EvidenceText(from_text(INVOICE_TEXT, locale="fr-FR"))
            with patch.object(main, "document_text", return_value=text), \
                    patch.object(main, "predict_invoice", return_value=PREDICTION):
                first = list(calls["run_extraction_ui"](self.example, {}))[-1]
                changed = main._edit_action(
                    first[11], "supplier_name", "Session corrigée", "edit",
                )[0]
                second = list(calls["run_extraction_ui"](self.example, changed))[-1]
        self.assertNotEqual(
            first[11]["human_edit"]["row_order"],
            second[11]["human_edit"]["row_order"],
        )
        self.assertEqual(
            second[11]["human_edit"]["draft"]["supplier_name"],
            second[0]["supplier_name"],
        )
        self.assertNotEqual(
            second[11]["human_edit"]["draft"]["supplier_name"],
            "Session corrigée",
        )


if __name__ == "__main__":
    unittest.main()
