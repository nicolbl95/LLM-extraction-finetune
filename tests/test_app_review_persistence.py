from copy import deepcopy
from pathlib import Path
import os
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import main
from app.local_review import launch_local
from app.review_persistence import PersistenceUI
import invoice_review_edit as edit
import invoice_review_store as store
from test_invoice_review_store import DOCUMENT, FLAGS, MODEL, REVIEW, STRICT


class PersistenceUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".pytest_cache")
        self.addCleanup(self.temp.cleanup)
        self.store = store.DraftStore(Path(self.temp.name) / "storage")
        self.store.prepare()
        self.boundary = SimpleNamespace(authorize=Mock(), store=self.store)
        self.ui = PersistenceUI(self.boundary, main)
        state = edit.initialize_edit_state(STRICT, REVIEW, MODEL)
        snapshot = store.snapshot_for(state, DOCUMENT)
        self.session = {
            "human_edit": state, "persistence": {
                "snapshot": snapshot, "snapshot_hash": store.digest(snapshot),
                "draft_id": None, "revision": None, "saved_hash": None,
                "verified": True, "warnings": [],
            },
        }

    def test_explicit_save_open_failed_open_and_delete(self):
        with patch.object(main, "document_text") as ocr, patch.object(main, "predict_invoice") as model:
            unconsented = self.ui.save(self.session, False)
            self.assertEqual(self.store.list_drafts(), [])
            self.assertIs(unconsented[0], self.session)
            saved = self.ui.save(self.session, True)
            draft_id = saved[0]["persistence"]["draft_id"]
            self.assertIsNotNone(draft_id)
            opened = self.ui.open({}, draft_id, False)
            self.assertEqual(len(opened), 25)
            self.assertEqual(opened[1], STRICT)
            self.assertFalse(opened[0]["human_edit"]["document_verified"])
            self.assertFalse(opened[16]["interactive"])
            self.assertIsNone(opened[18], "old source preview must be cleared")
            self.assertIsNone(opened[19], "old JSON download must be cleared")
            refused = self.ui.open(saved[0], "missing-id", True)
            self.assertEqual(refused[0], main.gr.skip())
            self.ui.delete(saved[0], True)
            stale = self.ui.save(saved[0], True)
            self.assertIn("deleted", stale[1])
            self.assertEqual(self.store.list_drafts(), [])
            ocr.assert_not_called()
            model.assert_not_called()

    def test_snapshot_mutation_and_wrong_source_do_not_replace_session(self):
        original = deepcopy(self.session)
        forged = deepcopy(self.session)
        forged["persistence"]["snapshot"]["strict"]["supplier_name"] = "changed"
        result = self.ui.save(forged, True)
        self.assertIn("snapshot mismatch", result[1])
        self.assertEqual(self.session, original)
        saved = self.ui.save(self.session, True)[0]
        loaded = self.ui.open({}, saved["persistence"]["draft_id"], False)[0]
        with patch.object(main, "checked_path", return_value="ignored"), \
                patch("app.review_persistence.document_identity",
                      return_value=dict(DOCUMENT, sha256="f" * 64)):
            result = self.ui.attach(loaded, "ignored")
        self.assertEqual(result[0], main.gr.skip())
        self.assertFalse(loaded["human_edit"]["document_verified"])


class LaunchTests(unittest.TestCase):
    def test_ordinary_launcher_refuses_persistence(self):
        with patch.dict(os.environ, FLAGS, clear=True):
            with self.assertRaisesRegex(store.StoreError, "app.local_review"):
                main.build_app()

    def test_forbidden_launch_does_not_create_database(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".pytest_cache") as folder:
            db = store.DraftStore(Path(folder) / "storage")
            for extra in ({"SPACE_ID": "private/space"}, {"GRADIO_SERVER_NAME": "0.0.0.0"},
                          {"GRADIO_SHARE": "1"}, {"HF_SPACE_ID": "test"}):
                with patch.dict(os.environ, {**FLAGS, **extra}, clear=True):
                    with self.assertRaises(store.StoreError):
                        launch_local(store=db)
                    self.assertFalse(db.path.exists())

    @unittest.skipUnless(os.name == "nt", "Approved launcher is Windows-local only")
    def test_effective_loopback_config_and_real_file_route_block(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".pytest_cache") as folder, \
                patch.dict(os.environ, FLAGS), \
                patch.object(main, "ocr_availability", return_value=(False, "test")), \
                patch.object(main, "runtime_versions", return_value={
                    "Tesseract": "test", "langues": "", "Pillow": "",
                    "NumPy": "", "OpenCV": "", "PyMuPDF": "",
                }):
            db = store.DraftStore(Path(folder) / "storage")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            demo, boundary = launch_local(port=port, store=db)
            try:
                boundary.authorize()
                self.assertEqual(demo.server.config.host, "127.0.0.1")
                self.assertFalse(demo.share)
                base = f"http://127.0.0.1:{port}"
                for url, headers in (
                    (base + "/gradio_api/file=" + quote(str(db.path), safe=""), {}),
                    (base + "/config", {"X-Forwarded-For": "127.0.0.1"}),
                    (base + "/config", {"Origin": "https://example.invalid"}),
                    (base + "/config", {"Host": "example.invalid"}),
                ):
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(Request(url, headers=headers), timeout=10)
                    self.assertEqual(caught.exception.code, 403)
                demo.server_name = "0.0.0.0"
                with self.assertRaises(store.StoreError):
                    boundary.authorize()
            finally:
                boundary.active = False
                demo.close()


if __name__ == "__main__":
    unittest.main()
