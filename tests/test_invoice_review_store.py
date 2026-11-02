import copy
from contextlib import closing
from decimal import Decimal
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import invoice_review_edit as edit
import invoice_review_store as store
from app.local_review import LocalBoundary

FLAGS = {key: "1" for key in
         ("INVOICE_REVIEW_UI", "INVOICE_REVIEW_EDIT", "INVOICE_REVIEW_PERSIST")}
STRICT = {
    "supplier_name": "Synthetic", "invoice_date": "2026-03-12",
    "total_ht": 12.0, "total_ttc": None,
    "line_items": [{"description": "Repeated", "quantity": 2, "unit_price": 3, "total_ht": 6},
                   {"description": "Repeated", "quantity": 2, "unit_price": 3, "total_ht": 6}],
}
MODEL = copy.deepcopy(STRICT)
MODEL["supplier_name"] = "Synthetic Ltd"
MODEL["line_items"][0]["quantity"] = Decimal("2.00")
REVIEW = {"line_items": [{}, {}], "field_assessments": {
    "supplier_name": {"status": "proposed", "evidence": {"text": "Synthetic"}},
    "line_items[1].quantity": {"status": "verified", "reason": "final_status"},
}}
DOCUMENT = {"sha256": "a" * 64, "size": 20, "media_type": "image/png"}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".pytest_cache")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "store"
        self.store = store.DraftStore(self.directory)
        self.store.prepare()
        self.state = edit.initialize_edit_state(STRICT, REVIEW, MODEL)
        self.snapshot = store.snapshot_for(self.state, DOCUMENT)

    def test_roundtrip_restart_decisions_confirmations_and_rows(self):
        state = edit.accept_proposal(self.state, "supplier_name")
        state = edit.confirm_field(state, "supplier_name")
        row = state["row_order"][1]
        state = edit.reject_proposal(state, f"row:{row}:quantity")
        state = edit.add_line_item(state)
        originals = copy.deepcopy((STRICT, REVIEW, MODEL))
        saved = self.store.save(self.snapshot, state)
        new_store = store.DraftStore(self.directory)
        new_store.prepare()
        with patch.object(edit, "_CONFIRMATION_KEY", b"new-process-key"):
            loaded = new_store.open(saved["draft_id"])
            self.assertFalse(loaded["human_edit"]["confirmed"]["supplier_name"])
            with self.assertRaises(edit.EditStateError):
                edit.confirm_field(loaded["human_edit"], "supplier_name")
            attached = new_store.open(saved["draft_id"], document=DOCUMENT)
            self.assertTrue(attached["human_edit"]["confirmed"]["supplier_name"])
            self.assertNotEqual(
                attached["human_edit"]["confirmation_tokens"]["supplier_name"],
                state["confirmation_tokens"]["supplier_name"],
            )
            self.assertFalse(edit.confirmed_view(attached["human_edit"])["finalized"])
        self.assertEqual(attached["human_edit"]["row_order"], state["row_order"])
        self.assertEqual(attached["human_edit"]["decisions"], state["decisions"])
        self.assertEqual(attached["snapshot"], self.snapshot)
        self.assertEqual((STRICT, REVIEW, MODEL), originals)
        with closing(sqlite3.connect(self.store.path)) as db, db:
            serialized = db.execute("SELECT data FROM drafts").fetchone()[0]
            self.assertNotIn("confirmation_tokens", serialized)
            self.assertNotIn("pages", serialized)

    def test_exact_numeric_codec_and_reserved_keys(self):
        data = {"x": Decimal("123456789012345678.1234500"), "float": 0.1,
                "big": 10 ** 80, "text": "2.00", "null": None,
                "nested": {"decimal": "not a tag"}, "bool": True}
        restored = store.loads(store.dumps(data))
        self.assertEqual(data, restored)
        self.assertEqual(type(restored["float"]), float)
        self.assertEqual(restored["x"].as_tuple(), data["x"].as_tuple())
        for value in (float("nan"), Decimal("Infinity"), Path("no-paths")):
            with self.assertRaises(store.StoreError):
                store.dumps(value)

    def test_incomplete_and_inconsistent_values_are_saved_without_correction(self):
        row = self.state["row_order"][0]
        state = edit.apply_correction(self.state, f"row:{row}:quantity", 9)
        saved = self.store.save(self.snapshot, state)
        loaded = self.store.open(saved["draft_id"], document=DOCUMENT)
        self.assertEqual(loaded["human_edit"]["draft"][f"row:{row}:quantity"], 9)
        self.assertTrue(edit.validate_draft(loaded["human_edit"])["errors"])

    def test_source_and_extraction_mismatches(self):
        saved = self.store.save(self.snapshot, self.state)
        with self.assertRaisesRegex(store.StoreError, "Source document mismatch"):
            self.store.open(saved["draft_id"], document=dict(DOCUMENT, sha256="b" * 64))
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["provenance"]["new"] = "changed"
        with self.assertRaisesRegex(store.StoreError, "different extraction"):
            self.store.save(snapshot, self.state, draft_id=saved["draft_id"], revision=1)

    def test_tampering_and_schema_versions(self):
        saved = self.store.save(self.snapshot, self.state)
        with closing(sqlite3.connect(self.store.path)) as db, db:
            db.execute("UPDATE snapshots SET data=?", (store.dumps({}),))
        with self.assertRaisesRegex(store.StoreError, "integrity"):
            self.store.open(saved["draft_id"])
        with closing(sqlite3.connect(self.store.path)) as db, db:
            db.execute("PRAGMA user_version=999")
        with self.assertRaisesRegex(store.StoreError, "schema"):
            self.store.prepare()

    def test_record_version_and_malformed_rows_rejected(self):
        saved = self.store.save(self.snapshot, self.state)
        data = store.draft_data(self.state)
        for replacement in (dict(data, schema_version=999), dict(data, row_order=["wrong"])):
            with closing(sqlite3.connect(self.store.path)) as db, db:
                db.execute("UPDATE drafts SET data=?,hash=?",
                           (store.dumps(replacement), store.digest(replacement)))
            with self.assertRaises(store.StoreError):
                self.store.open(saved["draft_id"])

    def test_version_change_requires_renewed_confirmation(self):
        state = edit.confirm_field(self.state, "supplier_name")
        saved = self.store.save(self.snapshot, state)
        with patch.object(store, "compatibility", return_value={"new": "version"}):
            loaded = self.store.open(saved["draft_id"], document=DOCUMENT)
            self.assertFalse(loaded["human_edit"]["confirmed"]["supplier_name"])
            self.assertTrue(loaded["warnings"])
            renewed = edit.confirm_field(loaded["human_edit"], "supplier_name")
            self.store.save(self.snapshot, renewed, draft_id=saved["draft_id"], revision=1)
            reopened = self.store.open(saved["draft_id"], document=DOCUMENT)
            self.assertTrue(reopened["human_edit"]["confirmed"]["supplier_name"])
            self.assertTrue(reopened["human_edit"]["confirmation_history"])

    def test_forged_runtime_confirmation_cannot_be_saved(self):
        state = copy.deepcopy(self.state)
        state["confirmed"]["supplier_name"] = True
        with self.assertRaises(edit.EditStateError):
            self.store.save(self.snapshot, state)
        self.assertEqual(self.store.list_drafts(), [])

    def test_concurrent_revision_and_delete_stale_save(self):
        saved = self.store.save(self.snapshot, self.state)
        second = self.store.save(self.snapshot, self.state,
                                 draft_id=saved["draft_id"], revision=1)
        self.assertEqual(second["revision"], 2)
        with self.assertRaises(store.RevisionConflict):
            self.store.save(self.snapshot, self.state, draft_id=saved["draft_id"], revision=1)
        with self.assertRaises(store.StoreError):
            self.store.delete(saved["draft_id"], 2)
        self.store.delete(saved["draft_id"], 2, confirmed=True)
        with self.assertRaises(store.RevisionConflict):
            self.store.save(self.snapshot, self.state, draft_id=saved["draft_id"], revision=2)
        with closing(sqlite3.connect(self.store.path)) as db, db:
            self.assertEqual(db.execute("SELECT count(*) FROM snapshots").fetchone()[0], 0)

    def test_interrupted_transaction_preserves_previous_revision(self):
        saved = self.store.save(self.snapshot, self.state)
        with closing(sqlite3.connect(self.store.path)) as db, db:
            db.execute("CREATE TRIGGER fail_update BEFORE UPDATE ON drafts "
                       "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
        with self.assertRaises(store.StoreError):
            self.store.save(self.snapshot, self.state, draft_id=saved["draft_id"], revision=1)
        self.assertEqual(self.store.open(saved["draft_id"])["revision"], 1)

    def test_permission_failure_no_fallback(self):
        new_path = Path(self.temp.name) / "unavailable"
        with patch.object(store, "secure_permissions", side_effect=store.StoreError("permissions")):
            with self.assertRaises(store.StoreError):
                store.DraftStore(new_path).prepare()
        self.assertFalse((new_path / "drafts.sqlite3").exists())

    @unittest.skipUnless(os.name == "nt", "Windows DACL check")
    def test_real_nonprivate_acl_is_rejected(self):
        subprocess.run(
            ["icacls.exe", str(self.directory), "/grant", "*S-1-1-0:(OI)(CI)R"],
            check=True, capture_output=True,
        )
        with self.assertRaisesRegex(store.StoreError, "exclusively"):
            self.store.list_drafts()

    def test_document_fingerprint_is_bytes_not_name(self):
        source = Path(self.temp.name) / "synthetic.png"
        source.write_bytes(b"synthetic first content")
        first = store.document_identity(source)
        source.write_bytes(b"synthetic other content")
        self.assertNotEqual(first, store.document_identity(source))


class BoundaryTests(unittest.TestCase):
    def test_disabled_default_and_bad_environments(self):
        self.assertFalse(store.persistence_requested({}))
        for extra in ({}, {"SPACE_ID": "private/space"}, {"GRADIO_SHARE": "1"},
                      {"GRADIO_SERVER_NAME": "0.0.0.0"}, {"GRADIO_ROOT_PATH": "/proxy"}):
            env = {} if not extra else {**FLAGS, **extra}
            with self.assertRaises(store.StoreError):
                store.require_local_environment(env)

    def test_request_guards_and_inactive_boundary(self):
        with patch.dict(os.environ, FLAGS, clear=True):
            boundary = LocalBoundary()
            base = {"client": ("127.0.0.1", 1234), "scheme": "http",
                    "headers": [(b"host", b"127.0.0.1:7862")]}
            self.assertTrue(boundary.request_allowed(base))
            for change in (
                {"client": ("192.168.1.1", 1234)},
                {"headers": [(b"host", b"evil.example")]},
                {"headers": base["headers"] + [(b"origin", b"https://evil.example")]},
                {"headers": base["headers"] + [(b"x-forwarded-for", b"127.0.0.1")]},
            ):
                self.assertFalse(boundary.request_allowed({**base, **change}))
            with self.assertRaises(store.StoreError):
                boundary.authorize()


if __name__ == "__main__":
    unittest.main()
