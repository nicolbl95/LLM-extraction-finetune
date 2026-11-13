"""PR141: the generated JSON must never become a file the file route can serve.

Gradio's file route is the exposure: `route_utils.file_fetch` only asks whether a path was
created by this app, never who created it or whether they are still here. A generated JSON
written in the upload folder was therefore downloadable by every visitor for as long as it
existed, whatever its name. The download is now serialised in memory and handed to the
browser as text, where the `gr.Code` toolbar turns it into a local blob; nothing reaches the
served folder, so there is no URL left to replay.

These tests pin the absence of a file, which is what buys the isolation. Uploads keep the
bounded lifetime `delete_cache` gives them — that residual exposure is documented in the
PR141 report, not here.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app import main

PAYLOAD = {
    "document_type": "invoice",
    "supplier_name": None,
    "invoice_number": None,
    "invoice_date": None,
    "currency": "EUR",
    "total_ht": None,
    "line_items": [],
}
_FRAMES = [(None, main.PROCESSING_STATUS), (PAYLOAD, "Lecture terminée.")]


@pytest.fixture
def served_app():
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ, {"GRADIO_TEMP_DIR": directory}), \
                patch.object(main, "get_extractor", side_effect=AssertionError("No model load")):
            yield main.build_app(), Path(directory)


def recruiter_frames(served_root):
    with patch.object(main, "run_extraction", lambda *a, **k: iter(_FRAMES)), \
            patch.object(main, "deployed_revision", return_value="0" * 40), \
            patch.object(main, "ocr_availability", return_value=(True, "ok")), \
            patch.object(main, "ocr_language_plan", return_value={
                "requested": ["fra"], "effective": ["fra"], "missing": [],
            }), patch.dict(os.environ, {"GRADIO_TEMP_DIR": str(served_root)}):
        return list(main.run_recruiter_extraction_ui("document.png"))


def test_no_download_file_is_registered_for_a_sweep_anymore():
    """The sweep was PR140's mitigation; PR141 removes the file instead of its lifetime."""
    assert not hasattr(main, "GENERATED_DOWNLOAD_FILES")


def test_the_download_component_is_the_native_client_side_carrier(served_app):
    app, _ = served_app
    downloads = [block for block in app.blocks.values()
                 if isinstance(block, main.gr.Code) and block.label == "Télécharger le JSON"]
    assert len(downloads) == 1
    assert "download" in downloads[0].buttons


def test_the_download_value_is_json_text_and_not_a_served_path(served_app):
    _, root = served_app
    value = main._download_json(PAYLOAD)
    assert json.loads(value) == PAYLOAD
    assert not Path(value).exists()
    assert str(root) not in value


def served_files(root):
    """The example buttons already cache the three public PNGs; the extraction must add nothing."""
    return {path for path in root.rglob("*") if path.is_file()}


def test_an_extraction_leaves_the_served_folder_untouched(served_app):
    """No new file, therefore no URL: this is what PR141 Phase 4 verifies against a live server."""
    _, root = served_app
    before = served_files(root)
    frames = recruiter_frames(root)
    assert json.loads(frames[-1][-1]) == PAYLOAD
    assert served_files(root) == before, "a generated piece reached the folder the file route serves"
    assert not list(root.rglob("*.json"))


def test_two_visitors_get_their_own_value_and_no_common_reference(served_app):
    _, root = served_app
    first = main._download_json({**PAYLOAD, "supplier_name": "Visitor A"})
    second = main._download_json({**PAYLOAD, "supplier_name": "Visitor B"})
    assert json.loads(first)["supplier_name"] == "Visitor A"
    assert json.loads(second)["supplier_name"] == "Visitor B"
    assert "Visitor B" not in first and "Visitor A" not in second
    assert not list(root.rglob("*.json"))


def test_uploads_still_keep_their_bounded_lifetime(served_app):
    """PR140's `delete_cache` is kept for the documents the OCR needs on disk."""
    app, _ = served_app
    frequency, age = app.delete_cache
    assert (frequency, age) == (60, 600)
    assert age > frequency
