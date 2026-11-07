"""Standard example downloads must survive cleanup; user uploads must not."""

import hashlib
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from gradio import route_utils

from app import main


@pytest.fixture
def cache_app():
    with tempfile.TemporaryDirectory(dir=main.ROOT / "tests") as directory:
        root = Path(directory)
        with patch.dict(main.os.environ, {"GRADIO_TEMP_DIR": str(root / "cache")}), \
                patch.object(main, "get_extractor", side_effect=AssertionError("No model load")):
            app = main.build_app()
        dependencies = {item["api_name"]: item for item in app.config["dependencies"]}
        event = dependencies["load_example"]
        upload = app.blocks[event["outputs"][0]]
        yield root, app, upload, app.fns[event["id"]].fn


def test_standard_example_file_delivery_survives_cache_cleanup(cache_app):
    _, app, upload, select = cache_app
    for index, (_, original) in enumerate(main.REGIONAL_EXAMPLES):
        selected = select((index, [str(original)]))["value"]
        cached = Path(selected["path"])
        assert cached.is_file()
        assert hashlib.sha256(cached.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()
        route_utils.delete_files_created_by_app(app, age=0)
        assert cached.is_file(), "Scheduled cleanup invalidated the example button's cached file"
        response = route_utils.file_fetch(
            str(cached), SimpleNamespace(headers={}), app, upload.GRADIO_CACHE,
        )
        assert response.status_code == 200
        assert select((index, [str(original)]))["value"]["path"] == str(cached)


def test_cache_protection_does_not_retain_new_uploads_or_json(cache_app):
    root, app, upload, _ = cache_app
    ordinary = root / "ordinary.png"
    ordinary.write_bytes(b"ordinary uploaded content")
    uploaded_cache = Path(upload.move_resource_to_block_cache(str(ordinary)))
    download = next(block for block in app.blocks.values() if isinstance(block, main.gr.DownloadButton))
    result = root / "result.json"
    result.write_text('{"supplier_name": null}', encoding="utf-8")
    result_cache = Path(download.move_resource_to_block_cache(str(result)))
    assert str(uploaded_cache) not in upload.keep_in_cache
    time.sleep(1.1)
    route_utils.delete_files_created_by_app(app, age=0)
    assert not uploaded_cache.exists()
    assert not result_cache.exists()
    assert ordinary.exists() and result.exists()
