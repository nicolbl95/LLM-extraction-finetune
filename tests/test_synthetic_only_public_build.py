"""PR142 §2: a hosted build reads nothing but the synthetic documents it shipped with.

PR141 stopped the generated JSON from becoming a public file; the documents a visitor uploads are
still written into the one cache folder that `/gradio_api/file=` serves to everyone, and Gradio has
no session, token or signature to isolate them with. Until that isolation exists, a Space refuses
every document it did not ship.

The refusal is a server gate, not a warning: it lives in `checked_path`, which every read entry
point goes through, so a direct HTTP call to `run_extraction_ui` or `preview_ui` meets it too. The
bundled documents are recognised by content because selecting an example hands the app Gradio's
copy of it, never `app/examples/...`. Local development keeps free upload, and
`INVOICE_FREE_UPLOAD=1` reopens it on a Space once upload isolation exists.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from gradio import processing_utils

from app import main

ROOT = Path(main.__file__).resolve().parents[1]
SPACE = "nicolbl95/llm-extraction-finetune"
GATE = "Cette démonstration n'accepte que ses trois factures d'exemple synthétiques."


@contextmanager
def deployment(**values):
    """Environment exactly as given: a `None` value removes the variable instead of ignoring it."""
    with patch.dict(os.environ, {k: v for k, v in values.items() if v is not None}, clear=False):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
        yield


@pytest.fixture
def on_a_space(tmp_path):
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        yield tmp_path


@pytest.fixture
def locally(tmp_path):
    with deployment(SPACE_ID=None, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        yield tmp_path


@pytest.fixture
def visitor_document(tmp_path):
    """A document the visitor brought — bytes are never read, only gated, so a stub is enough."""
    path = tmp_path / "ma-facture.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"synthetic visitor document")
    return path


def test_a_space_refuses_a_document_it_did_not_ship(on_a_space, visitor_document):
    with pytest.raises(ValueError) as error:
        main.checked_path(str(visitor_document))
    assert str(error.value) == GATE


def test_a_space_still_reads_the_three_bundled_examples(on_a_space):
    assert len(main.EXAMPLES) == 3
    for _, path in main.EXAMPLES:
        assert path.is_relative_to(ROOT / "app/examples"), path
        assert main.checked_path(str(path)) == path


def test_an_example_reaches_the_space_as_gradios_own_copy(tmp_path):
    """What the browser really sends: the cached copy, under a hashed directory, not the repo path."""
    source = ROOT / "app/examples/france.png"
    copy = tmp_path / "0a1b2c3d4e5f6789" / source.name
    copy.parent.mkdir(parents=True)
    copy.write_bytes(source.read_bytes())
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        assert main.checked_path(str(copy)).name == source.name


def test_a_space_refuses_a_renaming_of_a_bundled_example(tmp_path):
    """The gate is about content; a copied fixture with an allowed name is not an allowed document."""
    disguised = tmp_path / "france.png"
    disguised.write_bytes((ROOT / "tests/fixtures/layout-joanna-binet.png").read_bytes())
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        with pytest.raises(ValueError) as error:
            main.checked_path(str(disguised))
        assert str(error.value) == GATE


def test_local_development_keeps_free_upload(locally, visitor_document):
    assert main.free_upload_enabled() is True
    assert main.checked_path(str(visitor_document)) == visitor_document


def test_the_override_reopens_upload_on_a_space(on_a_space, visitor_document):
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD="1"):
        assert main.free_upload_enabled() is True
        assert main.checked_path(str(visitor_document)) == visitor_document


def test_the_override_closes_upload_locally_too(locally, visitor_document):
    with deployment(SPACE_ID=None, INVOICE_FREE_UPLOAD="0"):
        assert main.free_upload_enabled() is False
        with pytest.raises(ValueError) as error:
            main.checked_path(str(visitor_document))
        assert str(error.value) == GATE


def test_a_refused_document_is_never_opened(on_a_space, visitor_document):
    """The gate sits before the reader, so a refusal cannot spend OCR time or touch the file."""
    with patch.object(main, "open_image", side_effect=AssertionError("the reader ran")):
        with patch.object(main, "ocr_image", side_effect=AssertionError("the reader ran")):
            frames = list(main.run_recruiter_extraction_ui(str(visitor_document)))
    assert frames[-1][0] is None
    assert GATE in frames[-1][1]


def test_both_exposed_api_names_refuse(on_a_space, visitor_document):
    """`run_extraction_ui` and `preview_ui` are the two routes a visitor can call directly."""
    with patch.object(main, "ocr_availability", return_value=(True, "ok")), \
            patch.object(main, "ocr_language_plan", return_value={
                "requested": ["fra"], "effective": ["fra"], "missing": [],
            }):
        extraction = list(main.run_recruiter_extraction_ui(str(visitor_document)))
        preview = main.preview_recruiter_ui(str(visitor_document))
    assert GATE in extraction[-1][1]
    assert preview[0] is None
    assert GATE in preview[2]
    assert "Fichier ou demande invalide" not in preview[2]


def test_the_other_read_entry_points_are_gated(on_a_space, visitor_document):
    with pytest.raises(ValueError) as error:
        main.document_text(str(visitor_document))
    assert str(error.value) == GATE
    pages, limitations, _ = main.review_pages(str(visitor_document))
    assert pages == []
    assert limitations


def test_the_upload_widget_is_disabled_on_a_space_but_the_examples_remain(on_a_space):
    """Presentation only: the widget is greyed out, and the refusal above is what protects."""
    with patch.object(main, "get_extractor") as backend:
        configuration = main.build_app().config
    backend.assert_not_called()
    upload = [c for c in configuration["components"] if c["type"] == "file"]
    assert len(upload) == 1
    assert upload[0]["props"]["interactive"] is False
    assert "Téléversement désactivé" in upload[0]["props"]["label"]
    dataset = [c for c in configuration["components"] if c["type"] == "dataset"]
    assert len(dataset) == 1
    assert [sample[0] for sample in dataset[0]["props"]["samples"]] == [
        path.name for _, path in main.EXAMPLES
    ]


def test_the_upload_widget_stays_usable_in_local_development(locally):
    with patch.object(main, "get_extractor"):
        configuration = main.build_app().config
    upload = [c for c in configuration["components"] if c["type"] == "file"]
    assert upload[0]["props"]["interactive"] is not False
    assert upload[0]["props"]["label"] == "Importer une facture — PDF ou image"


def test_only_the_three_public_documents_are_allowlisted():
    """A bundled development layout is not a public document: `EXAMPLES` is the whole allowlist."""
    assert len(main.SYNTHETIC_DOCUMENT_DIGESTS) == 3
    for _, path in main.REGIONAL_EXAMPLES:
        assert processing_utils.hash_file(path) in main.SYNTHETIC_DOCUMENT_DIGESTS
    for _, path in main.ADDITIONAL_EXAMPLES:
        assert processing_utils.hash_file(path) not in main.SYNTHETIC_DOCUMENT_DIGESTS
