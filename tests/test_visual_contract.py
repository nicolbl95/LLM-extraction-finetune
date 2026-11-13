"""PR135 phase 3: the recruiter landing page is frozen by assertions, not by prose.

`tests/test_app.py` pins the strings that are rendered today. This module pins the strings the
recruiter page is *allowed* to render, so a future edit cannot quietly reintroduce a capability
claim. Two contract items are known-unmet against the current `app/main.py`; they carry
`xfail(strict=True)` so the gap stays registered in the suite, the suite stays failure-free, and
an implementation that satisfies the contract has to delete the marker to stay green.

The model-backend claim is no longer one of them: PR140 phase 4 made `INVOICE_MODEL=0` render
the OCR wording instead, so `test_no_model_backend_claim` now asserts the contract directly.
"""

import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import main  # noqa: E402

TITLE = "Extraction standardisée de données de factures"
SUBTITLE_CONTRACT = (
    "Recruiter demo",
    "OCR demonstration only",
    "French and English invoices",
    "Démonstration recruteur",
    "Démonstration OCR",
    "factures françaises et anglaises",
)
MODEL_CLAIMS = (
    "modèle spécialisé",
    "modèle fine-tuné",
    "model spécialisé",
    "fine-tuned model",
    "fine-tune",
)
THREE_LOCALE_CLAIMS = ("américain", "britannique", "états-unis")
INTERNAL_DIAGNOSTICS = (
    "tesseract",
    "rapidocr",
    "tessdata",
    "lora",
    "adapter",
    "checkpoint",
    "gpu",
    "diagnostic",
    "ocr_calls",
    "fallback",
)
REVIEW_UI_KINDS = ("accordion", "dataframe", "textbox", "tabs", "checkbox", "dropdown", "html")
RECRUITMENT_WORDING = (
    "candidature",
    "postuler",
    "offre d'emploi",
    "hiring",
    "join us",
    "alternance",
    "stage",
    "cdi",
    "cdd",
    "salaire",
    "rémunération",
)
QUALITY_METRIC_PATTERNS = (
    re.compile(r"precision", re.I),
    re.compile(r"recall", re.I),
    re.compile(r"\bf1\b", re.I),
    re.compile(r"accuracy", re.I),
    re.compile(r"exactitude", re.I),
    re.compile(r"précision", re.I),
    re.compile(r"fiabilit", re.I),
    re.compile(r"\d+(?:[.,]\d+)?\s*%"),
    re.compile(r"\d+(?:[.,]\d+)?\s*points? de pourcen", re.I),
)
CURRENT_TAGLINE = (
    "Transformez vos factures françaises, britanniques et américaines en données JSON structurées "
    "avec un modèle spécialisé pour obtenir un output clair des données importantes standardisées."
)
OCR_ONLY_CLAIMS = ("réalisée par OCR", "recoupées entre plusieurs passes de lecture")
MODEL_NOTICE = "traitement OCR et modèle"


@contextmanager
def environment(**values):
    with patch.dict(os.environ, {k: v for k, v in values.items() if v is not None}, clear=False):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
        yield


def landing_page(space_id=None, invoice_model="1"):
    """The recruiter page exactly as `build_app()` renders it, without touching a model backend.

    `invoice_model` pins `INVOICE_MODEL` so the copy is deterministic instead of following the
    ambient deployment switch; None removes the variable and inherits its default of on.
    """
    with environment(SPACE_ID=space_id, INVOICE_MODEL=invoice_model):
        with patch.object(main, "get_extractor") as backend:
            demo = main.build_app()
    backend.assert_not_called()
    return demo


def components(demo, kind):
    return [c for c in demo.config["components"] if c["type"] == kind]


def markdown_texts(demo):
    return [c["props"]["value"] for c in components(demo, "markdown")]


def landing_text(demo):
    return "\n".join(text for text in markdown_texts(demo) if text).casefold()


def test_exact_title_rendered():
    demo = landing_page()
    assert demo.config["title"] == TITLE
    heading = markdown_texts(demo)[0]
    assert heading.startswith(f"# {TITLE}\n"), heading
    assert heading.count(TITLE) == 1, heading


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PR135.1 recruiter subtitle is not rendered anywhere in app/main.py; the tagline is "
        f"'{CURRENT_TAGLINE}'. Writing it is a production edit this campaign forbids."
    ),
)
def test_bilingual_recruiter_subtitle():
    text = landing_text(landing_page())
    for phrase in SUBTITLE_CONTRACT:
        assert phrase.casefold() in text, phrase


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Three-locale claim rendered twice (app/main.py:1505 'françaises, britanniques et "
        "américaines', :1525 'France / Royaume-Uni / États-Unis'); the validated OCR language "
        "set is fra+eng."
    ),
)
def test_no_three_locale_claim():
    text = landing_text(landing_page())
    for claim in THREE_LOCALE_CLAIMS:
        assert claim not in text, claim


def test_no_model_backend_claim():
    for space_id in (None, "dummy/space"):
        text = landing_text(landing_page(space_id=space_id, invoice_model="0"))
        for claim in MODEL_CLAIMS:
            assert claim.casefold() not in text, claim
        for claim in OCR_ONLY_CLAIMS:
            assert claim.casefold() in text, claim
        assert MODEL_NOTICE.casefold() not in text, space_id


def test_model_backend_claim_only_accompanies_an_enabled_model():
    """The honest copy is a switch, not a deletion: with the model on the claim stays true."""
    for space_id in (None, "dummy/space"):
        text = landing_text(landing_page(space_id=space_id, invoice_model="1"))
        assert "modèle spécialisé" in text, space_id
        for claim in OCR_ONLY_CLAIMS:
            assert claim.casefold() not in text, claim
        if space_id:
            assert MODEL_NOTICE.casefold() in text, space_id


@pytest.mark.parametrize("space_id", [None, "dummy/space"])
def test_no_fabricated_quality_metrics(space_id):
    text = landing_text(landing_page(space_id=space_id))
    for pattern in QUALITY_METRIC_PATTERNS:
        assert pattern.search(text) is None, pattern.pattern


@pytest.mark.parametrize("seconds", [7.01, None, float("nan")])
def test_performance_block_states_only_a_duration(seconds):
    rendered = main._performance_summary(seconds)
    for pattern in QUALITY_METRIC_PATTERNS:
        assert pattern.search(rendered) is None, pattern.pattern
    assert rendered.startswith("### Performances\n"), rendered
    assert "Temps de traitement" in rendered, rendered


@pytest.mark.parametrize("space_id", [None, "dummy/space"])
def test_no_internal_diagnostics(space_id):
    text = landing_text(landing_page(space_id=space_id))
    for marker in INTERNAL_DIAGNOSTICS:
        assert marker not in text, marker


def test_review_ui_absent():
    demo = landing_page()
    for kind in REVIEW_UI_KINDS:
        assert not components(demo, kind), kind


@pytest.mark.parametrize("space_id", [None, "dummy/space"])
def test_no_recruitment_wording(space_id):
    text = landing_text(landing_page(space_id=space_id))
    for wording in RECRUITMENT_WORDING:
        assert wording not in text, wording


def test_single_upload_and_extraction_controls():
    demo = landing_page()
    uploads = components(demo, "file")
    assert len(uploads) == 1, uploads
    assert uploads[0]["props"]["label"] == "Importer une facture — PDF ou image"
    assert uploads[0]["props"]["file_count"] == "single"
    assert uploads[0]["props"]["file_types"] == [".pdf", ".png", ".jpg", ".jpeg"]
    buttons = components(demo, "button")
    assert [c["props"]["value"] for c in buttons] == ["Extraire les données"]
    assert buttons[0]["props"]["variant"] == "primary"
    assert len(components(demo, "code")) == 1
    assert len(components(demo, "json")) == 1
    assert len(components(demo, "image")) == 1
