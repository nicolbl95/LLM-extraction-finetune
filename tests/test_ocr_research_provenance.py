"""PR82: a research OCR run must state, and be cached by, the languages it really used.

`ocr-policy-5` has no language field, so its fingerprint cannot separate a full
`fra+eng` measurement from a silent English-only degradation. These tests pin the
research provenance layer without any OCR subprocess, network or GPU.
"""

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/stress"))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_ocr
import ocr_provenance
from ocr_config import OCR_CONFIG

FAKE_CMD = r"C:\Fake-Tesseract\tesseract.exe"
FAKE_VERSION = "5.4.0.20240606"
FRENCH_READ = {"text": ["marinière"], "left": [20], "top": [80], "width": [40], "height": [12], "conf": [93]}
ENGLISH_READ = {"text": ["mariniere"], "left": [20], "top": [80], "width": [40], "height": [12], "conf": [93]}
IMAGE = Image.new("RGB", (40, 20), "white")


@pytest.fixture(autouse=True)
def isolated_installation(monkeypatch):
    """One pinned installation per test: no listing subprocess, no ambient tessdata."""
    monkeypatch.setattr(ocr_provenance, "_LANGUAGES_CACHE", {})
    monkeypatch.setattr(ocr_provenance, "_ENVIRONMENT_CACHE", {})
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", FAKE_CMD)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: FAKE_VERSION)
    monkeypatch.setattr(pytesseract, "image_to_data", lambda image, **kwargs: dict(FRENCH_READ))


AUTO = object()


@contextmanager
def installed(languages, *, prefix=AUTO):
    """A self-consistent fake Tesseract: a distinct tessdata directory per distinct language set."""
    if prefix is AUTO:
        suffix = "-".join(sorted(languages)) or "none"
        prefix = str(ROOT / ".venv" / f"fake-tessdata-{suffix}")
    with patch.dict(os.environ, {"TESSDATA_PREFIX": prefix}), patch.object(
        pytesseract, "get_languages", return_value=sorted(languages)
    ):
        yield


def prepared_reads():
    """Keep the preprocessing step out of a cache/provenance test; the read itself is the subject."""
    return patch.object(experiment_ocr, "prepare", side_effect=lambda image, method: (image, {"method": method}))


def french_environment(*, require_all=True):
    with installed({"fra", "eng", "osd"}):
        return ocr_provenance.environment(require_all=require_all)


def english_environment():
    with installed({"eng", "osd"}):
        return ocr_provenance.environment(require_all=False)


def test_french_available_produces_a_complete_language_plan():
    with installed({"fra", "eng", "osd"}):
        plan = ocr_provenance.language_plan()
        strict = ocr_provenance.require_languages()
    assert plan["requested"] == ["fra", "eng"]
    assert plan["effective"] == ["fra", "eng"]
    assert plan["missing"] == [] and plan["fallback"] is False
    assert plan["installed"] == ["eng", "fra", "osd"]
    assert strict["effective"] == ["fra", "eng"]


def test_missing_french_is_an_explicit_fallback_that_is_never_labelled_as_french():
    with installed({"eng", "osd"}):
        plan = ocr_provenance.language_plan()
        relaxed = ocr_provenance.environment(require_all=False)
    assert plan["effective"] == ["eng"]
    assert plan["missing"] == ["fra"]
    assert plan["fallback"] is True
    assert relaxed["reader_languages"] == "eng"
    assert relaxed["missing"] == ["fra"], "a degraded run must record that it was degraded"


def test_strict_research_gate_refuses_an_english_only_french_run():
    with installed({"eng", "osd"}, prefix=str(ROOT / ".venv/absent-tessdata")):
        with pytest.raises(RuntimeError, match="fra") as refused:
            ocr_provenance.environment()
        assert "lacks required languages" in str(refused.value)
        assert "absent-tessdata" in str(refused.value), "the refusal must name the tessdata it looked in"
        assert "eng" in str(refused.value), "the refusal must show what was actually installed"

    with installed(set()):
        with pytest.raises(RuntimeError, match="No usable OCR language"):
            ocr_provenance.environment(require_all=False)

    with installed({"eng", "osd"}):
        assert ocr_provenance.language_plan(requested=("eng",))["missing"] == []


@pytest.mark.parametrize("available", [
    frozenset({"fra", "eng", "osd"}), frozenset({"eng", "osd"}), frozenset({"eng"}), frozenset(),
])
def test_research_plan_agrees_with_the_application_plan_in_the_same_environment(available):
    from app import main

    with patch.object(main, "_OCR_LANGUAGES_CACHE", {}), installed(available):
        assert main.ocr_language_plan() == ocr_provenance.language_plan(), (
            "research tooling and the application must split the requested languages identically"
        )


def test_research_and_application_refuse_a_french_run_under_the_same_condition():
    from app import main

    with installed({"eng", "osd"}), patch.object(main, "_OCR_LANGUAGES_CACHE", {}):
        with pytest.raises(RuntimeError, match="fra") as research:
            ocr_provenance.require_languages()
        with pytest.raises(RuntimeError, match="fra") as application:
            main.require_ocr_languages(require_all=True)
    assert "fra" in str(research.value) and "fra" in str(application.value)


def test_cache_keys_separate_an_english_only_read_from_a_french_read():
    image_hash = hashlib.sha256(b"same image bytes").hexdigest()
    french = french_environment()
    english = english_environment()
    assert french["reader_languages"] == "fra+eng"
    assert english["reader_languages"] == "eng"

    def keys(environment):
        return {
            ocr_provenance.cache_key(image_hash, method, psm, environment)
            for method in ("deskew", "clahe") for psm in (4, 6)
        }

    assert len(keys(french)) == 4 and len(keys(english)) == 4
    assert not keys(french) & keys(english), "an English-only cached read must never be served to a French run"

    legacy = {
        hashlib.sha256(f"{image_hash}/{method}/{psm}/v1".encode()).hexdigest()
        for method in ("deskew", "clahe") for psm in (4, 6)
    }
    assert not (keys(french) | keys(english)) & legacy, "versioned keys must not collide with historical entries"


def test_cache_keys_separate_engine_and_policy_changes():
    image_hash = hashlib.sha256(b"same image bytes").hexdigest()
    base = french_environment()
    variants = (
        {**base, "tesseract_version": "5.5.0"},
        {**base, "tesseract_cmd": r"C:\Other-Tesseract\tesseract.exe"},
        {**base, "ocr_fingerprint": "0" * 64},
        {**base, "reader_languages": "fra"},
    )
    keys = {ocr_provenance.cache_key(image_hash, "deskew", 4, variant) for variant in variants}
    assert len(keys) == 4
    assert ocr_provenance.cache_key(image_hash, "deskew", 4, base) not in keys
    assert ocr_provenance.cache_key(image_hash, "deskew", 4, dict(base)) == (
        ocr_provenance.cache_key(image_hash, "deskew", 4, base)
    ), "an unchanged configuration must stay reproducible, not churn keys"


def test_the_same_installation_written_with_different_slashes_keeps_one_cache_namespace():
    """A benchmark re-run from another shell must reuse, not silently fork, its own cache."""
    image_hash = hashlib.sha256(b"same image bytes").hexdigest()
    env = french_environment()
    forward = {**env, "tessdata_prefix": "C:/Fake/tessdata-fr"}
    backward = {**env, "tessdata_prefix": "C:\\Fake\\tessdata-fr"}
    assert ocr_provenance.cache_key(image_hash, "deskew", 4, forward) == (
        ocr_provenance.cache_key(image_hash, "deskew", 4, backward)
    )
    assert ocr_provenance.reusable({"environment": forward}, backward) is True
    assert ocr_provenance.reusable({"environment": {**forward, "tessdata_prefix": "C:/Other/tessdata"}}, backward) is False


def test_entries_without_language_provenance_are_never_served():
    env = french_environment()
    assert ocr_provenance.reusable({"data": ENGLISH_READ}, env) is False
    assert ocr_provenance.reusable({"data": ENGLISH_READ, "environment": None}, env) is False
    assert ocr_provenance.reusable({"data": ENGLISH_READ, "environment": {"reader_languages": "fra+eng"}}, env) is False
    assert ocr_provenance.reusable({"data": FRENCH_READ, "environment": dict(env)}, env) is True


def test_french_read_requests_both_languages_and_records_them(tmp_path, monkeypatch):
    used = []

    def reader(image, **kwargs):
        used.append(kwargs["lang"])
        return dict(FRENCH_READ)

    monkeypatch.setattr(experiment_ocr, "CACHE", tmp_path)
    monkeypatch.setattr(pytesseract, "image_to_data", reader)
    with prepared_reads(), installed({"fra", "eng", "osd"}):
        result = experiment_ocr.read_variant(IMAGE, "deskew", 4)

    assert used == ["fra+eng"]
    assert result["data"]["text"] == ["marinière"]
    assert result["environment"]["effective"] == ["fra", "eng"]
    assert result["environment"]["missing"] == []
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_research_refuses_to_cache_an_english_only_french_read(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_ocr, "CACHE", tmp_path)
    monkeypatch.setattr(pytesseract, "image_to_data", lambda image, **kwargs: dict(ENGLISH_READ))
    with prepared_reads(), installed({"eng", "osd"}):
        with pytest.raises(RuntimeError, match="fra"):
            experiment_ocr.read_variant(IMAGE, "deskew", 4)
    assert list(tmp_path.glob("*.json")) == [], "a refused run must not leave a baseline behind"


def test_legacy_entry_at_the_old_key_is_preserved_and_never_reused(tmp_path, monkeypatch):
    """A historical, provenance-free cache file stays on disk untouched and cannot answer a French read."""
    image_hash = hashlib.sha256(IMAGE.tobytes() + str(IMAGE.size).encode()).hexdigest()
    legacy_label = f"{image_hash}/deskew/4/v1"
    legacy_path = tmp_path / (hashlib.sha256(legacy_label.encode()).hexdigest() + ".json")
    legacy_payload = json.dumps({"data": ENGLISH_READ, "input_sha256": image_hash, "psm": 4})
    legacy_path.write_text(legacy_payload, encoding="utf-8")

    used = []

    def reader(image, **kwargs):
        used.append(kwargs["lang"])
        return dict(FRENCH_READ)

    monkeypatch.setattr(experiment_ocr, "CACHE", tmp_path)
    monkeypatch.setattr(pytesseract, "image_to_data", reader)
    with prepared_reads(), installed({"fra", "eng", "osd"}):
        first = experiment_ocr.read_variant(IMAGE, "deskew", 4)
        second = experiment_ocr.read_variant(IMAGE, "deskew", 4)

    assert used == ["fra+eng"], "the legacy English-only entry must not answer a French read"
    assert first["data"]["text"] == second["data"]["text"] == ["marinière"]
    assert legacy_path.read_text(encoding="utf-8") == legacy_payload, (
        "historical cache entries are segregated, never deleted or rewritten"
    )


def test_environment_records_the_whole_reproducibility_signature():
    env = french_environment()
    assert env["ocr_policy"] == OCR_CONFIG.version
    assert env["ocr_fingerprint"] == OCR_CONFIG.fingerprint()
    assert env["tesseract_cmd"] == FAKE_CMD
    assert env["tesseract_version"] == FAKE_VERSION
    assert env["tessdata_prefix"] == str(ROOT / ".venv" / "fake-tessdata-eng-fra-osd")
    for field in ("requested", "effective", "missing", "fallback", "installed", "reader_languages"):
        assert field in env

    exported = json.dumps(OCR_CONFIG.export(), sort_keys=True)
    assert "fra" not in exported and "language" not in exported, (
        "the frozen policy really carries no language, which is why research keys must"
    )


def test_language_listing_is_queried_once_per_run_not_per_read():
    listings = []

    def get_languages(config=""):
        listings.append(config)
        return ["fra", "eng", "osd"]

    with patch.object(pytesseract, "get_languages", side_effect=get_languages):
        for _ in range(5):
            ocr_provenance.environment()
            ocr_provenance.language_plan()
    assert len(listings) == 1, "provenance must not add a subprocess per image or per call"
