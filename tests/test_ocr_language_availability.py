"""PR81: a missing OCR language must never degrade French extraction silently."""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

FRENCH_TOKENS = [
    ("FACTURE", 20, 10), ("FR-001", 90, 10),
    ("Joanna", 20, 30), ("Binet", 70, 30), ("29/01/2019", 200, 30),
    ("Description", 20, 60), ("Quantity", 200, 60), ("Net", 280, 60), ("amount", 330, 60),
    ("Petit", 20, 80), ("marinière", 70, 80), ("uniforme", 150, 80), ("en", 220, 80), ("bleu", 250, 80),
    ("2", 200, 80), ("15,00", 270, 80), ("30,00", 340, 80),
    ("Sous", 20, 110), ("total", 70, 110), ("30,00", 340, 110),
    ("Total", 20, 130), ("TTC", 70, 130), ("30,00", 340, 130),
]


def tsv_for(language: str) -> dict:
    """Reproduce the one real difference between the two families: an English read has no diacritics."""
    words = [
        word.replace("marinière", "mariniere") if "fra" not in language else word
        for word, _, _ in FRENCH_TOKENS
    ]
    return {
        "text": words,
        "left": [left for _, left, _ in FRENCH_TOKENS],
        "top": [top for _, _, top in FRENCH_TOKENS],
        "width": [40] * len(words),
        "height": [12] * len(words),
        "conf": [93] * len(words),
    }


def language_reader(installed: frozenset[str]):
    """A Tesseract stand-in that only produces diacritics for a read that actually requested fra."""
    used: list[str] = []

    def image_to_data(image, **kwargs):
        used.append(kwargs["lang"])
        return tsv_for(kwargs["lang"])

    return used, image_to_data


def read_with(app_main, installed: frozenset[str]):
    from PIL import Image

    used, image_to_data = language_reader(installed)
    with patch.object(app_main, "installed_ocr_languages", return_value=installed), patch.object(
        app_main.pytesseract, "image_to_data", side_effect=image_to_data
    ):
        evidence_text = app_main.ocr_image(Image.new("RGB", (400, 160)))
    return evidence_text, set(used)


def test_french_language_available_keeps_the_full_requested_read():
    from app import main

    with patch.object(main, "installed_ocr_languages", return_value=frozenset({"fra", "eng", "osd"})):
        plan = main.ocr_language_plan()
        available, reason = main.ocr_availability()
    assert plan["requested"] == ["fra", "eng"]
    assert plan["effective"] == ["fra", "eng"]
    assert plan["missing"] == [] and plan["fallback"] is False
    assert (available, reason) == (True, "fra+eng disponibles")

    evidence_text, languages = read_with(main, frozenset({"fra", "eng"}))
    assert languages == {"fra+eng"}
    assert evidence_text.evidence.invoice_number == "FR-001"


def test_missing_french_is_reported_as_an_explicit_fallback(caplog):
    from app import main

    with patch.object(main, "installed_ocr_languages", return_value=frozenset({"eng", "osd"})):
        plan = main.ocr_language_plan()
        available, reason = main.ocr_availability()
        with caplog.at_level(logging.WARNING, logger=main.LOGGER.name):
            evidence_text, languages = read_with(main, frozenset({"eng", "osd"}))

    assert plan["effective"] == ["eng"]
    assert plan["missing"] == ["fra"]
    assert plan["fallback"] is True
    assert available is False
    assert "fra" in reason and "eng" in reason
    assert languages == {"eng"}, "an English-only read must never be labelled fra+eng"
    assert "OCR language fallback: requested fra+eng, effective eng, missing fra" in caplog.text
    assert "marinière" not in evidence_text.evidence.text


def test_permitted_fallback_still_reads_the_document_and_extracts_evidence():
    from app import main

    with patch.object(main, "installed_ocr_languages", return_value=frozenset({"eng"})):
        plan = main.require_ocr_languages()
    assert plan["effective"] == ["eng"]

    evidence_text, languages = read_with(main, frozenset({"eng"}))
    evidence = evidence_text.evidence
    assert languages == {"eng"}
    assert evidence.source_kind in {"ocr", "consensus"}
    assert evidence.tokens
    assert evidence.totals.get("total_ht") or evidence.totals.get("total_ttc")


def test_strict_benchmark_mode_refuses_a_missing_required_language(monkeypatch):
    from app import main

    monkeypatch.setenv("TESSDATA_PREFIX", str(ROOT / ".venv/absent-tessdata"))
    with patch.object(main, "installed_ocr_languages", return_value=frozenset({"eng", "osd"})):
        with pytest.raises(RuntimeError, match="fra") as raised:
            main.require_ocr_languages(require_all=True)
        assert "absent" in str(raised.value)
        assert "absent-tessdata" in str(raised.value)
        assert main.require_ocr_languages()["effective"] == ["eng"], (
            "public extraction keeps its permitted fallback; only strict mode refuses"
        )

    with patch.object(main, "installed_ocr_languages", return_value=frozenset({"fra", "eng"})):
        assert main.require_ocr_languages(require_all=True)["missing"] == []

    with patch.object(main, "installed_ocr_languages", return_value=frozenset()):
        with pytest.raises(RuntimeError, match="OCR indisponible"):
            main.require_ocr_languages()


def test_report_states_the_effective_language_configuration():
    from app import main
    from invoice_evidence import EvidenceText, from_tsv

    evidence = from_tsv(tsv_for("eng"), locale="fr-FR")
    evidence.source_kind = "ocr"
    quota = main.gr.Error("ZeroGPU quota exceeded; try again in 36 min")
    cases = {
        frozenset({"fra", "eng"}): (["Langues de lecture : fra+eng."], ["fra+eng demandées"]),
        frozenset({"eng", "osd"}): (["fra+eng demandées", "absentes : fra"], ["Langues de lecture : fra+eng."]),
    }
    for installed, (expected, forbidden) in cases.items():
        with patch.object(main, "document_text", return_value=EvidenceText(evidence)), patch.object(
            main, "predict_invoice", side_effect=quota
        ), patch.object(main, "installed_ocr_languages", return_value=installed):
            outputs = list(main.run_extraction_ui(str(main.EXAMPLES[0][1])))
        technical = outputs[-1][2]
        for fragment in expected:
            assert fragment in technical, (sorted(installed), technical)
        for fragment in forbidden:
            assert fragment not in technical, (sorted(installed), technical)


def test_accented_french_text_survives_only_when_french_is_installed():
    from app import main

    with_accent, languages = read_with(main, frozenset({"fra", "eng"}))
    assert languages == {"fra+eng"}
    assert "marinière" in with_accent.evidence.text

    without_accent, languages = read_with(main, frozenset({"eng"}))
    assert languages == {"eng"}
    assert "marinière" not in without_accent.evidence.text
    assert "mariniere" in without_accent.evidence.text
    assert without_accent.evidence.invoice_number == with_accent.evidence.invoice_number


def test_english_only_request_is_never_flagged_as_degraded():
    from app import main

    with patch.object(main, "installed_ocr_languages", return_value=frozenset({"eng", "osd"})):
        plan = main.ocr_language_plan(requested=("eng",))
        strict = main.require_ocr_languages(requested=("eng",), require_all=True)
    assert plan == {
        "requested": ["eng"], "effective": ["eng"], "missing": [], "fallback": False,
        "installed": ["eng", "osd"], "tessdata_prefix": plan["tessdata_prefix"],
    }
    assert strict["missing"] == []


def test_language_listing_is_cached_and_not_a_per_image_subprocess(monkeypatch):
    from app import main
    from PIL import Image

    monkeypatch.setattr(main, "_OCR_LANGUAGES_CACHE", {})
    listings: list[str] = []

    def get_languages(config=""):
        listings.append(config)
        return ["eng", "fra"]

    _, image_to_data = language_reader(frozenset({"fra", "eng"}))
    with patch.object(main.pytesseract, "get_languages", side_effect=get_languages), patch.object(
        main.pytesseract, "image_to_data", side_effect=image_to_data
    ):
        for _ in range(3):
            main.ocr_image(Image.new("RGB", (40, 20)))
        assert len(listings) == 1, "availability must be cached per installation, not queried per image"

        monkeypatch.setenv("TESSDATA_PREFIX", str(ROOT / ".venv/other-tessdata"))
        main.ocr_image(Image.new("RGB", (40, 20)))
    assert len(listings) == 2, "a different tessdata installation must not reuse the stale listing"
