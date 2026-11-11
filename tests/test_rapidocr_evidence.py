"""PR101 RapidOCR adapter tests: deterministic, offline, no engine and no private fixture.

A stand-in result object replaces the engine so every assertion is about the adapter's own work:
region-to-word geometry, the 0-100 confidence scale, the TSV hand-off into the unmodified
``invoice_evidence.from_tsv``, the header-continuation repair, and the safe fallback when RapidOCR
is unavailable.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pytest

import rapidocr_evidence as adapter
from invoice_evidence import Token


class Result:
    """The three arrays RapidOCR returns: boxes as four corner points, scores as 0-1 floats."""

    def __init__(self, regions):
        self.boxes = [region[0] for region in regions]
        self.txts = [region[1] for region in regions]
        self.scores = [region[2] for region in regions]


def region(text, left, top, right, bottom, score=0.95):
    return ([[left, top], [right, top], [right, bottom], [left, bottom]], text, score)


def words(result):
    return adapter.words(result)


def test_one_region_is_split_into_one_token_per_word():
    result = Result([region("Invoice Date 09/01/2025", 0, 10, 300, 30)])
    tokens = words(result)
    assert [token.text for token in tokens] == ["Invoice", "Date", "09/01/2025"]


def test_word_boxes_are_allocated_proportionally_inside_the_region():
    tokens = words(Result([region("ab cdef", 0, 0, 100, 10)]))
    assert tokens[0].box[0] == 0
    assert tokens[0].box[2] == pytest.approx(33.33, abs=0.01)
    assert tokens[1].box[0] == pytest.approx(33.33, abs=0.01)
    assert tokens[1].box[2] == pytest.approx(100, abs=0.01)
    assert all(token.box[1] == 0 and token.box[3] == 10 for token in tokens)


def test_confidence_is_reported_on_the_tesseract_zero_to_hundred_scale():
    tokens = words(Result([region("40,00 €", 0, 0, 60, 10, score=0.874)]))
    assert all(token.confidence == pytest.approx(87.4, abs=0.01) for token in tokens)


def test_empty_and_unscored_regions_are_dropped():
    tokens = words(Result([region("   ", 0, 0, 10, 10), region("kept", 20, 0, 40, 10)]))
    assert [token.text for token in tokens] == ["kept"]


def test_tokens_keep_the_original_monetary_formatting():
    tokens = words(Result([region("15,95 $ 191,40 $", 0, 0, 200, 10)]))
    assert [token.text for token in tokens] == ["15,95", "$", "191,40", "$"]


def test_tsv_handoff_matches_the_shape_from_tsv_expects():
    tokens = words(Result([region("Montant 1 404,00", 0, 0, 160, 12)]))
    data = adapter.to_tsv(tokens)
    assert set(data) == {"text", "conf", "left", "top", "width", "height"}
    assert data["text"] == ["Montant", "1", "404,00"]
    assert len(data["text"]) == len(data["conf"]) == len(data["left"]) == 3
    assert all(width > 0 for width in data["width"])
    assert data["left"] == sorted(data["left"])


def test_regions_are_kept_for_debugging_with_engine_and_box():
    read = adapter.read_regions(Result([region("Sous-total 1 170,00 €", 4, 50, 204, 70)]))
    assert [region_row.engine for region_row in read.regions] == ["rapidocr"]
    assert read.regions[0].box == (4, 50, 204, 70)
    assert read.regions[0].confidence == pytest.approx(95, abs=0.01)
    assert read.evidence.source_kind == "ocr"
    assert read.evidence.preprocessing_used == ["rapidocr"]
    assert read.evidence.rows_provenance == "rapidocr"


def test_a_rate_label_split_over_two_regions_rejoins_the_header_line():
    # ``Prix`` and ``unitaire`` on consecutive bands: the production continuation route needs the
    # second band to start below the first one's bottom edge.
    separated = Result([
        region("Description Quantité Prix Montant", 0, 100, 400, 118),
        region("unitaire", 250, 114, 320, 130),
    ])
    tokens = adapter.repair_header_continuation(words(separated))
    evidence = adapter.evidence_from_tokens(tokens)
    assert any("unitaire" in line.text and "Prix" in line.text for line in evidence.lines)

    distant = Result([
        region("Description Quantité Prix Montant", 0, 100, 400, 118),
        region("unitaire", 250, 400, 320, 416),
    ])
    evidence = adapter.evidence_from_tokens(adapter.repair_header_continuation(words(distant)))
    assert not any(line.text.strip() == "Prix unitaire" or "unitaire Montant" in line.text
                   for line in evidence.lines[:2])


def test_continuation_is_left_alone_when_the_header_has_no_prix_token():
    separated = Result([
        region("Description Quantité Montant", 0, 100, 400, 118),
        region("unitaire", 250, 114, 320, 130),
    ])
    tokens = adapter.repair_header_continuation(words(separated))
    assert [token.box[1] for token in tokens if token.text == "unitaire"] == [114]


def test_read_returns_nothing_when_the_engine_is_unavailable(monkeypatch):
    monkeypatch.setattr(adapter, "_ENGINE", None)
    monkeypatch.setattr(adapter, "_ENGINE_FAILED", "RapidOCR non importable : No module named rapidocr")
    assert adapter.read(Path("whatever.png")) is None
    assert "non importable" in adapter.unavailable_reason()


def test_read_reports_a_failure_instead_of_raising_when_inference_breaks(monkeypatch):
    class Broken:
        def __call__(self, source):
            raise RuntimeError("detectors exploded")

    monkeypatch.setattr(adapter, "_ENGINE", Broken())
    monkeypatch.setattr(adapter, "_ENGINE_FAILED", None)
    read = adapter.read(Path("whatever.png"))
    assert read.evidence is None
    assert "detectors exploded" in read.failure


def test_the_engine_is_initialised_once_and_reused(monkeypatch):
    built = []

    class Fake:
        def __init__(self):
            built.append(self)

        def __call__(self, source):
            return Result([region("Total 10,00", 0, 0, 80, 10)])

    monkeypatch.setattr(adapter, "_ENGINE", None)
    monkeypatch.setattr(adapter, "_ENGINE_FAILED", None)
    monkeypatch.setattr("rapidocr.RapidOCR", Fake)
    assert adapter.read(Path("a.png")).evidence is not None
    assert adapter.read(Path("b.png")).evidence is not None
    assert len(built) == 1, "the second read must reuse the initialised engine"


def test_a_second_read_reuses_the_module_level_engine(monkeypatch):
    engine = lambda source: Result([region("Rien 0,00", 0, 0, 40, 8)])
    monkeypatch.setattr(adapter, "_ENGINE", engine)
    monkeypatch.setattr(adapter, "_ENGINE_FAILED", None)
    assert adapter.read(Path("a.png")).evidence is not None
    assert adapter.read(Path("b.png")).evidence is not None
    assert adapter._ENGINE is engine


def test_tokens_are_page_zero_and_keep_the_region_vertical_extent():
    tokens = words(Result([region("a bb", 0, 40, 40, 55), region("c", 0, 80, 10, 95)]))
    assert all(token.page == 0 for token in tokens)
    assert [token.box[1] for token in tokens] == [40, 40, 80]
