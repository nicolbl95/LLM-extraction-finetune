"""Experimental RapidOCR adapter: engine output into the project's evidence structures.

The adapter never invents a value. It converts RapidOCR regions into word-level tokens with boxes and
confidences and hands them to the unmodified ``invoice_evidence.from_tsv``, so the existing header,
column, row and confidence gates decide what becomes a candidate row. Each region keeps its raw text,
score, box, page and engine, and the monetary formatting is preserved in the token text exactly as the
engine printed it.

Everything here is optional: when RapidOCR cannot be imported, cannot start, or fails on an image,
``read`` reports the failure (or returns ``None``) and the caller continues with Tesseract alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

ENGINE_NAME = "rapidocr"

# The only header continuation the production builder already understands, whose precondition is that
# the continuation line starts below the header line's bottom edge. RapidOCR detection boxes are
# looser than Tesseract's and often overlap by a couple of pixels, so the adapter lifts the
# continuation to sit just under that edge. It changes no text, no column order and no rule: it only
# restores the geometry the existing route was written for.
_CONTINUATION_LABELS = frozenset({"unitaire"})


@dataclass(frozen=True)
class Region:
    """One recognised element, with everything the engine actually reported about it."""

    text: str
    box: tuple[float, float, float, float]
    confidence: float
    page: int = 0
    engine: str = ENGINE_NAME


@dataclass
class RapidOcrRead:
    evidence: Any = None
    regions: list[Region] = field(default_factory=list)
    seconds: float = 0.0
    engine_version: str | None = None
    failure: str | None = None


_ENGINE = None
_ENGINE_FAILED: str | None = None


def _engine():
    """Build (once) and return the RapidOCR engine, or ``None`` when it is unavailable."""
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None or _ENGINE_FAILED:
        return _ENGINE
    try:
        from rapidocr import RapidOCR
    except Exception as exc:  # pragma: no cover - depends on the local install
        _ENGINE_FAILED = f"RapidOCR non importable : {exc}"
        return None
    try:
        _ENGINE = RapidOCR()
    except Exception as exc:  # pragma: no cover - model/runtime availability
        _ENGINE_FAILED = f"RapidOCR indisponible : {exc}"
    return _ENGINE


def unavailable_reason() -> str | None:
    _engine()
    return _ENGINE_FAILED


def version() -> str | None:
    try:
        from importlib.metadata import version as distribution_version

        return distribution_version("rapidocr")
    except Exception:
        return None


def _region_box(region: Any) -> tuple[float, float, float, float]:
    points = list(region)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def regions(result: Any) -> list[Region]:
    return [
        Region(str(text).strip(), _region_box(box), round(float(score) * 100, 1))
        for box, text, score in zip(result.boxes, result.txts, result.scores)
        if str(text).strip()
    ]


def words(result: Any) -> list:
    """RapidOCR regions to word tokens, allocating each word a slice of its region box.

    RapidOCR returns whole regions (often a visual line or a cell), while the evidence builder needs
    per-word geometry to split columns. Width is shared proportionally to character count; vertical
    position and height stay the region's own. Confidence is the region score on the 0-100 scale the
    project's thresholds are written for.
    """
    from invoice_evidence import Token

    tokens = []
    for region in regions(result):
        left, top, right, bottom = region.box
        parts = region.text.split()
        characters = sum(len(word) for word in parts)
        used = 0
        for word in parts:
            start = left + (right - left) * (used / characters)
            end = start + (right - left) * (len(word) / characters)
            used += len(word)
            tokens.append(Token(word, region.page,
                                (round(start, 2), top, round(end, 2), bottom), region.confidence))
    return tokens


def to_tsv(tokens: list) -> dict:
    """The array shape ``invoice_evidence.from_tsv`` already reads from Tesseract."""
    return {
        "text": [token.text for token in tokens],
        "conf": [token.confidence for token in tokens],
        "left": [token.box[0] for token in tokens],
        "top": [token.box[1] for token in tokens],
        "width": [token.box[2] - token.box[0] for token in tokens],
        "height": [token.box[3] - token.box[1] for token in tokens],
    }


def evidence_from_tokens(tokens: list, *, locale: str | None = None):
    """Hand the tokens to the unmodified production builder; it decides headers, columns and rows."""
    from invoice_evidence import from_tsv

    evidence = from_tsv(to_tsv(tokens), locale=locale)
    evidence.source_kind = "ocr"
    evidence.preprocessing_used = [ENGINE_NAME]
    evidence.rows_provenance = ENGINE_NAME
    return evidence


def _header_like(line: Any) -> bool:
    from invoice_evidence import _header_matches

    return all(_header_matches(key, line.text) for key in ("description", "quantity", "total_ht"))


def folded_text(text: str) -> str:
    from invoice_evidence import folded

    return folded(text)


def repair_header_continuation(tokens: list) -> list:
    """Let a lone ``unitaire`` region satisfy the production continuation precondition.

    Only the continuation tokens move, and only vertically; text, order and every other token are
    untouched, and new tokens are built because ``Token`` is frozen.
    """
    from invoice_evidence import positioned_lines

    lines = positioned_lines(tokens)
    for index, line in enumerate(lines[:-1]):
        following = lines[index + 1]
        if folded_text(following.text) not in _CONTINUATION_LABELS or not _header_like(line):
            continue
        prix = next((token for token in line.tokens if folded_text(token.text) == "prix"), None)
        if prix is None or line.box[3] - following.box[1] <= 0:
            continue
        shift = line.box[3] - following.box[1] + 1
        moved = {id(token) for token in following.tokens}
        return [
            type(token)(token.text, token.page,
                        (token.box[0], token.box[1] + shift, token.box[2], token.box[3] + shift),
                        token.confidence)
            if id(token) in moved else token
            for token in tokens
        ]
    return tokens


def read_regions(result: Any, *, locale: str | None = None) -> RapidOcrRead:
    """Convert one engine result into evidence, keeping the reported regions."""
    started = time.perf_counter()
    try:
        evidence = evidence_from_tokens(repair_header_continuation(words(result)), locale=locale)
    except Exception as exc:
        return RapidOcrRead(None, [], 0.0, version(), f"{type(exc).__name__}: {exc}")
    return RapidOcrRead(evidence, regions(result), round(time.perf_counter() - started, 2),
                        version(), None)


def read(image: Any, *, locale: str | None = None) -> RapidOcrRead | None:
    """Run RapidOCR on an image and return pipeline-compatible evidence.

    ``None`` means the engine is unavailable; a returned read with ``evidence is None`` means the
    engine was present but failed on this image. Neither case raises.
    """
    engine = _engine()
    if engine is None:
        return None
    try:
        return read_regions(engine(image), locale=locale)
    except Exception as exc:
        return RapidOcrRead(None, [], 0.0, version(), f"{type(exc).__name__}: {exc}")
