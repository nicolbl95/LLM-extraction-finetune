"""Experimental line-item recovery: geometric reconstruction, confirmed by an independent engine.

OFF by default (`INVOICE_TABLE_RECOVERY`). When a document has no production rows, the geometric
candidate proposes rows; each cell must also be read identically by an independent OCR engine on the
same printed cell. The engine can only veto: the value written into the evidence is always the
corroborated Tesseract text, never the engine's output, never a product. A cell whose readings
conflict is dropped, a row without confirmed cells is dropped, and every refusal is recorded.
"""

import logging
import os
from decimal import Decimal
from functools import lru_cache

import numpy as np
from PIL import Image

import table_reconstruction as reconstruction
import table_shadow
from invoice_evidence import Number, TableRow, TextLine

LOGGER = logging.getLogger(__name__)

FLAG_NAME = "INVOICE_TABLE_RECOVERY"
_TRUTHY = {"1", "true", "yes", "on"}
CROP_SCALE = 2


def enabled() -> bool:
    return os.environ.get(FLAG_NAME, "").strip().casefold() in _TRUTHY


@lru_cache(maxsize=1)
def get_engine():
    """The independent reader, loaded once. None when the research dependency is absent."""
    try:
        from rapidocr import RapidOCR
    except ImportError:  # pragma: no cover - depends on the research venv
        return None
    try:
        return RapidOCR()
    except Exception:  # pragma: no cover - engine failure must never block a document
        LOGGER.exception("Independent OCR engine unavailable")
        return None


def _frame_box(position):
    factor = CROP_SCALE / table_shadow.scale_of(position["family"])
    x0, y0, x1, y1 = position["pixel"]
    return (x0 * factor, y0 * factor, x1 * factor, y1 * factor)


def _window(report, row, role, size, pad_x=6, pad_y=0.25):
    cell = (row.get("cells") or {}).get(role)
    if not cell:
        return None
    boxes = [_frame_box(pos) for candidate in (row.get("cells") or {}).values()
             for pos in candidate.get("positions", [])]
    if not boxes:
        return None
    top = min(box[1] for box in boxes)
    bottom = max(box[3] for box in boxes)
    height = max(8.0, bottom - top)
    top, bottom = top - pad_y * height, bottom + pad_y * height
    span = report["axes"][report["pinned_roles"][role]]["span"]
    width, page_height = size
    return (int(max(0, span[0] * width - pad_x)), int(max(0, top)),
            int(min(width, span[1] * width + pad_x)), int(min(page_height, bottom)))


def _engine_value(engine, image):
    out = engine(image, use_det=False, use_cls=False, use_rec=True)
    texts = list(out.txts or [])
    if len(texts) != 1:
        return None
    cleaned = reconstruction.CURRENCY.sub("", texts[0]).strip()
    if len(cleaned.split()) > 1:
        return None
    return reconstruction.parse_money(cleaned)


def _row_ordinals(rows, floor):
    """Document ordinals from geometry, or None when the document cannot prove them.

    Ordinal 0 is the first row only when it actually sits directly under the table floor (within
    one and a half row heights). Further rows need a reliable pitch: the gaps between consecutive
    rows must be regular, otherwise a missing row is indistinguishable from a normal gap and the
    ordinal stays unresolved rather than guessed.
    """
    tops = [float(row["band_top"]) for row in rows]
    heights = [float(row["band_bottom"]) - float(row["band_top"]) for row in rows]
    ordinals, confidences = [], []
    pitch = None
    if len(tops) >= 3:
        gaps = sorted(b - a for a, b in zip(tops, tops[1:]) if b > a)
        if gaps and gaps[0] > 0 and gaps[-1] <= 2.5 * gaps[0]:
            pitch = gaps[0]
    for index, top in enumerate(tops):
        if index == 0:
            close_to_floor = bool(floor) and 0 <= top - floor < 1.5 * max(heights[0], 1e-6)
            ordinals.append(0 if close_to_floor else None)
            confidences.append("high" if close_to_floor else None)
            continue
        if pitch and tops[0] is not None:
            offset = top - tops[0]
            ordinal = int(round(offset / pitch))
            residual = abs(offset - ordinal * pitch)
            if ordinal >= 0 and residual <= 0.3 * pitch:
                ordinals.append(ordinal); confidences.append("high")
                continue
        ordinals.append(None); confidences.append(None)
    return ordinals, confidences


HEADER_CUES = __import__("re").compile(
    r"(description|d[ée]signation|libell[ée]|quantit|qty|prix|price|montant|amount|unit[ée]?)",
    __import__("re").I)
SUMMARY_CUES = __import__("re").compile(
    r"^(total|sous[- ]?total|subtotal|tva|tva|vat|tax|montant total|grand total)", __import__("re").I)


def _band_of(centre, rules):
    return next((index for index, (top, bottom) in enumerate(zip(rules, rules[1:]))
                 if top <= centre <= bottom), None)


def _lattice_ordinals(rows, rules, header_lines, evidence_lines, cell_boxes):
    """Ordinals counted from the first physical item band, never from a recovered candidate.

    The grid's first band is the first item row when every header label sits above the grid box;
    when the header lies inside the grid, the item region starts at the band below it. Bands are
    counted whether or not their cells were recognised, and a summary label ends the item region.
    """
    inside = {_band_of((line.box[1] + line.box[3]) / 2, rules) for line in header_lines}
    inside.discard(None)
    above = [line for line in header_lines if line.box[3] < rules[0]]
    if inside:
        if min(inside) != 0:
            return [None] * len(rows), [None] * len(rows)
        origin = max(inside) + 1
    elif above and len(above) == len(header_lines):
        # The whole header sits above the detected grid: its first band is the first item row.
        origin = 0
    else:
        return [None] * len(rows), [None] * len(rows)
    summary_bands = set()
    for line in evidence_lines:
        if SUMMARY_CUES.match(line.text.strip()):
            band = _band_of((line.box[1] + line.box[3]) / 2, rules)
            if band is not None:
                summary_bands.add(band)
    ordinals, confidences = [], []
    for position, row in enumerate(rows):
        boxes = (cell_boxes or [])[position] if cell_boxes and position < len(cell_boxes) else []
        if boxes:
            # The confirmed cells are the row's physical footprint; a band that merged two printed
            # rows cannot be used, so every cell centre must fall inside one and the same band.
            bands = {_band_of((box[1] + box[3]) / 2, rules) for box in boxes}
            bands.discard(None)
            band = bands.pop() if len(bands) == 1 else None
        else:
            band = _band_of((float(row["band_top"]) + float(row["band_bottom"])) / 2, rules)
        if band is None or band < origin or any(origin <= other < band for other in summary_bands):
            ordinals.append(None); confidences.append(None); continue
        ordinals.append(band - origin)
        confidences.append("lattice")
    return ordinals, confidences

def recover(evidence, readings, *, rotated_size=None, prepared=None, engine=None) -> object:
    """Attach recovered rows to the evidence when the flag is on; return the same object."""
    if not enabled():
        return evidence
    if evidence.rows:
        evidence.table_recovery = {"flag": FLAG_NAME, "enabled": True, "status": "not_triggered",
                                   "reason": "the production parser already produced rows"}
        return evidence
    report = {"flag": FLAG_NAME, "enabled": True, "status": "skipped", "rows": 0, "cells": 0,
              "refused": [], "reason": None}
    if rotated_size is None or not rotated_size[0] or not prepared:
        report["reason"] = "no coordinate frame"
        evidence.table_recovery = report
        return evidence
    engine = engine if engine is not None else get_engine()
    if engine is None:
        report["reason"] = "independent OCR engine unavailable"
        evidence.table_recovery = report
        return evidence
    try:
        reads = table_shadow.reads_of(readings, rotated_size)
        families = {reconstruction.family_of(key) for key in reads}
        if len(families) < 2:
            report.update(status="skipped", reason="one preprocessing family only")
            evidence.table_recovery = report
            return evidence
        result = reconstruction.reconstruct(reads)
        rows = result.get("rows") or []
        cell_boxes = [
            [tuple(pos["box"]) for cell in (row.get("cells") or {}).values()
             for pos in cell.get("positions", [])]
            for row in rows
        ]
        ordinals, confidences = _row_ordinals(rows, result.get("table_floor"))
        lattice = None
        try:
            reference_pixels = np.asarray((prepared.get("deskew") or next(iter(prepared.values()))).convert("L"))
            triggered, box, _ = table_shadow.grid_trigger(reference_pixels)
            if triggered and box:
                lattice = table_shadow.rule_lattice(reference_pixels, box)
                frame_w, frame_h = reference.size
                normalise = lambda line: type("L", (), {
                    "text": line.text,
                    "box": (line.box[0] / frame_w, line.box[1] / frame_h,
                            line.box[2] / frame_w, line.box[3] / frame_h)})()
                header_lines = [normalise(line) for line in evidence.lines
                                if HEADER_CUES.search(line.text)]
                page_lines = [normalise(line) for line in evidence.lines]
                if len(lattice["rules"]) >= 2 and header_lines:
                    ordinals, confidences = _lattice_ordinals(
                        rows, lattice["rules"], header_lines, page_lines, cell_boxes)
        except Exception:  # noqa: BLE001 - diagnostics only, never block recovery
            lattice = None
        reference = prepared.get("deskew") or next(iter(prepared.values()))
        confirmed_rows, refusals = [], []
        for index, row in enumerate(rows):
            cells = {}
            for role in reconstruction.CELL_ROLES:
                cell = (row.get("cells") or {}).get(role)
                if not cell:
                    continue
                window = _window(result, row, role, reference.size)
                if window is None:
                    refusals.append({"row": index, "role": role, "reason": "no cell window"})
                    continue
                crop = reference.crop(window).convert("RGB")
                crop = crop.resize((crop.width * CROP_SCALE, crop.height * CROP_SCALE),
                                   Image.Resampling.LANCZOS)
                value = _engine_value(engine, np.asarray(crop))
                if value is None or abs(value - Decimal(str(cell["value"]))) > Decimal("0.01"):
                    refusals.append({"row": index, "role": role, "candidate": str(cell["value"]),
                                     "engine": str(value) if value is not None else None,
                                     "reason": "engines disagree"})
                    continue
                position = cell["positions"][0]
                cells[role] = [Number(cell["raw"][0] if isinstance(cell["raw"], list) else str(cell["raw"]),
                                      Decimal(str(cell["value"])), 0, tuple(position["box"]))]
            if not cells:
                refusals.append({"row": index, "reason": "no confirmed cell"})
                continue
            band_text = str(row.get("band_text") or "")
            confirmed_rows.append(TableRow(
                description=str(row.get("detected_text") or band_text)[:120],
                row_index=ordinals[index] if index < len(ordinals) else None,
                row_evidence={
                    "page": 0,
                    "row_index": ordinals[index] if index < len(ordinals) else None,
                    "row_index_confidence": confidences[index] if index < len(confidences) else None,
                    "printed": {role: cell[0].raw for role, cell in cells.items()},
                    "positions": {role: list(cell[0].box) for role, cell in cells.items()},
                    "refused_cells": [item for item in refusals if item.get("row") == index],
                },
                cells=cells,
                line=TextLine(band_text, [], 0, (0.0, float(row["band_top"]), 1.0,
                                                  float(row["band_bottom"]))),
            ))
        evidence.rows = confirmed_rows
        if confirmed_rows:
            evidence.rows_provenance = "recovered"
        report.update(status="recovered" if confirmed_rows else "refused",
                      rows=len(confirmed_rows),
                      cells=sum(len(row.cells) for row in confirmed_rows),
                      refused=refusals, lattice=lattice)
    except Exception:  # noqa: BLE001 - the experiment must never fail a document
        LOGGER.exception("Table recovery failed")
        evidence.rows = []
        report.update(status="error", reason="recovery failed")
    evidence.table_recovery = report
    return evidence
