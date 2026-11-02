"""Shadow-mode table reconstruction: measured and recorded, never merged into the extraction.

The flag (``INVOICE_TABLE_SHADOW``, off by default) turns on a second look at a document the
existing pipeline could not put into rows. It reuses the OCR reads the pipeline already paid for,
adds at most two extra reads when the printed rules prove a ruled grid exists, and writes what it
found into ``evidence.table_shadow`` as a diagnostic report.

Nothing here can reach the extraction result: the report is a separate field, no candidate row is
appended to ``evidence.rows``, no value is ever taken from arithmetic alone, descriptions stay
diagnostic, and every failure - including the shadow's own deadline - is recorded here instead of
being raised into the request. The model is never called from this module.
"""

import logging
import os
import time
from decimal import Decimal

import table_reconstruction as reconstruction
from invoice_evidence import DocumentEvidence, from_tsv

LOGGER = logging.getLogger(__name__)

FLAG_NAME = "INVOICE_TABLE_SHADOW"
_TRUTHY = {"1", "true", "yes", "on"}
# A shadow pass may never extend a user request: it stops itself rather than the pipeline.
MAX_SHADOW_SECONDS = 5.0
MIN_SECONDS_FOR_EXTRA_READS = 3.0
# The two preprocessings every document already received. A grid read stays inside its own family:
# two segmentations or two preprocessings of one engine are corroboration, never independent proof.
GRID_FAMILIES = ("deskew", "clahe")
MAX_EXTRA_CALLS = 2
TABLE_PSM = 6
STRIP_MARGIN = 0.02


class ShadowDeadline(Exception):
    """The shadow pass spent its own time allowance; recorded, never raised any further."""


def shadow_enabled() -> bool:
    return os.environ.get(FLAG_NAME, "").strip().casefold() in _TRUTHY


def scale_of(family: str) -> int:
    """The scale `ocr_preprocessing.transform` applies to this family."""
    return 3 if family == "scale3" else 2


def reads_of(readings: dict, rotated_size) -> dict:
    """Production OCR reads, each normalised against the frame it was read in."""
    reads = {}
    for family, entries in readings.items():
        scale = scale_of(family)
        size = (rotated_size[0] * scale, rotated_size[1] * scale)
        for index, evidence in enumerate(entries):
            key = "%s_%d" % (family, index)
            reads[key] = reconstruction.words_of(
                [(token.text, token.box, token.confidence) for token in evidence.tokens],
                family=family, read=key, size=size,
            )
    return {key: words for key, words in reads.items() if words}


def grid_trigger(pixels) -> tuple[bool, tuple | None, str]:
    """A ruled table: several horizontal rules sharing an x span, crossed by vertical rules.

    The measurement is purely geometric - long thin dark runs - so no supplier, template or
    vocabulary is involved, and a page with no grid is charged nothing but this pass.
    """
    import cv2

    ink = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = ink.shape
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 25), 1)))
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 40))))
    rows, columns = [], []
    for contour in cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= width * 0.3 and h <= 6:
            rows.append((x, y, w, h))
    for contour in cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= height * 0.05 and w <= 6:
            columns.append((x, y, w, h))
    rows.sort(key=lambda box: box[1])
    if len(rows) < 3:
        return False, None, "only %d horizontal rules" % len(rows)
    if len(columns) < 2:
        return False, None, "%d horizontal rules but %d vertical rules: no grid" % (
            len(rows), len(columns))
    left = min(box[0] for box in rows + columns)
    right = max(box[0] + box[2] for box in rows + columns)
    upper = min(box[1] for box in rows)
    lower = max(box[1] + box[3] for box in rows)
    if (right - left) < width * 0.3 or (lower - upper) < height * 0.05:
        return False, None, "rules too small to be a table"
    return True, (left, upper, right - left, lower - upper), "%d x %d rules" % (
        len(rows), len(columns))


def strip_grid(pixels, box):
    """Remove the rules inside the detected grid only; every other pixel of the page is untouched."""
    import cv2
    import numpy as np

    ink = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    x, y, width, height = box
    margin_y, margin_x = int(ink.shape[0] * STRIP_MARGIN), int(ink.shape[1] * STRIP_MARGIN)
    window = np.zeros_like(ink)
    window[max(0, y - margin_y):min(ink.shape[0], y + height + margin_y),
           max(0, x - margin_x):min(ink.shape[1], x + width + margin_x)] = 255
    band = cv2.bitwise_and(ink, window)
    horizontal = cv2.morphologyEx(
        band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 25), 1)))
    vertical = cv2.morphologyEx(
        band, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, ink.shape[0] // 40))))
    rules = cv2.dilate(cv2.add(horizontal, vertical), np.ones((3, 3), np.uint8))
    return cv2.bitwise_not(cv2.subtract(ink, rules))


def _evidence_of(cell: dict) -> dict:
    return {
        "value": cell["value"], "families": cell["families"], "reads": cell["reads"],
        "printed_text": cell["raw"], "confidences": cell["confidences"],
        "positions": cell["positions"], "competing_values": cell["competing"],
        "in_primary_read": cell["primary"],
    }


def _candidate_row(row: dict) -> dict:
    """The same row in the shape the extraction would use: for comparison, never exported."""
    cells = row["cells"]
    return {
        "band_top": row["band_top"],
        "band_text": row["band_text"],
        "quantity": cells["quantity"]["value"] if "quantity" in cells else None,
        "unit_price": cells["unit_price"]["value"] if "unit_price" in cells else None,
        "total_ht": cells["amount_ht"]["value"] if "amount_ht" in cells else None,
        "tax_amount": cells["amount_tax"]["value"] if "amount_tax" in cells else None,
        "gross_amount": cells["amount_gross"]["value"] if "amount_gross" in cells else None,
        # Diagnostic only: the families share one engine, so repeated glyph errors are invisible.
        "description": None,
        "description_for_review": row.get("detected_text"),
        "evidence": {role: _evidence_of(cell) for role, cell in cells.items()},
        # Printed values the rules refused, with the reason: a null must be traceable too.
        "rejected": row.get("rejected_cells", {}),
    }


def _comparison(result: dict, evidence: DocumentEvidence) -> dict:
    amounts = [
        Decimal(row["cells"]["amount_ht"]["value"]) for row in result["rows"]
        if "amount_ht" in row["cells"]
    ]
    production_totals = {
        key: sorted({str(number.value) for number in numbers})
        for key, numbers in evidence.totals.items() if numbers
    }
    total_sum = sum(amounts, Decimal(0)) if amounts else None
    # A shadow candidate that adds up to the total the pipeline already trusted is the strongest
    # agreement this experiment can show, so it is recorded - but it still changes nothing.
    matches = sorted(
        value for value in production_totals.get("total_ht", [])
        if total_sum is not None and abs(Decimal(value) - total_sum) <= Decimal("0.02")
    )
    return {
        "production_rows": len(evidence.rows),
        "production_totals": production_totals,
        "candidate_rows": len(result["rows"]),
        "candidate_numeric_fields": sum(len(row["cells"]) for row in result["rows"]),
        "candidate_amount_column_sum": str(total_sum) if total_sum is not None else None,
        "agrees_with_production_total_ht": matches or None,
    }


def _grid_reads(reads, prepared, reader, budget, report, check) -> dict:
    """Extra grid-only reads, taken when the printed rules prove a ruled table exists."""
    import numpy as np
    from PIL import Image

    families = [family for family in GRID_FAMILIES if family in prepared]
    if not families or reader is None:
        return {"triggered": False, "reason": "no prepared page to measure", "reads": []}
    triggered, box, reason = grid_trigger(np.asarray(prepared[families[0]]))
    detail = {"triggered": bool(triggered), "reason": reason,
              "box": list(box) if box else None, "reads": []}
    if not triggered:
        return detail
    # The document budget is only consulted, never consumed: `remaining()` raises once it is gone.
    left_in_budget = None if budget is None else budget.seconds - (budget.clock() - budget.started)
    if left_in_budget is not None and left_in_budget < MIN_SECONDS_FOR_EXTRA_READS:
        detail["reason"] += "; not read: document OCR budget below %d s" % MIN_SECONDS_FOR_EXTRA_READS
        return detail
    for family in families[:MAX_EXTRA_CALLS]:
        key = "%s_grid" % family
        try:
            check()
            started = time.perf_counter()
            stripped = Image.fromarray(strip_grid(np.asarray(prepared[family]), box))
            words = reconstruction.words_of(
                [(token.text, token.box, token.confidence)
                 for token in from_tsv(reader(stripped, TABLE_PSM)).tokens],
                family=family, read=key, size=prepared[family].size,
            )
            if words:
                reads[key] = words
            report["extra_ocr_calls"] = report.get("extra_ocr_calls", 0) + 1
            detail["reads"].append({"read": key, "tokens": len(words),
                                    "seconds": round(time.perf_counter() - started, 2)})
        except Exception as exc:
            reads.pop(key, None)
            detail["reads"].append({"read": key, "error": "%s: %s" % (type(exc).__name__, str(exc)[:90])})
    return detail


def observe(
    evidence: DocumentEvidence,
    readings: dict,
    *,
    rotated_size=None,
    prepared: dict | None = None,
    reader=None,
    budget=None,
    clock=time.monotonic,
) -> DocumentEvidence:
    """Record what a reconstruction would have found; return the very same evidence object."""
    if not shadow_enabled():
        return evidence
    if evidence.rows:
        evidence.table_shadow = {
            "flag": FLAG_NAME, "enabled": True, "status": "not_triggered",
            "reason": "the existing pipeline reconstructed %d row(s)" % len(evidence.rows),
        }
        return evidence
    started = clock()

    def check():
        if clock() - started > MAX_SHADOW_SECONDS:
            raise ShadowDeadline("shadow pass exceeded %.1f s" % MAX_SHADOW_SECONDS)

    report: dict = {"flag": FLAG_NAME, "enabled": True, "status": "running", "extra_ocr_calls": 0}
    try:
        if rotated_size is None or not rotated_size[0]:
            report.update({"status": "skipped", "reason": "no coordinate frame for the reads"})
        else:
            reads = reads_of(readings, rotated_size)
            families = {reconstruction.family_of(key) for key in reads}
            report["families_read"] = sorted(families)
            if len(families) < 2:
                report.update({
                    "status": "skipped",
                    "reason": "a cell needs two distinct preprocessing families and %d was read"
                             % len(families),
                    "reads": sorted(reads),
                })
            else:
                check()
                if prepared is not None:
                    report["grid"] = _grid_reads(reads, prepared, reader, budget, report, check)
                result = reconstruction.reconstruct(reads, check=check)
                report.update({key: value for key, value in result.items() if key != "rows"})
                report["candidate_rows"] = [_candidate_row(row) for row in result["rows"]]
                report["rejected"] = {
                    "columns_without_role": {
                        role: text for role, text in result["pinning_evidence"].items()
                        if role not in result["pinned_roles"]
                    },
                    "certification": result["certification_rejections"],
                    "row_arithmetic": result["row_arithmetic_vetoes"],
                }
                report["comparison"] = _comparison(result, evidence)
                report["status"] = "reconstructed"
    except ShadowDeadline as exc:
        report.update({"status": "deadline", "reason": str(exc)})
    except Exception as exc:  # a research component must never be able to fail a document
        LOGGER.exception("Shadow table reconstruction failed")
        report.update({"status": "error", "reason": "%s: %s" % (type(exc).__name__, str(exc)[:200])})
    report["elapsed_seconds"] = round(clock() - started, 3)
    evidence.table_shadow = report
    return evidence

def rule_lattice(pixels, box) -> dict:
    """Horizontal rule positions inside a detected grid, as normalised y centres.

    Only rules that actually cross the grid's width are kept, so a decorative short line or a
    partial edge never becomes a row boundary. The caller decides what the intervals mean.
    """
    import cv2
    import numpy as np

    ink = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    x, y, width, height = box
    band = np.zeros_like(ink)
    band[max(0, y):min(ink.shape[0], y + height), max(0, x):min(ink.shape[1], x + width)] = ink[
        max(0, y):min(ink.shape[0], y + height), max(0, x):min(ink.shape[1], x + width)]
    horizontal = cv2.morphologyEx(
        band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 2), 1)))
    rules = []
    for contour in cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rw >= width * 0.6 and rh <= 6:
            rules.append(round((ry + rh / 2) / ink.shape[0], 4))
    return {"rules": sorted(set(rules)), "box": list(box)}
