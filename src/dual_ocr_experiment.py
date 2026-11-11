"""Experimental dual-engine OCR arbitration (``EXPERIMENTAL_DUAL_OCR``, off by default).

Combines the Tesseract consensus evidence with a RapidOCR read into consensus-shaped rows that then
reach the unmodified grounding and strict-export stages. The unit of corroboration here is the OCR
**engine**: every Tesseract preprocessing variant is collapsed into one Tesseract source, so several
deskew/clahe/crop reads are never misrepresented as independent acquisitions.

The two engines do not share a coordinate space — the Tesseract anchor read may be deskewed, cropped
or upscaled, while RapidOCR reads the original pixels — so row identity between engines is decided by
printed text, never by absolute boxes.

Strategies:

``A`` cross-engine agreement — a monetary cell is accepted only when the same value is also printed on
    the Tesseract line that reads this row. RapidOCR alone never creates a value.
``B`` missing-evidence recovery — RapidOCR may fill a cell or a whole row where Tesseract printed
    nothing to contradict it, but never overrides a Tesseract value.
``C`` conflict-aware arbitration — B plus one documented conflict rule: on an unexplained Tesseract
    amount the cell is abstained unless RapidOCR's covering tokens are at least ``CONFLICT_MARGIN``
    points more confident and Tesseract shows that losing value nowhere else on the page.

Nothing here changes ``HEADERS``, ``OCR_CONFIG``, ``build_consensus``, ``reconcile`` or the export
policy; it only decides which evidence reaches them.
"""

from __future__ import annotations

import os
import re
import time
from copy import deepcopy
from decimal import Decimal
from difflib import SequenceMatcher

import invoice_evidence as ie
from invoice_evidence import DocumentEvidence, Number
from rapidocr_evidence import read as rapidocr_read

FLAG_NAME = "EXPERIMENTAL_DUAL_OCR"
STRATEGY_ENV = "EXPERIMENTAL_DUAL_OCR_STRATEGY"
STRATEGIES = ("A", "B", "C")
TESSERACT = "tesseract"
RAPIDOCR = "rapidocr"
MONETARY = ("quantity", "unit_price", "total_ht")
CONFLICT_MARGIN = 20.0
BOTH = "both"
CONFLICT = "conflict"
DROPPED = "dropped"
LINE_MATCH_FLOOR = 0.6
# A description shorter than this many letters could match inside an unrelated line by chance, so the
# whitespace-insensitive agreement rule below refuses it.
MIN_COMPACT_DESCRIPTION = 8


def enabled() -> bool:
    return os.environ.get(FLAG_NAME, "").strip().casefold() in {"1", "true", "yes", "on"}


def strategy(default: str = "A") -> str:
    value = os.environ.get(STRATEGY_ENV, "").strip().upper()
    return value if value in STRATEGIES else default


def _token_value(text: str, locale: str | None) -> Decimal | None:
    try:
        return ie.parse_number(text, locale)
    except ValueError:
        return None


MONETARY_TOKEN = re.compile(r"[$\u20ac\u00a3]|\d[.,]\d{2}\b|\b\d{1,3}[.,]\d{2}")


def _is_monetary(text: str) -> bool:
    return bool(MONETARY_TOKEN.search(text))


def _line_read(line, locale: str | None, *, hours: bool) -> dict[Decimal, tuple[float, bool]]:
    """Values printed on one Tesseract line: value -> (lowest covering-token confidence, monetary shape)."""
    from duration_parser import parse_duration

    found: dict[Decimal, tuple[float, bool]] = {}
    for token in line.tokens:
        monetary = _is_monetary(token.text)
        values = []
        value = _token_value(token.text, locale)
        if value is not None:
            values.append((value, monetary))
        if hours:
            duration = parse_duration(token.text, hours_column=True, in_table=True)
            if duration is not None:
                values.append((duration, False))
        for candidate, money in values:
            confidence, previous = found.get(candidate, (100.0, False))
            found[candidate] = (min(token.confidence, confidence), previous or money)
    return found


def _matched_line(rapid_row: ie.TableRow, tesseract: DocumentEvidence):
    reference = ie.folded(rapid_row.line.text)
    best, score = None, 0.0
    if not reference:
        return None
    for line in tesseract.lines:
        candidate = ie.folded(line.text)
        if not candidate:
            continue
        ratio = SequenceMatcher(None, reference, candidate).ratio()
        if ratio > score:
            best, score = line, ratio
    return best if score >= LINE_MATCH_FLOOR else None


def _text_agrees(text: str, evidence: DocumentEvidence) -> bool:
    """Cross-engine agreement on a description, allowing Tesseract's word merging.

    Tesseract routinely prints ``etla ville`` or ``Lapensée`` where the second engine reads
    ``et la ville`` / ``La pensée``: the letters are identical, only the segmentation differs. Comparing
    the letter sequence is still agreement between two independent readings, so it is accepted — but only
    for a description long enough that a chance substring match is implausible. A genuinely different
    reading (a misread year, a different product) still fails and the description stays blank.
    """
    reference = ie.folded(text)
    if not reference:
        return False
    words = reference.split()
    # Whole words, not a substring: "Ab" must not be corroborated by "abonnement".
    if any(words == ie.folded(line.text).split()[start:start + len(words)]
           for line in evidence.lines
           for start in range(max(0, len(ie.folded(line.text).split()) - len(words) + 1))):
        return True
    compact = reference.replace(" ", "")
    return len(compact) >= MIN_COMPACT_DESCRIPTION and any(
        compact in ie.folded(line.text).replace(" ", "") for line in evidence.lines)


def _date_agrees(iso: str, evidence: DocumentEvidence) -> bool:
    for _kind, raw, _line in ie.document_date_readings(evidence.lines):
        parsed, _ = ie.parse_document_date(raw, evidence.date_locale or evidence.locale)
        if parsed is not None and parsed.isoformat() == iso:
            return True
    return False


def _duration_number(row: ie.TableRow) -> Number | None:
    """A printed duration in an hours column, converted by the isolated parser (never invented)."""
    from duration_parser import parse_duration

    matches = [
        Number(token.text, value, token.page, token.box)
        for token in row.line.tokens
        if (value := parse_duration(token.text, hours_column=True, in_table=True)) is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _hourly_table(evidence: DocumentEvidence) -> bool:
    return any(
        ie._contextual_rate_price_header(line.text) and ie._header_matches("quantity", line.text)
        for line in evidence.lines
    )


def _single(row: ie.TableRow, key: str, hours: bool) -> tuple[Number | None, str]:
    numbers = row.cells.get(key, [])
    if len(numbers) == 1:
        return numbers[0], "printed"
    if not numbers:
        return None, "not read"
    if key == "quantity" and hours:
        duration = _duration_number(row)
        if duration is not None:
            return duration, "duration-parsed"
    return None, f"{len(numbers)} candidate values"


def _row_signature(row: ie.TableRow) -> tuple:
    return tuple(sorted(
        f"{key}:{number.value}" for key in MONETARY for number in row.cells.get(key, [])
    ))


def arbitrate(tesseract: DocumentEvidence, rapid: DocumentEvidence,
              strategy_name: str | None = None) -> tuple[DocumentEvidence, dict]:
    """Return evidence combining both engines, plus a provenance record of every decision."""
    strategy_name = (strategy_name or strategy()).upper()
    allow_new_rows = not tesseract.rows
    hours = _hourly_table(rapid) or _hourly_table(tesseract)
    cells: list[dict] = []
    row_log: list[dict] = []
    merged_rows: list[ie.TableRow] = []
    replacements: dict[int, ie.TableRow] = {}
    tesseract_signatures = {_row_signature(row) for row in tesseract.rows}

    for index, rapid_row in enumerate(rapid.rows):
        line = _matched_line(rapid_row, tesseract)
        line_values = _line_read(line, tesseract.locale, hours=hours) if line is not None else {}
        rapid_money = _money_sequence(rapid_row.line, rapid.locale)
        tesseract_money = _money_sequence(line, tesseract.locale) if line is not None else []
        comparable_columns = bool(rapid_money) and len(rapid_money) == len(tesseract_money)
        used_ranks: set[int] = set()
        matched = (_row_owning_line(line, tesseract.rows) or _row_textually(rapid_row, tesseract.rows)
                 or _matching_tesseract_row(rapid_row, tesseract))
        accepted: dict[str, Number] = {}
        states: dict[str, str] = {}
        explained = set(_line_read(rapid_row.line, rapid.locale, hours=hours))
        unexplained = sorted(
            (value for value, (_confidence, money) in line_values.items()
             if money and value not in explained),
            key=str,
        )
        for key in MONETARY:
            number, origin = _single(rapid_row, key, hours)
            if number is None:
                cells.append({"row": index, "field": key, "provenance": DROPPED, "reason": origin})
                continue
            rank = _cell_rank(number, rapid_money, used_ranks) if comparable_columns else None
            if rank is not None:
                used_ranks.add(rank)
            column = tesseract_money[rank] if rank is not None else None
            states[key] = _decide(cells, index, key, number, origin, accepted, states,
                                  line_values, unexplained, strategy_name, line, rapid_row, column, hours)
        if not any(key in accepted for key in ("unit_price", "total_ht")):
            row_log.append({"row": index, "action": "no-accepted-monetary-cell", "cells": dict(states)})
            continue
        if matched is None and _row_signature_with(accepted) in tesseract_signatures:
            row_log.append({"row": index, "action": "dropped-duplicate", "cells": dict(states)})
            continue
        if matched is not None:
            tesseract_amounts = {number.value for key in MONETARY for number in matched.cells.get(key, [])}
            rapid_amounts = set(_line_read(rapid_row.line, rapid.locale, hours=hours))
            reanchor = bool(accepted) and not tesseract_amounts - rapid_amounts
            filled, preserved = [], []
            if reanchor:
                # RapidOCR prints every amount Tesseract printed for this row, so the row can be
                # re-anchored on the second engine's reading without losing a printed value.
                merged = deepcopy(rapid_row)
                merged.cells = {key: [number] for key, number in accepted.items()}
                merged.family_votes = {
                    key: {number.value: _voters(states[key])} for key, number in accepted.items()
                }
                merged.description = matched.description or ""
                if not merged.description:
                    merged.description = rapid_row.description or ""
                    if not _text_agrees(merged.description, tesseract):
                        merged.description = ""
                action = "rapidocr-anchored-row"
            else:
                merged = deepcopy(matched)
                for key, number in accepted.items():
                    existing = {item.value for item in merged.cells.get(key, [])}
                    if not existing:
                        merged.cells[key] = [number]
                        merged.family_votes.setdefault(key, {})[number.value] = _voters(states[key])
                        filled.append(key)
                        continue
                    if number.value in existing:
                        voters = merged.family_votes.get(key, {}).get(number.value, ())
                        merged.family_votes.setdefault(key, {})[number.value] = tuple(
                            dict.fromkeys([*voters, TESSERACT, RAPIDOCR]))
                        continue
                    if strategy_name != "C":
                        preserved.append(key)
                        continue
                    # Strategy C: the two engines print different amounts in the same column of the
                    # same row. The documented rule may replace the value, otherwise the column is
                    # abstained. Neither branch ever picks an engine by name.
                    tesseract_value = min(existing)
                    margin = _rapid_confidence(rapid_row, number) - max(
                        (token.confidence for token in matched.line.tokens
                         if _token_value(token.text, tesseract.locale) == tesseract_value), default=0.0)
                    if _conflict_rule_applies(key, number, tesseract_value, rapid_row, margin, hours):
                        merged.cells[key] = [number]
                        merged.family_votes[key] = {number.value: (RAPIDOCR,)}
                        cells.append({"row": index, "field": key, "provenance": "rapidocr-conflict",
                                      "value": str(number.value), "raw": number.raw,
                                      "overridden_tesseract_value": str(tesseract_value),
                                      "confidence_margin": round(float(margin), 1)})
                    else:
                        # Abstaining a conflict means refusing to *change* the amount. Deleting the
                        # value Tesseract printed would turn a disagreement into a regression.
                        preserved.append(key)
                        cells.append({"row": index, "field": key, "provenance": CONFLICT,
                                      "rapidocr_value": str(number.value),
                                      "kept_tesseract_value": str(tesseract_value),
                                      "confidence_margin": round(float(margin), 1),
                                      "reason": "column conflict: the Tesseract amount is kept"})
                action = "kept-tesseract-row"
            states["columns"] = {
                "accepted": sorted(accepted),
                "tesseract_amounts": sorted(str(value) for value in tesseract_amounts),
                "rapidocr_amounts": sorted(str(value) for value in rapid_amounts),
                "reanchored": reanchor,
            }
            replacements[id(matched)] = merged
        else:
            if not allow_new_rows:
                row_log.append({"row": index, "action": "not-activated-zero-row-rule",
                                "cells": dict(states), "line": rapid_row.line.text})
                continue
            # A row Tesseract never built. It is only kept when Tesseract built no rows at all,
            # because that is the activation the evidence of PR92/PR100 supports: RapidOCR then backs
            # the whole document, and every value is verified against RapidOCR's own boxes.
            merged = deepcopy(rapid_row)
            merged.cells = {key: [number] for key, number in accepted.items()}
            merged.family_votes = {
                key: {number.value: _voters(states[key])} for key, number in accepted.items()
            }
            merged.description = rapid_row.description or ""
            if not _text_agrees(merged.description, tesseract):
                # A description Tesseract cannot see at all is never exported as text: the second
                # engine may fill amounts, it does not get to rename a line item.
                merged.description = ""
            action = "rapidocr-row"
            filled, preserved = sorted(accepted), []
        merged.description_confident = bool(merged.description) and not ie.has_replacement_character(
            merged.description)
        merged.row_evidence = {
            "engine": RAPIDOCR, "strategy": strategy_name, "rapidocr_line": rapid_row.line.text,
            "tesseract_line": line.text if line is not None else None,
            "tesseract_row_matched": matched is not None, "cell_provenance": dict(states),
            "unexplained_tesseract_amounts": unexplained,
        }
        if matched is not None:
            replacements[id(matched)] = merged
        else:
            merged_rows.append(merged)
        row_log.append({"row": index, "action": action, "cells": dict(states),
                        "filled_columns": filled, "preserved_columns": preserved,
                        "matched_tesseract_row": matched is not None})

    # RapidOCR backs the document as soon as it contributes a row, because grounding verifies each
    # exported value against the boxes of the evidence it is given, and a RapidOCR row carries
    # RapidOCR geometry. Both engines' lines and tokens are therefore present so that a Tesseract row
    # is still verifiable; the row-level provenance keeps them distinguishable.
    rapid_backed = allow_new_rows and bool(merged_rows)
    primary, other = (rapid, tesseract) if rapid_backed else (tesseract, rapid)
    result = deepcopy(primary)
    if rapid_backed:
        result.rows = merged_rows
        result.lines = _union_lines(rapid, tesseract)
        result.tokens = _union_tokens(rapid, tesseract)
    else:
        result.rows = [replacements.get(id(row), row) for row in tesseract.rows]
    result.rows_provenance = f"dual_ocr_{strategy_name}"
    result.preprocessing_used = list(dict.fromkeys([*tesseract.preprocessing_used, *rapid.preprocessing_used]))
    # The second engine is a real second, differently-built read of the same page: the evidence is
    # cross-checked only where the two engines actually agreed on a printed value. Without any
    # agreement nothing is claimed, and grounding's own document gate keeps refusing the reading.
    agreed = any(cell["provenance"] in (BOTH, "rapidocr-conflict") for cell in cells)
    result.ocr_cross_checked = bool(primary.ocr_cross_checked or agreed)
    if merged_rows and result.source_kind == "ocr":
        result.source_kind = "consensus"
    document = _combine_document(result, other, strategy_name)
    return result, {
        "strategy": strategy_name,
        "evidence_engine": RAPIDOCR if rapid_backed else TESSERACT,
        "hourly_table": hours,
        "cells": cells,
        "rows": row_log,
        "document": document,
        "rows_built": {"tesseract": len(tesseract.rows), "rapidocr": len(rapid.rows),
                       "experimental": len(result.rows)},
        "cell_provenance": {
            state: sum(1 for cell in cells if cell["provenance"] == state)
            for state in (BOTH, RAPIDOCR, CONFLICT, DROPPED, "rapidocr-conflict")
        },
    }


def _row_owning_line(line, rows):
    """The Tesseract row built on this exact line, when one exists (same object, same evidence)."""
    if line is None:
        return None
    return next((row for row in rows if row.line is line or row.line.box == line.box), None)


def _row_textually(rapid_row: ie.TableRow, rows):
    """The Tesseract row whose own line is the row RapidOCR re-read, matched by printed text."""
    reference = ie.folded(rapid_row.line.text)
    if not reference:
        return None
    best, score = None, 0.0
    for row in rows:
        candidate = ie.folded(row.line.text)
        if not candidate:
            continue
        ratio = SequenceMatcher(None, reference, candidate).ratio()
        if ratio > score:
            best, score = row, ratio
    return best if score >= LINE_MATCH_FLOOR else None


def _union_lines(rapid: DocumentEvidence, tesseract: DocumentEvidence) -> list:
    boxes = {line.box for line in rapid.lines}
    return [*rapid.lines, *[deepcopy(line) for line in tesseract.lines if line.box not in boxes]]


def _union_tokens(rapid: DocumentEvidence, tesseract: DocumentEvidence) -> list:
    seen = {(token.text, token.box) for token in rapid.tokens}
    return [*rapid.tokens, *[
        token for token in tesseract.tokens if (token.text, token.box) not in seen
    ]]


def _matching_tesseract_row(rapid_row: ie.TableRow, tesseract: DocumentEvidence):
    reference = _row_signature_with({
        key: numbers[0] for key, numbers in rapid_row.cells.items()
        if key in MONETARY and len(numbers) == 1
    })
    if not reference:
        return None
    scored = sorted(
        ((SequenceMatcher(None, ie.folded(rapid_row.description), ie.folded(row.description)).ratio(),
          _row_signature(row) == reference, row) for row in tesseract.rows),
        key=lambda item: (item[1], item[0]), reverse=True,
    )
    if not scored:
        return None
    best = scored[0]
    return best[2] if best[1] and best[0] >= LINE_MATCH_FLOOR else None


def _row_signature_with(accepted: dict) -> tuple:
    return tuple(sorted(f"{key}:{number.value}" for key, number in accepted.items()))


def _rapid_confidence(row: ie.TableRow, number: Number) -> float:
    return min((token.confidence for token in row.line.tokens
                if _overlaps(number.box, token.box)), default=0.0)


def _overlaps(inner: tuple, outer: tuple, *, tolerance: float = 6.0) -> bool:
    centre = ((outer[0] + outer[2]) / 2, (outer[1] + outer[3]) / 2)
    return (inner[0] - tolerance <= centre[0] <= inner[2] + tolerance
            and inner[1] - tolerance <= centre[1] <= inner[3] + tolerance)


def _formatting_corruption(candidate: Decimal, printed: Decimal) -> bool:
    """True only for two renderings of the *same* amount: ``15.955`` against ``15,95`` is one.

    Both conditions are required. The digit sequences must agree apart from a trailing digit beyond
    the separator, and the amounts must be within one percent of each other, so ``4`` against ``40``
    or ``540`` against ``40`` are different amounts and can never override through this path.
    """
    difference = abs(candidate - printed)
    if difference == 0:
        return True
    if printed and difference / max(abs(printed), Decimal(1)) > Decimal("0.01"):
        return False
    left, right = str(abs(candidate)).replace(".", ""), str(abs(printed)).replace(".", "")
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(longer) == len(shorter) + 1 and longer.startswith(shorter)


def _reconciles_with_printed_total(quantity: Decimal, price: Decimal, total: Decimal) -> bool:
    from decimal import ROUND_HALF_UP

    expected = (quantity * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return abs(expected - total) <= Decimal("0.02")


def _conflict_rule_applies(key: str, number: Number, disagreeing: Decimal, row: ie.TableRow,
                           margin: float, hours: bool) -> bool:
    """Strategy C's documented conflict rule. Every condition is necessary.

    A confident second engine is not a reason to change an amount. The rule fires only when the two
    readings are renderings of the same amount (``_formatting_corruption``), the second engine's
    covering tokens are much better, and the row's own printed operands agree with the replacement.
    The arithmetic is never the only evidence: it can refuse, and by itself it never accepts.
    """
    if margin < CONFLICT_MARGIN or not _formatting_corruption(number.value, disagreeing):
        return False
    quantity = _single(row, "quantity", hours)[0]
    price = _single(row, "unit_price", hours)[0]
    total = _single(row, "total_ht", hours)[0]
    if quantity is None or price is None or total is None:
        return False
    return _reconciles_with_printed_total(quantity.value, price.value, total.value)


def _money_sequence(line, locale: str | None) -> list[tuple[Decimal, float, tuple]]:
    """The amounts one engine printed on a line, left to right: value, confidence, cell box.

    Both engines read a row as a sequence of amounts in column order, so the ordinal of an amount in
    this sequence is the column it belongs to. That is how a cell is matched across engines whose
    coordinate systems differ.
    """
    items = []
    for token in sorted(line.tokens, key=lambda item: item.box[0]):
        value = _token_value(token.text, locale)
        if value is not None and _is_monetary(token.text):
            items.append((value, token.confidence, token.box))
    return items


def _cell_rank(number: Number, money, used: set[int]) -> int | None:
    """Where this cell sits in the row's left-to-right amount sequence.

    ``Number`` carries its whole line box, so the position cannot be read from geometry; the value
    itself is matched against the row's amounts in reading order, each amount used once.
    """
    ranks = [index for index, (value, _confidence, _box) in enumerate(money)
             if value == number.value and index not in used]
    return ranks[0] if ranks else None


def _decide(cells, index, key, number, origin, accepted, states, line_values, unexplained,
            strategy_name, line, rapid_row, column=None, hours=False) -> str:
    """One cell, one decision. Records it and returns the provenance state.

    ``column`` is the amount Tesseract printed in the same column of the same row, available when the
    two engines printed the same number of amounts on that row. Without it only the row-level reading
    is available, and an amount found nowhere on the row is treated as a single-engine value.
    """
    disagreeing = None
    if column is not None:
        agreed = column[0] == number.value
        disagreeing = None if agreed else column[0]
    else:
        agreed = number.value in line_values
    if agreed:
        accepted[key] = number
        states[key] = BOTH
        cells.append({"row": index, "field": key, "provenance": BOTH, "value": str(number.value),
                      "raw": number.raw, "origin": origin,
                      "column_matched": column is not None,
                      "tesseract_confidence": column[1] if column is not None
                      else line_values[number.value][0]})
        return BOTH
    if disagreeing is None and unexplained:
        disagreeing = max((value for value, (_confidence, money) in line_values.items() if money),
                          default=None)
    if disagreeing is not None:
        margin = _rapid_confidence(rapid_row, number) - (
            column[1] if column is not None else max(
                (confidence for confidence, _money in line_values.values()), default=0.0))
        if strategy_name == "C" and _conflict_rule_applies(key, number, disagreeing, rapid_row, margin, hours):
            accepted[key] = number
            states[key] = "rapidocr-conflict"
            cells.append({"row": index, "field": key, "provenance": "rapidocr-conflict",
                          "value": str(number.value), "raw": number.raw, "origin": origin,
                          "overridden_tesseract_value": str(disagreeing),
                          "confidence_margin": round(float(margin), 1)})
            return "rapidocr-conflict"
        states[key] = CONFLICT
        cells.append({"row": index, "field": key, "provenance": CONFLICT,
                      "rapidocr_value": str(number.value), "raw": number.raw,
                      "tesseract_value": str(disagreeing),
                      "reason": "the two engines print different amounts in this column",
                      "confidence_margin": round(float(margin), 1)})
        return CONFLICT
    if strategy_name == "A":
        states[key] = DROPPED
        cells.append({"row": index, "field": key, "provenance": DROPPED,
                      "rapidocr_value": str(number.value), "raw": number.raw,
                      "reason": "no Tesseract reading of this column"})
        return DROPPED
    if strategy_name == "C" and not any(state == BOTH for state in states.values()):
        states[key] = DROPPED
        cells.append({"row": index, "field": key, "provenance": DROPPED,
                      "rapidocr_value": str(number.value), "raw": number.raw,
                      "reason": "single-engine cell without a corroborated sibling on this row"})
        return DROPPED
    accepted[key] = number
    states[key] = RAPIDOCR
    cells.append({"row": index, "field": key, "provenance": RAPIDOCR, "value": str(number.value),
                  "raw": number.raw, "origin": origin})
    return RAPIDOCR


def _voters(state: str) -> tuple[str, ...]:
    return (TESSERACT, RAPIDOCR) if state == BOTH else (RAPIDOCR,)


def _combine_document(result: DocumentEvidence, other: DocumentEvidence,
                      strategy_name: str) -> list[dict]:
    """Fill only what the backing engine lacks; never overwrite a value it already printed.

    Supplier names are never adopted. On Fast Photography both engines read the same logo the same wrong
    way, so the agreement test passed and grounding presented a corrupted name as corroborated: agreement
    between two readings of the same pixels establishes an amount, not a name.
    """
    changes = []
    if not result.supplier and getattr(other, "supplier", None):
        changes.append({"field": "supplier", "provenance": DROPPED,
                        "reason": "engine agreement cannot establish a supplier name"})
    for key, agrees in (("invoice_date", _date_agrees),):
        candidate = getattr(other, key)
        if getattr(result, key) or not candidate:
            continue
        agreed = agrees(str(candidate), result)
        if strategy_name == "A" and not agreed:
            continue
        setattr(result, key, candidate)
        changes.append({"field": key, "value": str(candidate),
                        "provenance": BOTH if agreed else RAPIDOCR})
    own_values = {number.value for number in result.numbers}
    for key, numbers in other.totals.items():
        if len(numbers) != 1 or result.totals.get(key):
            continue
        value = numbers[0].value
        agreed = value in own_values
        if strategy_name == "A" and not agreed:
            continue
        result.totals[key] = [numbers[0]]
        changes.append({"field": f"totals.{key}", "value": str(value),
                        "provenance": BOTH if agreed else RAPIDOCR})
    if not result.taxes and other.taxes and strategy_name != "A":
        result.taxes = list(other.taxes)
        changes.append({"field": "taxes", "value": len(other.taxes), "provenance": RAPIDOCR})
    return changes


def experiment(image, tesseract: DocumentEvidence, *, strategy_name: str | None = None):
    """Run the second engine and arbitrate. Returns ``(evidence, record)``; never raises."""
    started = time.perf_counter()
    name = (strategy_name or strategy()).upper()
    try:
        outcome = rapidocr_read(image)
    except Exception as exc:
        # A timeout or any engine-level failure must leave the production answer intact, not propagate.
        return tesseract, {
            "strategy": name, "available": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - started, 2),
        }
    if outcome is None or outcome.evidence is None or not outcome.evidence.tokens:
        return tesseract, {
            "strategy": name, "available": False,
            "failure": (outcome.failure if outcome else None) or "RapidOCR non importable ou indisponible",
            "seconds": round(time.perf_counter() - started, 2),
        }
    record = {
        "strategy": name, "available": True, "engine": RAPIDOCR,
        "engine_version": outcome.engine_version, "ocr_seconds": outcome.seconds,
        "regions": len(outcome.regions), "rapidocr_rows": len(outcome.evidence.rows),
    }
    try:
        result, detail = arbitrate(tesseract, outcome.evidence, name)
    except Exception as exc:  # the experiment must never break extraction
        record.update({"failure": f"{type(exc).__name__}: {exc}",
                       "seconds": round(time.perf_counter() - started, 2)})
        return tesseract, record
    record.update(detail)
    record["seconds"] = round(time.perf_counter() - started, 2)
    result.dual_ocr = record
    return result, record
