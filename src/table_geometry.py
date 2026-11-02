"""Experimental geometry-first row recovery (OFF by default: INVOICE_TABLE_GEOMETRY).

Runs only when production reconstruction produced no documentary rows. It reads the primary OCR
family's existing tokens (no extra OCR call), derives numeric column axes and roles from the
printed geometry plus arithmetic corroboration, and injects rows with ``row_index=None``: no
absolute document ordinal is ever asserted. Every cell keeps its printed text, box, page, role
and corroboration reason in ``row_evidence``; multi-row evidence therefore stays sidecar-only
unless the document proves the ordinals.

The model is never consulted: cells come from printed tokens only, and no value is ever derived
from arithmetic.
"""

import logging
import os
import re
from decimal import Decimal, InvalidOperation

from invoice_evidence import Number, TableRow

LOGGER = logging.getLogger(__name__)

FLAG_NAME = "INVOICE_TABLE_GEOMETRY"
_TRUTHY = {"1", "true", "yes", "on"}
TOL = Decimal("0.02")
AXIS_TOL = 0.025
CURRENCY = {"EUR", "GBP", "USD", "€", "£", "$"}
SUMMARY = re.compile(
    r"\b(?:total|subtotal|sous.total|tax|vat|tva|discount|remise|due|date|invoice|facture|"
    r"commande|order|reference|référence|siret|siren|iban|echeance|échéance|net a payer|"
    r"montant)\b", re.IGNORECASE)
ADDRESS_LIKE = re.compile(
    r"\b(?:rue|avenue|boulevard|chemin|route|place|street|road|lane|drive|way|high\s+street)\b",
    re.IGNORECASE)
POSTAL = re.compile(r"\b\d{4,5}\s+[A-ZÀ-ÖØ-Ý]")
THOUSANDS_COMMA = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?$")
THOUSANDS_DOT = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?$")
PCT = re.compile(r"^[+-]?\d{1,3}\s*%$")


def enabled() -> bool:
    return os.environ.get(FLAG_NAME, "").strip().casefold() in _TRUTHY


class _Tok:
    __slots__ = ("text", "box", "page", "confidence")

    def __init__(self, text, box, page, confidence):
        self.text = text
        self.box = box
        self.page = page
        self.confidence = confidence


def number_value(text):
    raw = text.strip().replace("\u00a0", "").replace(" ", "").rstrip("%")
    if not raw or not any(ch.isdigit() for ch in raw):
        return None
    if THOUSANDS_COMMA.match(raw):
        raw = raw.replace(",", "")
    elif THOUSANDS_DOT.match(raw):
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", ".")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _numeric_tokens(line, width):
    tokens = [_Tok(t.text, t.box, t.page, t.confidence) for t in line.tokens
              if number_value(t.text) is not None]
    merged = []
    for token in tokens:
        if merged:
            previous = merged[-1]
            gap = token.box[0] - previous.box[2]
            if gap <= 0.02 * width and re.fullmatch(r"[+-]?\d{1,3}", previous.text.strip()) \
                    and re.fullmatch(r"\d{3}[.,]\d{2}", token.text.strip()):
                merged[-1] = _Tok(previous.text.strip() + " " + token.text.strip(),
                                  (previous.box[0], previous.box[1], token.box[2], token.box[3]),
                                  previous.page, min(previous.confidence, token.confidence))
                continue
        merged.append(token)
    return merged


def _description_tokens(line, first_numeric):
    return [t for t in line.tokens
            if t.box[2] <= first_numeric.box[0] + 1
            and t.text.strip().upper() not in CURRENCY]


def _item_lines(reading, width):
    selected = []
    for line in reading.lines:
        tokens = _numeric_tokens(line, width)
        if len(tokens) < 2 or SUMMARY.search(line.text):
            continue
        if ADDRESS_LIKE.search(line.text) or POSTAL.search(line.text):
            continue
        first = min(tokens, key=lambda t: t.box[0])
        if first.box[0] / width < 0.12:
            continue
        description = " ".join(t.text for t in _description_tokens(line, first))
        if len(re.findall(r"[^\W\d_]", description)) < 2:
            continue
        selected.append((line, tokens))
    return selected


def _build_axes(lines, width):
    if len(lines) == 1:
        return sorted(token.box[2] / width for token in lines[0][1])
    edges = sorted(token.box[2] / width for _, tokens in lines for token in tokens)
    clusters = []
    for edge in edges:
        if clusters and abs(edge - clusters[-1][-1]) <= AXIS_TOL:
            clusters[-1].append(edge)
        else:
            clusters.append([edge])
    return sorted(sum(c) / len(c) for c in clusters if len(c) >= 2)


def _axis_for(x, axes):
    if not axes:
        return None
    best = min(axes, key=lambda a: abs(a - x))
    return best if abs(best - x) <= AXIS_TOL else None


def _axis_shape(per_line, axis):
    values, decimals = [], []
    for _, cells in per_line:
        token = cells.get(axis)
        if token is None:
            continue
        value = number_value(token.text)
        if value is None:
            continue
        values.append(value)
        text = token.text.strip().replace(" ", "")
        decimals.append(len(text.rsplit(",", 1)[-1]) if "," in text else
                        len(text.rsplit(".", 1)[-1]) if "." in text else 0)
    return (sum(values) / len(values) if values else Decimal(0),
            sum(decimals) / len(decimals) if decimals else 0)


def _shape_preference(shapes, q, p):
    if shapes[q][1] < shapes[p][1]:
        return 2
    if shapes[q][1] == shapes[p][1] and shapes[q][0] < shapes[p][0]:
        return 1
    return 0


def _find_triple(per_line, axes, shapes):
    best = None
    for q in axes:
        for p in axes:
            if p == q:
                continue
            for a in axes:
                if a in (q, p):
                    continue
                score, error = 0, Decimal(0)
                for _, cells in per_line:
                    qt, pt, at = cells.get(q), cells.get(p), cells.get(a)
                    if not (qt and pt and at):
                        continue
                    qv, pv, av = number_value(qt.text), number_value(pt.text), number_value(at.text)
                    if None in (qv, pv, av) or qv <= 0 or pv <= 0 or av <= 0:
                        continue
                    delta = abs(qv * pv - av)
                    if delta <= TOL:
                        score += 1
                        error += delta
                key = (score, _shape_preference(shapes, q, p), -error, a)
                if best is None or key > best[0]:
                    best = (key, q, p, a)
    return best


def _printed_text(tokens):
    return {role: token.text for role, token in tokens.items() if token is not None}


def _positions(tokens, family):
    return {role: {"family": family, "pixel": list(token.box)}
            for role, token in tokens.items() if token is not None}


def _number(token):
    return Number(raw=token.text, value=number_value(token.text), page=token.page, box=token.box)


def _row(line, family, description, cells, corroboration, refused, confident):
    tokens = {role: cell for role, cell in cells.items()}
    return TableRow(
        description=description,
        cells={role: [_number(token)] for role, token in tokens.items()},
        line=line,
        description_confident=confident,
        row_index=None,
        row_evidence={
            "page": line.page,
            "method": "geometry",
            "family": family,
            "row_index": None,
            "row_index_confidence": None,
            "printed": _printed_text(tokens),
            "positions": _positions(tokens, family),
            "corroboration": corroboration,
            "refused_cells": refused,
        },
    )


def _description(line, first_numeric):
    tokens = _description_tokens(line, first_numeric)
    text = " ".join(t.text for t in tokens).strip(" |:;-")
    confident = bool(tokens) and all(t.confidence >= 60 for t in tokens)
    return text, confident


def recover(evidence, readings, *, rotated_size=None, prepared=None):
    """Inject geometry rows only when production found none. Returns the same evidence object."""
    if not enabled():
        return evidence
    report = {"flag": FLAG_NAME, "status": "skipped", "reason": "", "rows": 0}
    if evidence.rows:
        report["reason"] = "production rows already present"
        evidence.geometry = report
        return evidence
    if not readings or not prepared:
        report["reason"] = "no OCR readings or frames"
        evidence.geometry = report
        return evidence
    families = [f for f in readings if readings[f] and f in prepared]
    if not families:
        report["reason"] = "no prepared family"
        evidence.geometry = report
        return evidence
    sizes = {f: prepared[f].size for f in families}
    if len(set(sizes.values())) != 1:
        report["status"] = "frame_mismatch"
        report["reason"] = f"prepared frames differ: {sizes}"
        evidence.geometry = report
        return evidence
    family = families[0]
    reading = readings[family][0]
    width = sizes[family][0]
    lines = _item_lines(reading, width)
    report["family"] = family
    report["width"] = width
    report["item_lines"] = len(lines)
    if not lines:
        report["status"] = "no_table_lines"
        report["reason"] = "no item line with two numbers and a description"
        evidence.geometry = report
        return evidence
    axes = _build_axes(lines, width)
    report["axes"] = [round(a, 4) for a in axes]
    if len(axes) < 2:
        report["status"] = "no_axes"
        report["reason"] = "fewer than two shared numeric axes"
        evidence.geometry = report
        return evidence

    per_line, rate_axes = [], set()
    for line, tokens in lines:
        cells = {}
        for token in tokens:
            axis = _axis_for(token.box[2] / width, axes)
            if axis is not None and axis not in cells:
                cells[axis] = token
        for token in line.tokens:
            if token.text.strip() == "%":
                left = [t for t in tokens if t.box[2] <= token.box[0] + 1]
                if left:
                    axis = _axis_for(max(left, key=lambda t: t.box[2]).box[2] / width, axes)
                    if axis is not None:
                        rate_axes.add(axis)
        per_line.append((line, cells))

    printed = {n.value for key in ("total_ht", "total_candidate") for n in evidence.totals.get(key, [])}
    discount_total = sum((abs(d.value) for d in evidence.discounts), Decimal(0))
    rate_axis = min(rate_axes) if rate_axes else None
    report["rate_axis"] = rate_axis

    rows, refused = [], []
    if rate_axis is not None:
        money = [a for a in axes if a != rate_axis]
        left = [a for a in money if a < rate_axis]
        right = [a for a in money if a > rate_axis]
        price_axis = max(left) if left else None
        tax_axis = min(right) if right else None
        gross = [a for a in right if a != tax_axis]
        amount_axis = max(gross) if gross else None
        qty_axes = [a for a in axes if a not in (price_axis, tax_axis, amount_axis, rate_axis)]
        if price_axis is None or tax_axis is None or not qty_axes:
            report["status"] = "no_roles"
            report["reason"] = "rate column without price/tax/quantity columns"
            evidence.geometry = report
            return evidence
        q_axis = qty_axes[0]
        for line, cells in per_line:
            qt, pt, tt, rt = cells.get(q_axis), cells.get(price_axis), cells.get(tax_axis), cells.get(rate_axis)
            if not (qt and pt and tt and rt):
                continue
            qv, pv, tv, rv = (number_value(qt.text), number_value(pt.text),
                              number_value(tt.text), number_value(rt.text))
            if None in (qv, pv, tv, rv) or qv <= 0 or pv <= 0:
                continue
            if abs(tv - qv * pv * rv / Decimal(100)) > TOL:
                refused.append(f"{line.text[:40]}: tax does not corroborate quantity x price")
                continue
            description, confident = _description(line, min(_numeric_tokens(line, width),
                                                            key=lambda t: t.box[0]))
            rows.append(_row(line, family, description,
                             {"quantity": qt, "unit_price": pt},
                             {"rule": "tax = quantity x unit_price x rate",
                              "rate": rt.text, "tax": tt.text},
                             [], confident))
        if len(rows) < 2:
            report["status"] = "uncorroborated"
            report["reason"] = f"{len(rows)} corroborated rows (<2)"
            evidence.geometry = report
            return evidence
    else:
        shapes = {axis: _axis_shape(per_line, axis) for axis in axes}
        best = _find_triple(per_line, axes, shapes)
        if best is None:
            report["status"] = "no_roles"
            report["reason"] = "no quantity/price/amount triple"
            evidence.geometry = report
            return evidence
        (score, shape_pref, _error, _a), q_axis, p_axis, a_axis = best
        report["roles"] = {"quantity": q_axis, "unit_price": p_axis, "amount": a_axis}
        reversed_best = None
        for q, p, a in ((p_axis, q_axis, a_axis),):
            score_r, error_r = 0, Decimal(0)
            for _, cells in per_line:
                qt, pt, at = cells.get(q), cells.get(p), cells.get(a)
                if not (qt and pt and at):
                    continue
                qv, pv, av = number_value(qt.text), number_value(pt.text), number_value(at.text)
                if None in (qv, pv, av):
                    continue
                delta = abs(qv * pv - av)
                if delta <= TOL:
                    score_r += 1
                    error_r += delta
            reversed_best = (score_r, _shape_preference(shapes, q, p))
        ambiguous = (reversed_best is not None and reversed_best[0] == score
                     and reversed_best[1] == shape_pref)
        report["ambiguous_quantity_price"] = bool(ambiguous)
        if ambiguous:
            base = sum((number_value(cells[a_axis].text) for _, cells in per_line
                        if cells.get(a_axis) is not None), Decimal(0))
            total_ok = any(abs(base - value) <= TOL for value in printed)
            report["amount_sum"] = float(base)
            report["amount_sum_supported"] = bool(total_ok)
            if not total_ok:
                report["status"] = "ambiguous_roles"
                report["reason"] = "quantity and unit price indistinguishable; amount not anchored"
                evidence.geometry = report
                return evidence
            for line, cells in per_line:
                at = cells.get(a_axis)
                if at is None:
                    continue
                description, confident = _description(
                    line, min(_numeric_tokens(line, width), key=lambda t: t.box[0]))
                rows.append(_row(line, family, description, {"total_ht": at},
                                 {"rule": "amount sum equals a printed total",
                                  "amount_sum": str(base)},
                                 ["quantity", "unit_price"], confident))
        else:
            for line, cells in per_line:
                qt, pt, at = cells.get(q_axis), cells.get(p_axis), cells.get(a_axis)
                if not (qt and pt and at):
                    continue
                qv, pv, av = number_value(qt.text), number_value(pt.text), number_value(at.text)
                if None in (qv, pv, av) or qv <= 0 or pv <= 0 or av <= 0:
                    continue
                if abs(qv * pv - av) > TOL:
                    refused.append(f"{line.text[:40]}: quantity x price != amount")
                    continue
                description, confident = _description(
                    line, min(_numeric_tokens(line, width), key=lambda t: t.box[0]))
                rows.append(_row(line, family, description,
                                 {"quantity": qt, "unit_price": pt, "total_ht": at},
                                 {"rule": "quantity x unit_price = amount",
                                  "product": str(qv * pv)}, [], confident))
            if len(rows) == 1:
                amount = number_value(rows[0].cells["total_ht"][0].raw)
                single_ok = any(abs(amount - discount_total - value) <= TOL for value in printed)
                report["single_row"] = {"amount": float(amount), "discount": float(discount_total),
                                        "printed": sorted(float(v) for v in printed),
                                        "corroborated": bool(single_ok)}
                if not single_ok:
                    report["status"] = "uncorroborated"
                    report["reason"] = "single row not corroborated by the printed total"
                    evidence.geometry = report
                    return evidence
            elif len(rows) < 2:
                report["status"] = "uncorroborated"
                report["reason"] = "no corroborated rows"
                evidence.geometry = report
                return evidence

    if not rows:
        report["status"] = "no_rows"
        report["reason"] = "no corroborated row"
        evidence.geometry = report
        return evidence
    evidence.rows = rows
    evidence.rows_provenance = "recovered"
    # No absolute ordinal was proven: rows may anchor model cells but never document totals.
    evidence.rows_ordinal_proof = False
    report["status"] = "recovered"
    report["rows"] = len(rows)
    report["refused"] = refused
    evidence.geometry = report
    LOGGER.info("Geometry recovery: %d rows on %s", len(rows), family)
    return evidence
