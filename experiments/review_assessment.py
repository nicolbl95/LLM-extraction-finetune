"""Isolated, OFF-by-default review assessment implementing the frozen P1 policy."""

from __future__ import annotations

import copy
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

POLICY_ID = "review-assessment-p1"
POLICY_VERSION = "1.0.0"
SCHEMA_VERSION = "2.1"
VERIFIED_STATUSES = {"vérifié", "valide", "récupéré (corroboré par ocr)"}
TOP_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")
LINE_FIELDS = ("description", "quantity", "unit_price", "total_ht")
NUMERIC_LINE_FIELDS = {"quantity", "unit_price", "total_ht"}
ROW_LINE_MIN_MATCHES = 2
SUPPLIER_LINE_MAX_WORDS = 12
MONEY_TOLERANCE = Decimal("0.02")
CURRENCY_RE = re.compile(r"^(?:€|\$|£|usd|eur|gbp|chf)$", re.IGNORECASE)
NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*")
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value).casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def _words(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", _fold(value))


def _normalize_number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).replace("\u00a0", "").replace(" ", "").strip(".,;:()[]{}€$£")
    if not text:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit():
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _numeric_values(text: str) -> set[Decimal]:
    values = set()
    for token in re.split(r"\s+", text or ""):
        for match in NUMBER_TOKEN_RE.finditer(token):
            value = _normalize_number(match.group(0).rstrip(".,"))
            if value is not None:
                values.add(value)
    return values


def _line_description_prefix(text: str) -> str:
    tokens = text.split()
    while tokens and (re.fullmatch(r"^\d[\d.,]*$", tokens[0]) or CURRENCY_RE.match(tokens[0])
                      or tokens[0] in {"-", "—", ":", "€", "$", "£"}):
        tokens.pop(0)
    description = []
    for token in tokens:
        if re.fullmatch(r"^\d[\d.,]*$", token) or CURRENCY_RE.match(token):
            break
        description.append(token)
    return " ".join(description)


@dataclass(frozen=True)
class EvidenceLine:
    text: str
    confidence: float = 100.0
    page: int = 0
    box: tuple[float, ...] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any) -> "EvidenceLine":
        def get(key: str, default: Any = None) -> Any:
            return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)
        box = get("box")
        return cls(
            text=str(get("text") or ""),
            confidence=float(get("confidence") or 0.0),
            page=int(get("page") or 0),
            box=tuple(box) if box else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence, "page": self.page,
                "box": list(self.box) if self.box else None}


@dataclass(frozen=True)
class EvidenceView:
    text: str = ""
    lines: tuple[EvidenceLine, ...] = ()
    supplier: str | None = None
    supplier_explicit: bool = False
    invoice_date: str | None = None
    total_ht: tuple[Decimal, ...] = ()
    total_ttc: tuple[Decimal, ...] = ()
    total_candidate: tuple[Decimal, ...] = ()
    rows: tuple[Mapping[str, Any], ...] = ()
    quality_issues: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EvidenceView":
        if not value:
            return cls()
        lines = tuple(EvidenceLine.from_mapping(line) for line in (value.get("lines") or []))
        text = str(value.get("text") or "\n".join(line.text for line in lines))
        totals = value.get("totals") or {}

        def amounts(name: str) -> tuple[Decimal, ...]:
            values = []
            for item in totals.get(name) or []:
                number = _normalize_number(item)
                if number is not None:
                    values.append(number)
            return tuple(values)

        return cls(
            text=text,
            lines=lines,
            supplier=value.get("supplier"),
            supplier_explicit=bool(value.get("supplier_is_explicit") or value.get("supplier_explicit")),
            invoice_date=value.get("invoice_date"),
            total_ht=amounts("total_ht"),
            total_ttc=amounts("total_ttc"),
            total_candidate=amounts("total_candidate"),
            rows=tuple(value.get("rows") or ()),
            quality_issues=tuple(value.get("quality_issues") or ()),
        )

    @property
    def numbers(self) -> set[Decimal]:
        return _numeric_values(self.text)


@dataclass(frozen=True)
class FieldAssessment:
    path: str
    field: str
    row_index: int | None
    status: str
    value: Any
    model_value: Any = None
    reason: str = ""
    basis: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    review_required: bool = False
    verification_conflict: bool = False
    row: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "value": _json_value(self.value),
            "model_value": _json_value(self.model_value),
            "reason": self.reason,
            "basis": self.basis,
            "review_required": self.review_required,
            "verification_conflict": self.verification_conflict,
            "evidence": {key: _json_value(value) for key, value in self.evidence.items()},
            "checks": {key: _json_value(value) for key, value in self.checks.items()},
            "row": self.row,
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class _GateOutcome:
    status: str
    reason: str
    evidence: dict[str, Any]
    checks: dict[str, Any]


@dataclass(frozen=True)
class _RowContext:
    index: int
    source: str
    line: EvidenceLine | None
    identity: str

    def to_dict(self) -> dict[str, Any]:
        payload = {"index": self.index, "order": self.index, "source": self.source,
                   "identity": self.identity}
        if self.line is not None:
            payload.update({"line": self.line.text, "line_confidence": self.line.confidence,
                            "page": self.line.page, "box": list(self.line.box) if self.line.box else None})
        return payload


def _line_evidence(line: EvidenceLine | None, kind: str) -> dict[str, Any]:
    if line is None:
        return {"kind": kind}
    return {"kind": kind, **line.to_dict()}


def _best_description_line(raw: Any, evidence: EvidenceView) -> tuple[float, EvidenceLine | None]:
    if not isinstance(raw, str) or not raw.strip():
        return 0.0, None
    raw_words = _words(raw)
    if not raw_words:
        return 0.0, None
    best_score, best_line = 0.0, None
    for line in evidence.lines:
        prefix_words = _words(_line_description_prefix(line.text))
        if not prefix_words:
            continue
        coverage = sum(1 for word in raw_words if word in prefix_words) / len(raw_words)
        ratio = len(raw_words) / len(prefix_words)
        score = min(coverage, ratio)
        if score > best_score:
            best_score, best_line = score, line
    return best_score, best_line


def _row_line(values: Mapping[str, Any], evidence: EvidenceView) -> EvidenceLine | None:
    numbers = [_normalize_number(values.get(key)) for key in ("quantity", "unit_price", "total_ht")]
    numbers = [number for number in numbers if number is not None]
    description = values.get("description")
    description_line = _best_description_line(description, evidence)[1]
    if description_line is not None:
        line_numbers = _numeric_values(description_line.text)
        if sum(1 for number in numbers if number in line_numbers) >= ROW_LINE_MIN_MATCHES:
            return description_line
    best_count, best_line = 0, None
    for line in evidence.lines:
        line_numbers = _numeric_values(line.text)
        matches = sum(1 for number in numbers if number in line_numbers)
        if matches > best_count:
            best_count, best_line = matches, line
    if best_count < ROW_LINE_MIN_MATCHES:
        return None
    if best_line is not None and "total" in _fold(best_line.text) and best_count < 3:
        return None
    return best_line


def _description_complete(raw: Any, line: EvidenceLine | None) -> bool:
    if not isinstance(raw, str) or line is None:
        return False
    prefix_words = _words(_line_description_prefix(line.text))
    raw_words = _words(raw)
    return bool(prefix_words) and len(raw_words) >= len(prefix_words) and all(
        word in raw_words for word in prefix_words
    )


def _supplier_support(raw: Any, evidence: EvidenceView) -> tuple[bool, EvidenceLine | None, str]:
    if not isinstance(raw, str) or not raw.strip():
        return False, None, "not_found"
    folded = _fold(raw)
    for line in evidence.lines:
        text = _fold(line.text)
        if folded and folded in text and len(_words(text)) <= SUPPLIER_LINE_MAX_WORDS and not re.search(r"\d", text):
            return True, line, "line"
    if evidence.supplier and _fold(evidence.supplier) == folded:
        return True, None, "evidence_supplier"
    return False, None, "not_found"


def _date_supported(raw: Any, evidence: EvidenceView) -> bool:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(raw or ""))
    if not match:
        return False
    year, month, day = (int(part) for part in match.groups())
    text = _fold(evidence.text)
    variants = [
        f"{day:02d}/{month:02d}/{year}", f"{day}/{month}/{year}", f"{year}-{month:02d}-{day:02d}",
        f"{day:02d}-{month:02d}-{year}", f"{day}-{month}-{year}",
    ]
    if any(variant in text for variant in variants):
        return True
    for name, month_value in MONTHS.items():
        if month_value != month:
            continue
        if re.search(rf"\b{name}\s+0?{day}\b[^\n]{{0,8}}\b{year}\b", text):
            return True
        if re.search(rf"\b0?{day}\s+{name}\b[^\n]{{0,8}}\b{year}\b", text):
            return True
    return False


def _arithmetic_consistent(row: Mapping[str, Any]) -> bool | None:
    quantity = _normalize_number(row.get("quantity"))
    price = _normalize_number(row.get("unit_price"))
    total = _normalize_number(row.get("total_ht"))
    if None in (quantity, price, total):
        return None
    return abs(quantity * price - total) <= MONEY_TOLERANCE


def _gate(field: str, row_index: int | None, raw_value: Any, raw_row: Mapping[str, Any],
          evidence: EvidenceView, row_line: EvidenceLine | None) -> _GateOutcome:
    if field == "description":
        if row_index is None:
            return _GateOutcome("rejected", "unsupported_field", {}, {})
        if row_line is None:
            return _GateOutcome("rejected", "row_line_not_found", {}, {"row_attribution": "unverified_no_rows"})
        _, best_line = _best_description_line(raw_value, evidence)
        if best_line is not row_line:
            return _GateOutcome("rejected", "row_attribution_conflict", {}, {"row_attribution": "row_local_ocr_line"})
        if not _description_complete(raw_value, best_line):
            return _GateOutcome("rejected", "incomplete_description", _line_evidence(best_line, "ocr_line"),
                                {"row_attribution": "row_local_ocr_line"})
        return _GateOutcome("proposed", "ocr_complete_description", _line_evidence(best_line, "ocr_line"),
                            {"row_attribution": "row_local_ocr_line"})
    if field in NUMERIC_LINE_FIELDS and row_index is not None:
        if row_line is None:
            return _GateOutcome("rejected", "row_line_not_found", {}, {"row_attribution": "unverified_no_rows"})
        number = _normalize_number(raw_value)
        if number is None or number not in _numeric_values(row_line.text):
            return _GateOutcome("rejected", "not_on_row_line", _line_evidence(row_line, "ocr_line"),
                                {"row_attribution": "row_local_ocr_line"})
        if field == "total_ht":
            top_values = set(evidence.total_ht) | set(evidence.total_ttc) | set(evidence.total_candidate)
            if number in top_values:
                return _GateOutcome("rejected", "global_total_as_line", _line_evidence(row_line, "ocr_line"),
                                    {"row_attribution": "row_local_ocr_line"})
        if field == "quantity":
            quantity = _normalize_number(raw_row.get("quantity"))
            price = _normalize_number(raw_row.get("unit_price"))
            if quantity is not None and price is not None and quantity == price:
                return _GateOutcome("rejected", "quantity_equals_unit_price", _line_evidence(row_line, "ocr_line"),
                                    {"row_attribution": "row_local_ocr_line"})
        return _GateOutcome("proposed", "ocr_row_line", _line_evidence(row_line, "ocr_line"),
                            {"row_attribution": "row_local_ocr_line",
                             "arithmetic_consistent": _arithmetic_consistent(raw_row),
                             "line": row_line.text})
    if field == "supplier_name":
        if evidence.supplier and _fold(evidence.supplier) != _fold(raw_value):
            return _GateOutcome("rejected", "contradicted_supplier", {}, {})
        supported, line, source = _supplier_support(raw_value, evidence)
        if supported:
            return _GateOutcome("proposed", f"ocr_supplier_{source}", _line_evidence(line, "ocr_supplier"),
                                {"documentary_source": source})
        return _GateOutcome("rejected", "supplier_not_found", {}, {})
    if field == "invoice_date":
        if evidence.invoice_date and str(evidence.invoice_date) != str(raw_value):
            return _GateOutcome("rejected", "contradicted_date", {}, {})
        if _date_supported(raw_value, evidence):
            return _GateOutcome("proposed", "ocr_date", {"kind": "ocr_date", "text": str(raw_value)}, {})
        return _GateOutcome("rejected", "date_not_found", {}, {})
    if field in {"total_ht", "total_ttc"}:
        number = _normalize_number(raw_value)
        if number is None:
            return _GateOutcome("rejected", "unsupported_total", {}, {})
        if field == "total_ht":
            if number in evidence.total_ht:
                return _GateOutcome("proposed", "labelled_ht", {"kind": "ocr_total", "value": str(number), "label": "HT"},
                                    {"label_conflict": False})
            if number in evidence.total_ttc or number in evidence.total_candidate:
                return _GateOutcome("rejected", "ttc_as_ht", {}, {})
        else:
            if number in evidence.total_ttc:
                return _GateOutcome("proposed", "labelled_ttc", {"kind": "ocr_total", "value": str(number), "label": "TTC"},
                                    {"label_conflict": False})
            if number in evidence.total_ht:
                return _GateOutcome("rejected", "ht_as_ttc", {}, {})
            if number in evidence.total_candidate:
                return _GateOutcome("proposed", "unqualified_total", {"kind": "ocr_total", "value": str(number)},
                                    {"label_conflict": True})
        if number in evidence.numbers:
            return _GateOutcome("proposed", "missing_label", {"kind": "ocr_total", "value": str(number)},
                                {"label_conflict": True})
        return _GateOutcome("rejected", "not_in_document", {}, {})
    return _GateOutcome("rejected", "unsupported_field", {}, {})


def _same_value(field: str, left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if field in {"quantity", "unit_price", "total_ht", "total_ttc"}:
        a, b = _normalize_number(left), _normalize_number(right)
        return a is not None and b is not None and abs(a - b) <= MONEY_TOLERANCE
    return unicodedata.normalize("NFC", str(left)).strip() == unicodedata.normalize("NFC", str(right)).strip()


def _status_value(raw: Any) -> str:
    if isinstance(raw, Mapping):
        raw = raw.get("status") or raw.get("statut") or ""
    return str(raw).strip().casefold()


def _status_for(statuses: Mapping[str, Any], path: str, row_index: int | None, field: str) -> str:
    candidates = [path]
    if row_index is not None:
        candidates.extend([f"line_items.{row_index}.{field}", f"line_items[{row_index}].{field}"])
    for candidate in candidates:
        if candidate in statuses:
            return _status_value(statuses[candidate])
    return ""


def _items(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("line_items") or []
    return [item if isinstance(item, Mapping) else {} for item in items]


def _evidence_row(evidence: EvidenceView, index: int) -> Mapping[str, Any] | None:
    return evidence.rows[index] if index < len(evidence.rows) else None


def _missing_status(field: str, row_index: int | None, evidence: EvidenceView,
                    row: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if row_index is not None:
        if row is None:
            candidate = False
        elif field == "description":
            candidate = bool(row.get("description"))
        else:
            cells = row.get("cells") or {}
            candidate = bool(cells.get(field))
        return ("missing_printed" if candidate else "absent",
                {"documentary_candidate": candidate, "row_index": row_index})
    if field == "supplier_name":
        candidate = bool(evidence.supplier)
    elif field == "invoice_date":
        candidate = bool(evidence.invoice_date)
    elif field == "total_ht":
        candidate = bool(evidence.total_ht)
    elif field == "total_ttc":
        candidate = bool(evidence.total_ttc or evidence.total_candidate)
    else:
        candidate = False
    return ("missing_printed" if candidate else "absent",
            {"documentary_candidate": candidate, "row_index": row_index})


def _row_context(index: int, raw_row: Mapping[str, Any], strict_row: Mapping[str, Any],
                 evidence_row: Mapping[str, Any] | None, line: EvidenceLine | None) -> _RowContext:
    if evidence_row is not None:
        source, identity = "evidence_row", "evidence_row"
    elif any(value is not None for value in raw_row.values()):
        source, identity = "raw_model_order", "raw_model_order_unverified"
    elif any(value is not None for value in strict_row.values()):
        source, identity = "strict_order", "strict_order"
    else:
        source, identity = "none", "none"
    return _RowContext(index=index, source=source, line=line, identity=identity)


def assess(evidence: Mapping[str, Any] | EvidenceView | None,
           raw_payload: Mapping[str, Any] | None,
           strict_result: Mapping[str, Any] | None) -> dict[str, FieldAssessment]:
    """Classify every candidate field without mutating inputs or promoting proposals."""
    evidence_view = evidence if isinstance(evidence, EvidenceView) else EvidenceView.from_mapping(evidence)
    raw = raw_payload if isinstance(raw_payload, Mapping) else {}
    strict = strict_result if isinstance(strict_result, Mapping) else {}
    payload_candidate = strict.get("payload") if isinstance(strict, Mapping) else None
    if isinstance(strict, Mapping) and "payload" in strict:
        strict_payload = payload_candidate if isinstance(payload_candidate, Mapping) else {}
    else:
        strict_payload = strict
    statuses = strict.get("statuses") or strict.get("field_statuses") or {}
    if not isinstance(statuses, Mapping):
        statuses = {}
    raw_items = _items(raw)
    strict_items = _items(strict_payload)
    row_count = max(len(raw_items), len(strict_items), len(evidence_view.rows))
    assessments: dict[str, FieldAssessment] = {}

    for field in TOP_FIELDS:
        strict_value = strict_payload.get(field) if isinstance(strict_payload, Mapping) else None
        raw_value = raw.get(field) if isinstance(raw, Mapping) else None
        status = _status_for(statuses, field, None, field)
        verified = strict_value is not None and status in VERIFIED_STATUSES
        conflict = verified and raw_value is not None and not _same_value(field, strict_value, raw_value)
        if verified:
            assessments[field] = FieldAssessment(
                path=field, field=field, row_index=None, status="verified",
                value=strict_value, model_value=raw_value, reason="final_status", basis="document",
                checks={"strict_status": status, "verification_conflict": conflict},
                review_required=conflict, verification_conflict=conflict,
            )
            continue
        if raw_value is None:
            missing, checks = _missing_status(field, None, evidence_view, None)
            assessments[field] = FieldAssessment(
                path=field, field=field, row_index=None, status=missing, value=None,
                reason="no_model_value", checks=checks,
            )
            continue
        gate = _gate(field, None, raw_value, {}, evidence_view, None)
        assessments[field] = FieldAssessment(
            path=field, field=field, row_index=None,
            status=gate.status, value=raw_value if gate.status == "proposed" else None,
            model_value=raw_value, reason=gate.reason, basis="model_proposal" if gate.status == "proposed" else None,
            evidence=gate.evidence, checks=gate.checks,
            review_required=gate.status == "proposed",
        )

    for index in range(row_count):
        raw_row = raw_items[index] if index < len(raw_items) else {}
        strict_row = strict_items[index] if index < len(strict_items) else {}
        evidence_row = _evidence_row(evidence_view, index)
        line = _row_line(raw_row, evidence_view) if raw_row else None
        context = _row_context(index, raw_row, strict_row, evidence_row, line)
        for field in LINE_FIELDS:
            path = f"line_items[{index}].{field}"
            strict_value = strict_row.get(field) if isinstance(strict_row, Mapping) else None
            raw_value = raw_row.get(field) if isinstance(raw_row, Mapping) else None
            status = _status_for(statuses, path, index, field)
            verified = strict_value is not None and status in VERIFIED_STATUSES
            conflict = verified and raw_value is not None and not _same_value(field, strict_value, raw_value)
            if verified:
                assessments[path] = FieldAssessment(
                    path=path, field=field, row_index=index, status="verified",
                    value=strict_value, model_value=raw_value, reason="final_status", basis="document",
                    checks={"strict_status": status, "verification_conflict": conflict},
                    review_required=conflict, verification_conflict=conflict, row=context.to_dict(),
                )
                continue
            if raw_value is None:
                missing, checks = _missing_status(field, index, evidence_view, evidence_row)
                assessments[path] = FieldAssessment(
                    path=path, field=field, row_index=index, status=missing, value=None,
                    reason="no_model_value", checks=checks, row=context.to_dict(),
                )
                continue
            gate = _gate(field, index, raw_value, raw_row, evidence_view, line)
            assessments[path] = FieldAssessment(
                path=path, field=field, row_index=index,
                status=gate.status, value=raw_value if gate.status == "proposed" else None,
                model_value=raw_value, reason=gate.reason,
                basis="model_proposal" if gate.status == "proposed" else None,
                evidence=gate.evidence, checks=gate.checks,
                review_required=gate.status == "proposed", row=context.to_dict(),
            )
    return assessments


def serialize_strict(strict_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an exact deep copy of the existing verified-only payload."""
    return copy.deepcopy(dict(strict_payload))


def _normalized_assessments(assessments: Mapping[str, FieldAssessment] | Iterable[FieldAssessment]
                           ) -> dict[str, FieldAssessment]:
    if isinstance(assessments, Mapping):
        return dict(assessments)
    return {item.path: item for item in assessments}


def serialize_review(strict_payload: Mapping[str, Any],
                     assessments: Mapping[str, FieldAssessment] | Iterable[FieldAssessment],
                     schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    """Return the verified primary payload plus a separate review assessment map."""
    result = serialize_strict(strict_payload)
    normalized = _normalized_assessments(assessments)
    result["mode"] = "review_assisted"
    result["schema_version"] = schema_version
    result["field_assessments"] = {path: item.to_dict() for path, item in normalized.items()}
    counts = Counter(item.status for item in normalized.values())
    result["review_summary"] = {
        "verified": counts.get("verified", 0),
        "proposed": counts.get("proposed", 0),
        "rejected": counts.get("rejected", 0),
        "missing_printed": counts.get("missing_printed", 0),
        "absent": counts.get("absent", 0),
        "review_required": sum(1 for item in normalized.values() if item.review_required),
        "verification_conflicts": sum(1 for item in normalized.values() if item.verification_conflict),
    }
    return result
