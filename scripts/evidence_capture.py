"""Read-only serialization of documentary evidence for benchmark captures."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Any, Mapping

SCHEMA_VERSION = "evidence-capture-1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _number(number) -> dict[str, Any]:
    return {"raw": number.raw, "value": str(number.value), "page": number.page,
            "box": list(number.box)}


def _token(token) -> dict[str, Any]:
    return {"text": token.text, "page": token.page, "box": list(token.box),
            "confidence": token.confidence}


def _line(line, *, include_tokens: bool = False) -> dict[str, Any]:
    confidence = (sum(token.confidence for token in line.tokens) / len(line.tokens)
                  if line.tokens else 0.0)
    payload = {
        "text": line.text,
        "page": line.page,
        "box": list(line.box),
        "confidence": round(confidence, 2),
        "token_count": len(line.tokens),
    }
    if include_tokens:
        payload["tokens"] = [_token(token) for token in line.tokens]
    return payload


def _row(row) -> dict[str, Any]:
    return {
        "description": row.description,
        "description_confident": row.description_confident,
        "row_index": row.row_index,
        "secondary_description": row.secondary_description,
        "line": _line(row.line, include_tokens=True),
        "cells": {field: [_number(number) for number in numbers]
                  for field, numbers in row.cells.items()},
        "row_evidence": _jsonable(row.row_evidence),
        "family_votes": _jsonable(row.family_votes),
        "corroborated_amount_mismatch": _jsonable(row.corroborated_amount_mismatch),
    }


def _tax(tax) -> dict[str, Any]:
    return {
        "rate": _number(tax.rate) if tax.rate is not None else None,
        "amount": _number(tax.amount),
        "base": _number(tax.base) if tax.base is not None else None,
    }


def serialize_evidence(evidence, diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Serialize evidence without mutation; top-level fields are P1-compatible."""
    diagnostics = diagnostics or {}
    lines = list(evidence.lines)
    rows = list(evidence.rows)
    totals = {key: [_number(number) for number in values]
              for key, values in (evidence.totals or {}).items()}
    top_rows = [{
        "description": row.description,
        "description_confident": row.description_confident,
        "row_index": row.row_index,
        "secondary_description": row.secondary_description,
        "cells": {field: [str(number.value) for number in numbers]
                  for field, numbers in row.cells.items()},
    } for row in rows]
    serialized = {
        "schema_version": SCHEMA_VERSION,
        "text": evidence.text,
        "text_sha256": hashlib.sha256(evidence.text.encode("utf-8")).hexdigest(),
        "locale": evidence.locale,
        "supplier": evidence.supplier,
        "supplier_raw": evidence.supplier_raw,
        "supplier_is_explicit": evidence.supplier_is_explicit,
        "invoice_number": evidence.invoice_number,
        "invoice_date": evidence.invoice_date,
        "available": evidence.available,
        "source_kind": evidence.source_kind,
        "ocr_cross_checked": evidence.ocr_cross_checked,
        "quality_issues": list(evidence.quality_issues),
        "lines": [_line(line) for line in lines],
        "totals": {key: [str(number.value) for number in values]
                   for key, values in (evidence.totals or {}).items()},
        "rows": top_rows,
        "details": {
            "tokens": [_token(token) for token in evidence.tokens],
            "lines": [_line(line, include_tokens=True) for line in lines],
            "rows": [_row(row) for row in rows],
            "totals": totals,
            "taxes": [_tax(tax) for tax in evidence.taxes],
            "discounts": [_number(number) for number in evidence.discounts],
            "numbers": [_number(number) for number in evidence.numbers],
            "field_candidates": {
                "totals": totals,
                "taxes": [_tax(tax) for tax in evidence.taxes],
                "discounts": [_number(number) for number in evidence.discounts],
                "numbers": [_number(number) for number in evidence.numbers],
                "rows": [{"row_index": row.row_index, "cells": {
                    field: [_number(number) for number in numbers]
                    for field, numbers in row.cells.items()}}
                    for row in rows],
            },
            "supplier_raw": evidence.supplier_raw,
            "supplier_is_explicit": evidence.supplier_is_explicit,
            "corroborated_identifiers": _jsonable(evidence.corroborated_identifiers),
            "issues": list(evidence.issues),
            "ocr": {
                "calls": evidence.ocr_calls,
                "pages": evidence.ocr_pages,
                "elapsed_seconds": evidence.ocr_elapsed_seconds,
                "preprocessing_used": list(evidence.preprocessing_used),
                "fast_path": _jsonable(evidence.fast_path),
                "table_shadow": _jsonable(evidence.table_shadow),
            },
            "rows_provenance": evidence.rows_provenance,
            "source_size": _jsonable(getattr(evidence, "source_size", None)),
            "grounding": {
                "field_statuses": _jsonable(diagnostics.get("field_statuses")),
                "provenance": _jsonable(diagnostics.get("provenance")),
                "warnings": _jsonable(diagnostics.get("warnings")),
                "validation": _jsonable(diagnostics.get("validation")),
                "model_info": _jsonable(diagnostics.get("model_info")),
                "unverified_model_readings": _jsonable(diagnostics.get("unverified_model_readings")),
                "arithmetic_warnings": _jsonable(diagnostics.get("arithmetic_warnings")),
                "ocr_configuration": _jsonable(diagnostics.get("ocr_configuration")),
            },
        },
        "counts": {
            "tokens": len(evidence.tokens),
            "lines": len(lines),
            "rows": len(rows),
        },
    }
    # The top-level totals/rows are used by the frozen P1 assessor. Preserve empty keys
    # expected by that contract even when the evidence contains no candidate.
    for key in ("total_ht", "total_ttc", "total_candidate"):
        serialized["totals"].setdefault(key, [])
    return serialized
