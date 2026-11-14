"""Offline annotation validation and adjudication tools; no OCR/model imports."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


ANNOTATION_VERSION = "invoice-annotation-v1"
SELECTION_VERSION = "invoice-holdout-selection-v1"
ARBITRATION_VERSION = "invoice-arbitration-v1"
FIELD_STATES = {
    "present",
    "absent",
    "unreadable",
    "ambiguous",
    "derived_only",
    "unscorable",
}
TEXT_FIELDS = {"supplier_name", "description", "label", "currency"}
DATE_FIELDS = {"invoice_date"}
DECIMAL_FIELDS = {
    "quantity",
    "unit_price",
    "total_ht",
    "total_ttc",
    "rate",
    "base",
    "amount",
}
PRODUCTION_INVOICE_FIELDS = (
    "supplier_name",
    "invoice_date",
    "total_ht",
    "total_ttc",
)
PRODUCTION_LINE_FIELDS = ("description", "quantity", "unit_price", "total_ht")
SPLITS = ("development", "validation", "final_test")
LOCALES = ("fr-FR", "en-GB", "en-US")
CENT = Decimal("0.01")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
ROW_PATH_RE = re.compile(
    r"^(line_items|taxes|discounts|financial_summary)\[([A-Za-z0-9._-]+)\]\.(.+)$"
)


def _issue(path: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"severity": severity, "path": path, "message": message}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _decimal(value: Any) -> Decimal | None:
    if not isinstance(value, str) or not DECIMAL_RE.fullmatch(value):
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _parse_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _check_keys(
    value: Any,
    required: set[str],
    path: str,
    issues: list[dict[str, str]],
) -> bool:
    if not isinstance(value, dict):
        issues.append(_issue(path, "must be an object"))
        return False
    missing = required - value.keys()
    extra = value.keys() - required
    for key in sorted(missing):
        issues.append(_issue(f"{path}.{key}", "required property is missing"))
    for key in sorted(extra):
        issues.append(_issue(f"{path}.{key}", "additional property is not allowed"))
    return not missing and not extra


def _validate_box(box: Any, path: str, issues: list[dict[str, str]]) -> None:
    if not isinstance(box, list) or len(box) != 4:
        issues.append(_issue(path, "must be [x0, y0, x1, y1]"))
        return
    if any(
        isinstance(coord, bool)
        or not isinstance(coord, (int, float))
        or not math.isfinite(coord)
        or not 0 <= coord <= 1
        for coord in box
    ):
        issues.append(_issue(path, "coordinates must be finite numbers from 0 to 1"))
        return
    x0, y0, x1, y1 = box
    if x0 >= x1 or y0 >= y1:
        issues.append(_issue(path, "box must have positive width and height"))


def _validate_evidence(
    evidence: Any, path: str, issues: list[dict[str, str]]
) -> None:
    if not isinstance(evidence, list):
        issues.append(_issue(path, "must be an array"))
        return
    for index, item in enumerate(evidence):
        item_path = f"{path}[{index}]"
        if not _check_keys(
            item, {"page_index", "bbox", "label_text"}, item_path, issues
        ):
            continue
        if not _is_int(item["page_index"]) or item["page_index"] < 0:
            issues.append(_issue(f"{item_path}.page_index", "must be >= 0"))
        _validate_box(item["bbox"], f"{item_path}.bbox", issues)
        if item["label_text"] is not None and not isinstance(item["label_text"], str):
            issues.append(_issue(f"{item_path}.label_text", "must be a string or null"))


def _validate_field(
    field: Any,
    field_name: str,
    path: str,
    issues: list[dict[str, str]],
) -> None:
    required = {
        "state",
        "visible_text",
        "normalized_value",
        "evidence",
        "candidates",
        "derivation",
        "note",
    }
    if not _check_keys(field, required, path, issues):
        return

    state = field["state"]
    visible = field["visible_text"]
    normalized = field["normalized_value"]
    evidence = field["evidence"]
    candidates = field["candidates"]
    derivation = field["derivation"]
    note = field["note"]

    if not isinstance(state, str) or state not in FIELD_STATES:
        issues.append(_issue(f"{path}.state", "unknown field state"))
        return
    if visible is not None and not isinstance(visible, str):
        issues.append(_issue(f"{path}.visible_text", "must be a string or null"))
    if normalized is not None and not isinstance(normalized, str):
        issues.append(_issue(f"{path}.normalized_value", "must be a string or null"))
    _validate_evidence(evidence, f"{path}.evidence", issues)
    if not isinstance(candidates, list) or any(
        not isinstance(candidate, str) or not candidate.strip()
        for candidate in candidates
    ):
        issues.append(_issue(f"{path}.candidates", "must contain non-empty strings"))
    if note is not None and not isinstance(note, str):
        issues.append(_issue(f"{path}.note", "must be a string or null"))

    if state == "present":
        if not isinstance(visible, str) or not visible.strip():
            issues.append(_issue(f"{path}.visible_text", "present requires printed text"))
        if not isinstance(normalized, str) or not normalized.strip():
            issues.append(
                _issue(f"{path}.normalized_value", "present requires a normalized value")
            )
        if isinstance(normalized, str):
            if field_name in TEXT_FIELDS and isinstance(visible, str):
                if normalized != _normalized_text(visible):
                    issues.append(
                        _issue(
                            f"{path}.normalized_value",
                            "text must be NFC with whitespace collapsed; preserve wording",
                        )
                    )
            elif field_name in DATE_FIELDS:
                try:
                    if dt.date.fromisoformat(normalized).isoformat() != normalized:
                        raise ValueError
                except ValueError:
                    issues.append(
                        _issue(f"{path}.normalized_value", "date must be ISO YYYY-MM-DD")
                    )
            elif field_name in DECIMAL_FIELDS and _decimal(normalized) is None:
                issues.append(
                    _issue(
                        f"{path}.normalized_value",
                        "numeric value must be a finite canonical decimal string",
                    )
                )
        if isinstance(evidence, list) and not evidence:
            issues.append(_issue(f"{path}.evidence", "present requires visual evidence"))
        if candidates:
            issues.append(_issue(f"{path}.candidates", "present cannot have alternatives"))
        if derivation is not None:
            issues.append(_issue(f"{path}.derivation", "printed values cannot be derived"))
    else:
        if normalized is not None:
            issues.append(
                _issue(f"{path}.normalized_value", f"{state} must not assert a printed value")
            )
        if not isinstance(note, str) or not note.strip():
            issues.append(_issue(f"{path}.note", f"{state} requires an explanation"))
        if state == "absent":
            if visible is not None or candidates:
                issues.append(
                    _issue(path, "absent must not contain printed text or candidates")
                )
            if derivation is not None:
                issues.append(_issue(f"{path}.derivation", "absent cannot be derived"))
        elif state == "unreadable":
            if not isinstance(evidence, list) or not evidence:
                issues.append(_issue(f"{path}.evidence", "unreadable requires a source box"))
            if derivation is not None:
                issues.append(_issue(f"{path}.derivation", "unreadable cannot be derived"))
        elif state == "ambiguous":
            if visible is not None or not isinstance(candidates, list) or len(candidates) < 2:
                issues.append(
                    _issue(
                        path,
                        "ambiguous requires null visible_text and at least two candidates",
                    )
                )
            elif len(set(candidates)) != len(candidates):
                issues.append(
                    _issue(f"{path}.candidates", "ambiguous candidates must be distinct")
                )
            if not isinstance(evidence, list) or not evidence:
                issues.append(_issue(f"{path}.evidence", "ambiguous requires a source box"))
            if derivation is not None:
                issues.append(_issue(f"{path}.derivation", "ambiguous cannot be derived"))
        elif state == "derived_only":
            if visible is not None or candidates:
                issues.append(
                    _issue(path, "derived_only must not claim printed text or candidates")
                )
            if not isinstance(derivation, dict):
                issues.append(_issue(f"{path}.derivation", "derived_only requires a formula"))
            else:
                if set(derivation) != {"value", "formula", "source_paths"}:
                    issues.append(
                        _issue(f"{path}.derivation", "invalid derivation properties")
                    )
                if _decimal(derivation.get("value")) is None:
                    issues.append(
                        _issue(f"{path}.derivation.value", "must be a finite decimal string")
                    )
                if not isinstance(derivation.get("formula"), str) or not derivation[
                    "formula"
                ].strip():
                    issues.append(
                        _issue(f"{path}.derivation.formula", "must explain the calculation")
                    )
                if (
                    not isinstance(derivation.get("source_paths"), list)
                    or not derivation["source_paths"]
                    or any(not isinstance(item, str) or not item for item in derivation["source_paths"])
                ):
                    issues.append(
                        _issue(
                            f"{path}.derivation.source_paths",
                            "must identify its input fields",
                        )
                    )
        elif state == "unscorable":
            if visible is not None or candidates or derivation is not None:
                issues.append(
                    _issue(
                        path,
                        "unscorable cannot assert a value, candidates or derivation",
                    )
                )


def _validate_field_map(
    fields: Any,
    expected: tuple[str, ...],
    path: str,
    issues: list[dict[str, str]],
) -> None:
    if not _check_keys(fields, set(expected), path, issues):
        return
    for name in expected:
        _validate_field(fields[name], name, f"{path}.{name}", issues)


def validate_annotation(annotation: Any) -> list[dict[str, str]]:
    """Return structural and provenance diagnostics without reading an invoice."""
    issues: list[dict[str, str]] = []
    root_keys = {
        "schema_version",
        "annotation_kind",
        "annotation_id",
        "protocol_version",
        "created_at",
        "document",
        "reviewer",
        "provenance",
        "invoice",
        "line_items",
        "taxes",
        "discounts",
        "financial_summary",
        "notes",
    }
    if not _check_keys(annotation, root_keys, "$", issues):
        return issues

    if annotation["schema_version"] != ANNOTATION_VERSION:
        issues.append(_issue("$.schema_version", "unsupported schema version"))
    kind = annotation["annotation_kind"]
    if not isinstance(kind, str) or kind not in {
        "independent",
        "adjudicated",
        "synthetic_test",
    }:
        issues.append(_issue("$.annotation_kind", "unknown annotation kind"))
    for key in ("annotation_id", "protocol_version"):
        if not isinstance(annotation[key], str) or not annotation[key].strip():
            issues.append(_issue(f"$.{key}", "must be a non-empty string"))
    if not _parse_timestamp(annotation["created_at"]):
        issues.append(_issue("$.created_at", "must be an ISO timestamp with timezone"))

    document = annotation["document"]
    if _check_keys(
        document,
        {
            "document_id",
            "relative_path",
            "sha256",
            "file_type",
            "country",
            "language",
            "currency",
        },
        "$.document",
        issues,
    ):
        for key in ("document_id", "relative_path"):
            if not isinstance(document[key], str) or not document[key].strip():
                issues.append(_issue(f"$.document.{key}", "must be a non-empty string"))
        if isinstance(document["relative_path"], str):
            relative_path = Path(document["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                issues.append(
                    _issue(
                        "$.document.relative_path",
                        "must be relative and stay below the asset root",
                    )
                )
        if not isinstance(document["sha256"], str) or not SHA256_RE.fullmatch(
            document["sha256"]
        ):
            issues.append(_issue("$.document.sha256", "must be lowercase SHA-256 hex"))
        if not isinstance(document["file_type"], str) or document["file_type"] not in {
            "pdf",
            "png",
            "jpg",
            "jpeg",
        }:
            issues.append(_issue("$.document.file_type", "unsupported file type"))
        for key in ("country", "language"):
            if document[key] is not None and not isinstance(document[key], str):
                issues.append(_issue(f"$.document.{key}", "must be a string or null"))
        _validate_field(document["currency"], "currency", "$.document.currency", issues)

    reviewer = annotation["reviewer"]
    if _check_keys(
        reviewer,
        {"reviewer_id", "role", "attestation"},
        "$.reviewer",
        issues,
    ):
        if not isinstance(reviewer["reviewer_id"], str) or not reviewer[
            "reviewer_id"
        ].strip():
            issues.append(_issue("$.reviewer.reviewer_id", "must not be empty"))
        if not isinstance(reviewer["role"], str) or reviewer["role"] not in {
            "independent_annotator",
            "adjudicator",
            "test_fixture",
        }:
            issues.append(_issue("$.reviewer.role", "unknown reviewer role"))
        attestation = reviewer["attestation"]
        attestation_keys = {
            "source_image_viewed",
            "predictions_viewed",
            "prior_annotations_viewed",
            "peer_annotation_viewed",
            "pr64_to_pr69_viewed",
        }
        if _check_keys(attestation, attestation_keys, "$.reviewer.attestation", issues):
            if any(not isinstance(value, bool) for value in attestation.values()):
                issues.append(_issue("$.reviewer.attestation", "all attestations must be boolean"))
            if kind == "independent":
                if reviewer["role"] != "independent_annotator":
                    issues.append(
                        _issue("$.reviewer.role", "independent annotation needs annotator role")
                    )
                if attestation.get("source_image_viewed") is not True:
                    issues.append(
                        _issue("$.reviewer.attestation.source_image_viewed", "must be true")
                    )
                for key in (
                    "predictions_viewed",
                    "prior_annotations_viewed",
                    "peer_annotation_viewed",
                    "pr64_to_pr69_viewed",
                ):
                    if attestation.get(key) is not False:
                        issues.append(
                            _issue(
                                f"$.reviewer.attestation.{key}",
                                "independent annotator must attest false",
                            )
                        )
            elif kind == "adjudicated":
                if reviewer["role"] != "adjudicator":
                    issues.append(
                        _issue("$.reviewer.role", "adjudicated annotation needs adjudicator role")
                    )
                if attestation.get("source_image_viewed") is not True:
                    issues.append(
                        _issue("$.reviewer.attestation.source_image_viewed", "must be true")
                    )
                if attestation.get("predictions_viewed") is not False or attestation.get(
                    "prior_annotations_viewed"
                ) is not False:
                    issues.append(
                        _issue(
                            "$.reviewer.attestation",
                            "adjudicator must not view predictions or old references",
                        )
                    )
                if attestation.get("peer_annotation_viewed") is not True:
                    issues.append(
                        _issue(
                            "$.reviewer.attestation.peer_annotation_viewed",
                            "adjudicator must review both annotations",
                        )
                    )
                if attestation.get("pr64_to_pr69_viewed") is not False:
                    issues.append(
                        _issue(
                            "$.reviewer.attestation.pr64_to_pr69_viewed",
                            "adjudicator must not view prior experiment results",
                        )
                    )

    provenance = annotation["provenance"]
    if _check_keys(
        provenance,
        {
            "source_kind",
            "derived_from_existing_reference",
            "permission_status",
            "prior_use_status",
        },
        "$.provenance",
        issues,
    ):
        if not isinstance(provenance["source_kind"], str) or provenance["source_kind"] not in {
            "manual_visual",
            "human_adjudication",
            "synthetic_test",
        }:
            issues.append(_issue("$.provenance.source_kind", "unknown source kind"))
        if not isinstance(provenance["derived_from_existing_reference"], bool):
            issues.append(
                _issue("$.provenance.derived_from_existing_reference", "must be boolean")
            )
        if not isinstance(provenance["permission_status"], str) or provenance[
            "permission_status"
        ] not in {
            "approved",
            "pending",
            "not_applicable",
        }:
            issues.append(_issue("$.provenance.permission_status", "unknown permission state"))
        if not isinstance(provenance["prior_use_status"], str) or provenance[
            "prior_use_status"
        ] not in {
            "none_confirmed",
            "known_used",
            "unknown",
            "not_applicable",
        }:
            issues.append(_issue("$.provenance.prior_use_status", "unknown prior-use state"))
        if kind == "independent":
            if provenance["source_kind"] != "manual_visual":
                issues.append(
                    _issue("$.provenance.source_kind", "independent labels must be manual")
                )
            if provenance["derived_from_existing_reference"] is not False:
                issues.append(
                    _issue(
                        "$.provenance.derived_from_existing_reference",
                        "independent labels cannot derive from old references",
                    )
                )
            if provenance["permission_status"] != "approved":
                issues.append(_issue("$.provenance.permission_status", "must be approved"))
            if provenance["prior_use_status"] != "none_confirmed":
                issues.append(
                    _issue(
                        "$.provenance.prior_use_status",
                        "holdout prior use must be checked and clear",
                    )
                )
        elif kind == "adjudicated":
            if provenance["source_kind"] != "human_adjudication":
                issues.append(
                    _issue(
                        "$.provenance.source_kind",
                        "adjudicated labels require human adjudication",
                    )
                )
            if provenance["derived_from_existing_reference"] is not False:
                issues.append(
                    _issue(
                        "$.provenance.derived_from_existing_reference",
                        "adjudication must use the source image, not old truth",
                    )
                )
            if provenance["permission_status"] != "approved":
                issues.append(_issue("$.provenance.permission_status", "must be approved"))
            if provenance["prior_use_status"] != "none_confirmed":
                issues.append(
                    _issue(
                        "$.provenance.prior_use_status",
                        "holdout prior use must be checked and clear",
                    )
                )
        elif kind == "synthetic_test" and provenance["source_kind"] != "synthetic_test":
            issues.append(
                _issue("$.provenance.source_kind", "test fixture origin must be explicit")
            )

    _validate_field_map(
        annotation["invoice"],
        PRODUCTION_INVOICE_FIELDS,
        "$.invoice",
        issues,
    )
    line_items = annotation["line_items"]
    _validate_records(
        line_items,
        "line_items",
        {"row_id", "row_state", "page_index", "bbox", "fields"},
        "row_id",
        PRODUCTION_LINE_FIELDS,
        issues,
        row=True,
    )
    _validate_records(
        annotation["taxes"],
        "taxes",
        {"tax_id", "page_index", "bbox", "fields"},
        "tax_id",
        ("label", "rate", "base", "amount"),
        issues,
    )
    _validate_records(
        annotation["discounts"],
        "discounts",
        {"discount_id", "page_index", "bbox", "fields"},
        "discount_id",
        ("description", "amount"),
        issues,
    )
    summary = annotation["financial_summary"]
    if not isinstance(summary, list):
        issues.append(_issue("$.financial_summary", "must be an array"))
    else:
        seen_ids: set[str] = set()
        for index, entry in enumerate(summary):
            path = f"$.financial_summary[{index}]"
            if not _check_keys(
                entry,
                {"entry_id", "kind", "page_index", "bbox", "fields"},
                path,
                issues,
            ):
                continue
            entry_id = entry["entry_id"]
            if not isinstance(entry_id, str) or not entry_id.strip():
                issues.append(_issue(f"{path}.entry_id", "must not be empty"))
            elif entry_id in seen_ids:
                issues.append(_issue(f"{path}.entry_id", "must be unique"))
            seen_ids.add(entry_id)
            if not isinstance(entry["kind"], str) or entry["kind"] not in {
                "subtotal",
                "tax",
                "shipping",
                "discount",
                "balance",
                "total",
                "other",
            }:
                issues.append(_issue(f"{path}.kind", "unknown summary kind"))
            if not _is_int(entry["page_index"]) or entry["page_index"] < 0:
                issues.append(_issue(f"{path}.page_index", "must be >= 0"))
            _validate_box(entry["bbox"], f"{path}.bbox", issues)
            _validate_field_map(
                entry["fields"], ("label", "amount"), f"{path}.fields", issues
            )

    if not isinstance(annotation["notes"], list) or any(
        not isinstance(item, str) for item in annotation["notes"]
    ):
        issues.append(_issue("$.notes", "must be an array of strings"))
    return issues


def _validate_records(
    records: Any,
    collection_name: str,
    required: set[str],
    id_key: str,
    field_names: tuple[str, ...],
    issues: list[dict[str, str]],
    *,
    row: bool = False,
) -> None:
    path = f"$.{collection_name}"
    if not isinstance(records, list):
        issues.append(_issue(path, "must be an array"))
        return
    seen: set[str] = set()
    for index, record in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_keys(record, required, item_path, issues):
            continue
        record_id = record[id_key]
        if not isinstance(record_id, str) or not record_id.strip():
            issues.append(_issue(f"{item_path}.{id_key}", "must not be empty"))
        elif not re.fullmatch(r"[A-Za-z0-9._-]+", record_id):
            issues.append(
                _issue(f"{item_path}.{id_key}", "must use letters, digits, dot, underscore or hyphen")
            )
        elif record_id in seen:
            issues.append(_issue(f"{item_path}.{id_key}", "must be unique"))
        seen.add(record_id)
        if row and (
            not isinstance(record["row_state"], str)
            or record["row_state"] not in {"present", "ambiguous", "unscorable"}
        ):
            issues.append(_issue(f"{item_path}.row_state", "unknown row state"))
        if not _is_int(record["page_index"]) or record["page_index"] < 0:
            issues.append(_issue(f"{item_path}.page_index", "must be >= 0"))
        _validate_box(record["bbox"], f"{item_path}.bbox", issues)
        _validate_field_map(record["fields"], field_names, f"{item_path}.fields", issues)


def verify_document_hash(annotation: dict[str, Any], asset_root: str | Path) -> dict[str, Any]:
    """Hash a referenced file after enforcing that it stays below asset_root."""
    document = annotation.get("document")
    if not isinstance(document, dict):
        return {"ok": False, "error": "document must be an object"}
    relative = document.get("relative_path")
    if not isinstance(relative, str) or not relative:
        return {"ok": False, "error": "document.relative_path is required"}
    root = Path(asset_root).resolve()
    requested = Path(relative)
    if requested.is_absolute():
        return {"ok": False, "error": "absolute document paths are not allowed"}
    try:
        resolved = (root / requested).resolve(strict=True)
    except OSError:
        return {"ok": False, "error": "document file is missing or unreadable"}
    if not resolved.is_relative_to(root):
        return {"ok": False, "error": "document path escapes asset_root"}
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return {"ok": False, "error": "document file is unreadable"}
    expected = document.get("sha256")
    actual = digest.hexdigest()
    return {
        "ok": actual == expected,
        "expected": expected,
        "actual": actual,
        "error": None if actual == expected else "SHA-256 mismatch",
    }


def _resolve_asset_path(asset_root: str | Path, relative_path: str) -> Path:
    root = Path(asset_root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("absolute paths are not allowed")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("path escapes asset_root")
    return resolved


def _field_signature(field: Any) -> Any:
    if not isinstance(field, dict):
        return field
    return {
        key: field.get(key)
        for key in ("state", "visible_text", "normalized_value", "candidates", "derivation")
    }


def _compare_field(
    field_a: Any,
    field_b: Any,
    path: str,
    divergences: list[dict[str, Any]],
    evidence_differences: list[dict[str, Any]],
) -> None:
    if _field_signature(field_a) != _field_signature(field_b):
        divergences.append(
            {"path": path, "a": _field_signature(field_a), "b": _field_signature(field_b)}
        )
    if (
        isinstance(field_a, dict)
        and isinstance(field_b, dict)
        and field_a.get("evidence") != field_b.get("evidence")
    ):
        evidence_differences.append(
            {
                "path": path,
                "a": field_a.get("evidence"),
                "b": field_b.get("evidence"),
            }
        )


def _compare_collection(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
    collection: str,
    id_key: str,
    fields: tuple[str, ...],
    divergences: list[dict[str, Any]],
    evidence_differences: list[dict[str, Any]],
) -> None:
    by_id_a = {item[id_key]: item for item in a}
    by_id_b = {item[id_key]: item for item in b}
    for record_id in sorted(by_id_a.keys() | by_id_b.keys()):
        record_a = by_id_a.get(record_id)
        record_b = by_id_b.get(record_id)
        record_path = f"{collection}[{record_id}]"
        if record_a is None or record_b is None:
            divergences.append(
                {"path": f"{record_path}.$record", "a": record_a, "b": record_b}
            )
            continue
        if record_a.get("bbox") != record_b.get("bbox"):
            evidence_differences.append(
                {
                    "path": f"{record_path}.$bbox",
                    "a": record_a.get("bbox"),
                    "b": record_b.get("bbox"),
                }
            )
        if record_a.get("page_index") != record_b.get("page_index"):
            divergences.append(
                {
                    "path": f"{record_path}.$page_index",
                    "a": record_a.get("page_index"),
                    "b": record_b.get("page_index"),
                }
            )
        for field_name in fields:
            _compare_field(
                record_a.get("fields", {}).get(field_name),
                record_b.get("fields", {}).get(field_name),
                f"{record_path}.{field_name}",
                divergences,
                evidence_differences,
            )
        if collection == "line_items" and record_a.get("row_state") != record_b.get(
            "row_state"
        ):
            divergences.append(
                {
                    "path": f"{record_path}.$row_state",
                    "a": record_a.get("row_state"),
                    "b": record_b.get("row_state"),
                }
            )


def compare_annotations(
    annotation_a: dict[str, Any], annotation_b: dict[str, Any]
) -> dict[str, Any]:
    """Compare two valid annotations without selecting a winner."""
    errors_a = validate_annotation(annotation_a)
    errors_b = validate_annotation(annotation_b)
    if errors_a or errors_b:
        raise ValueError("both annotations must pass validation before comparison")
    doc_a = annotation_a["document"]
    doc_b = annotation_b["document"]
    if (
        doc_a["document_id"] != doc_b["document_id"]
        or doc_a["sha256"] != doc_b["sha256"]
    ):
        raise ValueError("annotations must refer to the same document ID and SHA-256")

    divergences: list[dict[str, Any]] = []
    evidence_differences: list[dict[str, Any]] = []
    for field_name in (*PRODUCTION_INVOICE_FIELDS, "currency"):
        field_a = (
            annotation_a["invoice"].get(field_name)
            if field_name in annotation_a["invoice"]
            else doc_a["currency"]
        )
        field_b = (
            annotation_b["invoice"].get(field_name)
            if field_name in annotation_b["invoice"]
            else doc_b["currency"]
        )
        prefix = "invoice" if field_name != "currency" else "document"
        _compare_field(
            field_a,
            field_b,
            f"{prefix}.{field_name}",
            divergences,
            evidence_differences,
        )

    _compare_collection(
        annotation_a["line_items"],
        annotation_b["line_items"],
        "line_items",
        "row_id",
        PRODUCTION_LINE_FIELDS,
        divergences,
        evidence_differences,
    )
    _compare_collection(
        annotation_a["taxes"],
        annotation_b["taxes"],
        "taxes",
        "tax_id",
        ("label", "rate", "base", "amount"),
        divergences,
        evidence_differences,
    )
    _compare_collection(
        annotation_a["discounts"],
        annotation_b["discounts"],
        "discounts",
        "discount_id",
        ("description", "amount"),
        divergences,
        evidence_differences,
    )
    _compare_collection(
        annotation_a["financial_summary"],
        annotation_b["financial_summary"],
        "financial_summary",
        "entry_id",
        ("label", "amount"),
        divergences,
        evidence_differences,
    )
    for collection in ("taxes", "discounts", "financial_summary"):
        entries_a = {item.get("tax_id", item.get("discount_id", item.get("entry_id"))): item
                     for item in annotation_a[collection]}
        entries_b = {item.get("tax_id", item.get("discount_id", item.get("entry_id"))): item
                     for item in annotation_b[collection]}
        for record_id in entries_a.keys() & entries_b.keys():
            if entries_a[record_id].get("kind") != entries_b[record_id].get("kind"):
                divergences.append(
                    {
                        "path": f"{collection}[{record_id}].$kind",
                        "a": entries_a[record_id].get("kind"),
                        "b": entries_b[record_id].get("kind"),
                    }
                )

    review_paths = sorted(
        {item["path"] for item in divergences}
        | {item["path"] for item in evidence_differences}
    )
    return {
        "document_id": doc_a["document_id"],
        "sha256": doc_a["sha256"],
        "value_divergences": divergences,
        "evidence_differences": evidence_differences,
        "review_paths": review_paths,
        "arithmetic_warnings_a": arithmetic_warnings(annotation_a),
        "arithmetic_warnings_b": arithmetic_warnings(annotation_b),
    }


def arithmetic_warnings(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Report arithmetic discrepancies without editing or rejecting printed values."""
    warnings: list[dict[str, Any]] = []
    for row in annotation.get("line_items", []):
        fields = row.get("fields", {})
        quantity = _present_decimal(fields.get("quantity"))
        price = _present_decimal(fields.get("unit_price"))
        printed_total = _present_decimal(fields.get("total_ht"))
        if quantity is None or price is None or printed_total is None:
            continue
        calculated = (quantity * price).quantize(CENT, rounding=ROUND_HALF_UP)
        difference = abs(calculated - printed_total)
        if difference > CENT:
            warnings.append(
                {
                    "code": "line_product_mismatch",
                    "path": f"line_items[{row['row_id']}].total_ht",
                    "printed": str(printed_total),
                    "calculated": str(calculated),
                    "difference": str(difference),
                    "action": "review_only; never replace the printed value",
                }
            )

    for tax in annotation.get("taxes", []):
        fields = tax.get("fields", {})
        rate = _present_decimal(fields.get("rate"))
        base = _present_decimal(fields.get("base"))
        printed_amount = _present_decimal(fields.get("amount"))
        if rate is None or base is None or printed_amount is None:
            continue
        calculated = (base * rate).quantize(CENT, rounding=ROUND_HALF_UP)
        difference = abs(calculated - printed_amount)
        if difference > CENT:
            warnings.append(
                {
                    "code": "tax_product_mismatch",
                    "path": f"taxes[{tax['tax_id']}].amount",
                    "printed": str(printed_amount),
                    "calculated": str(calculated),
                    "difference": str(difference),
                    "action": "review_only; never replace the printed value",
                }
            )
    return warnings


def _present_decimal(field: Any) -> Decimal | None:
    if not isinstance(field, dict) or field.get("state") != "present":
        return None
    return _decimal(field.get("normalized_value"))


def _get_path_value(annotation: dict[str, Any], path: str) -> Any:
    if path.startswith("invoice."):
        return annotation["invoice"].get(path.split(".", 1)[1])
    if path == "document.currency":
        return annotation["document"]["currency"]
    match = ROW_PATH_RE.fullmatch(path)
    if not match:
        return None
    collection, record_id, field_name = match.groups()
    id_key = {
        "line_items": "row_id",
        "taxes": "tax_id",
        "discounts": "discount_id",
        "financial_summary": "entry_id",
    }[collection]
    record = next(
        (item for item in annotation[collection] if item.get(id_key) == record_id),
        None,
    )
    if record is None:
        return None
    if field_name == "$record":
        return record
    if field_name == "$row_state":
        return record.get("row_state")
    if field_name == "$page_index":
        return record.get("page_index")
    if field_name == "$bbox":
        return record.get("bbox")
    if field_name == "$kind":
        return record.get("kind")
    return record.get("fields", {}).get(field_name)


def _same_selected_value(final_value: Any, source_value: Any) -> bool:
    if isinstance(source_value, dict) and "state" in source_value:
        return _field_signature(final_value) == _field_signature(source_value)
    if isinstance(source_value, dict):
        return final_value == source_value
    return final_value == source_value


def finalize_reference(
    annotation_a: dict[str, Any],
    annotation_b: dict[str, Any],
    adjudication: dict[str, Any],
    *,
    allow_synthetic_test: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build production JSON and a provenance sidecar after explicit arbitration."""
    if not isinstance(adjudication, dict):
        raise ValueError("adjudication must be an object")
    comparison = compare_annotations(annotation_a, annotation_b)
    final_annotation = adjudication.get("final_annotation")
    final_issues = validate_annotation(final_annotation)
    if final_issues:
        raise ValueError("adjudicated final annotation is invalid")

    kinds = (
        annotation_a["annotation_kind"],
        annotation_b["annotation_kind"],
        final_annotation["annotation_kind"],
    )
    real_workflow = kinds == ("independent", "independent", "adjudicated")
    synthetic_workflow = (
        allow_synthetic_test
        and kinds == ("synthetic_test", "synthetic_test", "synthetic_test")
    )
    if not real_workflow and not synthetic_workflow:
        raise ValueError("finalization requires two independent annotations and an adjudication")
    if (
        annotation_a["reviewer"]["reviewer_id"]
        == annotation_b["reviewer"]["reviewer_id"]
    ):
        raise ValueError("the two independent annotations need distinct reviewer IDs")

    document = annotation_a["document"]
    final_document = final_annotation["document"]
    for candidate in (annotation_b["document"], final_document):
        if (
            candidate["document_id"] != document["document_id"]
            or candidate["sha256"] != document["sha256"]
        ):
            raise ValueError("all annotations and adjudication must match document ID and SHA-256")
    if (
        adjudication.get("schema_version") != ARBITRATION_VERSION
        or adjudication.get("document_id") != document["document_id"]
        or adjudication.get("document_sha256") != document["sha256"]
    ):
        raise ValueError("arbitration schema, document ID and SHA-256 must match")

    expected_paths = set(comparison["review_paths"])
    resolutions = adjudication.get("resolutions")
    if not isinstance(resolutions, list):
        raise ValueError("adjudication.resolutions must be an array")
    by_path: dict[str, dict[str, Any]] = {}
    for resolution in resolutions:
        if not isinstance(resolution, dict):
            raise ValueError("each resolution must be an object")
        path = resolution.get("path")
        if not isinstance(path, str) or path in by_path:
            raise ValueError("resolution paths must be unique strings")
        if resolution.get("decision") not in {"agree_a", "agree_b", "new_manual"}:
            raise ValueError(f"invalid resolution decision for {path}")
        if not isinstance(resolution.get("rationale"), str) or not resolution[
            "rationale"
        ].strip():
            raise ValueError(f"resolution rationale required for {path}")
        by_path[path] = resolution
    if set(by_path) != expected_paths:
        missing = sorted(expected_paths - by_path.keys())
        extra = sorted(by_path.keys() - expected_paths)
        raise ValueError(f"resolution paths mismatch; missing={missing}, extra={extra}")

    for path, resolution in by_path.items():
        if resolution["decision"] == "new_manual":
            continue
        source = annotation_a if resolution["decision"] == "agree_a" else annotation_b
        if not _same_selected_value(
            _get_path_value(final_annotation, path), _get_path_value(source, path)
        ):
            raise ValueError(f"final annotation does not match {resolution['decision']} at {path}")

    reviewer = adjudication.get("reviewer")
    if not isinstance(reviewer, dict):
        raise ValueError("adjudication.reviewer is required")
    if (
        reviewer.get("role") != "adjudicator"
        or not isinstance(reviewer.get("reviewer_id"), str)
        or not reviewer["reviewer_id"].strip()
        or reviewer.get("source_image_viewed") is not True
        or reviewer.get("predictions_viewed") is not False
        or reviewer.get("prior_annotations_viewed") is not False
        or reviewer.get("peer_annotations_viewed") is not True
        or reviewer.get("pr64_pr69_viewed") is not False
    ):
        raise ValueError("adjudicator attestations are incomplete or violate blinding")
    if final_annotation["reviewer"]["reviewer_id"] != reviewer["reviewer_id"]:
        raise ValueError("final annotation reviewer must match the adjudicator")
    if not _parse_timestamp(adjudication.get("completed_at")):
        raise ValueError("adjudication.completed_at must be a timestamp with timezone")

    output = build_production_output(final_annotation)
    states = _field_states(final_annotation)
    sidecar = {
        "schema_version": "invoice-reference-metadata-v1",
        "document_id": document["document_id"],
        "source_sha256": document["sha256"],
        "benchmark_eligible": real_workflow,
        "field_states": states,
        "row_states": {
            row["row_id"]: row["row_state"] for row in final_annotation["line_items"]
        },
        "review_paths": comparison["review_paths"],
        "resolutions": resolutions,
        "arithmetic_warnings": arithmetic_warnings(final_annotation),
        "annotators": [annotation_a["reviewer"]["reviewer_id"], annotation_b["reviewer"]["reviewer_id"]],
        "adjudicator": reviewer.get("reviewer_id"),
        "protocol_version": final_annotation["protocol_version"],
    }
    return output, sidecar


def _production_value(field: dict[str, Any], field_name: str) -> Any:
    if field["state"] != "present":
        return None
    value = field["normalized_value"]
    if field_name in DECIMAL_FIELDS:
        decimal_value = Decimal(value)
        if decimal_value == decimal_value.to_integral_value():
            return int(decimal_value)
        return float(decimal_value)
    return value


def build_production_output(annotation: dict[str, Any]) -> dict[str, Any]:
    """Project adjudicated field annotations to the exact production JSON keys."""
    output = {
        name: _production_value(annotation["invoice"][name], name)
        for name in PRODUCTION_INVOICE_FIELDS
    }
    output["line_items"] = []
    for row in annotation["line_items"]:
        if row["row_state"] != "present":
            continue
        output["line_items"].append(
            {
                name: _production_value(row["fields"][name], name)
                for name in PRODUCTION_LINE_FIELDS
            }
        )
    return output


def _field_states(annotation: dict[str, Any]) -> dict[str, str]:
    states = {
        f"invoice.{name}": field["state"]
        for name, field in annotation["invoice"].items()
    }
    states["document.currency"] = annotation["document"]["currency"]["state"]
    for row in annotation["line_items"]:
        for name, field in row["fields"].items():
            states[f"line_items[{row['row_id']}].{name}"] = field["state"]
    for collection, id_key in (
        ("taxes", "tax_id"),
        ("discounts", "discount_id"),
        ("financial_summary", "entry_id"),
    ):
        for record in annotation[collection]:
            for name, field in record["fields"].items():
                states[f"{collection}[{record[id_key]}].{name}"] = field["state"]
    return states


def validate_selection(
    manifest: Any,
    known_manifest: dict[str, Any] | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate holdout eligibility and report missing locale/split counts."""
    diagnostics: list[dict[str, str]] = []
    if not isinstance(manifest, dict):
        return {
            "valid": False,
            "ready": False,
            "status": None,
            "diagnostics": [_issue("$", "must be an object")],
            "missing": {},
        }
    if manifest.get("schema_version") != SELECTION_VERSION:
        diagnostics.append(_issue("$.schema_version", "unsupported selection schema"))
    status = manifest.get("status")
    if not isinstance(status, str) or status not in {
        "not_ready",
        "collecting",
        "frozen",
    }:
        diagnostics.append(_issue("$.status", "must be not_ready, collecting or frozen"))
    plan = manifest.get("planned_splits")
    selected = manifest.get("selected_documents")
    if not isinstance(plan, dict) or set(plan) != set(SPLITS):
        diagnostics.append(_issue("$.planned_splits", "must define all three splits"))
        plan = {}
    if not isinstance(selected, list):
        diagnostics.append(_issue("$.selected_documents", "must be an array"))
        selected = []

    required_document_keys = {
        "document_id",
        "relative_path",
        "sha256",
        "split",
        "locale",
        "country",
        "language",
        "file_type",
        "genuine_invoice_verified",
        "permission_status",
        "prior_use_status",
        "supplier_group_id",
        "layout_family_id",
        "visual_quality_band",
        "edge_cases",
        "annotation_status",
        "annotation_a_path",
        "annotation_a_sha256",
        "annotation_b_path",
        "annotation_b_sha256",
        "adjudication_path",
        "adjudication_sha256",
        "scored_field_count",
        "line_item_count",
        "expected_proposal_count",
    }
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    seen_annotation_hashes: set[str] = set()
    seen_annotation_paths: set[str] = set()
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    split_docs: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    known_ids: set[str] = set()
    known_paths: set[str] = set()
    known_hashes: set[str] = set()
    known_layouts: set[str] = set()
    if isinstance(known_manifest, dict):
        known_documents = known_manifest.get("documents", [])
        if not isinstance(known_documents, list):
            known_documents = []
        for doc in known_documents:
            if not isinstance(doc, dict):
                continue
            for key, target in (
                ("document_id", known_ids),
                ("path", known_paths),
                ("file_sha256", known_hashes),
                ("source_document_id", known_ids),
                ("duplicate_of", known_ids),
                ("layout_category", known_layouts),
            ):
                value = doc.get(key)
                if isinstance(value, str) and value:
                    target.add(value)

    for index, doc in enumerate(selected):
        path = f"$.selected_documents[{index}]"
        if not _check_keys(doc, required_document_keys, path, diagnostics):
            continue
        document_id = doc["document_id"]
        digest = doc["sha256"]
        relative = doc["relative_path"]
        split = doc["split"]
        locale = doc["locale"]
        for key, value in (
            ("document_id", document_id),
            ("relative_path", relative),
            ("supplier_group_id", doc["supplier_group_id"]),
            ("layout_family_id", doc["layout_family_id"]),
        ):
            if not isinstance(value, str) or not value.strip():
                diagnostics.append(_issue(f"{path}.{key}", "must be a non-empty string"))
        valid_id = isinstance(document_id, str) and bool(document_id.strip())
        valid_path = isinstance(relative, str) and bool(relative.strip())
        valid_digest = isinstance(digest, str) and bool(SHA256_RE.fullmatch(digest))
        if not valid_digest:
            diagnostics.append(_issue(f"{path}.sha256", "must be lowercase SHA-256 hex"))
        if valid_path:
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                diagnostics.append(
                    _issue(f"{path}.relative_path", "must stay below the asset root")
                )
        if valid_id and (document_id in seen_ids or document_id in known_ids):
            diagnostics.append(
                _issue(f"{path}.document_id", "duplicate or previously inventoried ID")
            )
        if valid_digest and (digest in seen_hashes or digest in known_hashes):
            diagnostics.append(
                _issue(f"{path}.sha256", "duplicate or previously inventoried content")
            )
        if valid_path and (relative in seen_paths or relative in known_paths):
            diagnostics.append(
                _issue(f"{path}.relative_path", "duplicate or previously inventoried path")
            )
        if valid_id:
            seen_ids.add(document_id)
        if valid_digest:
            seen_hashes.add(digest)
        if valid_path:
            seen_paths.add(relative)
        if asset_root is not None and valid_path and valid_digest:
            hash_result = verify_document_hash(
                {"document": {"relative_path": relative, "sha256": digest}},
                asset_root,
            )
            if not hash_result["ok"]:
                diagnostics.append(
                    _issue(f"{path}.sha256", hash_result["error"] or "hash mismatch")
                )
        if split not in SPLITS:
            diagnostics.append(_issue(f"{path}.split", "unknown split"))
        elif locale not in LOCALES:
            diagnostics.append(_issue(f"{path}.locale", "unsupported locale"))
        else:
            counts[split][locale] += 1
            split_docs[split].append(doc)
        if (
            isinstance(split, str)
            and split in SPLITS
            and isinstance(locale, str)
            and locale in LOCALES
        ):
            expected_country = {
                "fr-FR": "France",
                "en-GB": "United Kingdom",
                "en-US": "United States",
            }[locale]
            if doc["country"] != expected_country:
                diagnostics.append(
                    _issue(f"{path}.country", "does not match the selected locale")
                )
        if doc["genuine_invoice_verified"] is not True:
            diagnostics.append(_issue(f"{path}.genuine_invoice_verified", "must be true"))
        if doc["permission_status"] != "approved":
            diagnostics.append(_issue(f"{path}.permission_status", "permission must be approved"))
        if doc["prior_use_status"] != "none_confirmed":
            diagnostics.append(_issue(f"{path}.prior_use_status", "prior use must be checked and absent"))
        if not isinstance(doc["file_type"], str) or doc["file_type"] not in {
            "pdf",
            "png",
            "jpg",
            "jpeg",
        }:
            diagnostics.append(_issue(f"{path}.file_type", "unsupported file type"))
        if not isinstance(doc["visual_quality_band"], str) or doc[
            "visual_quality_band"
        ] not in {
            "clear",
            "mixed",
            "degraded",
            "unassessed",
        }:
            diagnostics.append(
                _issue(f"{path}.visual_quality_band", "unknown human visual quality band")
            )
        if not isinstance(doc["edge_cases"], list) or any(
            not isinstance(tag, str) for tag in doc["edge_cases"]
        ):
            diagnostics.append(_issue(f"{path}.edge_cases", "must be an array of tags"))
        if not isinstance(doc["annotation_status"], str) or doc[
            "annotation_status"
        ] not in {
            "not_started",
            "double_annotated",
            "adjudicated",
        }:
            diagnostics.append(_issue(f"{path}.annotation_status", "unknown annotation status"))
        annotation_artifacts = (
            ("annotation_a_path", "annotation_a_sha256"),
            ("annotation_b_path", "annotation_b_sha256"),
            ("adjudication_path", "adjudication_sha256"),
        )
        artifact_present: dict[str, bool] = {}
        artifact_paths: dict[str, str] = {}
        verified_artifacts: set[str] = set()
        for path_key, hash_key in annotation_artifacts:
            artifact_path = doc[path_key]
            artifact_hash = doc[hash_key]
            present = artifact_path is not None or artifact_hash is not None
            artifact_present[path_key] = present
            if not present:
                continue
            if not isinstance(artifact_path, str) or not artifact_path.strip():
                diagnostics.append(
                    _issue(f"{path}.{path_key}", "artifact path must be a non-empty relative path")
                )
                continue
            if (
                not isinstance(artifact_hash, str)
                or not SHA256_RE.fullmatch(artifact_hash)
            ):
                diagnostics.append(
                    _issue(f"{path}.{hash_key}", "artifact hash must be lowercase SHA-256 hex")
                )
                continue
            artifact_relative = Path(artifact_path)
            if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
                diagnostics.append(
                    _issue(f"{path}.{path_key}", "artifact path must stay below asset_root")
                )
                continue
            artifact_paths[path_key] = artifact_path
            if artifact_path in seen_annotation_paths:
                diagnostics.append(_issue(f"{path}.{path_key}", "annotation path is reused"))
            if artifact_hash in seen_annotation_hashes:
                diagnostics.append(_issue(f"{path}.{hash_key}", "annotation hash is reused"))
            seen_annotation_paths.add(artifact_path)
            seen_annotation_hashes.add(artifact_hash)
            if asset_root is not None:
                hash_result = verify_document_hash(
                    {"document": {"relative_path": artifact_path, "sha256": artifact_hash}},
                    asset_root,
                )
                if not hash_result["ok"]:
                    diagnostics.append(
                        _issue(
                            f"{path}.{hash_key}",
                            hash_result["error"] or "annotation hash mismatch",
                        )
                    )
            elif hash_result["ok"]:
                    verified_artifacts.add(path_key)

        a_present = artifact_present["annotation_a_path"]
        b_present = artifact_present["annotation_b_path"]
        adjudication_present = artifact_present["adjudication_path"]
        if doc["annotation_status"] == "not_started" and any(
            (a_present, b_present, adjudication_present)
        ):
            diagnostics.append(
                _issue(path, "not_started documents must not have annotation artifacts")
            )
        elif doc["annotation_status"] == "double_annotated" and (
            not a_present or not b_present or adjudication_present
        ):
            diagnostics.append(
                _issue(path, "double_annotated requires A and B files, but no adjudication")
            )
        elif doc["annotation_status"] == "adjudicated" and not all(
            (a_present, b_present, adjudication_present)
        ):
            diagnostics.append(
                _issue(path, "adjudicated requires A, B and adjudication files")
            )
        if (
            asset_root is not None
            and a_present
            and b_present
            and "annotation_a_path" in artifact_paths
            and "annotation_b_path" in artifact_paths
            and {"annotation_a_path", "annotation_b_path"} <= verified_artifacts
            and doc["annotation_status"] in {"double_annotated", "adjudicated"}
        ):
            try:
                annotation_a = _load_json(
                    _resolve_asset_path(asset_root, artifact_paths["annotation_a_path"])
                )
                annotation_b = _load_json(
                    _resolve_asset_path(asset_root, artifact_paths["annotation_b_path"])
                )
                if validate_annotation(annotation_a) or validate_annotation(annotation_b):
                    raise ValueError("A or B annotation fails schema validation")
                if (
                    annotation_a.get("annotation_kind") != "independent"
                    or annotation_b.get("annotation_kind") != "independent"
                ):
                    raise ValueError("A and B files must be independent annotations")
                if (
                    annotation_a.get("document", {}).get("document_id") != document_id
                    or annotation_b.get("document", {}).get("document_id") != document_id
                    or annotation_a.get("document", {}).get("sha256") != digest
                    or annotation_b.get("document", {}).get("sha256") != digest
                ):
                    raise ValueError("annotation document ID/SHA-256 does not match selection")
                if doc["annotation_status"] == "double_annotated":
                    compare_annotations(annotation_a, annotation_b)
                    if (
                        annotation_a["reviewer"]["reviewer_id"]
                        == annotation_b["reviewer"]["reviewer_id"]
                    ):
                        raise ValueError("annotator IDs must be distinct")
                elif (
                    doc["annotation_status"] == "adjudicated"
                    and adjudication_present
                    and "adjudication_path" in artifact_paths
                    and "adjudication_path" in verified_artifacts
                ):
                    adjudication = _load_json(
                        _resolve_asset_path(
                            asset_root, artifact_paths["adjudication_path"]
                        )
                    )
                    finalize_reference(annotation_a, annotation_b, adjudication)
            except (OSError, ValueError, TypeError, KeyError) as error:
                diagnostics.append(_issue(path, f"annotation provenance invalid: {error}"))
        for count_key in (
            "scored_field_count",
            "line_item_count",
            "expected_proposal_count",
        ):
            count_value = doc[count_key]
            if count_value is not None and (
                not _is_int(count_value) or count_value < 0
            ):
                diagnostics.append(
                    _issue(f"{path}.{count_key}", "must be a non-negative integer or null")
                )
            if status == "frozen" and not _is_int(count_value):
                diagnostics.append(
                    _issue(f"{path}.{count_key}", "must be recorded before freezing")
                )
        if (
            isinstance(split, str)
            and split == "final_test"
            and isinstance(doc["layout_family_id"], str)
            and doc["layout_family_id"] in known_layouts
        ):
            diagnostics.append(
                _issue(
                    f"{path}.layout_family_id",
                    "final-test layout overlaps a known historical layout",
                )
            )

    missing: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_plan = plan.get(split, {})
        per_locale = split_plan.get("per_locale", {}) if isinstance(split_plan, dict) else {}
        missing[split] = {}
        for locale in LOCALES:
            target = per_locale.get(locale, 0) if isinstance(per_locale, dict) else 0
            if not _is_int(target) or target < 0:
                diagnostics.append(
                    _issue(
                        f"$.planned_splits.{split}.per_locale.{locale}",
                        "target must be a non-negative integer",
                    )
                )
                target = 0
            count = counts[split][locale]
            missing[split][locale] = max(0, target - count)
            if count > target:
                diagnostics.append(
                    _issue(
                        f"$.planned_splits.{split}.per_locale.{locale}",
                        f"selected {count}, target is {target}",
                    )
                )
            if status == "frozen" and count != target:
                diagnostics.append(
                    _issue(
                        f"$.selected_documents.{split}.{locale}",
                        f"frozen split needs {target}, found {count}",
                    )
                )

    for left_index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[left_index + 1 :]:
            for group_key in ("supplier_group_id", "layout_family_id"):
                left_groups = {
                    doc.get(group_key)
                    for doc in split_docs[left_split]
                    if isinstance(doc.get(group_key), str)
                }
                right_groups = {
                    doc.get(group_key)
                    for doc in split_docs[right_split]
                    if isinstance(doc.get(group_key), str)
                }
                shared = left_groups & right_groups
                if shared:
                    diagnostics.append(
                        _issue(
                            "$.selected_documents",
                            f"{group_key} overlaps {left_split}/{right_split}: "
                            + ", ".join(sorted(shared)),
                        )
                    )

    final_plan = plan.get("final_test", {}) if isinstance(plan, dict) else {}
    if not isinstance(final_plan, dict):
        diagnostics.append(_issue("$.planned_splits.final_test", "must be an object"))
        final_plan = {}
    if status == "frozen":
        if asset_root is None:
            diagnostics.append(
                _issue("$", "frozen selection requires --asset-root hash/content verification")
            )
        final_docs = split_docs["final_test"]
        layout_ids = {
            doc.get("layout_family_id")
            for doc in final_docs
            if isinstance(doc.get("layout_family_id"), str)
        }
        supplier_ids: dict[str, set[str]] = {locale: set() for locale in LOCALES}
        for doc in final_docs:
            locale = doc.get("locale")
            supplier_id = doc.get("supplier_group_id")
            if (
                isinstance(locale, str)
                and locale in supplier_ids
                and isinstance(supplier_id, str)
            ):
                supplier_ids[locale].add(supplier_id)
        min_layouts = final_plan.get("minimum_distinct_layout_families", 0)
        if not _is_int(min_layouts) or min_layouts < 0:
            diagnostics.append(
                _issue(
                    "$.planned_splits.final_test.minimum_distinct_layout_families",
                    "must be a non-negative integer",
                )
            )
        elif len(layout_ids) < min_layouts:
            diagnostics.append(
                _issue(
                    "$.selected_documents.final_test",
                    f"requires {min_layouts} distinct layout families, found {len(layout_ids)}",
                )
            )
        min_suppliers = final_plan.get("minimum_supplier_families_per_locale", 0)
        if not _is_int(min_suppliers) or min_suppliers < 0:
            diagnostics.append(
                _issue(
                    "$.planned_splits.final_test.minimum_supplier_families_per_locale",
                    "must be a non-negative integer",
                )
            )
        else:
            for locale in LOCALES:
                if len(supplier_ids[locale]) < min_suppliers:
                    diagnostics.append(
                        _issue(
                            f"$.selected_documents.final_test.{locale}",
                            f"requires {min_suppliers} supplier families",
                        )
                    )
        for check in (
            "minimum_scored_fields",
            "minimum_line_items",
            "minimum_expected_proposals",
        ):
            if not _is_int(final_plan.get(check)) or final_plan[check] <= 0:
                diagnostics.append(
                    _issue(f"$.planned_splits.final_test.{check}", "must be a positive integer")
                )
        actuals = {
            "minimum_scored_fields": sum(
                doc["scored_field_count"]
                for doc in final_docs
                if _is_int(doc.get("scored_field_count"))
            ),
            "minimum_line_items": sum(
                doc["line_item_count"]
                for doc in final_docs
                if _is_int(doc.get("line_item_count"))
            ),
            "minimum_expected_proposals": sum(
                doc["expected_proposal_count"]
                for doc in final_docs
                if _is_int(doc.get("expected_proposal_count"))
            ),
        }
        for key, actual in actuals.items():
            target = final_plan.get(key)
            if isinstance(target, int) and actual < target:
                diagnostics.append(
                    _issue(
                        f"$.selected_documents.final_test.{key}",
                        f"requires at least {target}, found {actual}",
                    )
                )
        for doc in selected:
            if not isinstance(doc, dict):
                continue
            if doc.get("annotation_status") != "adjudicated":
                diagnostics.append(
                    _issue(
                        f"$.selected_documents[{doc.get('document_id', '?')}].annotation_status",
                        "all frozen documents need adjudicated references",
                    )
                )
        edge_targets = manifest.get("edge_case_targets_for_final_test", {})
        if not isinstance(edge_targets, dict):
            diagnostics.append(
                _issue("$.edge_case_targets_for_final_test", "must be an object")
            )
        else:
            for edge_case, target in edge_targets.items():
                actual = sum(
                    edge_case in doc.get("edge_cases", []) for doc in final_docs
                )
                if not _is_int(target) or target < 0:
                    diagnostics.append(
                        _issue(
                            f"$.edge_case_targets_for_final_test.{edge_case}",
                            "target must be a non-negative integer",
                        )
                    )
                elif actual < target:
                    diagnostics.append(
                        _issue(
                            f"$.selected_documents.final_test.{edge_case}",
                            f"requires {target} examples, found {actual}",
                        )
                    )

    if "missing_documents" in manifest and manifest["missing_documents"] != missing:
        diagnostics.append(
            _issue("$.missing_documents", "stored missing counts are stale; regenerate them")
        )

    ready = status == "frozen" and not diagnostics and not any(
        missing[split][locale] for split in SPLITS for locale in LOCALES
    )
    return {
        "valid": not any(item["severity"] == "error" for item in diagnostics),
        "ready": ready,
        "status": status,
        "selected_document_count": len(selected),
        "missing": missing,
        "diagnostics": diagnostics,
    }


def _load_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source, parse_constant=reject_constant)


def _write_new_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate one annotation")
    validate.add_argument("annotation")
    validate.add_argument("--asset-root")

    compare = commands.add_parser("compare", help="compare two blind annotations")
    compare.add_argument("annotation_a")
    compare.add_argument("annotation_b")
    compare.add_argument("--asset-root")

    selection = commands.add_parser("validate-selection", help="check the cohort manifest")
    selection.add_argument("manifest")
    selection.add_argument("--known-manifest")
    selection.add_argument("--asset-root")

    finalize = commands.add_parser("finalize", help="export adjudicated reference and sidecar")
    finalize.add_argument("--a", required=True)
    finalize.add_argument("--b", required=True)
    finalize.add_argument("--adjudication", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--metadata-output", required=True)
    finalize.add_argument("--asset-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            annotation = _load_json(args.annotation)
            issues = validate_annotation(annotation)
            hash_result = (
                verify_document_hash(annotation, args.asset_root)
                if args.asset_root
                else None
            )
            if hash_result is not None and not hash_result["ok"]:
                issues.append(_issue("$.document.sha256", hash_result["error"]))
            _emit(
                {
                    "valid": not issues,
                    "diagnostics": issues,
                    "hash": hash_result,
                    "arithmetic_warnings": arithmetic_warnings(annotation)
                    if not issues
                    else [],
                }
            )
            return 0 if not issues else 1
        if args.command == "compare":
            annotation_a = _load_json(args.annotation_a)
            annotation_b = _load_json(args.annotation_b)
            issues = validate_annotation(annotation_a) + validate_annotation(annotation_b)
            if issues:
                _emit({"valid": False, "diagnostics": issues})
                return 1
            hashes = None
            if args.asset_root:
                hashes = [
                    verify_document_hash(annotation_a, args.asset_root),
                    verify_document_hash(annotation_b, args.asset_root),
                ]
                if not all(item["ok"] for item in hashes):
                    _emit({"valid": False, "hashes": hashes})
                    return 1
            result = compare_annotations(annotation_a, annotation_b)
            result.update(valid=True, hashes=hashes)
            _emit(result)
            return 0
        if args.command == "validate-selection":
            known = _load_json(args.known_manifest) if args.known_manifest else None
            result = validate_selection(_load_json(args.manifest), known, args.asset_root)
            _emit(result)
            return 0 if result["valid"] else 1
        if args.command == "finalize":
            annotation_a = _load_json(args.a)
            annotation_b = _load_json(args.b)
            adjudication = _load_json(args.adjudication)
            if args.asset_root:
                hashes = [
                    verify_document_hash(annotation_a, args.asset_root),
                    verify_document_hash(annotation_b, args.asset_root),
                    verify_document_hash(adjudication.get("final_annotation", {}), args.asset_root),
                ]
                if not all(item["ok"] for item in hashes):
                    _emit({"finalized": False, "hashes": hashes})
                    return 1
            output, metadata = finalize_reference(annotation_a, annotation_b, adjudication)
            _write_new_json(args.output, output)
            try:
                _write_new_json(args.metadata_output, metadata)
            except Exception:
                Path(args.output).unlink(missing_ok=True)
                raise
            _emit(
                {
                    "finalized": True,
                    "output": str(args.output),
                    "metadata_output": str(args.metadata_output),
                    "benchmark_eligible": metadata["benchmark_eligible"],
                }
            )
            return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        _emit({"valid": False, "error": str(error)})
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
