"""Score recorded invoice-extraction outputs without loading models or making network calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "invoice-benchmark-results-1"
TOP_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")
LINE_FIELDS = ("description", "quantity", "unit_price", "total_ht")
FIELD_STATES = {"present", "absent", "derived_only", "ambiguous", "unscorable"}
SCORABLE_NULL_STATES = {"absent", "derived_only"}
VARIANTS = {"ocr_only", "base_qwen", "qwen_lora", "production_grounded"}
MONEY_FIELDS = {"total_ht", "unit_price"}
MONEY_QUANTUM = Decimal("0.01")
QUANTITY_TOLERANCE = Decimal("0.000001")
ROW_MATCH_THRESHOLD = 0.88


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _finite_json_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _text(value: Any, *, substantive: bool) -> str:
    text = " ".join(unicodedata.normalize("NFC", str(value)).split())
    if substantive:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        text = text.casefold()
    return text


def _row_match_text(value: Any) -> str:
    text = _text(value, substantive=True)
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _equal(field: str, expected: Any, actual: Any, *, substantive: bool = False) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if field == "invoice_date":
        try:
            return date.fromisoformat(str(expected)).isoformat() == date.fromisoformat(str(actual)).isoformat()
        except ValueError:
            return False
    if field in MONEY_FIELDS or field == "total_ttc":
        left, right = _decimal(expected), _decimal(actual)
        if left is None or right is None:
            return False
        try:
            return left.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP) == right.quantize(
                MONEY_QUANTUM, rounding=ROUND_HALF_UP,
            )
        except InvalidOperation:
            return left == right
    if field == "quantity":
        left, right = _decimal(expected), _decimal(actual)
        return left is not None and right is not None and abs(left - right) <= QUANTITY_TOLERANCE
    return _text(expected, substantive=substantive) == _text(actual, substantive=substantive)


def validate_output(output: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return None, ["output is not valid JSON"]
    if not isinstance(output, dict):
        return None, ["output must be a JSON object"]

    errors = []
    expected_top = set(TOP_FIELDS) | {"line_items"}
    if set(output) != expected_top:
        missing = sorted(expected_top - set(output))
        extra = sorted(set(output) - expected_top)
        if missing:
            errors.append(f"missing top-level keys: {', '.join(missing)}")
        if extra:
            errors.append(f"unexpected top-level keys: {', '.join(extra)}")

    supplier = output.get("supplier_name")
    if supplier is not None and (not isinstance(supplier, str) or not supplier.strip()):
        errors.append("supplier_name must be a non-empty string or null")
    invoice_date = output.get("invoice_date")
    if invoice_date is not None:
        try:
            if not isinstance(invoice_date, str) or date.fromisoformat(invoice_date).isoformat() != invoice_date:
                errors.append("invoice_date must be an ISO date string or null")
        except ValueError:
            errors.append("invoice_date must be an ISO date string or null")
    for field in ("total_ht", "total_ttc"):
        value = output.get(field)
        if value is not None and not _finite_json_number(value):
            errors.append(f"{field} must be a finite JSON number or null")

    rows = output.get("line_items")
    if not isinstance(rows, list):
        errors.append("line_items must be a JSON array")
    else:
        expected_line = set(LINE_FIELDS)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"line_items[{index}] must be a JSON object")
                continue
            if set(row) != expected_line:
                missing = sorted(expected_line - set(row))
                extra = sorted(set(row) - expected_line)
                if missing:
                    errors.append(f"line_items[{index}] missing keys: {', '.join(missing)}")
                if extra:
                    errors.append(f"line_items[{index}] unexpected keys: {', '.join(extra)}")
            description = row.get("description")
            if description is not None and (
                not isinstance(description, str) or not description.strip()
            ):
                errors.append(f"line_items[{index}].description must be a non-empty string or null")
            for field in ("quantity", "unit_price", "total_ht"):
                value = row.get(field)
                if value is not None and not _finite_json_number(value):
                    errors.append(f"line_items[{index}].{field} must be a finite JSON number or null")
    return (output, []) if not errors else (None, errors)


def _validate_field_truth(field_truth: Any, path: str) -> tuple[str, Any]:
    if not isinstance(field_truth, dict):
        raise ValueError(f"{path}: field annotation must be an object")
    state = field_truth.get("state")
    if state not in FIELD_STATES:
        raise ValueError(f"{path}: unsupported annotation state")
    value = field_truth.get("value")
    if state == "present" and value is None:
        raise ValueError(f"{path}: present annotations require a value")
    if state in SCORABLE_NULL_STATES and value is not None:
        raise ValueError(f"{path}: {state} annotations must not carry a printed value")
    if state in {"ambiguous", "unscorable"} and value is not None:
        raise ValueError(f"{path}: {state} annotations cannot select one value")
    return state, value


def _metric_counts() -> dict[str, int]:
    return {
        "scorable_fields": 0,
        "expected_present": 0,
        "expected_null": 0,
        "strict_correct": 0,
        "substantive_correct": 0,
        "correct_substantive_only": 0,
        "false_null": 0,
        "correct_null": 0,
        "non_null_predictions": 0,
        "incorrect_non_null": 0,
        "unscored_fields": 0,
        "unscored_non_null_predictions": 0,
        "missing_output_fields": 0,
        "extra_row_non_null_predictions": 0,
    }


def _record_field(
    counts: dict[str, int],
    field: str,
    state: str,
    expected: Any,
    actual: Any,
    *,
    output_valid: bool,
    extra_row: bool = False,
) -> None:
    if state in {"ambiguous", "unscorable"}:
        counts["unscored_fields"] += 1
        if actual is not None:
            counts["unscored_non_null_predictions"] += 1
        return

    counts["scorable_fields"] += 1
    if state == "present":
        counts["expected_present"] += 1
        if not output_valid or actual is None:
            counts["false_null"] += 1
            if not output_valid:
                counts["missing_output_fields"] += 1
            return
        counts["non_null_predictions"] += 1
        strict_match = _equal(field, expected, actual)
        substantive_match = _equal(field, expected, actual, substantive=True)
        if strict_match:
            counts["strict_correct"] += 1
        if substantive_match:
            counts["substantive_correct"] += 1
        if substantive_match and not strict_match:
            counts["correct_substantive_only"] += 1
        if not strict_match:
            counts["incorrect_non_null"] += 1
    elif state in SCORABLE_NULL_STATES:
        counts["expected_null"] += 1
        if not output_valid:
            counts["missing_output_fields"] += 1
        elif actual is None:
            counts["correct_null"] += 1
        else:
            counts["non_null_predictions"] += 1
            counts["incorrect_non_null"] += 1
    if extra_row and actual is not None:
        counts["extra_row_non_null_predictions"] += 1


def _field_metrics(counts: dict[str, int]) -> dict[str, Any]:
    scorable = counts["scorable_fields"]
    present = counts["expected_present"]
    non_null = counts["non_null_predictions"]
    null_outputs = counts["correct_null"] + counts["false_null"]
    return {
        **counts,
        "accepted_value_precision": (
            counts["strict_correct"] / non_null if non_null else None
        ),
        "field_recall": counts["strict_correct"] / present if present else None,
        "exact_field_accuracy": counts["strict_correct"] / scorable if scorable else None,
        "correct_null_rate": counts["correct_null"] / null_outputs if null_outputs else None,
        "false_null_rate": counts["false_null"] / present if present else None,
        "incorrect_non_null_acceptance_rate": (
            counts["incorrect_non_null"] / non_null if non_null else None
        ),
        "substantive_accuracy": (
            counts["substantive_correct"] / scorable if scorable else None
        ),
    }


def align_rows(
    gold_rows: list[dict[str, Any]], predicted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    gold_text = []
    unscorable_gold_rows = []
    predicted_text = []
    for index, row in enumerate(gold_rows):
        state, value = _validate_field_truth(row.get("description"), "line_items.description")
        gold_text.append(_row_match_text(value) if state == "present" else "")
        if state != "present":
            unscorable_gold_rows.append(index)
    for row in predicted_rows:
        value = row.get("description")
        predicted_text.append(_row_match_text(value) if isinstance(value, str) else "")

    for gold_index, expected in enumerate(gold_text):
        if not expected:
            continue
        for predicted_index, actual in enumerate(predicted_text):
            if not actual:
                continue
            similarity = SequenceMatcher(None, expected, actual).ratio()
            if similarity >= ROW_MATCH_THRESHOLD:
                candidates.append((similarity, gold_index, predicted_index))

    best_for_gold: dict[int, float] = {}
    best_for_prediction: dict[int, float] = {}
    for similarity, gold_index, predicted_index in candidates:
        best_for_gold[gold_index] = max(best_for_gold.get(gold_index, 0.0), similarity)
        best_for_prediction[predicted_index] = max(
            best_for_prediction.get(predicted_index, 0.0), similarity,
        )
    tied_gold = set()
    tied_predictions = set()
    for gold_index, best in best_for_gold.items():
        if sum(score == best and index == gold_index for score, index, _ in candidates) > 1:
            tied_gold.add(gold_index)
    for predicted_index, best in best_for_prediction.items():
        if sum(score == best and index == predicted_index for score, _, index in candidates) > 1:
            tied_predictions.add(predicted_index)

    pairs = []
    used_gold = set()
    used_predictions = set()
    for similarity, gold_index, predicted_index in sorted(candidates, reverse=True):
        if (
            gold_index in tied_gold or predicted_index in tied_predictions
            or gold_index in used_gold or predicted_index in used_predictions
        ):
            continue
        pairs.append((gold_index, predicted_index, similarity))
        used_gold.add(gold_index)
        used_predictions.add(predicted_index)
    pairs.sort()
    missing_gold_rows = sorted(set(range(len(gold_rows))) - used_gold - set(unscorable_gold_rows))
    unmatched_predictions = sorted(set(range(len(predicted_rows))) - used_predictions)
    unresolved_predictions = unmatched_predictions if unscorable_gold_rows else []
    return {
        "pairs": pairs,
        "ambiguous_gold_rows": sorted(tied_gold),
        "ambiguous_predicted_rows": sorted(tied_predictions),
        "unscorable_gold_rows": unscorable_gold_rows,
        "missing_gold_rows": missing_gold_rows,
        "extra_predicted_rows": [] if unscorable_gold_rows else unmatched_predictions,
        "unscored_predicted_rows": unresolved_predictions,
        "gold_rows": len(gold_rows),
        "predicted_rows": len(predicted_rows),
        "row_count_match": len(gold_rows) == len(predicted_rows),
        "order_preserved": (
            len(gold_rows) == len(predicted_rows) == len(pairs)
            and [predicted for _, predicted, _ in pairs]
            == sorted(predicted for _, predicted, _ in pairs)
        ),
    }


def _truth_complete(truth: dict[str, Any]) -> bool:
    if truth.get("status") != "complete":
        return False
    top = truth.get("top_level")
    rows = truth.get("line_items")
    if not isinstance(top, dict) or set(top) != set(TOP_FIELDS) or not isinstance(rows, list):
        return False
    all_fields = list(top.items())
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(LINE_FIELDS):
            return False
        all_fields.extend((f"line_items[{index}].{key}", row[key]) for key in LINE_FIELDS)
    for path, annotation in all_fields:
        state, _ = _validate_field_truth(annotation, path)
        if state not in {"present", *SCORABLE_NULL_STATES}:
            return False
    return True


def score_document(
    document: dict[str, Any],
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    truth = document.get("ground_truth")
    if not isinstance(truth, dict) or truth.get("status") not in {"complete", "partial"}:
        raise ValueError(f"{document.get('document_id')}: selected document has no scorable ground truth")
    top_truth = truth.get("top_level")
    gold_rows = truth.get("line_items")
    if not isinstance(top_truth, dict) or not isinstance(gold_rows, list):
        raise ValueError(f"{document.get('document_id')}: malformed ground truth")

    metadata = response or {}
    parsed_output = None
    schema_errors = []
    output_valid = False
    if response is not None and response.get("inference_status") == "succeeded":
        parsed_output, schema_errors = validate_output(response.get("output"))
        output_valid = parsed_output is not None
    elif response is not None and response.get("output") is not None:
        parsed_output, schema_errors = validate_output(response["output"])
        output_valid = parsed_output is not None

    top_predictions = parsed_output or {}
    line_metric_fields = {field: f"line_items.{field}" for field in LINE_FIELDS}
    field_counts = {
        field: _metric_counts()
        for field in (*TOP_FIELDS, *line_metric_fields.values())
    }
    for field in TOP_FIELDS:
        state, expected = _validate_field_truth(top_truth.get(field), field)
        _record_field(
            field_counts[field], field, state, expected, top_predictions.get(field),
            output_valid=output_valid,
        )

    predicted_rows = (parsed_output or {}).get("line_items", [])
    if not isinstance(predicted_rows, list):
        predicted_rows = []
    alignment = align_rows(gold_rows, predicted_rows) if output_valid else {
        "pairs": [],
        "ambiguous_gold_rows": [],
        "ambiguous_predicted_rows": [],
        "unscorable_gold_rows": [],
        "missing_gold_rows": list(range(len(gold_rows))),
        "extra_predicted_rows": [],
        "unscored_predicted_rows": [],
        "gold_rows": len(gold_rows),
        "predicted_rows": 0,
        "row_count_match": False,
        "order_preserved": False,
    }
    gold_to_prediction = {gold: predicted for gold, predicted, _ in alignment["pairs"]}
    for gold_index, row in enumerate(gold_rows):
        predicted_index = gold_to_prediction.get(gold_index)
        predicted = predicted_rows[predicted_index] if predicted_index is not None else {}
        description_state, _ = _validate_field_truth(
            row.get("description"), f"line_items[{gold_index}].description",
        )
        if description_state != "present":
            for field in LINE_FIELDS:
                state, expected = _validate_field_truth(
                    row.get(field), f"line_items[{gold_index}].{field}",
                )
                field_counts[line_metric_fields[field]]["unscored_fields"] += 1
                if (
                    output_valid and predicted_index is not None
                    and predicted.get(field) is not None
                ):
                    field_counts[line_metric_fields[field]]["unscored_non_null_predictions"] += 1
            continue
        for field in LINE_FIELDS:
            state, expected = _validate_field_truth(
                row.get(field), f"line_items[{gold_index}].{field}",
            )
            _record_field(
                field_counts[line_metric_fields[field]], field, state, expected, predicted.get(field),
                output_valid=output_valid,
            )
    for predicted_index in alignment["extra_predicted_rows"]:
        row = predicted_rows[predicted_index]
        for field in LINE_FIELDS:
            counts = field_counts[line_metric_fields[field]]
            if row.get(field) is not None:
                counts["scorable_fields"] += 1
                counts["non_null_predictions"] += 1
                counts["incorrect_non_null"] += 1
                counts["extra_row_non_null_predictions"] += 1
    for predicted_index in alignment["unscored_predicted_rows"]:
        row = predicted_rows[predicted_index]
        for field in LINE_FIELDS:
            if row.get(field) is not None:
                field_counts[line_metric_fields[field]]["unscored_non_null_predictions"] += 1

    all_counts = _metric_counts()
    for counts in field_counts.values():
        for key, value in counts.items():
            all_counts[key] += value
    metrics = _field_metrics(all_counts)
    per_field = {field: _field_metrics(counts) for field, counts in field_counts.items()}

    expected_present = all_counts["expected_present"]
    correctly_extracted = all_counts["strict_correct"]
    coverage = correctly_extracted / expected_present if expected_present else None
    complete_truth = _truth_complete(truth)
    exact = bool(
        output_valid and complete_truth
        and all_counts["strict_correct"] + all_counts["correct_null"] == all_counts["scorable_fields"]
        and alignment["row_count_match"] and alignment["order_preserved"]
    )
    return {
        "document_id": document["document_id"],
        "response_present": response is not None,
        "inference_status": metadata.get("inference_status", "not_run"),
        "model_used": bool(metadata.get("model_used", False)),
        "ocr_fallback": bool(metadata.get("ocr_fallback", False)),
        "execution_mode": metadata.get("execution_mode", "missing"),
        "software_revision": metadata.get("software_revision"),
        "model_revision": metadata.get("model_revision"),
        "adapter_revision": metadata.get("adapter_revision"),
        "schema_valid": output_valid,
        "schema_errors": schema_errors,
        "field_metrics": per_field,
        "metrics": metrics,
        "document_coverage": coverage,
        "has_any_correct_value": correctly_extracted > 0,
        "complete_truth": complete_truth,
        "complete_invoice_correct": exact,
        "line_items": {
            "gold_rows": alignment["gold_rows"],
            "predicted_rows": alignment["predicted_rows"],
            "matched_rows": len(alignment["pairs"]),
            "alignment_accuracy": (
                len(alignment["pairs"]) / max(alignment["gold_rows"], alignment["predicted_rows"])
                if max(alignment["gold_rows"], alignment["predicted_rows"]) else 1.0
            ),
            "missing_rows": alignment["missing_gold_rows"],
            "extra_rows": alignment["extra_predicted_rows"],
            "unscored_predicted_rows": alignment["unscored_predicted_rows"],
            "ambiguous_matches": (
                alignment["ambiguous_gold_rows"] or alignment["ambiguous_predicted_rows"]
            ),
            "row_count_match": alignment["row_count_match"],
            "order_preserved": alignment["order_preserved"],
        },
        "latency": response.get("latency", {}) if response else {},
        "input_sha256": response.get("input_sha256") if response else None,
        "ocr_evidence_sha256": response.get("ocr_evidence_sha256") if response else None,
        "prompt_sha256": response.get("prompt_sha256") if response else None,
        "generation_settings_sha256": (
            response.get("generation_settings_sha256") if response else None
        ),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _latency_summary(values: list[Any]) -> dict[str, Any]:
    valid = []
    for value in values:
        if not _finite_json_number(value) or value < 0:
            continue
        valid.append(float(value))
    valid.sort()
    if not valid:
        return {"count": 0, "median_seconds": None, "p95_seconds": None}
    rank = max(0, math.ceil(0.95 * len(valid)) - 1)
    middle = len(valid) // 2
    median = valid[middle] if len(valid) % 2 else (valid[middle - 1] + valid[middle]) / 2
    return {
        "count": len(valid),
        "median_seconds": median,
        "p95_seconds": valid[rank],
    }


def _paired_comparison(
    variant_names: list[str],
    document_ids: list[str],
    per_variant_documents: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    compared = [name for name in ("base_qwen", "qwen_lora") if name in variant_names]
    if len(compared) < 2:
        return {"status": "not_proven", "reason": "base_qwen and qwen_lora outputs are both required"}
    records = {
        name: {row["document_id"]: row for row in per_variant_documents[name]}
        for name in compared
    }
    required_hashes = (
        "input_sha256", "ocr_evidence_sha256", "prompt_sha256",
        "generation_settings_sha256", "software_revision",
    )
    problems = []
    for document_id in document_ids:
        left = records[compared[0]].get(document_id)
        right = records[compared[1]].get(document_id)
        if not left or not right or not left["response_present"] or not right["response_present"]:
            problems.append(f"{document_id}: missing paired response")
            continue
        if left["inference_status"] != "succeeded" or right["inference_status"] != "succeeded":
            problems.append(f"{document_id}: inference did not succeed in both variants")
        if not left["model_used"] or not right["model_used"]:
            problems.append(f"{document_id}: model was not used in both variants")
        if left.get("execution_mode") != right.get("execution_mode"):
            problems.append(f"{document_id}: mismatched execution_mode")
        if left.get("execution_mode") == "simulated" or right.get("execution_mode") == "simulated":
            problems.append(f"{document_id}: simulated output cannot establish a model comparison")
        for key in required_hashes:
            if not left.get(key) or not right.get(key):
                problems.append(f"{document_id}: missing {key}")
            elif left[key] != right[key]:
                problems.append(f"{document_id}: mismatched {key}")
        model_revision = left.get("model_revision")
        adapter_revision = right.get("adapter_revision")
        right_model_revision = right.get("model_revision")
        if not model_revision or not adapter_revision:
            problems.append(f"{document_id}: immutable base and adapter revisions are required")
        elif (
            not isinstance(model_revision, str)
            or not isinstance(right_model_revision, str)
            or not isinstance(adapter_revision, str)
            or not re.fullmatch(r"[a-fA-F0-9]{40}", model_revision)
            or not re.fullmatch(r"[a-fA-F0-9]{40}", right_model_revision)
            or not re.fullmatch(r"[a-fA-F0-9]{40}", adapter_revision)
        ):
            problems.append(f"{document_id}: model revisions must be immutable 40-character commit IDs")
        elif model_revision != right_model_revision:
            problems.append(f"{document_id}: base model revision differs between variants")
    return {
        "status": "paired" if not problems else "not_proven",
        "variants": compared,
        "problems": problems,
    }


def evaluate(
    manifest: dict[str, Any],
    responses: list[dict[str, Any]],
    set_name: str,
    *,
    root: Path,
) -> dict[str, Any]:
    documents = manifest.get("documents")
    evaluation_sets = manifest.get("evaluation_sets")
    if not isinstance(documents, list) or not isinstance(evaluation_sets, dict):
        raise ValueError("manifest must contain documents and evaluation_sets")
    document_map = {}
    for document in documents:
        if not isinstance(document, dict) or not document.get("document_id"):
            raise ValueError("every document requires a document_id")
        document_id = document["document_id"]
        if document_id in document_map:
            raise ValueError(f"duplicate document_id: {document_id}")
        document_map[document_id] = document
    selected_set = evaluation_sets.get(set_name)
    if not isinstance(selected_set, dict):
        raise ValueError(f"unknown evaluation set: {set_name}")
    document_ids = selected_set.get("document_ids")
    if (
        not isinstance(document_ids, list)
        or any(not isinstance(document_id, str) or not document_id for document_id in document_ids)
        or len(document_ids) != len(set(document_ids))
    ):
        raise ValueError(f"{set_name}: document_ids must be a unique list")
    expected_variants = selected_set.get("systems", [])
    if not isinstance(expected_variants, list) or any(
        not isinstance(variant, str) or variant not in VARIANTS
        for variant in expected_variants
    ):
        raise ValueError(f"{set_name}: systems must be a list of supported variants")
    selected_documents = []
    for document_id in document_ids:
        if document_id not in document_map:
            raise ValueError(f"{set_name}: unknown document_id {document_id}")
        document = document_map[document_id]
        if not (document.get("ground_truth") or {}).get("scoreable"):
            raise ValueError(f"{document_id}: ground truth is not marked scoreable")
        relative = Path(document.get("path", ""))
        if not relative.parts or relative.is_absolute():
            raise ValueError(f"{document_id}: path must be relative to the benchmark root")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"{document_id}: path escapes the benchmark root") from exc
        if not path.is_file():
            raise ValueError(f"{document_id}: source file is missing")
        expected_hash = document.get("file_sha256")
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            raise ValueError(f"{document_id}: source file SHA-256 mismatch")
        ground_truth = document.get("ground_truth") or {}
        annotation_hash = ground_truth.get("annotation_sha256")
        if annotation_hash:
            annotation = {
                key: ground_truth[key] for key in ("status", "top_level", "line_items")
                if key in ground_truth
            }
            if canonical_sha256(annotation) != annotation_hash:
                raise ValueError(f"{document_id}: ground-truth annotation SHA-256 mismatch")
        source_file = ground_truth.get("source_file")
        source_hash = ground_truth.get("source_sha256")
        if source_file or source_hash:
            if not isinstance(source_file, str) or not isinstance(source_hash, str):
                raise ValueError(f"{document_id}: ground-truth source path and hash must be paired")
            source_relative = Path(source_file)
            if source_relative.is_absolute():
                raise ValueError(f"{document_id}: ground-truth source path must be relative")
            source_path = (root / source_relative).resolve()
            try:
                source_path.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"{document_id}: ground-truth source path escapes the root") from exc
            if not source_path.is_file() or file_sha256(source_path) != source_hash:
                raise ValueError(f"{document_id}: ground-truth source SHA-256 mismatch")
        if document.get("duplicate_of") and not selected_set.get("allow_related_documents", False):
            raise ValueError(f"{document_id}: related document is not eligible in this evaluation set")
        selected_documents.append(document)

    seen_hashes = {}
    if not selected_set.get("allow_exact_duplicates", False):
        for document in selected_documents:
            digest = document["file_sha256"]
            previous = seen_hashes.get(digest)
            if previous:
                raise ValueError(
                    f"exact duplicate documents in {set_name}: {previous} and {document['document_id']}",
                )
            seen_hashes[digest] = document["document_id"]

    response_map = {}
    if not isinstance(responses, list):
        raise ValueError("responses must be a list of JSON objects")
    for response in responses:
        if not isinstance(response, dict):
            raise ValueError("each response must be a JSON object")
        document_id = response.get("document_id")
        variant = response.get("model_variant")
        if not isinstance(document_id, str) or document_id not in document_ids:
            raise ValueError(f"response references a document outside {set_name}: {document_id}")
        if not isinstance(variant, str) or variant not in VARIANTS:
            raise ValueError(f"unsupported model_variant: {variant}")
        key = (document_id, variant)
        if key in response_map:
            raise ValueError(f"duplicate response for {document_id}/{variant}")
        response_map[key] = response
        expected_hash = document_map[document_id]["file_sha256"]
        if response.get("input_sha256") != expected_hash:
            raise ValueError(f"{document_id}/{variant}: input SHA-256 mismatch")
        if response.get("execution_mode") not in {"recorded", "simulated", "live"}:
            raise ValueError(f"{document_id}/{variant}: execution_mode must be recorded, simulated, or live")
        if response.get("inference_status") not in {"succeeded", "failed", "not_run"}:
            raise ValueError(f"{document_id}/{variant}: invalid inference_status")
        if not isinstance(response.get("model_used"), bool):
            raise ValueError(f"{document_id}/{variant}: model_used must be boolean")
        if not isinstance(response.get("ocr_fallback"), bool):
            raise ValueError(f"{document_id}/{variant}: ocr_fallback must be boolean")
        if not isinstance(response.get("software_revision"), str) or not response["software_revision"].strip():
            raise ValueError(f"{document_id}/{variant}: software_revision is required")
        if variant == "ocr_only" and response["model_used"]:
            raise ValueError(f"{document_id}/{variant}: OCR-only runs cannot use a model")
        latency = response.get("latency", {})
        if latency is not None and not isinstance(latency, dict):
            raise ValueError(f"{document_id}/{variant}: latency must be an object")
        if isinstance(latency, dict):
            for name, value in latency.items():
                if value is not None and (not _finite_json_number(value) or value < 0):
                    raise ValueError(f"{document_id}/{variant}: latency.{name} must be non-negative or null")

    actual_variants = [response["model_variant"] for response in responses]
    variant_names = list(dict.fromkeys([*expected_variants, *actual_variants]))
    if any(variant not in VARIANTS for variant in variant_names):
        raise ValueError(f"{set_name}: systems contains an unsupported variant")

    variant_reports = {}
    all_documents_by_variant = {}
    for variant in variant_names:
        scored_documents = []
        variant_responses = [
            response_map.get((document_id, variant)) for document_id in document_ids
        ]
        for document, response in zip(selected_documents, variant_responses):
            scored_documents.append(score_document(document, response))
        all_documents_by_variant[variant] = scored_documents
        aggregate_counts = _metric_counts()
        field_counts = {
            field: _metric_counts()
            for field in (*TOP_FIELDS, *(f"line_items.{name}" for name in LINE_FIELDS))
        }
        for result in scored_documents:
            for key, value in result["metrics"].items():
                if key in aggregate_counts:
                    aggregate_counts[key] += value
            for field, metrics in result["field_metrics"].items():
                for key, value in metrics.items():
                    if key in field_counts[field]:
                        field_counts[field][key] += value
        metric_summary = _field_metrics(aggregate_counts)
        complete_docs = [row for row in scored_documents if row["complete_truth"]]
        coverage_values = [
            row["document_coverage"] for row in scored_documents
            if row["document_coverage"] is not None
        ]
        response_rows = [row for row in variant_responses if row is not None]
        valid_schema_count = sum(row["schema_valid"] for row in scored_documents)
        successful = sum(
            response is not None and response.get("inference_status") == "succeeded"
            and response.get("model_used") for response in variant_responses
        )
        fallback_count = sum(
            response is not None and response.get("ocr_fallback") for response in variant_responses
        )
        modes = defaultdict(int)
        revisions = set()
        model_revisions = set()
        adapter_revisions = set()
        for response in response_rows:
            modes[response["execution_mode"]] += 1
            if response.get("software_revision"):
                revisions.add(response["software_revision"])
            if response.get("model_revision"):
                model_revisions.add(response["model_revision"])
            if response.get("adapter_revision"):
                adapter_revisions.add(response["adapter_revision"])
        latencies = {
            key: _latency_summary([
                (response.get("latency") or {}).get(key)
                for response in response_rows if isinstance(response.get("latency"), dict)
            ])
            for key in ("ocr_seconds", "generation_seconds", "end_to_end_seconds")
        }
        alignment_denominator = sum(
            max(row["line_items"]["gold_rows"], row["line_items"]["predicted_rows"])
            for row in scored_documents
        )
        alignment_numerator = sum(
            row["line_items"]["matched_rows"] for row in scored_documents
        )
        row_docs = [row for row in scored_documents if row["line_items"]["gold_rows"]]
        variant_reports[variant] = {
            "expected_documents": len(document_ids),
            "responses_received": len(response_rows),
            "missing_response_ids": [
                document_id for document_id, response in zip(document_ids, variant_responses)
                if response is None
            ],
            "model_inference_success_rate": (
                None if variant == "ocr_only" else _rate(successful, len(document_ids))
            ),
            "model_inference_successes": successful,
            "ocr_only_fallback_frequency": _rate(fallback_count, len(document_ids)),
            "ocr_only_fallback_count": fallback_count,
            "execution_modes": dict(modes),
            "software_revisions": sorted(revisions),
            "model_revisions": sorted(model_revisions),
            "adapter_revisions": sorted(adapter_revisions),
            "schema_valid_outputs": valid_schema_count,
            "schema_invalid_outputs": [
                {
                    "document_id": row["document_id"],
                    "errors": row["schema_errors"],
                }
                for row in scored_documents if row["schema_errors"]
            ],
            "metrics": metric_summary,
            "field_metrics": {
                field: _field_metrics(counts) for field, counts in field_counts.items()
            },
            "document_coverage": {
                "documents_with_any_correct_value": sum(
                    row["has_any_correct_value"] for row in scored_documents
                ),
                "document_fraction_with_any_correct_value": _rate(
                    sum(row["has_any_correct_value"] for row in scored_documents),
                    len(document_ids),
                ),
                "mean_fraction_of_present_fields_correct": (
                    sum(coverage_values) / len(coverage_values) if coverage_values else None
                ),
            },
            "complete_invoice_accuracy": {
                "correct": sum(row["complete_invoice_correct"] for row in complete_docs),
                "scorable_documents": len(complete_docs),
                "rate": _rate(
                    sum(row["complete_invoice_correct"] for row in complete_docs),
                    len(complete_docs),
                ),
            },
            "line_item_alignment_accuracy": _rate(
                alignment_numerator, alignment_denominator,
            ) if alignment_denominator else 1.0,
            "line_item_count_accuracy": _rate(
                sum(row["line_items"]["row_count_match"] for row in scored_documents),
                len(scored_documents),
            ),
            "line_item_order_accuracy": _rate(
                sum(row["line_items"]["order_preserved"] for row in row_docs),
                len(row_docs),
            ),
            "latency": latencies,
            "documents": scored_documents,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_set": set_name,
        "set_description": selected_set.get("description"),
        "independence_status": selected_set.get("independence_status", "unknown"),
        "document_count": len(document_ids),
        "comparison": _paired_comparison(
            variant_names, document_ids, all_documents_by_variant,
        ),
        "variants": variant_reports,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Invoice benchmark: {report['evaluation_set']}",
        f"Documents: {report['document_count']} · independence: {report['independence_status']}",
        f"Paired base/LoRA comparison: {report['comparison']['status']}",
    ]
    for variant, result in report["variants"].items():
        metrics = result["metrics"]
        lines.append(
            f"- {variant}: responses {result['responses_received']}/{result['expected_documents']}; "
            f"schema-valid {result['schema_valid_outputs']}; "
            f"accepted precision {metrics['accepted_value_precision']}; "
            f"field recall {metrics['field_recall']}; "
            f"exact field accuracy {metrics['exact_field_accuracy']}; "
            f"complete invoices {result['complete_invoice_accuracy']['rate']}"
        )
    return "\n".join(lines)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    responses = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc.msg}") from exc
        if not isinstance(response, dict):
            raise ValueError(f"{path}:{line_number}: each response must be a JSON object")
        responses.append(response)
    return responses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--set", dest="set_name", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Machine-readable JSON result")
    parser.add_argument("--summary", type=Path, help="Optional concise text report")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="Root for manifest-relative source files (defaults to repository root)",
    )
    args = parser.parse_args()
    report = evaluate(
        _load_json(args.manifest), _load_jsonl(args.responses), args.set_name, root=args.root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = format_report(report)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
