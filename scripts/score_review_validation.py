"""Generic offline scorer for a frozen independent P1 validation capture.

Expected manifest shape (all paths relative to the repository root unless absolute):

{
  "schema_version": "validation-manifest-1",
  "frozen": true,
  "policy_sha256": "...",
  "assessor_sha256": "...",
  "validation_target": {"documents": 15, "countries": {"FR": 5, "UK": 5, "US": 5},
                        "fields": 120, "rows": 30, "layouts": 8},
  "cases": [{
    "id": "...", "path": "validation/anonymized/....png",
    "document_class": "genuine_issued", "eligible_independent": true,
    "country": "FR", "language": "fr", "layout": "...",
    "truth": {"file": "...", "pointer": "documents/id", "sha256": "..."},
    "provenance": {"file": "...", "sha256": "..."},
    "privacy_approval": {"file": "...", "sha256": "..."},
    "original_sha256": "...", "anonymized_sha256": "..."
  }]
}

The capture is the output directory of scripts/bench_real_model.py --capture-evidence
or its bench.json. Detailed values are written only to the private output; the public
summary contains counts, paths, statuses and reason codes, never field values or OCR text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.review_assessment import assess  # noqa: E402

SCHEMA_VERSION = "review-scoring-1"
MANIFEST_SCHEMA = "validation-manifest-1"
DEFAULT_TARGET = {
    "documents": 15,
    "countries": {"FR": 5, "UK": 5, "US": 5},
    "fields": 120,
    "rows": 30,
    "layouts": 8,
}
LINE_FIELDS = ("description", "quantity", "unit_price", "total_ht")
TOP_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")
MONEY_FIELDS = {"total_ht", "total_ttc", "unit_price"}
QUANTITY_FIELDS = {"quantity"}
CURRENCY_SYMBOLS = "€$£"


class ScoringRefusal(RuntimeError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_truth_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError("truth must be an object")
    if "inline" in spec:
        truth = spec["inline"]
        expected = spec.get("sha256")
        if expected and expected != _sha256_json(truth):
            raise ValueError("inline truth sha256 mismatch")
        return truth
    if not spec.get("file"):
        raise ValueError("truth.file is missing")
    path = _path(spec["file"])
    if not path.is_file():
        raise ValueError(f"truth file missing: {path}")
    expected = spec.get("sha256")
    if not expected:
        raise ValueError("truth.sha256 is missing")
    if _sha256_file(path) != expected:
        raise ValueError("truth sha256 mismatch")
    data = _load_json(path)
    pointer = str(spec.get("pointer") or "").lstrip("/")
    for part in pointer.split("/") if pointer else []:
        if part:
            data = data[part]
    return data


def manifest_fingerprint(cases: list[Mapping[str, Any]]) -> str:
    identity = [
        {"id": case.get("id"), "path": case.get("path"),
         "truth": load_truth_spec(case.get("truth") or {}),
         "expected_flags": case.get("expected_flags") or [],
         "status": case.get("status", "diagnostic")}
        for case in cases
    ]
    return _sha256_json(identity)


def _normalize_path(path: str) -> str:
    text = str(path)
    while "[" in text and "]" in text:
        start = text.index("[")
        end = text.index("]", start)
        text = text[:start] + "." + text[start + 1:end] + text[end + 1:]
    return text


def _field_name(path: str) -> str:
    return _normalize_path(path).rsplit(".", 1)[-1]


def _as_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).replace("\u00a0", "").replace(" ", "").strip(".,;:()[]{}" + CURRENCY_SYMBOLS)
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        text = text.replace(",", "") if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() else text.replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _same(field: str, truth: Any, value: Any) -> bool:
    if truth is None or value is None:
        return truth is None and value is None
    if field in MONEY_FIELDS:
        left, right = _as_decimal(truth), _as_decimal(value)
        return left is not None and right is not None and left.quantize(Decimal("0.01")) == right.quantize(Decimal("0.01"))
    if field in QUANTITY_FIELDS:
        left, right = _as_decimal(truth), _as_decimal(value)
        return left is not None and right is not None and abs(left - right) <= Decimal("0.000001")
    return str(truth).strip() == str(value).strip()


def flatten_truth(truth: Mapping[str, Any]) -> dict[str, Any]:
    fields = {key: truth.get(key) for key in TOP_FIELDS}
    for index, row in enumerate(truth.get("line_items") or []):
        for key in LINE_FIELDS:
            fields[f"line_items.{index}.{key}"] = (row or {}).get(key)
    return fields


def _truth_rows(truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{key: (row or {}).get(key) for key in LINE_FIELDS}
            for row in (truth.get("line_items") or [])]


def _country_counts(cases: list[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(case.get("country") for case in cases))


def _validate_manifest(cases: list[Mapping[str, Any]], target: Mapping[str, Any],
                       allow_partial: bool) -> list[str]:
    problems = []
    ids = [case.get("id") for case in cases]
    if any(not isinstance(item, str) or not item for item in ids):
        problems.append("every case must have a non-empty id")
    if len(set(ids)) != len(ids):
        problems.append("case ids must be unique")
    if not allow_partial and len(cases) < int(target.get("documents") or 0):
        problems.append(f"manifest has {len(cases)} documents; target requires {target.get('documents')}")
    counts = _country_counts(cases)
    if not allow_partial:
        for country, required in (target.get("countries") or {}).items():
            if counts.get(country, 0) < int(required):
                problems.append(f"country {country}: {counts.get(country, 0)} documents; target requires {required}")
        if sum(counts.values()) < int(target.get("documents") or 0):
            problems.append("country counts do not reach the document target")
    for case in cases:
        case_id = case.get("id", "?")
        if case.get("document_class") != "genuine_issued":
            problems.append(f"{case_id}: document_class must be genuine_issued")
        if case.get("eligible_independent") is not True:
            problems.append(f"{case_id}: eligible_independent must be true")
        if not case.get("country"):
            problems.append(f"{case_id}: country is missing")
        if not case.get("layout"):
            problems.append(f"{case_id}: layout is missing")
        for key in ("truth", "provenance", "privacy_approval"):
            if not isinstance(case.get(key), Mapping) or not case[key].get("file") or not case[key].get("sha256"):
                problems.append(f"{case_id}: {key}.file and {key}.sha256 are required")
        if not case.get("original_sha256"):
            problems.append(f"{case_id}: original_sha256 is required")
        if not case.get("anonymized_sha256"):
            problems.append(f"{case_id}: anonymized_sha256 is required")
    return problems


def _read_evidence_case(case: Mapping[str, Any], capture_dir: Path,
                        case_id: str) -> dict[str, Any]:
    if not isinstance(case.get("final_payload"), Mapping):
        raise ValueError("final_payload is missing")
    if not isinstance(case["final_payload"].get("line_items"), list):
        raise ValueError("final_payload.line_items is missing")
    evidence = case.get("evidence_capture")
    evidence_source = f"capture:{case_id}"
    if not isinstance(evidence, Mapping):
        candidate = capture_dir / "evidence" / f"{case_id}.json"
        if candidate.is_file():
            evidence = _load_json(candidate)
            evidence_source = str(candidate)
        else:
            raise ValueError("evidence_capture is missing")
    if evidence.get("schema_version") != "evidence-capture-1":
        raise ValueError("unsupported evidence_capture schema")
    grounding = ((evidence.get("details") or {}).get("grounding") or {})
    statuses = grounding.get("field_statuses")
    if not isinstance(statuses, Mapping):
        raise ValueError("evidence_capture.details.grounding.field_statuses is missing")
    raw = (case.get("model") or {}).get("raw")
    raw_payload = {}
    raw_parse_valid = False
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                raw_payload = parsed
                raw_parse_valid = True
        except json.JSONDecodeError:
            pass
    return {"evidence": evidence, "statuses": statuses, "raw_payload": raw_payload,
            "raw_parse_valid": raw_parse_valid, "raw": raw,
            "evidence_source": evidence_source}


def _load_capture(capture_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    if capture_path.is_dir():
        bench_path = capture_path / "bench.json"
    else:
        bench_path = capture_path
    if not bench_path.is_file():
        raise ValueError(f"capture bench.json not found: {bench_path}")
    capture = _load_json(bench_path)
    cases = capture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("capture has no cases")
    by_id = {}
    for case in cases:
        if not isinstance(case, Mapping) or not case.get("id"):
            raise ValueError("capture case lacks id")
        by_id[case["id"]] = case
    return capture, by_id, bench_path


def _load_case_inputs(case: Mapping[str, Any], capture_case: Mapping[str, Any],
                      capture_dir: Path, problems: list[str]) -> dict[str, Any]:
    case_id = case.get("id", "?")
    loaded = {"id": case_id, "manifest": case, "capture": capture_case}
    try:
        loaded["truth"] = load_truth_spec(case.get("truth") or {})
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        problems.append(f"{case_id}: truth invalid: {exc}")
    provenance_path = None
    approval_path = None
    try:
        provenance_path = _path(case["provenance"]["file"])
        if _sha256_file(provenance_path) != case["provenance"]["sha256"]:
            problems.append(f"{case_id}: provenance sha256 mismatch")
        loaded["provenance"] = _load_json(provenance_path)
        flags = loaded["provenance"]
        if flags.get("genuinely_issued") is not True:
            problems.append(f"{case_id}: provenance does not confirm genuine issuance")
        if flags.get("previously_exposed_development") is not False:
            problems.append(f"{case_id}: provenance marks previous development exposure")
        if flags.get("used_in_p1_development") is not False:
            problems.append(f"{case_id}: provenance marks P1 development exposure")
        if flags.get("truth_independent_of_model") is not True:
            problems.append(f"{case_id}: provenance does not confirm independent truth")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        problems.append(f"{case_id}: provenance invalid: {exc}")
    try:
        approval_path = _path(case["privacy_approval"]["file"])
        if _sha256_file(approval_path) != case["privacy_approval"]["sha256"]:
            problems.append(f"{case_id}: privacy approval sha256 mismatch")
        loaded["approval"] = _load_json(approval_path)
        if loaded["approval"].get("approved") is not True:
            problems.append(f"{case_id}: privacy approval is not approved")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        problems.append(f"{case_id}: privacy approval invalid: {exc}")
    try:
        image_path = _path(case["path"])
        if not image_path.is_file():
            problems.append(f"{case_id}: document file missing: {image_path}")
        elif _sha256_file(image_path) != case["anonymized_sha256"]:
            problems.append(f"{case_id}: anonymized document sha256 mismatch")
        loaded["image_path"] = image_path
    except (ValueError, OSError, KeyError, TypeError) as exc:
        problems.append(f"{case_id}: document path invalid: {exc}")
    if case.get("original_file"):
        try:
            original = _path(case["original_file"])
            if not original.is_file() or _sha256_file(original) != case.get("original_sha256"):
                problems.append(f"{case_id}: original document hash mismatch")
            loaded["original_path"] = original
        except (ValueError, OSError, TypeError) as exc:
            problems.append(f"{case_id}: original document invalid: {exc}")
    try:
        loaded.update(_read_evidence_case(capture_case, capture_dir, case_id))
    except (ValueError, KeyError, TypeError) as exc:
        problems.append(f"{case_id}: capture invalid: {exc}")
    loaded["provenance_path"] = provenance_path
    loaded["approval_path"] = approval_path
    return loaded


def _validate_context(manifest_path: Path, capture_path: Path, policy_path: Path,
                      lock_path: Path, assessor_path: Path,
                      allow_partial: bool) -> dict[str, Any]:
    problems = []
    if not manifest_path.is_file():
        raise ScoringRefusal([f"manifest missing: {manifest_path}"])
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringRefusal([f"manifest invalid: {exc}"])
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        problems.append(f"manifest schema_version must be {MANIFEST_SCHEMA}")
    if manifest.get("frozen") is not True:
        problems.append("manifest must declare frozen=true")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ScoringRefusal(problems + ["manifest must contain a non-empty cases list"])
    target = manifest.get("validation_target") or DEFAULT_TARGET
    problems.extend(_validate_manifest(cases, target, allow_partial))
    policy_sha = _sha256_file(policy_path) if policy_path.is_file() else None
    lock_sha = lock_path.read_text(encoding="ascii").split()[0] if lock_path.is_file() else None
    assessor_sha = _sha256_file(assessor_path) if assessor_path.is_file() else None
    if not policy_sha:
        problems.append("frozen policy file is missing")
    if not lock_sha or policy_sha != lock_sha:
        problems.append("frozen policy hash lock mismatch")
    if manifest.get("policy_sha256") != policy_sha:
        problems.append("manifest policy_sha256 does not match the frozen policy")
    if not assessor_sha:
        problems.append("frozen assessor file is missing")
    if manifest.get("assessor_sha256") != assessor_sha:
        problems.append("manifest assessor_sha256 does not match the frozen assessor")
    try:
        capture, capture_by_id, bench_path = _load_capture(capture_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ScoringRefusal(problems + [f"capture invalid: {exc}"])
    manifest_ids = {case.get("id") for case in cases}
    if set(capture_by_id) != manifest_ids:
        problems.append("capture case ids do not exactly match manifest ids")
    expected_fingerprint = None
    try:
        expected_fingerprint = manifest_fingerprint(cases)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        problems.append(f"manifest truth/fingerprint invalid: {exc}")
    if expected_fingerprint and capture.get("manifest", {}).get("sha256") != expected_fingerprint:
        problems.append("capture manifest fingerprint does not match the validation manifest")
    loaded_cases = []
    for case in cases:
        capture_case = capture_by_id.get(case.get("id"))
        if capture_case is None:
            problems.append(f"{case.get('id')}: missing capture case")
            continue
        if capture_case.get("path") != case.get("path"):
            problems.append(f"{case.get('id')}: capture path does not match manifest path")
        loaded = _load_case_inputs(case, capture_case, bench_path.parent, problems)
        loaded_cases.append(loaded)
    if not allow_partial:
        fields = sum(len(flatten_truth(loaded.get("truth") or {})) for loaded in loaded_cases)
        rows = sum(len((loaded.get("truth") or {}).get("line_items") or []) for loaded in loaded_cases)
        layouts = len({loaded["manifest"].get("layout") for loaded in loaded_cases})
        if fields < int(target.get("fields") or 0):
            problems.append(f"scored fields {fields}; target requires {target.get('fields')}")
        if rows < int(target.get("rows") or 0):
            problems.append(f"line rows {rows}; target requires {target.get('rows')}")
        if layouts < int(target.get("layouts") or 0):
            problems.append(f"distinct layouts {layouts}; target requires {target.get('layouts')}")
    if problems:
        raise ScoringRefusal(problems)
    return {"manifest": manifest, "target": target, "cases": loaded_cases,
            "capture": capture, "capture_path": bench_path,
            "policy_sha256": policy_sha, "assessor_sha256": assessor_sha,
            "policy_path": policy_path, "assessor_path": assessor_path,
            "lock_path": lock_path, "manifest_path": manifest_path,
            "allow_partial": allow_partial}


def _input_paths(context: Mapping[str, Any]) -> list[Path]:
    paths = [context["manifest_path"], context["policy_path"], context["lock_path"],
             context["assessor_path"], context["capture_path"]]
    for loaded in context["cases"]:
        paths.append(loaded["image_path"])
        for key in ("provenance_path", "approval_path"):
            if loaded.get(key):
                paths.append(loaded[key])
        truth_spec = loaded["manifest"].get("truth") or {}
        if truth_spec.get("file"):
            paths.append(_path(truth_spec["file"]))
        if loaded.get("original_path"):
            paths.append(loaded["original_path"])
        evidence = loaded.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("schema_version"):
            paths.append(context["capture_path"])  # inline evidence is part of the capture
        source = loaded.get("evidence_source")
        if isinstance(source, str) and not source.startswith("capture:"):
            paths.append(Path(source))
    return sorted(set(paths), key=lambda path: str(path))


def _hash_inputs(context: Mapping[str, Any]) -> dict[str, str]:
    return {str(path): _sha256_file(path) for path in _input_paths(context) if path.is_file()}


def _evidence_ref(assessment, source: str) -> dict[str, Any] | None:
    evidence = assessment.evidence or {}
    payload = {key: evidence.get(key) for key in ("kind", "page", "box", "confidence")
               if evidence.get(key) is not None}
    if not payload:
        return None
    payload["source"] = source
    return payload


def _row_from_truth(truth: Mapping[str, Any], index: int) -> dict[str, Any]:
    rows = truth.get("line_items") or []
    return {key: (rows[index] or {}).get(key) for key in LINE_FIELDS} if index < len(rows) else {}


def _same_row(truth_row: Mapping[str, Any], visible: Mapping[str, Any]) -> bool:
    return all(visible.get(key) is not None and _same(key, truth_row.get(key), visible.get(key))
               for key in LINE_FIELDS)


def _row_metrics(fields: list[dict[str, Any]], truth: Mapping[str, Any]) -> dict[str, Any]:
    truth_rows = _truth_rows(truth)
    visible = [field for field in fields if field["tier"] in {"verified", "proposed"}
               and field.get("value") is not None]
    by_row = {}
    for field in visible:
        row = field.get("row")
        if row is not None:
            by_row.setdefault(row, {})[field["field"]] = field["value"]
    visible_rows = sorted(by_row)
    exact_same = 0
    numeric_same = 0
    description_same = 0
    for index, truth_row in enumerate(truth_rows):
        values = by_row.get(index, {})
        if _same_row(truth_row, values):
            exact_same += 1
        if all(values.get(key) is not None and _same(key, truth_row.get(key), values.get(key))
               for key in ("quantity", "unit_price", "total_ht")):
            numeric_same += 1
        if values.get("description") is not None and _same("description", truth_row.get("description"),
                                                           values.get("description")):
            description_same += 1
    used = set()
    exact_any = 0
    for index in visible_rows:
        values = by_row[index]
        for truth_index, truth_row in enumerate(truth_rows):
            if truth_index in used:
                continue
            if _same_row(truth_row, values):
                used.add(truth_index)
                exact_any += 1
                break
    same_index = misattributed = incorrect = 0
    for field in visible:
        row = field.get("row")
        truth_row = _row_from_truth(truth, row) if row is not None else {}
        if row is not None and _same(field["field"], truth_row.get(field["field"]), field["value"]):
            same_index += 1
        elif any(_same(field["field"], other.get(field["field"]), field["value"])
                 for other in truth_rows):
            misattributed += 1
        else:
            incorrect += 1
    return {
        "truth_rows": len(truth_rows),
        "visible_rows": len(visible_rows),
        "exact_rows_same_index": exact_same,
        "exact_rows_any_index": exact_any,
        "numeric_rows_same_index": numeric_same,
        "description_rows_same_index": description_same,
        "missing_rows": max(0, len(truth_rows) - len([i for i in range(len(truth_rows)) if i in by_row])),
        "extra_rows": len([index for index in visible_rows if index >= len(truth_rows)]),
        "row_fields_same_index": same_index,
        "row_fields_misattributed": misattributed,
        "row_fields_incorrect": incorrect,
        "exact_row_attribution_accuracy": round(exact_same / len(truth_rows), 4) if truth_rows else None,
        "exact_any_index_accuracy": round(exact_any / len(truth_rows), 4) if truth_rows else None,
    }


def _metrics(fields: list[dict[str, Any]], truth: Mapping[str, Any]) -> dict[str, Any]:
    truth_fields = flatten_truth(truth)
    printed = {path for path, value in truth_fields.items() if value is not None}
    verified = [f for f in fields if f["tier"] == "verified" and f.get("value") is not None]
    proposed = [f for f in fields if f["tier"] == "proposed" and f.get("value") is not None]
    visible = verified + proposed
    rejected = [f for f in fields if f["tier"] == "rejected"]
    missing = [f for f in fields if f["tier"] == "missing"]
    verified_correct = sum(1 for f in verified if f["outcome"] == "correct")
    proposed_correct = sum(1 for f in proposed if f["outcome"] == "correct")
    visible_correct = sum(1 for f in visible if f["outcome"] == "correct")
    verified_printed = sum(1 for f in verified if f["path"] in printed)
    proposed_printed = sum(1 for f in proposed if f["path"] in printed)
    visible_printed = sum(1 for f in visible if f["path"] in printed)
    correct_visible_printed = sum(1 for f in visible if f["path"] in printed and f["outcome"] == "correct")
    return {
        "fields": len(fields),
        "truth_fields": len(truth_fields),
        "printed_fields": len(printed),
        "verified_total": len(verified),
        "verified_correct": verified_correct,
        "verified_incorrect": len(verified) - verified_correct,
        "verified_precision": round(verified_correct / len(verified), 4) if verified else None,
        "proposed_total": len(proposed),
        "proposed_correct": proposed_correct,
        "proposed_incorrect": len(proposed) - proposed_correct,
        "proposed_precision": round(proposed_correct / len(proposed), 4) if proposed else None,
        "visible_total": len(visible),
        "visible_correct": visible_correct,
        "visible_incorrect": len(visible) - visible_correct,
        "visible_precision": round(visible_correct / len(visible), 4) if visible else None,
        "rejected_total": len(rejected),
        "correctly_rejected": sum(1 for f in rejected if f["outcome"] == "correctly_rejected"),
        "incorrectly_rejected": sum(1 for f in rejected if f["outcome"] == "incorrectly_rejected"),
        "missing_printed": sum(1 for f in missing if f["outcome"] == "missing_printed"),
        "correct_abstention": sum(1 for f in missing if f["outcome"] == "correct_abstention"),
        "verified_printed_coverage": round(verified_printed / len(printed), 4) if printed else None,
        "proposed_printed_coverage": round(proposed_printed / len(printed), 4) if printed else None,
        "visible_printed_coverage": round(visible_printed / len(printed), 4) if printed else None,
        "correct_visible_printed_coverage": round(correct_visible_printed / len(printed), 4) if printed else None,
        "visible_coverage_all_fields": round(len(visible) / len(fields), 4) if fields else None,
    }


def _sum_metrics(fields: list[dict[str, Any]], truth_field_count: int,
                 printed_field_count: int) -> dict[str, Any]:
    verified = [f for f in fields if f["tier"] == "verified" and f.get("value") is not None]
    proposed = [f for f in fields if f["tier"] == "proposed" and f.get("value") is not None]
    visible = verified + proposed
    rejected = [f for f in fields if f["tier"] == "rejected"]
    missing = [f for f in fields if f["tier"] == "missing"]
    verified_correct = sum(1 for f in verified if f["outcome"] == "correct")
    proposed_correct = sum(1 for f in proposed if f["outcome"] == "correct")
    visible_correct = sum(1 for f in visible if f["outcome"] == "correct")
    verified_printed = sum(1 for f in verified if f.get("truth") is not None)
    proposed_printed = sum(1 for f in proposed if f.get("truth") is not None)
    visible_printed = sum(1 for f in visible if f.get("truth") is not None)
    correct_visible_printed = sum(1 for f in visible
                                  if f.get("truth") is not None and f["outcome"] == "correct")
    return {
        "fields": len(fields), "truth_fields": truth_field_count,
        "printed_fields": printed_field_count,
        "verified_total": len(verified), "verified_correct": verified_correct,
        "verified_incorrect": len(verified) - verified_correct,
        "verified_precision": round(verified_correct / len(verified), 4) if verified else None,
        "proposed_total": len(proposed), "proposed_correct": proposed_correct,
        "proposed_incorrect": len(proposed) - proposed_correct,
        "proposed_precision": round(proposed_correct / len(proposed), 4) if proposed else None,
        "visible_total": len(visible), "visible_correct": visible_correct,
        "visible_incorrect": len(visible) - visible_correct,
        "visible_precision": round(visible_correct / len(visible), 4) if visible else None,
        "rejected_total": len(rejected),
        "correctly_rejected": sum(1 for f in rejected if f["outcome"] == "correctly_rejected"),
        "incorrectly_rejected": sum(1 for f in rejected if f["outcome"] == "incorrectly_rejected"),
        "missing_printed": sum(1 for f in missing if f["outcome"] == "missing_printed"),
        "correct_abstention": sum(1 for f in missing if f["outcome"] == "correct_abstention"),
        "verified_printed_coverage": round(verified_printed / printed_field_count, 4) if printed_field_count else None,
        "proposed_printed_coverage": round(proposed_printed / printed_field_count, 4) if printed_field_count else None,
        "visible_printed_coverage": round(visible_printed / printed_field_count, 4) if printed_field_count else None,
        "correct_visible_printed_coverage": round(correct_visible_printed / printed_field_count, 4) if printed_field_count else None,
        "visible_coverage_all_fields": round(len(visible) / len(fields), 4) if fields else None,
    }


def _field_record(path: str, truth: Any, assessment, source: str) -> dict[str, Any]:
    if assessment is None:
        tier = "missing"
        status = "missing_printed" if truth is not None else "absent"
        outcome = "missing_printed" if truth is not None else "correct_abstention"
        reason = "no_assessment_truth_only"
        value = None
        model_value = None
        evidence_ref = None
        row = None
        raw_class = None
    else:
        status = assessment.status
        value = assessment.value if status in {"verified", "proposed"} else None
        model_value = assessment.model_value
        reason = assessment.reason
        evidence_ref = _evidence_ref(assessment, source)
        row = assessment.row
        raw_class = getattr(assessment, "raw_class", None)
        if status in {"verified", "proposed"} and value is not None:
            tier = status
            outcome = "correct" if truth is not None and _same(_field_name(path), truth, value) else "incorrect"
        elif status == "rejected":
            tier = "rejected"
            outcome = ("incorrectly_rejected" if truth is not None
                       and _same(_field_name(path), truth, model_value) else "correctly_rejected")
        else:
            tier = "missing"
            outcome = "missing_printed" if truth is not None else "correct_abstention"
    row_context = None
    if isinstance(row, Mapping):
        row_context = {key: row.get(key) for key in
                       ("index", "order", "source", "identity", "line_confidence", "page", "box")
                       if row.get(key) is not None}
    return {
        "path": path,
        "field": _field_name(path),
        "row": None if row is None else row.get("index"),
        "tier": tier,
        "status": status,
        "outcome": outcome,
        "reason": reason,
        "value": value,
        "truth": truth,
        "model_value": model_value,
        "raw_class": raw_class,
        "evidence_ref": evidence_ref,
        "row_context": row_context,
    }


def _score_case(loaded: Mapping[str, Any]) -> dict[str, Any]:
    truth = loaded["truth"]
    evidence = loaded["evidence"]
    statuses = loaded["statuses"]
    raw_payload = loaded["raw_payload"]
    strict_payload = loaded["capture"]["final_payload"]
    assessments = assess(evidence, raw_payload, {"payload": strict_payload, "statuses": statuses})
    truth_fields = flatten_truth(truth)
    paths = sorted(set(truth_fields) | {_normalize_path(path) for path in assessments})
    source = loaded.get("evidence_source") or f"capture:{loaded['id']}"
    fields = []
    for path in paths:
        assessment = next((item for key, item in assessments.items()
                           if _normalize_path(key) == path), None)
        fields.append(_field_record(path, truth_fields.get(path), assessment, source))
    metrics = _metrics(fields, truth)
    rows = _row_metrics(fields, truth)
    public_fields = [{key: field.get(key) for key in ("path", "field", "row", "tier", "status",
                                                       "outcome", "reason", "evidence_ref")}
                     for field in fields]
    return {
        "id": loaded["id"],
        "country": loaded["manifest"].get("country"),
        "language": loaded["manifest"].get("language"),
        "layout": loaded["manifest"].get("layout"),
        "document_class": loaded["manifest"].get("document_class"),
        "raw_parse_valid": loaded["raw_parse_valid"],
        "metrics": metrics,
        "rows": rows,
        "fields": fields,
        "public_fields": public_fields,
    }


def score_validation(manifest_path: str | Path, capture_path: str | Path, *,
                     policy_path: str | Path = "experiments/review_assessment_policy.json",
                     lock_path: str | Path = "experiments/review_assessment_policy.sha256",
                     assessor_path: str | Path = "experiments/review_assessment.py",
                     allow_partial: bool = False) -> dict[str, Any]:
    context = _validate_context(_path(manifest_path), _path(capture_path),
                               _path(policy_path), _path(lock_path), _path(assessor_path),
                               allow_partial)
    before = _hash_inputs(context)
    scored = [_score_case(loaded) for loaded in context["cases"]]
    after = _hash_inputs(context)
    if before != after:
        raise ScoringRefusal(["an input file changed during scoring; refusing to trust the result"])
    all_fields = [field for case in scored for field in case["fields"]]
    truth_fields = sum(len(flatten_truth(case["truth"])) for case in context["cases"])
    printed_fields = sum(len([value for value in flatten_truth(case["truth"]).values()
                              if value is not None]) for case in context["cases"])
    overall = _sum_metrics(all_fields, truth_fields, printed_fields)
    rows = {key: sum(case["rows"].get(key) or 0 for case in scored)
            for key in ("truth_rows", "visible_rows", "exact_rows_same_index", "exact_rows_any_index",
                        "numeric_rows_same_index", "description_rows_same_index", "missing_rows",
                        "extra_rows", "row_fields_same_index", "row_fields_misattributed",
                        "row_fields_incorrect")}
    rows["exact_row_attribution_accuracy"] = round(rows["exact_rows_same_index"] / rows["truth_rows"], 4) if rows["truth_rows"] else None
    rows["exact_any_index_accuracy"] = round(rows["exact_rows_any_index"] / rows["truth_rows"], 4) if rows["truth_rows"] else None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "scored",
        "policy_sha256": context["policy_sha256"],
        "assessor_sha256": context["assessor_sha256"],
        "capture_manifest_sha256": context["capture"].get("manifest", {}).get("sha256"),
        "documents": [{key: case[key] for key in ("id", "country", "language", "layout",
                                                   "document_class", "raw_parse_valid")}
                      for case in scored],
        "overall": overall,
        "rows": rows,
        "public_fields": [{key: field[key] for key in ("path", "field", "row", "tier", "status",
                                                        "outcome", "reason")}
                          for field in all_fields],
        "input_hashes": before,
        "privacy": {"public_values": False, "public_ocr_text": False},
    }
    details = {
        "schema_version": SCHEMA_VERSION,
        "status": "scored",
        "privacy": "private; contains field values but no OCR line text",
        "policy_sha256": context["policy_sha256"],
        "assessor_sha256": context["assessor_sha256"],
        "capture_manifest_sha256": context["capture"].get("manifest", {}).get("sha256"),
        "documents": [{**{key: case[key] for key in ("id", "country", "language", "layout",
                                                      "document_class", "raw_parse_valid")},
                       "metrics": case["metrics"], "rows": case["rows"],
                       "fields": case["fields"]} for case in scored],
        "overall": overall,
        "rows": rows,
        "input_hashes": before,
    }
    return {"summary": summary, "details": details, "context": context}


def _default_output_root() -> Path:
    return ROOT / ".venv" / "validation-results"


def _validate_output_path(output: Path, allow_unignored: bool) -> None:
    root = _default_output_root().resolve()
    resolved = output.resolve()
    if not (resolved == root or root in resolved.parents):
        if not allow_unignored:
            raise ScoringRefusal([f"output must be under {root} (ignored by Git); "
                                  "use --allow-unignored-output only for local tests"])
    if resolved.exists():
        raise ScoringRefusal([f"output already exists: {resolved}"])


def _write_outputs(result: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    try:
        os.chmod(output, 0o700)
    except OSError:
        pass
    summary_path = output / "summary.json"
    details_path = output / "details.json"
    summary_path.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
    details_path.write_text(json.dumps(result["details"], ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
    for path in (summary_path, details_path):
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", type=Path, default=Path("experiments/review_assessment_policy.json"))
    parser.add_argument("--policy-lock", type=Path, default=Path("experiments/review_assessment_policy.sha256"))
    parser.add_argument("--assessor", type=Path, default=Path("experiments/review_assessment.py"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-unignored-output", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight_only:
            context = _validate_context(_path(args.manifest), _path(args.capture),
                                        _path(args.policy), _path(args.policy_lock),
                                        _path(args.assessor), args.allow_partial)
            before = _hash_inputs(context)
            print(json.dumps({
                "status": "preflight_ok",
                "documents": [case["id"] for case in context["cases"]],
                "capture_manifest_sha256": context["capture"].get("manifest", {}).get("sha256"),
                "policy_sha256": context["policy_sha256"],
                "assessor_sha256": context["assessor_sha256"],
                "input_hashes": before,
                "model_loaded": False,
                "ocr_executed": False,
            }, ensure_ascii=False, indent=1, sort_keys=True))
            return 0
        if args.output is None:
            parser.error("--output is required unless --preflight-only is used")
        _validate_output_path(args.output, args.allow_unignored_output)
        result = score_validation(args.manifest, args.capture, policy_path=args.policy,
                                  lock_path=args.policy_lock, assessor_path=args.assessor,
                                  allow_partial=args.allow_partial)
        _write_outputs(result, args.output)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=1, sort_keys=True))
        return 0
    except ScoringRefusal as exc:
        print(json.dumps({"status": "refused", "problems": exc.problems}, ensure_ascii=False,
                         indent=1, sort_keys=True))
        return 2
    except Exception as exc:  # never expose field values or OCR text in an error
        print(json.dumps({"status": "refused",
                          "problems": [f"internal scoring error: {type(exc).__name__}"]},
                         ensure_ascii=False, indent=1, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
