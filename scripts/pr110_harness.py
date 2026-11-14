"""PR110 shared harness: the pieces every PR110 script reuses, and nothing else.

PR109 established the requirement (twelve genuinely independent invoices) and built the contamination screen,
then stopped because the repository contains zero eligible documents. PR110 is benchmark preparation, so this
harness is deliberately thin: it holds the reference-annotation helpers the owner-facing CLI needs, the one
projector that turns a PR110 reference into the shape `src/evaluation_metrics.py` already consumes, the
cohort composition categories PR109 asked for, and the privacy screen that keeps invoice text out of tracked
reports.

Nothing here re-implements what already exists, and nothing here runs the pipeline:

* duplicate and contamination screening comes from `scripts/pr109_intake.py` (`sha256_file`, `perceptual`,
  `hamming`, `known_content`, `signed_population`) and is only *called*, never copied;
* the schema rules come from `scripts/pr103_validate_reference.py`, which PR110 extended additively for
  `version: 2`;
* the freeze helpers (source hashes, PR103 drift, environment, git state, artifact guards) come from
  `scripts/pr106_harness.py`;
* scoring comes from `src/evaluation_metrics.py`, reached through the projector below.

No production extraction file is imported for modification, and importing this module makes 0 OCR calls and
0 model calls.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr103_validate_reference as V  # noqa: E402
import pr106_harness as P  # noqa: E402
import pr109_intake as I  # noqa: E402

INVOICE_DIR = ROOT / "benchmark_prep" / "invoices"
REFERENCE_DIR = ROOT / "benchmark_prep" / "references"
ARTIFACT_DIR = ROOT / "results" / "product-review110"
TEMPLATE = ROOT / "benchmark_prep" / "pr110-reference-template.json"

TARGET_DOCUMENTS = I.TARGET_DOCUMENTS
PROVENANCE = sorted(V.PROVENANCE_V2)
NULL_PROVENANCE = sorted(V.NULL_VALUE_PROVENANCE_V2)
VALUE_PROVENANCE = [state for state in PROVENANCE if state not in NULL_PROVENANCE]
HEADER_FIELDS = V.VALUE_FIELDS
ROW_FIELDS = V.ROW_FIELDS
# The evaluator scores four header fields; currency and tax are recorded but never scored as header cells.
EVALUATED_HEADER_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")
QUALITY_SIGNALS = ("small_text", "low_contrast", "skewed_or_cropped")
SEPARATOR_TOKENS = ("|", ";", "\n", "\u2022")
GITIGNORE_RULES = ("benchmark_prep/invoices/", "benchmark_prep/references/", "results/product-review110/")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def display(path: Path) -> str:
    """A repo-relative path when there is one, otherwise the path as given (test directories live outside)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- annotation authoring


def blank(document_id: str, source_file: str | None = None, pages: list[int] | None = None) -> dict:
    """A version-2 template with its identity filled in, ready for the owner to annotate."""
    reference = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    document = reference["document"]
    document["id"] = document_id
    document["source_file"] = source_file
    document["pages"] = list(pages or [1])
    document["owner_permission"]["date"] = time.strftime("%Y-%m-%d", time.gmtime())
    # The template ships one example row; an annotation starts empty and grows as the owner adds lines.
    reference["reference"]["line_items"] = []
    return reference


def set_field(reference: dict, path: str, value, provenance: str,
              derived_from: str | None = None, note: str | None = None) -> dict:
    """Set one annotated field by path: `supplier_name`, `total_ht` or `line_items[0].unit_price`."""
    if provenance not in PROVENANCE:
        raise ValueError(f"unknown provenance {provenance!r}; expected one of {PROVENANCE}")
    if provenance in NULL_PROVENANCE:
        value = None
    if provenance == "printed" and value is None:
        raise ValueError("a printed field must carry the value the document prints")
    field = {"value": value, "provenance": provenance}
    if derived_from:
        field["derived_from"] = derived_from
    if note:
        field["note"] = note
    target = _resolve(reference["reference"], path)
    target.update(field)
    for key in ("derived_from", "note"):
        if key not in field:
            target.pop(key, None)
    return target


def _resolve(body: dict, path: str) -> dict:
    """The {value, provenance} object at `path`, creating it (or a row) when it does not exist yet.

    `tax.rate` and `shipping` start as plain nulls in the template; annotating one replaces the null with the
    object the validator asks for, while the sibling keys a note lives in (`printed_label`, `printed_form`)
    are kept.
    """
    head, _, tail = path.partition(".")
    if head.split("[")[0] == "line_items":
        if not tail:
            raise ValueError("a line-item path needs a field, e.g. line_items[0].unit_price")
        rows = body.setdefault("line_items", [])
        digits = head.partition("[")[2].rstrip("]")
        position = int(digits) if digits.isdigit() else len(rows)
        while len(rows) <= position:
            rows.append({"is_product_or_service": True})
        container, key = rows[position], tail
    else:
        container = body
        for part in path.split(".")[:-1]:
            if not isinstance(container, dict):
                raise ValueError(f"{path} passes through {part}, which is not an object")
            if not isinstance(container.get(part), dict):
                container[part] = {}
            container = container[part]
        key = path.split(".")[-1]
    if not isinstance(container, dict):
        raise ValueError(f"{path} is not a field path")
    if not isinstance(container.get(key), dict):
        container[key] = {}
    return container[key]


def add_row(reference: dict) -> int:
    """Append an empty product row and return its index."""
    rows = reference["reference"].setdefault("line_items", [])
    rows.append({"is_product_or_service": True})
    return len(rows) - 1


# --------------------------------------------------------------------------- freezing


def validate(annotation: dict, image: Path | None = None) -> list[str]:
    """Every PR103 rule plus the version-2 additions, through the one published validator."""
    return V.validate(annotation, image)


def seal(annotation: dict, image: Path | None = None) -> dict:
    """Fill `freeze` from the bytes on disk, refusing to write anything the validator does not accept.

    The freeze block is part of what the validator checks (`reference_sha256` must exist and
    `frozen_before_pipeline_run` must be true), so the sealed shape is validated before it is committed: the
    annotation the owner sees is either accepted whole or left exactly as it was.
    """
    document = annotation["document"]
    if image is not None:
        document["source_file"] = document.get("source_file") or image.name
        document["sha256"] = I.sha256_file(image)
    sealed = copy.deepcopy(annotation)
    without_freeze = copy.deepcopy(sealed)
    without_freeze["freeze"] = {}
    sealed["freeze"] = {
        "reference_sha256": digest(without_freeze),
        "body_sha256": digest(sealed["reference"]),
        "frozen_before_pipeline_run": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    problems = validate(sealed, image)
    if problems:
        raise SystemExit("REFUSED — the reference is not sealable:\n  " + "\n  ".join(problems))
    annotation.clear()
    annotation.update(sealed)
    return annotation


def verify_seal(annotation: dict, image: Path | None = None) -> dict:
    """Recompute the freeze from the bytes and report whether the sealed reference is still intact.

    A reference that was edited after sealing keeps its old `reference_sha256`, so the recomputed digest
    differs and the gate can refuse it. PR109's boundary is that the truth must be frozen *before* the first
    pipeline run; this is what makes that boundary checkable instead of a promise.
    """
    freeze = annotation.get("freeze") or {}
    without_freeze = copy.deepcopy(annotation)
    without_freeze["freeze"] = {}
    recomputed = digest(without_freeze)
    body = digest(annotation.get("reference") or {})
    document = annotation.get("document") or {}
    on_disk = I.sha256_file(image) if image is not None else None
    return {
        "sealed": bool(freeze.get("reference_sha256")),
        "reference_sha256_matches": freeze.get("reference_sha256") == recomputed,
        "body_sha256_matches": freeze.get("body_sha256") == body,
        "frozen_before_pipeline_run": freeze.get("frozen_before_pipeline_run") is True,
        "sha256_matches_image_on_disk": on_disk is not None and document.get("sha256") == on_disk,
        "intact": (bool(freeze.get("reference_sha256")) and freeze.get("reference_sha256") == recomputed
                   and freeze.get("body_sha256") == body
                   and freeze.get("frozen_before_pipeline_run") is True
                   and on_disk is not None and document.get("sha256") == on_disk),
    }


# --------------------------------------------------------------------------- projection into the evaluator


def project(annotation: dict) -> dict:
    """Turn a PR110 reference into the plain shape `evaluation_metrics.evaluate` already reads.

    `printed` and `derived` carry a value into scoring. `absent`, `illegible` and `ambiguous` become null,
    which the evaluator counts as unverified and excludes from every denominator - so a field the document
    does not print can never be scored as either a correct or an incorrect extraction.
    """
    body = annotation.get("reference") or {}
    counts = {state: 0 for state in PROVENANCE}
    counts["not_annotated"] = 0
    unverified: list[str] = []

    def value_of(path: str, field):
        if not isinstance(field, dict) or field.get("provenance") is None:
            counts["not_annotated"] += 1
            unverified.append(path)
            return None
        state = field.get("provenance")
        counts[state if state in counts else "not_annotated"] += 1
        if state in VALUE_PROVENANCE:
            return field.get("value")
        unverified.append(path)
        return None

    scored = {field: value_of(field, body.get(field)) for field in EVALUATED_HEADER_FIELDS}
    rows = []
    for index, row in enumerate(body.get("line_items") or []):
        rows.append({field: value_of(f"line_items[{index}].{field}", row.get(field)) for field in ROW_FIELDS})
    scored["line_items"] = rows
    return {
        "reference": scored,
        "counts": {
            "fields_total": sum(counts.values()),
            "by_provenance": {state: counts[state] for state in PROVENANCE},
            "not_annotated": counts["not_annotated"],
            "scored_values": counts["printed"] + counts["derived"],
            "left_unverified": len(unverified),
        },
        "unverified_fields": unverified,
        "note": "printed and derived values are scored; absent, illegible and ambiguous are null, which "
                "evaluation_metrics reports as unverified and excludes from its denominators",
    }


# --------------------------------------------------------------------------- cohort composition


def coverage(records: list[dict], geometry: dict[str, dict] | None = None) -> list[dict]:
    """PR109's twelve Phase-3 categories, expressed over version-2 annotations.

    PR109's own `coverage()` could not be reused: it reads the development corpus's record shape
    (`provenance_recorded`, `geometry`, `structure`), which only exists after a pipeline run. These twelve
    categories must be answerable from the annotation alone, before the pipeline is run once.

    `records` are `cohort()` entries; `geometry` maps a document id to `pr109_dev_cohort.ruled_geometry()`
    for the two table categories, which are the only ones that need the image rather than the annotation.
    """
    geometry = geometry or {}

    def annotated(record) -> dict:
        return (record.get("annotation") or {}).get("reference") or {}

    def document_field(record, name: str):
        return ((record.get("annotation") or {}).get("document") or {}).get(name)

    def declared(record, name: str):
        field = annotated(record).get(name) or {}
        return field.get("value") if field.get("provenance") in VALUE_PROVENANCE else None

    def row_list(record) -> list[dict]:
        return annotated(record).get("line_items") or []

    def table_found(record):
        return (geometry.get(str(record.get("id"))) or {}).get("bordered_table_detected_by_production")

    def where(predicate):
        return sorted(str(record["id"]) for record in records if predicate(record))

    def multi_segment(record):
        for row in row_list(record):
            text = str((row.get("description") or {}).get("value") or "")
            if any(token in text for token in SEPARATOR_TOKENS):
                return True
        return False

    categories = [
        ("French invoices", where(
            lambda r: document_field(r, "language") == "French" or declared(r, "currency") == "EUR")),
        ("British invoices", where(
            lambda r: document_field(r, "region") == "UK" or declared(r, "currency") == "GBP")),
        ("American invoices", where(lambda r: document_field(r, "region") == "US")),
        ("Canadian invoices", where(lambda r: document_field(r, "region") == "CA")),
        ("Genuinely issued invoices (not templates)",
         where(lambda r: document_field(r, "genuine_business_invoice") is not False)),
        ("Bordered tables (production finds a ruled region)", where(lambda r: table_found(r) is True)),
        ("Borderless tables (production finds no ruled region)", where(lambda r: table_found(r) is False)),
        ("Multi-line / multi-segment descriptions", where(multi_segment)),
        ("Additional reference columns beyond the exported four",
         where(lambda r: any(row.get("reference_number") for row in row_list(r)))),
        ("Explicit tax breakdown in the reference", where(
            lambda r: declared(r, "total_ht") is not None and declared(r, "total_ttc") is not None
            and str(declared(r, "total_ht")) != str(declared(r, "total_ttc")))),
        ("No explicit tax breakdown (single total, or equal HT and TTC)", where(
            lambda r: declared(r, "total_ttc") is None
            or str(declared(r, "total_ht")) == str(declared(r, "total_ttc")))),
        ("Two or more line items", where(lambda r: len(row_list(r)) >= 2)),
        ("Small text or low-confidence acquisition",
         where(lambda r: document_field(r, "quality_declaration") in QUALITY_SIGNALS)),
    ]
    return [{"category": label, "documents": names, "count": len(names), "underrepresented": len(names) < 2}
            for label, names in categories]


# --------------------------------------------------------------------------- privacy


def sensitive_strings(annotations: list[dict]) -> list[str]:
    """Supplier names and line descriptions: the strings that must never appear in a tracked PR110 report."""
    found: set[str] = set()
    for annotation in annotations:
        body = (annotation or {}).get("reference") or {}
        for key in ("supplier_name", "invoice_date"):
            value = (body.get(key) or {}).get("value")
            if isinstance(value, str) and len(value.strip()) >= 3:
                found.add(value.strip())
        for row in body.get("line_items") or []:
            value = (row.get("description") or {}).get("value")
            if isinstance(value, str) and len(value.strip()) >= 3:
                found.add(value.strip())
    return sorted(found)


def screen(text: str, sensitive: list[str]) -> list[str]:
    """The invoice strings still present in a draft report."""
    lowered = text.lower()
    return sorted({value for value in sensitive if value.lower() in lowered})


# --------------------------------------------------------------------------- intake state


def reference_files(directory: Path = REFERENCE_DIR) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*.json"))


def invoice_files(directory: Path = INVOICE_DIR) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in I.DOCUMENT_SUFFIXES)


def cohort(directory: Path = REFERENCE_DIR, invoice_dir: Path = INVOICE_DIR) -> list[dict]:
    """One record per reference file, paired with the invoice it annotates and checked by the validator."""
    on_disk = invoice_files(invoice_dir)
    by_sha256: dict[str, Path] = {}
    by_name: dict[str, Path] = {}
    for path in on_disk:
        by_name[path.name] = path
        by_sha256[I.sha256_file(path)] = path
    records = []
    for path in reference_files(directory):
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            records.append({"file": display(path), "id": path.stem,
                            "annotation": None, "image": None, "image_found": False,
                            "problems": [f"not valid JSON: {error}"], "eligible": False})
            continue
        document = annotation.get("document") or {}
        image = by_name.get(str(document.get("source_file"))) or by_sha256.get(document.get("sha256"))
        problems = validate(annotation, image)
        seal_state = verify_seal(annotation, image)
        records.append({
            "file": display(path),
            "id": document.get("id") or path.stem,
            "version": annotation.get("version") or 1,
            "annotation": annotation,
            "image": image,
            "image_found": image is not None,
            "sha256": document.get("sha256"),
            "problems": problems,
            "content_problems": [problem for problem in problems if not problem.startswith("freeze.")],
            "seal": seal_state,
            "eligible": image is not None and not problems and seal_state["intact"],
        })
    return records


# --------------------------------------------------------------------------- Phase 1 reuse map


def component_inventory() -> dict:
    """What PR110 reuses, measured from the files rather than asserted.

    Each entry asks the module for the attributes PR110 actually calls, so a dependency that has drifted
    away shows up as an unavailable attribute instead of as a broken script later on.
    """
    probes = [
        ("scripts/pr109_intake.py", I,
         ["sha256_file", "sha256_bytes", "intake_locations", "known_content", "perceptual", "hamming",
          "detector_self_test", "signed_population", "assess", "TARGET_DOCUMENTS",
          "PERCEPTUAL_MAX_DISTANCE", "DOCUMENT_SUFFIXES"], False, {}),
        ("scripts/pr106_harness.py", P,
         ["guard", "write_json", "source_hashes", "pr103_freeze", "environment", "git_state", "documents"],
         False, {}),
        ("scripts/pr103_validate_reference.py", V,
         ["validate", "sha256", "TEMPLATE", "TEMPLATES", "PROVENANCE", "PROVENANCE_V2",
          "NULL_VALUE_PROVENANCE", "NULL_VALUE_PROVENANCE_V2", "VALUE_FIELDS", "ROW_FIELDS"],
         True, {}),
        ("src/evaluation_metrics.py", _module("evaluation_metrics"),
         ["evaluate", "aggregate", "ROW_FIELDS", "HEADER_FIELDS", "CORRECT", "INCORRECT", "MISSING",
          "UNVERIFIED"], False,
         {"payload/reference wrapper": "the evaluator reads plain values, so PR110 adds project() rather "
                                      "than editing the evaluator"}),
        ("scripts/pr109_dev_cohort.py", _module("pr109_dev_cohort"),
         ["ruled_geometry", "wilson", "coverage"], False,
         {"coverage": "bound to the development record shape, which only exists after a pipeline run; "
                      "PR110 re-expresses the same thirteen categories over the annotation"}),
        ("scripts/pr102_holdout_intake.py", _module("pr102_holdout_intake"),
         ["main"], False,
         {"intake": "one keyed references JSON and an all-or-nothing refusal; it cannot carry one "
                    "validator-checked reference per document, which is what PR110's readiness gate needs"}),
        ("benchmark_prep/annotation_tools.py", _module("annotation_tools", ROOT / "benchmark_prep"),
         ["FIELD_STATES", "main"], False,
         {"schema": "declares invoice-annotation-v1 with its own state vocabulary; adopting it would fork a "
                    "second reference vocabulary next to PR103's. Its two-annotator `finalize` is the model "
                    "for PR110's second_review rule."}),
    ]
    inventory = {}
    for name, module, attributes, modified, not_reused in probes:
        availability = {attribute: hasattr(module, attribute) if module is not None else False
                        for attribute in attributes}
        missing = sorted(key for key, value in availability.items() if not value)
        inventory[name] = {
            "exists": (ROOT / name).is_file(),
            "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).is_file() else None,
            "importable": module is not None,
            "reused_by_pr110": attributes,
            "all_reused_attributes_present": not missing,
            "missing_attributes": missing,
            "modified_by_pr110": modified,
        }
        if not_reused:
            inventory[name]["not_reused_by_pr110"] = not_reused
    return inventory


def _module(name: str, search: Path | None = None):
    """Import a module for the inventory probe; `search` covers directories outside src/ and scripts/."""
    if search is not None and str(search) not in sys.path:
        sys.path.append(str(search))
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - environment report
        return None


# --------------------------------------------------------------------------- safety measurement


def experimental_switches_off() -> dict:
    """Every experimental path read through the predicate production itself consults, as PR109 read them."""
    import dual_ocr_experiment
    import invoice_review
    import ocr_fast_path
    import table_geometry
    import table_recovery
    import table_shadow
    from ocr_config import OCR_CONFIG

    state = {
        "dual_ocr_experiment_enabled": dual_ocr_experiment.enabled(),
        "ocr_fast_path_enabled": ocr_fast_path.fast_path_enabled(),
        "table_shadow_enabled": table_shadow.shadow_enabled(),
        "table_recovery_enabled": table_recovery.enabled(),
        "table_geometry_enabled": table_geometry.enabled(),
        "review_ui_enabled": invoice_review.review_ui_enabled(),
    }
    state["all_off"] = not any(state.values())
    state["policy_version"] = OCR_CONFIG.version
    state["confidence_threshold"] = OCR_CONFIG.confidence
    return state


def intake_present() -> dict:
    """What the owner has actually supplied, measured without touching a pipeline."""
    invoices = invoice_files()
    references = reference_files()
    return {
        "invoice_directory": display(INVOICE_DIR),
        "invoice_directory_exists": INVOICE_DIR.is_dir(),
        "invoice_files": [path.name for path in invoices],
        "invoice_count": len(invoices),
        "reference_directory": display(REFERENCE_DIR),
        "reference_directory_exists": REFERENCE_DIR.is_dir(),
        "reference_files": [path.name for path in references],
        "reference_count": len(references),
    }


def gitignore_covered() -> dict:
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return {
        "rules_present": {rule: rule in rules for rule in GITIGNORE_RULES},
        "all_covered": all(rule in rules for rule in GITIGNORE_RULES),
    }
