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
import pr110_authority as J  # noqa: E402

INVOICE_DIR = ROOT / "benchmark_prep" / "invoices"
REFERENCE_DIR = ROOT / "benchmark_prep" / "references"
ARTIFACT_DIR = ROOT / "results" / "product-review110"
TEMPLATE = ROOT / "benchmark_prep" / "pr110-reference-template.json"

TARGET_DOCUMENTS = I.TARGET_DOCUMENTS
# PR167: the protocol is a composition, not a count. The quota says which countries the campaign measures -
# four invoices billed under each of the three conventions it was built for - and the target count is its
# consequence, so `select_cohort()` refuses a quota that does not add up rather than silently measuring less.
COHORT_QUOTA = {"FR": 4, "UK": 4, "US": 4}
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


# --------------------------------------------------------------------------- cohort selection (PR167)


def region_of(record: dict) -> str | None:
    """The billing country the owner declared: `document.region`, the only region field the validator pins.

    `coverage()` above answers "which invoices are French?" from language *or* currency, which describes a
    corpus well and spends a quota badly - an invoice priced in EUR is not therefore a French one. The
    validator refuses a reference whose `document.region` is outside its closed vocabulary, so this is the
    only declared value a country slot can be filled from, and a document with no region is not an
    unclassified one: it is an invalid reference that never reaches a selection.
    """
    document = (record.get("annotation") or {}).get("document") or {}
    return document.get("region")


def _selection_key(record: dict) -> tuple:
    return (str(record.get("id") or ""), str(record.get("sha256") or ""))


def select_cohort(records: list[dict], quota: dict[str, int] | None = None,
                  *, target: int | None = None) -> dict:
    """The exact documents the evaluation will run, chosen deterministically from the eligible ones.

    Twelve is not the protocol; four invoices billed under each of the three conventions the campaign was
    built for is. Within a country the candidates are ordered by (document id, document sha256) - never by
    the order `rglob` returns them - and the first `quota[region]` are taken, so the same intake and the same
    sealed references always select the same cohort and a later run can be diffed against the freeze.

    Candidates beyond a country's quota are kept and named as a reserve, which is what PR110's intake was
    designed to allow; documents whose region carries no quota (CA, other) are kept out of the cohort rather
    than used to patch a shortfall, because a Canadian invoice measured as French would be reported as one.
    """
    quota = dict(COHORT_QUOTA if quota is None else quota)
    target = TARGET_DOCUMENTS if target is None else target
    candidates: dict[str, list[dict]] = {}
    off_quota: dict[str, list[str]] = {}
    regionless: list[str] = []
    for record in records:
        region = region_of(record)
        if region is None:
            regionless.append(str(record.get("id")))
        elif region in quota:
            candidates.setdefault(region, []).append(record)
        else:
            off_quota.setdefault(str(region), []).append(str(record.get("id")))

    selected: list[dict] = []
    reserve: list[dict] = []
    counts: dict[str, int] = {}
    missing: dict[str, int] = {}
    for region in sorted(quota):
        ordered = sorted(candidates.get(region) or [], key=_selection_key)
        taken, extra = ordered[:quota[region]], ordered[quota[region]:]
        selected += taken
        counts[region] = len(taken)
        if len(taken) < quota[region]:
            missing[region] = quota[region] - len(taken)
        for record in extra:
            reserve.append({"id": str(record.get("id")), "region": region, "sha256": record.get("sha256"),
                            "reason": f"a {region} candidate beyond the quota of {quota[region]}; "
                                      "not evaluated, kept as a named reserve"})
    for region in sorted(off_quota):
        for document_id in sorted(off_quota[region]):
            reserve.append({"id": document_id, "region": region, "sha256": None,
                            "reason": f"declared region {region!r} carries no quota in this protocol; "
                                      "it cannot replace a required invoice"})

    ordered_selected = sorted(selected, key=lambda record: (str(region_of(record)), ) + _selection_key(record))
    return {
        "quota": quota,
        "required_total": target,
        # the target is a consequence of the quota, so a protocol that no longer adds up is a finding, not a
        # cohort: 4+4+4 is 12 and a selection asked for 11 documents would silently drop one country
        "protocol_consistent": sum(quota.values()) == target,
        "counts": counts,
        "missing": missing,
        "candidates": {region: sorted(str(record.get("id")) for record in candidates.get(region) or [])
                       for region in sorted(quota)},
        "off_quota_candidates": {region: names for region, names in sorted(off_quota.items())},
        "without_a_declared_region": sorted(regionless),
        "selected": ordered_selected,
        "selected_ids": [str(record.get("id")) for record in ordered_selected],
        "reserve": sorted(reserve, key=lambda entry: (entry["region"], entry["id"])),
        "selection_sha256": digest([(str(region_of(record)), ) + _selection_key(record)
                                    for record in ordered_selected]),
        "ordered_by": "document id, then document sha256, within each declared region - never the order the "
                      "files were found in",
    }


def cohort_variant_pairs(records: list[dict], *, geometry: bool = True) -> dict:
    """Cohort members that are two renderings of one invoice, decided with the campaign's existing tests.

    Nothing new is measured here. An identical `document.sha256` or an identical `document.id` is one document
    by the campaign's own record; a homography at `pr110_authority`'s unchanged thresholds is one page
    re-rendered; a pair inside `PERCEPTUAL_MAX_DISTANCE` bits, or one whose correspondence clears
    `REVIEW_INLIERS` but not the conclusive band, is the resemblance PR164 chose to hand to a human rather
    than assert. Proven pairs make the cohort wrong; unconfirmed ones make it unproven, and a twelve-document
    cohort that cannot show its members are twelve different invoices has not been formed yet.
    """
    proven: list[dict] = []
    unconfirmed: list[dict] = []
    errors: list[dict] = []
    pairs = 0
    signatures: dict[str, dict | None] = {}

    def signature_of(record):
        name = str(record.get("id"))
        if name not in signatures:
            image = record.get("image")
            signatures[name] = I.perceptual(Path(image)) if image is not None and Path(image).is_file() else None
        return signatures[name]

    for index, left in enumerate(records):
        for right in records[index + 1:]:
            pairs += 1
            entry = {"pair": [str(left.get("id")), str(right.get("id"))],
                     "regions": [region_of(left), region_of(right)]}
            reasons = []
            if left.get("sha256") and left.get("sha256") == right.get("sha256"):
                reasons.append("identical document sha256")
            if left.get("id") and str(left.get("id")) == str(right.get("id")):
                reasons.append("the same document id is annotated twice")
            if reasons:
                proven.append({**entry, "basis": "; ".join(reasons)})
                continue
            if not geometry:
                continue
            first, second = signature_of(left), signature_of(right)
            if not first or not second:
                errors.append({**entry, "error": "no page signature could be computed for one of the two"})
                continue
            distance = min([value for value in (I.hamming(first.get("ahash"), second.get("ahash")),
                                                I.hamming(first.get("dhash"), second.get("dhash")))
                            if value is not None] or [None])
            if distance is None:
                errors.append({**entry, "error": "neither page carries an image hash "
                                                 "(a PDF has no page image to compare): the pair is unproven"})
                continue
            entry["hash_distance"] = distance
            if distance > J.GEOMETRY_PREFILTER_BITS:
                continue
            try:
                found = J.correspondence(Path(left["image"]), Path(right["image"]))
            except Exception as error:  # an unusable detector holds the cohort; it never clears it
                errors.append({**entry, "error": f"{type(error).__name__}: {error}"})
                continue
            if not found:
                errors.append({**entry, "error": "the geometric correspondence could not be computed"})
                continue
            fraction = found["inliers"] / max(1, found["good"])
            entry.update({"inliers": found["inliers"], "good": found["good"], "fraction": round(fraction, 3)})
            if found["inliers"] >= J.MIN_INLIERS and fraction >= J.MIN_INLIER_FRACTION:
                proven.append({**entry, "basis": f"one homography explains {found['inliers']} of "
                                                 f"{found['good']} correspondences at scale {found['scale']}"})
            elif found["inliers"] >= J.REVIEW_INLIERS or distance <= I.PERCEPTUAL_MAX_DISTANCE:
                unconfirmed.append({**entry, "basis": "a real but inconclusive correspondence, or a "
                                                     f"{distance}-bit resemblance: two invoices issued from "
                                                     "one template can reach this state and need the owner"})
    return {
        "computed": bool(geometry),
        "pairs_tested": pairs,
        "proven": sorted(proven, key=lambda item: item["pair"]),
        "unconfirmed": sorted(unconfirmed, key=lambda item: item["pair"]),
        "errors": sorted(errors, key=lambda item: item["pair"]),
        "thresholds": {"similarity_bits": I.PERCEPTUAL_MAX_DISTANCE,
                       "geometry_prefilter_bits": J.GEOMETRY_PREFILTER_BITS,
                       "min_inliers": J.MIN_INLIERS, "min_inlier_fraction": J.MIN_INLIER_FRACTION,
                       "review_inliers": J.REVIEW_INLIERS},
        "note": "thresholds are PR164's, read from pr110_authority rather than restated; a pair that only "
                "resembles is held for the owner, exactly as the intake screen holds a resemblance",
    }


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
