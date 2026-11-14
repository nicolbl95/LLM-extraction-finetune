"""PR109 Phase 3-5, executed on the development corpus and labelled as such.

Phase 2 found 0 eligible independent documents, so there is no cohort to compose, annotate blindly or freeze:
the brief's Phase 3-5 are conditional on passing the intake gate. What can still be delivered without
pretending is the *same three audits applied to the documents the campaign already owns*, because those
audits answer two questions the owner has to settle before supplying anything - which of the twelve requested
categories the existing corpus can never cover, and how much of the existing ground truth is evidence-derived
rather than source-derived (the circularity PR102 found once already).

Three rules keep this honest:

* **Development-only labelling.** Every document measured here is a development or regression document by
   record (PR103's inventory, PR102's exposure audit). None of these numbers may be reported as
   generalization performance, and the label is repeated in each block.
* **Recorded facts before measurement, measurement before inference.** Language, country, genuine-vs-template
   and the documented derived/ambiguous notes are taken from `experiments/independent-validation-audit.json`
   and the frozen truth files. Only two axes are measured here: the ruled-table geometry, through
   production's own `_ruled_region` on production's own prepared images, and the confidence profile, recomputed
   from the frozen PR105B Tesseract bytes. The inference that PR107 already refused - that a lost accent is
   a match - is kept as a separate disclosed count, never merged into the match totals.
* **0 OCR calls and 0 model calls.** The pipeline is not re-run: the published PR105B payloads are scored with
   the production association-aware evaluator, which is how PR107 and PR108 reported their row counts.

No production file is read for modification, no reference is edited, nothing is written outside
results/product-review109/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr105b_harness as H  # noqa: E402
import pr106_harness as P  # noqa: E402

ARTIFACT_DIR = ROOT / "results" / "product-review109"
PR105B_CACHE = ROOT / "results" / "product-review105b" / "pr105b-ocr-cache.json"
PR107_TRACE = ROOT / "results" / "product-review107" / "pr107-phase2-trace-corpus.json"
AUDIT = ROOT / "experiments" / "independent-validation-audit.json"

EXPORTED_LINE_FIELDS = H.ROW_FIELDS  # description, quantity, unit_price, total_ht
HEADER_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")
# Literal words only, used to *report* which optional columns appear in recorded text. Nothing here decides.
TAX_WORDS = ("tva", "vat", "tax", "iva", "impuesto", "sales tax", "ht/u", "hors taxe")


# ------------------------------------------------------------------ geometry, measured through production


def ruled_geometry(path: Path) -> dict:
    """Does production's own ruled-region detector find a bordered table on this document?

    `recover_image` calls `_ruled_region(prepared_images["deskew"])` only when the two fast reads built no
    row at all, so this is the geometry the zero-row documents would be offered, not a new detector.
    """
    import numpy
    from PIL import Image

    import ocr_recovery
    from ocr_preprocessing import deskew, transform

    with Image.open(path) as source:
        rotated, angle = deskew(source.convert("RGB"))
        prepared = {family: transform(rotated, angle, family)[0] for family in ("deskew", "clahe")}

    out = {"angle_degrees": round(float(angle), 3), "prepared_size": list(prepared["deskew"].size)}
    for family, image in prepared.items():
        region = ocr_recovery._ruled_region(image)
        mask = ocr_recovery._grid_mask(numpy.asarray(image))
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 0]
        qualifying = [r for r in rectangles if r[2] > image.width / 2 and r[3] > 40]
        out[family] = {
            "ruled_region": [int(v) for v in region] if region else None,
            "ruled_region_area_ratio": round(
                (region[2] - region[0]) * (region[3] - region[1]) / (image.width * image.height), 4)
            if region else None,
            "grid_contours_found": len(rectangles),
            "grid_contours_qualifying_as_region": len(qualifying),
        }
    out["bordered_table_detected_by_production"] = any(
        out[family]["ruled_region"] for family in ("deskew", "clahe"))
    return out


# ------------------------------------------------------------------ recorded OCR evidence


def recorded_reads(cache: dict, name: str) -> dict:
    """Every line and token confidence the production acquisition recorded for this document.

    Rebuilt from the frozen Tesseract bytes, so it is the same evidence PR105B-PR108 replayed; 0 new calls.
    """
    from invoice_evidence import from_tsv

    lines: list[str] = []
    confidences: list[float] = []
    calls = 0
    for key in cache["documents"][name]["ocr_call_keys"]:
        data = cache["entries"].get(key)
        if data is None:
            continue
        calls += 1
        evidence = from_tsv(data)
        lines.extend(line.text for line in evidence.lines if line.text.strip())
        confidences.extend(
            token.confidence for token in evidence.tokens if str(token.text or "").strip())
    floor = _confidence_floor()
    return {
        "recorded_ocr_calls": calls,
        "recorded_text_lines": len(lines),
        "recorded_tokens": len(confidences),
        "mean_token_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "min_token_confidence": min(confidences) if confidences else None,
        "share_of_tokens_below_production_confidence_floor": round(
            sum(1 for value in confidences if value < floor) / len(confidences), 4) if confidences else None,
        "lines": lines,
        "normalised_lines": [_flat(line) for line in lines],
        "pr107_lines": [_flat(_pr107_normalize(line)) for line in lines],
        "numeric_tokens": numeric_tokens(lines),
    }


def _confidence_floor() -> int:
    from ocr_config import OCR_CONFIG

    return OCR_CONFIG.confidence


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).replace("\u00a0", " ")).strip().lower()


def _fold_accents(text: str) -> str:
    return "".join(character for character in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(character))


# ------------------------------------------------------------------ value matching


def numeric_forms(value) -> tuple[set[str], set[str], str]:
    """The printed forms of a number, split into the forms that prove the decimals and the ones that do not.

    191,40 may print as `191,40`, `191.40`, `191,4` or `191.4`; all four carry the decimal. `191` does not,
    so matching only `191` cannot be credited as the printed amount - it is reported as the truncation risk
    instead. A whole number such as a quantity of 3 prints as `3` and is credited.
    """
    number = float(value)
    two_dp = f"{number:.2f}"
    decimal_forms = {two_dp, two_dp.replace(".", ",")}
    for form in (two_dp, two_dp.replace(".", ",")):
        trimmed = form.rstrip("0").rstrip(".")
        if trimmed:
            decimal_forms.add(trimmed)
    bare_forms = {str(int(number))} if number.is_integer() else set()
    return decimal_forms, bare_forms, two_dp


def date_forms(value) -> list[str]:
    """Numeric printed forms of an ISO date, then the written-month forms in both campaign languages.

    A French invoice prints `12 juillet 2023`, and the production parser turns that into `2023-07-12`. If the
    matcher only knows the numeric forms, a correctly parsed date looks like a value the annotation could not
    have read, which would be a matcher artefact reported as a provenance finding.
    """
    year, month, day = str(value).split("-")
    forms = [str(value), f"{day}/{month}/{year}", f"{month}/{day}/{year}", f"{day}.{month}.{year}",
             f"{day}/{month}/{year[-2:]}", f"{month}/{day}/{year[-2:]}"]
    index = int(month)
    for name in MONTH_NAMES.get(index, []):
        forms += [f"{int(day)} {name} {year}", f"{name} {int(day)}, {year}",
                  f"{name} {day} {year}", f"{int(day)}. {name} {year}"]
    return forms


MONTH_NAMES = {
    1: ("janvier", "jan", "january", "jan"),
    2: ("février", "fevrier", "febr", "february", "feb"),
    3: ("mars", "mar", "march", "mar"),
    4: ("avril", "avr", "april", "apr"),
    5: ("mai", "may"),
    6: ("juin", "jun", "june", "jun"),
    7: ("juillet", "jul", "july", "jul"),
    8: ("août", "aout", "aug", "august", "aug"),
    9: ("septembre", "sept", "sep", "september", "sep"),
    10: ("octobre", "oct", "october", "oct"),
    11: ("novembre", "nov", "november", "nov"),
    12: ("décembre", "decembre", "dec", "december", "dec"),
}


def numeric_tokens(lines: list[str]) -> set[str]:
    """Every number-shaped token the recorded acquisition printed, with its separators left intact.

    Token equality, not substring containment, is what stops the reference value 191,40 from being credited
    to a line that happens to contain 1912. Whitespace is excluded from a token on purpose: the money columns
    print adjacent on one line, and allowing spaces inside a token merges `$250.00 2 $500.00` into one
    meaningless string.
    """
    out: set[str] = set()
    for line in lines:
        for token in re.findall(r"\d[\d.,/]*", line):
            cleaned = token.strip().strip(".,")
            if cleaned:
                out.add(cleaned)
                out.add(cleaned.replace(" ", ""))
    return out


def classify(field: str, value, reads: dict) -> dict:
    """Where this annotated value demonstrably exists in the recorded acquisition."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return {"category": "absent_in_reference", "matched_form": None,
                "would_match_only_if_accents_folded": False, "notes": []}
    if field in ("supplier_name", "description"):
        return classify_text(value, reads)
    if field == "invoice_date":
        return classify_date(value, reads)
    return classify_number(field, value, reads)


def classify_text(value, reads: dict) -> dict:
    """Three separable views, never merged: printed as-is, printed once both sides carry the PR107
    comparison form, and printed only if accents are folded (which the frozen convention refuses)."""
    normalized = _flat(value)
    pr107_form = _flat(_pr107_normalize(value))
    folded = _fold_accents(normalized)
    for line in reads["normalised_lines"]:
        if normalized in line:
            return {"category": "exact_in_recorded_ocr", "matched_form": value, "notes": []}
    for line in reads["pr107_lines"]:
        if pr107_form and pr107_form in line:
            return {"category": "match_after_pr107_normalisation",
                    "matched_form": _pr107_normalize(value),
                    "notes": [] if pr107_form == normalized else [
                        "the PR107 comparison form matches; the raw printed strings differ in "
                        "separator tokens, case or spacing"]}
    joined = " ".join(reads["pr107_lines"])
    if pr107_form and pr107_form in joined:
        return {"category": "match_only_across_a_line_break", "matched_form": _pr107_normalize(value),
                "notes": ["the text is in the recorded acquisition but only when consecutive lines are "
                          "joined - this is the multi-line description case, not a missing-value case"]}
    return {
        "category": "not_in_any_recorded_read", "matched_form": None,
        "would_match_only_if_accents_folded": any(folded in _fold_accents(line)
                                                 for line in reads["normalised_lines"]),
        "notes": [],
    }


def classify_number(field: str, value, reads: dict) -> dict:
    try:
        decimal_forms, bare_forms, two_dp = numeric_forms(value)
    except (TypeError, ValueError):
        return {"category": "not_in_any_recorded_read", "matched_form": None,
                "would_match_only_if_accents_folded": False, "notes": ["not a number"]}
    tokens = reads["numeric_tokens"]
    for form in sorted(decimal_forms, key=len, reverse=True):
        if form in tokens:
            return {"category": "exact_in_recorded_ocr", "matched_form": form, "notes": []}
    for form in sorted(bare_forms, key=len, reverse=True):
        if form in tokens:
            return {"category": "exact_in_recorded_ocr", "matched_form": form,
                    "notes": ["a whole value printed without decimals"]}
    integer_part = str(int(float(value)))
    if integer_part in tokens:
        return {"category": "not_in_any_recorded_read", "matched_form": None,
                "would_match_only_if_accents_folded": False,
                "notes": [f"only the integer part {integer_part} appears as a token, never a decimal form of "
                          f"{two_dp}, so the printed amount is not demonstrated"]}
    return {"category": "not_in_any_recorded_read", "matched_form": None,
            "would_match_only_if_accents_folded": False, "notes": []}


def classify_date(value, reads: dict) -> dict:
    """A date is credited only in a printed form, never from the bare year."""
    for form in date_forms(value):
        if form in reads["numeric_tokens"] or _flat(form) in " ".join(reads["normalised_lines"]):
            return {"category": "exact_in_recorded_ocr", "matched_form": form, "notes": []}
    corrupted = _date_from_a_corrupted_token(value, reads)
    if corrupted is not None:
        token, rationale = corrupted
        return {"category": "printed_only_as_a_corrupted_token", "matched_form": token, "notes": [
            "no faithful printed form of the date survives in the recorded acquisition; the annotated value "
            "is spelled only by a misread digit run, which production's own parser still resolves",
            f"production's recorded rationale for that token: {rationale}",
            "reported as its own class, never as clean printed evidence and never as circularity: the "
            "annotation is not the source of the export, but a lenient separator-free parse carries no "
            "evidence that the digits it happened to read are the digits that were printed",
        ]}
    year = str(value).split("-")[0]
    return {"category": "not_in_any_recorded_read", "matched_form": None,
            "would_match_only_if_accents_folded": False,
            "notes": [f"the year {year} appears but no full printed date form does"
                      if year in reads["numeric_tokens"] else "no printed date form appears at all"]}


def _date_from_a_corrupted_token(value, reads: dict) -> tuple | None:
    """Ask the shipped parser, not a matcher of my own, whether a misread digit run still spells this date.

    fr-mon-entreprise prints `Date : 24/02/2021` and every recorded read shows `2422021`: the separators and
    one digit are gone, so no syntactic form can bridge the gap and the only honest test is the rule the
    production code already applies to that exact string. The locale is left None because that is the
    strictest of the shipped branches - this can under-credit a date, never over-credit one.
    """
    from datetime import date

    from invoice_validation import parse_date

    target = date.fromisoformat(str(value))
    for token in sorted(reads["numeric_tokens"]):
        digits = re.sub(r"\D", "", token)
        if not 6 <= len(digits) <= 9 or digits == str(value).replace("-", ""):
            continue
        parsed, rationale = parse_date(token, None)
        if parsed == target:
            return token, rationale
    return None


def _pr107_normalize(text):
    from pr107_measure import normalize_description

    return normalize_description(text)


def derived_from_reference(field: str, value, row: dict, document: dict) -> bool:
    """Is this value implied by the rest of the reference alone? The brief's second annotation class."""
    if value is None:
        return False
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    if field == "total_ht" and row is not None:
        quantity, price = row.get("quantity"), row.get("unit_price")
        if quantity is not None and price is not None:
            return abs(float(quantity) * float(price) - target) <= 0.01
    if field in ("total_ht", "total_ttc") and row is None:
        lines = document.get("line_items") or []
        summed = sum(float(item[field]) for item in lines if item.get(field) is not None)
        if summed and abs(summed - target) <= max(0.05, 0.005 * summed):
            return True
        return False
    if field == "line_total" and row is not None:
        return False
    return False


# ------------------------------------------------------------------ one document


def audit_document(name: str, entry: dict, cache: dict, pr107: dict, audit: dict) -> dict:
    from PIL import Image

    reference = entry["reference"]
    reads = recorded_reads(cache, name)
    payload = cache["documents"][name]["payload"]
    path = Path(entry["path"])
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        file_block = {
            "bytes": path.stat().st_size, "mode": image.mode, "size": [image.width, image.height],
            "dpi": [round(float(v), 1) for v in dpi] if dpi else None,
            "megapixels": round(image.width * image.height / 1e6, 3),
        }

    truth = {}
    if str(entry["reference_source"]).endswith(".json") and (ROOT / entry["reference_source"]).exists():
        truth = json.loads((ROOT / entry["reference_source"]).read_text(encoding="utf-8"))
    document = truth.get("document") or {}
    metadata = truth.get("metadata") or {}
    audited = (audit.get("documents") or {}).get(path.name) or {}

    header_rows = [{"field": field, "classification": classify(field, reference.get(field), reads),
                    "derived_arithmetically": derived_from_reference(
                        field, reference.get(field), None, reference),
                    "equals_exported_value": reference.get(field) == (payload or {}).get(field)}
                   for field in HEADER_FIELDS]

    line_rows = []
    for index, row in enumerate(reference.get("line_items") or []):
        cells = []
        exported_row = (payload or {}).get("line_items", [{}])[index] if (
            payload and len(payload.get("line_items") or []) > index) else {}
        for field in EXPORTED_LINE_FIELDS:
            cells.append({
                "field": field,
                "classification": classify(field, row.get(field), reads),
                "derived_arithmetically": derived_from_reference(field, row.get(field), row, reference),
                "equals_exported_value": row.get(field) == exported_row.get(field),
                "reference_only_columns_present": sorted(
                    key for key in row if key not in EXPORTED_LINE_FIELDS and row.get(key) is not None),
            })
        line_rows.append({"row": index, "cells": cells,
                          "description_equals_exported_value": row.get("description")
                          == exported_row.get("description")})

    separators = sum(
        1 for row in (reference.get("line_items") or [])
        if row.get("description") and _flat(_pr107_normalize(row["description"]))
        != _flat(row["description"]))
    tax_words = sorted({word for word in TAX_WORDS
                        if any(word in line for line in reads["normalised_lines"])})
    gates = (pr107.get("documents", {}).get(name) or {}).get("gates") or {}

    return {
        "label": "DEVELOPMENT DOCUMENT - never a validation cohort",
        "kind": entry["kind"],
        "reference_source": entry["reference_source"],
        "provenance_recorded": {
            "country": audited.get("country"),
            "language": audited.get("language"),
            "genuine_business_invoice": truth.get("genuine_business_invoice"),
            "source_type": truth.get("source_type"),
            "layout_family": truth.get("layout_family"),
            "currency": document.get("currency"),
            "document_type": document.get("document_type"),
            "classification_in_pr102_audit": audited.get("classification"),
            "eligible_independent": audited.get("eligible_independent"),
            "language_country_basis": "recorded in the PR102 audit for the five real scans; the six frozen "
                                      "layouts are template renders with no issuing country, so their "
                                      "currency and document_type are printed-form facts, not strata",
        },
        "file": file_block,
        "geometry": ruled_geometry(path),
        "acquisition_profile": {key: value for key, value in reads.items()
                                if key not in ("lines", "normalised_lines", "pr107_lines",
                                               "numeric_tokens")},
        "structure": {
            "reference_rows": len(reference.get("line_items") or []),
            "exported_rows": len((payload or {}).get("line_items") or []),
            "reference_columns_beyond_the_exported_four": sorted({
                key for row in (reference.get("line_items") or []) for key in row
                if key not in EXPORTED_LINE_FIELDS}),
            "truth_document_tax_fields": sorted(
                key for key in document if re.search(r"tax|vat|tva", key, re.I)),
            "truth_metadata_keys": sorted(metadata),
            "documented_derived_values_in_truth": bool(metadata.get("derived")),
            "documented_ambiguous_values_in_truth": len(metadata.get("ambiguities") or []),
            "stage5_continuation_gate": gates.get("stage5_continuation_lines", {}),
            "stage7_row_gate": gates.get("stage7_row_decisions", {}),
            "reference_descriptions_bearing_a_separator_token": separators,
            "tax_vocabulary_present_in_recorded_ocr": tax_words,
        },
        "annotation_provenance": {"header": header_rows, "lines": line_rows},
    }


# ------------------------------------------------------------------ category coverage


def coverage(documents: dict) -> list[dict]:
    """Phase 3's twelve requested categories, counted over the corpus that actually exists."""

    def where(predicate):
        return sorted(name for name, record in documents.items() if predicate(name, record))

    def reference(name):
        return H.corpus()[name]["reference"]

    def truth_field(name, key):
        return documents[name]["provenance_recorded"].get(key)

    categories = [
        ("French invoices", where(
            lambda n, r: truth_field(n, "language") == "French" or truth_field(n, "currency") == "EUR")),
        ("British invoices", where(lambda n, r: truth_field(n, "country") == "United Kingdom")),
        ("American invoices", where(lambda n, r: truth_field(n, "country") == "United States")),
        ("Canadian invoices", where(lambda n, r: truth_field(n, "country") == "Canada")),
        ("Genuinely issued invoices (not templates)", where(
            lambda n, r: r["provenance_recorded"]["genuine_business_invoice"] is not False)),
        ("Bordered tables (production finds a ruled region)", where(
            lambda n, r: r["geometry"]["bordered_table_detected_by_production"])),
        ("Borderless tables (production finds no ruled region)", where(
            lambda n, r: not r["geometry"]["bordered_table_detected_by_production"])),
        ("Multi-line / multi-segment descriptions", where(
            lambda n, r: r["structure"]["reference_descriptions_bearing_a_separator_token"] > 0
            or any(int(value) > 0 for key, value in
                   r["structure"]["stage5_continuation_gate"].items() if key == "merged"))),
        ("Additional reference columns beyond the exported four", where(
            lambda n, r: bool(r["structure"]["reference_columns_beyond_the_exported_four"]))),
        ("Explicit tax breakdown in the reference", where(
            lambda n, r: (reference(n).get("total_ht") is not None
                          and reference(n).get("total_ttc") is not None
                          and reference(n)["total_ht"] != reference(n)["total_ttc"]))),
        ("No explicit tax breakdown (single total, or equal HT and TTC)", where(
            lambda n, r: reference(n).get("total_ttc") is None
            or reference(n).get("total_ht") == reference(n).get("total_ttc"))),
        ("Two or more line items", where(lambda n, r: len(reference(n).get("line_items") or []) >= 2)),
        ("Small text or low-confidence acquisition", where(
            lambda n, r: (r["acquisition_profile"]["mean_token_confidence"] or 100) < _confidence_floor()
            or (r["acquisition_profile"]["share_of_tokens_below_production_confidence_floor"] or 0) > 0.5)),
    ]
    return [{"category": label, "documents": names, "count": len(names),
             "underrepresented": len(names) < 2} for label, names in categories]


# ------------------------------------------------------------------ development metrics with uncertainty


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4)]


def metrics(documents_payload: dict, corpus: dict) -> dict:
    from evaluation_metrics import CORRECT, UNVERIFIED, aggregate, evaluate

    results, per_document = [], {}
    for name, payload in documents_payload.items():
        result = evaluate(payload or {}, corpus[name]["reference"])
        header = result["header"]
        exact = (
            result["rows"]["expected"] == result["rows"]["matched_total"]
            == result["rows"]["exported"]
            and result["fields"]["incorrect"] == 0
            and result["fields"]["missing"] == 0
            and result["fields"]["incorrect_association"] == 0
            and all(verdict in (CORRECT, UNVERIFIED) for verdict in header.values())
        )
        per_document[name] = {
            "rows": {key: value for key, value in result["rows"].items() if not isinstance(value, list)},
            "line_cells": {key: result["fields"][key] for key in
                           ("correct", "incorrect", "missing", "missing_in_matched_rows",
                            "missing_in_unrecovered_rows", "unverified", "incorrect_monetary",
                            "incorrect_association")},
            "cells_by_field": result["fields"]["by_field"],
            "header": header,
            "header_summary": result["header_summary"],
            "positional_cells": result["positional"]["cells"],
            "document_exact_match": exact,
        }
        results.append(result)

    field_totals = {key: sum(d["line_cells"][key] for d in per_document.values()) for key in
                    ("correct", "incorrect", "missing", "missing_in_matched_rows",
                     "missing_in_unrecovered_rows", "unverified", "incorrect_monetary",
                     "incorrect_association")}
    cells_by_field = {}
    for field in results[0]["fields"]["by_field"]:
        cells_by_field[field] = {verdict: sum(
            d["cells_by_field"][field][verdict] for d in per_document.values())
            for verdict in ("correct", "incorrect", "missing", "unverified")}
    counts = {
        "rows_correctly_matched": sum(d["rows"]["correctly_matched"] for d in per_document.values()),
        "rows_matched_total": sum(d["rows"]["matched_total"] for d in per_document.values()),
        "rows_expected": sum(d["rows"]["expected"] for d in per_document.values()),
        "rows_exported": sum(d["rows"]["exported"] for d in per_document.values()),
    }
    header_verdicts = {verdict: sum(d["header_summary"][verdict] for d in per_document.values())
                       for verdict in ("correct", "incorrect", "missing", "unverified")}
    return {
        "label": "DEVELOPMENT BASELINE - the published PR105B payloads scored offline; 0 OCR calls",
        "document_exact_match_definition": "every reference row matched and exported (expected = matched = "
                                           "exported), 0 incorrect and 0 missing line cells, 0 incorrect "
                                           "associations, and no header field scored incorrect",
        "per_document": per_document,
        "aggregate_from_the_evaluator": aggregate(results),
        "row_counts": counts,
        "line_cell_verdicts": field_totals,
        "line_cells_by_field": cells_by_field,
        "header_verdicts_pooled": header_verdicts,
        "header_accuracy_per_field": {
            field: {verdict: sum(1 for d in per_document.values() if d["header"][field] == verdict)
                    for verdict in ("correct", "incorrect", "missing", "unverified")}
            for field in HEADER_FIELDS},
        "exact_match_documents": sum(1 for d in per_document.values() if d["document_exact_match"]),
        "uncertainty": {
            "method": "Wilson score interval, 95 %, computed on the pooled development counts; the corpus is "
                      "11 documents, so these intervals are wide and are reported instead of any claim of "
                      "production reliability",
            "row_coverage": wilson(counts["rows_correctly_matched"], counts["rows_expected"]),
            "line_cells_correct_of_nonnull_reference_cells": wilson(
                field_totals["correct"], field_totals["correct"] + field_totals["incorrect"]
                + field_totals["missing"]),
            "header_fields_correct_of_scored": wilson(
                header_verdicts["correct"],
                header_verdicts["correct"] + header_verdicts["incorrect"] + header_verdicts["missing"]),
            "document_exact_match": wilson(
                sum(1 for d in per_document.values() if d["document_exact_match"]), len(per_document)),
        },
    }


# ------------------------------------------------------------------ provenance roll-up


def provenance_summary(documents: dict) -> dict:
    tally: dict[str, int] = {}
    circular, accent_only, derived, printed_elsewhere, degraded = [], [], [], [], []
    for name, record in documents.items():
        blocks = list(record["annotation_provenance"]["header"])
        blocks += [cell for row in record["annotation_provenance"]["lines"] for cell in row["cells"]]
        for block in blocks:
            category = block["classification"]["category"]
            tally[category] = tally.get(category, 0) + 1
            unexplained = (block.get("equals_exported_value")
                           and category in ("not_in_any_recorded_read", "match_only_across_a_line_break")
                           and not block.get("derived_arithmetically"))
            if category == "printed_only_as_a_corrupted_token":
                degraded.append({"document": name, "field": block["field"],
                                 "recorded_token": block["classification"]["matched_form"],
                                 "equals_exported_value": bool(block.get("equals_exported_value")),
                                 "note": block["classification"]["notes"][1]})
            elif unexplained:
                circular.append({"document": name, "field": block["field"]})
            elif block.get("equals_exported_value") and category == "not_in_any_recorded_read":
                printed_elsewhere.append({"document": name, "field": block["field"],
                                          "basis": "implied by the rest of the reference, so the export is "
                                                   "consistent with a derived value rather than a copied one"})
            if block["classification"].get("would_match_only_if_accents_folded"):
                accent_only.append({"document": name, "field": block["field"]})
            if block.get("derived_arithmetically"):
                derived.append({"document": name, "field": block["field"]})
    return {
        "classification_tally": dict(sorted(tally.items())),
        "annotated_values_printed_only_in_a_corrupted_recorded_form": {
            "count": len(degraded), "examples": degraded[:20],
            "note": "neither clean printed evidence nor circularity: the acquisition lost the separators (and "
                    "here a digit), and production's own separator-free branch turned the misread into the "
                    "annotated date. It is excluded from the exact-match tally because the printed form is not "
                    "faithful, and from the circularity tally because the export was demonstrably derived from "
                    "the recorded text rather than copied from the reference",
        },
        "annotated_values_that_equal_the_exported_value_and_are_in_no_recorded_read_and_are_not_derived": {
            "count": len(circular), "examples": circular[:20],
            "note": "this is PR102's circularity pattern: a reference that agrees with the output it is "
                    "supposed to judge, with neither an OCR line nor an arithmetic rule behind it, cannot be "
                    "evidence that the output is right",
        },
        "exported_values_absent_from_the_reads_but_implied_by_the_reference": {
            "count": len(printed_elsewhere), "examples": printed_elsewhere[:20],
            "note": "the brief's second annotation class: a value established by arithmetic rather than by "
                    "print is legitimate, and is separated here so it is never counted as circular",
        },
        "annotated_values_that_would_match_only_if_accents_were_folded": {
            "count": len(accent_only), "examples": accent_only[:20],
            "note": "reported separately and excluded from every match total, because the frozen "
                    "pr107-desc-norm-v1 convention refuses to fold accents: a lost accent is a different word",
        },
        "annotated_values_implied_by_reference_arithmetic": {"count": len(derived), "examples": derived[:20]},
        "second_reviewer": {
            "independent_second_annotation": 0,
            "note": "Phase 4 requires a second reviewer on blind independent documents; the development "
                    "references were frozen by the campaign itself and cannot be re-reviewed into independence",
        },
    }


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="pr109-phase3-5-development-cohort.json")
    parser.add_argument("--documents", nargs="*", default=None)
    args = parser.parse_args()

    target = P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name)
    cache = json.loads(PR105B_CACHE.read_text(encoding="utf-8"))
    pr107 = json.loads(PR107_TRACE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    corpus = P.documents()
    names = args.documents or list(corpus)

    documents = {name: audit_document(name, corpus[name], cache, pr107, audit) for name in names}
    scored = metrics({name: cache["documents"][name]["payload"] for name in names}, corpus)

    artifact = {
        "phase": "PR109 Phase 3-5 executed on the development corpus, labelled development-only",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": P.git_state()["branch"],
        "head": P.git_state()["head"][:7],
        "intake_gate": "BLOCKED_AWAITING_INDEPENDENT_INVOICES - 0 eligible of 12 required, so the real "
                       "Phase 3-5 cohort does not exist",
        "why_this_is_not_a_validation_cohort": [
            "all eleven documents are the tuning corpus: every rule decision since PR83 was taken against them",
            "the five real scans appear by supplier identity in tests/test_supplier_identity.py (PR102 audit)",
            "the six layouts are template renders, `genuine_business_invoice` false, so they cannot cover the "
            "FR/UK/US/CA country strata at all",
            "these metrics reproduce the published PR105B payloads and therefore match PR107 and PR108 exactly",
        ],
        "data_sources": {
            "published_payloads_and_frozen_ocr": PR105B_CACHE.relative_to(ROOT).as_posix(),
            "stage_gates": PR107_TRACE.relative_to(ROOT).as_posix(),
            "provenance": AUDIT.relative_to(ROOT).as_posix(),
            "truth_files": "the frozen PR103-schema references named per document",
        },
        "documents": documents,
        "category_coverage_vs_phase_3_request": coverage(documents),
        "annotation_provenance_audit": provenance_summary(documents),
        "development_baseline_metrics": scored,
        "constraints_observed": {
            "production_files_modified": 0,
            "frozen_references_edited": 0,
            "ocr_calls": 0,
            "model_calls": 0,
            "experimental_modes_enabled": 0,
            "independent_certification_made": 0,
            "network_calls": 0,
            "files_written_outside_the_workspace": 0,
        },
    }
    P.write_json(target, artifact)

    print("documents audited:", len(documents))
    print("provenance tally:", json.dumps(artifact["annotation_provenance_audit"]["classification_tally"]))
    print("rows correct/matched/expected:", scored["row_counts"]["rows_correctly_matched"],
          scored["row_counts"]["rows_matched_total"], scored["row_counts"]["rows_expected"])
    print("line cells:", json.dumps(scored["line_cell_verdicts"]))
    print("uncertainty:", json.dumps(scored["uncertainty"]))
    print("categories underrepresented:",
          [block["category"] for block in artifact["category_coverage_vs_phase_3_request"]
           if block["underrepresented"]])
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
