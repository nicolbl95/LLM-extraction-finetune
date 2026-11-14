"""PR103 Phase 3: emit a blind annotation template and validate a submitted reference before freezing.

The validator enforces the rules that make a reference trustworthy: only printed values, provenance on every
field, absent distinguished from illegible, no product row for shipping or tax, printed inconsistencies
recorded rather than reconciled, no OCR output seen before annotation, disagreements resolved, and the image
hash matching the file on disk. A reference that fails any rule is refused, not silently repaired.

**PR110 added version 2 of the schema, additively.** A `version: 1` reference is checked by exactly the rules
PR103 published and produces exactly the problem strings PR103 produced - that equivalence is pinned by
`tests/test_pr110_benchmark_prep.py`. Version 2 adds the two provenance states PR109 needed to classify the
truth it inherited (`derived` with an arithmetic basis, `ambiguous` with a note), the page map a multi-page
invoice needs, and a recorded second review. No v1 rule was relaxed to obtain them.

No network, no OCR, no pipeline run. Writes only where asked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "pr103": ROOT / "benchmark_prep" / "pr103-reference-template.json",
    "pr110": ROOT / "benchmark_prep" / "pr110-reference-template.json",
}
TEMPLATE = TEMPLATES["pr103"]
PROVENANCE = {"printed", "absent", "illegible"}
PROVENANCE_V2 = PROVENANCE | {"derived", "ambiguous"}
NULL_VALUE_PROVENANCE = {"absent", "illegible"}
NULL_VALUE_PROVENANCE_V2 = NULL_VALUE_PROVENANCE | {"ambiguous"}
VALUE_FIELDS = ("supplier_name", "invoice_date", "currency", "total_ht", "total_ttc")
ROW_FIELDS = ("description", "quantity", "unit_price", "total_ht")
REGIONS_V1 = ("FR", "UK", "US", "other")
INPUT_KINDS_V1 = ("scanned_image", "native_pdf", "photograph")
QUALITY_V2 = ("clear", "small_text", "low_contrast", "skewed_or_cropped", "other")


def _or_list(values: tuple[str, ...]) -> str:
    """`a, b or c` - and for the v1 vocabulary it reproduces PR103's message string verbatim."""
    return ", ".join(values[:-1]) + " or " + values[-1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def validate(reference: dict, image: Path | None = None) -> list[str]:
    """Return every rule violation. An empty list means the reference may be frozen."""
    problems = []
    version = reference.get("version") or 1
    v2 = version >= 2
    states = PROVENANCE_V2 if v2 else PROVENANCE
    null_states = NULL_VALUE_PROVENANCE_V2 if v2 else NULL_VALUE_PROVENANCE
    regions = REGIONS_V1 + (("CA",) if v2 else ())
    input_kinds = INPUT_KINDS_V1 + (("mixed",) if v2 else ())
    document = reference.get("document") or {}
    body = reference.get("reference") or {}
    freeze = reference.get("freeze") or {}

    for field in ("id", "source_file", "sha256", "language", "region", "input_kind", "annotation_date"):
        if not document.get(field):
            problems.append(f"document.{field} is required")
    if document.get("ocr_output_seen_before_annotation") is not False:
        problems.append("document.ocr_output_seen_before_annotation must be false: annotation is blind")
    if not document.get("annotators"):
        problems.append("document.annotators must name at least one annotator")
    permission = document.get("owner_permission") or {}
    if permission.get("authorized_for_local_evaluation") is not True:
        problems.append("document.owner_permission.authorized_for_local_evaluation must be true")
    if not permission.get("evidence"):
        problems.append("document.owner_permission.evidence is required")
    if document.get("region") not in set(regions):
        problems.append(f"document.region must be {_or_list(regions)}")
    if document.get("input_kind") not in set(input_kinds):
        problems.append(f"document.input_kind must be {_or_list(input_kinds)}")
    for index, disagreement in enumerate(document.get("disagreements") or []):
        if not disagreement.get("resolution"):
            problems.append(f"document.disagreements[{index}] has no resolution")

    if v2:
        pages = document.get("pages")
        if not isinstance(pages, list) or not pages or not all(
                isinstance(page, int) and page >= 1 for page in pages):
            problems.append("document.pages must be a non-empty list of 1-based page numbers "
                            "covering this invoice and no other")
        if document.get("quality_declaration") not in set(QUALITY_V2) | {None}:
            problems.append(f"document.quality_declaration must be {_or_list(QUALITY_V2)} or null")
        review = document.get("second_review") or {}
        if review.get("performed") is None:
            problems.append("document.second_review.performed must be recorded true or false; "
                            "PR109 found 0 of 11 inherited references with a second reviewer")
        if document.get("genuine_business_invoice") not in (True, False):
            problems.append("document.genuine_business_invoice must be recorded true or false: PR109 could "
                            "only guess whether a cohort document was a template")
        if review.get("performed") is True and not review.get("reviewer"):
            problems.append("document.second_review.reviewer is required when a second review was performed")
        if review.get("performed") is True and review.get("disagreements_resolved") is not True:
            problems.append("document.second_review.disagreements_resolved must be true when a second "
                            "review was performed")

    if image is not None:
        if not image.exists():
            problems.append(f"image not found: {image}")
        elif document.get("sha256") != sha256(image):
            problems.append("document.sha256 does not match the image on disk")

    def check_field(where: str, field: dict, required: bool = True):
        if not isinstance(field, dict):
            problems.append(f"{where} must be an object with value and provenance")
            return
        provenance = field.get("provenance")
        if provenance not in states:
            problems.append(f"{where}.provenance must be one of {sorted(states)}")
        if provenance == "printed" and field.get("value") is None:
            problems.append(f"{where} is marked printed but has no value")
        if provenance in null_states and field.get("value") is not None:
            problems.append(f"{where} is marked {provenance} but carries a value")
        if required and provenance is None:
            problems.append(f"{where}.provenance is required")
        if provenance == "derived" and not str(field.get("derived_from") or "").strip():
            problems.append(f"{where} is marked derived but records no derived_from arithmetic basis")
        if provenance == "ambiguous" and not str(field.get("note") or "").strip():
            problems.append(f"{where} is marked ambiguous but records no note explaining what is undecidable")

    for field in VALUE_FIELDS:
        check_field(f"reference.{field}", body.get(field))
    for field in ("rate", "amount"):
        tax = body.get("tax") or {}
        if field in tax and not isinstance(tax.get(field), dict):
            check_field(f"reference.tax.{field}", tax.get(field), required=False)
        elif v2 and field in tax:
            # PR103 left an object-form tax.rate/tax.amount unchecked; version 2 closes that gap and
            # version 1 keeps the published behaviour, including the gap.
            check_field(f"reference.tax.{field}", tax.get(field), required=False)
    shipping = body.get("shipping")
    if shipping is not None:
        check_field("reference.shipping", shipping, required=False)

    rows = body.get("line_items")
    if not isinstance(rows, list) or not rows:
        problems.append("reference.line_items must be a non-empty list")
        rows = []
    for index, row in enumerate(rows):
        for field in ROW_FIELDS:
            check_field(f"reference.line_items[{index}].{field}", row.get(field))
        if row.get("is_product_or_service") is not True:
            problems.append(f"reference.line_items[{index}] is not a product/service row: "
                            "shipping, discounts and tax are recorded separately")
        quantity = number((row.get("quantity") or {}).get("value"))
        price = number((row.get("unit_price") or {}).get("value"))
        total = number((row.get("total_ht") or {}).get("value"))
        if quantity is not None and price is not None and total is not None:
            if abs(quantity * price - total) > Decimal("0.02"):
                described = any(str(index) in str(item) for item in body.get("document_inconsistencies") or [])
                if not described:
                    problems.append(
                        f"reference.line_items[{index}] quantity x unit_price != printed total; "
                        "record the printed inconsistency in document_inconsistencies instead of reconciling it")

    if not freeze.get("reference_sha256"):
        problems.append("freeze.reference_sha256 is required")
    if freeze.get("frozen_before_pipeline_run") is not True:
        problems.append("freeze.frozen_before_pipeline_run must be true")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, help="write a blank template for one document")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--validate", type=Path, help="a filled reference JSON to check")
    parser.add_argument("--image", type=Path, help="the invoice image, to verify the recorded hash")
    args = parser.parse_args()

    if args.emit:
        if args.emit.exists():
            raise SystemExit(f"refusing to overwrite {args.emit}")
        template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        if args.document_id:
            template["document"]["id"] = args.document_id
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("written blank template", args.emit)
        return 0

    if args.validate:
        reference = json.loads(args.validate.read_text(encoding="utf-8"))
        problems = validate(reference, args.image)
        if problems:
            print("REFUSED — the reference may not be frozen:")
            for problem in problems:
                print("  -", problem)
            return 1
        print("reference accepted; freeze it with scripts/pr102_holdout_intake.py --intake")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
