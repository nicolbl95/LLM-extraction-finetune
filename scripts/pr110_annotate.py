"""PR110 Phase 3: the owner's reference-annotation tool - one guided CLI, no web app, no predictions.

Why a CLI and not an interface: PR109 showed that the bottleneck is not data entry but the *provenance* an
entry needs. `benchmark_prep/annotation_tools.py` already implements a full two-annotator tool, but on its own
`invoice-annotation-v1` vocabulary, which would fork a second reference schema next to PR103's; PR110 therefore
keeps PR103's schema, adds the two states PR109 had to reconstruct by hand (`derived`, `ambiguous`) and drives
them from the command line the campaign already uses. A browser interface would add a server, a port and a
second place to leak invoice text, for no gain in rigor.

What it guarantees:

* **Blind by construction.** This module never imports the pipeline, OCR, or a model, so no prediction can be
  shown while a reference is being written. `--check` and `--seal` re-run the published PR103/PR110 validator.
* **Provenance is mandatory.** Every field is asked as value + state (printed / derived / absent / illegible /
  ambiguous); a state that carries no value forces `null`, so "genuinely unknown" is recorded as unverified
  rather than guessed.
* **Nothing is frozen silently.** `--seal` refuses while the validator reports any problem, then records the
  reference and body digests and the image hash.

Sub-commands: `new`, `guide`, `show`, `check`, `seal`, `summary` (positional, then the document id).
Files are written only under `benchmark_prep/references/`, which `.gitignore` excludes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import pr109_intake as I  # noqa: E402
import pr110_harness as K  # noqa: E402

STATES = K.PROVENANCE
NUMERIC = re.compile(r"^-?[\d .,]+(?:€|$)")


# --------------------------------------------------------------------------- value parsing


def parse_value(text: str):
    """Numbers become floats (French `1 234,56` included); everything else stays the printed string."""
    cleaned = str(text).strip()
    candidate = cleaned.replace("\u00a0", "").replace(" ", "")
    if NUMERIC.match(candidate) or re.match(r"^-?\d+(?:[.,]\d+)?$", candidate):
        if "," in candidate and "." in candidate:
            candidate = candidate.replace(".", "").replace(",", ".") if candidate.rfind(".") > candidate.rfind(
                ",") else candidate.replace(",", ".").replace(".", "", candidate.count(".") - 1)
        elif "," in candidate:
            candidate = candidate.replace(",", ".")
        try:
            return float(candidate)
        except ValueError:
            return cleaned
    return cleaned


def _as_int_list(text: str) -> list[int]:
    return [int(part) for part in re.findall(r"\d+", str(text))]


# --------------------------------------------------------------------------- the walkthrough


DOCUMENT_STEPS = (
    ("source_origin", "Where did this invoice come from? (a short description, not the file path)", str),
    ("genuine_business_invoice", "Is it an invoice a business actually issued, not a template or sample? "
                                 "(y/n)", lambda text: text.strip().lower() in ("y", "yes")),
    ("owner_permission.authorized_for_local_evaluation",
     "May this document be evaluated locally on this machine? (y/n)",
     lambda text: text.strip().lower() in ("y", "yes")),
    ("owner_permission.evidence", "What gives permission to evaluate it locally? (e.g. 'it is my own "
                                  "company's document', 'the client agreed by email on 2026-11-01')", str),
    ("language", "Which language is it printed in?", str),
    ("region", "Which VAT/invoicing convention: FR, UK, US, CA or other?", str),
    ("input_kind", "How was it acquired: scanned_image, native_pdf, photograph or mixed?", str),
    ("pages", "Which pages of the file does this invoice occupy? (e.g. 1, or 1-2)", _as_int_list),
    ("page_count_observed", "How many pages does the file have in total?", int),
    ("quality_declaration", "Image quality: clear, small_text, low_contrast, skewed_or_cropped or other?", str),
    ("annotators", "Who is annotating? (name, comma separated)",
     lambda text: [part.strip() for part in text.split(",") if part.strip()]),
    ("annotation_date", "Date of this annotation (YYYY-MM-DD):", str),
    ("second_review.performed", "Did a second person review it? (y/n)",
     lambda text: text.strip().lower() in ("y", "yes")),
    ("second_review.reviewer", "Who reviewed?", str),
    ("second_review.disagreements_resolved", "Were all disagreements resolved? (y/n)",
     lambda text: text.strip().lower() in ("y", "yes")),
)

HEADER_PROMPTS = (
    ("supplier_name", "The supplier name exactly as printed"),
    ("invoice_date", "The invoice date exactly as printed (keep the printed form)"),
    ("currency", "The currency code printed (EUR, GBP, USD, CAD...)"),
    ("total_ht", "The printed net total (HT / subtotal / before tax)"),
    ("total_ttc", "The printed grand total (TTC / total due / after tax)"),
    ("tax.rate", "The tax rate exactly as printed"),
    ("tax.amount", "The tax amount exactly as printed"),
    ("shipping", "The shipping/carriage line exactly as printed"),
)

ROW_PROMPTS = (
    ("description", "the line description exactly as printed"),
    ("quantity", "the printed quantity (or hours)"),
    ("unit_price", "the printed unit price"),
    ("total_ht", "the printed line total"),
)

REQUIRED_FIELDS = tuple(path for path, _ in HEADER_PROMPTS)

HELP = (
    "state: p=printed, d=derived by arithmetic, a=absent (not on the document), "
    "u=illegible, m=ambiguous (readable but undecidable); 's' leaves the field unannotated"
)


def ask(prompt: str, reader) -> str:
    return str(reader(prompt)).strip()


def field_answer(path: str, description: str, reader, annotation: dict) -> bool:
    """One annotated field. Returns True when the field was filled, False when the owner skipped it."""
    print(f"\n{path} - {description}")
    print(HELP)
    state = ask("state [p/d/a/u/m/s]: ", reader).lower()
    if state in ("", "s"):
        print("  left unannotated: the validator will report the missing provenance")
        return False
    state = {"p": "printed", "d": "derived", "a": "absent", "u": "illegible", "m": "ambiguous"}.get(state, state)
    if state not in STATES:
        print(f"  unknown state {state!r}: recorded as ambiguous, to be corrected with --check")
        state = "ambiguous"
    if state in K.NULL_PROVENANCE:
        note = "" if state != "ambiguous" else ask("what is undecidable here? ", reader)
        K.set_field(annotation, path, None, state, note=note or None)
        print(f"  {path} = null ({state})")
        return True
    value = parse_value(ask("value: ", reader))
    derived_from = ask("which printed values and what arithmetic? ", reader) if state == "derived" else None
    K.set_field(annotation, path, value, state, derived_from=derived_from)
    print(f"  {path} = {value!r} ({state})")
    return True


def guide(annotation: dict, reader, *, rows: bool = True, redo: bool = False) -> dict:
    """Fill an annotation through the walkthrough. `reader(prompt) -> str` makes it testable without stdin.

    Answered items are skipped unless `redo`, so the owner can leave a session and finish later; the guide is
    a conversation, not a form that has to be filled in one sitting.
    """
    document = annotation["document"]
    print("=" * 78)
    print(f"Document {document.get('id')} - provenance first, values second. {K.TARGET_DOCUMENTS} documents "
          "are needed in total.")
    print("Nothing here shows extraction output: the reference is written from the printed document alone.")
    print("=" * 78)
    for path, question, cast in DOCUMENT_STEPS:
        if path.startswith("second_review.") and not document.get("second_review", {}).get("performed"):
            if path.endswith("reviewer") or path.endswith("disagreements_resolved"):
                continue
        if not redo and _filled(_lookup(document, path)):
            print(f"  {path} already recorded: {_lookup(document, path)}")
            continue
        answer = ask(question + ": ", reader)
        if not answer:
            print(f"  {path}: empty - the validator will report it as missing provenance")
            continue
        try:
            value = cast(answer)
        except ValueError:
            print(f"  {path}: {answer!r} is not a valid answer, left empty")
            continue
        _put(document, path, value)

    for path, description in HEADER_PROMPTS:
        if not redo and _get(annotation["reference"], path).get("provenance"):
            print(f"  {path} already annotated as {_get(annotation['reference'], path)['provenance']}")
            continue
        field_answer(path, description, reader, annotation)

    if rows:
        print("\nLine items: only products or services. Shipping, discounts and tax belong in their own "
              "fields, never in a row.")
        index = 0
        while True:
            if ask(f"add line item {index}? [y/n]: ", reader).lower() not in ("y", "yes"):
                break
            index = K.add_row(annotation)
            for path, description in ROW_PROMPTS:
                field_answer(f"line_items[{index}].{path}", description, reader, annotation)
            reference_number = ask("an order/reference number printed on this line? (blank if none): ", reader)
            if reference_number:
                annotation["reference"]["line_items"][index]["reference_number"] = parse_value(reference_number)
            index += 1
    return annotation


def _put(document: dict, path: str, value) -> None:
    head, _, tail = path.partition(".")
    if not tail:
        document[path] = value
        return
    document.setdefault(head, {})[tail] = value


# --------------------------------------------------------------------------- reporting


def missing_provenance(annotation: dict) -> list[str]:
    """Every field the schema requires an answer for that carries no provenance yet."""
    body = annotation.get("reference") or {}
    gaps = [path for path in REQUIRED_FIELDS if not _get(body, path).get("provenance")]
    for index, row in enumerate(body.get("line_items") or []):
        gaps += [f"line_items[{index}].{field}" for field in K.ROW_FIELDS
                 if not (row.get(field) or {}).get("provenance")]
    return gaps


def _lookup(container: dict, path: str):
    """The raw value at a dotted path, or None when nothing is recorded there."""
    node = container
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _get(container: dict, path: str) -> dict:
    """The {value, provenance} object at a dotted path; an empty dict when nothing is there."""
    value = _lookup(container, path)
    return value if isinstance(value, dict) else {}


def _filled(value) -> bool:
    """A recorded answer: false and zero count as answers, blanks do not."""
    return not (value is None or value == "" or value == {} or value == [])


def summary(annotation: dict) -> dict:
    projected = K.project(annotation)
    problems = K.validate(annotation)
    return {
        "id": (annotation.get("document") or {}).get("id"),
        "version": annotation.get("version"),
        "provenance_counts": projected["counts"]["by_provenance"],
        "not_annotated": projected["counts"]["not_annotated"],
        "scored_values": projected["counts"]["scored_values"],
        "left_unverified": projected["counts"]["left_unverified"],
        "line_items": len((annotation.get("reference") or {}).get("line_items") or []),
        "missing_provenance": missing_provenance(annotation),
        "content_problems": [problem for problem in problems if not problem.startswith("freeze.")],
        "problems": problems,
    }


# --------------------------------------------------------------------------- file handling


def reference_path(name: str, directory: Path) -> Path:
    filename = name if name.endswith(".json") else f"{name}.json"
    return directory / filename


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, annotation: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("new", "guide", "show", "check", "seal", "summary"))
    parser.add_argument("name", help="document id or reference file (in benchmark_prep/references/)")
    parser.add_argument("--invoice", type=Path, default=None,
                        help="the invoice file, for --new (to record its hash) and --seal")
    parser.add_argument("--references", type=Path, default=K.REFERENCE_DIR)
    parser.add_argument("--pages", default="1")
    parser.add_argument("--redo", action="store_true",
                       help="with guide: ask every question again, even the ones already answered")
    args = parser.parse_args(argv)

    directory = args.references
    if args.command == "new":
        target = reference_path(args.name, directory)
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        pages = _as_int_list(args.pages)
        annotation = K.blank(args.name, args.invoice.name if args.invoice else None, pages)
        if args.invoice is not None:
            signature = I.perceptual(args.invoice)
            annotation["document"]["sha256"] = signature["sha256"]
            annotation["document"]["image_resolution_dpi"] = signature.get("dpi")
            annotation["document"]["pages"] = annotation["document"]["pages"] or [1]
            annotation["document"]["page_count_observed"] = signature.get("pages") or 1
        save(target, annotation)
        print("blank reference created:", target)
        print("next: python scripts/pr110_annotate.py guide", args.name)
        return 0

    path = reference_path(args.name, directory)
    if not path.is_file():
        raise SystemExit(f"no reference named {args.name} in {directory}")
    annotation = load(path)

    if args.command == "guide":
        guide(annotation, input, redo=args.redo)
        save(path, annotation)
        report = summary(annotation)
        print(f"\nsaved {path}")
        print(f"fields annotated: {report['scored_values']} | unverified: {report['left_unverified']} "
              f"| line items: {report['line_items']}")
        if report["missing_provenance"] or report["content_problems"]:
            print("still open before you can seal:")
            for gap in report["missing_provenance"]:
                print("  - no provenance:", gap)
            for problem in report["content_problems"]:
                print("  -", problem)
            print("run the guide again, or edit the JSON directly.")
        else:
            print("the validator accepts this reference: "
                  f"python scripts/pr110_annotate.py seal {args.name}")
        return 0

    if args.command == "summary":
        print(json.dumps(summary(annotation), ensure_ascii=False, indent=1))
        return 0

    if args.command == "show":
        image = (annotation.get("document") or {}).get("source_file")
        print(json.dumps(annotation, ensure_ascii=False, indent=1))
        if image:
            print(f"\nthe document itself is at {directory.parent / 'invoices' / image}; open it in any "
                  "viewer - this tool never shows extraction output.")
        return 0

    if args.command in ("check", "seal"):
        image = args.invoice
        if args.command == "seal":
            try:
                K.seal(annotation, image)
            except SystemExit as error:
                print(error)
                return 1
            save(path, annotation)
            print("sealed", args.name, "reference_sha256", annotation["freeze"]["reference_sha256"][:16])
            return 0
        problems = K.validate(annotation, image)
        if problems:
            print("REFUSED - the reference may not be frozen:")
            for problem in problems:
                print("  -", problem)
            return 1
        print("reference accepted by the PR103/PR110 validator; seal it with --seal")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
