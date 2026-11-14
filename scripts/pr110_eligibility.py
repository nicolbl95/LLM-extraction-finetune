"""PR110 Phase 4: is this document genuinely independent, and may the owner let us use it?

PR109 answered the same question for the whole repository and found zero eligible documents. Its screening
primitives were correct and are reused here unchanged - `perceptual()` for the byte and visual signatures,
`hamming()` for the near-duplicate distance, `known_content()` for the 640-file PR103 inventory, and
`signed_population()` for the images the campaign has already touched. What PR109 could not do was *decide*:
its `assess()` returns `eligible_as_independent: False` unconditionally, because in PR109 every candidate was
either absent or already used, so the field was never exercised. That is a latent bug, not a rule, and PR110
does not edit PR109's published behaviour - it composes the decision here, from the file, from the PR103
record of past use, and from what the blind annotation says about permission, origin and convention.

Three states, never merged:

* **eligible** - readable, unseen by the campaign, permitted, and annotated, validated and sealed.
* **excluded** - a hard reason exists: a byte-identical or visually similar file already used, a purpose in
  the PR103 inventory (training render, development scan, fixture, derivative), a permission refused, a
  document that is not an invoice a business issued, or a file that cannot be read at all.
* **pending_review** - no hard reason, but something is missing: no annotation, an unsealed or invalid one,
  permission or language left blank, a page map that does not add up.

Two rules the brief states and this module enforces:

* A new filename proves nothing. Independence is the hash plus the visual signature plus the PR103 purpose
  plus the owner's declaration; nothing here looks at a filename when deciding.
* A page-level signature is computed for every page of a multipage file, because a file whose page 2 is a
  second invoice is two documents and the intake refuses it until it is split.

Importing this module makes 0 OCR calls and 0 model calls; the geometry check uses production's own ruled-
region detector on preprocessed pixels, which is the same computation `recover_image` would make and reads
no text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr106_harness as P  # noqa: E402
import pr109_intake as I  # noqa: E402
import pr110_harness as K  # noqa: E402

ARTIFACT_DIR = K.ARTIFACT_DIR
ELIGIBLE = "eligible"
EXCLUDED = "excluded"
PENDING_REVIEW = "pending_review"

# Purposes the PR103 inventory assigns to files the campaign has already touched. Any of them means the
# document is not independent: it has already been used for training, development, regression or as a demo.
USED_PURPOSES_PREFIX = ("synthetic_", "development_", "regression_", "derivative_", "demo_", "unredacted_")

# Two further purposes are not "already used for training" but still cannot be independent evidence:
# `partial_document` is a crop of a document the campaign analysed and `not_an_invoice` covers the debug
# renders under results/. An annotation could not make either of them a real invoice, so the bytes exclude them.
NON_DOCUMENT_PURPOSES_PREFIX = ("partial_document", "not_an_invoice")
NOT_INDEPENDENT_PURPOSES_PREFIX = USED_PURPOSES_PREFIX + NON_DOCUMENT_PURPOSES_PREFIX


# --------------------------------------------------------------------------- what the campaign already used

def pr103_record(manifest_path: Path = I.PR103_INVENTORY) -> dict:
    """The authoritative inventory of past use, read from a PR103-format manifest exactly as published.

    `manifest_path` defaults to the frozen PR103 artefact. Passing a freshly rebuilt inventory narrows the
    window in which a file added after the freeze is invisible to `previous_use`; the report always names the
    manifest it consumed, so the two are never confused.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_sha = {entry["sha256"]: entry for entry in manifest["files"]}
    used_hashes = {entry["sha256"] for entry in manifest["files"]
                   if entry["purpose"].startswith(USED_PURPOSES_PREFIX)}
    return {
        "manifest": manifest["review"],
        "manifest_file": manifest_path.name,
        "verdict": manifest["verdict"],
        "files": len(manifest["files"]),
        "by_sha": by_sha,
        "used_hashes": used_hashes,
        "used_document_hashes": set(manifest["used_document_hashes"]),
    }


# --------------------------------------------------------------------------- page-level signatures

def page_signatures(path: Path) -> list[dict]:
    """One signature per page (or per image), so a multipage file cannot hide a second invoice on page 3."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import pymupdf

            document = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
            out = []
            for number, page in enumerate(document, start=1):
                text = page.get_text() or ""
                out.append({"page": number,
                            "text_sha256": I.sha256_bytes(text.encode("utf-8")),
                            "characters": len(text.strip()),
                            "has_text_layer": bool(text.strip())})
            document.close()
            return out
        return [{"page": 1, "text_sha256": None, "characters": None, "has_text_layer": False}]
    except Exception as error:  # an unreadable file is a finding, never an exception escaping the gate
        return [{"page": None, "error": f"{type(error).__name__}: {error}"}]


# --------------------------------------------------------------------------- the checks, one per requirement

def _check(name: str, status: str, detail: str, blocking: str | None = None) -> dict:
    """status: pass | finding | unanswered. blocking: 'exclude' | 'pending' | None."""
    return {"check": name, "status": status, "detail": detail, "blocks": blocking}


def screen_file(path: Path, record: dict, population: list[dict], siblings: dict[str, dict]) -> dict:
    """Everything the bytes alone establish: readability, hash, duplicates and prior use."""
    signature = I.perceptual(path)
    checks = []

    if path.suffix.lower() not in I.DOCUMENT_SUFFIXES:
        checks.append(_check("format_readable", "finding",
                             f"{path.suffix or 'no extension'} is not an accepted invoice format "
                             "(PDF, PNG or JPEG)", "exclude"))
    elif signature.get("pdf_error"):
        checks.append(_check("format_readable", "finding",
                             f"the PDF could not be opened: {signature['pdf_error']}", "exclude"))
    elif signature.get("ahash") is None and signature.get("pages") is None:
        checks.append(_check("format_readable", "finding",
                             "no page and no image signature could be computed", "exclude"))
    else:
        checks.append(_check("format_readable", "pass",
                             f"{signature['format']}, "
                             + (f"{signature['pages']} page(s)" if signature.get("pages") is not None
                                else f"{signature.get('width')}x{signature.get('height')} px")))

    checks.append(_check("sha256_recorded", "pass", signature["sha256"]))

    known = record["by_sha"].get(signature["sha256"])
    duplicates = []
    if known:
        duplicates.append({"path": known["path"], "purpose": known["purpose"],
                           "basis": "byte-identical to a file in the PR103 inventory"})
        for hit in known.get("byte_identical_to_a_used_document") or []:
            duplicates.append({"path": hit, "purpose": "a document the pipeline was already run on",
                               "basis": "PR103 recorded this file as byte-identical to it"})
    if signature["sha256"] in record["used_hashes"]:
        duplicates.append({"path": known["path"] if known else "?", "purpose": known["purpose"] if known else "?",
                           "basis": "the PR103 inventory classifies this hash as already used"})
    if duplicates:
        checks.append(_check("exact_duplicate", "finding",
                             "; ".join(f"{d['path']} ({d['purpose']})" for d in duplicates), "exclude"))
    else:
        checks.append(_check("exact_duplicate", "pass", "no byte-identical file in the PR103 inventory"))

    near = []
    for entry in population:
        best = min([value for value in (I.hamming(signature.get("ahash"), entry.get("ahash")),
                                        I.hamming(signature.get("dhash"), entry.get("dhash")))
                    if value is not None] or [99])
        if best <= I.PERCEPTUAL_MAX_DISTANCE:
            near.append({"path": entry["path"], "distance": best})
    if near:
        checks.append(_check("visually_similar_duplicate", "finding",
                             f"within {I.PERCEPTUAL_MAX_DISTANCE} bits of "
                             + ", ".join(f"{n['path']} ({n['distance']})" for n in near[:5]), "exclude"))
    else:
        checks.append(_check("visually_similar_duplicate", "pass",
                             f"nothing in the signed population within {I.PERCEPTUAL_MAX_DISTANCE} bits"))

    other = siblings.get(signature["sha256"])
    if other and other != path:
        checks.append(_check("duplicate_inside_the_intake", "finding",
                             f"the same bytes are also present as {K.display(other)}", "exclude"))
    elif other:
        checks.append(_check("duplicate_inside_the_intake", "pass", "the first copy of these bytes in the intake"))

    if known:
        detail = f"PR103 classifies this hash as {known['purpose']}"
        blocks = "exclude" if known["purpose"].startswith(NOT_INDEPENDENT_PURPOSES_PREFIX) else None
        checks.append(_check("previous_use", "finding" if blocks else "pass", detail, blocks))
    else:
        checks.append(_check("previous_use", "pass",
                             "this hash appears in no PR103 inventory entry, so no recorded training, "
                             "development, regression or evaluation use"))
    return {"signature": signature, "checks": checks, "pages": page_signatures(path)}


def screen_annotation(annotation: dict | None, problems: list[str], seal: dict,
                      image: Path | None) -> list[dict]:
    """What only the blind annotation can establish: permission, origin, convention, and the freeze."""
    if annotation is None:
        return [_check("reference_annotation", "unanswered",
                       "no sealed reference exists for this document", "pending")]
    document = annotation.get("document") or {}
    permission = document.get("owner_permission") or {}
    checks = [_check("reference_annotation", "pass" if not problems else "finding",
                     "the validator accepts this reference" if not problems
                     else "; ".join(problems[:6]), None if not problems else "pending")]
    checks.append(_check("seal", "pass" if seal.get("intact") else "finding",
                         json.dumps({key: value for key, value in seal.items() if key != "note"}),
                         None if seal.get("intact") else "pending"))
    checks.append(_check("image_paired", "pass" if image is not None else "finding",
                         "the annotated document is present in the intake directory" if image else
                         "the reference names a document that is not in the intake directory",
                         None if image else "pending"))

    if permission.get("authorized_for_local_evaluation") is True:
        checks.append(_check("permission", "pass",
                             f"granted, evidence recorded: {bool(permission.get('evidence'))}"))
    elif permission.get("authorized_for_local_evaluation") is False:
        checks.append(_check("permission", "finding", "the owner recorded that this document may not be used",
                             "exclude"))
    else:
        checks.append(_check("permission", "unanswered", "owner_permission.authorized_for_local_evaluation "
                             "is blank", "pending"))

    origin = document.get("source_origin")
    checks.append(_check("source", "pass" if origin else "unanswered",
                         f"declared origin: {bool(origin)}", None if origin else "pending"))

    genuine = document.get("genuine_business_invoice")
    if genuine is True:
        checks.append(_check("genuine_invoice", "pass", "declared an invoice a business actually issued"))
    elif genuine is False:
        checks.append(_check("genuine_invoice", "finding",
                             "declared a template, sample or demo - it cannot validate extraction", "exclude"))
    else:
        checks.append(_check("genuine_invoice", "unanswered", "genuine_business_invoice is blank", "pending"))

    language, region = document.get("language"), document.get("region")
    checks.append(_check("language_and_convention",
                         "pass" if language and region else "unanswered",
                         f"language={language!r}, region={region!r}; taken from the annotation, never guessed "
                         "from a filename or a currency symbol",
                         None if (language and region) else "pending"))

    pages = [page for page in (document.get("pages") or []) if isinstance(page, int)]
    observed = document.get("page_count_observed")
    if not pages:
        checks.append(_check("one_invoice_per_file", "unanswered",
                             "document.pages is blank: which pages of the file are this invoice?", "pending"))
    elif isinstance(observed, int) and observed and pages != list(range(1, observed + 1)):
        checks.append(_check("one_invoice_per_file", "unanswered",
                             f"the annotation says pages {pages} of {observed}: the file holds more than one "
                             "invoice and the intake refuses it until it is split", "pending"))
    else:
        checks.append(_check("one_invoice_per_file", "pass", f"pages {pages} of {observed!r}"))
    return checks


def screen_layout(path: Path | None, annotation: dict | None, signature: dict | None) -> dict:
    """Layout characteristics, from the geometry production itself would see plus the owner's declaration."""
    out: dict = {"computed": False}
    if path is None:
        out["note"] = "no image on disk: the layout cannot be described"
        return out
    try:
        import pr109_dev_cohort as C

        geometry = C.ruled_geometry(path)
        out.update({
            "computed": True,
            "bordered_table_detected_by_production": geometry["bordered_table_detected_by_production"],
            "deskew_angle_degrees": geometry["angle_degrees"],
            "ruled_region_area_ratio": geometry["deskew"]["ruled_region_area_ratio"]
            or geometry["clahe"]["ruled_region_area_ratio"],
            "grid_contours": geometry["deskew"]["grid_contours_found"],
        })
    except Exception as error:  # geometry is a description, never a gate: a failure is recorded, not guessed
        out["error"] = f"{type(error).__name__}: {error}"
    if signature:
        out["resolution"] = {"dpi": signature.get("dpi"), "megapixels": signature.get("megapixels"),
                             "pages": signature.get("pages"), "bytes": signature.get("bytes")}
        out["embedded_text_layer"] = signature.get("embedded_text_layer")
    out["owner_declared_quality"] = (annotation or {}).get("document", {}).get("quality_declaration")
    out["owner_declared_input_kind"] = (annotation or {}).get("document", {}).get("input_kind")
    return out


# --------------------------------------------------------------------------- the decision

def decide(checks: list[dict]) -> dict:
    """One status per document: a hard reason excludes, a missing answer holds, otherwise eligible."""
    excluded = [check["check"] for check in checks if check["blocks"] == "exclude"]
    pending = [check["check"] for check in checks if check["blocks"] == "pending"]
    status = EXCLUDED if excluded else PENDING_REVIEW if pending else ELIGIBLE
    return {"status": status, "hard_reasons": excluded, "open_questions": pending}


def eligibility(invoice_dir: Path = K.INVOICE_DIR, reference_dir: Path = K.REFERENCE_DIR,
                *, geometry: bool = True, inventory: Path = I.PR103_INVENTORY) -> dict:
    """Screen every document in the intake, paired with its annotation, and classify it."""
    record = pr103_record(inventory)
    population = I.signed_population()
    files = K.invoice_files(invoice_dir)
    siblings: dict[str, Path] = {}
    for path in files:
        siblings.setdefault(I.sha256_file(path), path)
    annotations = K.cohort(reference_dir, invoice_dir)
    by_image: dict[Path, dict] = {}
    for reference in annotations:
        if reference.get("image") is not None:
            by_image[Path(reference["image"]).resolve()] = reference

    documents = []
    for path in files:
        signature_block = screen_file(path, record, population, siblings)
        annotation_record = by_image.get(path.resolve())
        signature = signature_block["signature"]
        annotation = (annotation_record or {}).get("annotation")
        problems = (annotation_record["problems"] if annotation_record
                    else ["no reference file pairs with this document"])
        seal = (annotation_record or {}).get("seal") or {"intact": False, "sealed": False}
        checks = signature_block["checks"] + screen_annotation(annotation, problems, seal, path)
        documents.append({
            "document": path.stem,
            "path": str(path),
            "sha256": signature_block["signature"]["sha256"],
            "reference": (annotation_record or {}).get("file"),
            "id": (annotation or {}).get("document", {}).get("id"),
            **decide(checks),
            "checks": checks,
            "pages": signature_block["pages"],
            "layout": screen_layout(path, annotation, signature_block["signature"]) if geometry else
                      {"computed": False, "note": "skipped by --no-geometry"},
        })

    orphans = [reference for reference in annotations if reference.get("image") is None]
    counts = {status: sum(1 for document in documents if document["status"] == status)
              for status in (ELIGIBLE, EXCLUDED, PENDING_REVIEW)}
    return {
        "review": "PR110 Phase 4 - independent eligibility verification",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_documents": K.TARGET_DOCUMENTS,
        "intake": {"invoice_directory": K.display(invoice_dir), "documents_found": len(files),
                   "reference_directory": K.display(reference_dir), "references_found": len(annotations)},
        "screen_provenance": {
            "reused_from_pr109": ["perceptual", "hamming", "sha256_file", "known_content",
                                  "signed_population", "PERCEPTUAL_MAX_DISTANCE"],
            "pr109_assess_not_reused": "pr109_intake.assess() returns eligible_as_independent=False "
                                       "unconditionally - correct for PR109's empty intake, a latent bug "
                                       "for a real one; PR109's published artefact is left untouched and the "
                                       "decision is composed here from its primitives",
            "prior_use_authority": f"{record['manifest']} ({record['manifest_file']}): "
                                   f"{record['files']} files classified, verdict {record['verdict']}",
            "population_signatures": len(population),
        },
        "references_without_a_document": [
            {"file": reference.get("file"), "id": reference.get("id"), "problems": reference["problems"]}
            for reference in orphans],
        "counts": counts,
        "eligible_ids": [document["sha256"][:12] for document in documents if document["status"] == ELIGIBLE],
        "shortage": max(0, K.TARGET_DOCUMENTS - counts[ELIGIBLE]),
        "documents": documents,
        "self_test": None,
        "known_limitations": {
            "perceptual_screen_on_sparse_pages": "an average/difference hash of a nearly blank page is "
                                                 "dominated by whitespace: a blank synthetic test image sits "
                                                 "7 bits from app/examples/modele-vierge.png, a blank template, "
                                                 "so a very low-ink document can be flagged as similar to an "
                                                 "unrelated one. The distance is reported with the file it "
                                                 "matched, so the owner can look and decide; the screen never "
                                                 "clears a document, only questions it.",
            "permission_rule_lives_in_the_validator": "PR103's published validator refuses to seal a reference "
                                                     "whose permission is not true, so a refused document "
                                                     "reaches this screen unsealed; the explicit exclude branch "
                                                     "here covers an annotation edited directly.",
            "eligibility_never_reads_a_filename": "nothing in screen_file or screen_annotation uses path.name "
                                                  "as evidence; a new filename is not a reason for anything.",
            "prior_use_authority_is_a_frozen_snapshot": "PR103_INVENTORY is the manifest as PR103 published it "
                                                        "(640 files), so `previous_use` cannot see a file added "
                                                        "to the repository after that freeze - PR129 measured 6 "
                                                        "such files, all derived debug crops under results/. The "
                                                        "byte and perceptual screens still apply; screening with "
                                                        "--inventory <a rebuilt manifest> narrows the window to "
                                                        "the interval since the last inventory run.",
        },
    }


def detector_proof(sample_paths: list[Path]) -> dict | None:
    """Re-run PR109's synthetic duplicate test so the screen's own behaviour is measured, not assumed."""
    usable = [path for path in sample_paths if path.is_file() and path.suffix.lower() in I.IMAGE_SUFFIXES]
    if len(usable) < 2:
        return {"skipped": "two development images are needed to prove the perceptual screen; "
                           f"{len(usable)} found"}
    proof = I.detector_self_test(usable[:2])
    proof["samples"] = [K.display(path) for path in usable[:2]]
    return proof


def detector_samples() -> list[Path]:
    """Two ink-bearing images the campaign already owns, for the duplicate proof.

    Taken from `tests/fixtures/` and chosen by size rather than by name: a nearly blank page degenerates an
    average hash, and the proof writes the sample names into its artifact, so the fixtures keep development
    document names out of the report.
    """
    images = [path for path in sorted((ROOT / "tests" / "fixtures").rglob("*"))
              if path.is_file() and path.suffix.lower() in I.IMAGE_SUFFIXES]
    return sorted(images, key=lambda path: path.stat().st_size, reverse=True)[:2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invoices", type=Path, default=K.INVOICE_DIR)
    parser.add_argument("--references", type=Path, default=K.REFERENCE_DIR)
    parser.add_argument("--output-name", default="pr110-phase4-eligibility.json")
    parser.add_argument("--no-geometry", action="store_true",
                        help="skip production's ruled-region detector (it reads pixels, never text)")
    parser.add_argument("--inventory", type=Path, default=I.PR103_INVENTORY,
                        help="a PR103-format manifest of past use; defaults to the frozen PR103 artefact")
    parser.add_argument("--self-test", action="store_true",
                        help="also re-run PR109's synthetic duplicate proof and record the result")
    args = parser.parse_args(argv)

    report = eligibility(args.invoices, args.references, geometry=not args.no_geometry,
                         inventory=args.inventory)
    if args.self_test:
        report["self_test"] = detector_proof(detector_samples())

    for document in report["documents"]:
        print(f"{document['document']:<24} {document['sha256'][:12]}  {document['status']:<13} "
              f"hard={','.join(document['hard_reasons']) or '-'} "
              f"open={','.join(document['open_questions']) or '-'}")
    print(f"\nintake: {report['intake']['documents_found']} document(s), "
          f"{report['intake']['references_found']} reference(s) | target {report['target_documents']}")
    print(f"eligible {report['counts'][ELIGIBLE]} | excluded {report['counts'][EXCLUDED]} | "
          f"pending review {report['counts'][PENDING_REVIEW]} | shortage {report['shortage']}")
    print(f"prior-use authority: {report['screen_provenance']['prior_use_authority']}")
    if report["references_without_a_document"]:
        print("references naming a document that is not in the intake:",
              [reference["file"] for reference in report["references_without_a_document"]])
    if report["self_test"] and "skipped" not in report["self_test"]:
        proof = report["self_test"]
        print(f"duplicate screen self-test on {', '.join(proof['samples'])}: "
              f"{proof['variants_detected']}/{proof['variants_tested']} variants "
              f"caught, {proof['variants_a_byte_hash_would_miss']} would be missed by a byte hash, "
              f"{proof['false_positives_on_a_different_document']} false positive(s)")
    path = P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name)
    P.write_json(path, report)
    print("written:", K.display(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
