"""PR110 Phase 5: the runner the readiness gate hands over to - extraction once per frozen document, scored.

This is deliberately the thinnest possible layer. It is not a new evaluator: the engine is
`src/evaluation_metrics.py`, the one PR102 built and PR109 reused, and the extraction path is the production
entry point, `app.main.document_text` then `app.main.extract_invoice`, exactly as PR103's freeze smoke test
calls it. What this module adds is only the glue the campaign keeps needing:

* it reads the **freeze** PR110's readiness gate wrote and refuses to run against anything else, so a
  reference edited after sealing, a document swapped on disk, or a production source that changed since the
  freeze all stop the run rather than silently producing a different number;
* it turns a version-2 reference into the evaluator's plain shape through `pr110_harness.project()`, which is
  what keeps `absent`, `illegible` and `ambiguous` out of the denominators instead of scoring them as errors;
* it reports per document and in aggregate, and never writes extracted or printed text into its own output:
  verdicts and counts only.

An OCR-only run on twelve documents is roughly twelve times the several-seconds-per-document cost PR102
measured, single process, no model. That is the reason this script is separate from the readiness gate: the
gate is free to re-run, the evaluation is not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import evaluation_metrics as M  # noqa: E402
import pr106_harness as P  # noqa: E402
import pr109_intake as I  # noqa: E402
import pr110_harness as K  # noqa: E402

ARTIFACT_DIR = K.ARTIFACT_DIR
DEFAULT_FREEZE = ARTIFACT_DIR / "pr110-phase5-cohort-freeze.json"


def verify_freeze(freeze: dict, invoice_dir: Path, reference_dir: Path) -> dict:
    """Re-check everything the freeze claims, from the bytes on disk, before a single pipeline call."""
    problems: list[str] = []

    now = P.source_hashes()
    if K.digest(now) != (freeze.get("production_sources") or {}).get("sha256"):
        problems.append("production sources changed after the freeze: the numbers below would not be the "
                        "numbers the frozen cohort was prepared to measure")

    on_disk = {I.sha256_file(path): path for path in K.invoice_files(invoice_dir)}
    references = {reference.get("id"): reference for reference in K.cohort(reference_dir, invoice_dir)}
    documents = []
    for entry in freeze.get("documents") or []:
        document = {"id": entry.get("id"), "problems": []}
        image = on_disk.get(entry.get("document_sha256"))
        if image is None:
            document["problems"].append("the frozen document hash is not present in the intake directory")
        record = references.get(entry.get("id"))
        if record is None:
            document["problems"].append("no reference file carries this document id")
        else:
            if record["content_problems"]:
                document["problems"] += record["content_problems"]
            if not record["seal"]["intact"]:
                document["problems"].append("the reference seal does not recompute: it was edited after "
                                            "freezing, or the document on disk is not the one annotated")
            frozen = (record["annotation"].get("freeze") or {})
            if frozen.get("reference_sha256") != entry.get("reference_sha256"):
                document["problems"].append("freeze.digest on disk differs from the cohort freeze")
            document["annotation"] = record["annotation"]
            document["image"] = image
        documents.append(document)
        problems += [f"{entry.get('id')}: {problem}" for problem in document["problems"]]

    recomputed = K.digest([document["annotation"]["freeze"]["reference_sha256"] for document in documents
                           if document.get("annotation")])
    if recomputed != freeze.get("cohort_sha256"):
        problems.append("the cohort digest does not recompute from the sealed references on disk")
    return {"ok": not problems, "problems": problems, "documents": documents,
            "production_sources_now": {"count": len(now), "sha256": K.digest(now)}}


def run(freeze_path: Path, invoice_dir: Path, reference_dir: Path) -> dict:
    """Extract and score each frozen document, or refuse with the reason."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    verification = verify_freeze(freeze, invoice_dir, reference_dir)
    if not verification["ok"]:
        return {"verdict": "REFUSED", "reason": "the freeze does not recompute from the bytes on disk",
                "verification": verification, "documents": [], "aggregate": None}

    import app.main as application

    plan = application.ocr_language_plan()
    french = [str(document["annotation"]["document"].get("id")) for document in verification["documents"]
              if document["annotation"]["document"].get("region") == "FR"
              or str(document["annotation"]["document"].get("language") or "").lower().startswith("fr")]
    if french and "fra" in plan["missing"]:
        return {"verdict": "REFUSED",
                "reason": "the cohort contains French-language invoices but Tesseract cannot load `fra` "
                          "here; PR81 and PR82 showed that an English-only read of a French document "
                          "produces numbers that look like pipeline failures",
                "language_plan": plan, "french_documents": french,
                "documents": [], "aggregate": None}

    results = []
    for document in verification["documents"]:
        annotation = document["annotation"]
        projected = K.project(annotation)
        diagnostics: dict = {}
        started = time.perf_counter()
        text = application.document_text(str(document["image"]))
        payload = application.extract_invoice(text, diagnostics=diagnostics)
        seconds = round(time.perf_counter() - started, 2)
        scored = M.evaluate(payload, projected["reference"])
        results.append({
            "id": annotation["document"].get("id"),
            "sha256": annotation["document"].get("sha256"),
            "reference_sha256": annotation["freeze"]["reference_sha256"],
            "language": annotation["document"].get("language"),
            "region": annotation["document"].get("region"),
            "input_kind": annotation["document"].get("input_kind"),
            "quality_declaration": annotation["document"].get("quality_declaration"),
            "pages": annotation["document"].get("pages"),
            "reference_projection": projected["counts"],
            "exported_rows": len(payload.get("line_items") or []),
            # the evaluation dict is kept whole: `aggregate()` sums exactly these keys across documents
            "evaluation": scored,
            "runtime": {"seconds": seconds, "ocr_calls": diagnostics.get("ocr_calls"),
                        "ocr_pages": diagnostics.get("ocr_pages"),
                        "preprocessing_used": diagnostics.get("preprocessing_used"),
                        "source_kind": diagnostics.get("source_kind"),
                        "dual_ocr_record_present": diagnostics.get("dual_ocr") is not None},
        })

    return {
        "verdict": "COMPLETED",
        "review": "PR110 Phase 5 - independent evaluation on the frozen cohort",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entry_point": "app.main.document_text + app.main.extract_invoice (production path)",
        "engine": "src/evaluation_metrics.py (the PR102 evaluator; not duplicated here)",
        "cohort_sha256": freeze["cohort_sha256"],
        "freeze_file": K.display(freeze_path),
        "environment": {"switches": K.experimental_switches_off(),
                        "model_guard": P.environment().get("full_precision_qwen_on_cpu")},
        "documents": results,
        "aggregate": M.aggregate([document["evaluation"] for document in results]),
        "unverified_note": "fields the reference recorded as absent, illegible or ambiguous are counted "
                           "unverified and stay out of every denominator; the totals below are therefore "
                           "correct-out-of-scored, not correct-out-of-printed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--invoices", type=Path, default=K.INVOICE_DIR)
    parser.add_argument("--references", type=Path, default=K.REFERENCE_DIR)
    parser.add_argument("--output-name", default="pr110-phase5-evaluation.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify the freeze and report what would run, making 0 pipeline calls")
    args = parser.parse_args(argv)

    if not args.freeze.is_file():
        print(f"REFUSED: no cohort freeze at {K.display(args.freeze)}")
        print("run the readiness gate first: python scripts/pr110_ready.py")
        return 1

    if args.dry_run:
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        verification = verify_freeze(freeze, args.invoices, args.references)
        print("freeze verified:", verification["ok"])
        for problem in verification["problems"]:
            print("  -", problem)
        print(f"{len(verification['documents'])} document(s) would be extracted and scored")
        return 0 if verification["ok"] else 1

    report = run(args.freeze, args.invoices, args.references)
    if report["verdict"] == "REFUSED":
        print("REFUSED:", report["reason"])
        for problem in (report.get("verification") or {}).get("problems") or []:
            print("  -", problem)
        for document_id in report.get("french_documents") or []:
            print(f"  - document {document_id} is French-language")
        P.write_json(P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name), report)
        return 1

    aggregate = report["aggregate"]
    print(f"documents: {aggregate['documents']} | cohort {report['cohort_sha256'][:16]}")
    print("header   :", json.dumps(aggregate["header"]))
    print("rows     :", json.dumps({key: aggregate["rows"][key] for key in
                                    ("expected", "exported", "correctly_matched", "missing", "misassociated")}))
    print("fields   :", json.dumps({key: aggregate["fields"][key] for key in
                                    ("correct", "incorrect", "unverified", "incorrect_association")}))
    print("position :", json.dumps(aggregate["positional"]["cells"]))
    path = P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name)
    P.write_json(path, report)
    print("written:", K.display(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
