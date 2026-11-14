"""PR110 Phase 5: one command that says whether the independent benchmark may be run, and refuses if not.

The brief asks for a single validation command with two gates and one freeze:

* the **intake gate** - at least twelve documents the eligibility screen classifies as eligible;
* the **annotation gate** - every reference in the cohort passes the published PR103/PR110 validator and its
  seal still recomputes to what it recorded;
* the **freeze** - the cohort, the reference digests, the 29 production source hashes and the experimental
  switch state, all written before any pipeline run, so the evaluation that follows is measuring a benchmark
  that existed first.

If either gate fails this prints what is missing, records the finding, and exits non-zero. It does not
partially freeze: a cohort of nine is not a smaller version of a cohort of twelve, and a readiness report
that said "ready for nine" would be the kind of number a later agent could mistake for a result.

The evaluation itself is not implemented here. `scripts/pr110_evaluate.py` runs the pipeline once per frozen
document and scores it with `src/evaluation_metrics.py` - the engine PR102 built and PR109 kept; this module
only decides whether that run is allowed and writes the freeze it must run from.
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
import pr110_eligibility as E  # noqa: E402
import pr110_harness as K  # noqa: E402

ARTIFACT_DIR = K.ARTIFACT_DIR
READY = "READY_FOR_INDEPENDENT_EVALUATION"
NOT_READY = "NOT_READY"


def cohort_geometry(documents: list[dict]) -> dict[str, dict]:
    """The ruled-region answers the coverage categories need, keyed by document id."""
    import pr109_dev_cohort as C

    out: dict[str, dict] = {}
    for document in documents:
        if document["status"] != E.ELIGIBLE:
            continue
        try:
            out[str(document["id"] or document["sha256"][:12])] = C.ruled_geometry(Path(document["path"]))
        except Exception as error:  # a category the geometry cannot answer stays reported as unknown
            out[str(document["id"] or document["sha256"][:12])] = {"error": f"{type(error).__name__}: {error}"}
    return out


def readiness(invoice_dir: Path = K.INVOICE_DIR, reference_dir: Path = K.REFERENCE_DIR,
              *, geometry: bool = True, source_manifest: Path | None = None) -> dict:
    """Measure both gates, freeze the cohort when they pass, and name the command that runs the evaluation.

    `source_manifest` declares which source version the cohort is frozen against. Left unset it is PR103's
    published freeze, exactly as before; a benchmark that means to run a later source version names its own
    manifest instead of silently comparing itself to a baseline it has already left behind.
    """
    screen = E.eligibility(invoice_dir, reference_dir, geometry=geometry)
    records = K.cohort(reference_dir, invoice_dir)
    eligible = [record for record in records if record["eligible"]]

    intake_gate = {
        "requirement": f"at least {K.TARGET_DOCUMENTS} documents eligible as independent",
        "eligible": len(eligible),
        "excluded": screen["counts"][E.EXCLUDED],
        "pending_review": screen["counts"][E.PENDING_REVIEW],
        "shortage": screen["shortage"],
        "passed": screen["counts"][E.ELIGIBLE] >= K.TARGET_DOCUMENTS,
    }
    invalid = [{"file": record["file"], "problems": record["content_problems"],
                "seal": {key: value for key, value in record["seal"].items() if key != "note"}}
               for record in records if record["content_problems"] or not record["seal"]["intact"]]
    annotation_gate = {
        "requirement": "every reference in the cohort passes the PR103/PR110 validator and its seal "
                       "recomputes to the digest it recorded",
        "references": len(records),
        "validated_and_sealed": sum(1 for record in records
                                    if not record["content_problems"] and record["seal"]["intact"]),
        "invalid": invalid,
        "passed": bool(records) and not invalid,
        "note": "`bool(records)` is part of the condition on purpose: with zero references this gate fails "
                "rather than passing vacuously, because 'every reference is valid' is trivially true when "
                "there are none and an empty intake must never read as an annotated one",
    }

    if geometry and intake_gate["passed"]:
        coverage = K.coverage(eligible, geometry=cohort_geometry(screen["documents"]))
    else:
        coverage = K.coverage(eligible, geometry=None)
        if geometry:
            for category in coverage:
                if "ruled region" in category["category"]:
                    category["count"] = None
                    category["note"] = ("not measured: the intake gate failed, so no geometry was computed "
                                        "for these documents")
    gaps = [category for category in coverage if category["underrepresented"]]

    hashes = P.source_hashes()
    frozen_documents = sorted(
        [{"id": (record["annotation"].get("document") or {}).get("id"),
          "source_file": (record["annotation"].get("document") or {}).get("source_file"),
          "document_sha256": (record["annotation"].get("document") or {}).get("sha256"),
          "pages": (record["annotation"].get("document") or {}).get("pages"),
          "reference_sha256": (record["annotation"].get("freeze") or {}).get("reference_sha256"),
          "body_sha256": (record["annotation"].get("freeze") or {}).get("body_sha256"),
          "frozen_at": (record["annotation"].get("freeze") or {}).get("frozen_at")}
         for record in eligible],
        key=lambda entry: str(entry["id"]))
    freeze = {
        "documents": frozen_documents,
        "cohort_sha256": K.digest([entry["reference_sha256"] for entry in frozen_documents]),
        "production_sources": {"count": len(hashes), "sha256": K.digest(hashes),
                               "pr103_drift": P.pr103_freeze(hashes, source_manifest)},
        "experimental_switches": K.experimental_switches_off(),
        "git": P.git_state(),
        "frozen_before_any_pipeline_run": True,
    }
    gates_passed = intake_gate["passed"] and annotation_gate["passed"]
    return {
        "review": "PR110 Phase 5 - benchmark readiness",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": READY if gates_passed else NOT_READY,
        "intake_gate": intake_gate,
        "annotation_gate": annotation_gate,
        "coverage": coverage,
        "coverage_gaps": [category["category"] for category in gaps],
        "blocking_findings": (
            ([] if intake_gate["passed"] else
             [f"{intake_gate['shortage']} more eligible documents are required; "
              f"{intake_gate['pending_review']} document(s) are pending review and "
              f"{intake_gate['excluded']} excluded"])
            + ([] if annotation_gate["passed"] else
               [f"{len(invalid)} reference(s) fail the validator or have a broken seal" if records else
                "no reference file exists yet: the annotation gate fails on an empty cohort rather than "
                "passing vacuously"])),
        "freeze": freeze if gates_passed else None,
        "freeze_when_not_ready": None if gates_passed else {
            "note": "no cohort frozen: the gates failed, and a cohort of fewer than "
                    f"{K.TARGET_DOCUMENTS} documents is not a smaller version of a full one"},
        # the interpreter is named rather than assumed: a bare `python` can resolve to one without pytesseract,
        # and the command this prints is the owner's only route to the evaluation
        "launch_command": (f'"{sys.executable}" scripts/pr110_evaluate.py '
                           f"--freeze {K.display(ARTIFACT_DIR / 'pr110-phase5-cohort-freeze.json')}"
                           if gates_passed else None),
        "refusal": "this script exits with status 1 and writes no freeze when either gate fails; "
                   "scripts/pr110_evaluate.py refuses independently by re-checking the freeze it was given",
        "eligibility_screen": {key: screen[key] for key in
                               ("target_documents", "counts", "shortage", "screen_provenance",
                                "known_limitations", "references_without_a_document")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invoices", type=Path, default=K.INVOICE_DIR)
    parser.add_argument("--references", type=Path, default=K.REFERENCE_DIR)
    parser.add_argument("--output-name", default="pr110-phase5-readiness.json")
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--source-manifest", type=Path, default=None,
                        help="source-version manifest the cohort is frozen against; defaults to PR103's")
    args = parser.parse_args(argv)

    report = readiness(args.invoices, args.references, geometry=not args.no_geometry,
                       source_manifest=args.source_manifest)
    print(f"verdict: {report['verdict']}")
    gate = report["intake_gate"]
    print(f"intake gate     : {'PASS' if gate['passed'] else 'FAIL'} - eligible {gate['eligible']}"
          f"/{K.TARGET_DOCUMENTS}, excluded {gate['excluded']}, pending {gate['pending_review']}")
    annotation = report["annotation_gate"]
    print(f"annotation gate : {'PASS' if annotation['passed'] else 'FAIL'} - validated and sealed "
          f"{annotation['validated_and_sealed']}/{annotation['references']}")
    for entry in annotation["invalid"][:20]:
        print("   ", entry["file"], entry["problems"][:4])
    print(f"coverage gaps (< 2 documents): {report['coverage_gaps'] or 'none'}")
    for finding in report["blocking_findings"]:
        print("  BLOCKING:", finding)
    if report["freeze"]:
        sources = report["freeze"]["production_sources"]["pr103_drift"]
        print(f"source version  : {sources['manifest']} ({sources['manifest_sha256'][:12]})"
              f" - drifted {len(sources['drifted'])}, added {len(sources['added_since_pr103'])}")
        print(f"cohort frozen: {len(report['freeze']['documents'])} document(s), "
              f"cohort sha {report['freeze']['cohort_sha256'][:16]}")
        print("launch the independent evaluation with:")
        print("   ", report["launch_command"])
        P.write_json(P.guard(ARTIFACT_DIR, ARTIFACT_DIR / "pr110-phase5-cohort-freeze.json"),
                     report["freeze"])
    else:
        print(report["freeze_when_not_ready"]["note"])

    P.write_json(P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name), report)
    print("written:", K.display(ARTIFACT_DIR / args.output_name))
    return 0 if report["verdict"] == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
