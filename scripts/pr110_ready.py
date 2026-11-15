"""PR110 Phase 5: one command that says whether the independent benchmark may be run, and refuses if not.

The brief asks for a single validation command with two gates and one freeze:

* the **intake gate** - at least twelve documents the eligibility screen classifies as eligible;
* the **annotation gate** - every reference in the cohort passes the published PR103/PR110 validator and its
  seal still recomputes to what it recorded;
* the **freeze** - the cohort, the reference digests, the 29 production source hashes and the experimental
  switch state, all written before any pipeline run, so the evaluation that follows is measuring a benchmark
  that existed first.

PR167 added the third gate and made the freeze carry the selection:

* the **coverage gate** - the cohort actually evaluated is 4 FR + 4 UK + 4 US admissible, sealed, mutually
  distinct invoices, chosen deterministically, so a campaign can never report a convention it did not measure.

If a gate fails this prints what is missing, records the finding, and exits non-zero. It does not
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


def coverage_gate(selection: dict, *, geometry: bool = True) -> dict:
    """PR167: the cohort is a composition, and this is the gate that says whether it has one.

    The intake gate counts admissible documents and the annotation gate checks their references; neither of
    them looks at *which* invoices they are. Twelve French invoices would pass both and measure one convention
    three times over, which is why the protocol states 4 FR + 4 UK + 4 US and why that statement needs its own
    gate. It runs on the cohort actually selected for evaluation - the deterministic twelve that
    `pr110_harness.select_cohort()` takes from the admitted documents, passed in here so the freeze and the
    gate are built from one selection rather than two - not on an inventory of what happens to be on disk.

    Three things can fail it. A quota nobody fills is a cohort that does not measure what it claims. A selected
    document that is not admissible, or whose reference is not sealed, is a slot claimed by a document the
    other two gates refused - it cannot happen while the selection is drawn from the admitted list, and it is
    checked anyway because a gate that trusts its own input is not a gate. And two members that turn out to be
    one invoice - the same bytes, the same id, or one page re-rendered, which the campaign's own registered-
    variant test decides at PR164's unchanged thresholds - make a twelve-document cohort thirteen pages about
    eleven invoices.
    """
    chosen = selection["selected"]
    pairs = K.cohort_variant_pairs(chosen, geometry=geometry)
    not_admitted = [{"id": str(record.get("id")), "region": K.region_of(record),
                     "reasons": (record["content_problems"] or [])
                                + ([] if record["seal"]["intact"] else ["the reference seal does not recompute"])}
                    for record in chosen
                    if not (record["eligible"] and not record["content_problems"] and record["seal"]["intact"])]

    findings = []
    if not selection["protocol_consistent"]:
        findings.append(f"the protocol does not add up: the quota asks for {sum(selection['quota'].values())} "
                        f"document(s) and the target is {selection['required_total']}; one country would be "
                        "dropped without saying so")
    if selection["missing"]:
        findings.append("the selected cohort leaves a quota short: "
                        + ", ".join(f"missing {region}: {count}"
                                    for region, count in sorted(selection["missing"].items())))
    if selection["without_a_declared_region"]:
        findings.append(f"{len(selection['without_a_declared_region'])} admitted document(s) declare no region "
                        "and therefore fill no slot: " + ", ".join(selection["without_a_declared_region"][:10]))
    if len(chosen) > selection["required_total"]:
        findings.append(f"{len(chosen)} documents were selected for a cohort of "
                        f"{selection['required_total']}; the reserve is allowed, a larger evaluated cohort is not")
    if not_admitted:
        findings.append(f"{len(not_admitted)} selected document(s) are not admissible or not sealed: "
                        + ", ".join(entry["id"] for entry in not_admitted[:10]))
    if pairs["proven"]:
        findings.append(f"{len(pairs['proven'])} pair(s) of the selected cohort are one invoice: "
                        + "; ".join(f"{'/'.join(entry['pair'])} — {entry['basis']}" for entry in pairs["proven"]))
    if pairs["unconfirmed"]:
        findings.append(f"{len(pairs['unconfirmed'])} pair(s) of the selected cohort are not proven distinct and "
                        "need the owner: " + "; ".join(f"{'/'.join(entry['pair'])}" for entry in pairs["unconfirmed"]))
    if pairs["errors"]:
        findings.append(f"{len(pairs['errors'])} pair(s) of the selected cohort could not be measured: "
                        + "; ".join(f"{'/'.join(entry['pair'])} — {entry['error']}" for entry in pairs["errors"]))

    return {
        "requirement": " + ".join(f"{count} {region}" for region, count in sorted(selection["quota"].items()))
                       + f" = {selection['required_total']} admissible, sealed, mutually distinct invoices",
        "quota": selection["quota"],
        "counts": selection["counts"],
        "report": ", ".join(f"{region}: {selection['counts'][region]}/{selection['quota'][region]}"
                            for region in sorted(selection["quota"])),
        "missing": selection["missing"],
        "selected": len(chosen),
        "selected_ids": selection["selected_ids"],
        "selection_sha256": selection["selection_sha256"],
        "ordered_by": selection["ordered_by"],
        "candidates": selection["candidates"],
        "reserve": selection["reserve"],
        "off_quota_candidates": selection["off_quota_candidates"],
        "without_a_declared_region": selection["without_a_declared_region"],
        "not_admissible": not_admitted,
        "distinct": pairs,
        "findings": findings,
        "passed": not findings,
    }


def readiness(invoice_dir: Path = K.INVOICE_DIR, reference_dir: Path = K.REFERENCE_DIR,
              *, geometry: bool = True, source_manifest: Path | None = None) -> dict:
    """Measure three gates, freeze the cohort when they pass, and name the command that runs the evaluation.

    `source_manifest` declares which source version the cohort is frozen against. Left unset it is PR103's
    published freeze, exactly as before; a benchmark that means to run a later source version names its own
    manifest instead of silently comparing itself to a baseline it has already left behind.
    """
    screen = E.eligibility(invoice_dir, reference_dir, geometry=geometry)
    records = K.cohort(reference_dir, invoice_dir)
    # Two questions, two authorities. The screen answers "is this an independent document?"; the annotation
    # answers "is it admissible?". A reference can validate and seal against a document the screen excludes or
    # holds, so the cohort is the intersection and nothing reaches the freeze on its annotation alone.
    by_path = {str(Path(document["path"]).resolve()): document for document in screen["documents"]}
    admitted = {path for path, document in by_path.items() if document["status"] == E.ELIGIBLE}
    eligible, annotated_not_admitted, frozen_once = [], [], set()
    for record in records:
        if not record["eligible"]:
            continue
        path = str(Path(record["image"]).resolve())
        if path in admitted and path not in frozen_once:
            frozen_once.add(path)
            eligible.append(record)
            continue
        decision = by_path.get(path) or {}
        annotated_not_admitted.append({
            "file": record["file"], "document": decision.get("document"),
            "screen_status": decision.get("status"), "hard_reasons": decision.get("hard_reasons"),
            "open_questions": decision.get("open_questions"),
            "note": "the reference is valid and sealed but the screen does not classify its document as an "
                    "independent one; it is not counted and not frozen"})

    intake_gate = {
        "requirement": f"at least {K.TARGET_DOCUMENTS} documents eligible as independent",
        "eligible": len(eligible),
        "excluded": screen["counts"][E.EXCLUDED],
        "pending_review": screen["counts"][E.PENDING_REVIEW],
        "annotated_not_admitted": annotated_not_admitted,
        "shortage": max(0, K.TARGET_DOCUMENTS - len(eligible)),
        "passed": len(eligible) >= K.TARGET_DOCUMENTS,
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

    # One selection, read by the gate, the categories, the freeze and the evaluator, so the twelve documents
    # that get measured are the twelve that passed - not a count that was taken somewhere else in the run.
    selection = K.select_cohort(eligible)
    cohort_gate = coverage_gate(selection, geometry=geometry)
    cohort = selection["selected"]

    # The descriptive categories answer a different question from the gate - what kind of corpus this is, not
    # whether it may run - and they are reported on the selected cohort, because the cohort that will be
    # measured is the one a later reader needs a description of.
    if geometry and intake_gate["passed"]:
        descriptive = K.coverage(cohort, geometry=cohort_geometry(screen["documents"]))
    else:
        descriptive = K.coverage(cohort, geometry=None)
        if geometry:
            for category in descriptive:
                if "ruled region" in category["category"]:
                    category["count"] = None
                    category["note"] = ("not measured: the intake gate failed, so no geometry was computed "
                                        "for these documents")
    gaps = [category for category in descriptive if category["underrepresented"]]

    hashes = P.source_hashes()
    sources = {"count": len(hashes), "sha256": K.digest(hashes),
               "pr103_drift": P.pr103_freeze(hashes, source_manifest)}
    frozen_documents = sorted(
        [{"id": (record["annotation"].get("document") or {}).get("id"),
          "region": K.region_of(record),
          "source_file": (record["annotation"].get("document") or {}).get("source_file"),
          "document_sha256": (record["annotation"].get("document") or {}).get("sha256"),
          "pages": (record["annotation"].get("document") or {}).get("pages"),
          "reference_sha256": (record["annotation"].get("freeze") or {}).get("reference_sha256"),
          "body_sha256": (record["annotation"].get("freeze") or {}).get("body_sha256"),
          "frozen_at": (record["annotation"].get("freeze") or {}).get("frozen_at")}
         for record in cohort],
        key=lambda entry: str(entry["id"]))
    freeze = {
        "documents": frozen_documents,
        "cohort_sha256": K.digest([entry["reference_sha256"] for entry in frozen_documents]),
        # PR167: the freeze names the composition, not only the members. Without this an evaluator could only
        # re-check that twelve references still seal, and could not see that one of them is a second US
        # invoice standing in for the missing FR one.
        "selection": {key: cohort_gate[key] for key in
                      ("quota", "counts", "report", "selected", "selected_ids", "selection_sha256",
                       "ordered_by", "reserve")},
        "gates": {"intake_gate": intake_gate["passed"], "annotation_gate": annotation_gate["passed"],
                  "coverage_gate": cohort_gate["passed"],
                  "note": "the freeze exists because all three passed; scripts/pr110_evaluate.py re-runs the "
                          "coverage and annotation conditions against these bytes"},
        # PR167: the authority the cohort was certified against is part of the freeze. An evaluation that runs
        # after the private registry moved would otherwise report the same twelve documents while standing on
        # a different claim about what the campaign had already used.
        "document_authority": {key: value for key, value in screen["document_authority"].items()},
        "production_sources": sources,
        "experimental_switches": K.experimental_switches_off(),
        "git": P.git_state(),
        "frozen_before_any_pipeline_run": True,
    }
    gates_passed = intake_gate["passed"] and annotation_gate["passed"] and cohort_gate["passed"]
    return {
        "review": "PR110 Phase 5 - benchmark readiness",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": READY if gates_passed else NOT_READY,
        "intake_gate": intake_gate,
        "annotation_gate": annotation_gate,
        "coverage_gate": cohort_gate,
        "coverage": descriptive,
        "coverage_gaps": [category["category"] for category in gaps],
        "blocking_findings": (
            ([] if screen["prior_use_inventory"]["present"] else
             [f"PR103's prior-use inventory is not on this machine ({screen['prior_use_inventory']['file']}), "
              "so no document can be shown never to have been used; the screen holds every candidate for "
              "review rather than treating an absent record as an empty one"])
            + ([] if screen["document_authority"]["available"] else
             ["the document authority is not available: "
              + str(screen["document_authority"]["reason"])])
            + ([] if not screen["document_authority"]["available"]
                 or screen["document_authority"]["complete"] else
               ["the document authority is present but incomplete, so no document can be certified as "
                "independent: " + str(screen["document_authority"]["reason"])])
            + ([] if intake_gate["passed"] else
               [f"{intake_gate['shortage']} more eligible documents are required; "
                f"{intake_gate['pending_review']} document(s) are pending review and "
                f"{intake_gate['excluded']} excluded"])
            + ([] if annotation_gate["passed"] else
               [f"{len(invalid)} reference(s) fail the validator or have a broken seal" if records else
                "no reference file exists yet: the annotation gate fails on an empty cohort rather than "
                "passing vacuously"])
            + ([] if cohort_gate["passed"] else
               [f"{cohort_gate['report']} — {NOT_READY}, " + finding for finding in cohort_gate["findings"]])),
        "freeze": freeze if gates_passed else None,
        "freeze_when_not_ready": None if gates_passed else {
            "note": "no cohort frozen: the gates failed, and a cohort of fewer than "
                    f"{K.TARGET_DOCUMENTS} documents is not a smaller version of a full one"},
        # the interpreter is named rather than assumed: a bare `python` can resolve to one without pytesseract,
        # and the command this prints is the owner's only route to the evaluation
        "launch_command": (f'"{sys.executable}" scripts/pr110_evaluate.py '
                           f"--freeze {K.display(ARTIFACT_DIR / 'pr110-phase5-cohort-freeze.json')}"
                           if gates_passed else None),
        "refusal": "this script exits with status 1 and writes no freeze when any gate fails; "
                   "scripts/pr110_evaluate.py refuses independently by re-checking the freeze it was given",
        "eligibility_screen": {key: screen[key] for key in
                               ("target_documents", "counts", "shortage", "screen_provenance",
                                "prior_use_inventory", "document_authority", "known_limitations",
                                "references_without_a_document")},
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
    coverage = report["coverage_gate"]
    print(f"coverage gate   : {'PASS' if coverage['passed'] else 'FAIL'} - {coverage['report']} "
          f"of {coverage['requirement']}")
    if coverage["reserve"]:
        print(f"  {len(coverage['reserve'])} candidate(s) are held as a named reserve and are not evaluated: "
              + ", ".join(f"{entry['id']} ({entry['region']})" for entry in coverage["reserve"][:10]))
    for finding in coverage["findings"]:
        print("   COVERAGE:", finding)
    if gate["annotated_not_admitted"]:
        print(f"  {len(gate['annotated_not_admitted'])} reference(s) are valid and sealed on documents the "
              "screen does not admit; they are not counted:")
        for entry in gate["annotated_not_admitted"][:20]:
            print("    ", entry["file"], entry["screen_status"],
                  entry["hard_reasons"] or entry["open_questions"])
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
