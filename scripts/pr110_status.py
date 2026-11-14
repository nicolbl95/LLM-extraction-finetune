"""PR110 Phase 1: verify the state PR109 left, and measure what PR110 can reuse.

PR109 ended with a data blocker, not a code blocker: 0 of the 12 required independent invoices existed.
PR110 is benchmark preparation, so before adding anything it re-measures the ground PR110 must not disturb:

* the production sources are re-hashed and diffed twice - against the PR103 manifest (the campaign's freeze)
  and against PR109 Phase 1's own recorded hashes (what PR110 specifically must not change);
* every experimental path is read through the predicate production consults, and the full-precision Qwen
  question is answered from RAM and CUDA availability as PR106 measured it;
* the reuse map is built by importing each candidate module and asking it for the attributes PR110 calls,
  so a reused component that has drifted shows up here instead of breaking a later phase.

The command doubles as the owner's status readout: `--print` shows how many invoices and references are
currently supplied and how far the cohort is from the twelve-document target.

0 OCR calls, 0 model calls, no document is read for text. Writes only under `results/product-review110/`
and refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr105b_harness as H  # noqa: E402
import pr106_harness as P  # noqa: E402
import pr110_harness as K  # noqa: E402

PR109_BASELINE = ROOT / "results" / "product-review109" / "pr109-phase1-baseline.json"


def against_pr109(hashes: dict[str, dict]) -> dict:
    """PR110's own freeze anchor: the same sources PR109 Phase 1 recorded, re-hashed now."""
    if not PR109_BASELINE.is_file():
        return {"available": False, "note": f"not found: {PR109_BASELINE.relative_to(ROOT).as_posix()}"}
    recorded = json.loads(PR109_BASELINE.read_text(encoding="utf-8"))["production_source_hashes"]

    def value(entry):
        return entry if isinstance(entry, str) else entry["sha256"]

    drifted = sorted(name for name, entry in recorded.items()
                     if name in hashes and hashes[name]["sha256"] != value(entry))
    added = sorted(name for name in hashes if name not in recorded)
    removed = sorted(name for name in recorded if name not in hashes)
    return {
        "available": True,
        "baseline": PR109_BASELINE.relative_to(ROOT).as_posix(),
        "sources_recorded_by_pr109": len(recorded),
        "sources_hashed_now": len(hashes),
        "drifted": drifted,
        "added_since_pr109": added,
        "removed_since_pr109": removed,
        "production_files_changed": len(drifted) + len(added) + len(removed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="pr110-phase1-state.json")
    parser.add_argument("--print", dest="echo", action="store_true",
                       help="also print the owner-facing intake status")
    args = parser.parse_args()
    target = P.guard(K.ARTIFACT_DIR, K.ARTIFACT_DIR / args.output_name)

    hashes = P.source_hashes()
    integrity_block = H.read_json(PR109_BASELINE)["integrity"] if PR109_BASELINE.is_file() else None
    git = P.git_state()
    switches = K.experimental_switches_off()
    intake = K.intake_present()
    inventory = K.component_inventory()
    pr103 = P.pr103_freeze(hashes)
    pr109 = against_pr109(hashes)
    model_guard = P.environment().get("full_precision_qwen_on_cpu")

    artifact = {
        "phase": "PR110 Phase 1 - existing state verified and reusable components measured",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission": ("make collecting, annotating, validating and evaluating twelve independent invoices "
                    "straightforward for the project owner; this is benchmark preparation, not OCR "
                    "optimization"),
        "inherited_from_pr109": {
            "verdict": "BLOCKED_AWAITING_INDEPENDENT_INVOICES",
            "eligible_documents": 0,
            "target_documents": K.TARGET_DOCUMENTS,
            "shortage": K.TARGET_DOCUMENTS,
            "nature": "a data blocker: the intake machinery exists and ran, the documents do not exist",
        },
        "git": git,
        "git_note": "PR110 works on top of the user's existing uncommitted changes and preserves them; the "
                    "modified tracked files listed above are PR82-PR109 work, not PR110 work",
        "production_baseline": {
            "sources_hashed": len(hashes),
            "pr103_freeze": pr103,
            "pr109_freeze": pr109,
            "production_extraction_changed_by_pr110": pr109.get("production_files_changed"),
            "integrity_as_pr109_recorded": integrity_block,
            "integrity_measured_now": _integrity_now(),
        },
        "experimental_switches": switches,
        "model_guard": model_guard,
        "model_guard_note": "PR106's measurement: float32 7B weights are ~28 GB, this machine has a small "
                            "fraction of that available and no CUDA device, so a full-precision Qwen "
                            "cannot initialize by accident",
        "intake": intake,
        "gitignore": K.gitignore_covered(),
        "component_inventory": inventory,
        "components_reused_without_duplication": sorted(
            name for name, entry in inventory.items()
            if entry["all_reused_attributes_present"] and entry["importable"]),
        "components_modified_by_pr110": sorted(
            name for name, entry in inventory.items() if entry["modified_by_pr110"]),
        "not_reused_decisions": {name: entry["not_reused_by_pr110"]
                                 for name, entry in inventory.items() if entry.get("not_reused_by_pr110")},
        "ocr_calls": 0,
        "model_calls": 0,
        "documents_read_for_text": 0,
    }
    artifact["verdict"] = (
        "STATE_VERIFIED" if pr109.get("available") and pr109["production_files_changed"] == 0
        and switches["all_off"] else "CHECK REQUIRED")
    P.write_json(target, artifact)

    print(f"branch {git['branch']} @ {git['head'][:7]} | modified tracked {len(git['modified_tracked'])}")
    print(f"production sources: {len(hashes)} hashed | PR103 drift: {pr103['drifted'] or 'none'} "
          f"| changed since PR109: {pr109.get('production_files_changed')}")
    print(f"experimental switches all off: {switches['all_off']} | model guard: "
          f"{(model_guard or {}).get('verdict', 'not measured')}")
    print(f"intake: {intake['invoice_count']} invoice file(s), {intake['reference_count']} reference(s) "
          f"| target {K.TARGET_DOCUMENTS} | shortage {max(0, K.TARGET_DOCUMENTS - intake['reference_count'])}")
    unavailable = [name for name, entry in inventory.items()
                   if not entry["all_reused_attributes_present"] or not entry["importable"]]
    print(f"reuse map: {len(inventory)} components probed, "
          f"{len(artifact['components_reused_without_duplication'])} reusable as-is, "
          f"drifted or unimportable: {unavailable or 'none'}")
    if args.echo:
        print(json.dumps({"intake": intake, "gitignore": artifact["gitignore"]}, indent=1))
    print("verdict", artifact["verdict"])
    print("wrote", target)
    return 0


def _integrity_now() -> dict:
    """PR109's own integrity block recomputed on today's source, so the two can be compared."""
    from pr105b_capture import integrity

    block = integrity(H.PRODUCTION_SOURCE_PATH.read_text(encoding="utf-8"))
    block["verdict"] = ("FROZEN" if not block["pr103_source_drift"]
                        and block["ocr_configuration_matches_pr103"]
                        and block["ocr_consensus_matches_pr103"]
                        and not block["dual_ocr_experiment_enabled"] else "NOT FROZEN")
    return block


if __name__ == "__main__":
    raise SystemExit(main())
