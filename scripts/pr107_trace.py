"""PR107 Phase 2: find the earliest stage that loses a printed description continuation line.

Nothing in ``src/`` is modified. The traced program is the PR106 `C2` stage-5 candidate exactly as PR106
published it (derived module, same rules, same frozen bytes, same pinned acquisition), plus two
logging-only probes:

* in ``invoice_evidence.from_tokens``, immediately inside the continuation branch, recording every operand
  of the production merge condition for the line under test;
* in ``ocr_consensus.build_consensus``, immediately before the continuation is accepted or discarded,
  recording what each family contributed and what the two-family rule then did.

Honesty gates, all enforced before any number is reported:

1. every anchor must occur exactly once, or the script refuses to run;
2. the traced payload digest must equal the published PR106 `C2` digest for the document - a probe that
   changes behaviour is a broken instrument, not a finding;
3. live Tesseract calls must stay 0: the evidence is the frozen PR106 cache.

Writes only inside the git-ignored results/product-review107/ directory and refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import evaluation_metrics as metrics  # noqa: E402
import pr105b_harness as H  # noqa: E402
import pr106_candidates as C  # noqa: E402
import pr106_harness as P  # noqa: E402

INVOICE_EVIDENCE_PATH = ROOT / "src" / "invoice_evidence.py"
OCR_CONSENSUS_PATH = ROOT / "src" / "ocr_consensus.py"
EXPECTED_NAME = "pr106-phase6-candidates-v3.json"
# PR107 keeps its verbatim traces in its own git-ignored directory (see .gitignore).
ARTIFACT_DIR = ROOT / "results" / "product-review107"

EVIDENCE_ANCHOR = (
    '        elif evidence.rows and starts and cell_texts.get("description")'
    ' and not any(values.values()):\n'
    '            last = evidence.rows[-1]\n'
)
EVIDENCE_PROBE = (
    '        elif evidence.rows and starts and cell_texts.get("description")'
    ' and not any(values.values()):\n'
    '            _pr107_probe(line, evidence.rows, starts, cell_texts, values)\n'
    '            last = evidence.rows[-1]\n'
)

CONSENSUS_ANCHOR = '        if row.description_confident and len(secondary) == 1:\n'
# The note sits at the anchor's own indentation, so it runs for every consensus row - including rows whose
# two-family rule returned nothing, which is the only way to tell "never agreed" from "agreed, then voided".
CONSENSUS_PROBE = (
    '        _pr107_secondary_note(row, secondary, continuation_matches, matches)\n'
    + CONSENSUS_ANCHOR
)
# The accept branch sees `secondary` only after the all-pairs compatibility filter, so the filter's own
# effect is invisible there. This second probe reads the value that `_agreed` returned, one statement
# earlier, which is what separates "the two-family rule never agreed" from "the filter voided an agreement".
CONSENSUS_AGREE_ANCHOR = '        if secondary:\n'
CONSENSUS_AGREE_PROBE = (
    '        _pr107_agreed(row, secondary, continuation_matches)\n'
    + CONSENSUS_AGREE_ANCHOR
)

EVIDENCE_FOOTER = r'''

# ---- PR107 continuation probe. Everything above is the production source with one logging call inserted
# inside the continuation branch; no condition, threshold or value changed. The operands below are
# re-evaluated from the same objects the production `if` reads, purely for the record. ----
_pr107_log: list[dict] = []


def _pr107_probe(line, rows, starts, cell_texts, values):
    last = rows[-1]
    description_start = starts.get("description")
    description_text = cell_texts.get("description") or ""
    gap = line.box[1] - last.line.box[3]
    tolerance = 3 * max(line.box[3] - line.box[1], last.line.box[3] - last.line.box[1])
    leftmost = min(token.box[0] for token in line.tokens) if line.tokens else None
    _pr107_log.append({
        "stage": "stage5",
        "page": line.page,
        "box": [round(value, 1) for value in line.box],
        "height": round(line.box[3] - line.box[1], 1),
        "text": line.text,
        "tokens": [{"text": token.text, "conf": token.confidence,
                    "box": [round(value, 1) for value in token.box]} for token in line.tokens],
        "cell_texts": dict(cell_texts),
        "values_nonempty": sorted(key for key, value in values.items() if value),
        "previous_row_description": last.description,
        "previous_row_secondary": last.secondary_description,
        "previous_row_box": [round(value, 1) for value in last.line.box],
        "previous_row_height": round(last.line.box[3] - last.line.box[1], 1),
        "same_page": line.page == last.line.page,
        "gap": round(gap, 1),
        "gap_tolerance": round(tolerance, 1),
        "gap_in_range": bool(0 <= gap <= tolerance),
        "description_start": description_start,
        "leftmost_token": round(leftmost, 1) if leftmost is not None else None,
        "x_delta": (round(abs(leftmost - description_start), 1)
                    if leftmost is not None and description_start is not None else None),
        "x_within_8": bool(leftmost is not None and description_start is not None
                           and abs(leftmost - description_start) <= 8),
        "min_token_confidence": min([token.confidence for token in line.tokens] or [None]),
        "all_confidence_at_least_60": bool(line.tokens)
                                      and all(token.confidence >= 60 for token in line.tokens),
        "description_usable": description_is_usable(description_text),
        "replacement_character": has_replacement_character(description_text),
        "continuation_excluded": bool(CONTINUATION_EXCLUDED.search(folded(description_text))),
        "excluded_match": (CONTINUATION_EXCLUDED.search(folded(description_text)).group(0)
                           if CONTINUATION_EXCLUDED.search(folded(description_text)) else None),
    })
'''

CONSENSUS_FOOTER = r'''

# ---- PR107 consensus probes: two logging calls, each reading the locals the existing branch reads at that
# point. They mutate nothing. `_pr107_agreed` records what the two-family rule returned; the note records
# what the accept/discard branch sees after the all-pairs compatibility filter. ----
_pr107_log: list[dict] = []
_pr107_last_agreed: dict[str, object] = {}


def _pr107_agreed(row, secondary, continuation_matches):
    _pr107_last_agreed["value"] = list(secondary)
    _pr107_last_agreed["families"] = sorted(continuation_matches)


def _pr107_secondary_note(row, secondary, continuation_matches, matches):
    observed = [
        candidate.secondary_description for entries in matches.values() for candidate in entries
        if candidate.secondary_description
    ]
    agreed = list(_pr107_last_agreed.get("value") or [])
    _pr107_log.append({
        "stage": "consensus",
        "primary_description": row.description,
        "description_confident": row.description_confident,
        "families_matched": sorted(matches),
        "families_with_secondary": {
            family: [candidate.secondary_description for candidate in entries
                     if candidate.secondary_description]
            for family, entries in matches.items()
            if any(candidate.secondary_description for candidate in entries)
        },
        "continuation_families": sorted(continuation_matches),
        "agreed_before_compatibility": agreed,
        "agreed_families": list(_pr107_last_agreed.get("families") or []),
        "secondary_candidates": list(secondary),
        "observed_values": observed,
        "accepted": bool(row.description_confident and len(secondary) == 1),
    })
'''


def instrument(path: Path, substitutions: list[tuple[str, str]], footer: str = ""):
    source = path.read_text(encoding="utf-8")
    for anchor, _ in substitutions:
        if source.count(anchor) != 1:
            raise SystemExit(f"anchor occurs {source.count(anchor)} times, expected 1 in {path.name}: "
                             f"{anchor.strip()[:72]!r} - the production source changed, re-read it first")
    text = source
    for anchor, probe in substitutions:
        text = text.replace(anchor, probe, 1)
    return text + footer, source


def load_traced_evidence(rules: dict):
    text, source = instrument(INVOICE_EVIDENCE_PATH, [(EVIDENCE_ANCHOR, EVIDENCE_PROBE)], EVIDENCE_FOOTER)
    module = C.load_derived(text, rules)
    module.pr107_logging_lines = len(text.splitlines()) - len(source.splitlines())
    return module


def load_traced_consensus():
    text, source = instrument(
        OCR_CONSENSUS_PATH,
        [(CONSENSUS_AGREE_ANCHOR, CONSENSUS_AGREE_PROBE), (CONSENSUS_ANCHOR, CONSENSUS_PROBE)],
        CONSENSUS_FOOTER,
    )
    import types

    module = types.ModuleType("pr107_traced_ocr_consensus")
    module.__file__ = str(OCR_CONSENSUS_PATH)
    sys.modules[module.__name__] = module
    exec(compile(text, str(OCR_CONSENSUS_PATH), "exec"), module.__dict__)
    module.pr107_logging_lines = len(text.splitlines()) - len(source.splitlines())
    return module


class Tracer:
    """Read-only observer of the two instrumented stages; both return production results untouched."""

    def __init__(self, evidence_module):
        self.evidence_module = evidence_module
        self.reads: list[dict] = []

    def _row(self, row):
        return {
            "description": row.description,
            "secondary_description": row.secondary_description,
            "cells": {key: [number.raw for number in values]
                      for key, values in row.cells.items() if values},
            "line_box": [round(value, 1) for value in row.line.box],
            "min_token_confidence": min([token.confidence for token in row.line.tokens] or [None]),
        }

    def _stash(self, evidence, builder, log):
        self.reads.append({
            "call_index": len(self.reads),
            "builder": builder,
            "rows_built": len(evidence.rows),
            "rows": [self._row(row) for row in evidence.rows],
            "continuation_probes": json.loads(json.dumps(log, default=str)),
        })
        return evidence

    def from_tsv(self, data, *, page: int = 0, locale=None):
        self.evidence_module._pr107_log.clear()
        evidence = self.evidence_module.from_tsv(data, page=page, locale=locale)
        return self._stash(evidence, "from_tsv", self.evidence_module._pr107_log)

    def from_tokens(self, tokens, *, locale=None, raw_lines=None, literal_columns=True):
        self.evidence_module._pr107_log.clear()
        evidence = self.evidence_module.from_tokens(tokens, locale=locale, raw_lines=raw_lines,
                                                    literal_columns=literal_columns)
        return self._stash(evidence, "from_tokens", self.evidence_module._pr107_log)


class TracedFreeze(C.Stage5Freeze):
    """Production consensus, replaced by its instrumented twin; the readings going in are recorded."""

    def __init__(self, *args, consensus_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.consensus_module = consensus_module
        self.consensus_traces: list[dict] = []

    def consensus_call(self, readings, *, confidence):
        self.consensus_module._pr107_log.clear()
        result = super().consensus_call(readings, confidence=confidence)
        self.consensus_traces.append({
            "families": {family: [row.description for row in evidence.rows]
                         for family, entries in readings.items() for evidence in entries},
            "rows_after": [row.description for row in result.rows],
            "probes": json.loads(json.dumps(list(self.consensus_module._pr107_log), default=str)),
        })
        return result


def stage5_verdict(probe) -> str:
    """Which of the eight production conjuncts the line failed, or `merged` if none."""
    failed = [key for key in ("same_page", "gap_in_range", "x_within_8", "all_confidence_at_least_60",
                              "description_usable") if probe[key] is False]
    if probe["replacement_character"]:
        failed.append("has_replacement_character")
    if probe["continuation_excluded"]:
        failed.append(f"continuation_excluded:{probe['excluded_match']}")
    return "merged" if not failed else "refused:" + ",".join(failed)


def stage7_verdict(probe) -> str:
    """Which of the three stage-7 safeguards decided this row's continuation, from the recorded operands."""
    if probe["accepted"]:
        return "accepted"
    if not probe["description_confident"]:
        return "dropped:primary_description_not_confident"
    if not probe["families_with_secondary"]:
        return "dropped:no_family_carried_a_continuation"
    if probe["agreed_before_compatibility"] and not probe["secondary_candidates"]:
        return "dropped:all_pairs_compatibility_filter"
    if not probe["agreed_before_compatibility"]:
        if len(probe["continuation_families"]) < 2:
            return ("dropped:intra_family_prefix_test"
                    if len(probe["families_with_secondary"]) != len(probe["continuation_families"])
                    else "dropped:two_family_agreement_never_reached")
        return "dropped:two_family_agreement_never_reached"
    if len(probe["secondary_candidates"]) > 1:
        return "dropped:ambiguous_continuations"
    return "dropped:unclassified"


def gate_census(doc: dict) -> dict:
    stage5 = {}
    for read in doc["reads"]:
        for probe in read["continuation_probes"]:
            verdict = stage5_verdict(probe)
            stage5[verdict] = stage5.get(verdict, 0) + 1
    stage7 = {}
    for trace in doc["consensus_traces"]:
        for probe in trace["probes"]:
            verdict = stage7_verdict(probe)
            stage7[verdict] = stage7.get(verdict, 0) + 1
    return {"stage5_continuation_lines": stage5, "stage7_row_decisions": stage7}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="layout-a-sumup-avoir-anon")
    parser.add_argument("--configuration", default="C2", choices=list(C.CONFIGURATIONS))
    parser.add_argument("--output-name", default="pr107-phase2-trace.json")
    parser.add_argument("--cache", default="pr105b", choices=("pr105b", "pr106"),
                        help="frozen OCR bytes: pr105b is the eleven-document corpus freeze PR106 "
                             "measured its candidates on; pr106 is the three-target capture pass")
    args = parser.parse_args()

    import invoice_evidence
    import ocr_recovery

    # The eleven-document corpus was frozen by PR105B; PR106's own cache holds only its three targets.
    cache_path = C.PR105B_CACHE if args.cache == "pr105b" else P.ARTIFACT_DIR / P.CACHE_NAME
    cache = H.read_json(cache_path)
    published = H.read_json(P.ARTIFACT_DIR / EXPECTED_NAME)
    documents = P.documents()
    names = [name.strip() for name in args.documents.split(",") if name.strip()]
    unknown = [name for name in names if name not in documents]
    if unknown:
        raise SystemExit(f"unknown documents: {unknown}")

    rules = C.CONFIGURATIONS[args.configuration]["rules"]
    evidence_module = load_traced_evidence(rules)
    consensus_module = load_traced_consensus()
    print(f"instrumented: stage 5 +{evidence_module.pr107_logging_lines} lines, "
          f"consensus +{consensus_module.pr107_logging_lines} lines, production files untouched")

    # Reference pass: production stage 5 and production consensus, recording the acquisition path and the
    # consensus calls that the candidate configuration must then follow.
    reference_freeze = H.Freeze("reference", dict(cache["entries"]))
    recordings: dict[str, dict] = {}
    reference: dict[str, dict] = {}
    with (patch.object(ocr_recovery, "from_tsv", invoice_evidence.from_tsv),
          patch.object(ocr_recovery, "from_tokens", invoice_evidence.from_tokens)):
        for name in names:
            reference[name] = H.run_document(name, documents[name], reference_freeze)
            recordings[name] = {"keys": list(reference_freeze.keys),
                                "consensus": list(reference_freeze.consensus)}

    traced: dict[str, dict] = {}
    tracer = Tracer(evidence_module)
    freeze = TracedFreeze("pinned", dict(cache["entries"]), recordings, consensus_module=consensus_module)
    with (patch.object(ocr_recovery, "from_tsv", tracer.from_tsv),
          patch.object(ocr_recovery, "from_tokens", tracer.from_tokens),
          patch.object(ocr_recovery, "build_consensus", consensus_module.build_consensus)):
        for name in names:
            evidence_module._pr106_fires.clear()
            runs = H.run_document(name, documents[name], freeze)
            traced[name] = {**runs, "reads": list(tracer.reads),
                            "consensus_traces": list(freeze.consensus_traces),
                            "rules_fired": dict(evidence_module._pr106_fires)}
            tracer.reads.clear()
            freeze.consensus_traces.clear()

    parity = []
    for name in names:
        expected = published["frozen"]["per_document"][name][args.configuration]["payload_sha256"]
        if traced[name]["payload_sha256"] != expected:
            parity.append({"document": name, "published": expected,
                           "traced": traced[name]["payload_sha256"]})
    if parity:
        raise SystemExit("the traced module does not reproduce PR106's published payload; "
                         "the probe changed behaviour: " + json.dumps(parity, indent=2))

    artifact = {
        "phase": f"PR107 Phase 2 - continuation-line trace under PR106 {args.configuration}",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": args.configuration,
        "rules": rules,
        "instrumentation": {
            "invoice_evidence_logging_lines": evidence_module.pr107_logging_lines,
            "ocr_consensus_logging_lines": consensus_module.pr107_logging_lines,
            "parity_with_published_pr106": "IDENTICAL",
            "production_files_modified": 0,
        },
        "evidence": {"cache_file": str(cache_path.relative_to(ROOT)),
                     "live_tesseract_calls": freeze.live_calls},
        "reference_baseline": {name: {"payload_sha256": reference[name]["payload_sha256"],
                                      "exported_rows": reference[name]["exported_rows"]} for name in names},
        "documents": {
            name: {
                "payload_sha256": traced[name]["payload_sha256"],
                "exported_rows": traced[name]["exported_rows"],
                "payload": traced[name]["payload"],
                "rules_fired": traced[name]["rules_fired"],
                "expected_rows": len(documents[name]["reference"]["line_items"] or []),
                "evaluation": metrics.evaluate(traced[name]["payload"], documents[name]["reference"]),
                "gates": gate_census(traced[name]),
                # Every consensus call in the run is reported; if they all reach the same verdict the
                # conclusion does not depend on which call fed the exported evidence.
                "consensus_verdicts_per_call": [
                    {"rows_after": trace["rows_after"],
                     "verdicts": [{"description": probe["primary_description"],
                                   "verdict": stage7_verdict(probe)} for probe in trace["probes"]]}
                    for trace in traced[name]["consensus_traces"]],
                "reads": traced[name]["reads"],
                "consensus_traces": traced[name]["consensus_traces"],
            } for name in names
        },
        "live_call_count": freeze.live_calls,
    }
    target = P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name)
    P.write_json(target, artifact)
    print(f"parity ok | traced payload = published {args.configuration} for {len(names)} document(s), "
          f"{freeze.live_calls} live calls")
    for name in names:
        rows = [row for row in (traced[name]["payload"].get("line_items") or [])]
        print(f"  {name}: {len(rows)} exported rows")
        for row in rows:
            print("    ", json.dumps({key: row.get(key) for key in
                                      ("description", "quantity", "unit_price", "total_ht")}, ensure_ascii=False))
        print("     gates:", json.dumps(gate_census(traced[name]), ensure_ascii=False))
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
