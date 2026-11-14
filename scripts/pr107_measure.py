"""PR107 Phases 4-6: measure a stage-5 multiline probe against production, PR106 C2, and the combination.

Four things are held fixed so that only the rule under test can move a number:

* **One evidence policy.** A production reference pass records the acquisition sequence of the frozen
  PR105B bytes; every configuration then replays it positionally (`Stage5Freeze`, pinned). Live Tesseract
  calls must stay 0 and the coupling counters must stay empty.
* **Production consensus and production grounding.** `ocr_consensus.build_consensus` and
  `invoice_grounding.reconcile` are never replaced, so a candidate cannot win by relaxing evidence validation.
* **Production scoring.** Every count comes from `src/evaluation_metrics.py` as published. The normalized
  comparison view is applied to *both* sides of that same evaluator; nothing in it is edited.
* **PR106 C2 reused verbatim** (`pr106_candidates.CONFIGURATIONS["C2"]["rules"]`), with its own parity gate.

Two parity gates run before anything is reported: configuration A must reproduce the published PR106 `P`
payload digest for all eleven documents and configuration C must reproduce the published `C2` digest. If a
gate fails the script stops - the measurement would otherwise describe a different program.

The multiline probe is a *parameterisation*, not a new behaviour: the two stage-5 continuation guards keep
their production values (8 px, confidence 60) unless a rule overrides them, so configuration B1/B2/B/D are
the only places those numbers differ from production, and the relaxations are exactly the two that PR107
Phase 2 found were never the binding constraint on layout-a.

Writes only inside the git-ignored results/product-review107/ directory and refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import types
import unicodedata
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import evaluation_metrics as metrics  # noqa: E402
import pr105b_harness as H  # noqa: E402
import pr106_candidates as C  # noqa: E402
import pr106_harness as P  # noqa: E402
import pr107_trace as T  # noqa: E402
from pr105b_factorial import incorrect_total, payload_diff, unbacked  # noqa: E402

INVOICE_EVIDENCE_PATH = ROOT / "src" / "invoice_evidence.py"
PUBLISHED_NAME = "pr106-phase6-candidates-v3.json"
ARTIFACT_DIR = ROOT / "results" / "product-review107"
CELL_FIELDS = ("description", "quantity", "unit_price", "total_ht")
VERDICTS = ("correct", "incorrect", "missing", "unverified")

# --------------------------------------------------------------------------- comparison convention

NORMALIZATION_VERSION = "pr107-desc-norm-v1"
# A standalone separator token printed between two spaces carries no content: it is how an annotator joined
# two printed lines. Inside a word (a hyphenated compound, a code) it is content and stays.
SEPARATOR_TOKEN = re.compile(r"(?<=\s)[\u2010-\u2015\u2212\u00b7\u2022\u2043*]\s+|(?<=\s)[\u2010-\u2015\u2212]\s")


def normalize_description(text):
    """Versioned comparison form of one description: formatting only, never content.

    NFKC compatibility folding, casefold, a non-breaking space written as a space, one space between
    tokens, and the removal of standalone separator tokens. Accents are *not* folded (a lost accent is a
    different word, not a style), the token order is never changed, no containment or similarity is used,
    and nothing is ever inserted. Two strings that differ in content stay different.

    None stays None: an absent description is absent evidence, and rewriting it to the empty string
    would let the derived view turn a "missing" cell into an "incorrect" one, or match two absences.
    """
    if text is None:
        return None
    value = unicodedata.normalize("NFKC", str(text)).replace("\u00a0", " ").casefold()
    value = SEPARATOR_TOKEN.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def normalized_pair(payload: dict, reference: dict) -> tuple[dict, dict]:
    """The derived comparison view: both sides rewritten through the same function, originals untouched."""
    def rewrite(source: dict) -> dict:
        copy = json.loads(json.dumps(source))
        for row in copy.get("line_items") or []:
            row["description"] = normalize_description(row.get("description"))
        return copy
    return rewrite(payload), rewrite(reference)


# --------------------------------------------------------------------------- multiline probe

CONTINUATION_ANCHOR = (
    '                and abs(min(token.box[0] for token in line.tokens) - description_start) <= 8\n'
    '                and all(token.confidence >= 60 for token in line.tokens)\n'
)
CONTINUATION_PROBE = (
    '                and abs(min(token.box[0] for token in line.tokens) - description_start)\n'
    '                <= _pr107_alignment_bound()\n'
    '                and all(token.confidence >= _pr107_continuation_confidence()\n'
    '                        for token in line.tokens)\n'
)

CONTINUATION_FOOTER = r'''

# ---- PR107 multiline probe. The two stage-5 continuation guards of the merge branch are read through
# these helpers, whose defaults are the production literals. An empty `_pr107_rules` therefore reproduces
# production exactly; nothing else in the branch changed, and no gate, threshold or value outside it moved. ----
_pr107_rules: dict = {}
_pr107_fires: dict = {}

PR107_PRODUCTION_ALIGNMENT_BOUND = 8
PR107_PRODUCTION_CONFIDENCE_FLOOR = 60


def _pr107_alignment_bound() -> int:
    value = int(_pr107_rules.get("continuation_alignment_bound", PR107_PRODUCTION_ALIGNMENT_BOUND))
    if value != PR107_PRODUCTION_ALIGNMENT_BOUND:
        _pr107_fires["continuation_alignment_bound"] = value
    return value


def _pr107_continuation_confidence() -> int:
    value = int(_pr107_rules.get("continuation_confidence_floor", PR107_PRODUCTION_CONFIDENCE_FLOOR))
    if value != PR107_PRODUCTION_CONFIDENCE_FLOOR:
        _pr107_fires["continuation_confidence_floor"] = value
    return value
'''


def load(source: str, structural_rules: dict, multiline_rules: dict) -> types.ModuleType:
    """Production source -> PR107 probe -> PR106 candidate rules, in that order, with all anchors checked."""
    if source.count(CONTINUATION_ANCHOR) != 1:
        raise SystemExit(
            f"continuation anchor occurs {source.count(CONTINUATION_ANCHOR)} times, expected 1; "
            "the production source changed, re-read it before measuring again")
    module = C.load_derived(source.replace(CONTINUATION_ANCHOR, CONTINUATION_PROBE, 1) + CONTINUATION_FOOTER,
                            structural_rules)
    module._pr107_rules = dict(multiline_rules)
    return module


C2_RULES = dict(C.CONFIGURATIONS["C2"]["rules"])
ALIGNMENT_RELAXATION = {"continuation_alignment_bound": 24}
CONFIDENCE_RELAXATION = {"continuation_confidence_floor": 0}
BOTH_RELAXATIONS = {**ALIGNMENT_RELAXATION, **CONFIDENCE_RELAXATION}

CONFIGURATIONS = {
    "A": {"description": "production baseline: no structural rule, no multiline relaxation",
          "structural": {}, "multiline": {}, "parity": "P"},
    "B1": {"description": "multiline only: the 8 px continuation alignment bound widened to 24 px",
           "structural": {}, "multiline": ALIGNMENT_RELAXATION, "parity": None},
    "B2": {"description": "multiline only: the confidence-60 continuation floor removed",
           "structural": {}, "multiline": CONFIDENCE_RELAXATION, "parity": None},
    "B": {"description": "multiline only: both continuation guards relaxed",
          "structural": {}, "multiline": BOTH_RELAXATIONS, "parity": None},
    "C": {"description": "PR106 C2 structural candidate, reused unchanged",
          "structural": C2_RULES, "multiline": {}, "parity": "C2"},
    "D": {"description": "combined: PR106 C2 plus both continuation guard relaxations",
          "structural": C2_RULES, "multiline": BOTH_RELAXATIONS, "parity": None},
}
CONFIG_ORDER = ("A", "B1", "B2", "B", "C", "D")


# --------------------------------------------------------------------------- per-configuration scores

def expected_cells(reference: dict) -> dict:
    rows = reference.get("line_items") or []
    return {field: sum(1 for row in rows if row.get(field) is not None) for field in CELL_FIELDS}


def precision_recall(evaluation: dict, reference: dict) -> dict:
    want = expected_cells(reference)
    out = {}
    for field in CELL_FIELDS:
        counts = evaluation["fields"]["by_field"][field]
        emitted = counts["correct"] + counts["incorrect"]
        out[field] = {
            "precision": counts["correct"] / emitted if emitted else None,
            "recall": counts["correct"] / want[field] if want[field] else None,
        }
    totals = {verdict: sum(evaluation["fields"]["by_field"][field][verdict] for field in CELL_FIELDS)
              for verdict in VERDICTS}
    emitted = totals["correct"] + totals["incorrect"]
    out["__all__"] = {"precision": totals["correct"] / emitted if emitted else None,
                      "recall": totals["correct"] / sum(want.values())}
    return out


def description_census(payload: dict, reference: dict) -> dict:
    """Every exported description, in both views, with which reference row it would be equal to.

    A normalized match that lands on a different reference row than the strict match is a flag: the
    convention would then be doing more than erasing formatting.
    """
    exported = [row or {} for row in (payload.get("line_items") or [])]
    gold = [row or {} for row in (reference.get("line_items") or [])]
    strict_match = metrics.match_rows(exported, gold)
    norm_payload, norm_reference = normalized_pair(payload, reference)
    norm_exported = [row or {} for row in (norm_payload.get("line_items") or [])]
    norm_gold = [row or {} for row in (norm_reference.get("line_items") or [])]
    norm_match = metrics.match_rows(norm_exported, norm_gold)

    def equality(rows_a, rows_b):
        return [[bool(a.get("description")) and bool(b.get("description"))
                 and metrics._text(a.get("description")) == metrics._text(b.get("description"))
                 for b in rows_b] for a in rows_a]

    strict_equal = equality(exported, gold)
    norm_equal = equality(norm_exported, norm_gold)
    pairs = {pair["export_index"]: pair for pair in strict_match["pairs"]}
    norm_pairs = {pair["export_index"]: pair for pair in norm_match["pairs"]}
    rows = []
    for index, export in enumerate(exported):
        moved = (index in norm_pairs and index in pairs
                 and norm_pairs[index]["gold_index"] != pairs[index]["gold_index"])
        rows.append({
            "export_index": index,
            "exported": export.get("description"),
            "exported_normalized": normalize_description(export.get("description")) or "",
            "strict_equal_to_reference_rows": [i for i, ok in enumerate(strict_equal[index]) if ok],
            "normalized_equal_to_reference_rows": [i for i, ok in enumerate(norm_equal[index]) if ok],
            "strict_pair": pairs.get(index, {}).get("gold_index"),
            "normalized_pair": norm_pairs.get(index, {}).get("gold_index"),
            "strict_identity": pairs.get(index, {}).get("identity"),
            "normalized_identity": norm_pairs.get(index, {}).get("identity"),
            "pair_moved_under_normalization": moved,
        })
    return {
        "rows": rows,
        "exact_matches_strict": sum(1 for row in rows if row["strict_equal_to_reference_rows"]),
        "exact_matches_normalized": sum(1 for row in rows if row["normalized_equal_to_reference_rows"]),
        "pairs_moved": sum(1 for row in rows if row["pair_moved_under_normalization"]),
        "reference_forms_changed": sum(
            1 for row in (reference.get("line_items") or []) if row.get("description")
            and normalize_description(row["description"]) != metrics._text(row["description"])),
    }


def score(runs: dict, fires: dict, documents: dict) -> dict:
    per_document = {}
    for name in documents:
        per_document[name] = {}
        baseline = runs["A"][name]["payload"]
        for config in CONFIG_ORDER:
            payload = runs[config][name]["payload"]
            reference = documents[name]["reference"]
            strict = metrics.evaluate(payload, reference)
            norm_payload, norm_reference = normalized_pair(payload, reference)
            normalized = metrics.evaluate(norm_payload, norm_reference)
            per_document[name][config] = {
                "payload_sha256": runs[config][name]["payload_sha256"],
                "exported_rows": len(payload.get("line_items") or []),
                "rows_recovered_vs_A": len(payload.get("line_items") or [])
                                       - len(baseline.get("line_items") or []),
                "strict": {
                    "rows": {key: strict["rows"][key] for key in
                             ("expected", "exported", "correctly_matched", "matched_total",
                              "identity_by_amounts_only", "missing", "duplicate", "misassociated",
                              "unsupported")},
                    "fully_reconstructed_rows": sum(
                        1 for pair in strict["pairs"]
                        if all(pair["verdicts"][field] in ("correct", "unverified") for field in CELL_FIELDS)),
                    "cells": {field: {verdict: strict["fields"]["by_field"][field][verdict]
                                      for verdict in VERDICTS} for field in CELL_FIELDS},
                    "precision_recall": precision_recall(strict, reference),
                    "incorrect_total": incorrect_total(strict),
                    "unsupported_nonnull": unbacked(payload, reference),
                },
                "normalized": {
                    "rows": {key: normalized["rows"][key] for key in
                             ("expected", "exported", "correctly_matched", "matched_total",
                              "identity_by_amounts_only", "missing", "duplicate", "misassociated",
                              "unsupported")},
                    "fully_reconstructed_rows": sum(
                        1 for pair in normalized["pairs"]
                        if all(pair["verdicts"][field] in ("correct", "unverified") for field in CELL_FIELDS)),
                    "cells": {field: {verdict: normalized["fields"]["by_field"][field][verdict]
                                      for verdict in VERDICTS} for field in CELL_FIELDS},
                    "precision_recall": precision_recall(normalized, norm_reference),
                    "incorrect_total": incorrect_total(normalized),
                    "description_census": description_census(payload, reference),
                },
                "diff_from_A": payload_diff(baseline, payload),
                "rules_fired": fires[config][name],
            }

    aggregate = {}
    for config in CONFIG_ORDER:
        view = {}
        for kind in ("strict", "normalized"):
            cells = {field: {verdict: sum(per_document[name][config][kind]["cells"][field][verdict]
                                          for name in documents) for verdict in VERDICTS}
                     for field in CELL_FIELDS}
            cells_correct = sum(cells[field]["correct"] for field in CELL_FIELDS)
            emitted = sum(cells[field]["correct"] + cells[field]["incorrect"] for field in CELL_FIELDS)
            expected_total = sum(sum(expected_cells(documents[name]["reference"]).values())
                                 for name in documents)
            view[kind] = {
                "exported_rows": sum(per_document[name][config]["exported_rows"] for name in documents),
                "rows_correctly_matched": sum(per_document[name][config][kind]["rows"]["correctly_matched"]
                                              for name in documents),
                "fully_reconstructed_rows": sum(per_document[name][config][kind]["fully_reconstructed_rows"]
                                                for name in documents),
                "cells": cells,
                "cells_correct": cells_correct,
                "cells_incorrect": sum(cells[field]["incorrect"] for field in CELL_FIELDS),
                "field_precision": cells_correct / emitted if emitted else None,
                "field_recall": cells_correct / expected_total if expected_total else None,
                "description_exact_matches": sum(
                    per_document[name][config]["normalized"]["description_census"]
                    ["exact_matches_strict" if kind == "strict" else "exact_matches_normalized"]
                    for name in documents),
                "incorrect_total": sum(per_document[name][config][kind]["incorrect_total"]
                                       for name in documents),
                "unsupported_nonnull_values": sum(
                    len(values) for name in documents
                    for values in per_document[name][config][kind]["unsupported_nonnull"].values()
                    if values) if kind == "strict" else None,
                "pairs_moved_by_normalization": sum(
                    per_document[name][config]["normalized"]["description_census"]["pairs_moved"]
                    for name in documents) if kind == "normalized" else None,
            }
        aggregate[config] = view
    return {"per_document": per_document, "aggregate": aggregate}


def retention_ceiling(documents: dict, runs: dict, traces_path: Path) -> dict | None:
    """What keeping the value production's own two-family rule already produced would have scored.

    Built only from the Phase 2 consensus probes: for each row whose continuation `_agreed` returned as a
    single candidate and the all-pairs compatibility filter then voided, that recorded candidate is spliced
    onto the exported description and the row is re-scored. The text comes from production's intermediate,
    never from the reference, so this measures headroom rather than inventing an answer.
    """
    if not traces_path.exists():
        return None
    traces = H.read_json(traces_path)["documents"]
    out = {}
    for name, trace in traces.items():
        if name not in documents:
            continue
        payload = json.loads(json.dumps(runs["D"][name]["payload"]))
        applied = []
        for call in trace["consensus_traces"]:
            for probe in call["probes"]:
                agreed = probe["agreed_before_compatibility"]
                if len(agreed) != 1 or probe["secondary_candidates"] or not probe["description_confident"]:
                    continue
                for row in payload.get("line_items") or []:
                    if metrics._text(row.get("description")) != metrics._text(probe["primary_description"]):
                        continue
                    if metrics._text(agreed[0]) in metrics._text(row.get("description")):
                        break  # an earlier consensus call already contributed this candidate
                    row["description"] = f"{row['description']} {agreed[0]}"
                    applied.append({"exported_after_splice": row["description"],
                                    "spliced_from": "production _agreed intermediate (trace probe)",
                                    "voided_by": T.stage7_verdict(probe)})
                    break
        if not applied:
            continue
        reference = documents[name]["reference"]
        strict = metrics.evaluate(payload, reference)
        norm_payload, norm_reference = normalized_pair(payload, reference)
        normalized = metrics.evaluate(norm_payload, norm_reference)
        out[name] = {
            "splices": applied,
            "payload": payload,
            "strict": {"rows_correctly_matched": strict["rows"]["correctly_matched"],
                       "cells_correct": strict["fields"]["correct"],
                       "cells_incorrect": strict["fields"]["incorrect"]},
            "normalized": {"rows_correctly_matched": normalized["rows"]["correctly_matched"],
                           "cells_correct": normalized["fields"]["correct"],
                           "cells_incorrect": normalized["fields"]["incorrect"]},
            "configuration_D": {
                "strict": {"rows_correctly_matched":
                           metrics.evaluate(runs["D"][name]["payload"], reference)["rows"]["correctly_matched"],
                           "cells_correct": metrics.evaluate(runs["D"][name]["payload"],
                                                             reference)["fields"]["correct"]},
            },
        }
    return out or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default=",".join(sorted(P.documents())))
    parser.add_argument("--output-name", default="pr107-phase5-measurement.json")
    parser.add_argument("--traces", default=str(ARTIFACT_DIR / "pr107-phase2-trace-corpus.json"),
                        help="Phase 2 trace artifact, used only for the retention-ceiling measurement")
    args = parser.parse_args()

    import invoice_evidence
    import ocr_recovery

    cache = H.read_json(C.PR105B_CACHE)
    published = H.read_json(P.ARTIFACT_DIR / PUBLISHED_NAME)
    documents = P.documents()
    names = [name.strip() for name in args.documents.split(",") if name.strip()]
    unknown = [name for name in names if name not in documents]
    if unknown:
        raise SystemExit(f"unknown documents: {unknown}")
    documents = {name: documents[name] for name in names}
    source = INVOICE_EVIDENCE_PATH.read_text(encoding="utf-8")

    # --- reference pass: production stage 5 and production consensus, recording the acquisition sequence.
    reference_freeze = H.Freeze("reference", dict(cache["entries"]))
    recordings: dict[str, dict] = {}
    production: dict[str, dict] = {}
    with (patch.object(ocr_recovery, "from_tsv", invoice_evidence.from_tsv),
          patch.object(ocr_recovery, "from_tokens", invoice_evidence.from_tokens)):
        for name in names:
            production[name] = H.run_document(name, documents[name], reference_freeze)
            recordings[name] = {"keys": list(reference_freeze.keys),
                                "consensus": list(reference_freeze.consensus)}

    diverging = [name for name in names
                 if production[name]["payload_sha256"]
                 != published["frozen"]["per_document"][name]["P"]["payload_sha256"]]
    if diverging:
        raise SystemExit("the production reference pass diverges from the published PR106 P baseline: "
                         + json.dumps(diverging, indent=2))

    # --- candidate passes over the identical frozen evidence
    runs: dict[str, dict] = {}
    fires: dict[str, dict] = {}
    timings: dict[str, float] = {}
    live_calls: dict[str, int] = {}
    for config in CONFIG_ORDER:
        module = load(source, CONFIGURATIONS[config]["structural"], CONFIGURATIONS[config]["multiline"])
        freeze = C.Stage5Freeze("pinned", dict(cache["entries"]), recordings)
        runs[config] = {}
        fires[config] = {}
        started = time.perf_counter()
        with (patch.object(ocr_recovery, "from_tsv", module.from_tsv),
              patch.object(ocr_recovery, "from_tokens", module.from_tokens)):
            for name in names:
                module._pr106_fires.clear()
                module._pr107_fires.clear()
                runs[config][name] = H.run_document(name, documents[name], freeze)
                fires[config][name] = {**dict(module._pr106_fires), **dict(module._pr107_fires)}
        timings[config] = round(time.perf_counter() - started, 1)
        live_calls[config] = freeze.live_calls
        print(f"{config}: {timings[config]}s over {len(names)} documents, "
              f"{sum(1 for v in fires[config].values() if v)} documents with a rule firing, "
              f"{freeze.live_calls} live OCR calls")

    gates = []
    for config in CONFIG_ORDER:
        expected = CONFIGURATIONS[config]["parity"]
        if expected is None:
            continue
        bad = [{"document": name,
                "published": published["frozen"]["per_document"][name][expected]["payload_sha256"],
                "measured": runs[config][name]["payload_sha256"]}
               for name in names
               if runs[config][name]["payload_sha256"]
               != published["frozen"]["per_document"][name][expected]["payload_sha256"]]
        if bad:
            gates.append({"configuration": config, "expected_to_reproduce": expected, "documents": bad})
    if gates:
        raise SystemExit("a parity gate failed; the harness is not measuring the published program:\n"
                         + json.dumps(gates, indent=2))

    coupling = {}
    for config in CONFIG_ORDER:
        coupling[config] = {
            "off_sequence_calls": {name: runs[config][name]["off_sequence_calls"] for name in names
                                   if runs[config][name]["off_sequence_calls"]},
            "suppressed_calls": {name: runs[config][name]["suppressed_calls"] for name in names
                                 if runs[config][name]["suppressed_calls"]},
            "consensus_calls_beyond_recorded": {
                name: runs[config][name]["consensus_calls_beyond_recorded"] for name in names
                if runs[config][name]["consensus_calls_beyond_recorded"]},
        }

    scored = score(runs, fires, documents)
    identical = {config: [name for name in names
                          if runs[config][name]["payload_sha256"] == runs["A"][name]["payload_sha256"]]
                 for config in CONFIG_ORDER if config != "A"}

    artifact = {
        "phase": "PR107 Phase 4-6 - multiline continuation probe, PR106 C2, and the combination",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instrumentation": {
            "production_source": str(INVOICE_EVIDENCE_PATH.relative_to(ROOT)),
            "production_sha256": H.sha256_file(INVOICE_EVIDENCE_PATH),
            "consensus_module_modified": False,
            "grounding_module_modified": False,
            "evaluator_modified": False,
            "pr106_c2_rules_reused_verbatim": C2_RULES,
            "multiline_probe": {"anchor": CONTINUATION_ANCHOR.strip(),
                                "production_values": {"alignment_bound": 8, "confidence_floor": 60}},
            "parity_gates": {"A_vs_published_P": "IDENTICAL", "C_vs_published_C2": "IDENTICAL"},
        },
        "comparison_convention": {
            "version": NORMALIZATION_VERSION,
            "operations": ["NFKC", "casefold", "nbsp -> space", "collapse whitespace",
                           "drop standalone separator tokens"],
            "explicitly_not_done": ["accent folding", "token reordering", "containment",
                                    "similarity scoring", "inserting punctuation", "editing references"],
            "references_modified": 0,
            "derived_view": "both sides of the same production evaluator, originals preserved",
        },
        "evidence": {
            "cache_file": str(C.PR105B_CACHE.relative_to(ROOT)),
            "unique_cached_calls": len(cache["entries"]),
            "live_tesseract_calls": {"reference_pass": reference_freeze.live_calls,
                                     "candidate_passes": live_calls},
            "documents": len(names),
        },
        "configurations": {config: CONFIGURATIONS[config]["description"] for config in CONFIG_ORDER},
        "rules_by_configuration": {config: {"structural": CONFIGURATIONS[config]["structural"],
                                            "multiline": CONFIGURATIONS[config]["multiline"]}
                                   for config in CONFIG_ORDER},
        "seconds_by_configuration": timings,
        "acquisition_coupling": coupling,
        "payload_identical_to_A": identical,
        "retention_ceiling": retention_ceiling(documents, runs, Path(args.traces)),
        "aggregate": scored["aggregate"],
        "per_document": scored["per_document"],
    }
    target = P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name)
    P.write_json(target, artifact)

    print("\naggregate (strict | normalized description comparison)")
    for config in CONFIG_ORDER:
        row = scored["aggregate"][config]
        print(f"  {config}: rows={row['strict']['exported_rows']} "
              f"matched={row['strict']['rows_correctly_matched']}|{row['normalized']['rows_correctly_matched']} "
              f"full={row['strict']['fully_reconstructed_rows']}|{row['normalized']['fully_reconstructed_rows']} "
              f"cells={row['strict']['cells_correct']}|{row['normalized']['cells_correct']} "
              f"incorrect={row['strict']['cells_incorrect']}|{row['normalized']['cells_incorrect']} "
              f"desc_exact={row['strict']['description_exact_matches']}|{row['normalized']['description_exact_matches']}")
    for config in CONFIG_ORDER:
        if config != "A":
            print(f"  {config} identical to A on {len(artifact['payload_identical_to_A'][config])} "
                  f"of {len(names)} documents: "
                  f"{json.dumps(artifact['payload_identical_to_A'][config], ensure_ascii=False)}")
    for name, value in (artifact["retention_ceiling"] or {}).items():
        print(f"  retention ceiling {name}: D matched "
              f"{value['configuration_D']['strict']['rows_correctly_matched']} -> "
              f"{value['strict']['rows_correctly_matched']} strict / "
              f"{value['normalized']['rows_correctly_matched']} normalized; cells "
              f"{value['configuration_D']['strict']['cells_correct']} -> "
              f"{value['strict']['cells_correct']} / {value['normalized']['cells_correct']}")
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
