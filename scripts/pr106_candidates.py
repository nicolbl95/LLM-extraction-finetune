"""PR106 Phase 5-6: measure the smallest stage-5 structural candidates on identical frozen evidence.

Production code is never modified. Each candidate is a *derived copy* of `src/invoice_evidence.py`: the
module is exec'd from the production source text, a candidate changes behaviour only through the five
anchored substitutions in `SUBSTITUTIONS` and the two module-global rebindings in `VOCABULARY`, and the
helpers that implement the rules live in `FOOTER` and read one rules dictionary. Everything else - the
two-family consensus minimum, the monetary confidence gates, the arithmetic recovery, the operand veto, the
U+FFFD rejection and every grounding proof - stays production code, so a rule that never fires on this
corpus cannot silently diverge.

Every rule is a statement about a *printed column header*, never about a document: a bare "Prix" is read as
the unit price only inside a header that also prints a VAT column and an unqualified amount column, and such
an amount column is reclassified as the gross column so that a VAT-inclusive line amount can never be
exported as line HT. No rule ever turns a null into a value: where the amount column is gross the exported
`total_ht` stays null, exactly as the reference does.

Honesty gates, enforced before any number is reported:

1. configuration `P` (derived module, rules empty) must reproduce every PR105B payload digest - otherwise
   the derivation itself changed behaviour and no candidate can be trusted;
2. every OCR call is replayed positionally against the recorded production sequence of that document;
   pixels a document never recorded are answered with a no-text reading and counted, never served;
3. `suppressed_calls`, `off_sequence_calls` and extra consensus calls are reported per configuration: a
   stage-5 rule that changes which pixels the pipeline asks for is coupled to acquisition, and that is
   disclosed, not hidden;
4. the per-document rule-fire counters show whether a rule was exercised at all, so "inert" and "measured"
   are never conflated.

Writes only inside the git-ignored results/product-review106/ directory and refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import evaluation_metrics as metrics  # noqa: E402
import pr105b_harness as H  # noqa: E402
import pr106_harness as P  # noqa: E402
from pr105b_factorial import incorrect_total, payload_diff, unbacked  # noqa: E402

INVOICE_EVIDENCE_PATH = ROOT / "src" / "invoice_evidence.py"
PR105B_CACHE = ROOT / "results" / "product-review105b" / "pr105b-ocr-cache.json"

# --------------------------------------------------------------------------- substitutions
# Each anchor is checked to occur exactly once, so a production edit that moves a gate makes this harness
# stop instead of silently measuring something else.

SUBSTITUTIONS: list[tuple[str, str]] = [
    (
        '        partial_header = (\n'
        '            all(matches[key] for key in ("description", "quantity", "unit_price"))\n'
        '            and re.search(OPTIONAL_HEADERS["gross_amount"], normalized)\n'
        '        )\n',
        '        partial_header = _pr106_partial_header(matches, normalized, line.text)\n',
    ),
    (
        '                for key, pattern in OPTIONAL_HEADERS.items():\n'
        '                    for index in range(len(line.tokens)):\n'
        '                        suffix = " ".join(token.text for token in line.tokens[index:])\n'
        '                        if re.match(pattern, folded(suffix)):\n',
        '                for key, pattern in _pr106_optional_headers(line.text).items():\n'
        '                    for index in range(len(line.tokens)):\n'
        '                        suffix = " ".join(token.text for token in line.tokens[index:])\n'
        '                        if re.match(pattern, folded(suffix)):\n',
    ),
    (
        '            required_cells = ("quantity", "unit_price", "total_ht")\n',
        '            required_cells = _pr106_required_cells(set(columns) | set(starts))\n',
    ),
    (
        '                valid_header = (\n'
        '                    required.issubset(starts)\n'
        '                    and len(set(starts.values())) == len(starts)\n'
        '                )\n',
        '                starts = _pr106_relabel_amount_column(starts, line.text)\n'
        '                valid_header = (\n'
        '                    required.issubset(starts)\n'
        '                    and len(set(starts.values())) == len(starts)\n'
        '                )\n',
    ),
    (
        '        if sum(bool(values[key]) for key in ("quantity", "unit_price", "total_ht")) >= 2:\n',
        '        if sum(bool(values[key]) for key in _pr106_monetary_keys(\n'
        '                set(columns) | set(starts))) >= 2:\n',
    ),
]

# Module globals rebound after exec: every internal call site resolves them at call time, so one binding
# per predicate is exactly what a production edit to that predicate would do.
VOCABULARY = (("_header_matches", "_pr106_header_matches"),
              ("is_line_total_header", "_pr106_is_line_total_header"))

FOOTER = r'''

# ---- PR106 candidate rules. Everything above this line is the production source, unchanged except for the
# four anchored substitutions and the two global rebindings at the end. The helpers below read
# `_pr106_rules`, which is empty for configuration P. No threshold, confidence, consensus, arithmetic or
# grounding rule is touched, and no rule turns a null into a value. ----
_pr106_rules: dict = {}
_pr106_fires: dict = {}

_PR106_PRICE = r"\b(?:prix|price)\b"
_PR106_VAT = r"\b(?:tva|vat|tax)\b"
_PR106_AMOUNT = r"\b(?:montant|amount)\b"
_PR106_QUALIFIED = r"\b(?:ht|ttc|net)\b|\bhors\s+taxe\b|\btax\s+inclusive\b"
_PR106_MONETARY = ("quantity", "unit_price", "total_ht")


def _pr106_fire(key: str) -> None:
    _pr106_fires[key] = _pr106_fires.get(key, 0) + 1


def _pr106_context(text: str) -> dict:
    """The column families announced by the very string under test.

    The condition is evaluated on the string being matched - a whole header line, a token suffix or a single
    cell - so the rule cannot depend on which call site asked, and a bare "Prix" alone never qualifies: it
    needs a VAT column and an amount column printed in the same string.
    """
    normalized = _header_text(text)
    return {
        "price": bool(re.search(_PR106_PRICE, normalized)),
        "vat": bool(re.search(_PR106_VAT, normalized)),
        "amount": bool(re.search(_PR106_AMOUNT, normalized)),
        "qualified": bool(re.search(_PR106_QUALIFIED, normalized)),
    }


def _pr106_gross_amount_column(text: str) -> bool:
    """An unqualified amount column printed next to a price and a VAT column is a gross column."""
    if not _pr106_rules.get("ambiguous_amount_is_gross"):
        return False
    context = _pr106_context(text)
    return bool(context["amount"] and context["vat"] and context["price"] and not context["qualified"])


def _pr106_bare_price_column(text: str, at_start: bool) -> bool:
    if not _pr106_rules.get("unit_price_bare_prix"):
        return False
    normalized = _header_text(text)
    matcher = re.match if at_start else re.search
    if not matcher(_PR106_PRICE, normalized):
        return False
    context = _pr106_context(text)
    return bool(context["vat"] and context["amount"] and not context["qualified"])


def _pr106_header_matches(key: str, text: str, *, at_start: bool = False) -> bool:
    if key == "total_ht" and _pr106_gross_amount_column(text):
        _pr106_fire("total_ht_blocked_gross")
        return False
    if _pr106_production_header_matches(key, text, at_start=at_start):
        return True
    if key == "unit_price" and _pr106_bare_price_column(text, at_start):
        _pr106_fire(f"unit_price_bare_prix_at_start_{bool(at_start)}")
        return True
    return False


def _pr106_is_line_total_header(text: str, *, at_start: bool = False) -> bool:
    if _pr106_gross_amount_column(text):
        _pr106_fire("line_total_blocked_gross")
        return False
    return _pr106_production_is_line_total_header(text, at_start=at_start)


def _pr106_optional_headers(header_text: str) -> dict:
    """Production optional columns, plus the unqualified amount column only where total_ht lost it.

    Returning the production dictionary unchanged in every other case is what keeps a plain
    "Description | Quantity | Unit Price | Amount" header from claiming the same column twice, which would
    fail the distinct-position test and reject a header production accepts today.
    """
    if not _pr106_gross_amount_column(header_text):
        return OPTIONAL_HEADERS
    _pr106_fire("optional_headers_extended")
    extended = dict(OPTIONAL_HEADERS)
    extended["gross_amount"] = (
        r"\b(?:total\s+ttc|montant\s+ttc|gross\s+(?:amount|total))\b"
        r"|^(?:montant|amount)$"
    )
    return extended


def _pr106_partial_header(matches: dict, normalized: str, header_text: str) -> bool:
    if not all(matches.get(key) for key in ("description", "quantity", "unit_price")):
        return False
    if re.search(OPTIONAL_HEADERS["gross_amount"], normalized):
        return True
    if _pr106_gross_amount_column(header_text):
        _pr106_fire("partial_header_gross")
        return True
    return False


def _pr106_relabel_amount_column(starts: dict, header_text: str) -> dict:
    """Bind the printed amount column to the gross role when the header proves it is gross.

    The per-suffix and per-cell calls cannot see this by themselves: a lone "Montant" suffix carries no
    price and no VAT token, so it stays an amount candidate while the whole header line says otherwise, and
    both roles then claim the same x-position. Production rejects that header as ambiguous geometry, which
    is the right answer for two columns at one place; the correct reading is that there is only one column
    and it is gross. Nothing is moved to another column and nothing is invented: the position is kept and
    only its role changes, so the exported line HT stays null.
    """
    if not _pr106_gross_amount_column(header_text) or "total_ht" not in starts:
        return starts
    position = starts["total_ht"]
    if starts.get("gross_amount", position) != position:
        _pr106_fire("amount_column_kept")
        return starts
    relabelled = {key: value for key, value in starts.items() if key != "total_ht"}
    relabelled["gross_amount"] = position
    _pr106_fire("amount_column_relabelled")
    return relabelled


def _pr106_gross_stands_in(declared: set) -> bool:
    """A declared gross column may stand in for an undeclared HT column, never for a declared one."""
    return bool(
        _pr106_rules.get("declared_gross_counts")
        and "total_ht" not in declared
        and "gross_amount" in declared
    )


def _pr106_quantity_absent(declared: set) -> bool:
    return bool(
        _pr106_rules.get("quantity_column_absent_ok")
        and "quantity" not in declared
        and {"unit_price", "total_ht"} <= declared
    )


def _pr106_required_cells(declared: set) -> tuple:
    if _pr106_gross_stands_in(declared):
        _pr106_fire("required_cells_gross")
        return ("quantity", "unit_price", "gross_amount")
    if _pr106_quantity_absent(declared):
        _pr106_fire("required_cells_no_quantity")
        return ("unit_price", "total_ht")
    return _PR106_MONETARY


def _pr106_monetary_keys(declared: set) -> tuple:
    if _pr106_gross_stands_in(declared):
        _pr106_fire("monetary_keys_gross")
        return ("quantity", "unit_price", "gross_amount")
    if _pr106_quantity_absent(declared):
        _pr106_fire("monetary_keys_no_quantity")
        return ("unit_price", "total_ht")
    return _PR106_MONETARY
'''


CONFIGURATIONS = {
    "P": {"description": "production rules inside the derived module (parity gate)", "rules": {}},
    "C1": {"description": "a bare 'Prix'/'Price' column is the unit price when the same header also prints "
                          "a VAT column and an unqualified amount column",
           "rules": {"unit_price_bare_prix": True}},
    "C2": {"description": "C1 plus that unqualified amount column is bound to the gross role for the whole "
                          "header, so it is never a line HT candidate and the exported line HT stays null",
           "rules": {"unit_price_bare_prix": True, "ambiguous_amount_is_gross": True}},
    "C3": {"description": "C2 plus the completeness gates may count a declared gross column where the header "
                          "declared no HT column at all (the contextual row gate and the two-of-three "
                          "monetary count)",
           "rules": {"unit_price_bare_prix": True, "ambiguous_amount_is_gross": True,
                    "declared_gross_counts": True}},
    "C4": {"description": "a line keeps its row when the header declares no quantity column at all, on unit "
                          "price and amount evidence alone - the 'description + price + amount' structure "
                          "of Phase 3; quantity stays null because nothing prints it",
           "rules": {"quantity_column_absent_ok": True}},
}
CONFIG_ORDER = ("P", "C1", "C2", "C3", "C4")

CELL_FIELDS = ("description", "quantity", "unit_price", "total_ht")


def load_derived(source: str, rules: dict) -> types.ModuleType:
    derived = source
    for anchor, replacement in SUBSTITUTIONS:
        count = derived.count(anchor)
        if count != 1:
            raise SystemExit(
                f"anchor occurs {count} times, expected 1: {anchor[:72]!r}\n"
                "the production source changed; re-read it before measuring again")
        derived = derived.replace(anchor, replacement, 1)
    module = types.ModuleType(f"pr106_derived_{abs(hash(json.dumps(rules, sort_keys=True)))}")
    module.__file__ = str(INVOICE_EVIDENCE_PATH)
    sys.modules[module.__name__] = module
    exec(compile(derived + FOOTER, str(INVOICE_EVIDENCE_PATH), "exec"), module.__dict__)
    # The production predicates are the derived module's own untouched definitions; the wrappers call
    # those and then replace the module globals, which is what a production edit would do to every call
    # site at once.
    module._pr106_production_header_matches = module.__dict__["_header_matches"]
    module._pr106_production_is_line_total_header = module.__dict__["is_line_total_header"]
    for production_name, wrapper_name in VOCABULARY:
        setattr(module, production_name, module.__dict__[wrapper_name])
    module._pr106_rules = dict(rules)

    import invoice_evidence as production

    # Class identity stays with production: ocr_recovery and app.main catch and build these objects.
    for name in ("UnreadableOCR", "DocumentEvidence", "TableRow", "TextLine", "Token", "Number"):
        setattr(module, name, getattr(production, name))
    return module


class Stage5Freeze(H.Freeze):
    """Frozen bytes replayed positionally; consensus always production; stage 5 is the candidate."""

    def consensus_call(self, readings, *, confidence):
        if len(self.consensus) >= len(self.recording["consensus"]):
            self.consensus_calls_beyond_recorded += 1
        result = self._original_consensus(readings, confidence=confidence)
        self.consensus.append(result)
        return result


def measure(runs: dict, fires: dict, documents: dict, scored: dict) -> dict:
    """Baseline-relative scores for one evidence policy, computed identically for frozen and live runs."""
    per_document = {}
    for name in documents:
        per_document[name] = {}
        baseline_rows = len(runs["P"][name]["payload"].get("line_items") or [])
        baseline_evaluation = scored["P"][name]
        for config in CONFIG_ORDER:
            payload = runs[config][name]["payload"]
            evaluation = scored[config][name]
            rows = payload.get("line_items") or []
            per_document[name][config] = {
                "exported_rows": len(rows),
                "rows_recovered": len(rows) - baseline_rows,
                "correctly_matched_rows": evaluation["rows"]["correctly_matched"],
                "cells": {field: {verdict: evaluation["fields"]["by_field"][field][verdict]
                                  for verdict in ("correct", "incorrect", "missing", "unverified")}
                          for field in CELL_FIELDS},
                "fields_correct_delta": evaluation["fields"]["correct"] - baseline_evaluation["fields"]["correct"],
                "incorrect_total": incorrect_total(evaluation),
                "incorrect_total_delta": incorrect_total(evaluation) - incorrect_total(baseline_evaluation),
                "unsupported_nonnull": unbacked(payload, documents[name]["reference"]),
                "payload_diff_from_baseline": payload_diff(runs["P"][name]["payload"], payload),
                "rules_fired": fires[config][name],
                "source_kind": runs[config][name]["source_kind"],
                "consensus_calls": runs[config][name]["consensus_calls"],
                "payload_sha256": runs[config][name]["payload_sha256"],
            }

    aggregate = {}
    for config in CONFIG_ORDER:
        rows_baseline = sum(len(runs["P"][name]["payload"].get("line_items") or []) for name in documents)
        rows_now = sum(len(runs[config][name]["payload"].get("line_items") or []) for name in documents)
        aggregate[config] = {
            "documents": len(documents),
            "exported_rows": rows_now,
            "exported_rows_delta_vs_baseline": rows_now - rows_baseline,
            "cells_correct": sum(scored[config][name]["fields"]["correct"] for name in documents),
            "cells_incorrect": sum(scored[config][name]["fields"]["incorrect"] for name in documents),
            "cells_missing": sum(scored[config][name]["fields"]["missing"] for name in documents),
            "line_item_precision": (
                sum(scored[config][name]["rows"]["correctly_matched"] for name in documents) / rows_now
                if rows_now else None),
            "line_item_recall": (
                sum(scored[config][name]["rows"]["correctly_matched"] for name in documents)
                / sum(len(documents[name]["reference"]["line_items"] or []) for name in documents)),
            "incorrect_total": sum(incorrect_total(scored[config][name]) for name in documents),
            "unsupported_nonnull_values": sum(len(values)
                                             for name in documents
                                             for values in per_document[name][config]["unsupported_nonnull"].values()),
            "rules_fired": {
                key: sum(counts.get(key, 0) for counts in fires[config].values())
                for key in sorted({key for counts in fires[config].values() for key in counts})
            },
            "documents_changed": sorted(name for name in documents
                                        if per_document[name][config]["payload_sha256"]
                                        != per_document[name]["P"]["payload_sha256"]),
            "documents_regressed": sorted(
                name for name in documents
                if scored[config][name]["fields"]["correct"] < scored["P"][name]["fields"]["correct"]
                or incorrect_total(scored[config][name]) > incorrect_total(scored["P"][name])),
            "seconds": runs[config]["__seconds__"],
            "acquisition_coupling": runs[config].get("__coupling__"),
            "live_calls": runs[config].get("__live_calls__"),
        }

    return {
        "per_document": per_document,
        "aggregate": aggregate,
        "false_positive_watch": {
            "definition": "an exported row that no reference row supports on two or more exported fields",
            "per_configuration": {
                config: {
                    name: [
                        {field: (row or {}).get(field) for field in CELL_FIELDS}
                        for row in (runs[config][name]["payload"].get("line_items") or [])
                        if not row_supported(row, documents[name]["reference"])
                    ] for name in documents
                    if any(not row_supported(row, documents[name]["reference"])
                           for row in (runs[config][name]["payload"].get("line_items") or []))
                } for config in CONFIG_ORDER
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="", help="comma-separated subset of the eleven documents")
    parser.add_argument("--output-name", default="pr106-phase6-candidates.json")
    parser.add_argument("--live", action="store_true",
                        help="also run every configuration through real Tesseract: a stage-5 rule that "
                             "changes which crops the pipeline re-reads cannot be scored on frozen bytes alone")
    args = parser.parse_args()

    import invoice_evidence as production
    import ocr_recovery

    source = INVOICE_EVIDENCE_PATH.read_text(encoding="utf-8")
    cache = H.read_json(PR105B_CACHE)
    pr105b = H.read_json(PR105B_CACHE.parent / "pr105b-baseline-parity.json")
    expected = dict(pr105b["parity"]["payload_sha256"])

    documents = P.documents()
    if args.documents:
        wanted = [name.strip() for name in args.documents.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in documents]
        if unknown:
            raise SystemExit(f"unknown documents: {unknown}")
        documents = {name: documents[name] for name in wanted}

    # --- reference pass: production stage 5, recording the acquisition path every candidate follows
    reference_freeze = H.Freeze("reference", dict(cache["entries"]))
    reference: dict[str, dict] = {}
    recordings: dict[str, dict] = {}
    with (patch.object(ocr_recovery, "from_tsv", production.from_tsv),
          patch.object(ocr_recovery, "from_tokens", production.from_tokens)):
        for name, entry in documents.items():
            reference[name] = H.run_document(name, entry, reference_freeze)
            recordings[name] = {"keys": list(reference_freeze.keys),
                                "consensus": list(reference_freeze.consensus)}

    diverging = [name for name in documents
                 if reference[name]["payload_sha256"] != expected.get(name)]
    if diverging:
        raise SystemExit(
            "the reference replay diverges from the published PR105B baseline: "
            + json.dumps(diverging, indent=2))

    # --- candidate passes
    runs: dict[str, dict] = {}
    fires: dict[str, dict] = {}
    for config in CONFIG_ORDER:
        module = load_derived(source, CONFIGURATIONS[config]["rules"])
        freeze = Stage5Freeze("pinned", dict(cache["entries"]), recordings)
        runs[config] = {}
        fires[config] = {}
        started = time.perf_counter()
        with (patch.object(ocr_recovery, "from_tsv", module.from_tsv),
              patch.object(ocr_recovery, "from_tokens", module.from_tokens)):
            for name, entry in documents.items():
                module._pr106_fires.clear()
                runs[config][name] = H.run_document(name, entry, freeze)
                fires[config][name] = dict(module._pr106_fires)
        runs[config]["__seconds__"] = round(time.perf_counter() - started, 1)
        runs[config]["__coupling__"] = {
            "off_sequence_calls": {name: runs[config][name]["off_sequence_calls"]
                                   for name in documents if runs[config][name]["off_sequence_calls"]},
            "suppressed_calls": {name: runs[config][name]["suppressed_calls"]
                                 for name in documents if runs[config][name]["suppressed_calls"]},
            "consensus_calls_beyond_recorded": {
                name: runs[config][name]["consensus_calls_beyond_recorded"]
                for name in documents if runs[config][name]["consensus_calls_beyond_recorded"]},
        }
        print(f"{config}: {runs[config]['__seconds__']}s, "
              f"suppressed {sum(len(v) for v in runs[config]['__coupling__']['suppressed_calls'].values())}, "
              f"off-sequence {sum(len(v) for v in runs[config]['__coupling__']['off_sequence_calls'].values())}")

    parity_p = [name for name in documents
                if runs["P"][name]["payload_sha256"] != reference[name]["payload_sha256"]]
    if parity_p:
        raise SystemExit(
            "configuration P (no candidate rule) does not reproduce production on the frozen bytes; "
            "the derivation itself changed behaviour: " + json.dumps(parity_p, indent=2))

    scored = {config: {name: metrics.evaluate(runs[config][name]["payload"], documents[name]["reference"])
                       for name in documents} for config in CONFIG_ORDER}

    frozen = measure(runs, fires, documents, scored)

    live_result = None
    if args.live:
        entries = dict(cache["entries"])
        live_runs: dict[str, dict] = {}
        live_fires: dict[str, dict] = {}
        for config in CONFIG_ORDER:
            module = load_derived(source, CONFIGURATIONS[config]["rules"])
            freeze = H.Freeze("capture", entries)
            live_runs[config] = {}
            live_fires[config] = {}
            started = time.perf_counter()
            with (patch.object(ocr_recovery, "from_tsv", module.from_tsv),
                  patch.object(ocr_recovery, "from_tokens", module.from_tokens)):
                for name, entry in documents.items():
                    module._pr106_fires.clear()
                    live_runs[config][name] = H.run_document(name, entry, freeze)
                    live_fires[config][name] = dict(module._pr106_fires)
            live_runs[config]["__seconds__"] = round(time.perf_counter() - started, 1)
            live_runs[config]["__live_calls__"] = freeze.live_calls
            print(f"live {config}: {live_runs[config]['__seconds__']}s, "
                  f"{freeze.live_calls} real Tesseract calls")
        live_scored = {config: {name: metrics.evaluate(live_runs[config][name]["payload"],
                                                      documents[name]["reference"])
                                for name in documents} for config in CONFIG_ORDER}
        live_result = measure(live_runs, live_fires, documents, live_scored)
        P.write_json(P.guard(P.ARTIFACT_DIR, P.ARTIFACT_DIR / "pr106-live-ocr-cache.json"),
                     {"entries": entries,
                      "note": "real Tesseract bytes for the candidate configurations of this run"})

    artifact = {
        "phase": "PR106 Phase 5-6 - stage-5 structural candidates on frozen evidence",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instrumentation": {
            "production_source": str(INVOICE_EVIDENCE_PATH.relative_to(ROOT)),
            "production_sha256": H.sha256_file(INVOICE_EVIDENCE_PATH),
            "anchored_substitutions": [anchor.strip()[:72] for anchor, _ in SUBSTITUTIONS],
            "rebound_module_globals": [[name, wrapper] for name, wrapper in VOCABULARY],
            "class_identity_kept_with_production": ["UnreadableOCR", "DocumentEvidence", "TableRow",
                                                    "TextLine", "Token", "Number"],
            "configuration_P_parity": "IDENTICAL to production on all documents (verified before writing)",
        },
        "evidence": {
            "cache_file": str(PR105B_CACHE.relative_to(ROOT)),
            "unique_cached_calls": len(cache["entries"]),
            "reference_replay_matches_pr105b_baseline": True,
            "documents": len(documents),
        },
        "configurations": {config: CONFIGURATIONS[config]["description"] for config in CONFIG_ORDER},
        "rules": {config: CONFIGURATIONS[config]["rules"] for config in CONFIG_ORDER},
        "baseline": {
            name: {"expected_rows": len(documents[name]["reference"]["line_items"] or []),
                   "exported_rows": len(reference[name]["payload"].get("line_items") or []),
                   "payload_sha256": reference[name]["payload_sha256"],
                   "cells_correct": scored["P"][name]["fields"]["correct"],
                   "incorrect_total": incorrect_total(scored["P"][name])}
            for name in documents
        },
        "frozen": frozen,
        "live": live_result,
    }
    target = P.guard(P.ARTIFACT_DIR, P.ARTIFACT_DIR / args.output_name)
    P.write_json(target, artifact)
    for label, result in (("frozen", frozen), ("live", live_result)):
        if result is None:
            continue
        for config in CONFIG_ORDER:
            summary = result["aggregate"][config]
            print(f"{label} {config}: rows {summary['exported_rows']} "
                  f"({summary['exported_rows_delta_vs_baseline']:+d}), "
                  f"cells correct {summary['cells_correct']}, incorrect {summary['cells_incorrect']}, "
                  f"missing {summary['cells_missing']}, incorrect_total {summary['incorrect_total']}, "
                  f"unsupported {summary['unsupported_nonnull_values']}, "
                  f"changed {len(summary['documents_changed'])}, "
                  f"regressed {len(summary['documents_regressed'])}, fires {summary['rules_fired']}")
    print("wrote", target)
    return 0


def row_supported(row, reference) -> bool:
    """A row is supported when one reference row agrees with it on at least two exported fields.

    The comparison is the evaluator's own, so a reference quantity of 3 and an exported 3.0 agree;
    a string comparison here would call a correct numeric row unsupported.
    """
    for gold in reference.get("line_items") or []:
        agreements = sum(
            1 for field in CELL_FIELDS
            if (row or {}).get(field) is not None and metrics._text((row or {}).get(field)) != ""
            and metrics._equal(field, gold.get(field), (row or {}).get(field))
            and metrics._text(gold.get(field)) != ""
        )
        if agreements >= 2:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
