"""PR105B Phase 2+3+4+5: the A/B/C/D factorial on one frozen evidence policy, with the error matrix.

The production pipeline runs once in reference mode to record the acquisition path (the ordered OCR calls
and every consensus result); each configuration then replays those bytes positionally for each document and
runs the rule under test on every consensus call. Configuration A — the production rule inside the replay —
must reproduce the reference payloads byte-for-byte and must not diverge from the recorded sequence, or the
run aborts: that gate is what makes B, C and D comparable.

Counts, verdicts and digests only go to the main artifact; the description texts and the per-candidate
confidence logs sit in a separate git-ignored detail file because they carry verbatim invoice text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import evaluation_metrics as metrics  # noqa: E402
import pr105b_harness as H  # noqa: E402
from pr105b_capture import guard  # noqa: E402

CONFIG_ORDER = ("A", "B", "C", "D")
VERDICTS = ("correct", "incorrect", "missing", "unverified")
CELL_FIELDS = ("description", "quantity", "unit_price", "total_ht")


def incorrect_total(evaluation: dict) -> int:
    """Every exported non-null value that contradicts the reference: row cells, associations, headers."""
    return (evaluation["fields"]["incorrect_monetary"] + evaluation["fields"]["incorrect_association"]
            + evaluation["header_summary"]["incorrect"])


def cells(evaluation: dict) -> dict:
    return {field: {verdict: evaluation["fields"]["by_field"][field][verdict] for verdict in VERDICTS}
            for field in CELL_FIELDS}


def unbacked(payload: dict, reference: dict) -> dict:
    """Exported non-null values that no field of this document's reference carries.

    `evaluation_metrics` only grades cells inside matched rows, so a description that is close to the truth
    but not equal to it - an OCR table border glued to the first word, for instance - disappears from the
    counts while still reaching the user. Acceptance criterion 2 is about any additional incorrect non-null
    field, so it is measured here without the association step.
    """
    out: dict[str, list] = {}
    for field in ("supplier_name", "invoice_date", "total_ht", "total_ttc"):
        value = (payload or {}).get(field)
        if not metrics._text(value):
            continue
        gold = reference.get(field)
        if not (gold is not None and metrics._text(gold) == metrics._text(value)):
            out[field] = [str(value)]
    gold_rows = reference.get("line_items") or []
    for field in CELL_FIELDS:
        for row in (payload or {}).get("line_items") or []:
            value = (row or {}).get(field)
            if value is None or not metrics._text(value):
                continue
            if field == "description":
                backed = any(metrics._text(gold.get("description")) == metrics._text(value)
                             for gold in gold_rows)
            else:
                number = metrics._number(value)
                backed = number is not None and any(metrics._number(gold.get(field)) == number
                                                    for gold in gold_rows)
            if not backed:
                out.setdefault(f"line.{field}", []).append(str(value))
    return out


def payload_diff(baseline: dict, current: dict) -> dict:
    """Export-contract fields whose values moved relative to the baseline payload.

    A description rule must not touch money or headers, and the evaluator alone cannot prove that: it only
    grades rows it can associate. This compares every field of the two payloads directly.
    """
    def flatten(payload: dict) -> dict:
        fields: dict[str, list] = {}
        for key, value in (payload or {}).items():
            if key == "line_items":
                for row in value or []:
                    for field, cell in (row or {}).items():
                        fields.setdefault(f"line_items.{field}", []).append(str(cell))
            else:
                fields.setdefault(key, []).append(str(value))
        return fields

    a, b = flatten(baseline), flatten(current)
    return {key: {"baseline": sorted(a.get(key, [])), "candidate": sorted(b.get(key, []))}
            for key in sorted(set(a) | set(b)) if sorted(a.get(key, [])) != sorted(b.get(key, []))}


def mechanism(gate_log: list[dict]) -> dict:
    """Field-gate headroom: candidate readings whose description column is cleaner than their whole line."""
    return {
        "candidate_readings": len(gate_log),
        "gate_line_passing": sum(1 for entry in gate_log if entry["gate_line"]),
        "gate_field_passing": sum(1 for entry in gate_log if entry["gate_field"]),
        "newly_passing": sum(1 for entry in gate_log if entry["gate_field"] and not entry["gate_line"]),
        "column_evidence_used": sum(1 for entry in gate_log if entry["used_column"]),
        "whole_line_fallback": sum(1 for entry in gate_log if not entry["used_column"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="pr105b-factorial.json")
    parser.add_argument("--detail-name", default="pr105b-factorial-detail.json")
    args = parser.parse_args()

    target = guard(H.ARTIFACT_DIR, H.ARTIFACT_DIR / args.output_name)
    detail_target = guard(H.ARTIFACT_DIR, H.ARTIFACT_DIR / args.detail_name)
    cache = H.read_json(H.ARTIFACT_DIR / H.CACHE_NAME)
    documents = H.corpus()
    expected = cache["documents"]
    source = H.PRODUCTION_SOURCE_PATH.read_text(encoding="utf-8")
    variant = H.load_variant(source)

    # Reference pass: production, unpinned, replaying the frozen OCR bytes. It records the acquisition
    # path every configuration must follow and reproduces the Phase 1 payloads.
    reference_freeze = H.Freeze("reference", dict(cache["entries"]))
    reference, recordings = {}, {}
    for name, entry in documents.items():
        reference[name] = H.run_document(name, entry, reference_freeze)
        recordings[name] = {"keys": list(reference_freeze.keys),
                            "consensus": list(reference_freeze.consensus)}
    diverging = [name for name in documents
                 if reference[name]["payload_sha256"] != expected[name]["payload_sha256"]]
    if diverging:
        raise SystemExit(f"the reference replay diverges from the Phase 1 baseline: {diverging}")

    # Factorial pass: frozen bytes per document, four description rules, every consensus call under the rule.
    pinned = H.Freeze("pinned", dict(cache["entries"]), recordings)
    runs, divergence = {}, {}
    for config in CONFIG_ORDER:
        started = time.perf_counter()
        runs[config] = {name: H.run_document(name, entry, pinned, variant, config)
                        for name, entry in documents.items()}
        divergence[config] = {
            "off_sequence_calls": {name: runs[config][name]["off_sequence_calls"]
                                   for name in documents if runs[config][name]["off_sequence_calls"]},
            "suppressed_calls": {name: runs[config][name]["suppressed_calls"]
                                 for name in documents if runs[config][name]["suppressed_calls"]},
            "consensus_calls_beyond_recorded": {
                name: runs[config][name]["consensus_calls_beyond_recorded"]
                for name in documents if runs[config][name]["consensus_calls_beyond_recorded"]},
        }
        if config == "A":
            # The baseline rule through the frozen replay must be production, exactly.
            moved = [name for name in documents
                     if runs["A"][name]["payload_sha256"] != expected[name]["payload_sha256"]]
            frozen = divergence[config]["off_sequence_calls"] or divergence[config]["suppressed_calls"] \
                     or divergence[config]["consensus_calls_beyond_recorded"]
            if moved or frozen:
                raise SystemExit(
                    "the frozen replay is not faithful: configuration A left the production result "
                    f"(payloads moved: {moved}; divergence: {json.dumps(frozen, ensure_ascii=False)})")
        print(f"config {config}: {sum(len(v) for v in divergence[config]['off_sequence_calls'].values())} "
              f"re-ordered, {sum(len(v) for v in divergence[config]['suppressed_calls'].values())} "
              f"suppressed, {sum(divergence[config]['consensus_calls_beyond_recorded'].values())} extra "
              f"consensus calls, {pinned.live_calls} live calls, {time.perf_counter() - started:.1f}s")
    if any(divergence[config][kind] for config in CONFIG_ORDER for kind in divergence[config]):
        print("acquisition divergence (bytes beyond this document's frozen capture are never granted):",
              json.dumps({config: divergence[config] for config in CONFIG_ORDER[1:]
                          if any(divergence[config].values())}, ensure_ascii=False, indent=2))

    scored = {config: {name: metrics.evaluate(runs[config][name]["payload"], documents[name]["reference"])
                       for name in documents} for config in CONFIG_ORDER}
    aggregated = {config: metrics.aggregate(list(scored[config].values())) for config in CONFIG_ORDER}
    unbacked_values = {config: {name: unbacked(runs[config][name]["payload"], documents[name]["reference"])
                                 for name in documents} for config in CONFIG_ORDER}
    diffs = {config: {name: payload_diff(runs["A"][name]["payload"], runs[config][name]["payload"])
                      for name in documents} for config in CONFIG_ORDER}

    documents_out, regressions, improvements = {}, [], []
    for name, entry in documents.items():
        per_config = {}
        for config in CONFIG_ORDER:
            evaluation, baseline = scored[config][name], scored["A"][name]
            per_config[config] = {
                "technical_error": runs[config][name]["technical_error"],
                "exported_rows": runs[config][name]["exported_rows"],
                "nonempty_descriptions": runs[config][name]["nonempty_descriptions"],
                "payload_sha256": runs[config][name]["payload_sha256"],
                "suppressed_calls": list(runs[config][name]["suppressed_calls"]),
                "cells": cells(evaluation),
                "rows": evaluation["rows"],
                "header": evaluation["header"],
                "incorrect_total": incorrect_total(evaluation),
                "unbacked": {field: len(values) for field, values in unbacked_values[config][name].items()},
                "unbacked_total": sum(len(v) for v in unbacked_values[config][name].values()),
                "unbacked_delta_vs_baseline": sum(len(v) for v in unbacked_values[config][name].values())
                                              - sum(len(v) for v in unbacked_values["A"][name].values()),
                "delta": {field: {verdict: evaluation["fields"]["by_field"][field][verdict]
                                          - baseline["fields"]["by_field"][field][verdict]
                                  for verdict in VERDICTS} for field in CELL_FIELDS},
                "rows_correct_delta": evaluation["rows"]["correctly_matched"]
                                      - baseline["rows"]["correctly_matched"],
                "incorrect_delta": incorrect_total(evaluation) - incorrect_total(baseline),
            }
        documents_out[name] = {
            "kind": entry["kind"],
            "reference_source": entry["reference_source"],
            "expected_rows": len(entry["reference"]["line_items"] or []),
            "acquisition": {"ocr_calls": len(recordings[name]["keys"]),
                            "consensus_calls": len(recordings[name]["consensus"])},
            "mechanism_field_gate": mechanism(runs["B"][name]["gate_log"]),
            "per_configuration": per_config,
        }
        for config in CONFIG_ORDER:
            result = per_config[config]
            if config == "A":
                continue
            header_regressed = [field for field, verdict in result["header"].items()
                               if verdict == "incorrect" and scored["A"][name]["header"][field] != "incorrect"]
            cell_regressed = [field for field, delta in result["delta"].items()
                             if delta["correct"] < 0 or delta["incorrect"] > 0]
            if cell_regressed or header_regressed or result["rows_correct_delta"] < 0 \
               or result["incorrect_delta"] > 0 or result["unbacked_delta_vs_baseline"] > 0:
                regressions.append({
                    "document": name, "configuration": config, "row_fields": cell_regressed,
                    "header_fields": header_regressed,
                    "rows_correct_delta": result["rows_correct_delta"],
                    "incorrect_delta": result["incorrect_delta"],
                    "unbacked_delta": result["unbacked_delta_vs_baseline"],
                })
            if result["rows_correct_delta"] > 0 or any(
                delta["correct"] > 0 for delta in result["delta"].values()):
                improvements.append({
                    "document": name, "configuration": config,
                    "rows_correct_delta": result["rows_correct_delta"],
                    "incorrect_delta": result["incorrect_delta"],
                    "unbacked_delta": result["unbacked_delta_vs_baseline"],
                    "description_correct_delta": result["delta"]["description"]["correct"],
                    "monetary_correct_delta": sum(result["delta"][field]["correct"]
                                                  for field in CELL_FIELDS[1:]),
                })

    detail = {
        name: {
            "exported_descriptions": {
                config: [row.get("description")
                         for row in (runs[config][name]["payload"] or {}).get("line_items") or []]
                for config in CONFIG_ORDER
            },
            "field_gate_log": runs["B"][name]["gate_log"],
            "unbacked_values": {config: unbacked_values[config][name] for config in CONFIG_ORDER},
            "payload_diff_vs_baseline": {config: diffs[config][name] for config in CONFIG_ORDER},
            "pairs": {config: scored[config][name]["pairs"] for config in CONFIG_ORDER},
        }
        for name in documents
    }
    detail_target.write_text(json.dumps({
        "warning": "verbatim invoice text from real scans; git-ignored, never commit or quote externally",
        "documents": detail,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    artifact = {
        "phase": "PR105B Phase 2-5 - A/B/C/D factorial on one frozen evidence policy, per-field matrix",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definitions": {
            "A": "baseline whole-line description gate + unanimity requirement (production rule)",
            "B": "field-level (description-column) confidence gate + unanimity requirement",
            "C": "baseline whole-line gate + strict two-family majority (unanimity dropped)",
            "D": "field-level gate + strict two-family majority",
            "incorrect_total": "row cells rated incorrect + incorrect monetary associations "
                               "+ incorrect header fields",
        },
        "evidence_isolation": {
            "cache_file": str((H.ARTIFACT_DIR / H.CACHE_NAME).relative_to(ROOT)),
            "unique_cached_ocr_calls": len(cache["entries"]),
            "live_tesseract_calls_in_this_run": pinned.live_calls + reference_freeze.live_calls,
            "ocr_call_sequences_identical_across_configurations": all(
                runs[config][name]["ocr_call_keys"] == recordings[name]["keys"]
                for config in CONFIG_ORDER for name in documents),
            "consensus_pinning_abandoned_reason": "_agreed returns the original reading objects "
                                                  "(src/ocr_consensus.py:65), so a copied intermediate "
                                                  "decision loses the aliasing production depends on; "
                                                  "measured, it moved two payloads under the baseline rule",
            "acquisition_divergence": divergence,
            "configuration_A_parity": {
                "documents": len(documents),
                "payloads_equal_to_production_baseline": sum(
                    1 for name in documents
                    if runs["A"][name]["payload_sha256"] == expected[name]["payload_sha256"]),
                "run_aborts_on_any_mismatch": True,
            },
            "configuration_A_parity": {
                "documents": len(documents),
                "payloads_equal_to_production_baseline": sum(
                    1 for name in documents
                    if runs["A"][name]["payload_sha256"] == expected[name]["payload_sha256"]),
                "run_aborts_on_any_mismatch": True,
            },
            "payload_sha256_by_configuration": {
                name: {config: runs[config][name]["payload_sha256"] for config in CONFIG_ORDER}
                for name in documents},
        },
        "corpus_totals": {
            config: {
                "exported_rows": sum(run["exported_rows"] for run in runs[config].values()),
                "nonempty_descriptions": sum(
                    run["nonempty_descriptions"] for run in runs[config].values()),
                "fields": {key: aggregated[config]["fields"][key] for key in
                           ("correct", "incorrect", "missing", "unverified", "incorrect_monetary",
                            "incorrect_association")},
                "rows": {key: aggregated[config]["rows"][key] for key in
                         ("expected", "exported", "correctly_matched", "matched_total", "missing",
                          "duplicate", "misassociated", "unsupported")},
                "header": aggregated[config]["header"],
                "incorrect_total": sum(incorrect_total(scored[config][name]) for name in documents),
                "documents_with_changed_payload": sum(
                    1 for name in documents
                    if runs[config][name]["payload_sha256"] != expected[name]["payload_sha256"]),
            }
            for config in CONFIG_ORDER
        },
        "deltas_vs_baseline": {
            config: {
                "correct_cells": sum(scored[config][name]["fields"]["correct"] for name in documents)
                                 - sum(scored["A"][name]["fields"]["correct"] for name in documents),
                "incorrect_cells": sum(
                    sum(scored[config][name]["fields"]["by_field"][field]["incorrect"]
                        for field in CELL_FIELDS) for name in documents) - sum(
                    sum(scored["A"][name]["fields"]["by_field"][field]["incorrect"]
                        for field in CELL_FIELDS) for name in documents),
                "incorrect_total": sum(incorrect_total(scored[config][name]) for name in documents)
                                   - sum(incorrect_total(scored["A"][name]) for name in documents),
                "rows_correctly_matched": aggregated[config]["rows"]["correctly_matched"]
                                          - aggregated["A"]["rows"]["correctly_matched"],
                "description_correct": aggregated[config]["fields"]["correct"]
                                       - aggregated["A"]["fields"]["correct"],
                "misassociated_delta": aggregated[config]["rows"]["misassociated"]
                                       - aggregated["A"]["rows"]["misassociated"],
                "unsupported_delta": aggregated[config]["rows"]["unsupported"]
                                     - aggregated["A"]["rows"]["unsupported"],
                "header_incorrect_delta": aggregated[config]["header"]["incorrect"]
                                          - aggregated["A"]["header"]["incorrect"],
                "documents_with_changed_payload": sum(
                    1 for name in documents
                    if runs[config][name]["payload_sha256"] != expected[name]["payload_sha256"]),
            }
            for config in CONFIG_ORDER[1:]
        },
        "field_gate_mechanism": {
            "candidate_readings": sum(d["mechanism_field_gate"]["candidate_readings"]
                                      for d in documents_out.values()),
            "newly_passing": sum(d["mechanism_field_gate"]["newly_passing"]
                                 for d in documents_out.values()),
            "per_document": {name: documents_out[name]["mechanism_field_gate"] for name in documents},
        },
        "unbacked_audit": {
            "note": "exported non-null values with no equal value in this document's reference, counted "
                    "without row matching; evaluation_metrics only grades associated rows",
            **{config: {
                "total": sum(sum(len(v) for v in unbacked_values[config][name].values())
                             for name in documents),
                "delta_vs_baseline": sum(sum(len(v) for v in unbacked_values[config][name].values())
                                         for name in documents)
                                      - sum(sum(len(v) for v in unbacked_values["A"][name].values())
                                            for name in documents),
                "by_field": {field: sum(len(unbacked_values[config][name].get(field, []))
                                        for name in documents)
                             for field in sorted({f for name in documents
                                                  for f in unbacked_values[config][name]})},
            } for config in CONFIG_ORDER},
        },
        "monetary_safeguards": {
            config: {
                "documents_with_any_field_moved": sorted(n for n, moved in diffs[config].items() if moved),
                "fields_moved": sorted({field for moved in diffs[config].values() for field in moved}),
                "row_monetary_fields_moved": sorted(
                    field for moved in diffs[config].values() for field in moved
                    if field.startswith("line_items.") and field.rsplit(".", 1)[1]
                    not in {"description", "line_index"}),
                "header_or_total_fields_moved": sorted(
                    field for moved in diffs[config].values() for field in moved
                    if not field.startswith("line_items.")),
            }
            for config in CONFIG_ORDER
        },
        "regressions": regressions,
        "improvements": improvements,
        "documents": documents_out,
        "detail_file": str(detail_target.relative_to(ROOT)),
    }
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for config in CONFIG_ORDER:
        totals = artifact["corpus_totals"][config]
        line = (f" {config}: rows {totals['rows']['correctly_matched']:2}/{totals['rows']['expected']:2} "
                f"descriptions {totals['nonempty_descriptions']:2} non-null | cells "
                f"{totals['fields']['correct']:3} correct {totals['fields']['incorrect']:2} wrong "
                f"{totals['fields']['missing']:3} missing | {totals['incorrect_total']:2} incorrect total | "
                f"{totals['documents_with_changed_payload']:2} payloads changed")
        if config != "A":
            line += f" | delta {json.dumps(artifact['deltas_vs_baseline'][config])}"
        print(line)
    for config in CONFIG_ORDER[1:]:
        audit = artifact["unbacked_audit"][config]
        safe = artifact["monetary_safeguards"][config]
        print(f" {config}: unbacked non-null values {audit['total']} "
              f"({audit['delta_vs_baseline']:+} vs baseline) by field {json.dumps(audit['by_field'])}; "
              f"fields moved {safe['fields_moved']}; monetary {safe['row_monetary_fields_moved']}; "
              f"headers {safe['header_or_total_fields_moved']}")
    print("field gate newly passing candidate readings:",
          artifact["field_gate_mechanism"]["newly_passing"])
    print("regressions:", len(regressions), "| improvements:", len(improvements))
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
