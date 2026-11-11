"""PR84 offline row-building trace and variant comparison for the six frozen layouts.

Replays the *unmodified* production row builder (`from_tsv` / `from_tokens` /
`build_consensus` / `reconcile`) on the token-level inputs recorded by
`scripts/capture_row_building_baseline.py`, so every rejection is demonstrated by the
production predicate that fired, with the real geometry and confidence of the recorded
tokens. Nothing here modifies `src/`, calls a model or runs Tesseract.

The experimental variants patch exactly one production lookup at a time, inside this
research process only, and every variant is scored on all six layouts against the frozen
references. The "as_built" configuration must reproduce the recorded production export
before any variant number is reported.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import invoice_evidence
from capture_layout_row_baseline import score, serialize_row_candidate
from capture_row_building_baseline import replay_from_capture
from evaluate_grounding import equal
from invoice_grounding import reconcile
from ocr_config import OCR_CONFIG
from ocr_consensus import build_consensus

CAPTURE = ROOT / "results" / "product-review84" / "pr84-row-building-capture.json"
OUTPUT = ROOT / "results" / "product-review84" / "pr84-row-building-trace-v2.json"
ROW_KEYS = ("description", "quantity", "unit_price", "total_ht")

# One-level vocabulary additions, never a document-specific rule.
FRENCH_RATE_LABEL = r"|\bprix\b|\btarif\b"
HOURLY_RATE_LABEL = r"|(?:(?:[€$£]\s*)?/\s*(?:heure|heuree|hour|hr)\b|\bper\s+(?:heure|hour)\b)"
HOURLY_QUANTITY_LABEL = r"|\bheures?\b|\bhrs?\b"

original_rate_price_header = invoice_evidence._contextual_rate_price_header
original_rate_price_index = invoice_evidence._rate_price_token_index
BASE_HEADERS = dict(invoice_evidence.HEADERS)


# --------------------------------------------------------------------------- replay


def rebuild_read(read: dict):
    """Rebuild one recorded read through the production parser, from recorded input only.

    Recorded `tsv` reads always come from `from_tsv(data)` call sites that pass no locale
    (`src/ocr_recovery.py:74,167,205,301,384`), so an empty kwargs dict reproduces them;
    `capture_replay` in the artifact proves the recorded input yields the recorded rows.
    """
    return replay_from_capture({
        "kind": read["kind"],
        "tsv": read.get("tsv"),
        "tokens": read.get("tokens"),
        "locale": read.get("locale"),
        "literal_columns": read.get("literal_columns"),
        "kwargs": {},
    })


def all_reads(document: dict) -> list[dict]:
    """Every recorded read, in call order. `read_key` repeats across consensus calls."""
    return [
        dict(read, read_id=f"call{call['call_index']}/{read['read_key']}")
        for call in document["consensus_calls"] for read in call["reads"]
    ]


def candidate_signature(rows) -> list[str]:
    return [
        "|".join((
            row.description,
            *(str(row.cells[key][0].value) if row.cells.get(key) else "-" for key in ROW_KEYS[1:]),
        ))
        for row in rows
    ]


def recorded_candidate_signature(candidates: list[dict]) -> list[str]:
    return [
        "|".join((
            candidate["description"],
            *(candidate["cells"][key][0]["value"] if candidate["cells"].get(key) else "-"
              for key in ROW_KEYS[1:]),
        ))
        for candidate in candidates
    ]


def gate_read_replay(document: dict) -> dict:
    """Gate 1: the recorded input must reproduce the recorded row candidates exactly."""
    failures, checked = [], 0
    for read in all_reads(document):
        rebuilt = rebuild_read(read)
        recorded = recorded_candidate_signature(read["row_candidates"])
        checked += 1
        if candidate_signature(rebuilt.rows) != recorded:
            failures.append({
                "read_key": read["read_key"],
                "rebuilt": candidate_signature(rebuilt.rows),
                "recorded": recorded,
            })
    return {"reads_checked": checked, "failures": failures}


def consensus_read_group(document: dict) -> tuple[dict, dict]:
    """The last consensus call of a document: the multi-family pass production ended with."""
    call = document["consensus_calls"][-1]
    readings: dict[str, list] = {}
    for read in call["reads"]:
        readings.setdefault(read["family"], []).append(rebuild_read(read))
    return call, readings


def payload_row_signature(payload: dict) -> list[str]:
    return sorted(
        "|".join((
            str(row.get("description") or ""),
            *(("" if row.get(key) is None else str(Decimal(str(row[key])))) for key in ROW_KEYS[1:]),
        ))
        for row in (payload.get("line_items") or [])
    )


def gate_consensus_replay(document: dict, floor: int) -> dict:
    """Gate 2: unpatched production consensus + grounding on recorded reads must equal the capture."""
    call, readings = consensus_read_group(document)
    consensus = build_consensus(readings, confidence=floor)
    exported = reconcile(consensus, [{}]).payload
    return {
        "anchor_read_key": call["anchor_read_key"],
        "reads": len(call["reads"]),
        "consensus_rows": len(consensus.rows),
        "recorded_consensus_rows": len(call["result_rows"]),
        "consensus_rows_match": candidate_signature(consensus.rows) == recorded_candidate_signature(
            call["result_rows"]),
        "exported_rows": len(exported.get("line_items") or []),
        "recorded_exported_rows": len((document["payload"] or {}).get("line_items") or []),
        "exported_signature_match": payload_row_signature(exported) == payload_row_signature(
            document["payload"] or {}),
        "rebuilt_export_sample": payload_row_signature(exported)[:3],
        "recorded_export_sample": payload_row_signature(document["payload"] or {})[:3],
    }


# --------------------------------------------------------------------------- trace


def header_predicates(line) -> dict:
    """Every production header predicate `from_tokens` evaluates for one line (:1162-1231)."""
    headers = invoice_evidence.HEADERS
    matches = {key: invoice_evidence._header_matches(key, line.text) for key in headers}
    if not matches["total_ht"]:
        matches["total_ht"] = invoice_evidence.is_line_total_header(line.text)
    contextual_starts = invoice_evidence._contextual_table_header_starts(line)
    for key in contextual_starts:
        matches[key] = True
    partial = (
        all(matches.get(key) for key in ("description", "quantity", "unit_price"))
        and bool(re.search(invoice_evidence.OPTIONAL_HEADERS["gross_amount"],
                           invoice_evidence.folded(line.text)))
    )
    accepted = bool(all(matches.values()) or partial)
    required = {"description", "quantity", "unit_price"} if partial else set(headers)
    starts = dict(contextual_starts)
    if accepted and not partial:
        for key, pattern in headers.items():
            if key in starts:
                continue
            index = invoice_evidence._header_token_index(line, key)
            if index is not None:
                starts[key] = line.tokens[index].box[0]
    located = required.issubset(starts) and len(set(starts.values())) == len(starts)
    return {
        "matched": sorted(key for key, value in matches.items() if value),
        "missing": sorted(key for key, value in matches.items() if not value),
        "contextual_rate_header": bool(invoice_evidence._contextual_rate_price_header(line.text)),
        "rate_token_index": invoice_evidence._rate_price_token_index(line),
        "partial_header": bool(partial),
        "accepted_as_header": accepted,
        "columns_located": located,
    }


def token_facts(line, limit: int = 14) -> list[dict]:
    facts = []
    for token in line.tokens[:limit]:
        left, top, right, bottom = token.box
        facts.append({
            "text": token.text, "x0": round(left, 1), "x1": round(right, 1),
            "y0": round(top, 1), "y1": round(bottom, 1), "confidence": round(token.confidence, 1),
        })
    return facts


def reference_row_for(text: str, reference: dict) -> int | None:
    """Which frozen reference line item this OCR line describes (word overlap, never values)."""
    words = set(invoice_evidence.folded(text).split())
    if not words:
        return None
    best, best_overlap = None, 0.0
    for index, item in enumerate(reference.get("line_items") or []):
        description = invoice_evidence.folded(str(item.get("description") or ""))
        goal = set(description.split())
        if not goal:
            continue
        overlap = len(goal & words) / len(goal)
        if overlap > best_overlap:
            best, best_overlap = index, overlap
    return best if best_overlap >= 0.5 else None


def reference_values_in_line(text: str, reference: dict, row_index: int) -> dict:
    """Which reference cell values are printed somewhere on this OCR line (text containment)."""
    normalized = invoice_evidence.folded(text)
    numbers = {number.strip() for number in re.findall(r"\d+(?:[.,]\d+)?", normalized)}
    outcome = {}
    for key in ROW_KEYS[1:]:
        value = (reference.get("line_items") or [])[row_index].get(key)
        if value is None:
            outcome[key] = "reference_null"
            continue
        text_variants = {str(value), str(value).replace(".", ","), str(value).rstrip("0").rstrip(".")}
        outcome[key] = "printed" if any(
            variant.replace(" ", "") in numbers or variant in normalized for variant in text_variants
        ) else "absent"
    return outcome


def trace_document(document: dict) -> dict:
    """Per-read production header facts, tied to what the recorded read actually did."""
    reference = document["reference"]
    per_read, causes, closest = {}, {}, {}
    for read in all_reads(document):
        rebuilt = rebuild_read(read)
        lines = list(rebuilt.lines)
        verdicts = [header_predicates(line) for line in lines]
        accepted = [index for index, verdict in enumerate(verdicts) if verdict["accepted_as_header"]
                    and verdict["columns_located"]]
        header_candidates = [index for index, verdict in enumerate(verdicts) if verdict["accepted_as_header"]]
        near = max(range(len(lines)),
                   key=lambda index: (len(verdicts[index]["matched"]), -len(lines[index].text))) if lines else None
        data_rows = [
            {
                "line_index": index,
                "text": lines[index].text[:90],
                "reference_row": row,
                "reference_values": reference_values_in_line(lines[index].text, reference, row),
                "tokens": token_facts(lines[index], 10),
            }
            for index, line in enumerate(lines)
            if len(line.tokens) >= 2 and (row := reference_row_for(line.text, reference)) is not None
        ][:3]
        key = ",".join(near and verdicts[near]["missing"] or ["no_lines"]) if near is not None else "no_lines"
        if not accepted:
            causes[key] = causes.get(key, 0) + 1
            closest.setdefault(key, lines[near].text[:90] if near is not None else None)
        per_read[read["read_id"]] = {
            "kind": read["kind"],
            "family": read["family"],
            "read_label": read["read_label"],
            "recorded_rows": read["rows"],
            "rebuilt_rows": len(rebuilt.rows),
            "capture_replay": read["capture_replay"],
            "tsv_rows": len(read["tsv"]["text"]) if read.get("tsv") else None,
            "tokens_after_filter": len(rebuilt.tokens),
            "lines": len(lines),
            "header_candidate_lines": [
                {"line_index": index, "text": lines[index].text[:90], **verdicts[index]}
                for index in header_candidates
            ],
            "accepted_header_lines": [
                {"line_index": index, "text": lines[index].text[:90], "tokens": token_facts(lines[index], 12)}
                for index in accepted
            ],
            "closest_header": {
                "line_index": near, "text": lines[near].text[:90] if near is not None else None,
                **verdicts[near], "tokens": token_facts(lines[near], 12),
            } if near is not None else None,
            "reference_data_lines_without_rows": data_rows if not accepted else [],
            "responsible_condition": (
                "header line accepted, columns located (src/invoice_evidence.py:1178-1231)"
                if accepted else
                "no line satisfies all(matches) or partial_header (:1174-1178); every data line "
                f"is skipped by the header gate (:1241); missing labels on the closest line: {key}"
            ),
        }
    return {
        "reads": len(per_read),
        "reads_with_accepted_header": sum(
            1 for entry in per_read.values() if entry["accepted_header_lines"]),
        "reads_with_candidate_rows": sum(1 for entry in per_read.values() if entry["recorded_rows"]),
        "rows_in_reads": sum(entry["recorded_rows"] for entry in per_read.values()),
        "replay_rows_match_for_every_read": all(
            entry["rebuilt_rows"] == entry["recorded_rows"] for entry in per_read.values()),
        "missing_key_combinations_when_no_header": causes,
        "closest_line_per_combination": closest,
        "reference_data_lines_lost": sum(
            len(entry["reference_data_lines_without_rows"]) for entry in per_read.values()),
        "per_read": per_read,
    }


# --------------------------------------------------------------------------- variants


def french_rate_header(text: str) -> bool:
    """Mechanism M1-helper: the French rate word is a rate/price label."""
    if original_rate_price_header(text):
        return True
    normalized = invoice_evidence._header_text(text)
    headers = invoice_evidence.HEADERS
    if not (
        re.search(headers["description"], normalized)
        and re.search(headers["quantity"], normalized)
        and (re.search(headers["total_ht"], normalized)
             or invoice_evidence.is_line_total_header(text))
    ):
        return False
    aliases = list(re.finditer(r"\b(?:rate|price|prix|tarif)\b", normalized))
    return len(aliases) == 1


def french_rate_token_index(line) -> int | None:
    """The locating half of the same rate-label rule (`_rate_price_token_index`)."""
    index = original_rate_price_index(line)
    if index is not None:
        return index
    labels = [invoice_evidence.normalized_label(token.text).strip() for token in line.tokens]
    french = [position for position, label in enumerate(labels) if label in {"prix", "tarif"}]
    return french[0] if len(french) == 1 else None


def patched_headers(**changes):
    return patch.object(invoice_evidence, "HEADERS", {**BASE_HEADERS, **changes})


VARIANTS = {
    "as_built": {
        "description": "unpatched production row builder replayed from recorded input",
        "mechanism": "none (baseline)",
        "patches": (),
    },
    "unit_price_vocabulary": {
        "description": (
            "one mechanism, one vocabulary table: `HEADERS['unit_price']` also accepts the bare "
            "French rate column label (`Prix`, `Tarif`), so the plain header route both matches "
            "and locates the column"
        ),
        "mechanism": "M1 vocabulary (HEADERS only)",
        "patches": ((patched_headers(unit_price=BASE_HEADERS["unit_price"] + FRENCH_RATE_LABEL),),),
    },
    "tax_aware_labels": {
        "description": (
            "the M1 vocabulary edit plus the bare `TVA` column label in `OPTIONAL_HEADERS['tax_rate']`: "
            "two lookups, kept together because the question is whether labelling the tax column is "
            "enough for the VAT-inclusive amount column to stop being read as a line total"
        ),
        "mechanism": "M1 vocabulary + tax column label (probe of the amount-column semantics gap)",
        "patches": ((patched_headers(unit_price=BASE_HEADERS["unit_price"] + FRENCH_RATE_LABEL),
                     patch.object(invoice_evidence, "OPTIONAL_HEADERS", {
                         **invoice_evidence.OPTIONAL_HEADERS,
                         "tax_rate": invoice_evidence.OPTIONAL_HEADERS["tax_rate"] + r"|\btva\b|\bvat\b"}),),),
    },
    "hourly_vocabulary": {
        "description": (
            "one mechanism, one vocabulary table: `HEADERS['quantity']` also accepts `heures` and "
            "`HEADERS['unit_price']` also accepts an `€/heure`-style rate label (a time-billed "
            "table prints neither `Quantité` nor `Prix unitaire`)"
        ),
        "mechanism": "M2 vocabulary (HEADERS only)",
        "patches": ((patched_headers(
            quantity=BASE_HEADERS["quantity"] + HOURLY_QUANTITY_LABEL,
            unit_price=BASE_HEADERS["unit_price"] + HOURLY_RATE_LABEL),),),
    },
    "all_printed_labels": {
        "description": (
            "the two vocabulary edits together: M1 + M2 in `HEADERS`, no helper patched"
        ),
        "mechanism": "M1 + M2 vocabulary ceiling (HEADERS only)",
        "patches": ((patched_headers(
            quantity=BASE_HEADERS["quantity"] + HOURLY_QUANTITY_LABEL,
            unit_price=BASE_HEADERS["unit_price"] + FRENCH_RATE_LABEL + HOURLY_RATE_LABEL),),),
    },
    "french_rate_predicate": {
        "description": (
            "one mechanism, one lookup: `_contextual_rate_price_header` also accepts the French "
            "rate word; the column position is still located only by the unpatched index helper"
        ),
        "mechanism": "M1 contextual route, predicate half",
        "patches": ((patch.object(invoice_evidence, "_contextual_rate_price_header", french_rate_header),),),
    },
    "french_rate_column": {
        "description": (
            "one mechanism, both halves of the same contextual rate rule: the predicate accepts "
            "the French word and `_rate_price_token_index` locates it"
        ),
        "mechanism": "M1 contextual route, predicate + token index",
        "patches": (
            (patch.object(invoice_evidence, "_contextual_rate_price_header", french_rate_header),
             patch.object(invoice_evidence, "_rate_price_token_index", french_rate_token_index)),
        ),
    },
}


def cell_outcomes(payload: dict, reference: dict) -> dict:
    """Per-field verdicts for the rows `score()` matched, mirroring its best-agreement pairing."""
    export_rows = payload.get("line_items") or []
    gold_rows = reference.get("line_items") or []
    outcomes = {key: {"correct": [], "incorrect": [], "missing": [], "contradicts_null_reference": []}
                for key in ROW_KEYS}
    used = set()
    for gold_index, gold in enumerate(gold_rows):
        best, best_score = None, 0
        for index, row in enumerate(export_rows):
            if index in used:
                continue
            agreeing = sum(
                1 for key in ROW_KEYS
                if gold.get(key) is not None and equal((row or {}).get(key), gold.get(key))
            )
            if agreeing > best_score:
                best, best_score = index, agreeing
        if best is None:
            continue
        used.add(best)
        for key in ROW_KEYS:
            gold_value, exported_value = gold.get(key), (export_rows[best] or {}).get(key)
            record = {"document_field": key, "gold_index": gold_index, "export_index": best,
                      "reference": gold_value, "exported": exported_value,
                      "exported_description": (export_rows[best] or {}).get("description")}
            if gold_value is None:
                if exported_value is not None:
                    outcomes[key]["contradicts_null_reference"].append(record)
            elif exported_value is None:
                outcomes[key]["missing"].append(record)
            elif equal(exported_value, gold_value):
                outcomes[key]["correct"].append(record)
            else:
                outcomes[key]["incorrect"].append(record)
    unsupported = [
        {"export_index": index, "description": (row or {}).get("description"),
         "values": {key: (row or {}).get(key) for key in ROW_KEYS[1:]}}
        for index, row in enumerate(export_rows) if index not in used
    ]
    return {"cells": outcomes, "unsupported_exported_rows": unsupported,
            "matched_rows": len(used), "unmatched_reference_rows": len(gold_rows) - len(used)}


def variant_totals(scoring: dict, counts: dict) -> dict:
    """`counts` is the per-field outcome tally returned by `cell_outcomes()[\"cells\"]`."""
    return {
        "reference_rows": scoring["reference_rows"],
        "exported_rows": scoring["exported_rows"],
        "rows_with_any_support": scoring["rows_with_any_support"],
        "rows_missing": scoring["rows_missing"],
        "unsupported_exported_rows": scoring["unsupported_exported_rows"],
        "empty_exported_rows": scoring["empty_exported_rows"],
        "cells_correct": sum(items["correct"] for items in counts.values()),
        "cells_incorrect": sum(items["incorrect"] for items in counts.values()),
        "cells_missing": sum(items["missing"] for items in counts.values()),
        "cells_contradicting_null_reference": sum(
            items["contradicts_null_reference"] for items in counts.values()),
    }


def run_variant(capture: dict, variant: str, floor: int) -> dict:
    """Rebuild every document's final consensus and export under one patched mechanism."""
    spec = VARIANTS[variant]
    documents = {}
    stack = contextlib.ExitStack()
    try:
        for group in spec["patches"]:
            for entered in group:
                stack.enter_context(entered)
        for document in capture["documents"]:
            started = time.perf_counter()
            call, readings = consensus_read_group(document)
            candidates_by_read = {
                read["read_id"]: len(rebuild_read(read).rows) for read in all_reads(document)
            }
            final_candidates = {
                f"{family}[{position}]": len(evidence.rows)
                for family, group in readings.items() for position, evidence in enumerate(group)
            }
            reads_with_candidate_rows = sum(1 for count in candidates_by_read.values() if count)
            consensus = build_consensus(readings, confidence=floor)
            exported = reconcile(consensus, [{}]).payload
            scoring = score(exported, document["reference"])
            outcomes = cell_outcomes(exported, document["reference"])
            consistency = {
                "correct_matches_score": sum(
                    len(items["correct"]) for items in outcomes["cells"].values())
                == sum(scoring["cells"][key]["correct"] for key in ROW_KEYS),
                "reference_matches_score": sum(
                    len(items["correct"]) + len(items["incorrect"]) + len(items["missing"])
                    for items in outcomes["cells"].values())
                == sum(scoring["cells"][key]["reference"] for key in ROW_KEYS),
            }
            documents[document["document"]] = {
                "reads_with_accepted_header": sum(
                    1 for entry in trace_document_light(document).values() if entry),
                "reads_with_candidate_rows": reads_with_candidate_rows,
                "candidate_rows": sum(candidates_by_read.values()),
                "final_group_candidate_rows": sum(final_candidates.values()),
                "final_group_candidates": final_candidates,
                "consensus_rows": len(consensus.rows),
                "consensus_issue_strings": [str(issue)[:200] for issue in consensus.issues][:6],
                "candidate_row_survival": (
                    round(len(consensus.rows) / sum(final_candidates.values()), 3)
                    if sum(final_candidates.values()) else None),
                "scoring": scoring,
                "outcomes": {key: {outcome: len(items) for outcome, items in lists.items()}
                             for key, lists in outcomes["cells"].items()},
                "detail": {
                    "incorrect": [item for items in outcomes["cells"].values() for item in items["incorrect"]],
                    "contradicts_null_reference": [
                        item for items in outcomes["cells"].values()
                        if items["contradicts_null_reference"] for item in items["contradicts_null_reference"]],
                    "unsupported_exported_rows": outcomes["unsupported_exported_rows"],
                },
                "scoring_consistency": consistency,
                "parse_seconds": round(time.perf_counter() - started, 2),
            }
    finally:
        stack.close()
    return documents


def trace_document_light(document: dict) -> dict:
    """Header-gate verdict per read, evaluated with whatever patch is currently active."""
    outcome = {}
    for read in all_reads(document):
        lines = rebuild_read(read).lines
        outcome[read["read_id"]] = any(
            (facts := header_predicates(line))["accepted_as_header"] and facts["columns_located"]
            for line in lines
        )
    return outcome


# --------------------------------------------------------------------------- main


def main_() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--skip-variants", action="store_true")
    args = parser.parse_args()

    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    floor = capture["documents"][0]["consensus_calls"][0]["confidence_floor"]
    if floor != OCR_CONFIG.confidence:
        raise SystemExit(f"recorded confidence floor {floor} differs from OCR_CONFIG {OCR_CONFIG.confidence}")

    report = {
        "purpose": "PR84 row-building diagnosis: production builder replayed on recorded tokens, "
                   "one patched mechanism per variant, model never called",
        "model_called": False,
        "ocr_rerun": False,
        "ocr_policy": capture["ocr_policy"],
        "fingerprint": capture["fingerprint"],
        "ocr_environment": capture["ocr_environment"],
        "traineddata": capture["traineddata"],
        "pipeline_identity": capture["pipeline_identity"],
        "capture_artifact": args.capture.relative_to(ROOT).as_posix(),
        "confidence_floor": floor,
        "replay_scope": (
            "from_tsv/from_tokens on recorded TSV and token records, then the final multi-family "
            "build_consensus call and reconcile() grounding; the post-consensus quantity "
            "clarification and fast-path metadata of the live run are outside this replay"
        ),
        "production_entry_points": {
            "header_gate": "src/invoice_evidence.py:1241 (line skipped unless a header opened columns/starts)",
            "header_acceptance": "src/invoice_evidence.py:1174-1179",
            "column_location": "src/invoice_evidence.py:1207-1231",
            "rate_label_route": "src/invoice_evidence.py:475 and 558-559",
            "row_acceptance": "src/invoice_evidence.py:1322 (two of quantity/unit_price/total_ht)",
        },
        "variants": {name: {"description": spec["description"], "mechanism": spec["mechanism"]}
                     for name, spec in VARIANTS.items()},
        "documents": [],
    }

    for document in capture["documents"]:
        entry = {
            "document": document["document"],
            "layout_family": document["layout_family"],
            "reference_rows": len(document["reference"].get("line_items") or []),
            "read_replay_gate": gate_read_replay(document),
            "consensus_replay_gate": gate_consensus_replay(document, floor),
            "trace": trace_document(document),
            "as_built_scoring": score(document["payload"], document["reference"]),
            "as_built_outcomes": cell_outcomes(document["payload"], document["reference"]),
        }
        entry["baseline_export_matches_recorded"] = entry["consensus_replay_gate"]["exported_signature_match"]
        report["documents"].append(entry)
        print(
            f'{entry["document"]}: reads={entry["trace"]["reads"]} '
            f'with_header={entry["trace"]["reads_with_accepted_header"]} '
            f'with_rows={entry["trace"]["reads_with_candidate_rows"]} '
            f'rows={entry["trace"]["rows_in_reads"]} '
            f'lost_reference_lines={entry["trace"]["reference_data_lines_lost"]} '
            f'gate1={not entry["read_replay_gate"]["failures"]} '
            f'gate2={entry["consensus_replay_gate"]["consensus_rows_match"]} '
            f'export_match={entry["baseline_export_matches_recorded"]} '
            f'causes={entry["trace"]["missing_key_combinations_when_no_header"]}'
        )

    parity_failures = [
        {"document": entry["document"], "stage": "read_replay",
         "detail": entry["read_replay_gate"]["failures"][:3]}
        for entry in report["documents"] if entry["read_replay_gate"]["failures"]
    ] + [
        {"document": entry["document"], "stage": "consensus_replay",
         "detail": entry["consensus_replay_gate"]}
        for entry in report["documents"]
        if not (entry["consensus_replay_gate"]["consensus_rows_match"]
                and entry["baseline_export_matches_recorded"])
    ] + [
        {"document": entry["document"], "stage": "trace_replay",
         "detail": entry["trace"]["per_read"]}
        for entry in report["documents"]
        if not entry["trace"]["replay_rows_match_for_every_read"]
    ]
    report["replay_parity_failures"] = len(parity_failures)
    report["replay_parity_detail"] = parity_failures
    report["baseline_totals"] = {
        key: sum(entry["as_built_scoring"][key] for entry in report["documents"])
        for key in ("reference_rows", "exported_rows", "rows_with_any_support", "rows_missing",
                    "unsupported_exported_rows", "empty_exported_rows")
    }
    report["baseline_totals"]["cells"] = {
        field: {outcome: sum(entry["as_built_scoring"]["cells"][field][outcome]
                             for entry in report["documents"])
                for outcome in ("correct", "reference", "unverified")}
        for field in ROW_KEYS
    }
    report["baseline_totals"]["cell_outcomes"] = {
        outcome: sum(len(entry["as_built_outcomes"]["cells"][field][outcome])
                     for entry in report["documents"] for field in ROW_KEYS)
        for outcome in ("correct", "incorrect", "missing", "contradicts_null_reference")
    }
    report["baseline_by_document"] = {
        entry["document"]: variant_totals(entry["as_built_scoring"], {
            field: {outcome: len(items[outcome]) for outcome in
                    ("correct", "incorrect", "missing", "contradicts_null_reference")}
            for field, items in entry["as_built_outcomes"]["cells"].items()})
        for entry in report["documents"]
    }

    if not args.skip_variants:
        if parity_failures:
            raise SystemExit(f"variant comparison refused: {len(parity_failures)} replay parity failures")
        report["variant_scoring"] = {}
        for variant in VARIANTS:
            scored = run_variant(capture, variant, floor)
            report["variant_scoring"][variant] = {
                label: {key: data[key] for key in (
                    "reads_with_accepted_header", "reads_with_candidate_rows", "candidate_rows",
                    "final_group_candidate_rows", "consensus_rows", "consensus_issue_strings",
                    "candidate_row_survival", "parse_seconds")}
                | {"rows_with_any_support": data["scoring"]["rows_with_any_support"],
                   "rows_missing": data["scoring"]["rows_missing"],
                   "unsupported_exported_rows": data["scoring"]["unsupported_exported_rows"],
                   "cells": data["outcomes"],
                   "detail": data["detail"],
                   "scoring_consistency": data["scoring_consistency"]}
                for label, data in scored.items()
            }
            totals = {
                key: sum(data[key] for data in scored.values())
                for key in ("reads_with_accepted_header", "reads_with_candidate_rows",
                            "candidate_rows", "final_group_candidate_rows", "consensus_rows")
            }
            per_document = {
                label: variant_totals(data["scoring"], data["outcomes"])
                for label, data in scored.items()
            }
            report.setdefault("variant_totals", {})[variant] = {
                **{key: sum(items[key] for items in per_document.values()) for key in next(
                    iter(per_document.values()))},
                "by_document": per_document,
                "totals_view": totals,
            }
            print(f"variant {variant}: {totals} "
                  f"correct={sum(v['cells_correct'] for v in per_document.values())} "
                  f"incorrect={sum(v['cells_incorrect'] for v in per_document.values())} "
                  f"contradicting_null={sum(v['cells_contradicting_null_reference'] for v in per_document.values())}")

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {args.output}")
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("written", args.output.resolve().relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
