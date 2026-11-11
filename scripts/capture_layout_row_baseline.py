"""PR83 OCR-only capture runner for the six frozen new-layout documents.

Runs the current production pipeline without any extraction model and records the
exact `readings` dictionary each `build_consensus` call received, so row-level
consensus decisions can be audited and replayed offline. Writes only inside the
PR83 artifact directory and refuses to create a file that already exists: no
PR64-PR82 capture, artifact or fixture can be overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from app import main
from capture_product_review64_ocr_baseline import (
    TESSDATA_FILES, json_safe, pixel_fingerprint, removed_candidates, serialize_evidence,
    serialize_row, sha256_file,
)
from evaluate_grounding import equal
from invoice_evidence import from_tsv
from invoice_grounding import reconcile
from ocr_config import OCR_CONFIG
import ocr_provenance

FIXTURES = ROOT / "diagnostics" / "holdout" / "new-layout"
OUTPUT_DIR = ROOT / "results" / "product-review83"
CAPTURE_NAME = "pr83-layout-row-capture.json"
# Approved v2 references for the shared service/travel family; v1 stays frozen and unused.
LAYOUTS = {
    "layout-a": ("sumup-avoir", "truth-sumup-avoir.json"),
    "layout-b": ("consulting", "truth-consulting.json"),
    "layout-c": ("retail", "truth-retail.json"),
    "layout-d": ("service", "truth-service-v2.json"),
    "layout-e": ("travel", "truth-travel-v2.json"),
    "layout-f": ("photographer", "truth-photographer.json"),
}
CELL_FIELDS = ("quantity", "unit_price", "total_ht")


def guard(target: Path, directory: Path) -> Path:
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {target}")
    resolved = target.resolve()
    if directory.resolve() not in resolved.parents:
        raise SystemExit(f"artifact path outside the PR83 directory: {target}")
    return resolved


def traineddata_identity(prefix: Path | None) -> dict:
    """Hash the language data actually present, so a run can prove which files it read."""
    directory = prefix or (Path(pytesseract.pytesseract.tesseract_cmd or "").parent / "tessdata")
    identity = {}
    for language, pinned in TESSDATA_FILES.items():
        path = directory / f"{language}.traineddata"
        if not path.exists():
            identity[language] = {"present": False}
            continue
        digest = sha256_file(path)
        identity[language] = {
            "present": True,
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "matches_pinned_release": digest == pinned["sha256"],
        }
    return identity


def serialize_row_candidate(row, *, confidence: int, index: int = 0) -> dict:
    record = serialize_row(row, index)
    confidences = [token.confidence for token in row.line.tokens]
    record["line_text"] = row.line.text
    record["line_min_confidence"] = min(confidences) if confidences else None
    record["line_mean_confidence"] = (
        round(sum(confidences) / len(confidences), 2) if confidences else None
    )
    record["line_all_confident"] = bool(confidences) and all(
        value >= confidence for value in confidences
    )
    for field, numbers in record["cells"].items():
        for number in numbers:
            left, top, right, bottom = number["box"]
            covering = [
                token.confidence for token in row.line.tokens
                if left <= token.box[0] < right and top <= token.box[1] < bottom
            ]
            number["covering_min_confidence"] = min(covering) if covering else None
            number["confident"] = bool(covering) and all(value >= confidence for value in covering)
    return record


def row_population_score(evidence) -> int:
    return sum(
        sum(bool(row.cells.get(key)) for key in CELL_FIELDS) for row in evidence.rows
    )


def summarize_read(evidence, family: str, position: int, *, confidence: int) -> dict:
    tokens = list(evidence.tokens)
    return {
        "read_key": f"{family}[{position}]",
        "family": family,
        "rows": len(evidence.rows),
        "lines": len(evidence.lines),
        "populated_row_score": row_population_score(evidence),
        # The page text of this read only: it shows whether a reference value was
        # recognized at all before any table-row interpretation was attempted.
        "text": evidence.text,
        "mean_token_confidence": (
            round(sum(token.confidence for token in tokens) / len(tokens), 3) if tokens else None
        ),
        "row_candidates": [
            serialize_row_candidate(row, confidence=confidence, index=index)
            for index, row in enumerate(evidence.rows)
        ],
    }


def trace(path: Path, language: str, confidence: int) -> dict:
    """Run the production pipeline once, recording every consensus call it performs."""
    prepared_family: dict[int, str] = {}
    last_label = ["?"]
    calls: list[dict] = []
    funnel: list[dict] = []

    def record_transform(*args, **kwargs):
        image, meta = original_transform(*args, **kwargs)
        prepared_family[id(image)] = meta["method"]
        return image, meta

    def record_from_tsv(data, *args, **kwargs):
        evidence = original_from_tsv(data, *args, **kwargs)
        evidence.family = last_label[0]
        funnel.append({
            "label": last_label[0],
            "rows": len(evidence.rows),
            "tokens": len(evidence.tokens),
            "mean_token_confidence": (
                round(sum(t.confidence for t in evidence.tokens) / len(evidence.tokens), 2)
                if evidence.tokens else None
            ),
        })
        return evidence

    def record_build_consensus(readings, **kwargs):
        resolved_confidence = kwargs.get("confidence", confidence)
        live_reads = []
        for family, group in readings.items():
            for position, evidence in enumerate(group):
                live_reads.append((f"{family}[{position}]", evidence))
        call = {
            "call_index": len(calls),
            "readings": {
                family: [
                    summarize_read(evidence, family, position, confidence=resolved_confidence)
                    for position, evidence in enumerate(group)
                ]
                for family, group in readings.items()
            },
            "confidence_floor": resolved_confidence,
        }
        result = original_build_consensus(readings, **kwargs)
        call["result_rows"] = [
            serialize_row_candidate(row, confidence=resolved_confidence, index=index)
            for index, row in enumerate(result.rows)
        ]
        call["result_issue_strings"] = list(result.issues)
        call["result_quality_issues"] = list(result.quality_issues)
        call["consensus_evidence"] = serialize_evidence(result, include_tokens=False)
        serialized_by_key = {
            read["read_key"]: read["row_candidates"]
            for group in call["readings"].values() for read in group
        }
        call["removed_candidates"] = removed_candidates(
            [
                {
                    "read_index": index, "family": read_key,
                    "parsed_candidates": serialized_by_key[read_key],
                    "_parsed_object": evidence,
                }
                for index, (read_key, evidence) in enumerate(live_reads)
            ],
            result,
        )
        calls.append(call)
        return result

    def reader(prepared, psm):
        last_label[0] = f"{prepared_family.get(id(prepared), 'derived')}|psm{psm}"
        return pytesseract.image_to_data(
            prepared, lang=language, config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

    import ocr_recovery
    original_transform = ocr_recovery.transform
    original_from_tsv = from_tsv
    original_build_consensus = ocr_recovery.build_consensus
    started = time.perf_counter()
    with patch.object(ocr_recovery, "transform", record_transform), patch.object(
        ocr_recovery, "from_tsv", record_from_tsv
    ), patch.object(ocr_recovery, "build_consensus", record_build_consensus):
        evidence = main.recover_image(main.open_image(str(path)), reader)
    elapsed = time.perf_counter() - started
    payload = reconcile(evidence, [{}]).payload

    for call in calls:
        all_reads = [read for group in call["readings"].values() for read in group]
        anchor = max(
            all_reads,
            key=lambda read: (read["populated_row_score"], read["mean_token_confidence"] or 0),
        )
        call["anchor_read_key"] = anchor["read_key"]
        call["candidate_reads"] = len(all_reads)
        call["candidate_rows"] = sum(len(read["row_candidates"]) for read in all_reads)

    return {
        "seconds": round(elapsed, 1),
        "tesseract_calls": len(funnel),
        "read_funnel": funnel,
        "consensus_calls": calls,
        "payload": json_safe(payload),
    }


def score(payload: dict, reference: dict) -> dict:
    """Row-level scoring against the frozen reference, matched by the best agreeing cell set."""
    export_rows = payload.get("line_items") or []
    gold_rows = reference.get("line_items") or []
    cell_counts = {key: {"correct": 0, "reference": 0, "unverified": 0}
                   for key in ("description", *CELL_FIELDS)}
    used, matched, unmatched_gold = set(), [], []
    for gold_index, gold in enumerate(gold_rows):
        best, best_score = None, 0
        for index, row in enumerate(export_rows):
            if index in used:
                continue
            agreeing = sum(
                1 for key in ("description", *CELL_FIELDS)
                if gold.get(key) is not None and equal((row or {}).get(key), gold.get(key))
            )
            if agreeing > best_score:
                best, best_score = index, agreeing
        if best is None:
            unmatched_gold.append(gold_index)
            continue
        used.add(best)
        matched.append({"gold_index": gold_index, "export_index": best, "agreeing_cells": best_score})
        for key in cell_counts:
            gold_value = gold.get(key)
            if gold_value is None:
                cell_counts[key]["unverified"] += 1
                continue
            cell_counts[key]["reference"] += 1
            cell_counts[key]["correct"] += int(equal((export_rows[best] or {}).get(key), gold_value))
    empty_exported_rows = sum(
        1 for row in export_rows if not any((row or {}).get(key) is not None for key in ("description", *CELL_FIELDS))
    )
    return {
        "reference_rows": len(gold_rows),
        "exported_rows": len(export_rows),
        "rows_with_any_support": len(matched),
        "rows_missing": len(unmatched_gold),
        "unsupported_exported_rows": len(export_rows) - len(matched),
        "empty_exported_rows": empty_exported_rows,
        "cells": cell_counts,
        "header": {
            field: {
                "exported": payload.get(field),
                "reference": reference.get(field),
                "state": ("unverified" if reference.get(field) is None else
                          "correct" if equal(payload.get(field), reference.get(field)) else "incorrect"),
            }
            for field in ("supplier_name", "invoice_date", "total_ht", "total_ttc")
        },
    }


def main_() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--capture-name", default=CAPTURE_NAME)
    parser.add_argument("--only", nargs="*", choices=sorted(LAYOUTS), default=sorted(LAYOUTS))
    parser.add_argument(
        "--allow-language-fallback", action="store_true",
        help="measure a degraded language environment deliberately; the artifact then records fallback=true",
    )
    args = parser.parse_args()

    env = ocr_provenance.environment(require_all=not args.allow_language_fallback)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = guard(output_dir / args.capture_name, output_dir)
    prefix = Path(env["tessdata_prefix"]) if env["tessdata_prefix"] else None

    manifest_path = args.fixtures / "manifest-new-layout-six.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"documents": {}}
    )
    report = {
        "purpose": "PR83 six-layout OCR-only row baseline; production decisions untouched, model never called",
        "model_called": False,
        "artifact_name": args.capture_name,
        "ocr_policy": OCR_CONFIG.version,
        "fingerprint": OCR_CONFIG.fingerprint(),
        "ocr_environment": env,
        "traineddata": traineddata_identity(prefix),
        "pipeline_identity": {
            "tesseract_cmd": env["tesseract_cmd"],
            "tesseract_version": env["tesseract_version"],
            "source_sha256": {
                relative: sha256_file(ROOT / relative)
                for relative in (
                    "src/ocr_consensus.py", "src/ocr_recovery.py", "src/invoice_evidence.py",
                    "src/invoice_grounding.py", "src/ocr_config.py",
                )
            },
        },
        "fixtures": {
            "directory": args.fixtures.relative_to(ROOT).as_posix(),
            "frozen_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        },
        "documents": [],
    }

    for label in args.only:
        slug, truth_name = LAYOUTS[label]
        image_path = args.fixtures / f"{slug}-anon.png"
        truth_path = args.fixtures / truth_name
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        reference = dict(truth["document"])
        reference["total_ht"] = reference.get("subtotal_ht")
        expected_copy = manifest.get("documents", {}).get(slug, {}).get("copy_sha256")
        actual_copy = sha256_file(image_path)
        if expected_copy and expected_copy != actual_copy:
            raise SystemExit(f"fixture {image_path.name} no longer matches the frozen manifest")
        result = trace(image_path, env["reader_languages"], OCR_CONFIG.confidence)
        result.update({
            "document": label,
            "layout_family": truth.get("layout_family"),
            "source_type": truth.get("source_type"),
            "genuine_business_invoice": truth.get("genuine_business_invoice"),
            "copy_sha256": actual_copy,
            "pixel_fingerprint": pixel_fingerprint(main.open_image(str(image_path))),
            "truth_file": truth_name,
            "truth_sha256": sha256_file(truth_path),
            "reference": json_safe(reference),
            "scoring": score(result["payload"], reference),
        })
        report["documents"].append(result)
        scoring = result["scoring"]
        print(
            f"{label} ({slug}): rows {scoring['exported_rows']}/{scoring['reference_rows']} exported, "
            f"{scoring['rows_with_any_support']} matched, {scoring['unsupported_exported_rows']} unsupported "
            f"({scoring['empty_exported_rows']} empty) | calls={result['tesseract_calls']} "
            f"consensus_calls={len(result['consensus_calls'])} | {result['seconds']}s"
        )

    report["totals"] = {
        "documents": len(report["documents"]),
        "reference_rows": sum(d["scoring"]["reference_rows"] for d in report["documents"]),
        "exported_rows": sum(d["scoring"]["exported_rows"] for d in report["documents"]),
        "rows_missing": sum(d["scoring"]["rows_missing"] for d in report["documents"]),
        "unsupported_exported_rows": sum(d["scoring"]["unsupported_exported_rows"] for d in report["documents"]),
        "empty_exported_rows": sum(d["scoring"]["empty_exported_rows"] for d in report["documents"]),
        "cells": {
            key: {
                outcome: sum(d["scoring"]["cells"][key][outcome] for d in report["documents"])
                for outcome in ("correct", "reference", "unverified")
            }
            for key in ("description", *CELL_FIELDS)
        },
        "tesseract_calls": sum(d["tesseract_calls"] for d in report["documents"]),
        "pipeline_seconds": round(sum(d["seconds"] for d in report["documents"]), 1),
    }
    capture_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("written", capture_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
