"""PR84 token-level OCR-only capture for the six frozen new-layout documents.

PR83 recorded what `build_consensus` received. This records what `from_tsv` received:
the raw Tesseract TSV arrays of every read, so the *production* row builder can be
replayed offline with its real geometry and no OCR rerun.

Research tooling only. It patches nothing in `src/`, calls no model, writes only inside
`results/product-review84/` and refuses to create a file that already exists, so no
PR64-PR83 artifact, capture or fixture can be overwritten.
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
from capture_layout_row_baseline import (
    CELL_FIELDS, FIXTURES, LAYOUTS, row_population_score, serialize_row_candidate,
    traineddata_identity,
)
from capture_product_review64_ocr_baseline import json_safe, pixel_fingerprint, sha256_file
from invoice_evidence import Token, from_tokens, from_tsv
from invoice_grounding import reconcile
from ocr_config import OCR_CONFIG
import ocr_provenance

OUTPUT_DIR = ROOT / "results" / "product-review84"
CAPTURE_NAME = "pr84-row-building-capture.json"
TSV_COLUMNS = ("text", "conf", "left", "top", "width", "height")


def guard(target: Path, directory: Path) -> Path:
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {target}")
    resolved = target.resolve()
    if directory.resolve() not in resolved.parents:
        raise SystemExit(f"artifact path outside the PR84 directory: {target}")
    return resolved


def compact_tsv(data: dict) -> dict:
    """Keep only the columns `from_tsv` reads, with their original index alignment."""
    return {column: [str(value) for value in data[column]] for column in TSV_COLUMNS}


def serialize_tokens(tokens: list) -> list[dict]:
    return [
        {"text": token.text, "page": token.page, "box": [float(v) for v in token.box],
         "confidence": float(token.confidence)}
        for token in tokens
    ]


def replay_from_capture(record: dict):
    """Rebuild one recorded read through the unmodified production parser."""
    if record["kind"] == "tsv":
        return from_tsv({column: list(values) for column, values in record["tsv"].items()},
                        **(record.get("kwargs") or {}))
    tokens = [
        Token(item["text"], item["page"], tuple(item["box"]), item["confidence"])
        for item in record["tokens"]
    ]
    return from_tokens(tokens, locale=record["locale"], literal_columns=record["literal_columns"])


def trace(path: Path, language: str, confidence: int) -> dict:
    """Run the production pipeline once, recording the TSV handed to every read."""
    prepared_family: dict[int, str] = {}
    last_label = ["?"]
    recorded: dict[int, dict] = {}
    keep_alive: list = []
    calls: list[dict] = []
    funnel: list[dict] = []

    def record_transform(*args, **kwargs):
        image, meta = original_transform(*args, **kwargs)
        prepared_family[id(image)] = meta["method"]
        return image, meta

    def register(kind, evidence, record):
        keep_alive.append(evidence)  # id() is a lookup key: recycled ids would misattribute reads
        evidence.family = last_label[0]
        rebuilt = replay_from_capture(dict(record, kind=kind))
        record = dict(record, kind=kind, label=last_label[0], rows=len(evidence.rows),
                      lines=len(evidence.lines), text=evidence.text,
                      capture_replay={
                          "rows_match": len(rebuilt.rows) == len(evidence.rows),
                          "text_match": rebuilt.text == evidence.text,
                      })
        recorded[id(evidence)] = record
        funnel.append({
            "label": last_label[0],
            "kind": kind,
            "rows": len(evidence.rows),
            "tokens": len(evidence.tokens),
            "recorded_rows": len(record.get("tsv", {}).get("text", record.get("tokens", []))),
            "mean_token_confidence": (
                round(sum(t.confidence for t in evidence.tokens) / len(evidence.tokens), 2)
                if evidence.tokens else None
            ),
        })
        return evidence

    def record_from_tsv(data, *args, **kwargs):
        evidence = original_from_tsv(data, *args, **kwargs)
        return register("tsv", evidence, {"tsv": compact_tsv(data), "kwargs": dict(kwargs)})

    def record_from_tokens(tokens, *args, **kwargs):
        evidence = original_from_tokens(tokens, *args, **kwargs)
        return register("tokens", evidence, {
            "tokens": serialize_tokens(tokens),
            "locale": kwargs["locale"] if "locale" in kwargs else (args[0] if args else None),
            "literal_columns": kwargs.get("literal_columns", True),
        })

    def record_build_consensus(readings, **kwargs):
        resolved_confidence = kwargs.get("confidence", confidence)
        all_reads = []
        for family, group in readings.items():
            for position, evidence in enumerate(group):
                record = recorded.get(id(evidence))
                if record is None:
                    raise SystemExit(f"read {family}[{position}] has no recorded input")
                read = {
                    "read_key": f"{family}[{position}]",
                    "family": family,
                    "kind": record["kind"],
                    "read_label": record["label"],
                    "rows": record["rows"],
                    "lines": record["lines"],
                    "populated_row_score": row_population_score(evidence),
                    "text": record["text"],
                    "capture_replay": record["capture_replay"],
                    "tsv": record.get("tsv"),
                    "tokens": record.get("tokens"),
                    "locale": record.get("locale"),
                    "literal_columns": record.get("literal_columns"),
                    "row_candidates": [
                        serialize_row_candidate(row, confidence=resolved_confidence, index=index)
                        for index, row in enumerate(evidence.rows)
                    ],
                }
                all_reads.append(read)
        call = {
            "call_index": len(calls),
            "confidence_floor": resolved_confidence,
            "reads": all_reads,
        }
        result = original_build_consensus(readings, **kwargs)
        call["result_rows"] = [
            serialize_row_candidate(row, confidence=resolved_confidence, index=index)
            for index, row in enumerate(result.rows)
        ]
        call["result_issue_strings"] = list(result.issues)
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
    original_from_tokens = from_tokens
    original_build_consensus = ocr_recovery.build_consensus
    started = time.perf_counter()
    with patch.object(ocr_recovery, "transform", record_transform), patch.object(
        ocr_recovery, "from_tsv", record_from_tsv
    ), patch.object(ocr_recovery, "from_tokens", record_from_tokens), patch.object(
        ocr_recovery, "build_consensus", record_build_consensus
    ):
        evidence = main.recover_image(main.open_image(str(path)), reader)
    elapsed = time.perf_counter() - started
    payload = reconcile(evidence, [{}]).payload

    for call in calls:
        anchor = max(
            call["reads"],
            key=lambda read: (read["populated_row_score"],
                              read["row_candidates"] and 1 or 0),
        )
        call["anchor_read_key"] = anchor["read_key"]

    return {
        "seconds": round(elapsed, 1),
        "tesseract_calls": len(funnel),
        "read_funnel": funnel,
        "consensus_calls": calls,
        "payload": json_safe(payload),
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
    report = {
        "purpose": "PR84 token-level six-layout OCR-only capture; production decisions untouched, model never called",
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
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"documents": {}}
        )
        expected_copy = manifest.get("documents", {}).get(slug, {}).get("copy_sha256")
        actual_copy = sha256_file(image_path)
        if expected_copy and expected_copy != actual_copy:
            raise SystemExit(f"fixture {image_path.name} no longer matches the frozen manifest")
        result = trace(image_path, env["reader_languages"], OCR_CONFIG.confidence)
        result.update({
            "document": label,
            "layout_family": truth.get("layout_family"),
            "truth_file": truth_name,
            "truth_sha256": sha256_file(truth_path),
            "copy_sha256": actual_copy,
            "pixel_fingerprint": pixel_fingerprint(main.open_image(str(image_path))),
            "reference": json_safe(reference),
        })
        report["documents"].append(result)
        reads = [read for call in result["consensus_calls"] for read in call["reads"]]
        print(
            f"{label} ({slug}): reads={len(reads)} "
            f"rows_in_reads={sum(read['rows'] for read in reads)} "
            f"capture_replay_ok={all(read['capture_replay']['rows_match'] for read in reads)} "
            f"calls={result['tesseract_calls']} | {result['seconds']}s"
        )

    report["totals"] = {
        "documents": len(report["documents"]),
        "reads": sum(len(call["reads"]) for d in report["documents"] for call in d["consensus_calls"]),
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
