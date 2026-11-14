"""PR105B Phase 1: freeze the production acquisition path and report the authoritative baseline.

The live production pipeline runs once over the eleven development documents while every Tesseract call is
recorded. The frozen evidence is then replayed through the same production entry point and the payloads
must match byte-for-byte, which is what makes the four-configuration comparison in
`scripts/pr105b_factorial.py` a measurement of the consensus rule rather than of OCR luck.

Writes only inside results/product-review105b/ and refuses to overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import evaluation_metrics as metrics  # noqa: E402
import pr105b_harness as H  # noqa: E402
from dual_ocr_experiment import enabled as dual_ocr_enabled  # noqa: E402
from ocr_config import OCR_CONFIG  # noqa: E402


def guard(directory: Path, target: Path) -> Path:
    resolved = target.resolve()
    if directory.resolve() not in resolved.parents:
        raise SystemExit(f"artifact path outside the PR105B directory: {target}")
    if resolved.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {target}")
    return resolved


def as_list(value):
    """JSON and dataclasses disagree on tuple vs list, which is not a configuration difference."""
    return list(value) if isinstance(value, (tuple, list)) else value


def integrity(source: str) -> dict:
    manifest = H.read_json(H.ARTIFACT_DIR.parent / "product-review103" / "pr103-implementation-manifest.json")
    recorded = manifest["source_hashes"].get("src/ocr_consensus.py")
    consensus_sha256 = H.sha256_file(H.PRODUCTION_SOURCE_PATH)
    return {
        "pr103_sources_hashed": len(manifest["source_hashes"]),
        "pr103_source_drift": [
            name for name, value in manifest["source_hashes"].items()
            if (ROOT / name).exists()
            and H.sha256_file(ROOT / name) != (value if isinstance(value, str) else value["sha256"])
        ],
        "pr103_fingerprint": manifest["ocr_configuration_fingerprint"],
        "ocr_configuration_now": {
            name: as_list(getattr(OCR_CONFIG, name)) for name in manifest["ocr_configuration"]
        },
        "ocr_configuration_matches_pr103": {
            name: as_list(getattr(OCR_CONFIG, name)) for name in manifest["ocr_configuration"]
        } == {name: as_list(value) for name, value in manifest["ocr_configuration"].items()},
        "ocr_configuration_differences": [
            name for name, value in manifest["ocr_configuration"].items()
            if as_list(getattr(OCR_CONFIG, name)) != as_list(value)
        ],
        "ocr_consensus_sha256": consensus_sha256,
        "ocr_consensus_matches_pr103": consensus_sha256
                                        == (recorded if isinstance(recorded, str) else recorded["sha256"]),
        "ocr_consensus_bytes": H.PRODUCTION_SOURCE_PATH.stat().st_size,
        "consensus_source_lines": len(source.splitlines()),
        "environment_flags": {key: value for key, value in os.environ.items()
                             if key.startswith(("INVOICE_", "OCR_", "RAPIDOCR_"))},
        "dual_ocr_experiment_enabled": dual_ocr_enabled(),
    }


def save(cache_path: Path, freeze: H.Freeze, records: dict) -> None:
    cache_path.write_text(json.dumps({
        "version": 2,
        "entries": freeze.entries,
        "documents": {name: {key: record[key] for key in
                            ("payload_sha256", "payload", "ocr_call_keys", "consensus_calls",
                             "exported_rows", "nonempty_descriptions", "technical_error", "source_kind")}
                      for name, record in records.items()},
    }, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-cache", action="store_true", help="skip the live Tesseract pass")
    parser.add_argument("--output-name", default="pr105b-baseline-parity.json")
    args = parser.parse_args()

    if dual_ocr_enabled():
        raise SystemExit("the experimental dual OCR engine is active; PR105B measures Tesseract consensus only")

    H.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = H.ARTIFACT_DIR / H.CACHE_NAME
    target = guard(H.ARTIFACT_DIR, H.ARTIFACT_DIR / args.output_name)
    documents, source = H.corpus(), H.PRODUCTION_SOURCE_PATH.read_text(encoding="utf-8")

    if args.reuse_cache:
        cache = H.read_json(cache_path)
        freeze = H.Freeze("capture", dict(cache["entries"]))
        live = {name: dict(cache["documents"][name]) for name in documents}
        live_calls = 0
    else:
        freeze = H.Freeze("capture")
        live = {}
        started = time.perf_counter()
        for name, entry in documents.items():
            live[name] = H.run_document(name, entry, freeze)
            save(cache_path, freeze, {**live})
        live_calls = freeze.live_calls
        print(f"captured {live_calls} live Tesseract calls in {time.perf_counter() - started:.1f}s")

    replay = H.Freeze("reference", dict(freeze.entries))
    frozen = {name: H.run_document(name, entry, replay) for name, entry in documents.items()}

    mismatches = [
        {"document": name,
         "live_payload_sha256": live[name]["payload_sha256"],
         "frozen_payload_sha256": frozen[name]["payload_sha256"]}
        for name in documents
        if live[name]["payload_sha256"] != frozen[name]["payload_sha256"]
        or live[name]["ocr_call_keys"] != frozen[name]["ocr_call_keys"]
    ]
    if mismatches:
        print("PARITY FAILED", json.dumps(mismatches, ensure_ascii=False, indent=2))
        return 1

    scored = {name: metrics.evaluate(frozen[name]["payload"], documents[name]["reference"])
              for name in documents}
    aggregate = metrics.aggregate(list(scored.values()))
    print(f"parity ok: {len(documents)} documents, "
          f"{sum(len(record['ocr_call_keys']) for record in frozen.values())} replayed calls, 0 live")
    for name in documents:
        result = scored[name]
        print(f" {name:28} rows {result['rows']['correctly_matched']:2}/{result['rows']['expected']:2} "
              f"descriptions {result['fields']['by_field']['description']['correct']:2} correct "
              f"{result['fields']['by_field']['description']['missing']:2} missing | "
              f"{result['fields']['correct']:3} cells correct, {result['fields']['incorrect']} wrong, "
              f"{result['fields']['missing']} missing")
    print("aggregate:", json.dumps({key: aggregate["fields"][key] for key in
                                    ("correct", "incorrect", "missing", "incorrect_association")}))

    artifact = {
        "phase": "PR105B Phase 1 - frozen acquisition path, harness parity, production baseline",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "integrity": integrity(source),
        "corpus": {
            name: {key: entry.get(key) for key in ("kind", "reference_source", "reference_mapping",
                                                   "image_sha256", "frozen_image_sha256")}
            for name, entry in documents.items()
        },
        "ocr_freeze": {
            "cache_file": str(cache_path.relative_to(ROOT)),
            "live_calls_captured": live_calls,
            "unique_cached_calls": len(freeze.entries),
            "replayed_calls": sum(len(record["ocr_call_keys"]) for record in frozen.values()),
            "live_tesseract_calls_in_replay": replay.live_calls,
            "per_document_calls": {name: len(record["ocr_call_keys"]) for name, record in frozen.items()},
            "per_document_consensus_calls": {
                name: {"calls": record["consensus_calls"],
                       "authoritative_index": record["consensus_calls"] - 1}
                for name, record in frozen.items()},
        },
        "parity": {
            "documents": len(documents),
            "payload_mismatches": 0,
            "call_sequence_mismatches": 0,
            "payload_sha256": {name: frozen[name]["payload_sha256"] for name in documents},
        },
        "harness": {
            "description_block_occurrences": source.count(H.DESCRIPTION_BLOCK),
            "production_source_matches_pr103": integrity(source)["ocr_consensus_matches_pr103"],
            "transformed_added_lines": len(H.transform(source).splitlines()) - len(source.splitlines()),
            "configurations": {key: list(value) for key, value in H.CONFIGS.items()},
        },
        "baseline": {
            "per_document": {
                name: {key: frozen[name][key] for key in
                       ("technical_error", "source_kind", "exported_rows", "nonempty_descriptions",
                        "payload_sha256")}
                | {"evaluation": scored[name]}
                for name in documents
            },
            "aggregate": aggregate,
        },
        "privacy": "the cache holds raw Tesseract output of real scans and stays in the git-ignored "
                   "artifact directory; the per-document evaluation objects carry counts and verdicts only",
    }
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
