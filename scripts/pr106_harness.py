"""PR106 shared harness: structural line-item coverage on the three zero-row development documents.

Phase 1 runs the authoritative production pipeline live on `us-microspective`, `layout-a` and `layout-f`,
records the exact exported JSON and freezes the Tesseract bytes those three documents need, so every later
phase measures row-building rules against the same evidence rather than against OCR luck.

Isolation rules, inherited from PR105B and not re-litigated here:

* **One evidence policy.** `ocr_recovery.recover_image` feeds intermediate consensus results back into its
  own decisions, so a stage-5 rule could otherwise win by reading *more of the page*. Every OCR call is
  replayed positionally against the recorded production sequence for that document; bytes the document
  never recorded are answered with a no-text reading and counted in `suppressed_calls`.
* **The production payload is the baseline.** Phase 1 compares the live payload digests against the
  digests PR105B recorded for the same documents through the same entry point, so a candidate starts from
  a reproduction of the published baseline instead of a re-derivation of it.
* **Sensitive bytes stay in the git-ignored artifact directory.** Nothing written here leaves
  `results/product-review106/`.

No production file is modified by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr105b_harness as H  # noqa: E402

ARTIFACT_DIR = ROOT / "results" / "product-review106"
CACHE_NAME = "pr106-ocr-cache.json"
PR103_MANIFEST = ROOT / "results" / "product-review103" / "pr103-implementation-manifest.json"

# Phase 1 targets: the documents the PR105C census found with zero reconstructed stage-5 rows.
TARGETS = ("us-microspective", "layout-a-sumup-avoir-anon", "layout-f-photographer-anon")

ROW_FIELDS = H.ROW_FIELDS


def guard(directory: Path, target: Path) -> Path:
    resolved = target.resolve()
    if directory.resolve() not in resolved.parents:
        raise SystemExit(f"artifact path outside the PR106 directory: {target}")
    if resolved.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {target}")
    return resolved


def write_json(target: Path, value) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def documents() -> dict[str, dict]:
    return H.corpus()


# ----------------------------------------------------------------------------- integrity


def source_hashes() -> dict[str, dict]:
    """Every file that can change what the pipeline extracts, as PR103 hashed them."""
    import hashlib

    out = {}
    for pattern in ("src/*.py", "app/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            out[path.relative_to(ROOT).as_posix()] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    return out


def display(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT.resolve()) else str(path)


def pr103_freeze(hashes: dict[str, dict], manifest: Path | None = None) -> dict:
    """Drift between the working tree and an explicitly selected source-version manifest.

    With no `manifest` the comparison is exactly what it has always been - PR103's published freeze - so a
    research run that names nothing still measures itself against the historical baseline. A gate that
    benchmarks a *later* source version declares which manifest it means, and the block then names and
    digests the file it actually read; the `pr103_*` key names are kept so every consumer that reads them
    keeps working, and they now mean "the selected manifest" rather than "PR103 by assumption".
    """
    path = Path(manifest) if manifest else PR103_MANIFEST
    manifest_bytes = path.read_bytes()
    recorded = json.loads(manifest_bytes.decode("utf-8"))["source_hashes"]

    def digest(value):
        return value if isinstance(value, str) else value["sha256"]

    drift = [name for name, value in recorded.items()
             if name in hashes and hashes[name]["sha256"] != digest(value)]
    added = [name for name in hashes if name not in recorded]
    removed = [name for name in recorded if name not in hashes]
    return {
        "manifest": display(path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_defaulted_to_pr103": manifest is None,
        "pr103_sources_hashed": len(recorded),
        "sources_hashed_now": len(hashes),
        "drifted": sorted(drift),
        "added_since_pr103": sorted(added),
        "removed_since_pr103": sorted(removed),
        "unchanged": len(hashes) - len(drift) - len(added),
    }


def environment() -> dict:
    """Hardware and flag state, measured rather than assumed (PR106 Phase 1 items 4-6)."""
    import dual_ocr_experiment as experiment
    from ocr_config import OCR_CONFIG

    flags = {
        "EXPERIMENTAL_DUAL_OCR": os.environ.get("EXPERIMENTAL_DUAL_OCR"),
        "INVOICE_TABLE_SHADOW": os.environ.get("INVOICE_TABLE_SHADOW"),
        "INVOICE_TABLE_GEOMETRY": os.environ.get("INVOICE_TABLE_GEOMETRY"),
        "INVOICE_REVIEW_FAST_PATH": os.environ.get("INVOICE_REVIEW_FAST_PATH"),
        "INVOICE_MODULE_PRELOAD": os.environ.get("INVOICE_MODULE_PRELOAD"),
    }
    info: dict = {
        "process_flags": flags,
        "windows_scopes": _windows_scopes(),
        "dual_ocr_enabled": experiment.enabled(),
        "dual_ocr_strategy": experiment.strategy(),
        "ocr_configuration": {name: list(getattr(OCR_CONFIG, name))
                              for name in ("families", "psm_sequence", "region_modes")
                              if hasattr(OCR_CONFIG, name)},
        "confidence_threshold": OCR_CONFIG.confidence,
        "language_data": os.environ.get("TESSDATA_PREFIX"),
    }
    try:
        import torch

        info["torch"] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    except Exception as error:  # pragma: no cover - environment report
        info["torch"] = {"error": f"{type(error).__name__}: {error}"}
    for module in ("bitsandbytes", "rapidocr_onnxruntime", "onnxruntime"):
        try:
            __import__(module)
            info[f"import_{module}"] = "available"
        except Exception as error:
            info[f"import_{module}"] = f"{type(error).__name__}"
    info["memory"] = _memory()
    info["full_precision_qwen_on_cpu"] = _qwen_feasibility(info)
    return info


def _windows_scopes() -> dict:
    if os.name != "nt":
        return {"scope": "not windows"}
    import winreg

    found = {}
    for hive, name, label in (
        (winreg.HKEY_CURRENT_USER, "Environment", "user"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", "machine"),
    ):
        try:
            key = winreg.OpenKey(hive, name)
            values = {winreg.EnumValue(key, index)[0]: winreg.EnumValue(key, index)[1]
                      for index in range(winreg.QueryInfoKey(key)[1])}
        except OSError:
            values = {}
        found[label] = {flag: values[flag] for flag in values
                        if "EXPERIMENTAL" in flag.upper() or flag.upper().startswith("INVOICE_")}
    return found


def _memory() -> dict:
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"error": "GlobalMemoryStatusEx failed"}
    gib = 1024 ** 3
    return {"total_gb": round(status.ullTotalPhys / gib, 1),
            "available_gb": round(status.ullAvailPhys / gib, 1),
            "memory_load_percent": status.dwMemoryLoad}


def _qwen_feasibility(info: dict) -> dict:
    """7B parameters in float32 needs ~28 GB of weights before any activation or KV cache."""
    weights_gb_float32 = round(7.0 * 1e9 * 4 / 1024 ** 3, 1)
    available = info.get("memory", {}).get("available_gb", 0)
    cuda = info.get("torch", {}).get("cuda_available")
    return {
        "parameters_billions_assumed": 7,
        "float32_weight_gb": weights_gb_float32,
        "cuda_available": cuda,
        "available_ram_gb": available,
        "verdict": ("IMPOSSIBLE: no CUDA device and available RAM is a small fraction of the float32 "
                    "weight size" if not cuda and available < weights_gb_float32 else "not excluded"),
    }


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "head": run("rev-parse", "HEAD"),
        "modified_tracked": [line[3:] for line in status.splitlines() if line[:2] not in ("??",)],
        "untracked": [line[3:] for line in status.splitlines() if line.startswith("?? ")],
    }


# ----------------------------------------------------------------------------- phase 1


def main() -> int:
    import evaluation_metrics as metrics
    from pr105b_capture import integrity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-cache", action="store_true",
                       help="skip the live Tesseract pass and replay the PR106 cache")
    parser.add_argument("--output-name", default="pr106-phase1-baseline.json")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name)
    cache_path = ARTIFACT_DIR / CACHE_NAME

    corpus = documents()
    targets = {name: corpus[name] for name in TARGETS}
    production_source = H.PRODUCTION_SOURCE_PATH.read_text(encoding="utf-8")
    hashes = source_hashes()

    # --- live production pass over the three targets, recording the OCR bytes
    if args.reuse_cache:
        cache = H.read_json(cache_path)
        freeze = H.Freeze("capture", dict(cache["entries"]))
        live = {name: dict(cache["documents"][name]) for name in targets}
        live_calls, elapsed = 0, 0.0
    else:
        freeze = H.Freeze("capture")
        started = time.perf_counter()
        live, partial = {}, {}
        for name, entry in targets.items():
            record = H.run_document(name, entry, freeze)
            partial[name] = {key: record[key] for key in
                             ("payload_sha256", "payload", "ocr_call_keys", "consensus_calls",
                              "exported_rows", "nonempty_descriptions", "technical_error", "source_kind")}
            cache_path.write_text(json.dumps(
                {"version": 2, "entries": freeze.entries, "documents": partial},
                ensure_ascii=False), encoding="utf-8")
            live[name] = record
            print(f"  captured {name}: {len(record['ocr_call_keys'])} OCR calls, "
                  f"{record['exported_rows']} exported rows")
        elapsed = time.perf_counter() - started
        live_calls = freeze.live_calls
        print(f"live pass: {live_calls} Tesseract calls in {elapsed:.1f}s")

    # --- replay parity: the frozen bytes must reproduce the same payloads
    replay = H.Freeze("reference", dict(freeze.entries))
    frozen = {name: H.run_document(name, entry, replay) for name, entry in targets.items()}
    mismatches = [
        {"document": name,
         "live_payload_sha256": live[name]["payload_sha256"],
         "frozen_payload_sha256": frozen[name]["payload_sha256"]}
        for name in targets
        if live[name]["payload_sha256"] != frozen[name]["payload_sha256"]
        or live[name]["ocr_call_keys"] != frozen[name]["ocr_call_keys"]
    ]

    # --- agreement with the published PR105B baseline for the same documents
    pr105b = H.read_json(ARTIFACT_DIR.parent / "product-review105b" / "pr105b-baseline-parity.json")
    recorded = pr105b["parity"]["payload_sha256"]
    against_pr105b = {
        name: {"pr105b_payload_sha256": recorded.get(name),
               "pr106_payload_sha256": frozen[name]["payload_sha256"],
               "identical": recorded.get(name) == frozen[name]["payload_sha256"]}
        for name in targets
    }

    scored = {name: metrics.evaluate(frozen[name]["payload"], targets[name]["reference"])
              for name in targets}
    write_json(ARTIFACT_DIR / "pr106-target-json.json", {
        "note": "exact exported JSON of the three PR106 targets through app.main.document_text; "
                "contains real invoice text and stays in the git-ignored artifact directory",
        "documents": {name: {"payload": frozen[name]["payload"],
                             "payload_sha256": frozen[name]["payload_sha256"],
                             "technical_error": frozen[name]["technical_error"],
                             "source_kind": frozen[name]["source_kind"],
                             "reference": targets[name]["reference"]}
                      for name in targets},
    })

    integrity_block = integrity(production_source)
    integrity_block["verdict"] = (
        "FROZEN" if not integrity_block["pr103_source_drift"]
        and integrity_block["ocr_configuration_matches_pr103"]
        and integrity_block["ocr_consensus_matches_pr103"]
        and not integrity_block["dual_ocr_experiment_enabled"]
        else "NOT FROZEN"
    )

    artifact = {
        "phase": "PR106 Phase 1 - reproduction of the three zero-row failures and evidence freeze",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission": ("investigate invoices whose legitimate line items are rejected before export; "
                    "targets us-microspective, layout-a, layout-f; they are not assumed to fail alike"),
        "git": git_state(),
        "environment": environment(),
        "integrity": integrity_block,
        "pr103_freeze": pr103_freeze(hashes),
        "source_hashes": hashes,
        "targets": {
            name: {
                "kind": entry["kind"],
                "image": entry["path"].name,
                "image_sha256": entry.get("image_sha256") or H.sha256_file(entry["path"]),
                "reference_source": entry["reference_source"],
                "expected_rows": len(entry["reference"]["line_items"] or []),
                "expected_row_fields": [
                    {field: row.get(field) for field in ROW_FIELDS}
                    for row in entry["reference"]["line_items"] or []
                ],
                "exported_rows": frozen[name]["exported_rows"],
                "nonempty_descriptions": frozen[name]["nonempty_descriptions"],
                "payload_sha256": frozen[name]["payload_sha256"],
                "technical_error": frozen[name]["technical_error"],
                "source_kind": frozen[name]["source_kind"],
                "ocr_calls": len(frozen[name]["ocr_call_keys"]),
                "consensus_calls": frozen[name]["consensus_calls"],
                "failure_reproduced": frozen[name]["exported_rows"]
                                      < len(entry["reference"]["line_items"] or []),
                "evaluation_counts": {
                    "rows": scored[name]["rows"],
                    "fields": scored[name]["fields"],
                    "header": scored[name]["header_summary"],
                },
            } for name, entry in targets.items()
        },
        "ocr_freeze": {
            "cache_file": str(cache_path.relative_to(ROOT)),
            "live_calls_captured": live_calls,
            "live_seconds": round(elapsed, 1),
            "unique_cached_calls": len(freeze.entries),
            "replayed_calls": sum(len(record["ocr_call_keys"]) for record in frozen.values()),
            "live_tesseract_calls_in_replay": replay.live_calls,
            "per_document_calls": {name: len(frozen[name]["ocr_call_keys"]) for name in targets},
        },
        "parity": {
            "payload_mismatches": mismatches,
            "live_vs_frozen": "IDENTICAL" if not mismatches else "DIVERGED",
            "against_published_pr105b_baseline": against_pr105b,
            "agrees_with_pr105b": all(value["identical"] for value in against_pr105b.values()),
        },
        "not_executed": [
            "Phase 2 stage traces (next: scripts/pr106_traces.py)",
            "pytest suites (no source file has been modified in Phase 1)",
            "model inference of any kind (this phase is OCR-only)",
        ],
        "privacy": "the cache and pr106-target-json.json hold raw Tesseract output and exported text of "
                   "real invoices; they stay inside the git-ignored results/product-review106/ directory",
    }
    artifact["verdict"] = (
        "BASELINE REPRODUCED" if not mismatches and artifact["parity"]["agrees_with_pr105b"]
        and all(artifact["targets"][name]["failure_reproduced"] for name in targets)
        else "CHECK REQUIRED"
    )
    write_json(target, artifact)

    print(f"parity {'ok' if not mismatches else 'FAILED'} | "
          f"agrees with PR105B {artifact['parity']['agrees_with_pr105b']} | "
          f"PR103 freeze {artifact['pr103_freeze']['drifted'] or 'no drift'} | "
          f"experimental OCR on {artifact['environment']['dual_ocr_enabled']}")
    for name in targets:
        result = scored[name]
        print(f" {name:28} rows {result['rows']['correctly_matched']}/"
              f"{result['rows']['expected']} expected | exported {frozen[name]['exported_rows']} | "
              f"fields {result['fields']['correct']} correct, {result['fields']['incorrect']} wrong, "
              f"{result['fields']['missing']} missing")
    print("verdict", artifact["verdict"])
    print("wrote", target)
    return 0 if artifact["verdict"] == "BASELINE REPRODUCED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
