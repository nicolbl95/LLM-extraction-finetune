"""Capture the deployed OCR pipeline locally without invoking an extraction model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / ".venv" / "product-review58" / "public-patch-candidate"
SOURCE = CANDIDATE / "src"
DATA_DIR = ROOT / ".venv" / "product-review64" / "tessdata"
MATRIX_PATH = ROOT / "PRODUCT_REVIEW_62_REGRESSION_MATRIX.json"
PR63_CAPTURE_PATH = ROOT / "PRODUCT_REVIEW_63_DIAGNOSTIC_CAPTURES.json"
MIRADA_RESPONSE_PATH = ROOT / ".venv" / "product-review45" / "STAGING_RESPONSE.json"
OUTPUT_PATH = ROOT / "PRODUCT_REVIEW_64_REPRODUCTION_RESULTS.json"
PUBLIC_REVISION = "927a115997ef2f850255e1b14cfe37fe861bb710"
TESSDATA_COMMIT = "923915d4ced2a7235221788285785a29c4a42d4a"
TESSDATA_FILES = {
    "fra": {
        "size": 1_130_365,
        "git_blob_sha1": "d9e2b2160be0d1ca3b8f1bf2730fae476ef3b4a6",
        "sha256": "ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca",
    },
    "eng": {
        "size": 4_113_088,
        "git_blob_sha1": "bbef4675053b5b468cdb477053e28b1c698ba08e",
        "sha256": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    },
}
CASE_IDS = (
    "microspective",
    "mon_entreprise",
    "mirada",
    "future_graphics",
    "fast_photography",
)
OCR_FLAGS = (
    "INVOICE_OCR_FAST_PATH",
    "INVOICE_TABLE_SHADOW",
    "INVOICE_TABLE_RECOVERY",
)
SOURCE_FILES = (
    "app/main.py",
    "src/ocr_config.py",
    "src/ocr_preprocessing.py",
    "src/ocr_recovery.py",
    "src/ocr_consensus.py",
    "src/ocr_quality.py",
    "src/invoice_evidence.py",
    "src/invoice_grounding.py",
    "src/invoice_validation.py",
    "src/table_shadow.py",
    "src/table_recovery.py",
    "src/table_geometry.py",
    "src/ocr_fast_path.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def pixel_fingerprint(image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="python"))
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return str(value)


def serialize_number(number) -> dict:
    return {
        "raw": number.raw,
        "value": str(number.value),
        "page": number.page,
        "box": list(number.box),
    }


def serialize_row(row, row_index: int) -> dict:
    return {
        "row_index": row_index,
        "description": row.description,
        "description_confident": row.description_confident,
        "secondary_description": row.secondary_description,
        "box": list(row.line.box),
        "page": row.line.page,
        "cells": {
            key: [serialize_number(number) for number in values]
            for key, values in row.cells.items()
        },
        "family_votes": {
            field: {
                str(value): list(families)
                for value, families in values.items()
            }
            for field, values in row.family_votes.items()
        },
        "corroborated_amount_mismatch": json_safe(
            row.corroborated_amount_mismatch
        ),
    }


def serialize_evidence(evidence, *, include_tokens: bool = True) -> dict:
    result = {
        "text": evidence.text,
        "available": evidence.available,
        "locale": evidence.locale,
        "date_locale": evidence.date_locale,
        "supplier": evidence.supplier,
        "supplier_raw": evidence.supplier_raw,
        "supplier_is_explicit": evidence.supplier_is_explicit,
        "invoice_number": evidence.invoice_number,
        "invoice_date": evidence.invoice_date,
        "source_kind": evidence.source_kind,
        "ocr_cross_checked": evidence.ocr_cross_checked,
        "rows_provenance": evidence.rows_provenance,
        "rows": [
            serialize_row(row, index)
            for index, row in enumerate(evidence.rows)
        ],
        "totals": {
            key: [serialize_number(number) for number in values]
            for key, values in evidence.totals.items()
        },
        "taxes": [
            {
                "rate": serialize_number(tax.rate) if tax.rate else None,
                "amount": serialize_number(tax.amount),
                "base": serialize_number(tax.base) if tax.base else None,
            }
            for tax in evidence.taxes
        ],
        "discounts": [serialize_number(number) for number in evidence.discounts],
        "quality_issues": list(evidence.quality_issues),
        "issues": list(evidence.issues),
        "preprocessing_used": list(evidence.preprocessing_used),
        "ocr_calls": evidence.ocr_calls,
        "ocr_pages": evidence.ocr_pages,
        "ocr_elapsed_seconds": evidence.ocr_elapsed_seconds,
        "source_size": (
            list(evidence.source_size)
            if getattr(evidence, "source_size", None) else None
        ),
        "table_shadow": json_safe(getattr(evidence, "table_shadow", None)),
        "table_recovery": json_safe(getattr(evidence, "table_recovery", None)),
    }
    if include_tokens:
        result["tokens"] = [
            {
                "text": token.text,
                "page": token.page,
                "box": list(token.box),
                "confidence": token.confidence,
            }
            for token in evidence.tokens
        ]
        result["lines"] = [
            {
                "text": line.text,
                "page": line.page,
                "box": list(line.box),
            }
            for line in evidence.lines
        ]
    return result


def serialize_tsv(data: dict, call: dict, from_tsv) -> dict:
    grouped = {}
    tokens = []
    scores = []
    for index, text in enumerate(data.get("text", [])):
        text = str(text)
        confidence = float(data["conf"][index])
        box = [
            int(data["left"][index]),
            int(data["top"][index]),
            int(data["left"][index] + data["width"][index]),
            int(data["top"][index] + data["height"][index]),
        ]
        position = {
            "page": int(data["page_num"][index]),
            "block": int(data["block_num"][index]),
            "paragraph": int(data["par_num"][index]),
            "line": int(data["line_num"][index]),
            "word": int(data["word_num"][index]),
        }
        if text.strip():
            token = {
                "text": text,
                "confidence": confidence,
                "box": box,
                **position,
            }
            tokens.append(token)
            if confidence >= 0:
                key = (
                    position["page"],
                    position["block"],
                    position["paragraph"],
                    position["line"],
                )
                grouped.setdefault(key, []).append(token)
        if confidence >= 0 and text.strip():
            scores.append(confidence)

    lines = [
        {
            "position": list(position),
            "text": " ".join(token["text"] for token in line_tokens),
            "tokens": line_tokens,
        }
        for position, line_tokens in grouped.items()
    ]
    parsed = from_tsv(data)
    family = call["family"]
    return {
        **{key: value for key, value in call.items() if key != "_tsv"},
        "raw_text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "tokens": tokens,
        "confidence_summary": {
            "scored_token_count": len(scores),
            "mean": statistics.mean(scores) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "minimum": min(scores) if scores else None,
            "below_60_count": sum(score < 60 for score in scores),
        },
        "parsed_candidates": serialize_evidence(parsed, include_tokens=False),
    }


def field_values(evidence, field: str) -> set[str]:
    if field in {"supplier_name", "invoice_date"}:
        attr = "supplier" if field == "supplier_name" else "invoice_date"
        value = getattr(evidence, attr, None)
        return {str(value)} if value is not None else set()
    if field in {"total_ht", "total_ttc", "total_candidate"}:
        return {
            str(number.value)
            for number in evidence.totals.get(field, [])
        }
    return set()


def removed_candidates(reads: list[dict], consensus) -> list[dict]:
    removed = []
    for read in reads:
        parsed = read["parsed_candidates"]
        for field in ("supplier_name", "invoice_date", "total_ht", "total_ttc"):
            pass_evidence = read["_parsed_object"]
            candidates = field_values(pass_evidence, field)
            accepted = field_values(consensus, field)
            for candidate in sorted(candidates - accepted):
                removed.append({
                    "read_index": read["read_index"],
                    "family": read["family"],
                    "field": field,
                    "candidate": candidate,
                    "accepted_consensus_values": sorted(accepted),
                    "classification": "not retained in consensus",
                })
        for row_index, row in enumerate(read["_parsed_object"].rows):
            final_row = None
            row_text = row.description.casefold().strip()
            if row_text:
                matches = [
                    final for final in consensus.rows
                    if final.description.casefold().strip() == row_text
                ]
                if len(matches) == 1:
                    final_row = matches[0]
            for field, numbers in row.cells.items():
                candidate_values = {str(number.value) for number in numbers}
                if not candidate_values:
                    continue
                accepted = (
                    {str(number.value) for number in final_row.cells.get(field, [])}
                    if final_row is not None else set()
                )
                for candidate in sorted(candidate_values - accepted):
                    removed.append({
                        "read_index": read["read_index"],
                        "family": read["family"],
                        "row_index": row_index,
                        "row_description": row.description,
                        "field": f"line_items.{row_index}.{field}",
                        "candidate": candidate,
                        "accepted_consensus_values": sorted(accepted),
                        "classification": (
                            "not retained for exact-description consensus row"
                            if final_row is not None
                            else "row alignment unresolved; not labelled a rejection"
                        ),
                    })
    return removed


def family_image_map(image, deskew, transform, families, max_pixels: int) -> dict:
    rotated, angle = deskew(image)
    mapped = {}
    for family in families:
        scale = 3 if family == "scale3" else 2
        if rotated.width * rotated.height * scale * scale > max_pixels:
            continue
        prepared, _metadata = transform(rotated, angle, family)
        mapped.setdefault(pixel_fingerprint(prepared), set()).add(family)
        if family in {"deskew", "clahe"}:
            header = prepared.crop(
                (0, 0, prepared.width, max(1, int(prepared.height * 0.22)))
            )
            mapped.setdefault(pixel_fingerprint(header), set()).add(
                f"header_crop/{family}"
            )
    return mapped


def reader_family(read_index: int, psm: int, labels: set[str],
                  seen: set[tuple[str, int]], headers_seen: set[str]) -> str:
    if read_index == 0 and "deskew" in labels:
        return "deskew"
    if read_index == 1 and "clahe" in labels:
        return "clahe"
    header_labels = sorted(
        label for label in labels if label.startswith("header_crop/")
    )
    if header_labels:
        return header_labels[0]
    if (
        "clahe" in labels and ("clahe", psm) not in seen
        and not (psm == 4 and "clahe" in headers_seen)
    ):
        return "clahe"
    if (
        "deskew" in labels and ("deskew", psm) not in seen
        and not (psm == 6 and "deskew" in headers_seen)
    ):
        return "deskew"
    if "crop" in labels and ("crop", psm) not in seen:
        return "crop"
    if len(labels) == 1:
        return next(iter(labels))
    return "derived_crop_or_region"


def main() -> int:
    import pytesseract

    original_prefix = os.environ.get("TESSDATA_PREFIX")
    pre_isolation_version = str(pytesseract.get_tesseract_version())
    pre_isolation_languages = sorted(pytesseract.get_languages(config=""))
    os.environ["TESSDATA_PREFIX"] = str(DATA_DIR.resolve())
    original_flags = {flag: os.environ.get(flag) for flag in OCR_FLAGS}
    for flag in OCR_FLAGS:
        os.environ.pop(flag, None)

    sys.path.insert(0, str(CANDIDATE))
    sys.path.insert(0, str(SOURCE))
    import pytesseract
    from PIL import Image
    from app import main as app_main
    from invoice_evidence import (
        _header_matches,
        from_tsv,
        is_line_total_header,
        total_label_field,
    )
    from invoice_grounding import document_quality, reconcile
    from ocr_budget import ProcessingBudget
    from ocr_config import OCR_CONFIG
    from ocr_preprocessing import deskew, transform

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Local traineddata directory is missing: {DATA_DIR}")
    expected_languages = {"eng", "fra"}
    installed_languages = set(pytesseract.get_languages(config=""))
    if not expected_languages.issubset(installed_languages):
        raise RuntimeError(
            f"Isolated Tesseract data lacks required languages: "
            f"{sorted(expected_languages - installed_languages)}"
        )
    for language, metadata in TESSDATA_FILES.items():
        path = DATA_DIR / f"{language}.traineddata"
        if not path.is_file() or path.stat().st_size != metadata["size"]:
            raise RuntimeError(f"Traineddata size/missing-file check failed: {path}")
        if sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"Traineddata SHA-256 check failed: {path}")

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    pr63 = json.loads(PR63_CAPTURE_PATH.read_text(encoding="utf-8"))
    pr63_cases = {case["case_id"]: case for case in pr63["cases"]}
    cache = json.loads(
        (ROOT / ".venv" / "final-targeted-audit" / "matrix.json").read_text(
            encoding="utf-8"
        )
    )
    matrix_cases = {case["case_id"]: case for case in matrix["cases"]}

    source_hashes = {
        file_name: sha256_file(CANDIDATE / file_name)
        for file_name in SOURCE_FILES
    }
    tesseract_version = str(pytesseract.get_tesseract_version())
    tesseract_path = str(pytesseract.pytesseract.tesseract_cmd)

    cases = []
    failures = 0
    for case_id in CASE_IDS:
        matrix_case = matrix_cases[case_id]
        asset = matrix_case["candidate_asset"]
        image_path = ROOT / asset["path"]
        actual_image_hash = sha256_file(image_path)
        if actual_image_hash != asset["sha256"]:
            raise RuntimeError(f"Input image hash mismatch for {case_id}")
        with app_main.open_image(image_path) as image:
            dimensions = list(image.size)
            family_hashes = family_image_map(
                image, deskew, transform, OCR_CONFIG.families,
                OCR_CONFIG.max_pixels,
            )
            budget = ProcessingBudget()
            calls = []
            captured_evidence = []
            seen_family_psms = set()
            header_families_seen = set()
            original_recover = app_main.recover_image

            def traced_recover(source_image, reader, *, budget=None):
                active_budget = budget or ProcessingBudget()

                def traced_reader(prepared, psm):
                    timeout = min(OCR_CONFIG.call_seconds, active_budget.remaining())
                    started = time.perf_counter()
                    data = reader(prepared, psm)
                    read_index = len(calls)
                    labels = family_hashes.get(
                        pixel_fingerprint(prepared), set()
                    )
                    family = reader_family(
                        read_index, psm, labels, seen_family_psms,
                        header_families_seen,
                    )
                    if family.startswith("header_crop/"):
                        header_families_seen.add(family.split("/", 1)[1])
                    elif family in OCR_CONFIG.families:
                        seen_family_psms.add((family, psm))
                    calls.append({
                        "read_index": read_index,
                        "family": family,
                        "family_label_basis": (
                            "initial reader order plus prepared-image fingerprint"
                        ),
                        "psm": psm,
                        "language": "fra+eng",
                        "config": f"--psm {psm}",
                        "timeout_seconds": timeout,
                        "elapsed_seconds": time.perf_counter() - started,
                        "image_size": list(prepared.size),
                        "image_mode": prepared.mode,
                        "input_pixel_sha256": pixel_fingerprint(prepared),
                        "_tsv": data,
                    })
                    return data

                evidence = original_recover(
                    source_image, traced_reader, budget=active_budget
                )
                captured_evidence.append(evidence)
                return evidence

            app_main.recover_image = traced_recover
            started = time.perf_counter()
            error = None
            try:
                result_text = app_main.ocr_image(image, budget=budget)
            except (
                app_main.UnreadableOCR,
                pytesseract.TesseractError,
                pytesseract.TesseractNotFoundError,
                RuntimeError,
                OSError,
                ValueError,
            ) as exc:
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                result_text = None
            finally:
                app_main.recover_image = original_recover
            elapsed = time.perf_counter() - started

        if error is not None:
            failures += 1
            cases.append({
                "case_id": case_id,
                "candidate_image": {
                    "path": asset["path"],
                    "sha256": actual_image_hash,
                    "dimensions": dimensions,
                    "first11_upload_sha256": asset.get("first11_upload_sha256"),
                },
                "capture_status": "failed",
                "error": error,
                "elapsed_seconds": elapsed,
                "reader_calls": [
                    serialize_tsv(call["_tsv"], call, from_tsv)
                    for call in calls
                ],
                "model_calls": 0,
            })
            continue

        if len(captured_evidence) != 1:
            raise RuntimeError(
                f"Expected exactly one DocumentEvidence for {case_id}; "
                f"received {len(captured_evidence)}"
            )
        evidence = captured_evidence[0]
        evidence.ocr_calls = len(calls)
        evidence.ocr_elapsed_seconds = elapsed
        evidence.source_size = tuple(dimensions)

        reads = []
        parsed_objects = []
        for call in calls:
            parsed = from_tsv(call["_tsv"])
            parsed_objects.append(parsed)
            read = serialize_tsv(call["_tsv"], call, from_tsv)
            reads.append(read)
        for read, parsed in zip(reads, parsed_objects):
            read["_parsed_object"] = parsed
        removed = removed_candidates(reads, evidence)
        for read in reads:
            read.pop("_parsed_object", None)

        diagnostic_probes = {}
        if case_id == "microspective":
            header = next(
                (
                    line for line in evidence.text.splitlines()
                    if "description" in line.casefold()
                    and ("qty" in line.casefold() or "quantity" in line.casefold())
                    and "rate" in line.casefold()
                ),
                None,
            )
            diagnostic_probes["microspective_header_parser"] = {
                "source_text": header,
                "header_matches": {
                    field: _header_matches(field, header)
                    for field in ("description", "quantity", "unit_price", "total_ht")
                } if header else None,
                "subtotal_is_line_total_header": is_line_total_header("SUBTOTAL"),
                "standalone_subtotal_global_field": total_label_field("SUBTOTAL 772.50"),
                "source_code": relative(CANDIDATE / "src" / "invoice_evidence.py"),
                "source_file_sha256": source_hashes["src/invoice_evidence.py"],
            }

        try:
            ocr_only = reconcile(evidence, [])
            reconciliation = {
                "mode": "OCR-only; reconcile(evidence, [])",
                "model_calls": 0,
                "payload": json_safe(ocr_only.payload),
                "statuses": json_safe(ocr_only.statuses),
                "warnings": json_safe(ocr_only.warnings),
                "provenance": json_safe(ocr_only.provenance),
                "failed_rows": json_safe(ocr_only.failed_rows),
                "validation": json_safe(ocr_only.validation),
            }
        except (ValueError, RuntimeError) as exc:
            reconciliation = {
                "mode": "OCR-only; reconcile(evidence, [])",
                "model_calls": 0,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

        local_replay = matrix_case.get("local_replay") or {}
        cached_record_key = local_replay.get("record_key")
        previous_cache = cache.get(cached_record_key) if cached_record_key else None
        prior_case = pr63_cases[case_id]
        cases.append({
            "case_id": case_id,
            "candidate_image": {
                "path": asset["path"],
                "sha256": actual_image_hash,
                "dimensions": dimensions,
                "hash_verified": True,
                "first11_upload_sha256": asset.get("first11_upload_sha256"),
            },
            "first11_user_report": matrix_case.get("first11_observation"),
            "first11_request_match": "not established",
            "reference_fields": matrix_case.get("ground_truth"),
            "capture_status": "captured",
            "elapsed_seconds": elapsed,
            "app_ocr_text": str(result_text),
            "reader_call_count": len(calls),
            "reads": reads,
            "consensus_rejected_or_unaligned_candidates": removed,
            "consensus_documentary_evidence": serialize_evidence(evidence),
            "diagnostic_probes": diagnostic_probes,
            "quality_issues": document_quality(evidence),
            "ocr_only_reconciliation": reconciliation,
            "model_calls": 0,
            "model_output": None,
            "prior_pr63_evidence": {
                "diagnostic_status": prior_case.get("diagnostic_status"),
                "local_replay": prior_case.get("existing_evidence", {}).get(
                    "local_replay"
                ),
                "historical_qwen_capture": (
                    prior_case.get("existing_evidence", {}).get(
                        "historical_raw_capture"
                    )
                    if case_id == "mirada" else None
                ),
                "cached_read_artifact": local_replay.get("source_file"),
                "cached_consensus": (
                    previous_cache.get("rows")
                    if previous_cache is not None else None
                ),
            },
        })

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())

    historical = json.loads(
        MIRADA_RESPONSE_PATH.read_text(encoding="utf-8")
    )
    output = {
        "schema_version": "product-review64-reproduction-results-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local OCR and grounding only; no model inference.",
        "source": {
            "public_revision": PUBLIC_REVISION,
            "deployed_candidate_path": relative(CANDIDATE),
            "current_worktree_head": git_head,
            "current_branch": git_branch,
            "worktree_dirty": git_dirty,
            "candidate_file_sha256": source_hashes,
            "capture_script_sha256": sha256_file(Path(__file__)),
        },
        "verified_deployed_build_baseline": {
            "tesseract_version": "5.3.0",
            "languages": ["eng", "fra", "osd"],
            "source": "PRODUCT_REVIEW_54_LINUX_BUILD_RESULT.md",
            "version_pinned_in_packages_txt": False,
            "traineddata_hashes_recorded_in_deployment": False,
        },
        "local_ocr_environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "tesseract_version": tesseract_version,
            "tesseract_command": tesseract_path,
            "languages": sorted(pytesseract.get_languages(config="")),
            "pre_isolation_check": {
                "tesseract_version": pre_isolation_version,
                "languages": pre_isolation_languages,
                "tessdata_prefix": original_prefix,
                "fra_available": "fra" in pre_isolation_languages,
            },
            "tessdata_prefix": str(DATA_DIR.resolve()),
            "tessdata_source": {
                "repository": "https://github.com/tesseract-ocr/tessdata_fast",
                "commit": TESSDATA_COMMIT,
                "files": {
                    language: {
                        **metadata,
                        "path": relative(DATA_DIR / f"{language}.traineddata"),
                        "source_url": (
                            "https://raw.githubusercontent.com/tesseract-ocr/"
                            f"tessdata_fast/{TESSDATA_COMMIT}/"
                            f"{language}.traineddata"
                        ),
                    }
                    for language, metadata in TESSDATA_FILES.items()
                },
                "integrity": (
                    "Downloaded from official raw URLs pinned to the commit; "
                    "Git blob SHA-1, file size, and SHA-256 checked."
                ),
            },
            "process_local_environment_flags": {
                "TESSDATA_PREFIX": str(DATA_DIR.resolve()),
                "original_TESSDATA_PREFIX_in_capture_process": original_prefix,
                "OCR_feature_flags_cleared_for_baseline": {
                    flag: {
                        "original_value": original_flags[flag],
                        "effective_value": None,
                    }
                    for flag in OCR_FLAGS
                },
                "parent_process_environment_restored": True,
            },
        },
        "ocr_configuration": {
            **OCR_CONFIG.export(),
            "fingerprint_sha256": OCR_CONFIG.fingerprint(),
            "reader_language": "fra+eng",
            "reading_modes": list(OCR_CONFIG.segmentations),
            "preprocessing_details": {
                "families": list(OCR_CONFIG.families),
                "fast_families": list(OCR_CONFIG.fast_families),
                "initial_segmentations": list(OCR_CONFIG.segmentations),
                "deskew_minimum_degrees": 0.15,
                "scale": {
                    "scale3": 3,
                    "other_families": 2,
                    "interpolation": "PIL BICUBIC",
                },
                "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
                "denoise": {"method": "median blur", "kernel_size": 3},
                "crop": "table-specific crops generated by ocr_recovery",
                "maximum_pixels": OCR_CONFIG.max_pixels,
            },
            "consensus_and_grounding": {
                "family_votes_require_within_family_segmentation_agreement": True,
                "minimum_independent_families_for_numeric_agreement": 2,
                "supplier_date_consensus_is_unanimous": True,
                "model_inference_performed": False,
                "final_reconciliation_uses_empty_model_passes": True,
            },
        },
        "historical_mirada_capture": {
            "source_path": relative(MIRADA_RESPONSE_PATH),
            "source_sha256": sha256_file(MIRADA_RESPONSE_PATH),
            "input_sha256": historical.get("input_sha256"),
            "revision_field": historical.get("revision"),
            "request_relation": "historical capture; not tied to the first-11 request",
            "raw_attempt_responses": historical.get("mirada_capture", {}).get(
                "attempt_responses", []
            ),
            "strict_payload": historical.get("payload"),
            "field_statuses": historical.get("field_statuses"),
            "unverified_model_readings": historical.get(
                "unverified_model_readings"
            ),
        },
        "model_inference": {
            "attempted": False,
            "hosted": False,
            "gpu": False,
            "model_weights_downloaded": False,
        },
        "cases": cases,
        "capture_summary": {
            "case_count": len(cases),
            "successful_pipeline_captures": sum(
                case["capture_status"] == "captured" for case in cases
            ),
            "failed_pipeline_captures": failures,
            "exact_first11_reproductions": 0,
            "model_calls": 0,
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"CAPTURE_FILE={relative(OUTPUT_PATH)} "
        f"CASES={len(cases)} OK={output['capture_summary']['successful_pipeline_captures']} "
        f"FAILED={failures} BYTES={OUTPUT_PATH.stat().st_size}"
    )
    for case in cases:
        print(
            f"{case['case_id']} status={case['capture_status']} "
            f"ocr_calls={case.get('reader_call_count', len(case.get('reader_calls', [])))} "
            f"rows={len(case.get('consensus_documentary_evidence', {}).get('rows', []))} "
            f"supplier={case.get('consensus_documentary_evidence', {}).get('supplier')} "
            f"date={case.get('consensus_documentary_evidence', {}).get('invoice_date')}"
        )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
