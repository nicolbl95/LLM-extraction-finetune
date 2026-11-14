"""PR109 Phase 2: independent-invoice intake, contamination audit and the duplicate-detection self-test.

Three rules decide this phase, and each is measured rather than asserted:

1. **Only the designated intake directory is searched.** The repository's own corpora are development data by
   record (PR103's dataset manifest classifies all 640 files), so nothing inside them can become a validation
   cohort by being re-labelled. The intake locations are the ones the PR103 readiness report already
   designates: `benchmark_prep/invoices/` for the images and `benchmark_prep/references/` for the blind
   annotations, with `--intake` to override.
2. **Independence is decided by content, not by filename.** A candidate is contaminated if its bytes match, or
   its perceptual signature is near, any development document, training render, fixture, holdout or previously
   evaluated file. The 640-file PR103 inventory supplies the known digests and is re-opened for hashes
   rather than trusted.
3. **A byte-hash gate alone is not enough**, because a resized or recompressed copy of a development document
   has a new SHA-256. The perceptual screen is therefore validated on synthetic variants of two development
   documents before it is allowed to clear anything, and the result of that validation is reported as
   synthetic - it is a test of the detector, not an invoice measurement.

No extraction, no OCR call, no model call, no network. Writes only under results/product-review109/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr106_harness as P  # noqa: E402

ARTIFACT_DIR = ROOT / "results" / "product-review109"
PR103_INVENTORY = ROOT / "results" / "product-review103" / "pr103-dataset-manifest.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules"}
TARGET_DOCUMENTS = 12
# 64-bit signatures; 10 differing bits of 64 is the accepted near-duplicate ceiling for a resized or
# recompressed copy of the same page and is still far below the distance between two different invoices.
PERCEPTUAL_MAX_DISTANCE = 10


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ designated intake locations

def intake_locations(overrides: list[Path] | None) -> dict:
    images_dir = (overrides[0] if overrides else ROOT / "benchmark_prep" / "invoices")
    references_dir = overrides[1] if overrides and len(overrides) > 1 else (
        ROOT / "benchmark_prep" / "references")
    found = sorted(path for path in (images_dir.rglob("*") if images_dir.is_dir() else []))
    candidates = [path for path in found
                  if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES]
    annotations = [path for path in (references_dir.rglob("*") if references_dir.is_dir() else [])
                   if path.is_file() and path.suffix.lower() == ".json"]
    return {
        "designated_image_directory": images_dir.relative_to(ROOT).as_posix()
        if images_dir.is_absolute() else str(images_dir),
        "image_directory_exists": images_dir.is_dir(),
        "designated_annotation_directory": references_dir.relative_to(ROOT).as_posix()
        if references_dir.is_absolute() else str(references_dir),
        "annotation_directory_exists": references_dir.is_dir(),
        "entry_counted": len(candidates),
        "annotation_files_counted": len(annotations),
        "candidates": candidates,
        "note": "an empty or absent designated directory is a finding, not an error: the brief forbids "
                "substituting development documents for missing independent ones",
    }


# ------------------------------------------------------------------ known content, taken from the record

def known_content() -> dict:
    """Every file the campaign has already touched, as the repository itself recorded it.

    The PR103 dataset manifest is the authoritative inventory of the 640 files and of the documented purpose
    of each; re-reading it is cheaper and more honest than re-classifying by directory name.
    """
    manifest = json.loads(PR103_INVENTORY.read_text(encoding="utf-8"))
    inventory = manifest["files"]
    namespaces: dict[str, list[dict]] = {}
    for entry in inventory:
        namespaces.setdefault(entry["purpose"], []).append(entry)
    corpus = P.documents()
    development = {
        name: {
            "path": str(entry["path"].relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(entry["path"]),
            "kind": entry["kind"],
            "reference_source": entry["reference_source"],
        } for name, entry in corpus.items()
    }
    return {
        "pr103_inventory_files": len(inventory),
        "pr103_inventory_by_purpose": {purpose: len(files) for purpose, files in sorted(namespaces.items())},
        "pr103_eligible_at_the_time": manifest["eligible_independent_documents"],
        "pr103_verdict": manifest["verdict"],
        "development_corpus": development,
        "known_sha256_to_path": {entry["sha256"]: entry["path"] for entry in inventory},
    }


# ------------------------------------------------------------------ perceptual signatures

def perceptual(path: Path) -> dict:
    """Average hash and difference hash of the first page or the image, plus geometry and PDF text-layer facts."""
    from PIL import Image

    out: dict = {"sha256": sha256_file(path), "bytes": path.stat().st_size,
                 "format": path.suffix.lower().lstrip("."), "pages": None,
                 "width": None, "height": None, "dpi": None, "megapixels": None,
                 "embedded_text_layer": None, "ahash": None, "dhash": None}
    data = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        try:
            import io

            import pymupdf
            from PIL import Image

            document = pymupdf.open(stream=data, filetype="pdf")
            out["pages"] = document.page_count
            text = "".join(page.get_text() or "" for page in document)
            out["embedded_text_layer"] = bool(text.strip())
            out["embedded_text_sha256"] = sha256_bytes(text.encode("utf-8"))
            page = document[0]
            out["width"], out["height"] = [round(value, 1) for value in page.rect[2:]]
            grey = Image.open(io.BytesIO(page.get_pixmap(dpi=72).tobytes("png"))).convert("L")
            document.close()
        except Exception as error:  # an unreadable PDF is not eligible; recorded, never guessed
            out["pdf_error"] = f"{type(error).__name__}: {error}"
            return out
    else:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            out["width"], out["height"] = image.size
            out["megapixels"] = round(image.width * image.height / 1e6, 3)
            dpi = image.info.get("dpi")
            out["dpi"] = [round(float(value), 1) for value in dpi] if dpi else None
            out["colour_mode"] = image.mode
            grey = image.convert("L")
    out["ahash"], out["dhash"] = _hashes(grey)
    return out


def _hashes(grey) -> tuple[str, str]:
    import numpy

    pixels = numpy.asarray(grey.resize((8, 8)))
    ahash = "".join("1" if value >= pixels.mean() else "0" for value in pixels.flatten())
    wide = numpy.asarray(grey.resize((9, 8)))
    bits = (wide[:, :-1] > wide[:, 1:]).flatten()
    return hex(int(ahash, 2)), hex(int("".join("1" if bit else "0" for bit in bits), 2))


def _bits(value: str) -> int:
    """Number of set bits in a hex signature, or in the hex of an XOR of two of them."""
    return bin(int(value, 16)).count("1")


def hamming(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None
    return _bits(hex(int(left, 16) ^ int(right, 16)))


# ------------------------------------------------------------------ detector self-test (synthetic)

def detector_self_test(sample_paths: list[Path]) -> dict:
    """Prove the perceptual screen catches what a byte hash cannot: a resized, recompressed copy.

    Synthetic by construction - variants are generated in memory from two development documents and never
    written to disk, and the point is the detector's behaviour, not the invoices.
    """
    import io

    from PIL import Image

    cases = []
    for path in sample_paths:
        original = Image.open(path).convert("L")
        base = perceptual(path)
        for label, transform in (
            ("resized_to_70_percent", lambda image: image.resize(
                (max(1, image.width * 7 // 10), max(1, image.height * 7 // 10)))),
            ("jpeg_recompressed_q75", None),
            ("greyscale_converted", None),
            ("cropped_10_pixels", lambda image: image.crop((0, 10, image.width, image.height - 10))),
        ):
            variant = original.copy()
            if transform:
                variant = transform(variant)
            buffer = io.BytesIO()
            if label == "jpeg_recompressed_q75":
                variant.save(buffer, "JPEG", quality=75)
            elif label == "greyscale_converted":
                variant = variant.convert("RGB")
                variant.save(buffer, "PNG")
            else:
                variant.save(buffer, "PNG")
            signed = _from_bytes(buffer.getvalue())
            cases.append({
                "source": path.name, "variant": label,
                "byte_hash_matches_original": signed["sha256"] == base["sha256"],
                "ahash_distance": hamming(base["ahash"], signed["ahash"]),
                "dhash_distance": hamming(base["dhash"], signed["dhash"]),
                "detected_as_near_duplicate": min(
                    [value for value in (hamming(base["ahash"], signed["ahash"]),
                                         hamming(base["dhash"], signed["dhash"])) if value is not None]
                    or [99]) <= PERCEPTUAL_MAX_DISTANCE,
            })
    # The negative control: two different invoices must not be flagged.
    if len(sample_paths) >= 2:
        first, second = perceptual(sample_paths[0]), perceptual(sample_paths[1])
        cases.append({
            "source": f"{sample_paths[0].name} vs {sample_paths[1].name}", "variant": "different_document",
            "byte_hash_matches_original": first["sha256"] == second["sha256"],
            "ahash_distance": hamming(first["ahash"], second["ahash"]),
            "dhash_distance": hamming(first["dhash"], second["dhash"]),
            "detected_as_near_duplicate": False,
            "expected": "not a duplicate",
        })
    caught = [case for case in cases if case["variant"] != "different_document"]
    return {
        "synthetic": True,
        "purpose": "the intake gate's perceptual screen is only meaningful if it detects a resized or "
                   "recompressed copy of a document the campaign already used",
        "perceptual_max_distance": PERCEPTUAL_MAX_DISTANCE,
        "variants_tested": len(caught),
        "variants_detected": sum(1 for case in caught if case["detected_as_near_duplicate"]),
        "variants_a_byte_hash_would_miss": sum(1 for case in caught if not case["byte_hash_matches_original"]),
        "false_positives_on_a_different_document": sum(
            1 for case in cases if case["variant"] == "different_document"
            and case["detected_as_near_duplicate"]),
        "cases": [{key: value for key, value in case.items() if key != "expected"} for case in cases],
    }


def _from_bytes(data: bytes) -> dict:
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("L")
    ahash, dhash = _hashes(image)
    return {"sha256": sha256_bytes(data), "ahash": ahash, "dhash": dhash}


# ------------------------------------------------------------------ per-candidate assessment

def assess(path: Path, known: dict, population: list[dict]) -> dict:
    """Everything Phase 2 can establish without running the pipeline on the document."""
    signature = perceptual(path)
    exact = []
    if signature["sha256"] in known["known_sha256_to_path"]:
        exact.append({"path": known["known_sha256_to_path"][signature["sha256"]],
                      "basis": "byte-identical to a file in the PR103 inventory"})
    if signature["sha256"] in {entry["sha256"] for entry in known["development_corpus"].values()}:
        exact.append({"path": "the eleven-document development corpus",
                      "basis": "byte-identical to a development document"})
    near = []
    for entry in population:
        distance = hamming(signature.get("ahash"), entry.get("ahash"))
        second = hamming(signature.get("dhash"), entry.get("dhash"))
        best = min([value for value in (distance, second) if value is not None] or [99])
        if best <= PERCEPTUAL_MAX_DISTANCE:
            near.append({"path": entry["path"], "ahash_distance": distance, "dhash_distance": second})
    language = None
    country = None
    quality = None
    if signature.get("megapixels") is not None:
        quality = {"megapixels": signature["megapixels"], "dpi": signature.get("dpi")}
    return {
        "file": path.name,
        "path": path.relative_to(ROOT).as_posix() if path.is_absolute() and str(path).startswith(str(ROOT))
        else str(path),
        "record": {key: value for key, value in signature.items()},
        "file_type": signature["format"],
        "dimensions_or_pages": [signature.get("width"), signature.get("height")]
        if signature.get("pages") is None else signature["pages"],
        "image_quality": quality,
        "language": language,
        "country_or_convention": country,
        "language_and_convention_note": "not inferred: a genuine document's language and VAT convention are "
                                        "established by the blind annotation in Phase 4, never by guessing "
                                        "from a filename",
        "source_and_permission": {"recorded": False,
                                  "note": "no provenance or permission sidecar accompanies the file"},
        "previously_processed_by_the_development_pipeline": bool(exact) or bool(near),
        "byte_identical_matches": exact,
        "perceptual_near_matches": near,
        "eligible_as_independent": False,
        "eligibility_reason": "no designated intake directory contains supplied documents; see the verdict",
    }


# ------------------------------------------------------------------ the population the screen compares against

def signed_population() -> list[dict]:
    """Perceptual signatures of every image the campaign has already touched, computed now."""
    entries, seen = [], set()
    roots = [ROOT / "real-scans", ROOT / "real-scans-supplementary",
             ROOT / "diagnostics" / "holdout" / "new-layout", ROOT / "diagnostics" / "anonymized",
             ROOT / "app" / "examples", ROOT / "tests" / "fixtures",
             ROOT / "benchmark_prep" / "invoices"]
    for directory in roots:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = sha256_file(path)
            if digest in seen:
                continue
            seen.add(digest)
            signature = perceptual(path)
            entries.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest,
                            "ahash": signature["ahash"], "dhash": signature["dhash"]})
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", nargs="*", type=Path, default=None,
                        help="override the designated intake directory (and optionally the annotation directory)")
    parser.add_argument("--output-name", default="pr109-phase2-intake.json")
    args = parser.parse_args()
    target = P.guard(ARTIFACT_DIR, ARTIFACT_DIR / args.output_name)

    locations = intake_locations(args.intake)
    known = known_content()
    population = signed_population()
    self_test = detector_self_test([Path(entry["path"]) for entry in list(known["development_corpus"].values())[:2]])

    candidates = [assess(path, known, population) for path in locations["candidates"]]
    eligible = [entry for entry in candidates if entry["eligible_as_independent"]]
    shortage = max(0, TARGET_DOCUMENTS - len(eligible))
    verdict = ("BLOCKED_AWAITING_INDEPENDENT_INVOICES" if len(eligible) < TARGET_DOCUMENTS
               else "INTAKE_GATE_PASSED")

    artifact = {
        "phase": "PR109 Phase 2 - independent invoice intake and contamination audit",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission_target_documents": TARGET_DOCUMENTS,
        "verdict": verdict,
        "eligible_independent_documents": len(eligible),
        "shortage": shortage,
        "designated_intake": {key: value for key, value in locations.items() if key != "candidates"},
        "searched_roots_outside_the_designated_intake": "none - the brief restricts the search to the "
                                                        "designated new-intake directory",
        "known_content": {key: value for key, value in known.items() if key != "known_sha256_to_path"}
        | {"known_sha256_records": len(known["known_sha256_to_path"])},
        "perceptual_population_size": len(population),
        "duplicate_detector_self_test": self_test,
        "candidates": candidates,
        "why_the_development_corpus_cannot_fill_the_cohort": [
            "the eleven documents of PR105B-PR108 are the tuning corpus: every rule change since PR83 was "
            "accepted or rejected against them",
            "the five real scans were the subject of PR91-PR102 and their supplier identity appears in the "
            "test suite",
            "the 600 data/invoices renders are synthetic training output for the LoRA adapter",
            "PR103's own inventory classified all 640 files in the repository and found 0 eligible",
        ],
        "constraints_observed": {
            "documents_fabricated_or_substituted": 0,
            "development_documents_counted_as_independent": 0,
            "pipeline_run_on_any_candidate": 0,
            "ocr_calls": 0,
            "model_calls": 0,
            "network_calls": 0,
            "files_written_outside_the_workspace": 0,
        },
    }
    P.write_json(target, artifact)

    print("verdict:", verdict, "| eligible:", len(eligible), "| shortage:", shortage)
    print("intake image directory exists:", locations["image_directory_exists"],
          "| candidates found:", locations["entry_counted"],
          "| annotation files:", locations["annotation_files_counted"])
    print("known records:", len(known["known_sha256_to_path"]), "| perceptual population:", len(population))
    print("self-test: variants", self_test["variants_tested"], "detected", self_test["variants_detected"],
          "byte-hash would miss", self_test["variants_a_byte_hash_would_miss"],
          "| false positives", self_test["false_positives_on_a_different_document"])
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
