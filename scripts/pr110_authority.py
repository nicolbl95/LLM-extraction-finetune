"""PR164: the document authority the PR110 eligibility screen decides from.

PR163 measured what the byte and perceptual screens cannot see: five of eleven supplied files were
already-used documents that reached the screen as downscaled re-renders, and two genuinely different
documents sat inside the 10-bit perceptual radius of an unrelated one. The cause is structural, not a
threshold: `signed_population()` answers "which images look alike", never "which document is this", and it
only knows the files that happen to sit in the repository.

This module supplies the missing layer. It records, per document, its identity, its known variants (the
original download, the anonymised copy the campaign evaluated, the development scan, the fixture), every
hash, and the recorded history. Two consequences:

* a byte-identical copy of an original download is now caught by hash even though the original is not in
  the repository - the hash is recorded, the file need not be present;
* a re-rendered copy that changed every byte still carries the page's geometry, so a minority-of-keypoints
  correspondence against a registered variant identifies it where a global hash cannot.

The correspondence is deliberately not "looks similar". It is a single homography that a majority of one
page's own keypoints agree on; on the campaign's corpus that separates 99-557 inliers for the same document
from at most 34 for look-alike documents. Phase 4 of the PR164 brief says a visual resemblance is a
suspicion, so the screen keeps a 10-bit similarity finding at `MANUAL_REVIEW_REQUIRED` and only a
registered variant excludes.

Nothing here reads a filename as evidence, uses a supplier name, or runs OCR, and PR166 Phase 2 checked
that against the committed registry. It holds, with one qualification that belongs on the record: verifying
a variant means opening the file, so every variant carries the path it was read from, and a corpus whose
files are named after the supplier that issued them leaves that name in a location field. Those fields are
the only place the registry locates a document, they stay inside the roots declared below, none is stored
absolute, and none carries anything printed on the page. The registry is generated from the repository's
own manifests so it can be rebuilt, diffed and committed.

PR167 separated the two halves of that sentence. The *format*, the *policy* and the *code* stay public;
the *content* - which private document is kept where, and which variants of it the campaign evaluated - moved
to a local file that Git ignores. `benchmark_prep/document-authority.private.json` is read from that location
(or from `$PR110_DOCUMENT_AUTHORITY`), and `benchmark_prep/document-authority.demo.json` is the tracked
demonstration: same schema, built only from the images the repository already publishes, and declared
`"scope": "demo"` so a screen that loads it refuses to admit anything. Discovery never picks the demonstration
up on its own, and a registry is trusted only when it declares `scope: "repository"` and every manifest in its
`built_from` is present and still hashes to what was recorded - the check that makes a clone without the
private authority report "I cannot tell" instead of "I have never seen this document".
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr109_intake as I  # noqa: E402

REGISTRY_VERSION = 1
# PR167: the real authority names private documents, so it lives where Git cannot see it. `REGISTRY_PATH`
# keeps its historical name for the callers and tests that mean "the authority of this machine".
REGISTRY_PATH = ROOT / "benchmark_prep" / "document-authority.private.json"
DEMO_REGISTRY_PATH = ROOT / "benchmark_prep" / "document-authority.demo.json"
# Where PR164-PR166 committed the authority. Nothing reads it any more; it is named so that `load()` can say
# why a registry found there is not used rather than fall back to it quietly.
LEGACY_TRACKED_PATH = ROOT / "benchmark_prep" / "document-authority.json"
REGISTRY_ENV = "PR110_DOCUMENT_AUTHORITY"
# The two roots whose images the repository itself publishes, so a registry built only from them can be
# committed without telling anyone where a private scan is kept.
PUBLIC_DEMO_ROOTS = (("app/examples", "demo_asset"), ("tests/fixtures", "regression_fixture"))

# A prefilter, not a decision: geometry is only computed for variants whose global hash already sits this
# close, so the expensive check stays bounded. The PR110 decision threshold (PERCEPTUAL_MAX_DISTANCE = 10)
# is untouched and is still the only thing that raises a similarity finding.
GEOMETRY_PREFILTER_BITS = 32
# A same page must own the majority of one image's keypoints in one homography. Measured on the campaign's
# own corpus: the same document scores 99-557 inliers at native scale, a different document of the same
# template family 15-27, unrelated documents 4-12.
MIN_INLIERS = 60
# A same rendering leaves almost no surviving correspondence unexplained; two invoices that merely share a
# template agree on the shared furniture but not on the text. Measured: 0.80-0.93 for the same document,
# 0.48 for two documents of the service/travel family, 0.16-0.32 for unrelated pages.
MIN_INLIER_FRACTION = 0.7
# Between these two numbers the correspondence is real but not conclusive - a same-template sibling reaches
# this band too. It is reported as a suspicion, never as an exclusion.
REVIEW_INLIERS = 30
MAX_SIDE = 1100
# Second attempt for a candidate rendered much smaller than the variant: both pages are resampled to one
# long side. It can only turn an unconclusive result into a suspicion, never into an exclusion, because two
# different documents of one template family reach 77 inliers at this scale.
NORMALISED_SIDE = 900
MAX_FEATURES = 3000
RATIO_TEST = 0.75
RANSAC_PIXELS = 5.0

# Directories whose images are development material by record. Each becomes one identity, and an
# `anonymized/<stem>` file is a variant of `real-scans*/<stem>` rather than a second document.
DEVELOPMENT_ROOTS = (
    ("real-scans", "development_scan"),
    ("real-scans-supplementary", "development_scan"),
    ("diagnostics/anonymized", "anonymized_copy"),
    ("tests/fixtures", "regression_fixture"),
    ("app/examples", "demo_asset"),
)
ANONYMIZED_PREFIX = "diagnostics/anonymized"

_FEATURE_CACHE: dict[tuple, dict | None] = {}
_MATCHERS: dict[str, object] = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(path: Path) -> dict:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(path.read_bytes())) as image:
        width, height = image.size
        grey = image.convert("L")
    ahash, dhash = I._hashes(grey)
    return {"width": width, "height": height, "ahash": ahash, "dhash": dhash}


def _display(path: Path) -> str:
    if path.is_absolute():
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return Path(path).as_posix()


def variant_record(path: Path, role: str, *, expect_present: bool = True) -> dict:
    record = {"path": _display(path), "role": role, "expect_present": expect_present,
              "sha256": None, "bytes": None, "width": None, "height": None, "ahash": None, "dhash": None}
    if path.is_file():
        signature = _signature(path)
        record.update({"sha256": sha256_file(path), "bytes": path.stat().st_size, **signature})
    return record


def _truth_identities() -> dict[str, dict]:
    """One identity per holdout document, from the truth manifest that already records it.

    `truth-*-v2.json` supersedes the first version of the same source, and an `evaluated_copy` is the
    variant the campaign actually ran, so both are read from the record instead of guessed from filenames.
    """
    identities: dict[str, dict] = {}
    manifests = sorted(ROOT.glob("diagnostics/holdout/**/truth-*.json"))
    for path in manifests:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = data.get("source") or {}
        evaluated = data.get("evaluated_copy") or {}
        document = data.get("document") or {}
        if not source.get("file") and not evaluated.get("file"):
            continue
        identity_id = path.stem.removeprefix("truth-").removesuffix("-v2")
        entry = identities.setdefault(identity_id, {
            "id": identity_id,
            "category": data.get("category"),
            "source_type": data.get("source_type"),
            "genuine_business_invoice": data.get("genuine_business_invoice"),
            "layout_family": data.get("layout_family"),
            "document_type": document.get("document_type"),
            "region": document.get("region"),
            "currency": document.get("currency"),
            "history": [],
            "variants": [],
        })
        for key in ("category", "source_type", "genuine_business_invoice", "layout_family"):
            if entry.get(key) is None and data.get(key) is not None:
                entry[key] = data.get(key)
        entry["history"].append({"kind": "holdout_evaluation",
                                 "record": path.relative_to(ROOT).as_posix()})

        original = source.get("file")
        if original and not any(v["role"] == "original_download" for v in entry["variants"]):
            record = variant_record(ROOT / "private_sources" / Path(original).name, "original_download",
                              expect_present=False)
            record["path"] = Path(original).name
            if not record["sha256"] and source.get("sha256"):
                record["sha256"] = source["sha256"]
            entry["variants"].append(record)
        copy = evaluated.get("file")
        if copy and not any(v["path"].endswith(copy) for v in entry["variants"]):
            bases = (ROOT / "diagnostics/holdout", ROOT / "diagnostics/holdout/new-layout")
            candidate = next((base / copy for base in bases if (base / copy).is_file()), bases[0] / copy)
            record = variant_record(candidate, "evaluated_copy", expect_present=True)
            if not record["sha256"] and evaluated.get("sha256"):
                record["sha256"] = evaluated["sha256"]
            entry["variants"].append(record)
    return identities


def _development_identities(roots: tuple = DEVELOPMENT_ROOTS) -> dict[str, dict]:
    """One identity per development image; `diagnostics/anonymized/<stem>` is a variant of its scan."""
    identities: dict[str, dict] = {}
    by_stem: dict[str, str] = {}
    for directory, kind in roots:
        root = ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in I.IMAGE_SUFFIXES:
                continue
            if directory == ANONYMIZED_PREFIX:
                identity_id = by_stem.get(path.stem)
                if identity_id:
                    identities[identity_id]["variants"].append(
                        variant_record(path, "anonymized_copy", expect_present=True))
                    identities[identity_id]["history"].append(
                        {"kind": "anonymized_for_evaluation", "record": path.relative_to(ROOT).as_posix()})
                    continue
                identity_id = f"anonymized/{path.stem}"
            else:
                identity_id = f"{Path(directory).name}/{path.stem}"
                by_stem.setdefault(path.stem, identity_id)
            identities.setdefault(identity_id, {
                "id": identity_id, "category": kind, "source_type": None,
                "genuine_business_invoice": None, "layout_family": None, "document_type": None,
                "region": None, "currency": None,
                "history": [{"kind": kind, "record": path.relative_to(ROOT).as_posix()}],
                "variants": [variant_record(path, kind, expect_present=True)],
            })
    return identities


def _attach_holdout_anonymised(identities: dict[str, dict]) -> None:
    """`diagnostics/holdout/<stem>-anon.png` belongs to `<stem>`, whether or not its truth records it."""
    root = ROOT / "diagnostics/holdout"
    if not root.is_dir():
        return
    by_stem = {}
    for identity_id, entry in identities.items():
        for variant in entry["variants"]:
            by_stem.setdefault(Path(variant["path"]).stem.removesuffix("-anon"), identity_id)
    for path in sorted(root.glob("*-anon.png")):
        stem = path.stem.removesuffix("-anon")
        identity_id = by_stem.get(stem, f"holdout/{stem}")
        entry = identities.setdefault(identity_id, {
            "id": identity_id, "category": "holdout_anonymized", "source_type": None,
            "genuine_business_invoice": None, "layout_family": None, "document_type": None,
            "region": None, "currency": None, "history": [], "variants": []})
        if any(v["sha256"] == sha256_file(path) for v in entry["variants"]):
            continue
        entry["variants"].append(variant_record(path, "anonymized_copy", expect_present=True))
        entry["history"].append({"kind": "anonymized_for_evaluation",
                                 "record": path.relative_to(ROOT).as_posix()})


def _documents(identities: dict[str, dict]) -> list[dict]:
    """Identities in sorted order, each variant owned by exactly one document."""
    seen: set[str] = set()
    documents = []
    for identity_id in sorted(identities):
        entry = identities[identity_id]
        unique = []
        for variant in entry["variants"]:
            marker = variant["sha256"] or f"{identity_id}:{variant['path']}:{variant['role']}"
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(variant)
        entry["variants"] = sorted(unique, key=lambda item: (item["role"], item["path"]))
        entry["history"] = sorted({(h["kind"], h["record"]) for h in entry["history"]})
        entry["history"] = [{"kind": kind, "record": record} for kind, record in entry["history"]]
        documents.append(entry)
    return documents


def _built_from() -> list[dict]:
    """The manifests this registry was generated from, hashed so a reader can check that claim."""
    built_from = []
    for path in sorted(list(ROOT.glob("diagnostics/holdout/*.json"))
                       + list(ROOT.glob("diagnostics/holdout/new-layout/*.json"))):
        if path.name.startswith(("manifest-", "truth-", "redaction-")):
            built_from.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)})
    return built_from


def _policy() -> dict:
    return {
        "similarity_threshold_bits": I.PERCEPTUAL_MAX_DISTANCE,
        "geometry_prefilter_bits": GEOMETRY_PREFILTER_BITS,
        "geometry_min_inliers": MIN_INLIERS,
        "note": "a global-hash resemblance raises MANUAL_REVIEW_REQUIRED; only a registered variant, "
                "by exact hash or by a single homography a majority of one page's keypoints agree on, "
                "excludes as previous use",
    }


def _envelope(documents: list[dict], *, scope: str, built_from: list[dict], purpose: str,
              note: str | None = None) -> dict:
    registry = {"version": REGISTRY_VERSION, "scope": scope, "purpose": purpose, "built_from": built_from,
                "policy": _policy(), "documents": documents}
    if note:
        registry["note"] = note
    return registry


def build() -> dict:
    """Rebuild the private registry from the repository's own manifests. Deterministic: no timestamp."""
    identities = _truth_identities()
    for key, value in _development_identities().items():
        if key not in identities:
            identities[key] = value
    _attach_holdout_anonymised(identities)
    return _envelope(_documents(identities), scope="repository", built_from=_built_from(),
                     purpose="documents the PR110 campaign has already used, their known variants "
                             "and their history")


def tracked_paths() -> set[str]:
    """The paths a clone receives, as Git understands them - not the files that happen to be on this disk.

    `demo_registry()` needs the distinction: a demonstration that names an image the repository does not ship
    arrives already accusing itself, because on another machine that variant is `missing` rather than verified.
    """
    import subprocess

    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    # Nothing listed means nothing to demonstrate, and a registry naming no document is the one thing `load()`
    # already refuses to certify independence from, so a silent Git degrades to the safe answer.
    return set() if listed.returncode != 0 else {line.strip() for line in listed.stdout.splitlines() if line.strip()}


def demo_registry() -> dict:
    """A registry in the published schema that knows only the images this repository already publishes.

    It exists so the format, the policy block and the screening code can be demonstrated and tested on a clone
    that holds no private authority. It is never a step in `resolve_registry()`, and it declares
    `scope: "demo"` and no `built_from`, so `load()` reports it as not the authority for this repository: a
    screen reading it admits nothing. That is the point - a demonstration must not be able to stand in for the
    record of what the campaign actually used.
    """
    tracked = tracked_paths()
    identities = {key: record for key, record in _development_identities(PUBLIC_DEMO_ROOTS).items()
                  if all(variant["path"] in tracked for variant in record["variants"])}
    return _envelope(_documents(identities), scope="demo", built_from=[],
                     purpose="a demonstration registry: the published schema, the published thresholds and "
                             "only the images this repository already tracks",
                     note="not the document authority. It cannot certify that a candidate is a document the "
                          "campaign never used, because it never looked at the private corpora; a screen that "
                          "loads it holds every document.")


def write(path: Path = REGISTRY_PATH, registry: dict | None = None) -> Path:
    registry = registry or build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def development_images(root: Path = ROOT) -> list[Path]:
    """The images `build()` looks at: the development roots and the top-level holdout anonymisations."""
    out: list[Path] = []
    for directory, _ in DEVELOPMENT_ROOTS:
        found = root / directory
        if found.is_dir():
            out += sorted(path for path in found.iterdir()
                          if path.is_file() and path.suffix.lower() in I.IMAGE_SUFFIXES)
    holdout = root / "diagnostics" / "holdout"
    if holdout.is_dir():
        out += sorted(holdout.glob("*-anon.png"))
    return out


def undeclared(registry: dict, root: Path = ROOT) -> list[str]:
    """Development images in a directory the registry describes, which the registry leaves out.

    `build()` walks those directories flat, so a directory the registry names images from is one it claims
    to have covered; a directory it names nothing from makes no claim and is not judged. This is what
    catches the one hazard a registry cannot catch from its own contents: a document that entered
    development use after the registry was written, which would otherwise read as never-used forever. A
    registry with no `built_from` is a fixture rather than a repository authority and claims nothing.
    """
    if not registry.get("built_from"):
        return []
    declared = {variant["path"] for document in registry.get("documents") or []
                for variant in document.get("variants") or []}
    covered = {str(Path(path).parent).replace("\\", "/") for path in declared}
    found = [path.relative_to(root).as_posix() for path in development_images(root)
             if path.relative_to(root).as_posix() not in declared
             and str(path.relative_to(root).parent).replace("\\", "/") in covered]
    return sorted(set(found))


def resolve_registry(path: Path | None = None) -> Path:
    """The authority to read: an explicit file, then `$PR110_DOCUMENT_AUTHORITY`, then the private default.

    The demonstration registry is deliberately not a step in this chain. A machine without the private
    authority has to end up at a file that does not exist, which `load()` reports as "nothing may be admitted",
    rather than at a public stand-in whose silence about the campaign's documents could be read as cleanliness.
    """
    if path is not None:
        return Path(path)
    from_environment = os.environ.get(REGISTRY_ENV)
    if from_environment:
        return Path(from_environment)
    return REGISTRY_PATH


def load(path: Path | None = None) -> dict:
    """Read the registry and verify it against the disk. Absence is reported, never replaced by emptiness."""
    path = resolve_registry(path)
    out = {"available": False, "path": str(path), "scope": None, "documents": [], "variants": [],
           "declared_variants": 0, "usable_variants": 0, "missing": [], "drift": [], "undeclared": [],
           "sources_declared": 0, "sources_verified": 0, "sources_unverifiable": [],
           "complete": False, "reason": None, "version": None}
    if not path.is_file():
        origin = (f"named by {REGISTRY_ENV}" if os.environ.get(REGISTRY_ENV) else
                  f"the local default, {REGISTRY_ENV} being unset")
        out["reason"] = (f"no document authority at {_display(path)} ({origin}); the screen cannot establish "
                         "which documents the campaign already used and must not admit anything until it can")
        return out
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        out["reason"] = f"the document authority could not be read ({type(error).__name__}: {error})"
        return out
    out["available"] = True
    out["version"] = registry.get("version")
    out["scope"] = registry.get("scope")
    out["documents"] = registry.get("documents") or []
    for document in out["documents"]:
        for variant in document.get("variants") or []:
            record = dict(variant)
            record["document"] = document.get("id")
            record["category"] = document.get("category")
            record["source_type"] = document.get("source_type")
            record["genuine_business_invoice"] = document.get("genuine_business_invoice")
            record["document_type"] = document.get("document_type")
            if not variant.get("expect_present") and not (ROOT / variant["path"]).is_file():
                record["state"] = "absent_by_design"
                out["variants"].append(record)
                continue
            out["declared_variants"] += 1
            on_disk = ROOT / variant["path"]
            if not on_disk.is_file():
                record["state"] = "missing"
                out["missing"].append(record)
            elif sha256_file(on_disk) != variant.get("sha256"):
                record["state"] = "drifted"
                out["drift"].append(record)
            else:
                record["state"] = "verified"
                out["usable_variants"] += 1
            out["variants"].append(record)
    out["undeclared"] = undeclared(registry)
    out["sources_unverifiable"] = verify_sources(registry)
    out["sources_declared"] = len(registry.get("built_from") or [])
    out["sources_verified"] = out["sources_declared"] - len(out["sources_unverifiable"])
    causes = []
    if out["scope"] not in (None, "repository"):
        causes.append(f"this registry declares scope {out['scope']!r}: it is not the record of what this "
                      "campaign used, so it cannot certify that a candidate document is new to it and "
                      "nothing may be admitted from it"
                      + (f" ({registry['note']})" if registry.get("note") else ""))
    if not out["documents"]:
        causes.append("this registry names no document"
                      + (f", though it says it was built from {out['sources_declared']} manifest(s)"
                         if out["sources_declared"] else "")
                      + "; a record of nothing cannot show a candidate to be new to the campaign, and an "
                        "empty match list against an empty registry is silence, not evidence of independence")
    if drifted := [entry for entry in out["sources_unverifiable"] if entry["state"] == "drifted"]:
        causes.append(f"{len(drifted)} of the {out['sources_declared']} manifest(s) this registry says it "
                      "was built from no longer hash to what it recorded ("
                      + ", ".join(entry["path"] for entry in drifted[:5])
                      + "); its claim to cover the corpora is stale, so nothing may be admitted until it is "
                        "rebuilt with scripts/pr110_authority.py")
    if out["missing"] or out["drift"]:
        causes.append(f"{len(out['missing'])} registered variant(s) are not on disk and "
                      f"{len(out['drift'])} no longer match their recorded hash; a re-rendered copy of "
                      "those documents could not be recognised here")
    if out["undeclared"]:
        causes.append(f"{len(out['undeclared'])} development image(s) in directories this registry covers are "
                      "absent from it (" + ", ".join(out["undeclared"][:5]) + "); the campaign has used "
                      "documents this registry has never heard of, so nothing may be admitted as independent "
                      "until it is rebuilt with scripts/pr110_authority.py")
    out["complete"] = not causes
    if causes:
        out["reason"] = " | ".join(causes)
    return out


def verify_sources(registry: dict, root: Path = ROOT) -> list[dict]:
    """The manifests a registry says it was generated from, checked against the bytes on disk.

    `built_from` is the only place the registry states where its own completeness comes from, so it is the
    claim worth verifying: a registry whose declared sources no longer hash as recorded is a registry that
    was not rebuilt after the corpora moved, which is exactly the state in which a used document reads as
    never-used. A source that is simply not on this machine is reported, not held against the registry: the
    private manifests are gitignored, so a clone legitimately cannot see them, and there a missing registry
    has already failed closed one step earlier.
    """
    out: list[dict] = []
    for entry in registry.get("built_from") or []:
        on_disk = root / entry["path"]
        if not on_disk.is_file():
            out.append({"path": entry["path"], "state": "absent"})
        elif sha256_file(on_disk) != entry.get("sha256"):
            out.append({"path": entry["path"], "state": "drifted"})
    return out


def empty() -> dict:
    """A declared-empty authority for tests: no known document, and nothing unverifiable."""
    return {"available": True, "path": None, "scope": "fixture", "documents": [], "variants": [],
            "declared_variants": 0, "usable_variants": 0, "missing": [], "drift": [], "undeclared": [],
            "sources_declared": 0, "sources_verified": 0, "sources_unverifiable": [],
            "complete": True, "reason": None, "version": REGISTRY_VERSION}


def exact_matches(digest: str, authority: dict) -> list[dict]:
    return [variant for variant in authority.get("variants") or [] if variant.get("sha256") == digest]


# --------------------------------------------------------------------------- geometric correspondence

def _features(path: Path, long_side: int = MAX_SIDE) -> dict | None:
    """Keypoints and descriptors, cached for the run: one candidate is compared against many variants."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size, long_side)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    try:
        import cv2
        import numpy

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            _FEATURE_CACHE[key] = None
            return None
        height, width = image.shape[:2]
        scale = long_side / max(height, width)
        if scale < 1:
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
        engine, detector = "sift", getattr(cv2, "SIFT_create", None)
        if detector is None:
            engine, detector = "orb", cv2.ORB_create(nfeatures=MAX_FEATURES)
            keypoints, descriptors = detector.detectAndCompute(image, None)
        else:
            keypoints, descriptors = detector(nfeatures=MAX_FEATURES).detectAndCompute(image, None)
        if descriptors is None or len(keypoints) < 8:
            _FEATURE_CACHE[key] = None
            return None
        entry = {"engine": engine, "points": numpy.float32([point.pt for point in keypoints]),
                 "descriptors": descriptors, "count": len(keypoints)}
        _FEATURE_CACHE[key] = entry
        return entry
    except Exception:  # an unusable image is reported by the caller, never turned into a match
        _FEATURE_CACHE[key] = None
        return None


def _matcher(engine: str):
    import cv2

    if engine not in _MATCHERS:
        norm = cv2.NORM_L2 if engine == "sift" else cv2.NORM_HAMMING
        _MATCHERS[engine] = cv2.BFMatcher(norm)
    return _MATCHERS[engine]


def correspondence(left: Path, right: Path, long_side: int = MAX_SIDE) -> dict | None:
    """One homography, and how many of the two pages' keypoints it explains."""
    import cv2
    import numpy

    first, second = _features(left, long_side), _features(right, long_side)
    if not first or not second or first["engine"] != second["engine"]:
        return None
    matcher = _matcher(first["engine"])
    raw = matcher.knnMatch(first["descriptors"], second["descriptors"], k=2)
    good = [match for match, rival in (pair for pair in raw if len(pair) == 2)
            if match.distance < RATIO_TEST * rival.distance]
    if len(good) < 8:
        return {"good": len(good), "inliers": 0, "scale": None, "covers": 0.0}
    source = numpy.float32([first["points"][match.queryIdx] for match in good]).reshape(-1, 1, 2)
    target = numpy.float32([second["points"][match.trainIdx] for match in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, RANSAC_PIXELS)
    inliers = int(mask.sum()) if mask is not None else 0
    scale = None
    if homography is not None:
        determinant = abs(float(numpy.linalg.det(homography[:2, :2])))
        scale = round(determinant ** 0.5, 3) if determinant > 0 else None
    return {"good": len(good), "inliers": inliers, "scale": scale,
            "covers": round(inliers / max(1, min(first["count"], second["count"])), 3)}


def variant_matches(path: Path, signature: dict, authority: dict) -> dict:
    """Registered variants that the candidate can be shown to be a re-render of."""
    best: list[dict] = []
    considered = 0
    for variant in authority.get("variants") or []:
        if variant.get("state") not in ("verified",):
            continue
        distance = min([value for value in (I.hamming(signature.get("ahash"), variant.get("ahash")),
                                            I.hamming(signature.get("dhash"), variant.get("dhash")))
                        if value is not None] or [99])
        if distance > GEOMETRY_PREFILTER_BITS:
            continue
        considered += 1
        found = correspondence(path, ROOT / variant["path"])
        if not found or found["inliers"] < REVIEW_INLIERS:
            continue
        fraction = found["inliers"] / max(1, found["good"])
        if found["inliers"] < MIN_INLIERS:
            # a candidate rendered much smaller than the variant: retry with both pages at one long side,
            # and keep the result only if it is conclusive on its own terms
            zoomed = correspondence(path, ROOT / variant["path"], long_side=NORMALISED_SIDE)
            if zoomed and zoomed["inliers"] >= MIN_INLIERS:
                found, fraction = zoomed, zoomed["inliers"] / max(1, zoomed["good"])
        strong = found["inliers"] >= MIN_INLIERS and fraction >= MIN_INLIER_FRACTION
        best.append({"document": variant.get("document"), "variant": variant["path"],
                     "role": variant.get("role"), "inliers": found["inliers"], "good": found["good"],
                     "fraction": round(fraction, 3), "scale": found["scale"], "covers": found["covers"],
                     "hash_distance": distance, "ambiguous": not strong, "strong": strong,
                     "category": variant.get("category"), "source_type": variant.get("source_type"),
                     "genuine_business_invoice": variant.get("genuine_business_invoice"),
                     "document_type": variant.get("document_type")})
    best.sort(key=lambda entry: (-int(entry["strong"]), -entry["inliers"]))
    return {"matches": best, "considered": considered}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help=f"where to write (default: the resolved authority, {REGISTRY_ENV} or "
                             f"{_display(REGISTRY_PATH)})")
    parser.add_argument("--check", action="store_true",
                        help="verify a registry against the disk without rewriting it")
    parser.add_argument("--demo", action="store_true",
                        help=f"write the tracked demonstration registry to {_display(DEMO_REGISTRY_PATH)} "
                             "instead of the private authority; it declares scope 'demo', so a screen that "
                             "loads it admits nothing")
    args = parser.parse_args(argv)

    if args.check:
        authority = load(args.output)
        print(f"registry: {authority['path']}")
        print(f"available {authority['available']} | complete {authority['complete']} | "
              f"scope {authority['scope']} | documents {len(authority['documents'])} | variants verified "
              f"{authority['usable_variants']}/{authority['declared_variants']} | sources verified "
              f"{authority['sources_verified']}/{authority['sources_declared']}")
        for entry in authority["missing"][:20]:
            print(f"  missing  {entry['document']} :: {entry['path']}")
        for entry in authority["drift"][:20]:
            print(f"  drifted  {entry['document']} :: {entry['path']}")
        for entry in authority["undeclared"][:20]:
            print(f"  undeclared development image :: {entry}")
        for entry in authority["sources_unverifiable"][:20]:
            print(f"  declared source {entry['state']} :: {entry['path']}")
        if authority["reason"]:
            print("  reason:", authority["reason"])
        return 0 if authority["complete"] else 1

    if args.demo:
        registry = demo_registry()
        write(DEMO_REGISTRY_PATH, registry)
        variants = sum(len(document["variants"]) for document in registry["documents"])
        print(f"wrote {_display(DEMO_REGISTRY_PATH)}: scope demo, {len(registry['documents'])} "
              f"document(s), {variants} variant(s) - a screen loading it admits nothing")
        return 0

    registry = build()
    output = Path(args.output) if args.output else resolve_registry()
    write(output, registry)
    variants = sum(len(document["variants"]) for document in registry["documents"])
    present = sum(1 for document in registry["documents"] for variant in document["variants"]
                  if variant["sha256"] and variant["bytes"])
    print(f"wrote {_display(output)}: scope repository, {len(registry['documents'])} document(s), "
          f"{variants} variant(s), {present} present on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
