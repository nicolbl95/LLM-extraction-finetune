"""PR105B shared harness: one frozen production acquisition path, four description rules, one scorer.

No production file is modified. Three isolation rules make the comparison honest:

* **One evidence policy.** `ocr_recovery.recover_image` feeds intermediate consensus results back into its
  own decisions (the ruled-region early return at `:237-249`, the `comparable and legible` probe at
  `:315-354`, the header re-reads and `_clarify_decimal_quantity`), so a description rule could otherwise
  win by reading *more of the page*. Every OCR call is replayed positionally from the recorded production
  sequence for that document, and a request whose bytes were never recorded for that document is answered
  with a no-text reading and counted in `suppressed_calls`: a configuration can only consume frozen
  evidence, never new pixels and never live Tesseract.
* **Consensus pinning was tried and is invalid.** `build_consensus` starts from `deepcopy(anchor)`
  (`src/ocr_consensus.py:152`), so its rows and tokens are its own objects, but the *values* it then adopts
  come straight from the readings: `_agreed` returns the original candidate objects (`:65`), and an identity
  probe on real documents confirms that every monetary `Number` in a consensus result is the same object as
  a reading's `Number`, while no row or token is. Replacing an intermediate decision with a `deepcopy` of
  the recorded result severs that sharing; measured, it moved the payloads of `us-fast-photography` and
  `layout-d-service-anon` under the *unchanged* production rule, so the pinned baseline no longer equalled
  production and no candidate could be trusted. Which shared object carries the difference was not isolated
  further because the whole design was discarded. Every consensus call therefore
  runs the variant module, and configuration A — the variant module running the production rule — is the
  gate that proves the replay is faithful.
* **Derived variant, not a mirror.** The experimental module is produced from the *source text of
  `src/ocr_consensus.py`* with exactly one contiguous block substituted (the description `_agreed` call at
  `:242-248`) plus a delimited helper footer. The two-family minimum, the segmentation collapse, the
  monetary confidence gates, the arithmetic recovery and the U+FFFD rejection stay production code, so a
  rule that never fires on this corpus cannot silently diverge (the PR83 lesson).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

FIXTURES = ROOT / "diagnostics" / "holdout" / "new-layout"
ARTIFACT_DIR = ROOT / "results" / "product-review105b"
CACHE_NAME = "pr105b-ocr-cache.json"

LAYOUTS = {
    "layout-a": "truth-sumup-avoir.json",
    "layout-b": "truth-consulting.json",
    "layout-c": "truth-retail.json",
    "layout-d": "truth-service-v2.json",
    "layout-e": "truth-travel-v2.json",
    "layout-f": "truth-photographer.json",
}
ROW_FIELDS = ("description", "quantity", "unit_price", "total_ht")

# configuration -> (description confidence token set, unanimity requirement)
CONFIGS = {
    "A": ("line", True),
    "B": ("field", True),
    "C": ("line", False),
    "D": ("field", False),
}

PRODUCTION_SOURCE_PATH = ROOT / "src" / "ocr_consensus.py"

DESCRIPTION_BLOCK = (
    "        descriptions = _agreed(\n"
    "            matches, lambda candidate: [candidate.description] if (\n"
    "                all(token.confidence >= confidence for token in candidate.line.tokens)\n"
    "                and not has_replacement_character(candidate.description)\n"
    "            ) else [], folded, unanimous=True, prefer=_accent_count,\n"
    "        )\n"
)
VARIANT_BLOCK = (
    "        descriptions = _agreed(\n"
    "            matches, lambda candidate: [candidate.description] if (\n"
    "                _pr105b_gate(candidate, confidence)\n"
    "                and not has_replacement_character(candidate.description)\n"
    "            ) else [], folded, unanimous=_pr105b_unanimous, prefer=_accent_count,\n"
    "        )\n"
)
FOOTER = '''

# ---- PR105B experimental footer. Everything above this line is the production source with the single
# description `_agreed` call substituted; `src/ocr_consensus.py` itself is untouched on disk. The two
# module globals below are the only behavioural difference between the four PR105B configurations. ----
_pr105b_tokens_mode = "line"
_pr105b_unanimous = True
_pr105b_gate_log: list[dict] = []


def _pr105b_description_tokens(row):
    """Row-line tokens that are not inside one of that row's own numeric cell boxes.

    With no numeric cell evidence the whole line is the description evidence, so the field-level rule never
    invents a column the row builder did not find: it is exactly the production rule in that case.
    """
    occupied = [
        number.box
        for key in ("quantity", "unit_price", "total_ht", *OPTIONAL_HEADERS)
        for number in row.cells.get(key, [])
    ]
    kept = [
        token for token in row.line.tokens
        if not any(
            box[0] <= (token.box[0] + token.box[2]) / 2 < box[2]
            and box[1] <= (token.box[1] + token.box[3]) / 2 < box[3]
            for box in occupied
        )
    ]
    return (kept or list(row.line.tokens)), bool(kept)


def _pr105b_gate(candidate, confidence):
    if _pr105b_tokens_mode == "line":
        return all(token.confidence >= confidence for token in candidate.line.tokens)
    tokens, used_column = _pr105b_description_tokens(candidate)
    _pr105b_gate_log.append({
        "tokens": len(tokens),
        "line_tokens": len(candidate.line.tokens),
        "used_column": used_column,
        "line_min_confidence": min((token.confidence for token in candidate.line.tokens), default=None),
        "description_min_confidence": min((token.confidence for token in tokens), default=None),
        "gate_line": all(token.confidence >= confidence for token in candidate.line.tokens),
        "gate_field": all(token.confidence >= confidence for token in tokens),
    })
    return all(token.confidence >= confidence for token in tokens)
'''


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dump(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def payload_digest(payload: dict) -> str:
    return hashlib.sha256(json_dump(payload).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- corpus


def corpus() -> dict[str, dict]:
    """The eleven development documents, each with its frozen reference annotation.

    Layout references name the printed HT subtotal `subtotal_ht`; the export contract calls the same value
    `total_ht`, so the reference is relabelled and the mapping is recorded in the artifact.
    """
    documents: dict[str, dict] = {}
    for name, reference in read_json(ROOT / "real-scans-truth.json")["documents"].items():
        documents[name.removesuffix(".png")] = {
            "path": ROOT / "real-scans" / name,
            "kind": "real-scan",
            "reference": reference,
            "reference_source": "real-scans-truth.json",
            "reference_mapping": None,
        }
    for key, truth_name in LAYOUTS.items():
        raw = read_json(FIXTURES / truth_name)
        source, copy = raw["document"], raw["evaluated_copy"]
        documents[f"{key}-{Path(copy['file']).stem}"] = {
            "path": FIXTURES / copy["file"],
            "kind": "frozen-layout",
            "reference": {
                **{field: source.get(field) for field in ("supplier_name", "invoice_date", "total_ttc")},
                "total_ht": source.get("subtotal_ht"),
                "line_items": [{field: row.get(field) for field in ROW_FIELDS} for row in source["line_items"]],
            },
            "reference_source": f"diagnostics/holdout/new-layout/{truth_name}",
            "reference_mapping": "document.subtotal_ht -> reference.total_ht",
            "image_sha256": sha256_file(FIXTURES / copy["file"]),
            "frozen_image_sha256": copy["sha256"],
        }
    missing = [name for name, entry in documents.items() if not entry["path"].exists()]
    if missing:
        raise SystemExit(f"corpus document images missing: {missing}")
    return documents


# ----------------------------------------------------------------------------- OCR bytes


def _call_key(image, lang: str, config: str) -> str:
    if hasattr(image, "mode") and hasattr(image, "size"):
        identity = f"pil|{image.mode}|{image.size}"
    else:
        identity = f"raw|{getattr(image, 'dtype', '')}|{getattr(image, 'shape', '')}"
    return f"{hashlib.sha256(identity.encode('utf-8') + image.tobytes()).hexdigest()}|{lang}|{config}"


def _empty_reading() -> dict:
    """A Tesseract response with no detected text, in the recorded `image_to_data` shape.

    `_clarify_decimal_quantity` (`src/ocr_recovery.py:128-134`) breaks on an empty reading without touching
    the evidence, so returning one is the only response to an unrecorded call that cannot hand a candidate
    pixels the baseline never saw. Such calls are counted and reported, never silently served.
    """
    return {key: [] for key in (
        "level", "page_num", "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height", "conf", "text",
    )}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise SystemExit("non-finite OCR confidence cannot be frozen")
        return float(value)
    raise SystemExit(f"unsupported OCR value type {type(value)}")


# ----------------------------------------------------------------------------- freeze


class Freeze:
    """One evidence policy shared by every configuration.

    mode "capture"  live Tesseract, records the OCR bytes, the ordered call keys and each consensus result.
    mode "reference" replays the recorded OCR bytes by content, unpinned consensus, and re-records the
                     ordered keys and consensus results, so a driver run is self-contained and repeatable.
    mode "pinned"    replays OCR positionally against the recorded production sequence *of that document*
                     and runs every consensus call through the variant module. Bytes never recorded for
                     that document are not granted: the call returns a no-text reading and is counted.
    """

    def __init__(self, mode: str, entries: dict | None = None, recordings: dict | None = None):
        if mode not in ("capture", "reference", "pinned"):
            raise ValueError(mode)
        if mode == "pinned" and not recordings:
            raise ValueError("pinned mode needs the reference recordings")
        self.mode = mode
        self.entries = {} if entries is None else entries
        self.recordings = recordings or {}
        self.recording: dict | None = None
        self.keys: list[str] = []
        self.consensus: list[object] = []
        self.consensus_calls_beyond_recorded = 0
        self.live_calls = 0
        self.off_sequence_calls: list[str] = []
        self.suppressed_calls: list[str] = []
        self.variant: types.ModuleType | None = None
        self.config: str | None = None
        self._original_reader = None
        self._original_consensus = None

    # -- OCR seam -------------------------------------------------------------------------
    def reader(self, image, *args, **kwargs):
        key = _call_key(image, kwargs.get("lang") or "", kwargs.get("config") or "")
        index = len(self.keys)
        if self.mode == "capture":
            data = _json_safe(self._original_reader(image, *args, **kwargs))
            self.live_calls += 1
            if key in self.entries and self.entries[key] != data:
                raise SystemExit(f"OCR key reused with a different output: {key}")
            self.entries[key] = data
        elif self.mode == "reference":
            if key not in self.entries:
                raise SystemExit(f"frozen cache has no entry for {key}: the evidence is not reproducible")
            data = self.entries[key]
        else:
            expected = self.recording["keys"]
            if index < len(expected) and expected[index] == key:
                data = self.entries[key]
            elif key in set(expected):
                # Bytes this document did record, at another position: the rule changed what the pipeline
                # chose to read, so the evidence stays frozen but the order no longer matches production.
                self.off_sequence_calls.append(
                    f"position {index}: {key} (production read {expected[index] if index < len(expected) else 'nothing'})")
                data = self.entries[key]
            else:
                # Pixels this document never recorded are not granted to any configuration.
                self.suppressed_calls.append(f"position {index}: {key}")
                data = _empty_reading()
        self.keys.append(key)
        return data

    # -- consensus seam -------------------------------------------------------------------
    def consensus_call(self, readings, *, confidence):
        """The rule under test decides every consensus call; only the bytes are frozen.

        Pinning an intermediate decision to a copy is not a faithful replay because production aliases the
        readings into its result; `consensus_calls_beyond_recorded` counts any call the recorded production
        path did not make, which means the rule changed a downstream decision.
        """
        if self.mode in ("capture", "reference"):
            result = self._original_consensus(readings, confidence=confidence)
            self.consensus.append(deepcopy(result))
            return result
        if len(self.consensus) >= len(self.recording["consensus"]):
            self.consensus_calls_beyond_recorded += 1
        result = self.variant.build_consensus(readings, confidence=confidence)
        self.consensus.append(result)
        return result

    @contextlib.contextmanager
    def active(self):
        import ocr_fast_path
        import ocr_recovery
        import pytesseract

        self._original_reader = pytesseract.image_to_data
        self._original_consensus = ocr_recovery.build_consensus
        with (
            patch.object(pytesseract, "image_to_data", side_effect=self.reader),
            patch.object(ocr_recovery, "build_consensus", side_effect=self.consensus_call),
            patch.object(ocr_fast_path, "build_consensus", side_effect=self.consensus_call),
        ):
            yield


# ----------------------------------------------------------------------------- variant module


def transform(source: str) -> str:
    if source.count(DESCRIPTION_BLOCK) != 1:
        raise SystemExit(
            f"production description block not found exactly once "
            f"(count {source.count(DESCRIPTION_BLOCK)}); the consensus source changed and this harness "
            "must be re-read before any measurement"
        )
    return source.replace(DESCRIPTION_BLOCK, VARIANT_BLOCK) + FOOTER


def load_variant(source: str) -> types.ModuleType:
    module = types.ModuleType("pr105b_ocr_consensus_variant")
    module.__file__ = str(PRODUCTION_SOURCE_PATH)
    module.pr105b_production_source = source
    module.pr105b_transformed_source = transform(source)
    sys.modules[module.__name__] = module
    exec(compile(module.pr105b_transformed_source, module.__file__, "exec"), module.__dict__)
    return module


# ----------------------------------------------------------------------------- one document


def run_document(name: str, entry: dict, freeze: Freeze, variant=None, config: str | None = None) -> dict:
    """`app.main.document_text` on the frozen acquisition path, scored through the production export."""
    from app import main
    from invoice_evidence import UnreadableOCR
    from invoice_grounding import reconcile

    freeze.keys, freeze.consensus, freeze.off_sequence_calls = [], [], []
    freeze.suppressed_calls, freeze.consensus_calls_beyond_recorded = [], 0
    freeze.variant, freeze.config = variant, config
    freeze.recording = freeze.recordings.get(name)
    if freeze.mode == "pinned" and freeze.recording is None:
        raise SystemExit(f"pinned mode has no reference recording for {name}")
    if variant is not None:
        variant._pr105b_tokens_mode, variant._pr105b_unanimous = CONFIGS[config][0], CONFIGS[config][1]
        variant._pr105b_gate_log.clear()
    operational_error = None
    with freeze.active():
        try:
            document = main.document_text(str(entry["path"]))
            evidence = document.evidence
        except UnreadableOCR as error:
            operational_error = f"{type(error).__name__}: {error}"
            evidence = None
        except (OSError, ValueError, RuntimeError) as error:
            main.LOGGER.exception("PR105B could not read %s", name)
            operational_error = f"{type(error).__name__}: {error}"
            evidence = None
    if evidence is None:
        return {"document": name, "technical_error": operational_error, "payload": None,
                "payload_sha256": None, "ocr_call_keys": list(freeze.keys),
                "gate_log": list(variant._pr105b_gate_log) if variant else [],
                "off_sequence_calls": list(freeze.off_sequence_calls),
                "suppressed_calls": list(freeze.suppressed_calls)}
    payload = reconcile(evidence, [{}]).payload
    return {
        "document": name,
        "technical_error": operational_error,
        "payload": payload,
        "payload_sha256": payload_digest(payload),
        "source_kind": evidence.source_kind,
        "ocr_calls": getattr(evidence, "ocr_calls", None),
        "ocr_call_keys": list(freeze.keys),
        "consensus_calls": len(freeze.consensus),
        "consensus_calls_beyond_recorded": freeze.consensus_calls_beyond_recorded,
        "off_sequence_calls": list(freeze.off_sequence_calls),
        "suppressed_calls": list(freeze.suppressed_calls),
        "exported_rows": len(payload.get("line_items") or []),
        "nonempty_descriptions": sum(
            1 for row in (payload.get("line_items") or []) if (row or {}).get("description")),
        "gate_log": list(variant._pr105b_gate_log) if variant else [],
    }
