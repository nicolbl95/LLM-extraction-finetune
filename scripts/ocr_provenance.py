#!/usr/bin/env python3
"""Language- and engine-bound provenance for research OCR runs.

`ocr-policy-5` has no language field, so its fingerprint cannot tell a full
`fra+eng` read from a silent English-only degradation. Research tooling therefore
binds the languages Tesseract actually loaded, plus the engine identity, into its
own cache keys and into every recorded artifact.

Plan semantics mirror `ocr_language_plan()` in `app/main.py`; the application is
not imported here so benchmark runs stay free of the Gradio dependency.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import pytesseract

from ocr_config import OCR_CONFIG

LOGGER = logging.getLogger(__name__)
REQUESTED_LANGUAGES = ("fra", "eng")
IDENTITY_FIELDS = (
    "reader_languages", "tesseract_cmd", "tesseract_version",
    "tessdata_prefix", "ocr_fingerprint",
)
_LANGUAGES_CACHE: dict[tuple[str, str | None], frozenset[str]] = {}
_ENVIRONMENT_CACHE: dict[tuple[tuple[str, str | None], tuple[str, ...]], dict] = {}


def _installation_key():
    return (str(pytesseract.pytesseract.tesseract_cmd or ""), os.environ.get("TESSDATA_PREFIX"))


def installed_languages(*, refresh: bool = False) -> frozenset[str]:
    key = _installation_key()
    if refresh or key not in _LANGUAGES_CACHE:
        try:
            _LANGUAGES_CACHE[key] = frozenset(pytesseract.get_languages(config=""))
        except (OSError, RuntimeError, ValueError, pytesseract.TesseractError) as exc:
            LOGGER.warning("Tesseract language listing failed (%s: %s)", type(exc).__name__, exc)
            _LANGUAGES_CACHE[key] = frozenset()
    return _LANGUAGES_CACHE[key]


def language_plan(*, requested=REQUESTED_LANGUAGES, refresh: bool = False) -> dict:
    """Same four-way distinction the application uses, computed for this research process."""
    installed = installed_languages(refresh=refresh)
    requested = tuple(requested)
    effective = tuple(language for language in requested if language in installed)
    missing = tuple(language for language in requested if language not in installed)
    return {
        "requested": list(requested),
        "effective": list(effective),
        "missing": list(missing),
        "fallback": bool(missing) and bool(effective),
        "installed": sorted(installed),
        "tessdata_prefix": os.environ.get("TESSDATA_PREFIX") or None,
    }


def require_languages(
    *, requested=REQUESTED_LANGUAGES, require_all: bool = True, refresh: bool = False,
) -> dict:
    """Refuse a measurement whose language environment cannot support what it claims to test."""
    plan = language_plan(requested=requested, refresh=refresh)
    if require_all and plan["missing"]:
        raise RuntimeError(
            "Isolated Tesseract data lacks required languages: "
            f"{', '.join(plan['missing'])} (requested: {', '.join(plan['requested'])}; "
            f"installed: {', '.join(plan['installed']) or 'none'}; "
            f"tessdata: {plan['tessdata_prefix'] or 'system default'}).",
        )
    if not plan["effective"]:
        raise RuntimeError(
            "No usable OCR language: install Tesseract with fra/eng (see packages.txt)."
        )
    return plan


def environment(*, require_all: bool = True, refresh: bool = False) -> dict:
    """Record once per resolved language set what a run can honestly claim to have read with."""
    plan = require_languages(require_all=require_all, refresh=refresh)
    key = (_installation_key(), tuple(plan["installed"]))
    if refresh or key not in _ENVIRONMENT_CACHE:
        _ENVIRONMENT_CACHE[key] = {
            **plan,
            "reader_languages": "+".join(plan["effective"]),
            "tesseract_cmd": str(pytesseract.pytesseract.tesseract_cmd or ""),
            "tesseract_version": str(pytesseract.get_tesseract_version()),
            "ocr_policy": OCR_CONFIG.version,
            "ocr_fingerprint": OCR_CONFIG.fingerprint(),
        }
    return dict(_ENVIRONMENT_CACHE[key])


def identity(env: dict) -> dict:
    """The fields that can change what a read really saw, with path spellings folded together.

    Tesseract opens `dir/sub` and `dir\\sub` as the same directory, so two shells that
    wrote the same installation differently must not end up in two cache namespaces;
    the recorded environment itself stays verbatim for the reader.
    """
    record = {field: env.get(field) for field in IDENTITY_FIELDS}
    for field in ("tesseract_cmd", "tessdata_prefix"):
        if record[field]:
            record[field] = str(Path(record[field]))
    return record


def cache_token(env: dict) -> str:
    return hashlib.sha256(json.dumps(identity(env), sort_keys=True).encode()).hexdigest()[:16]


def cache_key(image_hash: str, method: str, psm: int, env: dict) -> str:
    return hashlib.sha256(
        f"{image_hash}/{method}/{psm}/v2/{cache_token(env)}".encode()
    ).hexdigest()


def reusable(entry: dict, env: dict) -> bool:
    """Serve a cached read only when it records the engine and languages we are about to use."""
    recorded = entry.get("environment")
    if not isinstance(recorded, dict):
        return False
    return identity(recorded) == identity(env)
