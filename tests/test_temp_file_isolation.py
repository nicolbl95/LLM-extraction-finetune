"""Temporary files must not dirty the checkout, and dirtiness must not change results.

PR135 phase 2: the revision label rendered into the report came from ``git status``, so a test
that wrote its scratch directory inside the worktree changed the report text of a later run.
"""

import ast
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import main

ROOT = Path(__file__).resolve().parents[1]
CLEAN_REVISION = "0e40a0ec680abe9c3d217aca7c5697bbd22de0e1"
DIRTY_SUFFIX = " (modifications locales non commitées)"
_TEMPORARY_CALLS = {"TemporaryDirectory", "mkdtemp", "NamedTemporaryFile", "mkstemp"}


def _callee_name(func):
    if isinstance(func, (ast.Attribute, ast.Name)):
        return func.attr if isinstance(func, ast.Attribute) else func.id
    return None


def _worktree_base(node):
    if isinstance(node, ast.Name) and node.id == "ROOT":
        return ROOT
    if (isinstance(node, ast.Attribute) and node.attr == "ROOT"
            and isinstance(node.value, ast.Name) and node.value.id == "main"):
        return main.ROOT
    return None


def _declared_temporary_roots():
    """Every temporary root tests/ pins to an explicit worktree path."""
    sites = []
    for module in sorted((ROOT / "tests").glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _callee_name(node.func) not in _TEMPORARY_CALLS:
                continue
            keyword = next((k for k in node.keywords if k.arg == "dir"), None)
            if keyword is None:
                continue
            value = keyword.value
            if isinstance(value, ast.Call) and _callee_name(value.func) == "str":
                value = value.args[0]
            if (isinstance(value, ast.BinOp) and _worktree_base(value.left) is not None
                    and isinstance(value.right, ast.Constant) and isinstance(value.right.value, str)):
                sites.append((module.relative_to(ROOT).as_posix(), node.lineno,
                              _worktree_base(value.left) / value.right.value))
            else:
                sites.append((module.relative_to(ROOT).as_posix(), node.lineno, None))
    return sites


def _is_ignored_by_git(path: Path) -> bool:
    probe = path / "pr135-isolation-probe.txt"
    return subprocess.run(
        ["git", "check-ignore", "-q", str(probe)], cwd=ROOT, capture_output=True,
    ).returncode == 0


def test_tests_only_pin_temporary_roots_that_git_cannot_see():
    sites = _declared_temporary_roots()
    assert sites, "the scanner found no pinned temporary root; it stopped guarding anything"
    offenders = []
    for name, line, path in sites:
        if path is None:
            offenders.append(f"{name}:{line} uses a location this guard cannot resolve")
        elif _is_ignored_by_git(path):
            continue
        else:
            offenders.append(f"{name}:{line} writes {path} into the worktree and Git reports it")
    assert not offenders, "\n".join(offenders)


def test_pinned_scratch_roots_are_created_by_committed_configuration():
    """.pytest_cache is Git-ignored, so a fresh clone only has it because the session created it."""
    import conftest

    roots = {path for _, _, path in _declared_temporary_roots() if path is not None}
    assert roots == {conftest.SCRATCH_ROOT}, (
        f"roots pinned by tests: {roots}; root created by conftest: {conftest.SCRATCH_ROOT}")


def _porcelain() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
    ).stdout


def test_download_json_outside_the_worktree_leaves_git_status_unchanged():
    before = _porcelain()
    with tempfile.TemporaryDirectory(prefix="pr135-upload-") as directory, \
            patch.dict(os.environ, {"GRADIO_TEMP_DIR": directory}):
        created = main._download_json({"supplier_name": None, "line_items": []})
        assert Path(created).resolve().is_relative_to(Path(directory).resolve())
        assert Path(created).read_text(encoding="utf-8")
        assert _porcelain() == before
    assert _porcelain() == before


PAYLOAD = {
    "document_type": "invoice",
    "supplier_name": None,
    "invoice_number": None,
    "invoice_date": None,
    "currency": "EUR",
    "total_ht": None,
    "line_items": [],
}
_FRAMES = [(None, main.PROCESSING_STATUS), (PAYLOAD, "Lecture terminée.")]


def _frames(revision: str, call):
    diagnostics = {
        "evidence_available": True,
        "source_kind": "consensus",
        "ocr_lines": 12,
        "ocr_calls": 4,
        "ocr_elapsed_seconds": 1.25,
        "preprocessing_used": ["grayscale"],
        "field_statuses": {"supplier_name": "valide", "total_ht": "non vérifié"},
        "model_info": {"used": False, "passes_requested": 0, "passes_succeeded": 0,
                       "reason": "quota", "determinism": "répétabilité non mesurée"},
        "first_pass_raw": "",
        "validation": {"flags": [], "passed": [], "not_verified": []},
        "document_coverage": {"documented_rows": 0},
    }
    with tempfile.TemporaryDirectory(prefix="pr135-report-") as directory, patch.object(
        main, "run_extraction", lambda *a, **k: iter(_FRAMES),
    ), patch.object(main, "deployed_revision", return_value=revision), patch.object(
        main, "ocr_availability", return_value=(True, "ok"),
    ), patch.object(
        main, "ocr_language_plan",
        return_value={"requested": ["fra"], "effective": ["fra"], "missing": []},
    ), patch.dict(os.environ, {"GRADIO_TEMP_DIR": directory}):
        frames = []
        for frame in call(diagnostics):
            download = Path(frame[-1]).read_text(encoding="utf-8") if frame[-1] else None
            frames.append((
                json.dumps(frame[0], sort_keys=True) if frame[0] is not None else None,
                *frame[1:-1], json.loads(download) if download else None,
            ))
        return frames


def test_full_report_is_identical_in_a_clean_and_a_dirty_checkout():
    full_report = lambda diagnostics: main.run_extraction_ui("document.png", diagnostics)
    payload_clean, report_clean, details_clean, advanced_clean, raw_clean, validation_clean, download_clean = \
        _frames(CLEAN_REVISION, full_report)[1]
    payload_dirty, report_dirty, details_dirty, advanced_dirty, raw_dirty, validation_dirty, download_dirty = \
        _frames(CLEAN_REVISION + DIRTY_SUFFIX, full_report)[1]

    assert payload_clean == payload_dirty
    assert (report_clean, raw_clean, validation_clean) == (report_dirty, raw_dirty, validation_dirty)
    assert details_clean == details_dirty
    assert download_clean == download_dirty == PAYLOAD

    clean_lines, dirty_lines = advanced_clean.splitlines(), advanced_dirty.splitlines()
    assert len(clean_lines) == len(dirty_lines)
    changed = [
        (index, before, after) for index, (before, after) in enumerate(zip(clean_lines, dirty_lines))
        if before != after
    ]
    assert len(changed) == 1
    index, before, after = changed[0]
    assert clean_lines[index].startswith("- Révision déployée")
    assert before == f"- Révision déployée : `{CLEAN_REVISION}`"
    assert after == f"- Révision déployée : `{CLEAN_REVISION}{DIRTY_SUFFIX}`"


def test_recruiter_demo_ignores_checkout_dirtiness():
    recruiter = lambda _: main.run_recruiter_extraction_ui("document.png")
    clean = _frames(CLEAN_REVISION, recruiter)
    dirty = _frames(CLEAN_REVISION + DIRTY_SUFFIX, recruiter)

    assert [frame[0] for frame in clean] == [frame[0] for frame in dirty]
    assert [frame[1] for frame in clean] == [frame[1] for frame in dirty]
    assert clean[0][2] == dirty[0][2] == "### Performances"
    assert [frame[-1] for frame in clean] == [frame[-1] for frame in dirty] == [None, PAYLOAD]
