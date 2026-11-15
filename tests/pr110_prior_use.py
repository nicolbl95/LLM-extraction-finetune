"""PR169: the prior-use inventory the eligibility screen reads, supplied per test instead of from disk.

`pr110_eligibility.eligibility()` bound PR103's published inventory as an import-time default, so it could
not be redirected and every test that ran the screen - the intake gate, the coverage quota, the freeze, the
evaluator's re-checks - had to find that private artefact on disk. 48 of them failed in a clean clone while
the rule they measure is machine-independent. The default is now resolved per call, and the fixture below
points it at an inventory written in PR103's own format.

Nothing is simulated: an entry carries the real digest of a tracked public file, so `previous_use` blocks
for the reason the production code states, and a document absent from the inventory is admitted because the
record genuinely does not know it. The document authority is left to each module's own fixture - three of
them already declare `AU.empty()` - and tests that are about this machine's private artefacts read them
directly and skip with the missing path named.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def write(tmp_path: Path, used: tuple[Path, ...] = ()) -> Path:
    """An inventory of past use in PR103's format, naming exactly `used` and nothing else.

    `regression_fixture` is PR103's namespace for a file the development corpus already used, which is what
    makes the screen block it; an empty `used` is what makes the screen report a candidate as new to the
    campaign. Either way the recorded digest is the one the file has on disk.
    """
    import pr109_intake as I

    payload = {
        "review": "PR169 prior-use fixture",
        "verdict": "fixture",
        "note": "a test fixture in PR103's format, not PR103's published inventory",
        "files": [{"path": display(path), "sha256": I.sha256_file(path), "purpose": "regression_fixture",
                   "byte_identical_to_a_used_document": []} for path in used],
        "used_document_hashes": [],
    }
    path = tmp_path / "prior-use-inventory.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def install(monkeypatch, tmp_path: Path, *, used: tuple[Path, ...] = ()) -> Path:
    """Point the screen's prior-use authority at the fixture for the duration of one test."""
    import pr109_intake as I

    inventory = write(tmp_path, used)
    monkeypatch.setattr(I, "PR103_INVENTORY", inventory)
    return inventory
