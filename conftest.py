"""Session-wide test configuration for a checkout that has never run pytest.

`tests/` pins its scratch directories under `.pytest_cache` so that Git can never report them
(PR135 phase 2); the directory itself is Git-ignored, so a fresh clone does not contain it and
`tempfile.TemporaryDirectory(dir=...)` fails there before any production code is reached.
"""

from pathlib import Path

SCRATCH_ROOT = Path(__file__).resolve().parent / ".pytest_cache"


def pytest_configure(config):
    SCRATCH_ROOT.mkdir(exist_ok=True)
