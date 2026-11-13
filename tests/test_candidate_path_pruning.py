"""Pruned optional examples must not break stale-upload validation."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app import main


def test_checked_path_skips_missing_pruned_example_files():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stale_upload = root / "expired-upload.png"
        absent_example = root / "pruned-example.png"
        with patch.object(main.gr.utils, "get_upload_folder", return_value=str(root / "cache")), \
                patch.object(main, "REGIONAL_EXAMPLES", []), \
                patch.object(main, "ADDITIONAL_EXAMPLES", [("pruned", absent_example)]):
            with pytest.raises(ValueError, match="fichier n'est plus disponible"):
                main.checked_path(str(stale_upload))
