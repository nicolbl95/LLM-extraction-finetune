"""HF's isolated requirements mount must retain the validated package pins."""

from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_hf_requirements_are_self_contained():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    entries = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert not any(line.startswith("-") for line in entries)
    assert all(Requirement(line.split(" #", 1)[0]) for line in entries)


def test_hf_requirements_include_every_validated_constraint():
    def requirements(path):
        return [
            Requirement(line.split(" #", 1)[0])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

    installed = requirements(ROOT / "requirements.txt")
    for constraint in requirements(ROOT / "constraints-hf.txt"):
        assert any(
            item.name.lower() == constraint.name.lower()
            and item.specifier == constraint.specifier
            and item.marker == constraint.marker
            for item in installed
        ), f"Missing inline constraint: {constraint}"


def test_hf_system_packages_are_linux_xargs_compatible():
    content = (ROOT / "packages.txt").read_bytes()
    assert b"\r" not in content, "Linux xargs preserves carriage returns in apt package names"
    assert content.endswith(b"\n")
    assert content.decode("ascii").splitlines() == [
        "tesseract-ocr", "tesseract-ocr-fra", "tesseract-ocr-eng",
    ]
