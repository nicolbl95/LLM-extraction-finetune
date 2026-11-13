"""PR135 phase 4 — every third-party import of the tracked suite must be declared.

The offline suite is installed from `requirements.txt` (runtime) plus `requirements-dev.txt`
(runner). A test that imports a distribution absent from those files passes on a workstation
that happens to have it and errors in a clean environment, which is exactly how `pytest` itself
stayed undeclared while `README.md` told everyone to run it.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (ROOT, ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "data")
DECLARED_FILES = ("requirements.txt", "requirements-inference.txt", "requirements-dev.txt")
RESEARCH_ONLY = ("rapidocr", "onnxruntime")
MODULE_DISTRIBUTIONS = {
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "fitz": "pymupdf",
    "pymupdf": "pymupdf",
    "huggingface_hub": "huggingface-hub",
    "sklearn": "scikit-learn",
}


def normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def tracked_paths():
    try:
        result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git indisponible : la clôture d'imports ne peut pas être bornée aux tests suivis")
    return [ROOT / line for line in result.stdout.splitlines()]


def tracked_test_paths():
    return [path for path in tracked_paths() if path.relative_to(ROOT).parts[0] == "tests"
            and path.suffix == ".py"]


def first_party_names():
    """Module names the repository itself provides, including the namespace packages it uses."""
    names = set()
    for path in tracked_paths():
        if path.name == "__init__.py":
            names.add(normalize(path.parent.name))
        elif path.suffix == ".py":
            names.add(normalize(path.stem))
        for part in path.relative_to(ROOT).parts[:-1]:
            names.add(normalize(part))
    return names


def declared_distributions():
    distributions = set()
    for name in DECLARED_FILES:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            line = line.split(" #", 1)[0].strip()
            if line and not line.startswith(("#", "-")):
                distributions.add(normalize(line.split("=", 1)[0].split("!", 1)[0].split("<", 1)[0]
                                            .split(">", 1)[0].split("[", 1)[0]))
    return distributions


def imported_modules(paths):
    modules = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_every_third_party_test_import_is_declared():
    third_party = imported_modules(tracked_test_paths()) - set(getattr(sys, "stdlib_module_names", set()))
    first_party = first_party_names()
    declared = declared_distributions()
    undeclared = sorted(
        f"{module} -> {MODULE_DISTRIBUTIONS.get(module, module)}"
        for module in third_party
        if normalize(module) not in first_party
        and normalize(MODULE_DISTRIBUTIONS.get(module, module)) not in declared
    )
    assert third_party, "aucun import tiers repéré : le scan ne garde plus rien"
    assert not undeclared, f"imports tiers non declares dans {DECLARED_FILES}: {undeclared}"


def test_the_test_runner_is_declared_outside_the_space_build():
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").lower()
    assert "pytest" in dev
    assert "packaging" in dev, "tests/test_hf_build_requirements.py importe packaging directement"
    runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "pytest" not in runtime, "le build du Space ne doit pas installer le runner de tests"


def test_research_only_engines_stay_undeclared():
    declared = declared_distributions()
    assert not [name for name in RESEARCH_ONLY if normalize(name) in declared], (
        "rapidocr/onnxruntime sont des dependances de recherche : les declarer les rendrait "
        "obligatoires pour une suite qui ne les importe pas"
    )


def parsed_requirements(path):
    """Every declared requirement of one file, comments and `-r` includes left out."""
    return [
        Requirement(line.split(" #", 1)[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", "-"))
    ]


def test_declared_runtime_and_inference_requirements_are_compatible():
    """The three declared files must install one version per distribution, not three candidates.

    An inference post installs `requirements-inference.txt`, a test workstation adds
    `requirements-dev.txt` and the Space mounts only `requirements.txt`; nothing at install time
    compares them, so a pin edited in one file silently changes the environment being measured.
    """
    inference_lines = [line.strip() for line in
                       (ROOT / "requirements-inference.txt").read_text(encoding="utf-8").splitlines()]
    assert "-r requirements.txt" in inference_lines, (
        "requirements-inference.txt a cesse d'inclure le runtime : un poste d'inférence "
        "installerait un jeu de dépendances différent de celui du Space"
    )
    exact_pins: dict[str, list[tuple[str, str]]] = {}
    for name in DECLARED_FILES:
        for item in parsed_requirements(ROOT / name):
            if str(item.specifier).startswith("=="):
                exact_pins.setdefault(normalize(item.name), []).append(
                    (name, str(item.specifier)))
    conflicts = {distribution: spots for distribution, spots in exact_pins.items()
                 if len({spot[1] for spot in spots}) > 1}
    assert not conflicts, f"pins exacts divergents entre fichiers declares: {conflicts}"

    runtime = parsed_requirements(ROOT / "requirements.txt")
    ranges: dict[str, list] = {}
    for item in runtime:
        bounds = str(item.specifier)
        if bounds and not bounds.startswith("=="):
            ranges.setdefault(normalize(item.name), []).append(item.specifier)

    def pin_version(specifier: str) -> str:
        # `==4.14.0.*` is a prefix pin, not a version: the range has to admit the prefix itself.
        return specifier.removeprefix("==").removesuffix(".*")

    unsatisfied = [
        f"{distribution} {exact} vs {specifier}"
        for distribution, entries in ranges.items()
        for specifier in entries
        for _, exact in exact_pins.get(distribution, [])
        if not specifier.contains(pin_version(exact), prereleases=True)
    ]
    assert not unsatisfied, f"un pin exact contredit une plage du meme fichier: {unsatisfied}"

    torch = next((item for item in runtime if normalize(item.name) == "torch"), None)
    assert torch is not None, "torch n'est plus declare : la fenetre ZeroGPU n'est plus garde fou"
    bounds = {spec.operator: Version(spec.version) for spec in torch.specifier}
    assert bounds.get(">=") is not None and bounds[">="] >= Version("2.8"), (
        f"ZeroGPU rejette torch < 2.8 (fenetre declaree: {torch.specifier})")
    assert bounds.get("<=") is not None and bounds["<="] <= Version("2.13"), (
        f"ZeroGPU rejette torch 2.14+ (fenetre declaree: {torch.specifier})")
