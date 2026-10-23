#!/usr/bin/env python3
"""Review-only disjoint patches; never stage, commit, tag or push."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEMES = {
    "01-ancrage": "Preuves, ancrage, scoring et tests associés.",
    "02-rapport-documentaire": "Application Gradio, rapport documentaire et tests de présentation.",
    "03-stress-ocr": "Qualification, configuration, délais OCR, dépendances et stress.",
    "04-scripts-notebook": "Outils de mesure/validation réelle, Colab, protocole et audit.",
    "05-dependances": "Version de l'environnement Space, Pillow/OpenCV/NumPy, packages et exclusions.",
}


def theme(path):
    if path == ".gitignore" or path.startswith("requirements") or path == "docs/DEPLOY.md" or path == "results/PILLOW-VALIDATION.md":
        return "05-dependances"
    if path.startswith("results/"):
        if path in {
            "results/LATENCY.md", "results/STRESS-SUMMARY.md",
            "results/null-breakdown.md", "results/ocr-experiments.md",
        }:
            return "03-stress-ocr"
        raise ValueError(f"Raw/generated result excluded from commit plan: {path}")
    if path.startswith("scripts/") or path.startswith("notebooks/") or path in {
        "docs/REAL_SCANS.md", "docs/TEST_CHANGES.md",
        "src/diagnose_pipeline.py", "tests/test_real_scans.py", "tests/test_grounding_slow.py",
    }:
        return "04-scripts-notebook"
    if path.startswith("src/ocr_") or path.startswith("tests/stress/") or path in {
        "pytest.ini", "tests/test_ocr_budget.py",
    }:
        return "03-stress-ocr"
    if path.startswith("app/") or path in {
        "README.md", "docs/DESIGN.md", "src/invoice_validation.py", "tests/test_document_report.py", "tests/test_app.py",
    }:
        return "02-rapport-documentaire"
    if path.startswith(("src/", "data/", "tests/")):
        return "01-ancrage"
    raise ValueError(f"Unclassified review file: {path}")


def generate(output):
    tracked = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True,
    ).splitlines()
    files = sorted(set(tracked + untracked))
    files = [path for path in files if path == ".gitignore" or path.endswith(
        (".py", ".md", ".ipynb", ".json", ".ini", ".txt")
    )]
    output.mkdir(parents=True, exist_ok=True)
    chunks = {key: [] for key in THEMES}
    ownership = {key: [] for key in THEMES}
    for path in files:
        if path in untracked:
            command = ["git", "diff", "--no-index", "--binary", "--", "/dev/null", path]
        else:
            command = ["git", "diff", "--binary", "HEAD", "--", path]
        result = subprocess.run(command, cwd=ROOT, capture_output=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode())
        key = theme(path)
        chunks[key].append(result.stdout)
        ownership[key].append(path)
    manifest = {"base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "themes": {}}
    for key, pieces in chunks.items():
        content = b"".join(pieces)
        target = output / f"{key}.patch"
        target.write_bytes(content)
        checked = subprocess.run(["git", "apply", "--check", "--reverse", str(target)], cwd=ROOT, capture_output=True)
        if checked.returncode:
            raise RuntimeError(f"Patch does not match worktree: {target}: {checked.stderr.decode()}")
        manifest["themes"][key] = {"description": THEMES[key], "files": ownership[key],
                                   "sha256": hashlib.sha256(content).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Patchs de revue par thème\n\n"
        "Ces cinq patchs sont disjoints par fichier et comparent HEAD à l'arbre local complet. "
        "Ils sont destinés à la revue, **pas à cinq déploiements indépendants** : des imports "
        "croisés exigent l'ensemble. L'ordre conseillé est dépendances, ancrage, rapport, OCR/stress, scripts/notebook. "
        "Appliquer dans cet ordre sur le commit de base indiqué "
        "dans `manifest.json`, puis tester. Aucun commit/push n'est exécuté par ce générateur.\n\n"
        + "\n".join(f"- [{key}.patch](./{key}.patch) — {description}" for key, description in THEMES.items())
        + "\n\nSeuls les rapports Markdown sélectionnés sont inclus ; logs, caches, JSON bruts et poids restent exclus.\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/review-themes/five-commit-plan")
    args = parser.parse_args()
    manifest = generate(args.output_dir)
    print(json.dumps({"base": manifest["base"], "files_by_theme": {
        name: len(value["files"]) for name, value in manifest["themes"].items()
    }}))


if __name__ == "__main__":
    main()
