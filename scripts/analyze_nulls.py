#!/usr/bin/env python3
"""Attribute baseline nulls without changing acceptance thresholds or rewriting data."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CAUSES = {
    "unreadable": "1. Aucun token OCR exploitable",
    "confidence": "2. Confiance sous le seuil",
    "disagreement": "3. Segmentations en désaccord",
    "table": "4. Tableau/ligne non reconstruite",
    "constraint": "5. Contrainte sans solution unique",
    "other": "6. Autre / garde-fou documentaire",
}


def field_group(path):
    if path.startswith("line_items."):
        return "lignes." + path.split(".")[-1]
    return path


def primary_cause(case):
    issues = " ".join(case.get("quality_issues", [])).casefold()
    if not case.get("ocr_lines"):
        return "unreadable"
    if "confiance" in issues or case.get("ocr_mean_confidence", 100) < 85:
        return "confidence"
    if "concordent" in issues or "incohérent entre" in issues:
        return "disagreement"
    if not case.get("table_rows") or "cellules" in issues:
        return "table"
    if "interprétations" in issues or "combinaisons" in issues:
        return "constraint"
    return "other"


def write_breakdown(report, output):
    causes, fields, degradations, families = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    cause_fields = defaultdict(Counter)
    for case in report["cases"]:
        cause = primary_cause(case)
        family, degradation = case["case"].split("/", 1)
        degradation = "rotation" if degradation.startswith("rotate:") else degradation
        families[family]["expected"] += len(case["statuses"])
        families[family]["null"] += len(case["false_nulls"])
        families[family]["cases"] += 1
        for path in case["false_nulls"]:
            group = field_group(path)
            causes[cause] += 1
            fields[group] += 1
            cause_fields[cause][group] += 1
            degradations[degradation][group] += 1
    expected = sum(value["expected"] for value in families.values())
    nulls = sum(causes.values())
    lines = [
        "# Diagnostic des null à tort — référence figée avant réglage",
        "",
        f"**{nulls} null sur {expected} champs attendus, {len(report['cases'])} cas.** "
        "Seuils inchangés au moment du diagnostic : moyenne 85/100, token 60/100.",
        "",
        "Les **600 champs** correspondaient aux 34 cas antérieurs ; l'ajout de la "
        "remise monétaire séparée ajoute 12 champs. Le jeu fixe de 35 cas comporte **612 champs**.",
        "",
        "| Famille | Cas | Champs attendus | Null à tort |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, counts in families.items():
        lines.append(f"| {family} | {counts['cases']} | {counts['expected']} | {counts['null']} |")
    lines += ["", "## Cause principale exclusive", "",
              "Chaque null hérite de la première cause bloquante, dans l'ordre 1 → 6 ci-dessous. "
              "Il s'agit de l'attribution du refus du document entier, **pas** de la preuve que chaque champ "
              "était illisible. Les causes se recouvrent ; les messages bruts restent dans le JSON.",
              "", "| Cause | Null attribués |", "| --- | ---: |"]
    for cause, title in CAUSES.items():
        lines.append(f"| {title} | {causes[cause]} |")
    lines += ["", "### Trois causes principales", ""]
    lines += [f"- **{CAUSES[cause]} : {count} null**." for cause, count in causes.most_common(3)]
    all_fields = sorted(fields)
    lines += ["", "## Par dégradation et champ", "",
              "| Dégradation | " + " | ".join(all_fields) + " | Total |",
              "| --- | " + " | ".join("---:" for _ in all_fields) + " | ---: |"]
    for degradation, counts in sorted(degradations.items()):
        lines.append("| " + degradation + " | " + " | ".join(str(counts[field]) for field in all_fields)
                     + f" | {sum(counts.values())} |")
    lines += ["", "| Cause | " + " | ".join(all_fields) + " |",
              "| --- | " + " | ".join("---:" for _ in all_fields) + " |"]
    for cause, title in CAUSES.items():
        lines.append("| " + title + " | " + " | ".join(str(cause_fields[cause][field]) for field in all_fields) + " |")
    lines += ["", "## Décision préalable", "",
              "Ne pas baisser les seuils : beaucoup de valeurs justes sont annulées par un garde-fou "
              "**global**, surtout lorsque le tableau n'est pas reconstruit. Priorité au redressement "
              "et au recoupement par champ ; mesurer les prétraitements sans utiliser les références "
              "comme source de valeurs. Une courbe de seuil sera descriptive, et le choix ne pourra "
              "utiliser que la partition de réglage.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/stress-before-recovery.json"))
    parser.add_argument("--output", type=Path, default=Path("results/null-breakdown.md"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    write_breakdown(report, args.output)


if __name__ == "__main__":
    main()
