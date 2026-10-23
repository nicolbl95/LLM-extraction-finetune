#!/usr/bin/env python3
"""Publish before/after decisions with exact fixed-case denominators."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline = json.loads((ROOT / "results/stress-before-recovery.json").read_text())
    selection = json.loads((ROOT / "results/ocr-selection.json").read_text())
    split = json.loads((ROOT / "results/ocr-split.json").read_text())["cases"]
    tuning = json.loads((ROOT / "results/ocr-tuning-experiments.json").read_text())
    validation = json.loads((ROOT / "results/ocr-validation-experiments.json").read_text())
    curve = json.loads((ROOT / "results/ocr-threshold-curve.json").read_text())
    final = json.loads((ROOT / "results/stress-report.json").read_text())
    lines = [
        "# Expériences de récupération OCR — décisions et résultats",
        "",
        "Les références ne sont utilisées que pour le score. Aucun chiffre final n'est choisi "
        "par consultation de la référence. La sélection est figée dans `ocr-selection.json` "
        "avant l'exécution sur les 14 cas de validation.",
        "",
        "## Dénominateurs exacts",
        "",
        "| Famille | Cas | Champs attendus | Avant : null | Après : null | Faux gardés | Anomalies fantômes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ("france", "uk", "us", "layout"):
        before = [record for record in baseline["cases"] if record["case"].startswith(family + "/")]
        after = [record for record in final["cases"] if record["case"].startswith(family + "/")]
        fields = sum(len(record["statuses"]) for record in before)
        lines.append(
            f"| {family} | {len(before)} | {fields} | {sum(len(r['false_nulls']) for r in before)} | "
            f"{sum(len(r['false_nulls']) for r in after)} | {sum(len(r['wrong_values']) for r in after)} | "
            f"{sum(len(r['phantom_anomalies']) for r in after)} |"
        )
    lines += [
        "", "**35 cas = 612 champs** (24 images = 480 ; 11 layouts = 132). "
        "L'ancien chiffre 600 correspondait aux 34 cas, avant l'ajout d'un layout à 12 champs.",
        "", "## Essais isolés — aucun abaissement de seuil", "",
        "| Essai | Réglage (372 champs) : null avant → après | Validation (240 champs) : null avant → après | Faux / fantômes | Décision sur réglage |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    names = {"deskew": "Redressement géométrique", "otsu": "Otsu", "adaptive": "Seuil local",
             "denoise": "Débruitage médian", "clahe": "CLAHE", "scale3": "Agrandissement ×3",
             "crop": "Recadrage tableau / PSM 6"}
    for method, name in names.items():
        left, right = tuning[method]["summary"], validation[method]["summary"]
        decision = (
            "Gardé comme famille de consensus" if method in {"deskew", "clahe", "crop"} else
            "Rejeté seul ; gardé comme preuve contradictoire / support régional" if method in {"denoise", "scale3"} else
            "Rejeté"
        )
        lines.append(
            f"| {name} | 252 → {left['false_nulls']} | 164 → {right['false_nulls']} | "
            f"{left['wrong_values'] + right['wrong_values']} / {left['phantom_anomalies'] + right['phantom_anomalies']} | {decision} |"
        )
    lines += [
        "", "## Consensus progressif sur réglage seulement", "",
        "| Configuration | Null (372 champs) | Faux conservés | Décision |",
        "| --- | ---: | ---: | --- |",
        "| deskew + CLAHE + crop, premier consensus | 93 | 1 | **Rejeté** : `Ciavier sans fil` sur JPEG FR |",
        "| Ajout débruitage, veto textuel contradictoire, seuil 60 | 100 | 0 | Gardé comme garde-fou, pas un vote majoritaire |",
        "| Ajout ×3 comme support régional, seuil 60 | 95 | 0 | Gardé ; ne rend pas la demi-résolution lisible |",
        "| Configuration figée, seuil 40 | 94 | 0 | Retenue avant validation |",
        "", "Le crop change la région mais ne constitue pas une preuve statistiquement indépendante du deskew. "
        "Le consensus exige deux familles de transformation et rejette les variantes de texte contradictoires ; "
        "cela ne garantit pas l'absence d'une erreur commune au même moteur Tesseract.",
        "", "## Courbe du seuil — réglage uniquement", "",
        "| Confiance minimale | Champs justes exportés | Null | Faux acceptés | Anomalies fantômes |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in curve:
        correct = point["fields"] - point["false_nulls"] - point["wrong_values"]
        lines.append(
            f"| {point['threshold']} | {correct} | {point['false_nulls']} | {point['wrong_values']} | {point['phantom_anomalies']} |"
        )
    lines += [
        "", "**40 est le plus bas seuil testé sûr sur le réglage**, pas une limite universelle. "
        "La légère non-monotonie provient des vetos : une lecture supplémentaire peut introduire "
        "un désaccord et retirer une valeur, ce qui est voulu. Le seuil moyen global n'est pas "
        "abaissé : il est remplacé par le consensus régional pour ce chemin.",
        "", "## Résultats par partition", "",
        "| Partition | Cas | Champs | Null avant | Null après | Faux conservés | Fantômes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for partition in ("tuning", "validation"):
        before = [r for r in baseline["cases"] if split[r["case"]] == partition]
        after = [r for r in final["cases"] if split[r["case"]] == partition]
        lines.append(
            f"| {partition} | {len(before)} | {sum(len(r['statuses']) for r in before)} | "
            f"{sum(len(r['false_nulls']) for r in before)} | {sum(len(r['false_nulls']) for r in after)} | "
            f"{sum(len(r['wrong_values']) for r in after)} | {sum(len(r['phantom_anomalies']) for r in after)} |"
        )
    lines += [
        "", "### Images dégradées seulement", "",
        "| Partition | Images | Champs image | Null après | Taux de null image |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for partition in ("tuning", "validation"):
        images = [r for r in final["cases"] if split[r["case"]] == partition and not r["case"].startswith("layout/")]
        fields = sum(len(r["statuses"]) for r in images)
        nulls = sum(len(r["false_nulls"]) for r in images)
        lines.append(f"| {partition} | {len(images)} | {fields} | {nulls} | {nulls / fields:.2%} |")
    lines += [
        "", "La partition est 21/14 cas (60/40), stratifiée par famille, seed 62419. "
        "Les proportions de **champs** sont 372/240, pas exactement 60/40 car les factures ont des longueurs différentes. "
        "Les cas avaient été vus dans la passe précédente : c'est une validation d'ingénierie figée, "
        "**pas un vrai jeu externe aveugle**. Aucun réglage n'a été modifié après cette validation.",
        "", "## Limites et objectif non atteint", "",
        "Les erreurs communes aux prétraitements, descriptions tronquées et demi-résolution restent problématiques. "
        "Les trois images complètement détruites comptent déjà 60 champs impossibles à récupérer (12,5 % des champs image). "
        "Les trois demi-résolutions représentent 60 autres champs sans tableau lisible. "
        "Ces deux groupes imposent à eux seuls 25 % de null image avant toute autre incertitude.",
        "", "Le seuil cible de 30 % n'autorise pas à deviner les champs restants. "
        "Les refus et coûts de latence sont conservés et publiés. Les anomalies d'une image reconstruite "
        "par consensus ne sont pas affirmées comme des erreurs documentaires certaines sans confirmation.",
        "", "## Erreurs modèle supplémentaires", "",
        "Les profils `reported`, `coherent`, `permuted` rejouent les erreurs Qwen décrites par l'utilisateur "
        "et des triplets arithmétiquement cohérents mais absents du document. Ce ne sont **pas des générations "
        "capturées localement**. Le JSON détaillé final donne le résultat des 105 profils supplémentaires.",
        "", "Sélection figée : " + json.dumps(selection, ensure_ascii=False), "",
    ]
    (ROOT / "results/ocr-experiments.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
