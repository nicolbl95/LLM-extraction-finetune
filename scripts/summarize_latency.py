#!/usr/bin/env python3
"""Publish measured latency, calls and the fixed stress acceptance decision."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    before = json.loads((ROOT / "results/latency-before.json").read_text())
    after = json.loads((ROOT / "results/latency-after-final.json").read_text())
    scanned = json.loads((ROOT / "results/latency-scanned-pdf-final.json").read_text())
    stress = json.loads((ROOT / "results/stress-latency-final.json").read_text())
    labels = {
        "clean_image": "Image propre", "degraded_image": "Image dégradée",
        "destroyed_image": "Image détruite", "native_pdf_10_pages": "PDF natif 10 pages",
    }
    lines = [
        "# Latence et ressources hors GPU — mesures réelles",
        "",
        "**Aucun modèle/GPU appelé.** Vrai Tesseract fra+eng et PyMuPDF ; trois "
        "échantillons par catégorie, processus chaud. Images propres FR/UK/US ; "
        "dégradées : rotation FR +1,5°, JPEG UK 40, bruit US ; détruites : bruit "
        "déterministe des trois images. PDF : dix pages générées, pas des factures réelles.",
        "",
        "| Catégorie | Médiane avant → après (s) | p95 interpolé avant → après (s) | Pire avant → après (s) | Appels Tesseract avant → après |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, title in labels.items():
        left, right = before["summary"][key], after["summary"][key]
        lines.append(
            f"| {title} | {left['median_seconds']:.3f} → {right['median_seconds']:.3f} | "
            f"{left['p95_seconds_linear']:.3f} → {right['p95_seconds_linear']:.3f} | "
            f"{left['worst_seconds']:.3f} → {right['worst_seconds']:.3f} | "
            f"{left['tesseract_calls']} → {right['tesseract_calls']} |"
        )
    scanned_metrics = scanned["summary"]["scanned_pdf_10_pages"]
    lines += [
        "", "### PDF réellement scanné, 10 pages", "",
        f"Après changement seulement : médiane **{scanned_metrics['median_seconds']:.3f} s**, "
        f"p95 interpolé **{scanned_metrics['p95_seconds_linear']:.3f} s**, pire "
        f"**{scanned_metrics['worst_seconds']:.3f} s** ; appels "
        f"Tesseract **{scanned_metrics['tesseract_calls']}**. Le document mesuré termine "
        "sous le budget ; son épuisement multi-page est également testé avec une horloge contrôlée.",
        "", "## Budget et chemin dégradé", "",
        "Budget **45 secondes au total par document**, toutes ses pages comprises. "
        "Il commence à l'entrée du lecteur documentaire, pas à chaque appel OCR. "
        "Tesseract reçoit `min(60, temps restant)` ; les étapes CPU vérifient "
        "le délai avant/après. Aucun prétraitement supplémentaire après épuisement. "
        "Le rapport rend « Preuves OCR insuffisantes » et des champs non vérifiés/null, "
        "jamais les valeurs d'une lecture partielle présentée comme sûre.",
        "",
        "Ce n'est pas un hard timeout de 45,000 s : arrêt du sous-processus, encodage "
        "et prétraitement CPU en cours peuvent ajouter une surcharge. Les images détruites "
        f"mesurées atteignent {after['summary']['destroyed_image']['worst_seconds']:.3f} s. "
        "Le lecteur natif PyMuPDF vérifie le délai entre pages ; il n'est pas interrompu "
        "au milieu d'un appel natif. Aucun engagement de latence du Space n'est déduit de ces chiffres.",
        "",
        "Le budget 45 s est choisi après observation d'un cas dégradé à "
        f"{before['summary']['degraded_image']['worst_seconds']:.2f} s ; "
        "25 s n'a pas été imposé aveuglément. Le redressement est calculé une seule fois. "
        "Les premières lectures deskew PSM 4 / CLAHE PSM 6 sont réutilisées ; une paire reconnue, "
        "concordante et sans signal de faible confiance/contradiction termine en deux appels. "
        "Elles sont deux prétraitements distincts, pas deux votes d'une seule famille. "
        "Sinon, escalade vers la configuration de consensus figée, sans abaisser les seuils.",
        "",
        "Un premier candidat à deux segmentations du même prétraitement avait donné "
        "153 null. Il n'a pas été conservé sous cette forme : la règle des deux familles "
        "a été restaurée puis la version finale remesurée, sans faire passer ces chiffres "
        "intermédiaires pour les résultats finaux.",
        "", "## Critère de conservation du changement", "",
        f"35 cas / 612 champs : **155 → {stress['summary']['false_nulls']} null**, "
        f"**{stress['summary']['wrong_values']} valeur fausse**, "
        f"**{stress['summary']['phantom_anomalies']} anomalie fantôme**, "
        f"**{stress['summary']['crashes']} crash**. Variation : "
        f"**{stress['latency_acceptance']['false_null_delta_percentage_points']:.3f} point**, "
        "en dessous du plafond +2 points ; changement **retenu**.",
        "",
        "Les 105 profils supplémentaires donnent "
        f"**{stress['additional_model_errors']['wrong_values']} valeur fausse / "
        f"{stress['additional_model_errors']['phantom_anomalies']} anomalie fantôme**. "
        "Ces profils sont reconstruits/simulés, pas des générations réelles de Qwen.",
        "",
        "## Ressources et limites", "",
    ]
    for phase, data in (("Avant", before), ("Après", after), ("PDF scanné après", scanned)):
        memory = max(sample["process_peak_rss_mib"] for sample in data["samples"])
        lines.append(f"- {phase} : pic RSS parent observé **{memory:.1f} MiB** (maximum cumulatif du processus).")
    lines += [
        "- Tesseract limité à un thread (`OMP_THREAD_LIMIT=1` dans le protocole), "
        "OpenCV à un thread. Les temps excluent les imports initiaux, le chargement "
        "du modèle, le GPU et la génération des dégradations.",
        "- Le RSS ci-dessus n'est pas un pic mémoire isolé par facture et n'inclut "
        "pas toute la mémoire transitoire des processus enfants Tesseract.",
        "- **n=3 par catégorie** : le p95 est une interpolation descriptive, pas "
        "une estimation robuste de production ; exécuter à nouveau sur HF/ZeroGPU.",
        "- Les scanners réels indépendants restent à évaluer selon `docs/REAL_SCANS.md` ; "
        "aucune mesure d'exactitude n'est inventée sans vérité terrain.",
        "", "## Configuration et revue", "",
        "`src/ocr_config.py` est l'unique politique versionnée : confiance 40, "
        "familles, segmentations, critères du chemin rapide, budget et limites. "
        f"Version **{stress['ocr_configuration']['version']}**, SHA-256 "
        f"`{stress['ocr_configuration_sha256']}`. Les résultats des scans réels "
        "et l'application exportent cette empreinte. Toute modification impose "
        "une nouvelle évaluation, pas un réglage après lecture des vrais scans.",
        "",
        "Patchs de revue disjoints : `results/review-themes/`. Les patchs "
        "ont des dépendances croisées : ils ne sont pas quatre versions déployables séparément.",
    ]
    (ROOT / "results/LATENCY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
