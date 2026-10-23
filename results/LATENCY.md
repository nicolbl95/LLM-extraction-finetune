# Latence et ressources hors GPU — mesures réelles

**Aucun modèle/GPU appelé.** Vrai Tesseract fra+eng et PyMuPDF ; trois échantillons par catégorie, processus chaud. Images propres FR/UK/US ; dégradées : rotation FR +1,5°, JPEG UK 40, bruit US ; détruites : bruit déterministe des trois images. PDF : dix pages générées, pas des factures réelles.

| Catégorie | Médiane avant → après (s) | p95 interpolé avant → après (s) | Pire avant → après (s) | Appels Tesseract avant → après |
| --- | ---: | ---: | ---: | --- |
| Image propre | 11.357 → 2.279 | 15.981 → 11.059 | 16.495 → 12.035 | [12, 12, 12] → [2, 12, 2] |
| Image dégradée | 12.821 → 12.978 | 40.869 → 19.491 | 43.986 → 20.214 | [12, 12, 12] → [12, 12, 12] |
| Image détruite | 157.573 → 45.253 | 158.715 → 45.261 | 158.842 → 45.262 | [10, 10, 10] → [3, 3, 3] |
| PDF natif 10 pages | 0.010 → 0.029 | 0.013 → 0.031 | 0.014 → 0.032 | [0, 0, 0] → [0, 0, 0] |

### PDF réellement scanné, 10 pages

Après changement seulement : médiane **24.893 s**, p95 interpolé **25.606 s**, pire **25.686 s** ; appels Tesseract **[20, 20, 20]**. Le document mesuré termine sous le budget ; son épuisement multi-page est également testé avec une horloge contrôlée.

## Budget et chemin dégradé

Budget **45 secondes au total par document**, toutes ses pages comprises. Il commence à l'entrée du lecteur documentaire, pas à chaque appel OCR. Tesseract reçoit `min(60, temps restant)` ; les étapes CPU vérifient le délai avant/après. Aucun prétraitement supplémentaire après épuisement. Le rapport rend « Preuves OCR insuffisantes » et des champs non vérifiés/null, jamais les valeurs d'une lecture partielle présentée comme sûre.

Ce n'est pas un hard timeout de 45,000 s : arrêt du sous-processus, encodage et prétraitement CPU en cours peuvent ajouter une surcharge. Les images détruites mesurées atteignent 45.262 s. Le lecteur natif PyMuPDF vérifie le délai entre pages ; il n'est pas interrompu au milieu d'un appel natif. Aucun engagement de latence du Space n'est déduit de ces chiffres.

Le budget 45 s est choisi après observation d'un cas dégradé à 43.99 s ; 25 s n'a pas été imposé aveuglément. Le redressement est calculé une seule fois. Les premières lectures deskew PSM 4 / CLAHE PSM 6 sont réutilisées ; une paire reconnue, concordante et sans signal de faible confiance/contradiction termine en deux appels. Elles sont deux prétraitements distincts, pas deux votes d'une seule famille. Sinon, escalade vers la configuration de consensus figée, sans abaisser les seuils.

Un premier candidat à deux segmentations du même prétraitement avait donné 153 null. Il n'a pas été conservé sous cette forme : la règle des deux familles a été restaurée puis la version finale remesurée, sans faire passer ces chiffres intermédiaires pour les résultats finaux.

## Critère de conservation du changement

35 cas / 612 champs : **155 → 155 null**, **0 valeur fausse**, **0 anomalie fantôme**, **0 crash**. Variation : **0.000 point**, en dessous du plafond +2 points ; changement **retenu**.

Les 105 profils supplémentaires donnent **0 valeur fausse / 0 anomalie fantôme**. Ces profils sont reconstruits/simulés, pas des générations réelles de Qwen.

## Ressources et limites

- Avant : pic RSS parent observé **597.7 MiB** (maximum cumulatif du processus).
- Après : pic RSS parent observé **484.8 MiB** (maximum cumulatif du processus).
- PDF scanné après : pic RSS parent observé **815.1 MiB** (maximum cumulatif du processus).
- Tesseract limité à un thread (`OMP_THREAD_LIMIT=1` dans le protocole), OpenCV à un thread. Les temps excluent les imports initiaux, le chargement du modèle, le GPU et la génération des dégradations.
- Le RSS ci-dessus n'est pas un pic mémoire isolé par facture et n'inclut pas toute la mémoire transitoire des processus enfants Tesseract.
- **n=3 par catégorie** : le p95 est une interpolation descriptive, pas une estimation robuste de production ; exécuter à nouveau sur HF/ZeroGPU.
- Les scanners réels indépendants restent à évaluer selon `docs/REAL_SCANS.md` ; aucune mesure d'exactitude n'est inventée sans vérité terrain.

## Configuration et revue

`src/ocr_config.py` est l'unique politique versionnée : confiance 40, familles, segmentations, critères du chemin rapide, budget et limites. Version **ocr-policy-4-bounded-fastpath**, SHA-256 `d8ed5856dd39aec17e8d4caf845d89c77f522625798a52f5144a348ab4dca26f`. Les résultats des scans réels et l'application exportent cette empreinte. Toute modification impose une nouvelle évaluation, pas un réglage après lecture des vrais scans.

Patchs de revue disjoints : `results/review-themes/`. Les patchs ont des dépendances croisées : ils ne sont pas quatre versions déployables séparément.
