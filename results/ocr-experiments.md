# Expériences de récupération OCR — décisions et résultats

Les références ne sont utilisées que pour le score. Aucun chiffre final n'est choisi par consultation de la référence. La sélection est figée dans `ocr-selection.json` avant l'exécution sur les 14 cas de validation.

## Dénominateurs exacts

| Famille | Cas | Champs attendus | Avant : null | Après : null | Faux gardés | Anomalies fantômes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| france | 8 | 160 | 160 | 63 | 0 | 0 |
| uk | 8 | 128 | 112 | 43 | 0 | 0 |
| us | 8 | 192 | 144 | 49 | 0 | 0 |
| layout | 11 | 132 | 0 | 0 | 0 | 0 |

**35 cas = 612 champs** (24 images = 480 ; 11 layouts = 132). L'ancien chiffre 600 correspondait aux 34 cas, avant l'ajout d'un layout à 12 champs.

## Essais isolés — aucun abaissement de seuil

| Essai | Réglage (372 champs) : null avant → après | Validation (240 champs) : null avant → après | Faux / fantômes | Décision sur réglage |
| --- | ---: | ---: | --- | --- |
| Redressement géométrique | 252 → 213 | 164 → 124 | 0 / 0 | Gardé comme famille de consensus |
| Otsu | 252 → 276 | 164 → 180 | 0 / 0 | Rejeté |
| Seuil local | 252 → 300 | 164 → 180 | 0 / 0 | Rejeté |
| Débruitage médian | 252 → 252 | 164 → 156 | 0 / 0 | Rejeté seul ; gardé comme preuve contradictoire / support régional |
| CLAHE | 252 → 212 | 164 → 136 | 0 / 0 | Gardé comme famille de consensus |
| Agrandissement ×3 | 252 → 260 | 164 → 129 | 0 / 0 | Rejeté seul ; gardé comme preuve contradictoire / support régional |
| Recadrage tableau / PSM 6 | 252 → 169 | 164 → 156 | 0 / 0 | Gardé comme famille de consensus |

## Consensus progressif sur réglage seulement

| Configuration | Null (372 champs) | Faux conservés | Décision |
| --- | ---: | ---: | --- |
| deskew + CLAHE + crop, premier consensus | 93 | 1 | **Rejeté** : `Ciavier sans fil` sur JPEG FR |
| Ajout débruitage, veto textuel contradictoire, seuil 60 | 100 | 0 | Gardé comme garde-fou, pas un vote majoritaire |
| Ajout ×3 comme support régional, seuil 60 | 95 | 0 | Gardé ; ne rend pas la demi-résolution lisible |
| Configuration figée, seuil 40 | 94 | 0 | Retenue avant validation |

Le crop change la région mais ne constitue pas une preuve statistiquement indépendante du deskew. Le consensus exige deux familles de transformation et rejette les variantes de texte contradictoires ; cela ne garantit pas l'absence d'une erreur commune au même moteur Tesseract.

## Courbe du seuil — réglage uniquement

| Confiance minimale | Champs justes exportés | Null | Faux acceptés | Anomalies fantômes |
| ---: | ---: | ---: | ---: | ---: |
| 95 | 173 | 199 | 0 | 0 |
| 85 | 267 | 105 | 0 | 0 |
| 75 | 278 | 94 | 0 | 0 |
| 60 | 277 | 95 | 0 | 0 |
| 50 | 278 | 94 | 0 | 0 |
| 40 | 278 | 94 | 0 | 0 |

**40 est le plus bas seuil testé sûr sur le réglage**, pas une limite universelle. La légère non-monotonie provient des vetos : une lecture supplémentaire peut introduire un désaccord et retirer une valeur, ce qui est voulu. Le seuil moyen global n'est pas abaissé : il est remplacé par le consensus régional pour ce chemin.

## Résultats par partition

| Partition | Cas | Champs | Null avant | Null après | Faux conservés | Fantômes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tuning | 21 | 372 | 252 | 94 | 0 | 0 |
| validation | 14 | 240 | 164 | 61 | 0 | 0 |

### Images dégradées seulement

| Partition | Images | Champs image | Null après | Taux de null image |
| --- | ---: | ---: | ---: | ---: |
| tuning | 15 | 300 | 94 | 31.33% |
| validation | 9 | 180 | 61 | 33.89% |

La partition est 21/14 cas (60/40), stratifiée par famille, seed 62419. Les proportions de **champs** sont 372/240, pas exactement 60/40 car les factures ont des longueurs différentes. Les cas avaient été vus dans la passe précédente : c'est une validation d'ingénierie figée, **pas un vrai jeu externe aveugle**. Aucun réglage n'a été modifié après cette validation.

## Limites et objectif non atteint

Les erreurs communes aux prétraitements, descriptions tronquées et demi-résolution restent problématiques. Les trois images complètement détruites comptent déjà 60 champs impossibles à récupérer (12,5 % des champs image). Les trois demi-résolutions représentent 60 autres champs sans tableau lisible. Ces deux groupes imposent à eux seuls 25 % de null image avant toute autre incertitude.

Le seuil cible de 30 % n'autorise pas à deviner les champs restants. Les refus et coûts de latence sont conservés et publiés. Les anomalies d'une image reconstruite par consensus ne sont pas affirmées comme des erreurs documentaires certaines sans confirmation.

## Erreurs modèle supplémentaires

Les profils `reported`, `coherent`, `permuted` rejouent les erreurs Qwen décrites par l'utilisateur et des triplets arithmétiquement cohérents mais absents du document. Ce ne sont **pas des générations capturées localement**. Le JSON détaillé final donne le résultat des 105 profils supplémentaires.

Sélection figée : {"selection_partition": "tuning", "split_seed": 62419, "tuning_cases": 21, "validation_cases": 14, "preprocessing_families": ["deskew", "clahe", "crop", "denoise", "scale3"], "confidence_threshold": 40, "threshold_search": [95, 85, 75, 60, 50, 40], "threshold_scope": "lowest tested threshold with zero wrong retained values on tuning; not a universal safety guarantee", "tuning_reference_fields": 372, "tuning_false_nulls_before": 252, "tuning_false_nulls_after": 94, "tuning_wrong_values": 0, "tuning_phantom_anomalies": 0, "rejected_consensus": "Three/four preprocessing families retained Ciavier sans fil on French JPEG. Denoising supplies contradictory text and vetoes that export.", "limitations": ["Baseline cases had already been observed in previous passes; the holdout is a frozen engineering validation partition, not a new external sample.", "Distinct preprocessing families are not statistically independent OCR engines.", "No candidate is chosen using reference values at runtime.", "Validation results must not be used to tune this selection after it is frozen."]}
