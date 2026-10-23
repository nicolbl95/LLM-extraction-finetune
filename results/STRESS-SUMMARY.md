# Stress indépendant — récupération des null et limites

## Exécutions réelles

- `python scripts/run_stress_check.py --output results/stress-report.json`
- Passe précédente : **237 tests réussis**. Passe récupération :
  `python -m pytest -q` : **249 réussis, 3 slow exclus, 101 sous-tests réussis**,
  798,20 s dans cet environnement. Résultat de fin explicite conservé dans `ocr-final-tests.log`.
- OCR : vrai Tesseract `fra+eng`, cinq prétraitements, deux segmentations TSV par famille, ×2/×3 ;
  PDF : véritable lecture PyMuPDF, y compris un tableau sur deux pages.
- Modèle : **réponses fausses simulées**, aucun modèle réel chargé/téléchargé.
- Les tests et documents de stress sont séparés dans `tests/stress/` ; aucun
  ajout ou mélange avec `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

## Résultats chiffrés — historique

| Mesure | Avant qualification | Après qualification |
| --- | ---: | ---: |
| Cas | 34 | 35 |
| Valeurs fausses conservées | 8 | **0** |
| Anomalies fantômes erreur/avertissement | 6 | **0** |
| `null` à tort | 293 | **416** |
| Crashs | 0 | **0** |

La colonne finale ajoute le cas « remise monétaire séparée » : les dénominateurs
ne sont donc pas strictement identiques. Les 34 premiers cas sont inchangés.
Les détails historiques sont dans `stress-baseline.json` et `stress-before-recovery.json`.

Le stress de **35 cas comporte exactement 612 champs de référence** :
160 FR + 128 UK + 192 US + 132 layouts. Avant récupération, **416/612 (67,97 %)**
étaient neutralisés ; sur images dégradées : **416/480 (86,67 %)**.
L'ancien nombre de **600 champs** s'applique uniquement aux **34 cas antérieurs**.

| Groupe | Cas | Faux conservés | `null` à tort | Anomalies fantômes | Crash |
| --- | ---: | ---: | ---: | ---: | ---: |
| France, images dégradées | 8 | 0 | 160 | 0 | 0 |
| UK, images dégradées | 8 | 0 | 112 | 0 | 0 |
| US, images dégradées | 8 | 0 | 144 | 0 | 0 |
| Mises en page indépendantes / PDF | 11 | 0 | 0 | 0 | 0 |

## Nouvelle passe : réglage / validation figés

Le diagnostic des null a été publié **avant changement de seuil** :
`null-breakdown.md`. Principales causes : tableau non reconstruit 200,
désaccords des segmentations 64, confiance sous le seuil 60 (ex æquo avec
aucun token 60). Attribution exclusive du refus global, pas diagnostic
individuel d'illisibilité de chacun des 416 champs.

| Partition | Cas | Champs attendus | Null avant | Null après (application vérifiée) | Faux gardés | Fantômes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Réglage | 21 | 372 | 252 | 94 | 0 | 0 |
| Validation | 14 | 240 | 164 | 61 | 0 | 0 |
| Total | 35 | 612 | 416 | 155 | 0 | 0 |

La cible **sous 30 % sur images** n'est pas atteinte : **155/480 = 32,29 %**.
Réglage images : **94/300 = 31,33 %** ; validation images :
**61/180 = 33,89 %**. Aucun des deux ne passe sous 30 %.
La baisse est néanmoins nette, **−261 null** ; rapport total **155/612 = 25,33 %**.
Les onze layouts (132 champs) restent sans null. Le chemin applicatif complet
a été réexécuté avec le vrai OCR : **155 null, 0 valeur fausse conservée,
0 anomalie fantôme, 0 crash** sur les 35 cas. Les 105 profils modèle
supplémentaires conservent **0 valeur fausse et 0 anomalie fantôme**.
L'ensemble des résultats est dans `stress-report.json`.
Le tableau détaillé avant/après par essai et seuil est `ocr-experiments.md`.

Une première suite complète de cette passe a échoué avec **deux dépassements
du délai Tesseract sur des images détruites** (et un test de statuts dépendant
de ces cas). Ce résultat n'a pas été masqué par le score du runner isolé.
Le dépassement exact `Tesseract process timeout` est désormais journalisé et
converti en preuve OCR insuffisante ; les autres `RuntimeError` sont propagées.
Deux tests dédiés vérifient cette distinction. Le résultat final de suite
est **vert** et enregistré séparément dans `ocr-final-tests.log`.

| Famille (dénominateur exact) | Null avant | Null après | Faux / fantômes |
| --- | ---: | ---: | --- |
| France : 160 champs | 160 | 63 | 0 / 0 |
| UK : 128 champs | 112 | 43 | 0 / 0 |
| US : 192 champs | 144 | 49 | 0 / 0 |
| Layouts/PDF : 132 champs | 0 | 0 | 0 / 0 |

Les trois images détruites représentent 60 champs, les trois demi-résolutions
60 autres champs : ces deux familles non lisibles imposent déjà **25 %**
de refus sur les seuls champs image. Les 35 autres null d'essai ne sont pas
comblés artificiellement pour passer sous 30 %.

Le premier consensus avait gardé une fausse description (`Ciavier sans fil`) :
**essai rejeté**, conservé dans `ocr-tuning-consensus-rejected.json`.
Le débruitage apporte un veto textuel, ×3 aide certains champs ; ces deux
méthodes ne sont pas revendiquées comme améliorations isolées.
Otsu et le seuil local ont été rejetés pour leurs pertes de couverture.
La sélection est figée sur réglage dans `ocr-selection.json`, puis validée
sans changement sur les 40 % restants.

Images : rotations +1,5°/−2°, flou σ=0,8, JPEG qualité 40, contraste 0,25,
résolution divisée par deux, bruit gaussien σ=14 et bruit destructeur. Seed 713.
Layouts : colonnes inversées, Qté/Qty/P.U., remise textuelle déjà incluse,
remise monétaire séparée, TVA multi-taux, séparateurs inverses explicites,
préfixes/suffixes de devise et tableau PDF sur deux pages.

## Échecs réellement observés avant correction

| Cas | Problème observé | Cause |
| --- | --- | --- |
| france/rotate:1.5 | faux IBAN invalide | Ligne d'IBAN tronquée par regroupement géométrique. |
| france/rotate:-2 | fournisseur `Services`, dates invalides fantômes | Lignes fragmentées par rotation. |
| france/jpeg40 | fournisseur `Services 1zon SAS`, faux IBAN | Bruit JPEG/texte OCR tronqué ou chiffre mal lu. |
| uk/blur | `Oak Studio Lid`, quantité `78` au lieu de `7.8`, erreur de facture fantôme | Décimale perdue, malgré confiance OCR élevée. |
| uk/jpeg40 | quantité `78` au lieu de `7.8` | Décimale perdue, montant OCR manquant. |
| uk/half_resolution | date `2025-08-18` au lieu de `2025-05-18` | Résolution insuffisante, faible confiance moyenne. |
| uk/noise | faux écart HT + TVA/TTC | Ligne de TVA non récupérée ou interprétation discordante. |
| us/rotate:1.5 | fournisseur `Cedar Office LLC States` | Fusion de texte voisin. |
| us/half_resolution | fournisseur avec texte client fusionné | Résolution insuffisante et rôle/mise en page mal reconnu. |

Ces cas ne sont pas cachés : ils restent visibles dans le rapport avant
qualification. Aucun cas ne conserve de valeur fausse ou d'anomalie fantôme
dans l'exécution finale ; les cas douteux rendent `null` et `non vérifié`.

## Limites non résolues

- Deux segmentations du même moteur peuvent partager la même mauvaise lecture.
  Exemple concret : UK flou lit `78` dans les **deux** segmentations. Le filtre
  ne les considère pas comme deux preuves indépendantes de vérité.
- Une erreur de calcul lue sur une image peut être une vraie erreur imprimée
  ou une décimale perdue par OCR : sans confirmation, le rapport ne l'affirme
  pas. Une vraie erreur dans le PDF natif/texte est toujours signalée, valeurs
  imprimées conservées. Le rappel sur les images contenant de vraies erreurs
  reste un risque produit, à valider manuellement.
- La récupération régionale ne conserve un champ que s'il est lu identiquement
  par plusieurs familles et sans contradiction textuelle. La même erreur
  peut encore être commune au moteur : zéro erreur mesurée n'est pas une preuve
  universelle. Les cas de validation avaient déjà été observés avant cette
  passe : le split est figé, pas un vrai test externe aveugle.
- Latence accrue : jusqu'à dix OCR par image et recadrages supplémentaires.
  Le temps de la suite locale n'est pas
  une mesure de latence du Space, ni un benchmark GPU.
- Vrai modèle et déterminisme brut non vérifiés : Torch/Transformers/PEFT
  absents. `run_real_model_check.py --check-environment` échoue explicitement,
  sans charger ni télécharger de modèle. Le script GPU reste à exécuter.

## Revue et publication

- Audit complet : `docs/TEST_CHANGES.md` — **29 tests existants modifiés/renommés**,
  aucun supprimé sans remplacement, **21 couvertures historiques réduites**
  explicitement signalées. Un test vert ne rétablit pas une assertion retirée.
- Trois aperçus HTML statiques dans `results/report-previews/` ; le cas insuffisant
  montre l'instruction de scan et des lectures modèle simulées explicitement hors JSON ; documents
  de démonstration textuels, ni screenshots du navigateur ni sorties du modèle.
- Plan d'une page : `docs/DEPLOY.md`.
- Tag **local seulement** : `backup/pre-document-report-137b643` → `137b643`.
- Aucun commit ni push effectué. HEAD reste `137b643`, arbre local modifié.
- Script réel : `scripts/run_real_model_check.py --stress` traite aussi les
  images dégradées et mesure le vrai modèle même quand l'application aurait
  refusé de l'appeler (génération de mesure seulement, jamais ajoutée au JSON).
  Notebook `notebooks/real_model_check.ipynb` non exécuté, sans sorties.

## Passe latence et préparation des scans réels

La politique finale `ocr-policy-4-bounded-fastpath` est définie uniquement
dans `src/ocr_config.py` (confiance 40 et familles inchangées). Elle ajoute
un budget total **45 s par document**, partagé entre les pages, et compare
au départ deux familles distinctes (gris redressé PSM 4 / CLAHE PSM 6).
Le chemin complet n'est lancé que si cette confirmation est douteuse.
Les lectures déjà faites et le redressement sont réutilisés.

Mesure sur les mêmes 35 cas : **155 → 155 null**, **0 faux / 0 fantôme /
0 crash**, variation **0 point**, dans le plafond autorisé de +2 points.
Les 105 profils supplémentaires conservent zéro valeur fausse et zéro anomalie
fantôme. Le candidat intermédiaire à 153 null n'a pas été gardé sous sa forme
à deux lectures du même prétraitement ; il ne remplace pas ces résultats finaux.

Suite complète finale : **271 tests réussis, 3 slow exclus, 101 sous-tests**.
Les temps réels, nombre d'appels, limites du p95 à trois mesures et ressources
sont dans `results/LATENCY.md`, données brutes `latency-before.json`,
`latency-after-final.json`, `latency-scanned-pdf-final.json`. Résultat de
sûreté et durée par cas : `stress-latency-final.json`.

Le script `evaluate_real_scans.py` fonctionne avec référence manuelle optionnelle,
sans modèle ; huit tests utilisent des PDF générés, **pas de véritables nouvelles
factures**. Aucun scan réel indépendant n'a été fourni. Protocole préenregistré :
`docs/REAL_SCANS.md`. Les quatre patchs de revue sont dans `results/review-themes/`.
Point de déploiement restant : l'environnement local utilise Pillow 12.3.0,
contrairement au manifeste `<11.0` ; vérifier sur environnement neuf ou aligner
avec autorisation avant publication. Aucun commit ni push.
