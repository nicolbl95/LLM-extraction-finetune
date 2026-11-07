# PRODUCT-REVIEW-70A — Préparation du benchmark indépendant

**Date :** 2026-11-06  
**Décision :** NO-GO pour lancer la comparaison OCR tant que le corpus indépendant n'est pas acquis et annoté; GO pour la préparation, qui est désormais terminée et vérifiée (§6-§7).

## Synthèse

Les assets existants permettent de tester les scripts et de reproduire des cas historiques, mais **0 facture réelle est actuellement utilisable comme holdout frais et indépendant**. Les images déjà évaluées ne deviennent pas indépendantes parce qu'elles ont un nouveau nom, une copie anonymisée, ou une annotation reformatée.

Cette phase ajoute un schéma d'annotation manuelle, un protocole aveugle à deux annotateurs, un manifeste de sélection vide, et un validateur hors ligne avec tests synthétiques. Le code est isolé sous [`benchmark_prep/`](./benchmark_prep/) et utilise uniquement la bibliothèque standard. Il ne lance ni OCR ni modèle et n'importe aucun code de production. Aucune annotation historique n'a été transformée en vérité indépendante; le manifeste ne contient aucun document choisi ou inventé.

## 1. Inventaire disponible et lacunes

| Axe | Mesure du manifeste / documents | Conséquence |
|---|---|---|
| Assets | 637 chemins, tous PNG; aucun PDF/JPEG dans ce manifeste. 600 sont synthétiques, les 37 autres comprennent fixtures, scans historiques/anonymisés, templates web, exemples et fragment partiel. | Les PDFs et autres formats d'entrée ne sont pas représentés. |
| Pays/langue | Les 600 factures synthétiques sont 200 `fr-FR`, 200 `en-GB`, 200 `en-US`. Les 37 autres contiennent 27 entrées sans pays/langue; les 10 entrées historiques/template aux métadonnées remplies ne prouvent pas 10 factures réelles. Six scans historiques ont pays/langue absents dans le manifeste. | La distribution synthétique est équilibrée, mais pas la distribution réelle indépendante. Le test doit comprendre de vraies factures UK. |
| Références | 600 sorties synthétiques déterministes; 5 scans réels primaires déjà utilisés en développement; Pure Cleaning est annoté indépendamment sur un seul document, mais déjà réutilisé/layout connu; les templates web sont historiques ou d'authenticité incertaine. **0 holdout frais éligible.** | Régression seulement pour les données existantes. |
| Qualité visuelle | Un proxy image calculé pour 637/637 (variance Laplacienne min/mediane/max : 217.879 / 1056.277 / 5152.787). Le manifeste précise que ce proxy n'est pas une note humaine de lisibilité. | Aucune catégorie fiable « nette/dégradée/illisible » sans revue humaine indépendante. |
| Complexité (nombre de lignes indiqué) | Assets : 101 à 1–2 lignes, 147 à 3–5, 269 à 6–10, 103 à 11+, 17 inconnus; la majorité est synthétique. | La complexité synthétique ne remplace pas des tableaux réels FR/UK/US. |
| Taxes/remises | Les sorties synthétiques `output` ont les clés du JSON de production : HT, TTC, lignes; aucune n'a de clé `taxes` ou `discounts`. Le fichier historique `real-scans-truth.json` ne transcrit pas les composants fiscaux. | Présence/absence des taxes, remises, frais de port et soldes n'est pas annotée systématiquement. Elle doit être capturée comme preuve annexe; ne pas interpréter une clé manquante comme une absence imprimée. |
| Totaux | HT et TTC sont disponibles dans les labels synthétiques; les 5 transcriptions primaires contiennent HT/TTC mais aucune taxonomie complète des lignes de synthèse. | Annoter valeur visible, libellé et rôle distinctement; comparer les valeurs imprimées sans les réparer par somme. |
| Format de tableau | `line_item_count` n'est pas renseigné pour 17 assets; les 637 images sont PNG. | Ajouter PDF multi-pages, scans photographiés, descriptions multilignes, lignes répétées, taxes multi-taux, remises et avoirs lorsqu'ils sont obtenus avec permission. |

Les SHA-256 des images et références précédemment vérifiés dans PR69 restent la preuve d'identité des fichiers, pas de leur indépendance. Les anciennes factures de développement, leurs variantes anonymisées, les captures de PR64–PR69, les fixtures et les références synthétiques réutilisées sont exclues du holdout.

### Documents réutilisables et manquants

- **Utilisables comme vérité indépendante de test : 0.**
- **Utilisables seulement en développement/régression :** les 5 factures réelles de `real-scans/`, le document UK Pure Cleaning déjà réutilisé, les factures synthétiques déterministes et les cas contrôlés. Les templates historiques ne sont pas confirmés comme factures réelles.
- Les 637 assets sont des images PNG; aucun n'est promu par ce travail au rang de nouvel échantillon ou de référence.
- Le premier pilote proposé requiert **30 nouvelles factures permissionnées et non contaminées** : 6 de développement (2 par locale), 9 de validation (3 par locale) et 15 de test final scellé (5 par locale). Il faut donc acquérir 10 FR, 10 UK et 10 US, toutes avec provenance contrôlée. Le manifeste vide indique précisément ces quotas manquants.
- Le test initial doit viser au moins 8 familles de layout inédites, au moins 3 familles fournisseur par locale, 120 champs scorables, 30 lignes et 50 valeurs attendues. Inclure, entre autres, taxes multi-taux, taxes de vente/VAT, remises/frais, descriptions multilignes, quantités décimales, avoirs, montants négatifs, totaux non libellés, lignes répétées et cas de faible lisibilité.
- Ces 15 documents de test forment un **pilote**, pas une preuve statistique confirmatoire. Estimer après le pilote une taille fondée sur l'effet minimal et la variance, avec rééchantillonnage/power analysis par facture et stratification locale. Les cellules d'une facture ne sont pas indépendantes; une précision ±5 points sur une proportion de champs exige environ 385 observations indépendantes sans clustering, davantage avec l'effet de grappe. Un test final d'environ 30 factures par locale (90) est un point de départ à confirmer par cette analyse, non une garantie automatique.

Le manifeste [`holdout_selection_manifest.json`](./benchmark_prep/holdout_selection_manifest.json) ne contient que les plans et les quotas manquants; `selected_documents` est vide. Il ne contient aucun nom, hash, annotation ou document fictif.

## 2. Schéma et protocole d'annotation

[`annotation.schema.json`](./benchmark_prep/annotation.schema.json) associe aux champs de production (`supplier_name`, `invoice_date`, `total_ht`, `total_ttc`, puis description/quantité/prix/montant de chaque ligne) un état de référence, le texte exact visible, une valeur normalisée, les boîtes de preuve et une note. Le schéma conserve aussi devise, taxes, remises et lignes de synthèse comme informations documentaires **hors JSON de production**.

Chaque champ distingue `present`, `absent`, `unreadable`, `ambiguous`, `derived_only` et `unscorable`. Le texte imprimé et une éventuelle valeur calculée sont séparés. Une incohérence quantité × prix ou avec un total est seulement signalée; l'annotation imprimée n'est jamais réécrite.

Le protocole [`ANNOTATION_PROTOCOL.md`](./benchmark_prep/ANNOTATION_PROTOCOL.md) exige :

1. Authentification/permission, hash et sélection figés avant annotation.
2. Deux transcriptions indépendantes depuis l'image pleine résolution, sans prédictions, anciennes références, résultats PR64–PR69 ou annotation du pair.
3. Comparaison des états, valeurs et boîtes seulement après verrouillage des deux fichiers.
4. Arbitrage humain à partir du document original, avec justification par divergence; laisser `ambiguous`/`unscorable` si le désaccord persiste.
5. Export strict de production et sidecar d'états seulement après adjudication complète. Le finaliseur refuse toute divergence non résolue.

## 3. Outils isolés

[`annotation_tools.py`](./benchmark_prep/annotation_tools.py) utilise uniquement la bibliothèque standard et ne référence ni l'application, ni Tesseract, ni RapidOCR, ni Qwen. [`engine_run.schema.json`](./benchmark_prep/engine_run.schema.json) définit, sans créer de run, le contrat de capture futur commun aux systèmes `tesseract_current`, `rapidocr_independent` et `combined_proposals`.

| Commande | Fonction |
|---|---|
| `validate` | Vérifie l'annotation, états, types, dates, nombres, boîtes et attestations; optionnellement vérifie SHA-256 dans une racine d'assets. |
| `compare` | Compare deux annotations du même document et rapporte les divergences cellule par cellule et les divergences de boîtes sans sélectionner une valeur. |
| `validate-selection` | Vérifie les quotas de split/locale, doublons, hashes source/annotations, recouvrements fournisseur/layout entre splits, permission, authenticité et absence d'usage antérieur; rapporte les documents manquants. |
| `finalize` | Exige les deux annotations indépendantes, une résolution humaine de chaque divergence et une annotation complète d'arbitrage; écrit le JSON de production et un sidecar de provenance sans écraser des fichiers. |

Le contrôle arithmétique vérifie uniquement les produits de ligne et les montants de taxe lorsque toutes les cellules imprimées nécessaires sont présentes. Il émet un avertissement et conserve les montants transcrits. Les champs `ambiguous`, `unreadable`, `unscorable` ou non imprimés deviennent `null` dans le JSON strict, mais leur état reste dans le sidecar pour ne pas scorer une valeur inconnue comme absence confirmée. Pour chaque document, le manifeste conserve aussi les chemins/hash des deux annotations indépendantes et de l'arbitrage; la commande de sélection les vérifie sans afficher ni publier de facture.

Les tests de [`test_annotation_tools.py`](./benchmark_prep/tests/test_annotation_tools.py) utilisent uniquement des fixtures artificielles explicitement marquées `synthetic_test`; elles ne sont ni factures ni références du benchmark.

## 4. Comparaison OCR prévue

Les outils de vérité terrain sont indépendants des moteurs et ne les importent pas. PR70B pourra comparer ultérieurement, sur les mêmes hashes et après autorisation :

- **A — Tesseract actuel :** `fra+eng`, configuration/PSM/prétraitements et versions enregistrés; ne pas modifier le seuil 85, le seuil token 60, le consensus ou le grounding.
- **B — RapidOCR :** version et poids explicitement épinglés, modèle français/anglais sélectionné et hashé; RapidOCR par défaut n'est pas présumé compatible français.
- **C — propositions combinées :** sortie, boîtes et confiance de chaque moteur séparées; un désaccord entraîne abstention/revue. Le vote, l'arithmétique et la confiance ne sont jamais à eux seuls une preuve d'impression.

Comparer les systèmes en apparié sur chaque document exact : exactitude OCR dans la cellule annotée, affectation de colonne, rattachement de ligne, exactitude stricte/substantielle des descriptions, quantités (`1e-6`), prix et montants (centimes `ROUND_HALF_UP`), HT/TTC, précision des valeurs acceptées, rappel des champs présents, faux positifs monétaires, taux de `null`, lignes complètes/ordre et temps/coût médian/p95. Séparer résultats de développement, validation et test final; ne pas régler les méthodes sur le holdout.

## 5. Exécution et vérification

Commande ciblée prévue :

```powershell
.\.venv\Scripts\python.exe -m pytest -q benchmark_prep\tests\test_annotation_tools.py
```

Critères de sortie de PR70A :

- Le test ciblé passe sans OCR/modèle.
- Le schéma JSON est lisible; annotations invalides, hashes incohérents, chemins échappant au répertoire, doublons de documents et divergences non arbitrées sont refusés.
- Les erreurs arithmétiques ne modifient aucune valeur imprimée.
- Le manifeste reste `not_ready`, avec zéro document sélectionné et les quotas manquants FR/UK/US explicites.
- Le statut de tous les drapeaux et seuils de production reste inchangé.

Les conditions mesurables de réussite de la future comparaison (précision proposée ≥0,90, zéro valeur numérique de ligne fausse acceptée, couverture ≥0,65, rappel amélioré d'au moins 5 points, aucune perte nette de précision numérique par locale, aucune régression gelée et p95 dans le budget 45 s) sont détaillées dans [PR69](./PRODUCT_REVIEW_69_INDEPENDENT_OCR_STRATEGY.md). Elles ne sont pas évaluées dans PR70A.

## 6. Vérification d'intégrité exécutée (2026-11-06)

Toutes les vérifications ci-dessous ont été exécutées hors ligne, sans OCR, modèle, GPU, cloud, téléchargement, base de données, commit ni push.

| Contrôle | Commande / méthode | Résultat |
|---|---|---|
| Tests synthétiques | `.\.venv\Scripts\python.exe -m pytest -q benchmark_prep\tests\test_annotation_tools.py` | **12 passed** en 0,28 s |
| CLI | `annotation_tools.py --help` | 4 commandes exposées : `validate`, `compare`, `validate-selection`, `finalize` |
| Isolation | inspection des imports | bibliothèque standard uniquement ; aucune référence à `app`, `src`, Tesseract, RapidOCR, Qwen, torch ou transformers |
| Schéma | parsing JSON | valide ; 14 clés requises de premier niveau, dont `document`, `invoice`, `line_items`, `taxes`, `discounts`, `financial_summary` |
| Manifeste holdout | parsing JSON + `validate-selection` | `valid=true`, `ready=false`, `status=not_ready`, `selected_document_count=0`, manquants = 2/2/2 (développement), 3/3/3 (validation), 5/5/5 (test final) |
| Inventaire ré-agrégé | relecture de `INVOICE_BENCHMARK_MANIFEST.json` | 637 documents, 100 % PNG ; 600 synthétiques (200 fr-FR / 200 en-GB / 200 en-US) ; 37 autres ; lignes 101 / 147 / 269 / 103 / 17 inconnues ; proxy qualité numérique 637/637 |
| Éligibilité holdout | re-agrégation de `held_out_eligibility` | **0 document éligible** : 480 `ineligible_training`, 60 `ineligible_development`, 60 synthétiques de test à statut « inconnu » (manifeste d'entraînement de l'adaptateur et journal d'usage indisponibles), 14 déjà utilisés, 8 fixtures, 6 historiques, 4 authenticité non établie, 3 doublons, 1 non-facture, 1 partiel |
| Hashes d'assets | échantillon déterministe de 21 documents (pas de 31) | 21 présents, **21 SHA-256 identiques**, 0 écart |
| Spec comparaison OCR | `benchmark_prep/ocr_comparison_spec.json` | créé, déclaratif ; critères PR69 gelés ; aucun moteur installé ni exécuté |

**Portée des hashes :** PR69 a vérifié les 637 assets ; cet audit en ré-échantillonne 21 de façon déterministe pour confirmer qu'aucun fichier n'a bougé depuis. L'identité d'un fichier ne prouve pas son indépendance : elle est nécessaire, pas suffisante.

### Incident d'intégrité détecté et réparé

`benchmark_prep/annotation_tools.py` a été modifié par un autre processus pendant cet audit (mtime 16:04:12), après un premier passage de tests réussi. La modification a dupliqué un bloc d'attestation avec une indentation incorrecte et une parenthèse orpheline, rendant le module **non importable** (`SyntaxError` ligne 468). Les horodatages de `annotation.schema.json` (16:04:29) et de `test_annotation_tools.py` (16:05:45) montrent des écritures que cet audit n'a pas produites : un rédacteur concurrent était actif sur les livrables.

Réparation appliquée : suppression du bloc dupliqué, conservation du seul bloc correct pour `pr64_to_pr69_viewed`. Vérifications après réparation : `py_compile` OK, **12/12 tests**, CLI OK, `validate-selection` OK. Aucune autre duplication adjacente détectée (un `return` redondant équivalent subsiste dans `_same_selected_value`, sans effet de comportement ; laissé tel quel pour ne pas modifier du code hors périmètre). **Recommandation : s'assurer qu'aucun autre agent n'écrit dans `benchmark_prep/` avant de figer PR70A ou de démarrer PR70B.**

**Écarts détectés :** aucun autre. Le rapport, le schéma, le manifeste de sélection, les tests et le protocole sont cohérents avec le manifeste ré-agrégé ; aucune annotation automatique n'a été présentée comme vérité indépendante.

## 7. Statut final PR70A

| Question | Réponse |
|---|---|
| Documents actuellement utilisables comme holdout indépendant | **0** |
| Documents supplémentaires nécessaires | **30** nouvelles factures permissionnées : 10 FR, 10 UK, 10 US → 6 développement (2/locale), 9 validation (3/locale), 15 test final scellé (5/locale) |
| Statut des outils | **Prêts et vérifiés** : schéma (427 lignes), protocole (67 lignes), validateur (1 879 lignes, 4 commandes, 12/12 tests), manifeste de sélection (`not_ready`, 0 sélection), spec de comparaison OCR |
| Conditions restant à satisfaire avant PR70B | (1) acquérir 30 factures authentiques avec permission et provenance ; (2) figer hashes et IDs avant annotation ; (3) deux annotations aveugles complètes ; (4) arbitrage humain exhaustif ; (5) `validate-selection` → `ready=true` ; (6) épingler versions et hashes des moteurs ; (7) autorisation explicite de PR70B |
| Décision | **NO-GO** pour lancer la comparaison des moteurs OCR · **GO** pour la préparation, terminée |

## Handover

- **État initial :** branche `wip/layouts-reprise`, HEAD `9d7a641a52c114484ee614730d15882dbaa35d66`, worktree déjà sale; changements antérieurs préservés.
- **Fichiers créés :** le présent rapport et les six fichiers isolés sous `benchmark_prep/` (schéma, manifeste de sélection, protocole, outil, tests et spec de comparaison OCR).
- **Production :** aucun fichier d'application, d'OCR, de consensus, de grounding ou de modèle modifié.
- **Tests/commandes :** `pytest -q benchmark_prep/tests/test_annotation_tools.py` → 12/12 ; `annotation_tools.py --help` → 4 commandes ; `annotation_tools.py validate-selection benchmark_prep/holdout_selection_manifest.json` → `valid=true`, `ready=false`, quotas manquants affichés ; échantillon déterministe de 21 hashes → 21/21 identiques ; parsing JSON des livrables → valides. Aucun OCR, Qwen, téléchargement, GPU, cloud, déploiement, commit ou push exécuté.
- **Blocage :** aucun holdout réel frais; tous les 30 documents initiaux sont à collecter. Le lot final 15 documents est un pilote seulement.
- **Prochaine étape :** acquérir les 30 factures permissionnées, appliquer le protocole d'annotation aveugle, puis rendre `validate-selection` `ready=true`. PR70B requiert une nouvelle autorisation et n'a pas été commencé.
