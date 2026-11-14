# Protocole d'annotation indépendante des factures

Version : `invoice-blind-annotation-v1`  
Schéma : [`annotation.schema.json`](./annotation.schema.json)  
Contrat des exécutions OCR futures : [`engine_run.schema.json`](./engine_run.schema.json)  
Outils hors ligne : [`annotation_tools.py`](./annotation_tools.py)

## 1. Avant l'annotation

1. Le coordinateur vérifie que le document est une facture émise authentique, que son utilisation est autorisée, et qu'il n'a été utilisé ni pour entraînement, ni développement, ni réglage, ni débogage, ni test antérieur. Vérifier également l'absence de doublon par hash, de source identique, de fournisseur/layout connu et de dérivé recadré.
2. Créer un `document_id` opaque, calculer SHA-256 des octets originaux et enregistrer chemin relatif, format, pages, dimensions, pays/langue/devise si établis. Figer la sélection et les hashes avant toute annotation ou exécution d'un moteur.
3. Créer deux copies vierges du formulaire à partir du schéma. Les réviseurs A et B reçoivent uniquement l'image autorisée, le protocole et le même identifiant de document. Ils ne reçoivent aucune valeur préremplie, OCR, sortie modèle, ancienne vérité terrain, rapport PR64–PR69 ni annotation de l'autre réviseur.
4. Les images et annotations non publiques restent dans un stockage privé approuvé. Ne pas publier des factures clients ni leurs descriptions, montants, adresses ou coordonnées.

## 2. Annotation indépendante (A et B)

Chaque réviseur ouvre l'image à pleine résolution et transcrit indépendamment :

- Champs généraux correspondant au JSON de production : `supplier_name`, `invoice_date`, `total_ht`, `total_ttc`.
- Chaque ligne physique : identifiant visuel stable par page et ordre de haut en bas (`p001-r001`), boîte normalisée, `description`, `quantity`, `unit_price`, `total_ht`. Inclure une ligne dont les cellules sont illisibles; ne pas en inventer.
- Taxes, remises, livraison, solde antérieur et autres lignes de synthèse dans les collections annotation-only `taxes`, `discounts`, `financial_summary`. Ne pas ajouter ces éléments au JSON de production, qui ne possède pas ces clés.
- Pour chaque champ : état, texte imprimé exact, valeur normalisée distincte, et une ou plusieurs boîtes de preuve avec page et libellé voisin. Pour les nombres, `normalized_value` est une chaîne décimale sans devise ni séparateur de milliers; `visible_text` conserve exactement le texte de l'image. Date normalisée ISO `YYYY-MM-DD`. Texte normalisé en NFC avec espaces consécutifs réduits.

États autorisés :

- `present`: valeur lisible et visible, avec texte exact et boîte de preuve.
- `absent`: le champ est explicitement absent; noter pourquoi. Ne pas employer cet état seulement parce qu'un réviseur ne l'a pas trouvé.
- `unreadable`: la zone existe, mais sa valeur ne peut pas être lue; conserver sa boîte et tout fragment réellement lisible.
- `ambiguous`: plusieurs lectures visuelles restent plausibles; conserver au moins deux chaînes candidates et la zone.
- `derived_only`: valeur calculée possible mais non imprimée; garder `normalized_value=null`, noter formule et chemins sources dans `derivation`.
- `unscorable`: aucune référence fiable ne peut être établie; expliquer pourquoi.

Le format distingue toujours l'impression (`visible_text`) de sa normalisation (`normalized_value`) et d'un calcul (`derivation`). Si l'arithmétique contredit une valeur imprimée, transcrire le montant imprimé tel quel, enregistrer le calcul séparément et laisser l'outil signaler le désaccord. Ne jamais corriger silencieusement le document ni remplacer une cellule imprimée par le produit quantité × prix.

Les boîtes utilisent des coordonnées normalisées `[x0, y0, x1, y1]` entre 0 et 1, origine en haut à gauche, avec `page_index` commençant à zéro. Utiliser une même convention d'identifiants de ligne; les lignes indéterminées restent présentes avec `row_state="ambiguous"` ou `unscorable`.

Chaque réviseur signe les attestations dans `reviewer.attestation`: image source vue; prédictions, anciennes annotations, annotation pair et rapports PR64–PR69 non consultés. `provenance.derived_from_existing_reference` doit être `false`. Les annotations de test synthétiques sont marquées `annotation_kind="synthetic_test"` et ne sont jamais admises dans un manifeste de holdout.

## 3. Comparaison et arbitrage

1. Après remise et verrouillage des deux fichiers, lancer `compare`. L'outil vérifie que document ID et SHA correspondent, compare les états, textes, valeurs et candidats champ par champ, signale les lignes présentes d'un seul côté, et rapporte séparément les différences de boîtes.
2. L'outil ne choisit jamais A ou B, ne calcule pas une valeur de remplacement et ne transforme pas un désaccord en vérité. Ses contrôles arithmétiques sont des avertissements de revue seulement.
3. Un adjudicateur ouvre l'image originale et les deux transcriptions, mais pas les sorties OCR/modèle ni les anciennes références. Pour chaque chemin divergent, il enregistre une résolution explicite (`agree_a`, `agree_b` ou `new_manual`), une justification et, si nécessaire, les boîtes de preuve.
4. L'adjudicateur produit une annotation complète `annotation_kind="adjudicated"` depuis l'image. Les chemins divergents doivent correspondre exactement à la liste de résolutions. En cas de doute persistant, marquer `ambiguous` ou `unscorable`; ne pas imposer un consensus artificiel.
5. `finalize` refuse l'export s'il manque une résolution, si les documents/hashes diffèrent, si une annotation n'est pas valide ou si le final adjudicated n'atteste pas que l'image a été revue. Il écrit (sans écraser de fichiers) le JSON strict de production et un sidecar d'états/provenance/arbitrage. Pour les états non `present`, le JSON strict reçoit `null`; le sidecar préserve `absent`, `derived_only`, `unreadable`, `ambiguous` et `unscorable` afin que le scoreur ne les confonde pas.
6. Après chaque signature A, B et arbitrage, inscrire son chemin privé relatif et son SHA-256 dans le manifeste de sélection. `validate-selection --asset-root` vérifie les quatre artefacts contre leurs hashes et refuse les chemins absolus, les traversées et les réutilisations.

## 4. Séparation des jeux

- **Développement :** 6 documents, 2 par locale. Peut servir à corriger le câblage et choisir des paramètres; jamais compté comme test.
- **Validation :** 9 documents, 3 par locale. Peut servir au choix pré-enregistré des variantes; reste séparé du test final.
- **Test final :** 15 documents, 5 par locale, au moins 8 familles de layouts inédites et 3 familles de fournisseurs par locale. Scellé avant les runs, évalué une fois, sans retries/tuning.
- Les trois splits n'ont ni hash/document ID/source, ni fournisseur/layout partagé. Aucun document du manifeste historique, image de développement, doublon exact, template web ou dérivé recadré n'est réutilisé comme holdout.
- L'échantillon initial est un pilote, pas une preuve statistique définitive. Après le pilote, calculer la taille confirmatoire à partir de la variance observée, de l'effet minimum pertinent et d'un plan apparié par facture/locale. Les cellules d'une même facture ne sont pas des unités indépendantes.

## 5. Commandes hors ligne

Depuis la racine du dépôt sous Windows :

```powershell
.\.venv\Scripts\python.exe benchmark_prep\annotation_tools.py validate path\to\annotation.json --asset-root path\to\private-assets
.\.venv\Scripts\python.exe benchmark_prep\annotation_tools.py compare path\to\annotator-a.json path\to\annotator-b.json --asset-root path\to\private-assets
.\.venv\Scripts\python.exe benchmark_prep\annotation_tools.py validate-selection benchmark_prep\holdout_selection_manifest.json --known-manifest INVOICE_BENCHMARK_MANIFEST.json
.\.venv\Scripts\python.exe benchmark_prep\annotation_tools.py validate-selection benchmark_prep\holdout_selection_manifest.json --known-manifest INVOICE_BENCHMARK_MANIFEST.json --asset-root path\to\private-assets
.\.venv\Scripts\python.exe benchmark_prep\annotation_tools.py finalize --a path\to\annotator-a.json --b path\to\annotator-b.json --adjudication path\to\adjudication.json --output path\to\reference.json --metadata-output path\to\reference.metadata.json
```

Le validateur utilise uniquement la bibliothèque standard, le système de fichiers et les documents JSON fournis. Il n'importe pas le pipeline, Tesseract, RapidOCR ou Qwen, et ne lit aucun contenu OCR. `compare` peut afficher du texte sensible : l'exécuter seulement dans un environnement local approuvé.

## 6. Contrat de capture A/B/C

Chaque exécution future produit un enregistrement par document et système conforme à `engine_run.schema.json`. Il associe le hash source, split/locale, versions et hashes des moteurs/modèles, langue, prétraitements, PSM, seuils et flags aux tokens bruts, hypothèses de cellules, JSON strict, latences, ressources CPU/RAM/VRAM et coût. Les systèmes sont nommés `tesseract_current`, `rapidocr_independent` et `combined_proposals`; une hypothèse cite explicitement ses `engine_ids` et `token_ids`. Ce schéma est seulement un contrat de données : PR70A ne crée aucun enregistrement d'exécution OCR.
