# PRODUCT-REVIEW-75 — Récupération contrôlée du pipeline de bout en bout

**Date :** 2026-11-06  
**Décision :** **PARTIAL GO pour l'invocation synthétique; NO-GO pour les tests de factures réelles.** Une seule requête autorisée au callback Gradio a terminé et retourné quatre sorties en 31,494 s, mais leur sérialisation locale a échoué avant capture. Il n'est donc pas possible de confirmer que Qwen a généré une réponse ni d'évaluer le JSON. La révision base du fine-tuning n'est pas prouvée.

## Résumé

- Space inspecté en lecture seule : public, Gradio 6.30.0 / Python 3.12, `RUNNING`/`READY`, SHA `927a115997ef2f850255e1b14cfe37fe861bb710`, matériel `zero-a10g`.
- Les logs actuels n'ont pas été obtenus : `/logs/build` a retourné 401 et `/logs/container` 404 sans authentification. Les derniers logs historiques disponibles (PR60) montrent un démarrage et le préchargement de fichiers, mais **pas** de chargement Qwen, d'allocation GPU ou d'inférence.
- Le `adapter_config.json` de l'adaptateur actuel désigne `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`; son champ `revision` est `null`. Le commit courant `21d7790b...` n'est pas une preuve du commit utilisé lors du fine-tuning.
- Le canary image-only est dans [pr75-synthetic-invoice.png](./tests/fixtures/pr75-synthetic-invoice.png); sa vérité terrain est dans [pr75-synthetic-invoice.expected.json](./tests/fixtures/pr75-synthetic-invoice.expected.json). Il contient deux lignes arithmétiquement cohérentes. Le JSON attendu respecte `Invoice` et a été validé sans exécuter OCR/modèle.
- Une requête d'extraction synthétique a été exécutée après autorisation. Aucune facture réelle, installation, modification/déploiement du Space ou nouvelle tentative n'a eu lieu. Le téléchargement éventuel des poids et la consommation GPU/quota ne sont pas observables dans la réponse capturée.

## 1. Configuration du Space et état de préparation

### État et build

L'API publique du Hub a retourné :

| Propriété | Valeur observée |
|---|---|
| Space | `nicolbl95/LLM-extraction-finetune`, public, SDK Gradio |
| Runtime / domaine | `RUNNING`; domaine `nicolbl95-llm-extraction-finetune.hf.space` en `READY` |
| Révision | `927a115997ef2f850255e1b14cfe37fe861bb710` |
| Configuration déclarée | Python 3.12, Gradio 6.30.0 |
| Matériel demandé / courant | `zero-a10g` |
| Réplicas | 1 |

Sources lues sans ouvrir l'application : [API Space](https://huggingface.co/api/spaces/nicolbl95/LLM-extraction-finetune) et [API runtime](https://huggingface.co/api/spaces/nicolbl95/LLM-extraction-finetune/runtime). La révision correspond au déploiement vérifié en PR60; elle est distincte du HEAD local.

Le runtime `RUNNING` indique que le serveur Gradio tourne, pas que le modèle est chargé ou qu'une allocation ZeroGPU est active. Aucun champ public ne révèle ici la VRAM/RAM effective, le cache courant, ou les versions réellement résolues dans l'environnement.

Le rapport PR60 documente une build sans échec critique, le passage à `RUNNING`, des endpoints de santé HTTP 200 et trois avertissements non fatals `ValueError: Invalid file descriptor: -1` lors d'un hot reload. Cela valide le lancement/serving de cette révision, pas l'inférence. Pour cette inspection, la lecture anonyme des logs a échoué (`logs/build` : 401; `logs/container` : 404). Aucun token n'a été recherché/utilisé; les logs d'amorçage du runtime courant restent indisponibles.

### Cache des modèles et compatibilité

La capture historique PR60, prise au déploiement de cette même révision, indique que le démarrage normal a préchargé environ **131 Mo** d'adaptateur/tokenizer et **2,37 Go** de fichiers du modèle de base. Elle précise aussi qu'aucune inférence ni allocation GPU n'avait alors été faite. Cela prouve un préchargement historique, pas la présence actuelle de ces fichiers dans le conteneur.

Le `startup_prefetch()` de la révision Space résout l'adaptateur puis précharge le modèle de base hors de la fonction GPU lorsque `SPACE_ID` est défini; `INVOICE_PREFETCH` vaut 1 par défaut. `INVOICE_MODULE_PRELOAD` n'est pas activé par défaut : le préchargement de fichiers ne signifie donc pas que Qwen a été instancié ou placé en mémoire GPU. Le chemin de `predict_invoice()` appelle `get_extractor()` sous `@spaces.GPU`; le premier chargement/inférence reste à vérifier.

Le Space déclare les contraintes du projet : Torch `>=2.8,<=2.13`, Transformers `>=4.49,<6`, PEFT `>=0.14`, Accelerate `>=1.2`, bitsandbytes `>=0.48` sous Linux. `requirements-inference.txt` inclut `requirements.txt` et Gradio 6.30.0. Aucun inventaire des versions résolues du conteneur courant n'est disponible dans les métadonnées/logs anonymes.

### API et parcours effectif

Le code déployé expose des événements Gradio, pas un endpoint Hugging Face Inference séparé :

- `preview_ui` est lié à `upload.change` et ne lance pas l'inférence.
- `run_extraction_ui` est lié au bouton d'extraction, `concurrency_limit=1`, dans une queue Gradio (taille max 8).
- Cet événement appelle `run_recruiter_extraction_ui` → `run_extraction_ui` → `run_extraction`/`document_text` → `extract_invoice` → `run_grounded_inference` → `predict_invoice` (`@spaces.GPU`) → grounding, validation et JSON strict.

Aucun de ces événements n'a été appelé pendant PR75. Le point d'entrée de canary sera `run_extraction_ui` avec le PNG synthétique ci-dessus, après autorisation.

## 2. Modèle et provenance

| Artefact | Révision / référence | Informations vérifiées |
|---|---|---|
| Adaptateur | `nicolbl95/llm-extraction-weights@a1e56cc8c313cb194395047332cd8df2a0e34e63` | Dépôt public. `adapter_model.safetensors` : 119 801 528 octets, SHA-256 LFS `b45155157d66288ebcd054da2b29e5b07bbe4185c243b5a9d40b7cd519964d87`. |
| Base déclarée par l'adaptateur | `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` | `adapter_config.json` indique `base_model_name_or_path` ci-dessus, `revision=null`, `peft_type=LORA`, PEFT 0.21.2, rang 16, alpha 16. |
| Base, HEAD actuel | `21d7790b1332196a5cf65ba72f2b46b659f74ddf` | Métadonnées du Hub indiquent Qwen2 `Qwen2ForCausalLM`, bitsandbytes 4-bit; `model.safetensors` SHA-256 LFS `26e4f54c6213083a3861866bdf7ae9e84f74314fb4286095031643de391fccc5`. |
| Révision de base utilisée au fine-tuning | **Inconnue** | `revision` est `null`; le manifeste d'entraînement n'est pas présent dans le dépôt adaptateur ni dans les résultats locaux recensés en PR73. Le HEAD actuel ne peut pas être assimilé au commit d'entraînement. |

La recherche ciblée de PR75 dans `notebooks/`, `scripts/`, `results/`, `diagnostics/` et `experiments/` n'a retrouvé que le code du notebook qui écrit `base_model_revision` dans le manifeste et le lecteur optionnel de manifeste de [bench_real_model.py](./scripts/bench_real_model.py). Aucun manifeste concret ni commit base d'entraînement n'y a été retrouvé.

Le `models` listé dans les métadonnées du Space mentionne le nom Qwen non quantifié, mais la configuration LoRA est plus précise : elle référence la variante BNB 4-bit. Le code `LoRAInference` charge le modèle nommé par l'adaptateur et la voie CUDA utilise bitsandbytes/NF4. Il faut préserver cette base et cette architecture; aucun autre modèle n'est substitué.

Le Space précharge à `main` sans SHA de base spécifié, et le fichier LoRA ne fixe pas de révision base. Si les poids doivent être téléchargés, l'application peut récupérer l'adaptateur/base depuis le Hub. Le canary autorisé sur le Space actuel ne serait donc une reproduction exacte que si le cache/runtime existant correspond aux révisions attendues; cette correspondance n'est pas actuellement observable.

## 3. Canary synthétique et vérité terrain

### Fichier et contenu

Le document local est explicitement étiqueté **synthétique**, **test uniquement** et **sans valeur commerciale**. Il s'agit d'un PNG image-only de 1600 × 2200 px, SHA-256 `e602a21cac9c8e8f918aedadbb77d5628ad41cae49445f01c2fd708646f08470` (110 412 octets). Sa vérité terrain est [pr75-synthetic-invoice.expected.json](./tests/fixtures/pr75-synthetic-invoice.expected.json), SHA-256 `CB18B889F00D26E5B92BDDB85F7D86D015B1C32119516F0B99D94CEDD35F3366`. Aucun OCR n'a été lancé sur le fichier.

Le motif global `.gitignore:11:*.png` ignore cette fixture image; elle reste présente localement, mais n'apparaît pas dans `git status`. Le `.gitignore` n'a pas été modifié et le fichier n'a pas été forcé dans l'index.

Valeurs imprimées :

- Fournisseur : `Atelier Lumen SARL`
- Client : `Client Demo SAS`
- Numéro : `TEST-PR75-001`
- Date : `6 NOVEMBRE 2026` (attendu ISO `2026-11-06`)
- Devise : EUR
- `Cahier A4`: quantité 2, prix unitaire HT 12,50 EUR, montant HT 25,00 EUR
- `Stylo bleu`: quantité 3, prix unitaire HT 4,00 EUR, montant HT 12,00 EUR
- Sous-total HT 37,00 EUR; TVA 20 % 7,40 EUR; total TTC 44,40 EUR

Calcul manuel : 2 × 12,50 = 25,00; 3 × 4,00 = 12,00; 25,00 + 12,00 = 37,00; 37,00 × 20 % = 7,40; 37,00 + 7,40 = 44,40.

### JSON strict attendu

```json
{
  "supplier_name": "Atelier Lumen SARL",
  "invoice_date": "2026-11-06",
  "total_ht": 37.0,
  "total_ttc": 44.4,
  "line_items": [
    {
      "description": "Cahier A4",
      "quantity": 2.0,
      "unit_price": 12.5,
      "total_ht": 25.0
    },
    {
      "description": "Stylo bleu",
      "quantity": 3.0,
      "unit_price": 4.0,
      "total_ht": 12.0
    }
  ]
}
```

Le schéma de production `Invoice` ne contient pas `customer_name`, `invoice_number`, `currency`, `vat_rate` ou `vat_amount`. Ces données sont visibles dans l'image et consignées dans le sidecar de vérité terrain, mais elles ne peuvent pas figurer dans le JSON strict sans changement de schéma (hors périmètre et non demandé). Le JSON attendu, lui, ne comprend que les champs du schéma actuel.

Le sidecar JSON a été validé par `Invoice.model_validate`; les vérifications arithmétiques Pydantic passent. Cette validation ne teste ni l'image, ni OCR, ni le modèle, ni le grounding.

### Résultats de canary

| Étape/champ | Attendu | Observé |
|---|---|---|
| OCR texte / positions | Texte visible ci-dessus et deux rangées distinctes | Non capturé; le wrapper ne le renvoie pas |
| Fournisseur | `Atelier Lumen SARL` | Non capturé |
| Date | `2026-11-06` | Non capturée |
| Ligne 1 | Cahier A4; 2; 12.5; 25.0 HT | Non capturée |
| Ligne 2 | Stylo bleu; 3; 4.0; 12.0 HT | Non capturée |
| Total HT / TTC | 37.0 / 44.4 | Non capturé |
| TVA, client, numéro, devise | Imprimés; non représentés dans le JSON strict actuel | Non comparés |
| Sortie brute, décisions de grounding, JSON final | Une réponse du modèle et le payload issu du pipeline réel | Quatre sorties ont été retournées par le client, mais leurs valeurs ont été perdues avant affichage/sauvegarde; non capturés |
| Durée, pic VRAM, erreurs/warnings | Mesures du véritable appel GPU | Temps client total 31,494 s; pic VRAM, durée GPU, quota et résultat backend non capturés |

Il n'y a donc **aucun résultat OCR-only, mocké ou simulé présenté comme end-to-end**. L'invocation du callback est constatée, mais aucun résultat n'est revendiqué comme preuve de génération Qwen ou d'exactitude.

### Exécution autorisée — une seule tentative

- L'utilisateur a autorisé un canary best-effort sur le Space actuel, avec téléchargement possible si le cache était froid. Le runtime a été re-vérifié avant l'appel : toujours `RUNNING`/`READY`, SHA `927a115...`.
- Le client Gradio, sans token HF local et avec analytics désactivées, a appelé une fois `run_extraction_ui` avec le PNG synthétique. `Client.predict` est revenu sans exception en **31,494 s** et le résultat avait **quatre composants**, correspondant à la sortie du callback.
- La commande a ensuite échoué en imprimant les composants : `UnicodeEncodeError` de l'encodage console Windows `cp1252` sur des caractères français. L'objet de retour était en mémoire dans le processus terminé mais n'a pas été écrit dans un fichier. **Aucune nouvelle requête ne sera faite sous l'autorisation d'une seule canary.**
- Cette erreur d'affichage locale ne prouve pas une erreur du Space. Cependant, les valeurs des quatre composants sont perdues; impossible de dire si le modèle a été utilisé, si l'appel s'est terminé en OCR seul, ou si le JSON retenu correspond au ground truth.
- Le code déployé `run_recruiter_extraction_ui` ne retourne que le JSON, le résumé compact, les performances et le fichier téléchargeable; il écarte les sorties détaillées `details`, `advanced`, `first_pass_raw` et `validation` du helper interne. Ainsi, le texte OCR, l'entrée exacte du modèle, la réponse brute et les statuts détaillés de grounding ne sont pas disponibles via l'API active, même si le client est configuré en UTF-8.
- Aucun modèle brut, quota consommé, téléchargement éventuel, durée GPU ou pic mémoire ne peut être confirmé. Le dernier statut public après l'appel reste `RUNNING`/`READY`; les logs actuels demeurent inaccessibles via les endpoints anonymes déjà vérifiés.

## 4. Gate d'autorisation avant le canary

### Capacité apparente

Le Space **semble capable en principe** : son conteneur est RUNNING, la même révision a déjà préchargé adaptateur et base et le callback complet est câblé. Une invocation du callback a rendu quatre composants, mais les valeurs ont été perdues par le client. Il n'est pas **prouvé prêt à l'inférence** : logs courants et cache non accessibles, quota actuel inconnu, provenance base non épinglée, utilisation Qwen non confirmée.

### Opération exécutée et coût observé

L'opération autorisée a été une soumission de `pr75-synthetic-invoice.png` au callback Gradio `run_extraction_ui` du Space actuel. Le callback peut télécharger poids/tokenizer s'ils ne sont pas dans le cache (maximum estimé pour les motifs actuels : **environ 2,503 Go**). Le cache courant n'est pas exposé; le téléchargement éventuel n'a pas pu être vérifié.

### Coût / quota estimé

- ZeroGPU : $0 direct dans le quota inclus; quota actuel non vérifié. Le dernier relevé historique disponible était 9,524590 s, tandis que le callback demande initialement 18 s, avec une durée bornée de 8–22 s.
- Si crédits payants éligibles : tarif documenté $1/10 min; 18–22 s correspondraient à environ **$0.03–$0.04** si la consommation est facturée exactement à cette durée. C'est une estimation, pas un coût garanti; chargement et ordonnancement peuvent modifier le résultat.
- Fichiers modèle : aucun téléchargement en PR75. Un cache froid pourrait déclencher environ 2,503 Go de récupération Hub; le téléchargement lui-même n'a pas de tarif Hub par opération confirmé dans cette revue.

### Autorisation reçue et limite restante

L'utilisateur a autorisé une canary best-effort sur le Space actuel, acceptant la base courante non prouvée et un téléchargement des poids si le cache était froid. Cette autorisation a été consommée par l'unique requête ci-dessus. Le quota actuel n'a pas été interrogé; la session peut avoir consommé du quota ou des crédits éligibles. Aucune nouvelle inférence n'est autorisée par cette décision.

Le manifeste d'entraînement (ou un nouvel accord sur la provenance) reste nécessaire pour revendiquer une reproduction exacte. Toute seconde requête, instrumentation, méthode autorisée de capturer les diagnostics bruts ou changement/déploiement du Space nécessitera une autorisation distincte. Aucun document réel ne sera envoyé.

## Décision finale

**PARTIAL GO — invocation du callback uniquement.** Le Space a accepté une requête et le client a reçu quatre composants, mais la sortie a été perdue; Qwen, grounding et validité des valeurs ne sont pas vérifiés. Ce résultat ne satisfait pas le critère **GO** et n'autorise pas les tests de factures réelles. Le blocage observé est la capture/visibilité des résultats; aucune incompatibilité du modèle n'est démontrée. Une seconde tentative avec capture UTF-8, ou tout instrument diagnostique/déploiement, demande une nouvelle autorisation. Après un canary dont les valeurs sont effectivement comparées au ground truth, les factures FR/UK/US historiques resteraient des régressions de développement, non un benchmark indépendant.

## Handover

- **Branche / HEAD :** `wip/layouts-reprise`, `9d7a641a52c114484ee614730d15882dbaa35d66`, branche en avance de deux commits. Modifications préexistantes conservées.
- **Fichiers créés :** `PRODUCT_REVIEW_75_PIPELINE_RECOVERY.md`, `tests/fixtures/pr75-synthetic-invoice.expected.json` et `tests/fixtures/pr75-synthetic-invoice.png` (ce dernier est local et ignoré par le motif global `*.png`). Aucune modification du code de production.
- **Validation :** le PNG s'ouvre en 1600 × 2200 et son SHA-256 a été calculé; le JSON attendu passe `Invoice.model_validate` et l'arithmétique. Une unique requête `run_extraction_ui` a été exécutée; le client a retourné quatre sorties en 31,494 s, puis échoué en sérialisant vers la console cp1252. Aucune sortie n'a été conservée; aucune comparaison de champs n'a été faite.
- **Logs :** API build anonyme 401; API container 404; les logs PR60 existants montrent un démarrage et un préchargement historiques, pas une inférence. Aucun credential n'a été consulté.
- **Blocages :** cache courant inaccessible; quota consommé inconnu; `adapter_config.revision=null`; manifeste de fine-tuning absent des emplacements vérifiés; les valeurs du callback ne sont pas capturées et la réponse brute/les diagnostics internes ne sont pas exposés par le wrapper public.
- **Prochaines étapes :** fournir le manifeste ou accepter explicitement la base best-effort; corriger le harness de capture local (UTF-8) et obtenir une nouvelle autorisation avant toute seconde canary. Pour capturer OCR brut, entrée/sortie modèle et décisions détaillées, proposer d'abord une instrumentation non-production avec tests et autorisation distincte; ne pas modifier ou redéployer le Space sans accord. Ne pas lancer de factures réelles avant un canary dont la sortie a été conservée et comparée.
