# PRODUCT-REVIEW-74 — Sélection du runtime GPU et plan de récupération

**Date :** 2026-11-06  
**Décision :** NO-GO pour l'inférence réelle pendant PR74. Le Space est disponible sans modification, mais le quota courant et sa mémoire GPU effective ne sont pas établis. La révision de base du fine-tuning reste inconnue (`revision: null`). PR75 peut préparer un essai technique après décision sur cette provenance et autorisation des téléchargements/du runtime.

## Résumé décisionnel

- Le Space public `nicolbl95/LLM-extraction-finetune` est actuellement **RUNNING**, domaine **READY**, révision `927a115997ef2f850255e1b14cfe37fe861bb710`, matériel demandé/courant `zero-a10g`. Ce statut a été lu via les endpoints metadata/runtime; aucune page n'a été ouverte et aucune fonction GPU n'a été appelée.
- Les données de l'adaptateur à sa révision `a1e56cc8c313cb194395047332cd8df2a0e34e63` indiquent le modèle de base exact **`unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`**, quantification bitsandbytes 4-bit, rang 16, alpha 16, PEFT 0.21.2. La révision base de l'entraînement n'est pas épinglée (`revision: null`).
- **Correction à PR73 :** les 6,32 Go calculés avec `unsloth/Qwen2.5-3B-Instruct` non quantifié ne sont pas la meilleure cible pour cet adaptateur. Le dépôt BNB 4-bit référencé par `adapter_config.json` ferait environ **2,503 Go** avec les motifs de téléchargement de l'application, adaptateur inclus. Ne pas télécharger le dépôt non quantifié avant d'avoir une raison documentée.
- Pour un test contrôlé, l'option technique la plus fiable est un Pod Linux temporaire A5000 24 Go / 25 Go RAM : prix public affiché **$0.27/h**, soit environ **$0.27 pour 1 h** ou **$0.54 pour 2 h**, hors stockage/taxes/variation de disponibilité. Aucun achat ni démarrage n'a eu lieu.
- L'option la moins chère est le Space ZeroGPU existant (coût GPU affiché $0), mais elle dépend d'un quota et d'une file d'attente; la dernière mesure de quota documentée était de 9,524590 secondes. La capacité VRAM du `zero-a10g` spécifique n'est pas exposée par le catalogue matériel public.
- Le poste local conserve l'avantage de confidentialité et n'a pas de coût cloud, mais 4 Go de VRAM sous Windows avec bitsandbytes absent ne constituent pas une configuration d'inférence certifiée.

## 1. Space Hugging Face : état, capacité et relance

### État vérifié sans réveil

La consultation en lecture seule de l'API Space et de son endpoint `/runtime` a retourné :

| Propriété | Valeur |
|---|---|
| Visibilité / SDK | Public, Gradio |
| Runtime | `RUNNING`; domaine `nicolbl95-llm-extraction-finetune.hf.space` en `READY` |
| Révision Space | `927a115997ef2f850255e1b14cfe37fe861bb710`, identique à la révision recensée dans PR60 |
| Matériel demandé / courant | `zero-a10g` |
| Réplicas / garbage-collection | 1; `gcTimeout=172800` secondes |
| Environnement déclaré | Python 3.12, Gradio 6.30.0 |

Sources : [API Space](https://huggingface.co/api/spaces/nicolbl95/LLM-extraction-finetune), [API runtime](https://huggingface.co/api/spaces/nicolbl95/LLM-extraction-finetune/runtime). La requête de statut n'a pas ouvert le Space, déclenché une extraction ou alloué un GPU. `RUNNING` décrit le serveur web et **ne prouve pas** qu'une allocation ZeroGPU est active ni que le modèle est chargé.

### Mémoire ZeroGPU et limites

Le [catalogue matériel Hugging Face](https://huggingface.co/api/spaces/hardware) identifie `zero-a10g` comme **ZeroGPU**, avec ressources CPU dynamiques, RAM `-`, accélérateur `null` et coût affiché de $0 par minute. Il ne révèle pas de modèle de GPU ni de VRAM pour cette instance.

La documentation [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) actuelle décrit les tailles ZeroGPU génériques `large` (demi RTX PRO 6000 Blackwell, 48 Go) et `xlarge` (96 Go), mais les métadonnées de ce Space ne révèlent pas une taille qui permette d'associer ces valeurs au slug `zero-a10g`. La similitude du nom ne suffit pas à lui attribuer les 24 Go d'un A10G payé. **Mémoire utilisable par invocation : non vérifiée.** Si le Space reçoit la capacité ZeroGPU documentée, Qwen 3B BNB 4-bit devrait avoir de la marge; cela n'annule ni les risques de file d'attente ni de quota.

Les limites documentées par Hugging Face sont une file partagée et un quota quotidien lié au compte : 2 min pour un utilisateur non authentifié, 5 min pour un compte gratuit, 40 min pour PRO/Team et 60 min pour Enterprise; la priorité dépend aussi du quota restant. Les comptes PRO/Team/Enterprise peuvent acheter des crédits à **$1 par 10 min** après le quota inclus, soit **$6 par heure de GPU** (une session de 18–22 s représenterait environ $0.03–$0.04 si elle est facturée exactement à sa durée GPU). Cette dernière valeur est un calcul indicatif, pas une estimation du temps réellement consommé par une inférence. Le quota actuel de l'utilisateur n'a pas été interrogé.

### Erreurs antérieures et cause démontrée

- Le dernier contrôle de quota consigné dans [FINAL_PUBLIC_RELEASE_READINESS.md](./FINAL_PUBLIC_RELEASE_READINESS.md) a retourné **9,524590 secondes disponibles**, 0 seconde de dépassement, avec reset à `2026-11-05T20:10:45Z`. Cette observation historique n'est pas un relevé du quota actuel. Elle est inférieure à la durée GPU initiale de 18 s utilisée par l'application et insuffisante pour un essai reproductible; un faible quota réduit aussi la priorité en file.
- Les logs publics/staging déjà examinés ne montraient aucun marqueur reconnu de quota dépassé, OOM ou timeout. Ils contenaient deux traces `ValueError` dont la cause n'était pas récupérable. Il n'est donc **pas établi** que les erreurs précédentes aient été causées par un manque de mémoire ou par le modèle.
- [PRODUCT_REVIEW_60_PUBLIC_PATCH_RESULT.md](./PRODUCT_REVIEW_60_PUBLIC_PATCH_RESULT.md) documente trois avertissements non fatals `ValueError: Invalid file descriptor: -1` lors d'un hot reload Gradio; le Space est ensuite passé à RUNNING et a servi les endpoints HTTP 200. Ces avertissements ne sont pas une preuve d'échec du scheduler GPU.
- Le démarrage de la révision 927a a historiquement préchargé environ **131 Mo** d'adaptateur/tokenizer et **2,37 Go** de fichiers base, sans charger le modèle sur GPU ni lancer d'inférence. Le cache existe ou a existé lors de ce démarrage; sa persistance au runtime actuel n'est pas vérifiée.
- Risque technique possible, mais non démontré comme cause d'une erreur passée : `startup_prefetch()` télécharge les fichiers hors de l'appel GPU; `INVOICE_MODULE_PRELOAD` est désactivé par défaut et le modèle reste chargé paresseusement dans `get_extractor()` appelé depuis `@spaces.GPU`. La durée dynamique commence à 18 s, s'appuie ensuite sur le temps de génération passé et est plafonnée à 22 s. Un premier chargement du modèle dans cette fenêtre peut être serré. Aucun test n'a été fait pour vérifier ce risque.

### Peut-on relancer sans modifier le dépôt ?

**Techniquement oui, sous conditions**, sans redéploiement : le Space est déjà RUNNING sur la révision connue, son UI contient le flux d'extraction, et le préchargement historique a réussi. Il faut néanmoins que le cache de poids soit encore présent et que le quota/ordonnancement autorise la durée requise. Le premier chargement du modèle est toujours non validé; les logs disponibles ne prouvent pas que l'inférence ait déjà réussi. La relance et toute invocation GPU sont hors du périmètre PR74.

Les rapports précédents indiquaient qu'aucune variable publique n'était configurée; aucune valeur secrète n'a été lue. La configuration distante récupérée en texte brut reste Gradio 6.30.0 / Python 3.12 et les contraintes de dépendances du projet.

## 2. Provenance exacte du modèle

### Adaptateur

Le dépôt public `nicolbl95/llm-extraction-weights` est non privé et non gated. La révision `main` observée est `a1e56cc8c313cb194395047332cd8df2a0e34e63`. Les métadonnées du Hub indiquent :

| Fichier | Taille | SHA disponible |
|---|---:|---|
| `adapter_model.safetensors` | 119 801 528 octets | SHA-256 LFS `b45155157d66288ebcd054da2b29e5b07bbe4185c243b5a9d40b7cd519964d87` |
| `tokenizer.json` | 11 422 166 octets | SHA-256 LFS `6b4360dd6a184650ffc48056c2569bc603f896c5adfe94b10f1c79f809638aa5` |
| `adapter_config.json` | 1 334 octets | Git blob `98c39088eb86fd429c82ae7f69ef7410fc74f09d` |
| `tokenizer_config.json` | 4 412 octets | Git blob `e4920204ed8a492fc5912075a01451d56d9693bc` |
| `chat_template.jinja` | 2 507 octets | Git blob `bdf7919a96cfe43d50914a007b9c0877bd0ec27e` |

Le `adapter_config.json` correspondant à ce SHA précis a été lu (1 334 octets, **pas un poids**). Il déclare :

- `base_model_name_or_path = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"`;
- `revision = null`, `peft_type = LORA`, `peft_version = 0.21.2`, `r = 16`, `lora_alpha = 16`;
- classe de base `Qwen2ForCausalLM` et mêmes cibles q/k/v/o/up/gate/down que le notebook.

Référence : [adapter_config.json à la révision a1e56cc8](https://huggingface.co/nicolbl95/llm-extraction-weights/resolve/a1e56cc8c313cb194395047332cd8df2a0e34e63/adapter_config.json). Aucun fichier de poids n'a été téléchargé.

### Base réellement déclarée et incertitude sur le commit d'entraînement

Les métadonnées publiques de `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` décrivent `Qwen2ForCausalLM`, `quantization_config.load_in_4bit=true` et `quant_method=bitsandbytes`. Le `main` courant observé est le commit `21d7790b1332196a5cf65ba72f2b46b659f74ddf`, dernière modification publiée le 6 février 2025.

| Fichier base | Taille | SHA-256 LFS |
|---|---:|---|
| `model.safetensors` | 2 355 738 764 octets | `26e4f54c6213083a3861866bdf7ae9e84f74314fb4286095031643de391fccc5` |
| `tokenizer.json` | 11 421 896 octets | `9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa` |

Les fichiers correspondant aux motifs de préchargement de l'application totalisent **2 371 619 620 octets** pour la base. Avec les fichiers correspondant aux motifs adaptateur, le total annoncé est **2 502 851 567 octets (2,331 Gio)**. Ce sont les tailles du snapshot courant et non des fichiers présents ou vérifiés localement.

**Provenance manquante :** le champ `revision` du LoRA est `null`; aucun `training_manifest.json` n'est dans le dépôt adaptateur, et PR73 n'en a pas trouvé dans les `results/` locaux. Le commit `21d7790b...` est la meilleure cible actuelle pour le dépôt de base indiqué, mais il n'est pas prouvé qu'il s'agisse du commit exact consommé pendant le fine-tuning d'octobre 2026. Son ancienne date de modification rend le même snapshot plausible, pas certain. Il faut retrouver le manifeste original ou obtenir l'accord explicite d'effectuer un essai avec le `main` actuel comme reconstruction best-effort.

**Correction de PR73 :** PR73 avait calculé le volume à partir du dépôt `unsloth/Qwen2.5-3B-Instruct` non quantifié du notebook. Le `adapter_config.json` actuel référence au contraire la variante BNB 4-bit ci-dessus. L'API Space `models` peut également lister le nom générique non quantifié; pour l'association LoRA, la valeur `base_model_name_or_path` de l'adaptateur est la référence plus directe. Ne pas substituer le dépôt non quantifié à la base déclarée.

### Quantification et dépendances

Le dépôt base indique bitsandbytes 4-bit; l'inférence applicative passe `load_in_4bit=True`, NF4, double quantification, dtype bf16 si supporté sinon fp16, contexte 4096 et maximum de génération 2048. PEFT applique l'adaptateur. Cette combinaison est cohérente au niveau des métadonnées, mais n'a pas été initialisée ni testée.

Le `requirements-inference.txt` du Space inclut `requirements.txt` et Gradio 6.30.0. Les bornes modèle sont PyTorch `>=2.8,<=2.13`, Transformers `>=4.49,<6`, PEFT `>=0.14`, Accelerate `>=1.2`, bitsandbytes `>=0.48` sur Linux. Le Space annonce Python 3.12. Le `requirements.txt` ne constitue pas un lock complet : les versions résolues du runtime n'ont pas été vérifiées. L'environnement local PR72/73 avait Torch 2.14.1 (hors borne) et pas bitsandbytes.

**Cible candidate d'environnement isolé :** Linux x86-64, Python 3.12, une roue CUDA de Torch dans la borne projet, Transformers 4.49–<6, PEFT 0.21.2 (version inscrite dans le LoRA), Accelerate >=1.2 et bitsandbytes >=0.48. Cette pile doit être résolue et verrouillée séparément après autorisation; aucune installation n'a eu lieu. Pour RunPod, rester dans les bornes projet permet de conserver la configuration attendue, même si le plafond Torch a été imposé principalement par ZeroGPU.

## 3. Comparaison des environnements

| Critère | A. Space ZeroGPU existant | B. RunPod Pod temporaire A5000 | C. RTX 3050 locale |
|---|---|---|---|
| GPU / VRAM | `zero-a10g`; VRAM exacte et accélérateur non exposés pour ce Space. ZeroGPU générique documente 48/96 Go, sans taille reliée au slug actuel. | RTX A5000, **24 Go VRAM** selon l'offre publique consultée. | RTX 3050 Laptop, **4 Go VRAM**. |
| RAM système | Dynamique/non publiée par le catalogue pour `zero-a10g`. | Offre listée avec **25 Go RAM**, 9 vCPU. | **7,79 Go utilisables**. |
| Modèle | Dépôts publics; préchargement historique réussi, cache actuel inconnu. Scheduler, quotas et chargement paresseux limitent la fiabilité. | Adaptateur BNB 4-bit et base déclarée compatibles en principe; réserve de VRAM suffisante, à valider par un chargement autorisé. | La taille disque est suffisante (79 Go libres), mais 4 Go VRAM est marginal et bitsandbytes est Linux-only dans les dépendances du projet. CPU FP32 dépasse la RAM. |
| Dépendances | Python 3.12 / Gradio 6.30; contraintes projet pour Torch, Transformers, PEFT, Accelerate et bitsandbytes Linux. Versions exactes chargées non connues. | Environnement Linux/Python 3.12 à préparer; mêmes bornes, BNB pour la quantification. | Windows actuel : Torch 2.14.1 hors borne et BNB absent. WSL2/CUDA n'a pas été vérifié. |
| Coût publié | **$0** par minute GPU pour ZeroGPU; quota partagé. Des crédits peuvent coûter $1/10 min au-delà du quota pour les tiers admissibles. | RunPod affiche **$0.27/h** pour RTX A5000. À ce tarif : 1 h = **$0.27**, 2 h = **$0.54**. Le tarif final dépend de l'offre/zone et doit être confirmé avant lancement. | Pas de location; coût d'électricité non calculé. |
| Accès | Space public; quota lié au compte, file partagée, cache modèle non garanti. | Compte/facturation fournisseur et disponibilité du GPU nécessaires; aucun accès n'a été testé. | Déjà local; mise en place Linux/WSL2 et compatibilité GPU à établir. |
| Complexité | Faible si quota/cache sont prêts; aucun changement de code ne semble requis pour une invocation, mais le test serait peu contrôlable. | Moyenne : créer un pod jetable, installer l'environnement, récupérer les révisions, vérifier les hashes et exporter les résultats. | Moyenne/élevée : WSL2, CUDA et BNB, avec risque de VRAM insuffisante. |
| Risque principal | Quota/scheduling, durée de chargement dans 18–22 s, état du cache persistant inconnu. | Dépendance réseau/package, coût pendant installation, sécurité des factures envoyées à un tiers. | OOM/initialisation GPU, incompatibilité Windows/BNB, RAM CPU insuffisante. |
| Logs/résultats | Logs runtime Space, rétention/persistance non garantie; ne pas y exposer de données sensibles. | Volume privé temporaire possible; exporter les journaux puis arrêter/supprimer le pod. Stockage facturé séparément selon configuration. | Contrôle local complet; conserver dans un emplacement local restreint. |

Sources tarifaires consultées :

- La page [Hugging Face Pricing](https://huggingface.co/pricing) affiche ZeroGPU gratuit; A10G-small payé à **$1.00/h** (4 vCPU, 15 Go RAM, 24 Go VRAM). Cette option payée est distincte du `zero-a10g` gratuit actuel et demanderait une modification matérielle ou un autre Space; elle n'est pas recommandée comme premier choix coût.
- La page [RunPod Pricing](https://www.runpod.io/pricing), mise à jour le 4 novembre 2026, affiche l'A5000 24 Go/25 Go RAM/9 vCPU à $0.27/h. Les montants 1 h/2 h ci-dessus sont des calculs à partir du prix affiché, pas une facture/garantie de disponibilité. Le stockage de volume/container est indiqué séparément (à partir de $0.10/Go/mois sur la page consultée).

### Évaluation du poste local

Les 4 Go de VRAM pourraient suffire à un modèle 3B quantifié dans certains contextes, mais le fichier BNB fait 2,36 Go et l'application doit encore conserver les buffers, les activations, le cache de contexte et les allocations CUDA. La carte partage aussi ses ressources avec Windows. Cette configuration est donc **non certifiée et à risque d'OOM**. Le chemin CPU de l'application est FP32; il ne constitue pas un repli adapté aux 7,79 Go disponibles. Ne pas classer l'option locale comme impossible, mais ne pas la choisir comme plan fiable sans un préflight spécifique autorisé.

## 4. Plan de test futur (non exécuté)

1. **Verrouiller la provenance avant téléchargement.** Chercher le `training_manifest.json` original auprès du système ayant fait le fine-tuning. L'adaptateur actuel est fixé au SHA `a1e56cc8...`; le modèle BNB courant est au SHA candidat `21d7790b...`, mais son commit d'entraînement n'est pas établi. Si le manifeste est introuvable, demander l'accord explicite pour tester avec ce snapshot `main` candidat sans l'appeler reproduction exacte.
2. **Choisir le runtime.** Priorité recommandée : un Pod Linux temporaire A5000, une fois autorisés les dépenses et le transfert éventuel des données. Repli sans coût : Space existant seulement après contrôle de quota/cache et acceptation des risques ZeroGPU. Choix privé : test local seulement après validation WSL2/CUDA/BNB. Ne pas modifier le Space actuel.
3. **Préparer et vérifier les poids.** Dans un cache/volume privé, récupérer les commits autorisés de l'adaptateur et de la base BNB 4-bit; activer ensuite `HF_HUB_OFFLINE=1`; vérifier les SHA-256 LFS ci-dessus. Ne pas utiliser un modèle non quantifié ou un autre checkpoint.
4. **Préflight technique sans facture.** Initialiser `LoRAInference` et enregistrer les versions, commit Space, commit adaptateur/base, quantification, device et temps de chargement. Effectuer une seule petite sonde JSON sans données de facture, avec uniquement des champs `null` si une valeur est inconnue, puis valider avec le validateur JSON strict existant. Ne pas accepter de valeurs de démonstration comme résultat métier.
5. **Essai de pipeline, après autorisation séparée.** Utiliser une facture historique de développement choisie et identifiée comme non indépendante. Pour un GPU distant, obtenir avant envoi une autorisation explicite de traiter le document chez ce fournisseur; privilégier une facture anonymisée ou un cas non sensible si possible.
6. **Capturer les preuves.** Dans un dossier privé et restreint, conserver l'OCR brut/TSV et coordonnées, texte et cellules reconstruits, qualité et consensus, diagnostics/raisons de rejet, sortie brute du modèle, statut de grounding/validation, JSON final, horodatages, temps mural, temps GPU, pic VRAM, RAM hôte, quota consommé et coût. N'envoyer aucun contenu facture dans des logs publics; exporter les résultats puis arrêter le runtime.
7. **Non-régression.** Ne modifier ni le schéma JSON, ni OCR, seuils, consensus, grounding ou preuves; aucune candidate géométrique rejetée en PR67/PR68. Une erreur de quota/worker doit être enregistrée comme échec technique, jamais comme succès OCR-only.

## 5. Recommandation et autorisations pour PR75

### Environnement recommandé

**Option B — Pod RunPod A5000 temporaire**, pour une première vérification reproductible et non limitée par une petite VRAM/queue ZeroGPU. Le coût indicatif est **$0.27 par heure d'exécution** (environ $0.27–$0.54 pour 1–2 h), hors stockage et toute variation fournisseur. Créer le volume uniquement pour la session, y télécharger les fichiers autorisés, garder les logs privés et arrêter le Pod dès l'export.

Cette recommandation est conditionnelle à deux points : (1) retrouver ou accepter explicitement l'absence de preuve du commit base d'entraînement; (2) obtenir l'autorisation de transfert des données si une facture réelle est envoyée au fournisseur. Pour minimiser le coût, tester d'abord le Space existant seulement si le quota a été réinitialisé et si le cache est encore disponible; son résultat restera moins déterministe.

### Autorisations requises

Avant PR75, demander confirmation pour :

1. Fournir le manifeste original de fine-tuning, ou autoriser une expérience best-effort avec la base BNB `main` observée à `21d7790b...` sans la présenter comme reproduction exacte.
2. Choisir le Pod distant (ou autoriser une lecture courante du quota Space et tenter ultérieurement le Space sans le modifier); confirmer le budget de 1–2 heures au tarif affiché et le plafond de coût.
3. Télécharger environ **2,503 Go** de fichiers modèles vers un stockage temporaire privé et installer/résoudre un environnement Linux/Python 3.12 isolé. Aucun poids ni paquet n'a été récupéré/ installé en PR74.
4. Traiter une facture historique sur un fournisseur externe, le cas échéant. Sans accord de confidentialité/transfert, ne pas téléverser de document réel; limiter le préflight à la sonde sans facture ou utiliser un environnement local.
5. Exécuter un chargement modèle et une sonde JSON, puis obtenir une autorisation distincte pour tout test fonctionnel de facture.

## Décision finale

**NO-GO pour préparer/lancer maintenant une inférence réelle.** Le runtime recommandé et le budget de référence sont identifiés, mais l'empreinte exacte du modèle base utilisée en entraînement manque (`revision: null`), le cache actuel du Space n'est pas vérifié et la capacité effective de `zero-a10g` n'est pas publiée. **GO conditionnel pour PR75** une fois la provenance ou l'acceptation best-effort confirmée, le runtime/budget choisi et les autorisations de téléchargement, installation et données obtenues.

## Handover

- **État Git au départ :** `wip/layouts-reprise`, HEAD `9d7a641a52c114484ee614730d15882dbaa35d66`, branche en avance de deux commits. Les modifications préexistantes ont été préservées.
- **État des opérations :** seules lectures locales, API de statut/métadonnées et lecture du petit `adapter_config.json`; aucun poids téléchargé, aucune installation, allocation GPU, inférence, appel payant, déploiement, commit ou push.
- **Fichier créé :** `PRODUCT_REVIEW_74_GPU_RUNTIME_SELECTION.md`, synthèse de l'état Space, coûts, provenance corrigée, risques et protocole proposé.
- **Tests :** aucun test applicatif/model/GPU ou facture n'a été exécuté; aucune validation de chargement réelle n'est revendiquée.
- **Blocages :** révision base d'entraînement non enregistrée; quota/cache Space actuels inconnus; `zero-a10g` n'expose pas sa VRAM/RAM; système local Windows/4 Go non certifié; données de facture non autorisées pour transfert externe.
- **Suite exacte :** obtenir les décisions énumérées en section 5, retrouver le manifeste d'entraînement, puis préparer seulement l'environnement isolé et le préflight sans facture. Attendre une autorisation distincte avant l'inférence réelle ou les tests de factures.
