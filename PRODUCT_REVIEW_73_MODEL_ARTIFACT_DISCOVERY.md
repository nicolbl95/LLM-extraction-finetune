# PRODUCT-REVIEW-73 — Découverte des artefacts modèle et plan d'inférence

**Date :** 2026-11-06  
**Décision :** NO-GO pour lancer le pipeline complet. Les métadonnées des poids distants sont disponibles et les révisions courantes sont identifiées, mais aucun poids local n'a été trouvé dans les chemins du workspace inspectés. Le cache Hugging Face standard est hors workspace et n'a pas été consulté.

## Résumé exécutif

- **Adaptateur attendu :** `nicolbl95/llm-extraction-weights`, révision `main` observée à `a1e56cc8c313cb194395047332cd8df2a0e34e63`. Le dépôt est public et non gated. Son poids LoRA fait 119 801 528 octets; son SHA-256 LFS est connu.
- **Modèle de base attendu :** `unsloth/Qwen2.5-3B-Instruct`, révision `main` observée à `7548fff1f997f57b2e9e8ab1ec7be96949b00ed0`. Le snapshot comprend deux shards BF16 totalisant 6 171 926 992 octets.
- **Volume probable de récupération :** environ 6,319 Go décimaux (5,885 Gio) avec les motifs utilisés par l'application, modèle de base et adaptateur réunis. Ce sont des tailles déclarées par l'API du Hub, pas des fichiers téléchargés ou vérifiés localement.
- **Compatibilité :** la cible de base du notebook correspond à la famille du dépôt de base. La compatibilité exacte de l'adaptateur, son `base_model_name_or_path` et la révision de base utilisée à l'entraînement ne peuvent pas être certifiés sans son `adapter_config.json` et le manifeste original; ces contenus n'ont pas été téléchargés.
- **Machine :** CPU seul insuffisant pour le chemin FP32 observé. La RTX 3050 Laptop 4 Gio pourrait être très contrainte en 4-bit, mais n'a pas été testée. L'environnement courant est Windows, Torch 2.14.1 hors borne et bitsandbytes absent.
- **Conclusion A/B/C :** A n'est pas démontré, B est probable si le cache utilisateur est vide, C est la voie la moins risquée pour une exécution fiable, mais le GPU local n'est pas prouvé impossible. Aucun modèle, GPU, facture ou endpoint d'inférence n'a été utilisé.

## 1. Périmètre et méthode

Lecture complète de [AGENTS.md](./AGENTS.md) et [PRODUCT_REVIEW_72_ENVIRONMENT_RECOVERY.md](./PRODUCT_REVIEW_72_ENVIRONMENT_RECOVERY.md). Les emplacements locaux examinés sont les chemins d'artefacts du projet, `results/`, les snapshots locaux `.local-review/`, les variables d'environnement de cache et les résultats `.venv` documentés en PR72. La recherche dans `results/` et `.local-review/` était limitée aux noms `adapter_config.json`, `adapter_model.safetensors`, `model*.safetensors`, `pytorch_model*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.gguf` et aux manifestes de checkpoint; aucune recherche sur l'ensemble du disque n'a été faite.

Les seules requêtes distantes ont ciblé les métadonnées JSON publiques des deux dépôts Hugging Face :

- [Métadonnées de l'adaptateur à `main`](https://huggingface.co/api/models/nicolbl95/llm-extraction-weights/revision/main?blobs=true)
- [Métadonnées du modèle de base à `main`](https://huggingface.co/api/models/unsloth/Qwen2.5-3B-Instruct/revision/main?blobs=true)

Aucun poids, fichier de configuration brut, modèle ou donnée de facture n'a été téléchargé. Il s'agissait de requêtes de métadonnées, sans inférence ni calcul cloud/GPU.

## 2. Artefacts locaux et caches

| Emplacement | Résultat |
|---|---|
| Chemin applicatif `results/qwen25-3b-international-v3-20261021T053959288046Z/lora_adapter` | Absent. C'est le chemin `DEFAULT_ADAPTER` de [app/main.py](./app/main.py). |
| `results/` | Aucun `adapter_config.json`, poids `.safetensors`, `pytorch_model*.bin`, checkpoint `.pt`/`.pth`/`.ckpt`, GGUF, `training_manifest.json` ou `trainer_state.json` trouvé. Le seul sous-répertoire est `evaluation-3/`; son [finetuned_metrics.json](./results/evaluation-3/finetuned_metrics.json) référence l'ancien chemin d'adaptateur, mais ne contient pas les poids, hashes ni la provenance du checkpoint. |
| `.local-review/` | Aucun des fichiers modèle/manifeste recherchés dans ses snapshots. |
| `.venv/` | PR72 avait déjà cherché les artefacts de modèle dans ce répertoire sans trouver de poids/adaptateur complets. Les environnements qu'il contient ne sont pas un cache de modèle confirmé. |
| Variables de cache | `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME` et `TORCH_HOME` ne sont pas définies dans les portées Process/User/Machine interrogées. |
| Cache Hugging Face par défaut | Le cache Windows standard se situe sous le profil utilisateur (`.cache/huggingface/hub`), donc hors du workspace. Il n'a pas été inspecté : `AGENTS.md` limite l'accès aux fichiers du workspace. Des snapshots déjà téléchargés à cet emplacement restent donc possibles. |

### Manifeste et provenance d'entraînement

Le notebook [fine_tune_unsloth.ipynb](./notebooks/fine_tune_unsloth.ipynb) déclare `MODEL_NAME='unsloth/Qwen2.5-3B-Instruct'`, crée le run `qwen25-3b-international-v3-<timestamp>`, et sauvegarde `lora_adapter/` et `training_manifest.json` sous `results/<RUN_NAME>/`. Le manifeste doit contenir `base_model_revision` et les versions de paquets; il n'est pas présent dans le `results/` local.

Le dépôt Hub de l'adaptateur ne liste pas de `training_manifest.json`. Les métriques locales pointent vers le nom de run attendu, mais ne permettent pas de vérifier les poids ni la révision de base. Ainsi :

- Aucun snapshot complet n'est confirmé comme déjà présent dans le workspace.
- Aucune identité, révision ou somme de contrôle ne peut être attribuée à un artefact modèle local inexistant.
- La présence éventuelle d'un cache utilisateur hors workspace reste indéterminée.

## 3. Métadonnées Hugging Face

### Adaptateur LoRA

| Propriété | Métadonnée actuelle |
|---|---|
| Dépôt / visibilité | `nicolbl95/llm-extraction-weights`, `private=false`, `gated=false`, `disabled=false`; accès anonyme aux métadonnées et fichiers publics, sans approbation d'accès signalée. |
| Révision observée | `main` → commit `a1e56cc8c313cb194395047332cd8df2a0e34e63`; dernière modification `2026-10-21T18:50:26Z`. Utiliser le SHA complet plutôt que `main` pour une récupération reproductible. |
| Licence | L'API du dépôt adaptateur n'expose pas de champ de licence ni de fiche modèle. Les droits propres à l'adaptateur ne sont donc pas établis par ces métadonnées. |

Fichiers du dépôt à cette révision :

| Fichier | Taille | Identité exposée par l'API |
|---|---:|---|
| `adapter_config.json` | 1 334 octets | Git `blobId` `98c39088eb86fd429c82ae7f69ef7410fc74f09d`; contenu non récupéré. |
| `adapter_model.safetensors` | 119 801 528 octets | SHA-256 LFS `b45155157d66288ebcd054da2b29e5b07bbe4185c243b5a9d40b7cd519964d87`. |
| `tokenizer.json` | 11 422 166 octets | SHA-256 LFS `6b4360dd6a184650ffc48056c2569bc603f896c5adfe94b10f1c79f809638aa5`. |
| `tokenizer_config.json` | 4 412 octets | Git `blobId` `e4920204ed8a492fc5912075a01451d56d9693bc`; contenu non récupéré. |
| `chat_template.jinja` | 2 507 octets | Git `blobId` `bdf7919a96cfe43d50914a007b9c0877bd0ec27e`. |

L'application liste neuf `ADAPTER_FILES` en `allow_patterns`; cinq correspondances existent dans ce snapshot. Les fichiers requis par `check_adapter` (`adapter_config.json`, `adapter_model.safetensors`, `tokenizer_config.json`) sont listés, et `tokenizer.json` est présent. `special_tokens_map.json`, `added_tokens.json`, `vocab.json` et `merges.txt` ne figurent pas dans ce dépôt à la révision observée. Le code de téléchargement ne les impose pas individuellement, mais seul un contrôle de chargement ultérieur, autorisé, pourra confirmer le tokenizer effectif.

La taille totale des fichiers correspondant aux motifs d'adaptateur de l'application est **131 231 947 octets (125,15 Mio)**. Les tailles et SHA LFS viennent du JSON du Hub; pour les fichiers non-LFS, l'API ne fournit qu'un identifiant Git, pas le SHA-256 du contenu.

### Modèle de base

| Propriété | Métadonnée actuelle |
|---|---|
| Dépôt / visibilité | `unsloth/Qwen2.5-3B-Instruct`, `private=false`, `gated=false`, `disabled=false`. |
| Révision observée | `main` → commit `7548fff1f997f57b2e9e8ab1ec7be96949b00ed0`; dernière modification `2025-03-06T00:08:34Z`. |
| Architecture | `Qwen2ForCausalLM`, `model_type=qwen2`; fiche Hub déclare `Qwen/Qwen2.5-3B-Instruct` comme base. Le nombre de paramètres déclaré est 3 085 938 688 en BF16. |
| Licence | Fiche Hub `qwen-research`, avec lien vers la licence Qwen. Vérifier ses conditions d'usage avant un usage de production ou une redistribution. |

Fichiers de poids et empreintes annoncés :

| Fichier | Taille | SHA-256 LFS |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 3 968 658 944 octets | `67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2` |
| `model-00002-of-00002.safetensors` | 2 203 268 048 octets | `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d` |
| `tokenizer.json` | 11 421 896 octets | `9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa` |

Les deux shards pèsent **6 171 926 992 octets**. Avec les fichiers correspondant aux motifs de préchargement base de l'application (`*.json`, `*.safetensors`, `*.txt`, `*.jinja`), la taille annoncée est **6 187 842 759 octets**. L'API fournit également des Git blob IDs pour les petits fichiers, mais pas leur SHA-256. Ces tailles décrivent le snapshot courant de `main`, pas nécessairement celui qui a servi à l'entraînement.

### Compatibilité et révisions à employer

La cible de base déclarée dans le notebook et le dépôt public de base concordent au niveau du nom de modèle et de l'architecture attendue. Cela rend le couple plausible, mais **ne prouve pas** que l'adaptateur publié déclare ce même `base_model_name_or_path` ni qu'il a été entraîné sur le commit `7548fff1...`: la fiche du dépôt adaptateur n'expose pas ces champs. Aucun fichier de configuration brut n'a été récupéré.

La cible de récupération recommandée, après vérification de l'`adapter_config.json`, est :

- Adaptateur : `nicolbl95/llm-extraction-weights@a1e56cc8c313cb194395047332cd8df2a0e34e63`.
- Base : `unsloth/Qwen2.5-3B-Instruct@7548fff1f997f57b2e9e8ab1ec7be96949b00ed0`, **uniquement si** la config ou le manifeste confirme que c'est bien la révision d'entraînement.

Le run de fine-tuning a une date de nommage le 21 octobre 2026, tandis que le `main` base observé n'a pas changé depuis mars 2025. Cette chronologie est compatible avec l'emploi de ce commit, sans le prouver. L'adaptateur ne contient pas le manifeste de provenance qui permettrait de lever l'incertitude.

Le code applicatif [app/main.py](./app/main.py) retombe sur `main` si `INVOICE_ADAPTER_REVISION` n'est pas fixé. Le chargement base dans [evaluate_finetuned.py](./src/evaluate_finetuned.py) utilise la révision enregistrée dans `adapter_config.json` ou `training_manifest.json`; sans l'une ou l'autre, `from_pretrained` retombe aussi sur la branche par défaut. Il n'existe pas de variable séparée pour imposer la révision base. Ne pas traiter les deux `main` comme des références immuables.

## 4. Environnement d'inférence

### Versions vérifiées

Versions relues dans le venv courant sans import du modèle :

| Paquet | Version actuelle | Contrainte projet | État |
|---|---:|---|---|
| Python | 3.12.0 | Environnement projet 3.12 | Compatible avec le plan du projet. |
| PyTorch | 2.14.1 | `>=2.8,<=2.13` | Hors borne supérieure; ne pas modifier ce venv. |
| Transformers | 5.21.0 | `>=4.49,<6` | Dans la plage déclarée, mais pas testé avec les poids visés. |
| PEFT | 0.21.2 | `>=0.14` | Dans la borne déclarée. |
| Accelerate | 1.15.0 | `>=1.2` | Dans la borne déclarée. |
| bitsandbytes | absent | `>=0.48`, marqueur Linux seulement | Requis par le chemin CUDA 4-bit actuel; indisponible dans cet environnement Windows. |
| huggingface-hub | 2.3.0 | Non épinglé par le projet | Présent; version exacte compatible avec la future pile non validée. |
| safetensors | 0.8.0 | Dépendance attendue pour les poids | Présent; aucune lecture de poids effectuée. |

Les contraintes de référence sont dans [requirements.txt](./requirements.txt). [constraints-hf.txt](./constraints-hf.txt) épingle surtout les paquets de déploiement/UI, pas une pile d'inférence Torch/PEFT complète. La borne `torch<=2.13` ne certifie pas à elle seule une combinaison de roues CUDA.

### Configuration minimale candidate (à isoler)

Le choix le plus directement aligné sur le code actuel serait :

1. **Linux x86-64 + CUDA**, Python 3.12, dans un environnement neuf et isolé; ne pas remplacer ni mettre à niveau `.venv`.
2. PyTorch dans `>=2.8,<=2.13` (choisir une version/roue CUDA disponible dans cette borne), Transformers `>=4.49,<6`, PEFT `>=0.14`, Accelerate `>=1.2`.
3. bitsandbytes `>=0.48` dans cet environnement Linux pour `load_in_4bit=True`; safetensors et huggingface-hub pour les fichiers et le cache.
4. Fixer les révisions SHA de l'adaptateur et de la base; précharger les fichiers dans un cache isolé puis utiliser `HF_HUB_OFFLINE=1`. Vérifier les hashes après récupération.

Les versions plancher ci-dessus sont celles du manifeste projet, pas une matrice d'intégration testée. Une résolution exacte et un `pip check` seraient à faire dans ce nouvel environnement uniquement après autorisation. `unsloth`, TRL et datasets sont nécessaires au notebook de fine-tuning, pas au chemin d'inférence applicatif examiné ici.

Dans le code, CUDA sélectionne le chargement 4-bit NF4 via bitsandbytes, une seule carte (`device_map={"": "cuda"}`), `n_ctx=4096` et `max_tokens=2048`. En CPU, le code utilise FP32 et n'appelle pas bitsandbytes; cela ne rend pas pour autant l'exécution possible sur la machine actuelle.

## 5. Capacités matérielles observées

Les faits statiques proviennent de l'audit PR72; aucun contrôle CUDA, allocation GPU ou chargement de modèle n'a été lancé pendant PR73.

- RAM physique : environ **7,8 Gio**.
- GPU : NVIDIA RTX 3050 Laptop, **4 Gio de VRAM**; Intel Iris Xe intégré.
- Chemin CPU : 3 085 938 688 paramètres en FP32 représentent environ **12,34 Go décimaux (11,49 Gio)** de poids bruts, avant tokenizer, activations et runtime. C'est supérieur à la RAM disponible; le mode CPU de l'application n'est donc pas raisonnable sur cette machine.
- Chemin CUDA 4-bit : il évite les poids FP32 complets, mais 4 Gio de VRAM laisse peu de marge pour le runtime, le contexte et Windows. Sa faisabilité n'a pas été établie; le manque de bitsandbytes et l'OS actuel empêchent un essai dans le venv courant.

**Recommandation prudente :** utiliser un runtime Linux/CUDA avec au moins 8 Gio de VRAM disponibles, sans considérer ce seuil comme une garantie formelle. Une installation WSL2 ne serait une option locale que si son accès CUDA est confirmé séparément. Le matériel externe GPU n'a pas été utilisé, et ses tarifs éventuels n'ont pas été recherchés.

## 6. Décision opérationnelle A/B/C

| Situation | Verdict | Justification |
|---|---|---|
| **A — Tous les poids sont déjà locaux** | **Non démontré.** | Aucun poids complet dans les emplacements projet contrôlés. Le cache Hugging Face par défaut, hors workspace, n'a pas été inspecté; il pourrait contenir des snapshots. |
| **B — Des poids manquent et doivent être récupérés** | **Probable, sous réserve du cache utilisateur.** | Le chemin adaptateur local est absent et le dépôt de résultats ne contient pas les checkpoints. Si le cache externe est vide, les motifs applicatifs demanderaient environ **6 319 074 706 octets (5,885 Gio)** au total. Les deux dépôts sont publics et non gated; le coût direct observé est stockage/réseau local, pas un service d'inférence. |
| **C — Un GPU externe est préférable** | **Voie la moins risquée, pas une impossibilité locale démontrée.** | Le CPU local est insuffisant. La RTX 3050 4 Gio pourrait éventuellement charger le modèle en 4-bit, mais Windows ne satisfait pas la dépendance Linux bitsandbytes du projet et cette capacité n'a pas été vérifiée. Un poste/runtime Linux CUDA plus généreux est le choix robuste; un service GPU payant dépendrait du fournisseur et de la durée, et n'a pas été évalué. |

**Solution directe recommandée :** d'abord faire vérifier par l'opérateur (ou autoriser l'inspection de) son cache Hugging Face standard. Si les snapshots complets et compatibles s'y trouvent, réutiliser l'adaptateur via `INVOICE_ADAPTER_PATH`, confirmer les révisions, et effectuer le préflight en mode hors-ligne. Sinon, après autorisation, récupérer uniquement les deux révisions figées ci-dessus, sous réserve de confirmer la config de l'adaptateur, puis créer l'environnement Linux isolé. Pour un parcours fiable, préférer un GPU Linux disposant d'une marge supérieure à la carte 4 Gio actuelle.

## 7. Actions nécessitant une autorisation

Pour rendre le pipeline opérationnel, il faut autoriser explicitement les actions nécessaires ci-dessous :

1. **Établir l'état du cache externe** : soit fournir le résultat d'un inventaire local de `%USERPROFILE%\.cache\huggingface\hub`, soit autoriser une dérogation au périmètre workspace pour cette lecture. Aucun fichier de ce cache n'a été ouvert ici.
2. **Confirmer la provenance de l'adaptateur** : autoriser la récupération ciblée de `adapter_config.json` (1 334 octets, non-poids) et, si disponible dans l'emplacement d'entraînement, du `training_manifest.json`. Vérifier le `base_model_name_or_path`, le champ `revision` et le commit d'entraînement avant de sélectionner la base.
3. **Récupérer les poids uniquement s'ils manquent** : autoriser le téléchargement des fichiers des deux snapshots immuables — jusqu'à environ **6,319 Go décimaux** avec les motifs actuels — puis vérifier les SHA-256 LFS annoncés. Les poids ne sont pas gated selon les métadonnées actuelles.
4. **Préparer l'environnement isolé** : autoriser la création d'un nouvel environnement Linux/Python 3.12 et l'installation/résolution de Torch, Transformers, PEFT, Accelerate et bitsandbytes sans modifier le `.venv` existant.
5. **Choisir le matériel** : décider entre un essai local Linux/WSL2 si CUDA est disponible et suffisamment de VRAM est libre, ou un runtime GPU local/externe plus généreux. Aucun fournisseur ni prix n'est présumé.
6. Après disponibilité des poids et validation de l'environnement, demander une autorisation distincte avant tout chargement de modèle/inférence ou test de facture.

**Aucune de ces actions n'a été exécutée dans PR73.** Pas de téléchargement, installation, inférence, appel GPU, traitement de facture, modification de production, commit ou push. PR74 et les tests de factures ne sont pas commencés.

## Handover

- **Branche / HEAD :** `wip/layouts-reprise`, `9d7a641a52c114484ee614730d15882dbaa35d66`, branche en avance de deux commits.
- **État Git :** modifications suivies et fichiers non suivis préexistants conservés; seul ce rapport a été ajouté pour PR73.
- **Actions :** lecture complète des instructions et rapport PR72; inventaire ciblé de `results/`, `.local-review/`, variables d'environnement et sources de modèle; lecture des métadonnées HF adapter/base sans récupération de poids; relecture des versions du venv.
- **Tests/commandes :** recherche locale limitée par noms d'artefacts; interrogation de `importlib.metadata` du venv (versions listées ci-dessus); agrégation des tailles depuis les métadonnées. Aucun test applicatif, test OCR, téléchargement ou inference.
- **Blocages :** cache HF standard hors workspace non inspecté; fichier `adapter_config.json`/manifeste d'entraînement absent localement; révision de base réellement utilisée non confirmée; pile Windows incompatible avec le chemin CUDA 4-bit courant; GPU local 4 Gio non qualifié.
- **Prochaines étapes exactes :** obtenir les autorisations 1–4 ci-dessus; vérifier le cache et la config/manifestes; si nécessaire récupérer les SHA exactes; établir un environnement Linux isolé; vérifier la disponibilité et la marge CUDA sans lancer de facture; demander une autorisation distincte avant toute inférence.
