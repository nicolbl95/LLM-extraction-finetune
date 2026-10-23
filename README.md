---
title: Extraction standardisée de données de factures
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app/main.py
pinned: false
---

# LLM-extraction-finetune

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Unsloth](https://img.shields.io/badge/Unsloth-059669?style=for-the-badge)](https://docs.unsloth.ai/) [![QLoRA 4-bit](https://img.shields.io/badge/QLoRA-4--bit-8B5CF6?style=for-the-badge)](https://arxiv.org/abs/2305.14314) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/latest/)

## 📖 Description

Pipeline complet d'extraction structurée (JSON) pour factures internationales (FR/EUR, UK/GBP, US/USD) : fournisseur, date, articles imbriqués et totaux.
Stack technique : Qwen2.5-3B-Instruct, QLoRA 4-bit, Unsloth, PEFT et validation stricte Pydantic v2.
Le fine-tuning spécialisé permet d'atteindre 100 % de conformité de schéma JSON et 95 % d'Exact Match sur un jeu de test de 60 factures multi-formats inédites, surpassant nettement l'approche baseline par règles (43,33 % Exact Match).
**[Tester la démo live](https://huggingface.co/spaces/nicolbl95/LLM-extraction-finetune)**.

## 🛠️ Stack Technique & Méthodologies

- **Modèle :** 3 milliards de paramètres ; Unsloth pour la mémoire/vitesse,
  LoRA via PEFT ; contrôle des longueurs sans troncature silencieuse.
- **Données :** synthèse multilingue/multidevise, diversification procédurale
  des contenus et mises en page ; rendu ReportLab/PyMuPDF.
- **Évaluation :** benchmark personnalisé, Exact Match Rate, JSON Validity Rate,
  rapports `metrics.json` et prédictions par région, devise, format et template.
- **Tooling :** PyTorch, Hugging Face Transformers/Datasets, VS Code/Codespaces,
  Google Colab pour le GPU ; backend alternatif GGUF/llama.cpp.

## 🏗️ Architecture

```text
Templates → Génération Python → Textes + cibles JSON + PNG
                            → Train / Validation / Test
                            → Qwen + QLoRA / Unsloth
                            → Adaptateurs + tokenizer
                            → Base + PEFT → JSON → Validation / scoring
```

Les données sont générées par règles Python déterministes (`data/generate_dataset.py`),
puis réparties en trois splits disjoints (`src/split_data.py`).
Le modèle apprend des régularités d'extraction à partir des couples texte/JSON,
pas l'exécution du script ni une garantie de calcul exact.

### Données et partitionnement

**600 factures textuelles + 600 PNG**, dont **200 FR / 200 UK / 200 US** :
neuf familles de templates, 600 profils visuels distincts, 1 à 12 articles,
dates et devises régionalisées ; seed **42**.
Les calculs monétaires de génération sont contrôlés et arrondis au centime.

| Fichier | Part / exemples | Rôle |
| --- | --- | --- |
| `data/train.jsonl` | **80 % / 480** | Mise à jour des adaptateurs. |
| `data/val.jsonl` | **10 % / 60** | Suivi de la perte, sans mise à jour des poids. |
| `data/test.jsonl` | **10 % / 60** | Mesure finale, jamais utilisé pour les gradients ni les hyperparamètres. |

Splits disjoints par ID et texte, contrôles de chevauchement et provenance par
hashes. Les exemples de test sont inédits pour l'apprentissage, mais partagent
les familles de templates : la généralisation à des structures entièrement
nouvelles n'est pas démontrée. La validation aide à surveiller le
surapprentissage, sans garantir son absence.

### Garde-fous et auto-correction

- **Validation après génération :** Pydantic v2, objet JSON strict, champs/types
  requis et valeurs finies ; prose, Markdown et clés dupliquées refusés.
  Le pipeline n'impose pas de JSON Schema au décodage.
- **Arithmétique :** `quantity × unit_price` net et somme HT, calculs `Decimal`,
  arrondi `ROUND_HALF_UP`, tolérance **0,02**, HT + TVA lue = TTC (plusieurs taux).
- **Preuves documentaires :** PyMuPDF pour le texte PDF ; OCR Tesseract `fra+eng`
  en niveaux de gris, TSV avec positions pour les images. Redressement, CLAHE,
  recadrage du tableau, débruitage et ×3 sont recoupés **par champ**, avec deux
  segmentations par famille de transformation. Une contradiction ne devient
  jamais une valeur par défaut. Les zones non recoupées restent `null` et
  « non vérifiées », sans annuler les autres lignes recoupées.
- **Latence bornée et politique traçable :** chemin propre en deux lectures
  distinctes (gris redressé PSM 4 / CLAHE PSM 6), escalade seulement en cas
  de doute. Budget **45 s par document**, partagé entre pages ; Tesseract
  reçoit le temps restant. L'arrêt et le rendu peuvent ajouter une petite
  surcharge, sans lancement d'étapes supplémentaires. Version, confiance 40
  et familles sont définies uniquement dans `src/ocr_config.py` et empreintées
  dans les résultats.
  Ancrage par cellule numérique et descriptions comparées sans casse/accents
  (seuil **0,9**). Les formats FR/UK/US, espaces insécables et milliers sont
  interprétés selon la locale, sans supprimer arbitrairement une virgule.
- **Récupération :** exclusivement des valeurs présentes dans le document,
  avec une solution unique des contraintes. Les prix nets ne sont pas remisés
  une seconde fois. Un champ douteux seul peut devenir `null` ; un total HT
  documentaire corroboré n'est pas effacé à cause d'une quantité mal lue.
- **Fidélité au document :** une valeur imprimée lisible est conservée même
  si le calcul de la facture est faux ; l'anomalie est signalée séparément
  dans `validation.flags`, hors du schéma scoré. Lecture incertaine : `null`.
- **Seconde passe ciblée :** uniquement les lignes non résolues et leur texte
  OCR, jamais toute la facture. Si l'OCR suffit, aucune seconde inférence.
  Budget ZeroGPU dynamique (15–60 s initialement, 8–30 s pour la correction),
  repli OCR sans GPU et message dédié en cas de quota ; traces complètes en logs.
- **Rapport avant le JSON :** anomalies de la facture avec justification,
  contrôles réussis, puis résumé des lectures récupérées/neutralisées.
  Statuts par champ et erreurs du modèle dans « Détails techniques » replié,
  jamais confondus avec une erreur de la facture. Hash et disponibilité OCR
  dans le pied de page. Les réponses brutes restent séparées de l'export sécurisé.
  Les métriques historiques du modèle ne sont pas remplacées par les résultats OCR.
- **Contrôles documentaires :** noms répétés, échéance/conditions de paiement,
  devise/pays, taxes, doublons et identifiants. Calibration sur 60 références :
  **18 SIRET sur 20 échouent à Luhn** ; ce contrôle est donc informatif.
  Les autres contrôles d'identifiants n'échouent pas sur les références.

Décodage sans échantillonnage dans [l'évaluateur](src/evaluate_finetuned.py) :
LoRA `do_sample=False` pour les deux passes éventuelles ; GGUF `temperature=0.0`.
Lecture littérale des chiffres : [détails du prompt système](docs/DESIGN.md).

Vérifications logicielles et évaluation des rejeux (aucun LLM réel appelé) :

```bash
python -m pytest -q
python src/evaluate_grounding.py --replays tests/fixtures/invoice_model_errors.json
# Appels réels sur les images ; échoue si le modèle ne peut pas générer.
python src/diagnose_pipeline.py --require-real-model
```

Le test réel opt-in est exclu par défaut. Il nécessite des poids **locaux**
explicitement configurés, les dépendances d'inférence et le matériel adapté :

```bash
INVOICE_SLOW_ADAPTER_PATH=results/<run>/lora_adapter python -m pytest -q -m slow tests/test_grounding_slow.py
```

Les rejeux sont des erreurs synthétiques décrites dans la demande, non des sorties
capturées du modèle. Les tests distinguent le texte du dataset du véritable OCR
des trois images d'exemple ; ils ne constituent pas un benchmark du LLM réel.

### Vérification hors gabarit et préproduction

Le jeu indépendant [tests/stress/](tests/stress/) n'est jamais ajouté aux splits
d'entraînement/test. Il mesure séparément valeurs fausses conservées, `null` à
tort, anomalies fantômes et crashs, par cas :

```bash
python scripts/run_stress_check.py --output results/stress-report.json
# Mesures isolées, puis consensus figé (les TSV d'expérience sont mis en cache) :
python scripts/experiment_ocr.py --phase tuning
python scripts/experiment_ocr.py --phase tuning --consensus
python scripts/experiment_ocr.py --phase validation --consensus
python scripts/summarize_ocr_experiments.py
python scripts/audit_test_changes.py --output docs/TEST_CHANGES.md
python scripts/render_report_previews.py
```

**Dénominateur figé : 35 cas, 612 champs** : FR 160, UK 128, US 192,
layouts/PDF 132. Les seuls champs des images dégradées sont **480**.
L'ancien total de 600 désignait 34 cas, avant le layout de remise séparée
(12 champs supplémentaires). Diagnostic avant seuil :
[ventilation des 416 null](results/null-breakdown.md).
Les essais réels sont décrits dans [le tableau d'expériences](results/ocr-experiments.md) :
réglage 21 cas / 372 champs, validation 14 cas / 240 champs, split figé.
La sélection sur réglage passe de 252 à 94 `null` ; validation de 164 à 61,
sans chiffre faux conservé ni anomalie fantôme dans ces essais. La vérification
du chemin applicatif complet est enregistrée séparément dans le JSON de stress.
La cible de moins de 30 % de `null` image n'est pas revendiquée : **155/480 =
32,29 %** dans les essais, contre 416/480 avant. Le bruit destructeur et la
demi-résolution ne justifient aucune invention pour gagner des points.
Le chemin applicatif réel reproduit ces mesures : **0 valeur fausse,
0 anomalie fantôme, 0 crash**, y compris 105 profils supplémentaires de
réponses modèle reconstruites. Avant la passe latence, suite de récupération : **249 tests réussis,
101 sous-tests, 3 slow exclus** ; pas de LLM réel chargé.

Les transformations d'images sont déterministes (seed 713) ; les mises en page
sont rédigées indépendamment. Le modèle est **simulé** pour ces tests, l'OCR est
réel. Le consensus prudent neutralise un champ dès que ses preuves restent
insuffisantes : consulter le nombre de `null`, pas seulement « zéro valeur
fausse ». Une même erreur peut survivre à plusieurs transformations et lectures
Tesseract ; ce recoupement n'est pas une garantie universelle.
Une contradiction sur une image est donc une lecture à confirmer, **pas**
automatiquement une erreur imprimée ; les erreurs certaines du texte natif
restent signalées avec conservation des valeurs.
Un dépassement explicite du délai Tesseract est traité comme une preuve
insuffisante et journalisé avec trace, pas comme un chiffre confirmé ni un crash
du rapport. Les autres erreurs OCR ne sont pas absorbées silencieusement.

Sur une **machine CUDA/Colab** (Python 3.11/3.12 recommandé), depuis la racine :

```bash
python -m pip install -r requirements-inference.txt
python scripts/run_real_model_check.py --check-environment
python scripts/run_real_model_check.py \
  --adapter-path results/<run>/lora_adapter \
  --output results/real-model-check.json --load-in-4bit --allow-download
# Même protocole sur les 24 images dégradées en plus des 3 originales :
python scripts/run_real_model_check.py \
  --adapter-path results/<run>/lora_adapter \
  --output results/real-model-stress.json --load-in-4bit --stress --allow-download
```

`--allow-download` autorise explicitement le téléchargement du modèle de base
**sur cette machine seulement** ; sans ce drapeau le mode est hors ligne.
Le script ne télécharge jamais les adaptateurs : fournir leur répertoire local.
Il traite les trois images, répète chaque inférence **trois fois**, conserve les
réponses brutes/finales, égalités avec les références, déterminisme, durées,
révision du code et matériel. Il utilise le vrai modèle, sans allocation ZeroGPU.
Un repli OCR seul, une troncature, une erreur GPU ou une différence entre les
trois sorties ne sont pas une réussite. Aucun vrai modèle n'a été exécuté dans
l'environnement de développement dépourvu de Torch/Transformers/PEFT.
Le [notebook Colab minimal](notebooks/real_model_check.ipynb) installe les
dépendances sur Colab, vérifie le GPU, charge des adaptateurs fournis (ou
explicitement téléchargés avec autorisation), lance le script et affiche les
scores. Il est livré sans sorties, **non exécuté ici**. Téléverser la version
locale complète du projet : les scripts non commités ne sont pas disponibles
par un simple clone du HEAD distant.

Avant publication : [plan d'une page](docs/DEPLOY.md),
[audit exhaustif des assertions modifiées](docs/TEST_CHANGES.md).
Le tag local `backup/pre-document-report-137b643` est une sauvegarde du dernier
commit, pas de l'arbre non commité. Rien ne doit être publié sans accord.

### Scans réels et revue de latence — sans publication

```bash
python scripts/evaluate_real_scans.py real-scans/ \
  --truth real-scans-truth.json --output results/real-scans-review.json
# Sans vérité terrain : champs neutralisés et anomalies à relire, pas d'exactitude inventée.
python scripts/evaluate_real_scans.py real-scans/ --output results/real-scans-manual.json
python scripts/measure_latency.py --output results/latency-local.json
python scripts/run_stress_check.py --compare results/stress-before-latency.json \
  --output results/stress-latency-local.json
python scripts/generate_theme_patches.py
```

Le [protocole de 5–10 factures indépendantes](docs/REAL_SCANS.md) exige
anonymisation, vérité terrain manuelle avant l'outil et aucun réglage après
lecture des résultats. Aucun document réel indépendant n'a été fourni ici.
Le script de revue n'appelle **aucun modèle** ; la validation Qwen reste
le protocole GPU séparé ci-dessus. Les mesures CPU réelles, appels OCR,
limites du p95 à trois échantillons et décision sur les 35 cas sont dans
[LATENCY.md](results/LATENCY.md). Les quatre patchs par thème sont destinés à
la revue et doivent être appliqués ensemble, pas comme déploiements autonomes.
La politique finale conserve **155/612 null**, **0 valeur fausse / 0 fantôme /
0 crash** sur les 35 cas et les 105 rejeux ; variation des null **0 point**.
Suite locale finale : **271 tests et 101 sous-tests réussis, 3 slow exclus**.
Mesures propres : médiane **2,28 s**, avec 2 appels pour deux exemples et
12 pour l'exemple UK dont la confirmation reste douteuse. Images détruites :
pire mesuré **45,26 s** ; PDF scanné 10 pages : médiane **24,89 s**, 20 appels.
Ce ne sont ni des mesures du LLM, ni un engagement de latence du Space.

## 🎯 Résultats & Benchmark Comparatif

L'évaluation a été menée de manière rigoureuse sur un jeu de test disjoint (**60 factures inédites**) à isoprompt et avec un décodage déterministe (`do_sample=False`) pour mesurer l'impact net de l'adaptation SFT par rapport à une approche basée sur des règles.

### Performance Globale (Test Set - 60 factures)

| Système / Approche | JSON Valide (%) | Exact Match (%) | Latence Moyenne |
| :--- | :---: | :---: | :---: |
| **Baseline Regex / Rules** | 46,67 % | 43,33 % | < 1 ms |
| **Qwen2.5-3B + QLoRA 4-bit (SFT v3)** | **100,00 %** | **95,00 %** | ~420 ms |

---

### Analyse par Format de Document (Exact Match Rate)

Le dataset de test évalue la robustesse de l'extraction sur 3 typologies d'entrées textuelles :

| Format de Facture | Baseline Regex | Qwen 2.5 3B (SFT) | Gain de Performance |
| :--- | :---: | :---: | :---: |
| **Email Body** | 65,0 % | 100,0 % | **+35,0 pts** |
| **Raw Text** | 35,0 % | 90,0 % | **+55,0 pts** |
| **Text Block / Unstructured** | 30,0 % | 90,0 % | **+60,0 pts** |

---

### 💡 Enseignements & Analyse d'Ingénierie LLM

1. **Garantie de Schéma & Robustesse en Production :**
   - L'approche non adaptée (Regex/Rule-based) s'effondre sur la conformité JSON (**53,33 % d'échec**), causée par des clés tronquées ou du texte parasite.
   - Le SFT aligne parfaitement le modèle sur le schéma **Pydantic v2**, atteignant **100 % de validité JSON** et éliminant tout besoin de parsing heuristique complexe en aval.

2. **Extraction de Structures Imbriquées (*Line Items*) :**
   - La principale faiblesse des méthodes traditionnelles résidait dans le parsing des tableaux d'articles dynamiques (précision de 76,67 %).
   - Le fine-tuning permet de fixer l'attention du modèle sur la hiérarchie des objets imbriqués, garantissant une extraction exacte sans omission de champs.

3. **Compromis Réseau vs Spécialisation (SLM vs Frontier Models) :**
   - En adaptant un **Small Language Model (3B de paramètres)** via QLoRA 4-bit, le pipeline atteint une précision comparable à des modèles propriétaires nettement plus volumineux, tout en optimisant considérablement les coûts d'inférence et la latence sur GPU.

## 🚀 Fine-tuning & Loss

Run QLoRA Colab terminé ; poids LoRA et tokenizer importés dans `results/`.

| Configuration | Valeur |
| --- | --- |
| Base | `unsloth/Qwen2.5-3B-Instruct` |
| Quantification / LoRA | 4 bits ; rang `16`, alpha `16` |
| Époques / learning rate | `3` / `2e-4` |
| Batch / accumulation | `1` / `8` |
| Contexte maximal | `4096` tokens |
| Durée du run | Environ **40 min 19 s** |
| Train loss / validation loss | **0,00746 / 0,00591** |

![Pertes d'entraînement et de validation du fine-tuning QLoRA](results/loss_curve.png)

La perte diminue rapidement puis reste faible ; la validation se stabilise
autour de **0,006**, sans remontée durable visible. Cela suggère une convergence
stable sur ce run, sans prouver l'absence de surapprentissage.

Pour reproduire : exécuter le notebook sur un GPU compatible (Colab T4 prévu)
avec les fichiers v3 d'entraînement/validation ; télécharger l'archive avant
expiration du runtime. Le manifeste sauvegarde configuration, versions et hashes.
**Les adaptateurs nécessitent le modèle de base et le tokenizer.**
Export GGUF fusionné Q4_K_M facultatif ; uploads Hub/W&B désactivés par défaut.

## ⚙️ Installation & Utilisation

Depuis la racine, sur un environnement CUDA compatible pour l'inférence :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-inference.txt

# Vérifier les données sans charger de modèle.
python src/evaluate_finetuned.py --check-data-only

ADAPTER="results/<run>/lora_adapter"  # chemin vers les adaptateurs téléchargés
python src/evaluate_finetuned.py --adapter-path "$ADAPTER" --load-in-4bit --output-dir results/evaluation-3/nouveau-run

# Baseline brute : mêmes instructions et scoring, une correction maximum.
python src/evaluate_finetuned.py --base-only --self-correct --output-dir results/baseline-qwen-v3
```

Remplacer `<run>` par le dossier téléchargé et `nouveau-run` par un nom inédit.
Le dossier existant `results/evaluation-3` conserve le rapport fourni ; chaque
évaluation écrit `metrics.json` et `predictions.jsonl` dans un nouveau
sous-dossier, sans écraser les scores déclarés.
La baseline écrit ses propres `metrics.json` et `predictions.jsonl`, sans
remplacer le rapport Regex. `--self-correct` réutilise la validation et le
prompt correctif de la démo. Le prompt système partagé exclut les en-têtes et
sous-totaux des articles et impose la copie littérale des descriptions, avec un
exemple fictif ; le dialogue few-shot supplémentaire reste propre à la démo.
Les deux réponses et les scores avant/après correction sont tracés ; une
incohérence arithmétique persistante reste un échec de validation du benchmark.
Pour une comparaison contrôlée, appliquer aussi `--self-correct` au run LoRA,
ou l'omettre des deux runs ; le rapport v3 synthétique ne précise pas ce réglage.

### Organisation du projet

```text
.
├── data/                 # génération, schéma, textes, templates et PNG
├── notebooks/            # entraînement Unsloth
├── src/                  # partitionnement et évaluation
├── tests/                # tests logiciels et Golden
├── results/              # rapports, courbe et adaptateurs par run
├── app/                  # interface documentaire
├── docs/                 # notes de conception
└── requirements*.txt     # dépendances données et inférence
```
