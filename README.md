# LLM-extraction-finetune

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Unsloth](https://img.shields.io/badge/Unsloth-059669?style=for-the-badge)](https://docs.unsloth.ai/) [![QLoRA 4-bit](https://img.shields.io/badge/QLoRA-4--bit-8B5CF6?style=for-the-badge)](https://arxiv.org/abs/2305.14314) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/latest/)

## 📖 Description

Extraction de factures **FR/EUR, UK/GBP et US/USD** vers un JSON structuré : fournisseur, date, articles et totaux.
Stack : **Qwen2.5-3B-Instruct, QLoRA 4 bits, Unsloth et PEFT**, avec validation Pydantic v2.
Rapport v3 : **100 % de validité JSON et 95 % d'Exact Match déclarés** ; la supériorité sur des modèles plus volumineux reste à démontrer.
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
  arrondi `ROUND_HALF_UP`, tolérance **0,02**.
- **Correction :** une seule passe corrective, deux inférences maximum, à partir
  de la facture, de la réponse et des erreurs ; pas d'ajustement artificiel des
  totaux. Le dernier JSON structurellement valide est conservé avec
  avertissements si nécessaire ; sans résultat valide, une erreur est affichée.
  Données, évaluation et tests Golden restent stricts par défaut.
- **Erreurs opérationnelles :** chargement, troncature ou dépassement de contexte
  ne déclenchent pas la correction ; aucun contexte n'est tronqué silencieusement.

Décodage sans échantillonnage dans [l'évaluateur](src/evaluate_finetuned.py) :
LoRA `do_sample=False` pour les deux passes éventuelles ; GGUF `temperature=0.0`.
Lecture littérale des chiffres : [détails du prompt système](docs/DESIGN.md).

## 🎯 Résultats

| Système | JSON valide | Exact Match | Source |
| --- | ---: | ---: | --- |
| Baseline Regex (rapport historique) | 46,67 % | 43,33 % | [src/evaluate_baseline.py](src/evaluate_baseline.py) |
| Qwen2.5-3B-Instruct brut (sans fine-tuning) | à mesurer | à mesurer | [mode `--base-only`](src/evaluate_finetuned.py) |
| Qwen2.5-3B + QLoRA (v3) | 100 % (déclaré) | 95 % (déclaré) | [results/evaluation-3/finetuned_metrics.json](results/evaluation-3/finetuned_metrics.json) |

Valeurs Regex reprises du [rapport existant](results/baseline_metrics.json).
Les rapports historiques Regex/GGUF ne constituent pas une comparaison
contrôlée v3. Le rapport v3 identifie les adaptateurs, mais ne fournit ni
effectif évalué, ni hash du test, ni latence, ni détail par segment :
les deux taux sont repris tels que déclarés, sans extrapolation.

Baseline à produire : évaluer le modèle de base avec le même prompt et le
même test set (60 exemples), sans adaptateur. Le mode `--base-only` charge
`unsloth/Qwen2.5-3B-Instruct` en NF4 sur CUDA, sans charger ni fusionner de LoRA.
Le lancement local est bloqué par l'absence de PyTorch et de GPU NVIDIA :
aucun score mesuré n'est disponible ; la baseline Regex reste distincte.

L'évaluateur distingue JSON strict, validité du schéma et exactitude métier ;
les articles sont comparés dans leur ordre. Sa tolérance numérique de
**0,01** sert au scoring des prédictions contre les cibles, contrairement à
la tolérance arithmétique ci-dessus qui contrôle la cohérence interne.

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
prompt correctif de la démo, sans son prompt système enrichi ni son few-shot.
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
