---
title: Extraction intelligente de factures
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app/main.py
---

# 🧾 LLM Invoice Extraction & Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Unsloth](https://img.shields.io/badge/Unsloth-059669?style=for-the-badge)](https://docs.unsloth.ai/) [![QLoRA 4-bit](https://img.shields.io/badge/QLoRA-4--bit-8B5CF6?style=for-the-badge)](https://arxiv.org/abs/2305.14314) [![Qwen 2.5 · 3B](https://img.shields.io/badge/Qwen_2.5-3B-6366F1?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/latest/) [![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) [![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)](https://colab.research.google.com/) [![ReportLab](https://img.shields.io/badge/ReportLab-1D4ED8?style=for-the-badge)](https://docs.reportlab.com/) [![GGUF / llama.cpp](https://img.shields.io/badge/GGUF_%2F_llama.cpp-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://github.com/ggml-org/llama.cpp)

## 📖 Description

Extraction de factures internationales **FR/EUR, UK/GBP et US/USD** vers un
JSON structuré : fournisseur, date, articles et totaux. Le projet met en œuvre
les étapes d'un workflow LLM / IA Engineering, de la synthèse de données
jusqu'à la mesure de la qualité d'extraction.

**100 % de validité JSON et 95 % d'Exact Match déclarés dans le rapport v3 :
un résultat remarquable pour Qwen2.5-3B-Instruct, un modèle léger de seulement
3 milliards de paramètres.** Ce résultat illustre l'intérêt d'un fine-tuning
ciblé LoRA/Unsloth pour spécialiser un petit modèle sur une extraction
structurée complexe ; une supériorité sur des modèles plus volumineux reste
à démontrer par un benchmark comparatif.

> **Périmètre : extraction textuelle avec interface documentaire.** La démo
> Gradio accepte les PDF et les images : le texte est lu dans les PDF natifs
> ou extrait par OCR français/anglais avant l'inférence. Le modèle reste
> textuel ; les scores ci-dessus ne mesurent pas la chaîne OCR + extraction.

## Démo live — Gradio

**[Ouvrir la démo sur Hugging Face Spaces](https://huggingface.co/spaces/nicolbl95/LLM-extraction-finetune)**.
L'URL est celle du Space cible ; sa disponibilité dépend du déploiement des
sources et de la configuration des poids décrits ci-dessous.

L'application [app/main.py](app/main.py) propose un upload PDF/PNG/JPEG,
trois exemples cliquables France/EUR, Royaume-Uni/GBP et États-Unis/USD,
un aperçu visuel et le JSON validé par le schéma Pydantic `Invoice`.
Les exemples suivent exactement le même parcours OCR + modèle que les uploads :
aucune réponse attendue du dataset n'est utilisée pour simuler l'extraction.

### Lancement local

Utiliser Python 3.11 ou 3.12, puis installer les dépendances Python et Tesseract
avec les langues française et anglaise (Debian/Ubuntu) :

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
python -m pip install -r requirements.txt
python app/main.py
```

Ouvrir <http://localhost:7860>. Le modèle est chargé à la première extraction,
pas au démarrage de l'interface. Par défaut, les adaptateurs locaux du run
`results/qwen25-3b-international-v3-20261021T053959288046Z/lora_adapter`
sont chargés sur CUDA en 4 bits s'ils sont présents et complets. Sinon, les
adaptateurs et le tokenizer sont téléchargés automatiquement depuis
[nicolbl95/llm-extraction-weights](https://huggingface.co/nicolbl95/llm-extraction-weights)
et conservés dans le cache Hugging Face. Les poids ne sont pas suivis dans
le dépôt de l'application.

| Variable | Valeur par défaut | Rôle |
| --- | --- | --- |
| `INVOICE_ADAPTER_PATH` | Chemin du run ci-dessus | Dossier des adaptateurs et du tokenizer, absolu ou relatif au dépôt |
| `INVOICE_ADAPTER_REPO` | `nicolbl95/llm-extraction-weights` | Dépôt Hub utilisé lorsque les poids locaux sont absents ou incomplets |
| `INVOICE_ADAPTER_REVISION` | `main` | Branche, tag ou hash de commit du dépôt des poids |
| `HF_TOKEN` | Non défini | Secret de lecture pour un dépôt Hub privé ; inutile pour le dépôt public |
| `INVOICE_DEVICE` | `cuda` | `cuda` (4 bits) ou `cpu` (float32, RAM et latence importantes) |
| `PORT` | `7860` | Port du serveur Gradio |

Le chemin retourné par `snapshot_download` est transmis au backend LoRA après
vérification des fichiers. Le dépôt des poids doit contenir à sa racine
`adapter_config.json`, `adapter_model.safetensors`, `tokenizer_config.json`
et les fichiers du tokenizer (notamment `tokenizer.json` et le template de chat).
Le modèle de base indiqué par les adaptateurs est téléchargé via Transformers
si nécessaire. Un téléchargement échoué ou un dépôt incomplet produit une
erreur explicite ; aucun modèle non fine-tuné ni résultat simulé n'est substitué.

### Hébergement sur Hugging Face Spaces

Créer un Space **Gradio**, sélectionner Python **3.11** et définir dans
l'en-tête YAML du README du Space :

```yaml
---
title: Extraction intelligente de factures
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app/main.py
---
```

Utiliser la même version Gradio 6.x dans `sdk_version` et dans l'environnement
déployé. Publier les sources, `requirements.txt`, `packages.txt` et
`app/examples/`. Les trois PNG distribuables sont des copies des factures du
split test : France `invoice-0406`, UK `invoice-0110`, US `invoice-0435`.
Ils ne dépendent pas du dossier `data/invoices/`, ignoré par Git.

Prévoir un **Space GPU**. Aucun poids n'a besoin d'être inclus dans le dépôt
du Space : ils sont téléchargés depuis le dépôt Hub configuré. Pour un dépôt
privé, ajouter `HF_TOKEN` dans les **Secrets** du Space ; ne pas l'écrire dans
le code ou le README. Un Space CPU reste possible avec
`INVOICE_DEVICE=cpu`, suffisamment de RAM et une latence plus élevée.
Sur **ZeroGPU**, sélectionner ce matériel dans les paramètres du Space et
conserver `INVOICE_DEVICE=cuda`. Le paquet `spaces` est inclus dans les
dépendances et `extract_invoice` porte le décorateur `@spaces.GPU` :
le chargement du modèle et la génération ont lieu dans la réservation GPU,
tandis que l'OCR et le téléchargement des adaptateurs restent hors de cette
réservation. Le décorateur est sans
effet hors de ZeroGPU, ce qui préserve le lancement local et les GPU dédiés.
Le lien ci-dessus pointe vers le Space cible
`nicolbl95/LLM-extraction-finetune`.

Limites de la démo : 20 Mo, 10 pages PDF, 25 mégapixels par image/page rendue,
une extraction à la fois et une file d'attente de 8 requêtes. L'aperçu PDF
montre la première page ; l'extraction lit toutes les pages, avec OCR pour
les pages sans texte. Un document dépassant le contexte du modèle est refusé,
sans troncature silencieuse. Utiliser des factures fictives ou anonymisées ;
les fichiers transitent par le serveur Gradio. La validation Pydantic garantit
la structure, pas l'exactitude métier. Les performances publiées ne sont pas
un benchmark de cette interface ni de son OCR.

## 🛠️ Stack Technique & Méthodologies

- **LLM & Fine-Tuning :** Qwen2.5-3B-Instruct, Unsloth pour l'optimisation
  mémoire/vitesse, adaptation LoRA avec PEFT et quantification 4 bits
  du modèle de base (QLoRA).
- **Structuration & Validation :** Pydantic v2 pour définir le contrat
  d'extraction ; contrôle du JSON strict et validation du schéma après
  génération pour les entités, montants HT/TTC et tableaux d'articles.
  Le pipeline n'impose pas de JSON Schema au décodage.
- **Pipeline de Données :** génération synthétique multilingue et multidevise,
  diversification procédurale des contenus et mises en page (data augmentation),
  partitionnement Train/Val/Test reproductible et contrôles de chevauchement.
- **Évaluation & Métriques :** benchmark personnalisé, Exact Match Rate,
  JSON Validity Rate et rapports `metrics.json`, avec prédictions détaillées
  et ventilation par région, devise, format et template.
- **Environnement & Tooling :** PyTorch, Hugging Face Transformers et Datasets,
  VS Code / GitHub Codespaces pour le développement et Google Colab pour
  l'entraînement GPU.

## 🏗️ Architecture

```text
9 références de templates internationaux
                  │
     Génération procédurale déterministe
                  │
    600 textes + cibles JSON + 600 PNG
                  │
      Train 480 / Validation 60 / Test 60
                  │
      Qwen2.5-3B + QLoRA / Unsloth
                  │
     Adaptateurs LoRA + tokenizer
                  │
   Base + PEFT → JSON → Pydantic / scoring
                  │
       Métriques + prédictions détaillées
```

- **Data Engineering :** neuf familles de structures, 600 profils visuels
  distincts, rendu ReportLab/PyMuPDF, 1 à 12 articles, dates et devises
  régionalisées. Calculs monétaires contrôlés avec `Decimal` et arrondi au
  centime ; seed **42**.
- **Optimisation :** base en 4 bits, supervision des réponses assistant,
  accumulation de gradients et contrôle des longueurs sans troncature
  silencieuse.
- **Fiabilité :** contrat Pydantic v2, scoring du JSON strict, erreurs
  explicites et provenance par hashes. Les splits sont disjoints par ID et
  texte ; les templates restent partagés entre les sous-ensembles.

### Organisation du projet

```text
.
├── assets/loss_curve.png
├── data/
│   ├── generate_dataset.py / render_invoice.py / schema.py
│   ├── dataset.jsonl                  # 600 factures
│   ├── train.jsonl / val.jsonl / test.jsonl
│   ├── templates/                     # 1.png à 9.png
│   └── invoices/                      # 600 rendus PNG uniques
├── notebooks/fine_tune_unsloth.ipynb
├── src/
│   ├── split_data.py
│   ├── evaluate_baseline.py            # baseline Regex, pas Qwen brut
│   └── evaluate_finetuned.py           # LoRA / GGUF
├── tests/                             # données, notebook, évaluation
├── results/
│   ├── evaluation-3/finetuned_metrics.json
│   ├── qwen25-3b-international-v3-20261021T053959288046Z.zip
│   └── qwen25-3b-international-v3-20261021T053959288046Z/
│       ├── lora_adapter/              # poids et tokenizer
│       ├── checkpoints/
│       ├── training_manifest.json
│       ├── train_log.json
│       └── loss_curve.png
├── requirements.txt
└── requirements-inference.txt
```

### Contrat de sortie

Le schéma Pydantic v2 définit `Invoice` et ses `LineItem` :

| Champ | Type | Contenu |
| --- | --- | --- |
| `supplier_name` | `str` | Nom du fournisseur |
| `invoice_date` | `date` | Date normalisée `AAAA-MM-JJ` |
| `total_ht`, `total_ttc` | `float` | Totaux hors taxes et taxes comprises |
| `line_items` | `list[LineItem]` | Articles ordonnés ; liste vide par défaut dans le schéma |
| `line_items[].description` | `str` | Désignation |
| `line_items[].quantity` | `float` | Quantité |
| `line_items[].unit_price` | `float` | Prix unitaire net après remise, hors taxes |
| `line_items[].total_ht` | `float` | Montant net de la ligne |

Les contrôles arithmétiques Pydantic utilisent `Decimal`, l'arrondi
`ROUND_HALF_UP` au centime pour `quantity × unit_price`, et une tolérance
absolue de **0,02 unité monétaire** par ligne et sur le total HT global.
La somme des montants HT de toutes les lignes doit correspondre au total HT
de la facture ; une liste vide n'est donc pas valide pour un total HT non nul
au-delà de cette tolérance. Les valeurs non finies (`NaN`, infinis) sont refusées.
La démo utilise un prompt renforcé exigeant une lecture littérale des nombres
et un exemple fictif complet : `7,2 × 12,50 = 90,00` devient
`quantity=7.2`, `unit_price=12.5`, `total_ht=90.0`. Le prompt d'évaluation,
identique à celui de l'entraînement, reste inchangé.

Si le JSON est mal formé ou échoue à la validation Pydantic, **une seule**
nouvelle inférence est tentée. La conversation conserve la facture originale
et reçoit le JSON rejeté ainsi que l'erreur exacte du parseur ou de Pydantic.
La seconde réponse est soumise aux mêmes contrôles, sans troisième tentative.
Les erreurs de chargement, réponses tronquées et dépassements du contexte ne
déclenchent pas cette correction. Le contexte reste limité sans troncature
silencieuse, y compris pendant la seconde passe.

En cas d'échec persistant, l'interface affiche l'erreur et ne présente pas
le JSON comme valide. Aucune correction arithmétique forcée n'est appliquée.
Le calcul du TTC n'est pas vérifié : les taux de taxe ne font pas partie du schéma.

La génération utilise déjà un décodage sans échantillonnage (`do_sample=False`
en LoRA, `temperature=0.0` en GGUF). Ce réglage et les contrôles mathématiques
ne garantissent pas l'absence d'hallucinations : des valeurs erronées mais
arithmétiquement cohérentes nécessitent toujours une vérification du document.

**Champs extraits ≠ métadonnées.** Client, coordonnées bancaires et références
sont présents dans les documents, mais ne sont pas des champs cibles.
`locale`, `currency`, template, chemin d'image et détails taxes/remises servent
à la traçabilité du dataset, pas au prompt du modèle. Les montants prédits
restent dans la devise du document ; aucun champ `currency` n'est extrait.

## 🎯 Résultats clés

| Indicateur | Résultat |
| --- | ---: |
| Validité JSON déclarée | **100 %** |
| Exact Match déclaré | **95 %** |
| Dataset international v3 | **600 factures + 600 PNG** |
| Répartition par région | **200 FR / 200 UK / 200 US** |
| Split train / validation / test | **480 / 60 / 60** |

Source : [rapport final v3](results/evaluation-3/finetuned_metrics.json),
qui identifie les adaptateurs du run Colab. Ce rapport synthétique ne fournit
ni effectif évalué, ni hash du test, ni latence, ni détail par segment :
les deux taux sont repris tels que déclarés, sans extrapolation.

L'évaluateur prépare les ventilations région/devise/format/template et distingue
JSON strict, validité du schéma et exactitude métier. Il compare les nombres
avec une tolérance de **0,01** et les articles dans leur ordre. Les rapports
historiques Regex/GGUF ne constituent pas une comparaison contrôlée v3.

## 🚀 Fine-tuning & Loss

Entraînement QLoRA réalisé sur Colab avec Unsloth ; les poids LoRA et le
tokenizer sont importés dans `results/`.

| Configuration | Valeur |
| --- | --- |
| Base | `unsloth/Qwen2.5-3B-Instruct` |
| Quantification / LoRA | 4 bits ; rang `16`, alpha `16` |
| Époques / learning rate | `3` / `2e-4` |
| Batch / accumulation | `1` / `8` |
| Contexte maximal | `4096` tokens |
| Durée du run | Environ **40 min 19 s** |
| Train loss / validation loss | **0,00746 / 0,00591** |

![Pertes d'entraînement et de validation du fine-tuning QLoRA](assets/loss_curve.png)

La perte diminue rapidement puis reste faible ; la validation se stabilise
autour de **0,006**, sans remontée durable visible. Cela suggère une convergence
stable sur ce run, sans prouver l'absence de surapprentissage.

Pour reproduire : ouvrir le notebook sur un GPU compatible (Colab T4 prévu),
fournir les fichiers v3 `train.jsonl` et `val.jsonl`, puis exécuter les cellules
dans l'ordre. Le manifeste sauvegarde configuration, versions et hashes ;
l'archive doit être téléchargée avant expiration du runtime.
**Les adaptateurs nécessitent le modèle de base.** L'export GGUF fusionné
Q4_K_M est facultatif ; les uploads Hub/W&B sont désactivés par défaut.

## ⚙️ Installation / Utilisation

Commandes depuis la racine du projet, avec un Python compatible avec la pile
GPU utilisée pour l'entraînement.

### Installer et vérifier les données

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/evaluate_finetuned.py --check-data-only
python -m unittest discover -s tests -v
```

Pour **régénérer** les données et les images :

```bash
python data/generate_dataset.py
python src/split_data.py
```

La régénération remplace les fichiers existants : conserver les données et
hashes du run avant toute modification.

### Charger les adaptateurs et évaluer

Sur un environnement CUDA avec suffisamment de VRAM :

```bash
python -m pip install -r requirements-inference.txt

RUN_NAME="qwen25-3b-international-v3-20261021T053959288046Z"
ADAPTER_DIR="results/$RUN_NAME/lora_adapter"

# Vérification des fichiers et de la provenance, sans charger le modèle.
python src/evaluate_finetuned.py \
  --adapter-path "$ADAPTER_DIR" --check-data-only

# Smoke test de chargement et de génération.
python src/evaluate_finetuned.py \
  --adapter-path "$ADAPTER_DIR" --load-in-4bit --limit 3 \
  --output-dir "results/$RUN_NAME/smoke-test"

# Évaluation complète du split de test.
python src/evaluate_finetuned.py \
  --adapter-path "$ADAPTER_DIR" --load-in-4bit \
  --output-dir "results/$RUN_NAME/evaluation-v3"
```

Chaque dossier de sortie doit être **nouveau**. L'évaluation produit
`metrics.json` et `predictions.jsonl`, sans écraser les rapports existants.
Le précontrôle vérifie les splits voisins et les hashes disponibles du
manifeste ; il ne garantit pas le chargement des poids.

Le modèle de base est téléchargé si nécessaire. Par défaut : décodage sans
échantillonnage, contexte **4096**, budget de réponse **2048** tokens et une
requête de warmup exclue du timing. En LoRA, un dépassement prompt + budget
provoque une erreur explicite ; ajuster les limites au contexte entraîné,
sans accepter de sorties tronquées comme benchmark complet.

Le backend GGUF est accessible avec `--model-path chemin/modele.gguf`
à la place de `--adapter-path` et nécessite `llama-cpp-python`.
La tentative locale initiale était bloquée par l'absence de PyTorch/GPU ;
elle n'est pas la source des scores du rapport fourni.

> **Limites à retenir :** test synthétique avec familles de templates partagées,
> pas de benchmark OCR ni de preuve de généralisation aux factures réelles.
> Pydantic détecte des écarts de structure, sans garantir l'absence
> d'hallucinations. Les tests logiciels utilisent des modèles simulés ;
> ils ne remplacent pas une évaluation des vrais poids.
