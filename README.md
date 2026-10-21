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
dépendances et `predict_invoice` porte le décorateur `@spaces.GPU` :
le chargement du modèle et la génération ont lieu dans la réservation GPU,
tandis que l'OCR et le téléchargement des adaptateurs restent hors de cette
réservation. La validation Pydantic et l'orchestration de l'auto-correction
restent dans le processus Gradio : les erreurs de validation ne traversent
pas la frontière du worker ZeroGPU. Chaque passe réserve le GPU uniquement
pour sa génération. Le décorateur est sans
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

**Champs extraits ≠ métadonnées.** Client, coordonnées bancaires et références
sont présents dans les documents, mais ne sont pas des champs cibles.
`locale`, `currency`, template, chemin d'image et détails taxes/remises servent
à la traçabilité du dataset, pas au prompt du modèle. Les montants prédits
restent dans la devise du document ; aucun champ `currency` n'est extrait.

## Mécanismes de fiabilisation & déterminisme

### Décodage sans échantillonnage et précision numérique

Le décodage est explicitement verrouillé dans
[src/evaluate_finetuned.py](src/evaluate_finetuned.py) :

- **LoRA / Transformers :** `do_sample=False` pour les deux passes éventuelles.
- **GGUF :** `temperature=0.0` pour la génération.

Dans [app/main.py](app/main.py), le prompt système impose une lecture
littérale des chiffres : une quantité `7,2` doit être transcrite `7.2`,
et non `142` ou `72`. Un exemple few-shot fictif complet illustre le calcul
`7,2 × 12,50 = 90,00`, soit `quantity=7.2`, `unit_price=12.5`, `total_ht=90.0`.
Les valeurs de cet exemple ne doivent pas être recopiées dans la facture réelle.
Le prompt d'évaluation, identique à celui de l'entraînement, reste inchangé.

### Garde-fous arithmétiques Pydantic

Le schéma partagé [data/schema.py](data/schema.py) valide la structure et
croise les champs numériques avant l'affichage du résultat :

| Contrôle | Règle appliquée |
| --- | --- |
| Cohérence d'une ligne | Comparaison de `quantity × unit_price` avec le `total_ht` de la ligne |
| Cohérence de la facture | Comparaison de la somme des `line_items[].total_ht` avec le `total_ht` global |
| Tolérance monétaire | Écart absolu maximal de **0,02 unité monétaire**, par ligne et sur le total HT global ; tout écart strictement supérieur est rejeté |
| Valeurs non finies | Rejet de `NaN`, `+Infinity` et `-Infinity` dans les quantités, prix et totaux |

Les calculs utilisent `Decimal` à partir des représentations décimales des
valeurs. Le produit quantité × prix unitaire est arrondi au centime avec
`ROUND_HALF_UP`, comme dans le générateur du dataset. Le prix unitaire cible
est le prix **net après remise**. Une facture sans articles est rejetée si
son total HT diffère de zéro de plus de 0,02.

L'application exige également un objet JSON strict : le texte explicatif,
les balises Markdown et les clés dupliquées ne sont pas acceptés. Aucun JSON
ayant échoué aux contrôles n'est affiché comme un résultat validé, et aucune
quantité ni somme n'est recalculée de force pour masquer une incohérence.

**Limites :** le TTC doit être fini, mais son calcul n'est pas vérifié, car
les taux de taxe ne font pas partie du schéma. Le décodage sans échantillonnage
ne garantit ni des résultats identiques entre tous les matériels et versions,
ni l'absence d'hallucinations. Des valeurs erronées mais arithmétiquement
cohérentes nécessitent toujours une vérification du document.

## Boucle d'auto-correction (Self-Correction)

La démo limite la correction à **un unique essai**, soit au maximum
**deux passes d'inférence** :

1. Générer une première réponse, la parser comme JSON strict, puis appliquer
   les validations Pydantic de structure et de cohérence arithmétique.
2. Si le JSON est mal formé ou échoue à Pydantic, conserver la facture
   originale et ajouter à la conversation **le JSON erroné et l'erreur exacte
   du parseur ou de Pydantic**, avec les champs et incohérences signalés.
3. Demander une correction ciblée respectant les valeurs littérales de la
   facture, notamment les décimales, sans modifier artificiellement les montants
   pour satisfaire la validation.
4. Revalider la seconde réponse avec les mêmes contrôles. Si elle échoue,
   afficher une erreur ou un avertissement détaillé dans Gradio et laisser
   le résultat JSON vide. **Aucune troisième tentative n'est lancée.**

Les erreurs de chargement, réponses tronquées et dépassements du contexte
ne déclenchent pas cette correction. Le contexte reste limité sans troncature
silencieuse, y compris pendant la seconde passe.

Sur ZeroGPU, seules les opérations de chargement et de génération du modèle
sont exécutées dans `predict_invoice`, décorée avec `@spaces.GPU`. La
validation et l'orchestration des deux passes restent dans le processus
Gradio, afin de préserver les erreurs Pydantic sans les faire traverser
la frontière du worker GPU.

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

### Golden Dataset de non-régression

Les quatre cas synthétiques de [tests/golden_dataset/](tests/golden_dataset/)
couvrent les quantités `7,2` et `4,2`, les accents et caractères spéciaux,
les prix nets après remise, les taxes multiples et exonérations, ainsi que
les formats FR/EUR, UK/GBP et US/USD avec séparateurs de milliers.

| Cas de référence | Difficultés couvertes |
| --- | --- |
| [France — quantités décimales](tests/golden_dataset/fr_decimal_quantities.json) | `7,2` et `4,2`, virgule décimale, accents, guillemets et symbole euro |
| [France — remises et taxes multiples](tests/golden_dataset/fr_discounts_mixed_taxes.json) | Prix nets après remise de 10 %, TVA à 20 %, 5,5 % et 0 % |
| [Royaume-Uni — TVA et exonération](tests/golden_dataset/uk_fractional_zero_vat.json) | Quantité `7.2`, livre sterling, date en anglais, TVA à 20 % et 0 % |
| [États-Unis — remises et milliers](tests/golden_dataset/us_discount_thousands.json) | Quantité `4.2`, remise de 20 %, taxe à 8.875 %, format `$1,250.00` |

Chaque fixture versionne le texte de facture, une réponse JSON de référence
figée et le résultat attendu, plus les détails de taxes et remises vérifiables.

[tests/test_golden_dataset.py](tests/test_golden_dataset.py) est découvert
automatiquement par `unittest`. Il exécute le pipeline réel de prompt,
validation Pydantic et auto-correction avec un backend d'inférence simulé,
puis compare tous les champs aux attentes par égalité exacte, sans tolérance
sur les nombres, les textes ou l'ordre des articles. Les cas négatifs couvrent
les quantités décalées, une différence d'un centime et les erreurs cohérentes
mathématiquement qui échappent au seul schéma Pydantic.
La comparaison Golden n'utilise donc **pas** la tolérance de 0,02 du schéma :
une différence numérique d'un seul centime par rapport à la référence est
une régression, même si Pydantic accepte encore le résultat.

```bash
python -m unittest discover -s tests -p test_golden_dataset.py -v
```

Ces tests restent hors ligne, sans OCR, téléchargement de poids ni GPU.
Les réponses figées ne sont pas des mesures de qualité du modèle : cette
brique sécurise le pipeline logiciel et ne remplace pas une évaluation des
vraies prédictions. Pour ajouter un cas, conserver séparément le texte source,
la réponse brute et le JSON attendu revu ; ne pas régénérer les attentes depuis
la sortie courante du modèle.

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
