# LLM-extraction-finetune

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Unsloth](https://img.shields.io/badge/Unsloth-059669?style=for-the-badge)](https://docs.unsloth.ai/) [![QLoRA 4-bit](https://img.shields.io/badge/QLoRA-4--bit-8B5CF6?style=for-the-badge)](https://arxiv.org/abs/2305.14314) [![Qwen 2.5 · 3B](https://img.shields.io/badge/Qwen_2.5-3B-6366F1?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/latest/) [![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) [![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)](https://colab.research.google.com/) [![ReportLab](https://img.shields.io/badge/ReportLab-1D4ED8?style=for-the-badge)](https://docs.reportlab.com/) [![GGUF / llama.cpp](https://img.shields.io/badge/GGUF_%2F_llama.cpp-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://github.com/ggml-org/llama.cpp)

## 📖 Description

Extraction de factures internationales **FR/EUR, UK/GBP et US/USD** vers un
JSON structuré : fournisseur, date, articles et totaux. Le projet met en œuvre
les étapes d'un workflow LLM / IA Engineering, de la synthèse de données
jusqu'à la mesure de la qualité d'extraction.
**[Tester la démo live](https://huggingface.co/spaces/nicolbl95/LLM-extraction-finetune)**.

**100 % de validité JSON et 95 % d'Exact Match déclarés dans le rapport v3 :
un résultat remarquable pour Qwen2.5-3B-Instruct, un modèle léger de seulement
3 milliards de paramètres.** Ce résultat illustre l'intérêt d'un fine-tuning
ciblé LoRA/Unsloth pour spécialiser un petit modèle sur une extraction
structurée complexe ; une supériorité sur des modèles plus volumineux reste
à démontrer par un benchmark comparatif.

## 🛠️ Stack Technique & Méthodologies

- **LLM & Fine-Tuning :** Qwen2.5-3B-Instruct, Unsloth pour l'optimisation
  mémoire/vitesse, adaptation LoRA avec PEFT et quantification 4 bits
  du modèle de base (QLoRA).
- **Structuration & Validation :** Pydantic v2 pour définir le contrat
  d'extraction ; contrôle du JSON strict et validation du schéma après
  génération pour les entités, montants HT/TTC et tableaux d'articles.
  Le pipeline n'impose pas de JSON Schema au décodage.
- **Garde-fous & Auto-correction :** validation Pydantic stricte et contrôles
  arithmétiques avec `Decimal`, tolérance de **0,02** et arrondi
  `ROUND_HALF_UP`. Une seule passe corrective relit les données sources ;
  les incohérences persistantes sont signalées, sans ajustement forcé.
- **Pipeline de Données :** génération synthétique multilingue et multidevise,
  diversification procédurale des contenus et mises en page (data augmentation),
  partitionnement Train/Val/Test reproductible et contrôles de chevauchement.
- **Rôle des jeux de données (Split 80/10/10) :** le script Python crée les
  exemples ; Qwen apprend sur **Train (480)**, est suivi sur
  **Validation (60)**, l'« examen blanc », puis évalué sur
  **Test (60)**, l'« examen final » réservé à des exemples non utilisés
  pour l'apprentissage ni le réglage.
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

### Méthodologie : génération et apprentissage

**Le script crée les exercices ; le modèle apprend à les résoudre.**
`data/generate_dataset.py` fabrique les factures et leurs cibles JSON par des
règles Python explicites. `src/split_data.py` les répartit en trois jeux
disjoints ; aucun apprentissage n'a lieu pendant cette génération.
Lors du fine-tuning, Qwen ajuste ses adaptateurs à partir des couples
texte/JSON : il apprend des régularités d'extraction, pas l'exécution du
script ni une garantie de raisonnement arithmétique exact.

| Fichier | Part / exemples | Rôle pédagogique et technique |
| --- | --- | --- |
| `data/train.jsonl` | **80 % / 480** | **Apprentissage des règles** : exemples corrigés utilisés pour calculer la perte et mettre à jour les adaptateurs. |
| `data/val.jsonl` | **10 % / 60** | **Examen blanc** : mesure la perte sans mise à jour des poids ; aide à surveiller le surapprentissage et à guider les choix d'entraînement, sans garantir son absence. |
| `data/test.jsonl` | **10 % / 60** | **Examen final scellé vis-à-vis du réglage** : réservé à la mesure finale, jamais utilisé pour les gradients ni pour choisir les hyperparamètres. |

Les exemples de test sont inédits pour l'apprentissage, mais partagent les
familles de templates du dataset : cette évaluation ne démontre pas à elle
seule une généralisation à des structures entièrement nouvelles.

### Organisation du projet

```text
.
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
et non `142` ou `72` ; `4,2` devient `4.2`, et non `42` ou `4.02`.
Les parties entière et décimale doivent être préservées, sans concaténation
ni mélange des colonnes. Les quantités et prix ne doivent pas être déduits
par division des totaux. Un exemple few-shot fictif complet illustre le calcul
`7,2 × 12,50 = 90,00`, soit `quantity=7.2`, `unit_price=12.5`, `total_ht=90.0`.
Les valeurs de cet exemple ne doivent pas être recopiées dans la facture réelle.
Le prompt d'évaluation, identique à celui de l'entraînement, reste inchangé.

### Garde-fous arithmétiques Pydantic

- **Structure :** objet JSON strict, champs et types requis, valeurs finies ;
  prose, Markdown et clés dupliquées sont refusés.
- **Calculs :** contrôle de `quantity × unit_price` net et de la somme HT,
  avec `Decimal`, arrondi `ROUND_HALF_UP` et tolérance de **0,02**.
- **Affichage :** après correction, un JSON structurellement valide peut être
  affiché avec des avertissements arithmétiques détaillés, sans recalcul forcé.
  Les données, l'évaluation et les tests Golden restent stricts par défaut.

## Boucle d'auto-correction (Self-Correction)

La démo autorise **une seule correction**, soit deux passes maximum :
le modèle reçoit la facture originale, sa réponse et les erreurs de validation
pour relire les valeurs sources, sans ajuster artificiellement les totaux.
Le dernier JSON structurellement valide est conservé avec avertissements si
nécessaire ; sans résultat valide, une erreur est affichée.
Les erreurs de chargement, troncatures et dépassements de contexte ne
déclenchent pas de correction ; aucun contexte n'est tronqué silencieusement.

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

La perte diminue rapidement puis reste faible ; la validation se stabilise
autour de **0,006**, sans remontée durable visible. Cela suggère une convergence
stable sur ce run, sans prouver l'absence de surapprentissage.

Pour reproduire : ouvrir le notebook sur un GPU compatible (Colab T4 prévu),
fournir les fichiers v3 `train.jsonl` et `val.jsonl`, puis exécuter les cellules
dans l'ordre. Le manifeste sauvegarde configuration, versions et hashes ;
l'archive doit être téléchargée avant expiration du runtime.
**Les adaptateurs nécessitent le modèle de base.** L'export GGUF fusionné
Q4_K_M est facultatif ; les uploads Hub/W&B sont désactivés par défaut.

## ⚙️ Installation & Utilisation

Depuis la racine du projet :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-inference.txt

# Vérifier les données sans charger de modèle.
python src/evaluate_finetuned.py --check-data-only

# Évaluer les adaptateurs téléchargés sur un environnement CUDA compatible.
python src/evaluate_finetuned.py \
  --adapter-path results/qwen25-3b-international-v3-20261021T053959288046Z/lora_adapter \
  --load-in-4bit \
  --output-dir results/evaluation-v3
```

Les adaptateurs nécessitent le modèle de base et leur tokenizer. Choisir un
nouveau dossier de sortie pour chaque évaluation ; les rapports existants
ne sont pas écrasés.
