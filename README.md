# 🧾 LLM Invoice Extraction & Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Unsloth](https://img.shields.io/badge/Unsloth-059669?style=for-the-badge)](https://docs.unsloth.ai/) [![QLoRA 4-bit](https://img.shields.io/badge/QLoRA-4--bit-8B5CF6?style=for-the-badge)](https://arxiv.org/abs/2305.14314) [![Qwen 2.5 · 3B](https://img.shields.io/badge/Qwen_2.5-3B-6366F1?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/latest/) [![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/) [![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)](https://colab.research.google.com/) [![ReportLab](https://img.shields.io/badge/ReportLab-1D4ED8?style=for-the-badge)](https://docs.reportlab.com/) [![GGUF / llama.cpp](https://img.shields.io/badge/GGUF_%2F_llama.cpp-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://github.com/ggml-org/llama.cpp) [![Démo Live · À configurer](https://img.shields.io/badge/D%C3%A9mo_Live-%C3%80_configurer-F59E0B?style=for-the-badge)](#demo-en-ligne)

**Factures internationales · JSON structuré · Adaptation efficiente · Évaluation traçable**

---

<a id="description"></a>

## 📖 Description

Pipeline de génération de données, d'adaptation d'un modèle de langage et
d'évaluation pour convertir des factures **françaises et anglophones** en JSON
validable avec **Pydantic v2**.

L'objectif professionnel est d'automatiser la transformation de documents
hétérogènes en données exploitables : fournisseur, date, lignes d'articles
et totaux. Le projet met en œuvre les étapes d'un workflow **LLM / IA
Engineering**, de la synthèse de données jusqu'à la mesure de la qualité
d'extraction, pour des cas de démonstration recruteur ou d'intégration métier.

La version **`international-v3`** comprend **600 factures synthétiques** et leurs
**600 rendus PNG**, répartis entre **France / EUR**, **Royaume-Uni / GBP** et
**États-Unis / USD**. Le modèle cible est **Qwen2.5-3B-Instruct**, adapté en
**QLoRA 4 bits avec Unsloth**, puis chargé avec ses adaptateurs LoRA pour
l'évaluation.

> **Périmètre actuel : extraction textuelle.** Les PNG simulent des documents
> utilisateurs, mais le notebook et l'évaluateur consomment leur texte, pas les
> pixels. Un parcours d'upload image/PDF nécessite encore une étape OCR ou un
> modèle vision-langage. Aucune API ni interface d'upload n'est implémentée.

<a id="demo-en-ligne"></a>

> [!NOTE]
> **ℹ️ Démo en ligne — lien à renseigner**
>
> Une fois la démo publiée, vous pourrez tester directement le modèle sans
> installation ici : `<LIEN_DEMO_OU_SPACE_HUGGINGFACE>`.
>
> Aucun lien de démo déployée n'est actuellement référencé. Le badge
> « Démo Live » pointe vers cet encadré, pas vers un service actif.

---

<a id="ingenierie-llm"></a>

## 🎯 Compétences démontrées — LLM / IA Engineer

| Axe d'ingénierie | Réalisation et valeur technique |
| --- | --- |
| **Data Engineering & Synthèse** | Génération procédurale de **600 factures textuelles + 600 PNG uniques**, compositions ReportLab et rasterisation PyMuPDF. Neuf familles de structures, 600 profils visuels distincts, régionalisation FR/UK/US et contrôles financiers : [générateur](data/generate_dataset.py), [rendu](data/render_invoice.py). |
| **Fine-Tuning & Optimisation** | Adaptation **QLoRA 4 bits de Qwen2.5-3B avec Unsloth**, notebook prévu pour GPU Colab T4, accumulation de gradients et supervision des réponses assistant. Contrôle explicite des longueurs de contexte, sans troncature silencieuse : [notebook](notebooks/fine_tune_unsloth.ipynb). |
| **Fiabilité & Production** | Contrat de sortie **Pydantic v2**, contrôle du JSON strict, détection des montants non finis, erreurs explicites et provenance des poids/données. Les réponses non conformes sont mesurées et tracées, pas présentées comme des succès : [schéma](data/schema.py), [évaluateur](src/evaluate_finetuned.py). |
| **Évaluation Rigoureuse** | Rapports par **région, devise, format et template** ; baseline Regex distincte du LLM brut ; protocole de comparaison avec le modèle adapté. Splits Train/Val/Test disjoints par ID et texte, contrôles de chevauchement lorsque les fichiers sont présents et vérification des hashes d'entraînement disponibles : [répartition](src/split_data.py), [tests d'évaluation](tests/test_evaluate_finetuned.py). |

> [!IMPORTANT]
> La diversité vise à **réduire le risque d'overfitting**, pas à l'éliminer.
> La validation détecte les écarts de structure ; elle ne garantit ni
> l'absence d'hallucinations métier ni une génération JSON toujours conforme.
> Les poids v3 sont reçus et leur provenance vérifiée ; le rapport fourni
> annonce **100 % de validité JSON et 95 % d'Exact Match**. La comparaison
> contrôlée avec les baselines reste à établir. Les détails et limites sont explicités
> dans les sections [évaluation](#evaluation), [résultats](#resultats) et
> [validation](#validation).

---

## 🧭 Sommaire

- [📖 Description et démo](#description)
- [🎯 Compétences LLM / IA Engineer](#ingenierie-llm)
- [🏗️ Architecture et état du projet](#architecture)
- [📂 Arborescence](#arborescence)
- [🌍 Dataset international v3](#dataset-v3)
- [📐 Contrat de données et schéma Pydantic v2](#schema)
- [⚙️ Installation et génération](#installation)
- [🚀 Fine-tuning sur Colab ou Kaggle](#fine-tuning)
- [🔎 Inférence et évaluation](#evaluation)
- [📊 Comparaison des modèles et métriques](#resultats)
- [🛡️ Validation, limites et mise en production](#validation)

---

<a id="architecture"></a>

## 🏗️ Architecture et état du projet

```text
Références de structure : data/templates/1.png … 9.png
                          │
                  Génération déterministe
                          │
             ┌────────────┴─────────────┐
             │                          │
     dataset.jsonl                 600 factures PNG
   texte + cible + métadonnées       même contenu métier
             │                          │
     train / val / test          futurs tests OCR/upload
             │
     Unsloth + Qwen2.5-3B
       entraînement QLoRA
             │
   adaptateurs LoRA + tokenizer + manifeste
             │
   modèle de base + PEFT → génération JSON
             │
   validation Pydantic + scoring sur test.jsonl
             │
      metrics.json + predictions.jsonl
```

| Composant | État et responsabilités |
| --- | --- |
| Génération | 600 exemples v3 et 600 PNG présents ; contrôles de cohérence financière et de diversité |
| Répartition | 480 exemples d'entraînement, 60 de validation, 60 de test |
| Fine-tuning | Run Colab terminé : 3 époques ; archive et adaptateurs v3 extraits localement, hashes train/validation vérifiés |
| Inférence | Backend Hugging Face / PEFT pour LoRA et backend GGUF via `llama-cpp-python` |
| Évaluation | Précontrôle sans modèle, scoring structuré, rapports globaux et par segment |
| Performances v3 | Rapport final fourni : **100 % de validité JSON, 95 % d'Exact Match** ; détail et limites de traçabilité dans la section résultats |
| OCR, application, API | Hors implémentation actuelle |

**Séparation des responsabilités :** le générateur produit les données ; le
notebook entraîne le modèle ; l'évaluateur mesure ses sorties. Générer les
factures ou réussir les tests logiciels n'entraîne pas le modèle et ne prouve
pas sa précision.

---

<a id="arborescence"></a>

## 📂 Arborescence

Arborescence applicative actuelle, hors métadonnées Git et caches Python.
Les séries de PNG sont regroupées pour ne pas énumérer 600 fichiers.

```text
llm-extraction-finetune/
├── .github/
│   └── copilot-instructions.md
├── README.md
├── requirements.txt
├── requirements-inference.txt
├── assets/
│   └── loss_curve.png             # courbes de perte train/validation
├── Qwen2.5-3B-Instruct.Q4_K_M.gguf   # fichier local ; provenance à vérifier
├── data/
│   ├── schema.py                   # schéma cible Pydantic
│   ├── generate_dataset.py         # contenu, annotations et génération
│   ├── render_invoice.py           # compositions visuelles et rendu PNG
│   ├── dataset.jsonl               # 600 exemples complets
│   ├── train.jsonl                 # 480 exemples
│   ├── val.jsonl                   # 60 exemples
│   ├── test.jsonl                  # 60 exemples
│   ├── templates/
│   │   └── 1.png … 9.png           # neuf références internationales
│   └── invoices/
│       └── invoice-0001.png … invoice-0600.png
├── notebooks/
│   └── fine_tune_unsloth.ipynb
├── src/
│   ├── split_data.py
│   ├── evaluate_baseline.py        # extracteur historique à règles
│   └── evaluate_finetuned.py       # inférence/évaluation LoRA ou GGUF
├── tests/
│   ├── test_invoices.py
│   ├── test_training_notebook.py
│   └── test_evaluate_finetuned.py
└── results/
    ├── baseline_metrics.json       # rapport historique, pas Qwen brut
    ├── finetuned_metrics.json      # rapport historique non certifié v3
    ├── evaluation-3/
    │   └── finetuned_metrics.json  # rapport final v3 fourni
    ├── qwen25-3b-international-v3-20261021T053959288046Z.zip
    └── qwen25-3b-international-v3-20261021T053959288046Z/
        ├── lora_adapter/           # poids LoRA et tokenizer reçus
        ├── checkpoints/
        ├── training_manifest.json
        ├── train_log.json
        └── loss_curve.png
```

Après téléchargement d'un run Colab/Kaggle et exécution de l'évaluation,
les artefacts attendus sont :

```text
results/
├── <RUN_NAME>.zip
└── <RUN_NAME>/
    ├── lora_adapter/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer_config.json
    │   └── ...                     # fichiers complémentaires du tokenizer
    ├── checkpoints/                # au plus deux checkpoints d'époque
    ├── training_manifest.json
    ├── train_log.json
    ├── loss_curve.png
    ├── gguf_q4_k_m/                # seulement si export GGUF activé
    ├── smoke-test/                 # après test limité
    └── evaluation-v3/              # après évaluation complète
        ├── metrics.json
        └── predictions.jsonl
```

Cette seconde arborescence décrit la convention de sortie de l'évaluateur.
Le rapport final fourni est enregistré séparément dans
[results/evaluation-3/finetuned_metrics.json](results/evaluation-3/finetuned_metrics.json),
et non dans le dossier conventionnel `evaluation-v3/`. Le run Colab et ses
poids sont extraits localement. La présence locale des fichiers ne signifie
pas qu'ils sont versionnés ou publiés sur GitHub.

---

<a id="dataset-v3"></a>

## 🌍 Dataset international v3

### Répartition et diversité

| Région | Locale | Langue | Devise | Total | Train | Validation | Test |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| France | `fr-FR` | Français | EUR / € | 200 | 160 | 20 | 20 |
| Royaume-Uni | `en-GB` | Anglais britannique | GBP / £ | 200 | 160 | 20 | 20 |
| États-Unis | `en-US` | Anglais américain | USD / $ | 200 | 160 | 20 | 20 |
| **Total** | | | | **600** | **480** | **60** | **60** |

Les exemples couvrent trois représentations textuelles : `text_block`,
`raw_text` et `email`, **200 exemples chacune sur le dataset complet**.
Les sous-ensembles ne garantissent pas une égalité exacte entre ces formats.
La répartition combine allocation par plus forts restes pour les templates
et entrelacement des formats. Chaque région de chaque sous-ensemble couvre
les neuf familles de templates.

Les compositions originales s'inspirent des **neuf références, soit une
dizaine de templates internationaux**, de [data/templates/](data/templates/).
Leurs visuels et identités d'exemple ne sont pas reproduits à l'identique.

| Référence | Variation structurelle |
| --- | --- |
| 1 | Grand en-tête, blocs fournisseur/client, prix avant quantité |
| 2 | Mise en page minimale, quantité en première colonne |
| 3 | Colonnes de taxes et montants TTC par ligne |
| 4 | Titre centré, client avant fournisseur |
| 5 | Bloc de paiement avant le tableau d'articles |
| 6 | Accent latéral, client en premier, totaux avant les lignes |
| 7 | Tableau de prestations et alternance de fonds |
| 8 | Prix catalogue, remise explicite et prix unitaire net |
| 9 | Facture pro forma avec taxes et montants bruts |

La génération fait varier les libellés, l'ordre des colonnes, les positions
des blocs, les couleurs, les marges, les espacements, les tailles de police
et trois familles typographiques : Helvetica, Times et Courier.
L'échantillonnage sans remplacement produit **600 profils visuels distincts**.
Les PNG mesurent **893 × 1263 pixels**.

Les factures comportent **1 à 12 lignes**, des quantités entières ou
fractionnaires, des remises, plusieurs échéances et différents formats de
date : `JJ/MM/AAAA`, `MM/JJ/AAAA`, ISO, séparateurs à points et mois en lettres.
Les dates numériques suivent la convention de la région.

### Cohérence financière et traçabilité

- Seed de génération et de répartition : **42**.
- Prix unitaire cible : prix **net après remise**.
- Arrondi monétaire : `Decimal`, au centime, `ROUND_HALF_UP`.
- Montant HT de ligne : quantité × prix unitaire net arrondi.
- Total HT : somme des montants nets des lignes.
- Total TTC : total HT + somme des taxes arrondies par ligne.
- Taxes simulées : France 20 % et 5,5 % pour les livres imprimés ;
  Royaume-Uni 20 % et 0 % pour les livres imprimés ; États-Unis 7,25 %,
  8,875 % ou 0 % selon l'État simulé, sans surtaxes locales ni règles
  inter-États.

Ces scénarios sont destinés à l'extraction, **pas à la conformité fiscale**.
Les identités et coordonnées bancaires sont synthétiques et ne doivent pas
être utilisées pour des paiements.

Chaque texte et son image partagent le même contenu métier. Le format email
ajoute une enveloppe textuelle de salutation/signature. Les cibles JSON ne
sont pas imprimées dans les images ; un débordement de page provoque une
erreur explicite plutôt qu'un contenu silencieusement coupé.

---

<a id="schema"></a>

## 📐 Contrat de données et schéma Pydantic v2

### Cible effectivement prédite

Le contrat implémenté dans [data/schema.py](data/schema.py) est volontairement
compact. **La v3 internationalise les documents sans élargir ce schéma.**

```python
from datetime import date
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total_ht: float


class Invoice(BaseModel):
    supplier_name: str
    invoice_date: date
    total_ht: float
    total_ttc: float
    line_items: list[LineItem] = Field(default_factory=list)
```

| Champ cible | Type | Sémantique |
| --- | --- | --- |
| `supplier_name` | `str` | Nom du fournisseur, forme juridique comprise lorsqu'elle est présente |
| `invoice_date` | `date` | Date de facture ; sérialisation JSON `AAAA-MM-JJ` |
| `total_ht` | `float` | Total net hors taxes, dans la devise du document |
| `total_ttc` | `float` | Total taxes comprises, dans la même devise |
| `line_items` | `list[LineItem]` | Lignes ordonnées ; liste vide par défaut au niveau du modèle Pydantic |
| `line_items[].description` | `str` | Désignation de l'article ou de la prestation |
| `line_items[].quantity` | `float` | Quantité entière ou fractionnaire |
| `line_items[].unit_price` | `float` | Prix unitaire net après remise, hors taxes |
| `line_items[].total_ht` | `float` | Montant net hors taxes de la ligne |

Exemple de sortie attendue :

```json
{
  "supplier_name": "Exemple Services SAS",
  "invoice_date": "2025-09-23",
  "total_ht": 100.0,
  "total_ttc": 120.0,
  "line_items": [
    {
      "description": "Prestation de conseil",
      "quantity": 2.0,
      "unit_price": 50.0,
      "total_ht": 100.0
    }
  ]
}
```

Le schéma seul ne vérifie pas les identités financières, la positivité ou
l'interdiction des champs supplémentaires. Le générateur contrôle les calculs
des annotations ; l'évaluateur ajoute des vérifications de types et de
montants finis. Il ne faut pas confondre validation Pydantic et conformité
comptable exhaustive.

### Fournisseur, client, métadonnées, devise et banque : où sont les données ?

Chaque ligne JSONL est un enregistrement contenant `input`, `output` et des
attributs de traçabilité. **Seuls le texte `input` et la cible `output`
participent à la construction des exemples d'entraînement.**

| Information | Présence dans la v3 | Extraite dans `output` ? |
| --- | --- | --- |
| Fournisseur | Nom, adresse, contact et immatriculation dans le texte/PNG | **Nom uniquement** : `supplier_name` |
| Client | Nom et adresse dans le texte/PNG | Non ; aucun modèle `Customer` |
| Facture | Numéro, date, référence de commande, échéance et conditions dans le texte/PNG | **Date uniquement** : `invoice_date` |
| Devise | Symbole/code dans le document ; `currency` au niveau JSONL | Non ; aucun champ `currency` dans `Invoice` |
| Langue/région | `locale` au niveau JSONL | Non |
| Articles et totaux | Dans le document et la cible structurée | Oui, selon les champs ci-dessus |
| Remises et taxes détaillées | Dans le document ; `line_details` au niveau JSONL | Non ; seuls les prix nets et totaux associés sont prédits |
| Banque | IBAN/BIC français, sort code/compte britannique ou instructions ACH américaines dans le texte/PNG | Non ; aucun modèle `BankDetails` |
| Traçabilité | Identifiant, version, template, rendu et chemin d'image au niveau JSONL | Non |

Les attributs JSONL hors cible sont :

| Attribut | Rôle |
| --- | --- |
| `id`, `dataset_version` | Identifiant d'exemple et version `international-v3` |
| `format`, `locale`, `currency` | Segmentation textuelle, linguistique et monétaire |
| `template_id`, `template_reference` | Famille de structure et référence visuelle |
| `date_style`, `visual` | Format de date et paramètres typographiques/graphiques |
| `line_details` | Prix catalogue, pourcentage de remise, taux et montant de taxe par ligne |
| `image_path` | Chemin relatif à la racine du projet vers le PNG correspondant |
| `input`, `output` | Texte source et objet cible d'extraction |

L'inférence reçoit le texte et les instructions d'extraction, **jamais la
cible ni les métadonnées de segmentation**. Le modèle doit lire la devise
dans le document pour interpréter les montants, mais ne la restitue pas dans
le schéma actuel. Ajouter client, devise ou banque à la sortie nécessiterait
une évolution coordonnée du schéma, des annotations, des prompts et du scoring.

---

<a id="installation"></a>

## ⚙️ Installation et génération

Toutes les commandes ci-dessous s'exécutent **depuis la racine du projet**.
Pour la pile GPU, privilégier un runtime Python compatible avec PyTorch,
Unsloth et les versions du manifeste d'entraînement ; les manifests de
dépendances ne constituent pas un environnement verrouillé.

### Données, rendu et contrôles sans entraînement

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Réécrit les 600 exemples et leurs images.
python data/generate_dataset.py

# Réécrit train.jsonl, val.jsonl et test.jsonl.
python src/split_data.py

# Contrôle le split de test sans charger ni télécharger de modèle.
python src/evaluate_finetuned.py --check-data-only
```

**Attention à la provenance :** la régénération remplace les fichiers de
données. Conserver les versions et hashes utilisés pour entraîner un modèle
avant de produire un nouveau dataset. Après génération, relancer la
répartition ; ne pas mélanger des splits de générations différentes.

### Inférence LoRA sur GPU

Les dépendances GPU sont séparées de celles de génération :

```bash
python -m pip install -r requirements-inference.txt
```

Ce fichier ajoute PyTorch, Transformers, PEFT, Accelerate et bitsandbytes
sous Linux. Vérifier le pilote CUDA, la compatibilité des paquets et la VRAM
dans le runtime cible. Les versions réellement utilisées pour l'entraînement
sont consignées dans `training_manifest.json` ; les réutiliser autant que
possible pour l'inférence.

Le backend GGUF nécessite séparément `llama-cpp-python`, avec une compilation
adaptée à la machine :

```bash
python -m pip install llama-cpp-python
```

L'installation locale de ces paquets ne lance ni entraînement ni évaluation.

---

<a id="fine-tuning"></a>

## 🚀 Fine-tuning sur Colab ou Kaggle

Le point d'entrée d'entraînement est
[notebooks/fine_tune_unsloth.ipynb](notebooks/fine_tune_unsloth.ipynb).

1. Activer un GPU compatible, par exemple Colab T4 ; sur Kaggle, activer
   également Internet.
2. Fournir les **fichiers v3 locaux**
   [train.jsonl](data/train.jsonl) et [val.jsonl](data/val.jsonl).
   Un clone du dépôt n'inclut pas les changements locaux non publiés.
3. Sur Colab, placer les JSONL dans `/content/data/` ou `/content/`.
   Sur Kaggle, les joindre comme dataset privé.
4. Régler `DATA_DIR_OVERRIDE` si plusieurs copies sont détectées.
5. Exécuter les cellules dans l'ordre ; redémarrer le runtime si
   l'installation des dépendances le demande.
6. Récupérer l'archive du run avant expiration du runtime et l'extraire dans
   [results/](results/). Aucun fichier distant n'est automatiquement
   synchronisé vers cet espace de travail.

| Paramètre du notebook | Valeur par défaut |
| --- | --- |
| Modèle | `unsloth/Qwen2.5-3B-Instruct` |
| Adaptation | QLoRA, modèle chargé en 4 bits |
| Rang / alpha LoRA | `16` / `16` |
| Modules ciblés | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Époques / learning rate | `3` / `2e-4` |
| Batch par GPU / accumulation | `1` / `8` |
| Longueur maximale | `4096` tokens |
| Optimiseur / scheduler | `adamw_8bit` / linéaire, 5 pas de warmup |
| Supervision | Réponses assistant uniquement |
| Évaluation / checkpoints | Tous les 10 pas / chaque époque, 2 checkpoints conservés |
| Seed | `42` |

Le notebook vérifie les effectifs, la version, l'équilibre régional et
l'absence de chevauchement train/validation. Les longueurs des deux
sous-ensembles sont contrôlées avant entraînement : un exemple trop long
arrête l'exécution, sans troncature silencieuse.

Les adaptateurs et le tokenizer sont sauvegardés avant l'évaluation finale
et le tracé des courbes. Le manifeste décrit le modèle de base, sa révision
lorsqu'elle est disponible, les hashes des données, la configuration, les
versions des paquets et les métriques de train/validation.

**Les poids LoRA ne sont pas un modèle autonome** : ils nécessitent le modèle
de base déclaré dans `adapter_config.json`. `EXPORT_GGUF=True` active une
conversion fusionnée Q4_K_M facultative et actualise l'archive. Les envois
Hub et W&B sont désactivés par défaut. Sur Colab, `DOWNLOAD_ARCHIVE=True`
permet de déclencher le téléchargement du ZIP ; sur Kaggle, sauvegarder la
version avec ses outputs.

### 📉 Courbe de loss et stabilité de l'apprentissage

![Courbes de perte d'entraînement et de validation du fine-tuning QLoRA](assets/loss_curve.png)

*Évolution des pertes d'entraînement (bleu) et de validation (orange) au fil
des pas d'optimisation.*

La perte d'entraînement diminue rapidement, puis reste faible malgré quelques
fluctuations ponctuelles. La perte de validation baisse au début du run et
se stabilise autour de **0,006**, sans remontée durable visible en fin
d'entraînement. Ces courbes suggèrent une **convergence globalement stable
sur ce run** ; elles ne suffisent pas à exclure le surapprentissage ni à
garantir la généralisation à des factures réelles ou à des templates inédits.
La qualité d'extraction se mesure séparément dans le
[rapport d'évaluation v3](results/evaluation-3/finetuned_metrics.json).

---

<a id="evaluation"></a>

## 🔎 Inférence et évaluation

Le script [src/evaluate_finetuned.py](src/evaluate_finetuned.py) utilise le
même prompt d'extraction que le notebook. Il prend en charge **soit un
répertoire LoRA, soit un fichier GGUF**.

### Précontrôle des données et des poids reçus

Remplacer la valeur ci-dessous par le nom réel du dossier extrait :

```bash
RUN_NAME="qwen25-3b-international-v3-REMPLACER"
ADAPTER_DIR="results/$RUN_NAME/lora_adapter"

python src/evaluate_finetuned.py \
  --adapter-path "$ADAPTER_DIR" \
  --check-data-only
```

Ce mode vérifie les 60 exemples de test, les devises, les cibles, les IDs et
textes uniques, ainsi que l'absence de chevauchement avec les fichiers
train/validation voisins **lorsqu'ils sont présents**. Conserver les trois
splits ensemble pour bénéficier de ce contrôle.

Pour les poids, le précontrôle vérifie la présence de fichiers non vides
et la configuration LoRA. Si un manifeste est présent, il doit déclarer un
run v3 terminé ; les hashes train/validation disponibles doivent correspondre
aux fichiers locaux. Un écart bloque l'évaluation ; une provenance manquante
produit un avertissement et apparaît dans `training_hashes_verified`.

**Ce contrôle ne désérialise pas les safetensors et ne prouve pas le
chargement du modèle.** Sans `--adapter-path`, il reste utilisable avant
réception des poids, sans dépendances GPU.

### Smoke test puis évaluation complète

Dans le même shell, avec les variables définies ci-dessus :

```bash
# Vérifier chargement, génération et écriture des résultats.
python src/evaluate_finetuned.py \
  --adapter-path "$ADAPTER_DIR" \
  --load-in-4bit \
  --limit 3 \
  --output-dir "results/$RUN_NAME/smoke-test"

# Mesurer les performances sur les 60 factures de test.
python src/evaluate_finetuned.py \
  --adapter-path "$ADAPTER_DIR" \
  --load-in-4bit \
  --output-dir "results/$RUN_NAME/evaluation-v3"
```

L'évaluateur charge le modèle de base déclaré par PEFT, puis les adaptateurs
en mode inférence et le tokenizer sauvegardé. Une révision du manifeste est
réutilisée lorsque l'identifiant de base correspond. Si Unsloth a résolu un
autre identifiant, un avertissement signale l'écart ; le script utilise la
base/révision déclarée par l'adaptateur plutôt qu'une révision potentiellement
étrangère. Vérifier cet avertissement contre la provenance de l'entraînement.

Le modèle de base est téléchargé s'il n'est pas en cache ; les factures et
prédictions ne sont pas envoyées à un service d'inférence.
`--device cpu`, sans `--load-in-4bit`, est prévu pour une machine disposant
d'assez de RAM pour les poids pleine précision, pas pour le petit workspace
CPU actuel.

### Paramètres et sorties

| Option | Défaut / comportement |
| --- | --- |
| `--adapter-path` | Dossier `lora_adapter` ; incompatible avec `--model-path` |
| `--model-path` | Fichier GGUF ; backend alternatif |
| `--test-path` | `data/test.jsonl` ; attend un split v3 complet de 60 factures |
| `--check-data-only` | Vérification sans chargement ni génération |
| `--device` | `cuda` pour le backend LoRA ; `cpu` possible |
| `--load-in-4bit` | Quantification NF4 du modèle de base avec bitsandbytes ; LoRA sur CUDA |
| `--n-ctx` | `4096` tokens |
| `--max-new-tokens` | `2048` tokens |
| `--warmup` | `1` requête exclue des mesures |
| `--limit` | 1 à 60 premiers exemples ; `full_test=false` si sous-ensemble |
| `--output-dir` | Nouveau dossier ; un dossier existant est refusé |

Le décodage n'échantillonne pas. En LoRA, si le prompt et le budget demandé
dépassent le contexte autorisé, une erreur explicite impose d'ajuster les
limites ; aucun texte n'est coupé automatiquement. Ne pas dépasser le contexte
entraîné ni interpréter des sorties tronquées comme un benchmark complet.

Chaque évaluation produit :

- **`predictions.jsonl`** : ID, segments, réponse brute, compteurs de tokens,
  motif d'arrêt, exactitude par champ, erreurs JSON/schéma et latence ;
- **`metrics.json`** : métriques, hashes du test et des poids, provenance,
  paramètres de génération, statut de couverture complète et agrégations
  `per_locale`, `per_currency`, `per_format`, `per_template`.

Sans `--output-dir`, un dossier horodaté est créé sous [results/](results/).
Les rapports historiques ne sont pas écrasés par cet évaluateur.

### Backend GGUF

```bash
python src/evaluate_finetuned.py \
  --model-path "chemin/vers/modele.gguf" \
  --output-dir results/evaluation-gguf-v3
```

La simple présence du GGUF local ne certifie ni son origine ni l'application
des nouveaux adaptateurs. Vérifier sa provenance avant de le qualifier de
modèle brut ou fine-tuné.

---

<a id="resultats"></a>

## 📊 Comparaison des modèles et métriques

### Comparaison v3 : base brute versus adaptation

Le [rapport final v3 fourni](results/evaluation-3/finetuned_metrics.json)
déclare **100 % de validité JSON** et **95 % d'Exact Match** pour les
adaptateurs du run Colab. Seuls ces deux scores sont disponibles :
les autres indicateurs restent « À mesurer ».

| Modèle | Adaptation métier | Validité JSON déclarée | Exactitude moyenne des champs | Exact Match déclaré | Latence / tokens·s⁻¹ | Statut v3 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B-Instruct brut | Aucune, prompt d'extraction seul | À mesurer | À mesurer | À mesurer | À mesurer | Référence LLM à établir |
| **Qwen2.5-3B-Instruct + LoRA** | QLoRA sur 480 factures, validation sur 60 ; 3 époques terminées | **100 %** | À mesurer | **95 %** | À mesurer | Rapport v3 fourni, `status=success` ; adaptateurs identifiés |

> [!NOTE]
> Le rapport synthétique contient le statut, le chemin des adaptateurs et les
> deux taux ci-dessus, mais pas l'effectif évalué, le nom du split, le hash
> du test, les paramètres de scoring ou les résultats par segment. Les scores
> sont repris tels que fournis, sans déduire un nombre de factures correctes
> ni assimiler le taux JSON à une validation Pydantic prouvée. Archiver le
> rapport détaillé et les prédictions pour une comparaison reproductible.

### Run Colab reçu et provenance des résultats

Les adaptateurs sont disponibles dans
[lora_adapter/](results/qwen25-3b-international-v3-20261021T053959288046Z/lora_adapter/).
Le [manifeste du run](results/qwen25-3b-international-v3-20261021T053959288046Z/training_manifest.json)
confirme un entraînement `international-v3` terminé :

| Indicateur d'entraînement | Valeur enregistrée dans le manifeste |
| --- | ---: |
| Exemples train / validation | 480 / 60 |
| Époques | 3 |
| Durée d'entraînement | 2 419,4351 s, soit environ 40 min 19 s |
| `train_loss` | 0,00746108 |
| `eval_loss` | 0,00590958 |

**Ces pertes ne sont ni des taux d'extraction ni des résultats sur le test.**
Le précontrôle des 60 factures et la correspondance des hashes train/validation
avec le manifeste ont réussi. Les trois fichiers attendus de l'adaptateur
sont présents ; l'en-tête safetensors contient 504 tenseurs avec des offsets
dans les limites du fichier. Cela ne remplace pas un chargement effectif.

La tentative d'évaluation locale précédente s'était arrêtée sur
`ModuleNotFoundError: No module named 'torch'`, avant génération, dans un
environnement sans GPU NVIDIA disponible. Elle n'a pas produit les scores
affichés ici : ceux-ci proviennent du rapport final désormais fourni dans
[results/evaluation-3/finetuned_metrics.json](results/evaluation-3/finetuned_metrics.json).
Ce rapport identifie bien le répertoire des adaptateurs du run reçu.
Les rapports historiques ci-dessous restent distincts de ces résultats v3.

L'[extracteur à règles](src/evaluate_baseline.py) est une baseline distincte :
**ce n'est pas Qwen brut**. Son script reste historique ; son exécution
réécrit [results/baseline_metrics.json](results/baseline_metrics.json).

Pour une comparaison contrôlée :

1. Conserver exactement les mêmes 60 factures et le même hash de test.
2. Fixer prompts, scoring, budgets de génération, matériel et warmup.
3. Comparer à backend et quantification identiques, ou déclarer explicitement
   ces différences.
4. Mesurer séparément Qwen sans adaptation et Qwen adapté, sans réglage des
   hyperparamètres sur le test.
5. Archiver les rapports et leur provenance avant toute mise à jour du tableau.

Le CLI actuel n'a pas de mode Hugging Face « base seule » :
`--adapter-path` applique toujours LoRA. Une comparaison sans modifier le
code est possible avec **deux GGUF de provenance vérifiée**, base et modèle
fusionné fine-tuné, quantifiés de manière comparable :

```bash
python src/evaluate_finetuned.py \
  --model-path "chemin/vers/base-verifie.Q4_K_M.gguf" \
  --output-dir results/comparaison-v3-base

python src/evaluate_finetuned.py \
  --model-path "chemin/vers/finetune-verifie.Q4_K_M.gguf" \
  --output-dir results/comparaison-v3-finetune
```

Le libellé générique `model` du rapport GGUF n'atteste pas l'adaptation :
consigner le rôle base/fine-tuné avec le chemin, le hash et la provenance.

### Définitions des métriques

| Métrique | Définition |
| --- | --- |
| `json_validity_rate` | Réponse entière constituée d'un objet JSON strict, sans prose/Markdown, clés dupliquées ni constantes NaN/Infinity |
| `schema_validity_rate` | Objet direct ou récupéré validé par Pydantic, avec montants finis |
| `recovered_json_rate` | Objet JSON récupérable, y compris dans une réponse enveloppée |
| `field_accuracy` | Exactitude pour chacun des cinq champs cibles |
| `mean_field_accuracy` | Moyenne non pondérée des cinq exactitudes par champ |
| `exact_match_rate` | JSON strict, schéma valide et tous les champs corrects selon les règles de comparaison |
| `recovered_exact_match_rate` | Même exactitude métier, en autorisant un JSON récupéré dans une réponse non strictement JSON |
| `truncation_rate` | Part des générations arrêtées par la limite de longueur |
| Latence et débit | Temps par requête et tokens générés/seconde, hors chargement et warmup |

Les noms et dates sont comparés exactement. Les nombres sont comparés avec
une tolérance absolue **0,01**, sans accepter chaînes numériques ou booléens
comme nombres corrects. `line_items` exige les mêmes lignes, dans le même
ordre, avec descriptions et valeurs correctes ; son score est **tout ou rien
par facture**. L'exact match n'est pas une égalité octet par octet du JSON.

Le scoring de champs peut exploiter un objet récupéré : consulter les taux
stricts et de récupération conjointement pour ne pas masquer une violation
du format de sortie. Le timing LoRA synchronise CUDA et inclut tokenisation,
génération et décodage, mais pas le scoring.

Les ventilations par langue/région et devise reposent sur les métadonnées
connues du test. Elles ne constituent pas une mesure d'extraction d'un champ
`currency`, absent de la cible.

### Rapports historiques : chiffres à ne pas attribuer à la v3

| Artefact | JSON valide déclaré | Exactitude moyenne déclarée | Exact match déclaré | Latence moyenne | Débit déclaré |
| --- | ---: | ---: | ---: | ---: | ---: |
| [baseline_metrics.json](results/baseline_metrics.json), règles | 46,67 % | 73,33 % | 43,33 % | 0,542 ms | Sans génération |
| [finetuned_metrics.json](results/finetuned_metrics.json), ancien GGUF | 100 % | 95 % | 100 % | 250 ms | 45 tokens/s |

Ces valeurs sont conservées pour traçabilité, **pas comme preuve d'un gain
sur la v3**. Le rapport GGUF n'indique ni split, ni effectif, ni hash du test.
Son exact match de 100 % et son exactitude moyenne de 95 % sont incompatibles
avec les règles actuelles sur un même ensemble d'exemples : cette incohérence
doit être levée par une nouvelle mesure. Les anciennes règles JSON diffèrent
également du scoring strict actuel. Aucun gain chiffré fiable n'est donc
déduit de ces deux fichiers.

---

<a id="validation"></a>

## 🛡️ Validation, limites et mise en production

### Tests logiciels

Après génération et répartition des données :

```bash
python -m unittest discover -s tests -v
```

Les suites couvrent :

- diversité, équilibre, reproductibilité, calculs, rendus et fichiers persistés ;
- configuration du notebook, format des prompts et contrôle des longueurs ;
- chargement LoRA simulé, scoring, erreurs, provenance et écriture des rapports.

Les tests de modèles utilisent des doubles de test. **Ils ne démontrent ni
la compatibilité des vrais poids téléchargés ni les performances du modèle.**

### Limites connues et critères avant production

- Les neuf familles de structures sont partagées entre train, validation et
  test : on mesure une généralisation **intra-famille**, pas des layouts
  entièrement inconnus.
- Les 60 factures de test donnent des segments réduits ; les scores doivent
  toujours être interprétés avec leurs effectifs.
- Les données sont synthétiques et les rendus propres : scans, rotations,
  bruit OCR, tableaux multipages et factures réelles restent à tester.
- La couverture FR/EUR, UK/GBP et US/USD ne garantit pas une extraction
  universelle, même pour un recruteur ou un client test.
- Client, devise explicite et banque ne font pas partie des champs extraits.
- Les dépendances GPU, la mémoire requise et le chargement réel des adaptateurs
  doivent être vérifiés sur l'environnement d'exécution.

Avant mise en production : charger les poids importés sur GPU, réaliser le
smoke test, évaluer le test complet, établir une référence
Qwen brut comparable, puis tester des documents réels et des familles de
templates inédites. Un parcours utilisateur image/PDF doit enfin être évalué
**de bout en bout**, OCR compris, avec gestion explicite des erreurs.
