# LLM-extraction-finetune

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Gradio](https://img.shields.io/badge/Gradio-6.30.0-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/) [![Tesseract OCR](https://img.shields.io/badge/Tesseract-fra%20%2B%20eng-22C55E?style=for-the-badge)](https://tesseract-ocr.github.io/) [![OpenCV](https://img.shields.io/badge/OpenCV-headless-10B981?style=for-the-badge)](https://opencv.org/) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/latest/)

[![Claude Code](https://img.shields.io/badge/Claude_Code-outil_de_developpement-D97757?style=for-the-badge&logo=claude&logoColor=white)](https://github.com/anthropics/claude-code)

## 📖 Description

**LLM-extraction-finetune** est un projet d'extraction automatisée de données de factures PDF et
images, développé en Python. Il transforme des documents français, britanniques et américains en
données JSON structurées grâce à un pipeline OCR combinant analyse documentaire, extraction des
champs et validation des résultats.

Le système identifie fournisseur, date, articles, quantités, prix, taxes et totaux, et rend avec le JSON
un **rapport de vérification** qui distingue ce qui est confirmé, ce qui est anormal et ce qui reste
insuffisamment étayé.

La démo publique fonctionne **sans chargement de modèle de langage** (`INVOICE_MODEL=0`) : seul le
pipeline OCR traite le document. Les visiteurs peuvent téléverser leurs propres factures — PDF, PNG ou
JPEG, 20 Mo au plus — ou partir des exemples fournis. Le document est traité sur l'infrastructure
Hugging Face de l'Espace, et les fichiers créés pour un visiteur restent à l'écart des autres visiteurs
par le contrôle d'accès lié à la session.

**Démo publique — pipeline OCR en ligne :** <https://nicolbl95-llm-extraction-finetune.hf.space>

## 📸 Aperçu de l'interface

![Interface après extraction de l'exemple France : rapport de vérification, JSON structuré et durée mesurée](docs/assets/apercu-interface-extraction.png)

*Capture réelle prise après une extraction complète de l'exemple synthétique « France · EUR », dans
la configuration de la démo publique. Fournisseur, client, montants et identifiants y sont fictifs.*

## ✨ Fonctionnalités clés

- **Lecture PDF et image** — jusqu'à 10 pages et 20 Mo, PDF à couche texte native et PDF scanné.
- **OCR positionné** — Tesseract `fra` + `eng` avec coordonnées et confiances ; les prétraitements (redressement, contraste, recadrage, échelle) ne sont escaladés qu'en cas de doute.
- **Devises et écritures régionales** — FR/EUR, UK/GBP et US/USD, dates et numéros traités par locale.
- **Rapport de vérification** — statut champ par champ, anomalies documentaires, contrôle de clé SIRET, JSON exportable.

## 🧠 Architecture du système

```text
Document (PDF, image) → Prétraitement → OCR positionné → Extraction → Vérification → JSON + rapport
```

L'architecture est modulaire : acquisition et prétraitement ([`src/ocr_preprocessing.py`](src/ocr_preprocessing.py),
[`src/ocr_recovery.py`](src/ocr_recovery.py)), preuve documentaire et lignes ([`src/invoice_evidence.py`](src/invoice_evidence.py)),
ancrage des montants sur leurs opérandes ([`src/invoice_grounding.py`](src/invoice_grounding.py)), cohérence arithmétique
et documentaire ([`src/invoice_validation.py`](src/invoice_validation.py)), contrat de sortie ([`data/schema.py`](data/schema.py))
et interface Gradio unique ([`app/main.py`](app/main.py)). Étapes et lecture des mesures : [docs/DESIGN.md](docs/DESIGN.md).

**Expérimentation LLM :** le projet a également exploré le fine-tuning de Qwen2.5-3B avec QLoRA sur des
factures synthétiques. Cette approche expérimentale n'est pas utilisée dans la démo publique actuelle, qui
repose sur un pipeline OCR et des règles de validation. Le notebook d'entraînement
([`notebooks/fine_tune_unsloth.ipynb`](notebooks/fine_tune_unsloth.ipynb)), l'évaluateur
([`src/evaluate_finetuned.py`](src/evaluate_finetuned.py)) et leurs mesures historiques
([`results/baseline_metrics.json`](results/baseline_metrics.json), [`results/evaluation-3/finetuned_metrics.json`](results/evaluation-3/finetuned_metrics.json))
restent dans le dépôt ; ils exigent un GPU et aucun poids adapté n'est versionné.

## 🛠️ Technologies utilisées

**Moteur déployé** (Space et local) :

| Technologie | Rôle |
| --- | --- |
| Python 3.12 | interpréteur déployé par le Space (`python_version: "3.12"`) |
| Gradio 6.30 | interface et serveur ; `app/main.py` est le point d'entrée du Space |
| Tesseract OCR (`pytesseract`) | OCR positionné en TSV, langues `fra` et `eng` |
| OpenCV (headless) | redressement, contraste, recadrage, débruitage, mise à l'échelle |
| PyMuPDF | ouverture et rendu des pages de PDF, couche texte native |
| Pydantic v2 | contrat de sortie et validation stricte |
| Hugging Face Spaces | hébergement de la démo, `requirements.txt` et `packages.txt` |

**Technologies expérimentées** — non exécutées par la démo publique :

| Technologie | Rôle |
| --- | --- |
| Qwen2.5-3B-Instruct | modèle de base adapté par QLoRA, mesuré sur le split de test |
| PyTorch, Transformers, PEFT | entraînement et vérification modèle sur GPU |
| Unsloth | accélération de l'entraînement exploratoire |
| pytest | tests logiciels hors ligne (`requirements-dev.txt`, jamais installée par le Space) |

**Outils de développement** : [Claude Code](https://github.com/anthropics/claude-code), ayant assisté le
développement du projet — aucun composant exécuté par l'application.

## 📂 Structure du projet

- **`app/`** — interface Gradio, point d'entrée du Space et garde d'import ASGI.
- **`src/`** — pipeline OCR, preuve documentaire, ancrage des montants, cohérence, évaluation.
- **`data/` · `tests/`** — contrat Pydantic, générateur du corpus synthétique, suite hors ligne et fixtures.
- **`scripts/` · `notebooks/`** — latence, robustesse, diagnostics, outillage du benchmark indépendant ; entraînement QLoRA (travaux de recherche).
- **`docs/`** — environnement, conception, déploiement, protocoles et capture d'interface.

## 🚀 Installation locale

Prérequis : **Python 3.12** et **Tesseract** avec les données `fra` **et** `eng`. L'installation
Windows par défaut ne livre qu'`eng` et `osd` ; sans `fra`, Tesseract lit l'anglais sans lever d'erreur.

```powershell
git clone https://github.com/nicolbl95/LLM-extraction-finetune.git
cd LLM-extraction-finetune
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs   # par défaut : eng et osd seulement
$env:TESSDATA_PREFIX = "$PWD\.venv\tessdata"   # dossier où l'on dépose fra.traineddata, hors Git
$env:INVOICE_MODEL   = "0"                     # comme la démo publique : OCR déterministe seul
.\.venv\Scripts\python.exe app\main.py         # interface sur http://127.0.0.1:7860
```

Sous Linux, `packages.txt` donne les paquets apt et `fra` y est installé. Dépendances réellement importées,
données de langue, variables d'environnement et commande de référence des tests : [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## 🛡️ Validation et contrôle des erreurs

- **Justification par le document** — une valeur n'est retenue que si le document l'imprime ou la recoupe.
- **Recalculs contrôlés** — quantité × prix unitaire, somme des lignes, net + TVA, en arithmétique `Decimal`.
- **Anomalies signalées** — chaque incohérence est nommée dans le rapport, jamais masquée silencieusement.
- **Aucune invention** — un champ incertain sort `null` ; le moteur n'exécute pas d'auto-correction par LLM.

## 🧪 Corpus, tests et résultats

**Un test qui passe n'est pas une mesure de précision sur une facture inédite.** Trois registres à ne pas confondre :

| Registre | Ce qui a réellement été exécuté | Résultat |
| --- | --- | --- |
| **Tests logiciels** | 1 328 tests automatisés hors ligne, sans appel de modèle de langage, plus 26 qui réclament les ressources non versionnées (corpus PNG régénéré, diagnostics privés) et 3 marqués `slow` écartés d'office | **1 328 réussis, 26 ignorés, 0 échec, 0 erreur** |
| **Factures synthétiques** | 600 textes et 600 PNG générés pour le développement ([`data/generate_dataset.py`](data/generate_dataset.py)) ; les tests vérifient leur intégrité et leur partitionnement (480 / 60 / 60), rendent 27 variantes gabarit × locale et extraient 8 images de fixtures versionnées | **aucune évaluation d'exactitude sur les 600 documents** ; les PNG sont régénérés localement, non versionnés |
| **Factures réelles** | une dizaine essayées manuellement dans l'interface ; la validation indépendante exige douze factures annotées à l'aveugle ([`scripts/pr110_ready.py`](scripts/pr110_ready.py)) | **0 / 12** admissibles, donc **aucun taux de précision publié** pour le moteur servi |

```powershell
pip install -r requirements-dev.txt
python -X utf8 -m pytest -q -p no:cacheprovider tests   # TESSDATA_PREFIX doit rester défini
```

## 👨‍💻 Auteur et développement

Développé par **Nicolas DAVOINE** ([`nicolbl95`](https://github.com/nicolbl95)), avec l'assistance de
**Claude Code** — démo : <https://nicolbl95-llm-extraction-finetune.hf.space>
