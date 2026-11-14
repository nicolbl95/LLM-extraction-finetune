---
title: Extraction standardisée de données de factures
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app/main.py
pinned: false
---

# LLM-extraction-finetune

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Gradio](https://img.shields.io/badge/Gradio-6.30.0-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/) [![Tesseract OCR](https://img.shields.io/badge/Tesseract-fra%20%2B%20eng-22C55E?style=for-the-badge)](https://tesseract-ocr.github.io/) [![OpenCV](https://img.shields.io/badge/OpenCV-headless-10B981?style=for-the-badge)](https://opencv.org/) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/latest/)

**Démo publique — pipeline OCR en ligne :** <https://nicolbl95-llm-extraction-finetune.hf.space>

## 📖 Description

**Le problème.** Une facture est un document — PDF de logiciel de facturation ou photo d'un imprimé —
dont la mise en page porte l'information autant que le texte. En extraire fournisseur, date, totaux et
lignes d'articles demande une lecture documentaire, pas une expression régulière.

**La solution.** Un pipeline dédié transforme ce document en **JSON structuré et audité** : rendu et
prétraitement, OCR positionné, regroupement des fragments en champs et en lignes, ancrage des montants
sur les valeurs réellement imprimées, contrôles de cohérence arithmétique et documentaire, contrat
Pydantic. Chaque exécution rend aussi un rapport distinguant ce qui est confirmé, anormal ou invérifiable,
sur les formats FR/EUR, UK/GBP et US/USD.

**Le volet LLM.** Le projet confronte cette voie déterministe à un petit modèle de langage :
**Qwen2.5-3B-Instruct** adapté par **QLoRA** (Transformers + PEFT) sur un corpus de 600 factures
synthétiques, puis mesuré sur le même split de test que la baseline par règles. Actuellement,
**la démo publique n'exécute aucun modèle de langage** : elle sert le pipeline OCR déterministe seul
(`INVOICE_MODEL=0`).

## 📸 Aperçu de l'interface

![Interface après extraction de l'exemple France : rapport de vérification, JSON structuré et durée mesurée](docs/assets/apercu-interface-extraction.png)

*Capture réelle prise après une extraction complète de l'exemple synthétique « France · EUR », dans la
configuration de la démo publique. Fournisseur, client, montants et identifiants y sont fictifs.*

## ✨ Fonctionnalités clés

- **Lecture PDF et image** — jusqu'à 10 pages et 20 Mo ; un PDF à couche texte native et un PDF scanné
  empruntent deux chemins distincts.
- **OCR positionné** — Tesseract `fra` + `eng` avec coordonnées et confiances ; les prétraitements
  (redressement, contraste, recadrage, échelle) ne sont escaladés qu'en cas de doute.
- **Champs et lignes de facturation** — fournisseur, date, total net et total TTC, puis description,
  quantité, prix unitaire et montant net de chaque ligne, dans l'ordre imprimé.
- **Vérification des montants et des taxes** — quantité × prix unitaire = total de ligne, somme des
  lignes = total net, net + TVA lue = total TTC, en arithmétique `Decimal`.
- **Incertitude assumée** — une valeur insuffisamment étayée sort `null` et est **expliquée** dans le
  rapport ; elle n'est jamais recalculée ni remplacée par une hypothèse.
- **Sorties exportables** — JSON validé par contrat, statut compact des champs recoupés, anomalies
  documentaires et contrôle de clé SIRET.

## 🧠 Architecture du système

```text
Document (PDF, image) → Prétraitement → OCR positionné → Extraction → Vérification → JSON + rapport
```

L'architecture est modulaire : acquisition et prétraitement ([`src/ocr_preprocessing.py`](src/ocr_preprocessing.py),
[`src/ocr_recovery.py`](src/ocr_recovery.py)), preuve documentaire et segmentation en lignes
([`src/invoice_evidence.py`](src/invoice_evidence.py)), ancrage des montants sur leurs opérandes
([`src/invoice_grounding.py`](src/invoice_grounding.py)), cohérence arithmétique et documentaire
([`src/invoice_validation.py`](src/invoice_validation.py)), contrat de sortie ([`data/schema.py`](data/schema.py)) —
le tout orchestré par une interface Gradio unique ([`app/main.py`](app/main.py)).

Le volet expérimental reste hors du chemin servi : un notebook QLoRA entraîne les adaptateurs
([`notebooks/fine_tune_unsloth.ipynb`](notebooks/fine_tune_unsloth.ipynb)), un évaluateur les compare à la
baseline par règles ([`src/evaluate_finetuned.py`](src/evaluate_finetuned.py)) ; ces scripts exigent un GPU
et des poids non versionnés. Étapes, seuils et mécanismes de recoupement : [docs/DESIGN.md](docs/DESIGN.md).

## 🛠️ Technologies utilisées

**Pipeline déployé** (Space et local) :

| Technologie | Rôle |
| --- | --- |
| Python 3.12 | interpréteur déployé par le Space (`python_version: "3.12"`) |
| Gradio 6.30 | interface et serveur ; `app/main.py` est le point d'entrée du Space |
| Tesseract OCR (`pytesseract`) | OCR positionné en TSV, langues `fra` et `eng` |
| OpenCV (headless) | redressement, égalisation de contraste, recadrage, débruitage, mise à l'échelle |
| PyMuPDF | ouverture et rendu des pages de PDF, exploitation de la couche texte native |
| Pydantic v2 | contrat de sortie et validation stricte |
| Hugging Face Spaces | hébergement de la démo, installation de `requirements.txt` et `packages.txt` |

**Expérimentation LLM et qualification logicielle** — non exécutées par la démo publique :

| Technologie | Rôle |
| --- | --- |
| Qwen2.5-3B-Instruct | modèle de base adapté par QLoRA, puis mesuré sur le split de test |
| PyTorch | exécution de l'entraînement et des vérifications modèle sur GPU |
| Hugging Face Transformers | chargement du modèle et du tokenizer, quantification 4 bits |
| PEFT / QLoRA (via Unsloth) | adaptateurs rang 16, alpha 16, entraînés sur 480 factures |
| pytest | suite hors ligne (`requirements-dev.txt`, jamais installée par le Space) |

## 📂 Structure du projet

- **`app/`** — interface Gradio, point d'entrée du Space et garde d'import ASGI.
- **`src/`** — pipeline OCR, preuve documentaire, ancrage des montants, cohérence, évaluation.
- **`data/`** — contrat Pydantic de sortie et générateur du corpus synthétique.
- **`tests/`** — suite hors ligne, fixtures et corpus de robustesse indépendant.
- **`scripts/`** — latence, robustesse, diagnostics et outillage du benchmark indépendant.
- **`notebooks/`** — entraînement QLoRA et vérification du modèle réel sur GPU.
- **`docs/`** — environnement, conception, déploiement, protocoles et capture d'interface.

## 🚀 Installation locale

Prérequis : **Python 3.12** et **Tesseract** avec les données `fra` **et** `eng`. L'installation Windows
par défaut ne livre qu'`eng` et `osd` : Tesseract lit alors l'anglais **sans lever d'erreur**, et le dépôt
ne fournit pas les binaires de langue.

```powershell
git clone https://github.com/nicolbl95/LLM-extraction-finetune.git
cd LLM-extraction-finetune
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs   # par défaut : eng et osd seulement
$env:TESSDATA_PREFIX = "$PWD\.venv\tessdata"   # dossier où l'on dépose fra.traineddata, hors Git
$env:INVOICE_MODEL   = "0"                     # comme la démo publique : OCR déterministe seul
.\.venv\Scripts\python.exe app\main.py         # interface sur http://127.0.0.1:7860
```

Sous Linux, `packages.txt` donne les paquets apt — `fra` y est installé et `TESSDATA_PREFIX` n'est pas
nécessaire. La démo publique tourne en OCR déterministe seul : garde `INVOICE_MODEL=0` pour la reproduire.

## 🧪 Tests et résultats

**Un test qui passe n'est pas une mesure de précision sur une facture inédite.**

| Registre | Ce qui est réellement mesuré | Résultat |
| --- | --- | --- |
| Tests logiciels | 1 249 tests hors ligne, aucun modèle de langage appelé | **1 249 réussis, 0 échec, 0 erreur** |
| Expérimentation LLM | QLoRA contre la baseline par règles, sur les 60 factures synthétiques du split `test` | validité JSON **46,67 % → 100 %**, Exact Match **43,33 % → 95 %** ([baseline](results/baseline_metrics.json), [fine-tuned](results/evaluation-3/finetuned_metrics.json)) |
| Validation indépendante | exactitude sur des factures étrangères au générateur du dépôt | **en attente** : 12 documents admissibles annotés à l'aveugle |

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# TESSDATA_PREFIX doit rester défini : sans lui la suite s'interrompt plutôt que de simuler
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests
```

Le premier chiffre est une mesure rejouée sur ce poste, sans appel de modèle. Le second est consigné par
un run d'entraînement sur GPU : il mesure une adaptation à des gabarits synthétiques, pas une
généralisation. Le troisième registre n'a **aucun résultat** : la porte `scripts/pr110_ready.py` exige
douze factures indépendantes annotées à l'aveugle et les collectes n'en ont fourni aucune.

## ⚠️ Limites et améliorations possibles

- **La démo publique ne traite que ses trois factures synthétiques** autorisées : un choix de sécurité,
  pas une limite du moteur — en local, le téléversement est libre.
- **La confidentialité entre visiteurs d'un Space public n'est pas démontrée** ; la route fichier de
  Gradio ne connaissant pas les sessions, aucun téléversement arbitraire n'y est accepté.
- **Aucune exactitude n'est établie sur facture inédite** : valider douze factures indépendantes
  annotées à l'aveugle reste le premier chantier ([protocole](benchmark_prep/OWNER_GUIDE.md)).
- **La comparaison OCR / LLM est à approfondir** sur ce même corpus indépendant, et non sur des jeux
  synthétiques partagés, pour mesurer une généralisation plutôt qu'une adaptation.
- **Les poids QLoRA ne font pas partie du déploiement** : non versionnés, reproduction supposant un GPU.

## 👨‍💻 Auteur et documentation

Développé par **nicolbl95** — [<https://github.com/nicolbl95>](https://github.com/nicolbl95) ·
Démo : [<https://nicolbl95-llm-extraction-finetune.hf.space>](https://nicolbl95-llm-extraction-finetune.hf.space)

- [docs/DESIGN.md](docs/DESIGN.md) — conception du pipeline et lecture des mesures.
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — environnement reproductible : dépendances réellement
  importées, binaire Tesseract, données `fra`, empreintes.
- [benchmark_prep/OWNER_GUIDE.md](benchmark_prep/OWNER_GUIDE.md) — annotation à l'aveugle des douze
  factures indépendantes attendues.
