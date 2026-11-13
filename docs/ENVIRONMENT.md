# Environnement reproductible des tests

Documenté pour la campagne PR135 (reproductibilité) : tout ce que la suite hors ligne
exige, et rien qui ne dépende d'un fichier local non documenté. Les valeurs chiffrées
de ce document sont mesurées sur le poste de développement Windows du 2026-11-12 ;
les commandes de vérification sont données à chaque section.

## 1. Python

- Le Space déploie **Python 3.12** (`python_version: "3.12"` dans l'en-tête YAML du `README.md`).
- Le poste local vérifié : `.venv` en **3.12.0**.
- La suite s'exécute avec l'interpréteur du venv, jamais avec un Python système :
  l'en-tête UTF-8 (`-X utf8`) est requis car des assertions comparent des accents français.

```powershell
.\.venv\Scripts\python.exe --version
```

## 2. Fichiers de dépendances

| Fichier | Rôle | Installé par |
| --- | --- | --- |
| `requirements.txt` | runtime autonome du Space (aucune ligne `-r`) | Hugging Face Space, uniquement celui-ci |
| `requirements-inference.txt` | ajoute le pin Gradio validé pour l'inférence | poste d'inférence / notebook |
| `constraints-hf.txt` | versions observées sur le staging Linux ; **pas** un lockfile transitif | aucune installation directe |
| `requirements-dev.txt` | runner de tests : `pytest`, `packaging` | poste de test uniquement |
| `packages.txt` | paquets apt Linux (Tesseract + `fra` + `eng`) | build Linux du Space |

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Sur Linux, installer en plus le contenu de `packages.txt` (`apt-get install -y $(xargs -a packages.txt)`).

`tests/test_hf_build_requirements.py` interdit toute ligne `-r` dans `requirements.txt` :
le fichier de développement ne peut donc pas entrer dans le build du Space, et le build
ne fournit pas `pytest`.

### Obligatoire, optionnel, absent

- **Obligatoire pour la suite hors ligne** : `requirements.txt` + `requirements-dev.txt`.
  Les import tiers réellement utilisés par les tests suivis sont `pytest`, `packaging`,
  `gradio`, `numpy`, `pydantic`, `PyMuPDF`, `Pillow`, `pytesseract` — contrôlé par
  `tests/test_declared_test_dependencies.py`.
- **Déclaré mais non importé par la suite** : `torch`, `transformers`, `peft`,
  `accelerate`, `bitsandbytes`. Ils ne servent qu'aux tests `slow` opt-in (vrai modèle)
  et à l'évaluateur d'inférence.
- **Délibérément non déclarées** : `rapidocr`, `onnxruntime`. Recherche seule, modules
  expérimentaux derrière flag (`src/dual_ocr_experiment.py`) ; la suite ne doit jamais
  les importer. `datasets`, `huggingface_hub`, `reportlab`, `spaces`, `pandas`,
  `requests` sont déclarées pour la génération de données et le Space.

## 3. Découverte du binaire Tesseract

- Windows : `app/main.py:32-33` fixe `pytesseract.pytesseract.tesseract_cmd` sur
  `C:\Program Files\Tesseract-OCR\tesseract.exe`. C'est le seul chemin absolu Windows
  littéral du code suivi ; il suppose l'emplacement d'installation par défaut.
- Linux / Space : rien n'est épinglé, le binaire est trouvé sur le `PATH` via apt.
- Version mesurée localement : `tesseract v5.4.0.20240606`, `leptonica-1.84.1`.

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

## 4. Données de langue `fra` et `eng` — le point le plus fragile

La production demande `OCR_REQUESTED_LANGUAGES = ("fra", "eng")` (`app/main.py:60`).
L'installation Windows par défaut ne contient que `eng` et `osd` : sans donnée française
supplémentaire, **Tesseract lit en anglais sans lever d'erreur** (cause du diagnostic
PR80, non une régression du pipeline). PR81 rend ensuite la dégradation visible par
`ocr_availability()` et `ocr_language_plan()`.

- Linux : `packages.txt` installe `tesseract-ocr-fra` et `tesseract-ocr-eng` ; le
  `TESSDATA_PREFIX` n'est pas nécessaire.
- Windows : le poste local place les `.traineddata` dans `.venv/tessdata/`, exporté dans
  `TESSDATA_PREFIX`. Ce répertoire est hors Git (`.venv/` ignoré) : **le dépôt ne
  fournit pas les binaires**, l'opérateur doit les provisionner, puis vérifier :

```powershell
$env:TESSDATA_PREFIX = "$PWD\.venv\tessdata"
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs   # doit lister eng, fra, osd
.\.venv\Scripts\python.exe -X utf8 -c "import sys; sys.path[:0]=['.','src']; from app.main import ocr_language_plan; print(ocr_language_plan())"
# attendu : effective == ['eng', 'fra'], fallback False, missing []
```

Empreintes (octets bruts) de la copie locale utilisée par les baselines enregistrées,
pour reconnaître un poste équivalent — pas une promesse d'approvisionnement :

| Fichier | Taille | SHA-256 |
| --- | --- | --- |
| `fra.traineddata` | 1 130 365 | `ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca` |
| `eng.traineddata` | 4 113 088 | `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` |
| `osd.traineddata` | 10 562 727 | `9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff` |

## 5. Flags de fonctionnement

Valeurs vraies : `{"1", "true", "yes", "on"}`. La suite documentée se lance avec
**aucune** variable `INVOICE_*` définie dans l'environnement.

| Variable | Défaut | Lecture |
| --- | --- | --- |
| `INVOICE_PREFETCH` | **activé** sauf faux explicite (`"0"`/`"false"`/`"no"`/`"off"`) | `app/main.py:460` — seul défaut « on » |
| `INVOICE_OCR_FAST_PATH` | arrêté | `src/ocr_fast_path.py:14` |
| `INVOICE_TABLE_SHADOW` | arrêté | `src/table_shadow.py:24` |
| `INVOICE_TABLE_GEOMETRY` | arrêté | `src/table_geometry.py:23` |
| `INVOICE_TABLE_RECOVERY` | arrêté | `src/table_recovery.py:24` |
| `INVOICE_REVIEW_UI`, `INVOICE_REVIEW_EDIT`, `INVOICE_REVIEW_PERSIST` | arrêtés | `src/invoice_review.py:18`, `src/invoice_review_store.py:44` |
| `INVOICE_MODULE_PRELOAD` | arrêté | `app/main.py:453` |
| `INVOICE_ADAPTER_PATH`, `INVOICE_ADAPTER_REPO`, `INVOICE_ADAPTER_REVISION`, `INVOICE_DEVICE`, `INVOICE_LABEL`, `INVOICE_SLOW_ADAPTER_PATH` | sans effet sur la lecture OCR | chargement du modèle, donc tests `slow` seulement |

## 6. Variables d'environnement utilisées par les tests

`GRADIO_TEMP_DIR` (10 modules, cache de téléchargement Gradio), `TESSDATA_PREFIX`
(6 modules), `INVOICE_*` reprises ci-dessus. Aucune variable d'atelier HF
(`HF_HOME`, `TRANSFORMERS_OFFLINE`) n'est lue par la suite.

## 7. Fichiers temporaires — règle PR135 phase 2

Un test ne peut pas écrire son répertoire de travail dans l'arborescence versionnée :
`deployed_revision()` (`app/main.py:1446`) ajoute « (modifications locales non commitées) »
dès que `git status --porcelain` n'est pas vide, donc un artifact égaré modifiait le texte
du rapport d'un autre test. Convention : `tempfile.TemporaryDirectory()` sans `dir=`, donc
le répertoire système. Les seules racines encore épinglées sont ignorées par Git
(`.pytest_cache`). Garde-fou : `tests/test_temp_file_isolation.py` (scan statique des
`dir=` des tests + résultat identique dans un checkout propre et sale).
Cette racine épinglée n'existe pas dans un clone frais — elle est donc créée au démarrage
de la session par `conftest.py` (`pytest_configure`), ce qui rend les tests de persistance
autonomes d'un fichier local non versionné.

## 8. Fins de ligne et normalisation

`.gitattributes` impose `* text=auto eol=lf` : les objets versionnés sont en LF et tout
clone est rendu en LF. Le poste local garde `core.autocrlf=true`, donc un fichier de
l'arborescence peut être CRLF sur disque alors que son blob est LF : **un SHA-256 de
fichier travaillé n'est pas un SHA-256 de blob**. Les quatre définitions sont documentées
et distinguées séparément :

| Définition | Calcul | Reproductible depuis |
| --- | --- | --- |
| octets bruts | SHA-256 des octets exacts du fichier sur disque | le même mode de checkout |
| texte LF | SHA-256 après `\r\n`/`\r` → `\n` (ce que produit `read_text()` puis ré-encodage UTF-8) | n'importe quel checkout |
| blob Git | SHA-256 de `git show <rev>:<path>` | le commit, sur toute machine |
| empreinte de configuration | `OCR_CONFIG.fingerprint()`, aucun fichier | le code de cette révision |

`scripts/verify_freeze_hashes.py` recalcule les quatre pour chaque digest enregistré par un
manifeste de gel et déclare laquelle le reproduit. Mesure PR135 sur le manifeste PR133
(49 digests) : 37 ne sont reproduits que par les **octets bruts CRLF** de ce poste — un clone
propre en calcule d'autres —, 6 sont neutres aux fins de ligne, 1 (`regression_tests.sha256`)
n'existe que sous la définition **texte LF**, 1 est une empreinte de configuration et 3 ont
vu leur contenu changer depuis la capture.

**Règle PR135 :** privilégier les hashes en octets bruts, et les calculer là où les trois
définitions concordent — donc dans un checkout gouverné par `.gitattributes` (voir phase 7).
Un hash pris ainsi s'authentifie aussi par `git show <rev>:<path> | sha256sum`.

`packages.txt` doit rester octet en LF (xargs Linux), contrôlé par
`tests/test_hf_build_requirements.py`.

## 9. Empreintes de configuration runtime

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "import sys; sys.path[:0]=['.','src']; from ocr_config import OCR_CONFIG; print(OCR_CONFIG.version, OCR_CONFIG.fingerprint())"
```

- Politique OCR : `ocr-policy-5-bounded-fastpath-quota` (5 familles, segmentations 4/6,
  budget 45 s par document, seuils rapide 85/60).
- Consensus (Baseline B, `src/ocr_consensus.py`) : SHA-256 octets bruts
  `003fd535e3d46f2db58a9cb3abca494a0ec71f6149b53ef0ca11783fb75205ea`, préfixe long
  `003fd535e3d46f2d`.

## 10. Données locales hors Git dont la suite a besoin

- `data/invoices/*.png` : 0 fichier versionné, répertoire ignoré. Les 600 PNG sont
  **reproductibles** : `python scripts\prepare_offline_test_assets.py` les régénère et
  refuse d'écrire si les enregistrements générés diffèrent de `data/dataset.jsonl` (suivi,
  lui). Les cibles JSON ne sont jamais réécrites.
- `app/examples/{france,uk,us,modele-vierge}.png` : 4 fichiers **suivis**, aucune
  provisionnement nécessaire.
- `results/`, `experiments/*.json`, `benchmark_prep/` : livrables de recherche, ignorés
  ou hors suite ; aucune dépendance de test.
- `.venv/`, poids du modèle, adaptateurs, secrets : jamais dans Git.

## 11. Commande de référence de la suite

```powershell
$env:TESSDATA_PREFIX = "$PWD\.venv\tessdata"
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider tests
```

bash/Linux :

```bash
export TESSDATA_PREFIX="$PWD/.venv/tessdata"
python -X utf8 -m pytest -q -p no:cacheprovider tests
```

Lancer `pytest tests`, pas `pytest` à la racine : la collecte racine importerait des
copies archivées de modules de test hors suite. `pytest.ini` ajoute `-m "not slow"`,
donc les trois tests de vrai modèle restent exclus tant que
`INVOICE_SLOW_ADAPTER_PATH` n'est pas explicitement configuré.
