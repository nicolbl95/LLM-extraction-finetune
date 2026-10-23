# Notes de conception

## Prompt système : lecture littérale des chiffres

Dans [app/main.py](../app/main.py), le prompt système impose une lecture
littérale des chiffres : une quantité `7,2` doit être transcrite `7.2`,
et non `142` ou `72` ; `4,2` devient `4.2`, et non `42` ou `4.02`.
Les parties entière et décimale doivent être préservées, sans concaténation
ni mélange des colonnes. Les quantités et prix ne doivent pas être déduits
par division des totaux. Un exemple few-shot fictif complet illustre le calcul
`7,2 × 12,50 = 90,00`, soit `quantity=7.2`, `unit_price=12.5`, `total_ht=90.0`.
Les valeurs de cet exemple ne doivent pas être recopiées dans la facture réelle.
Le prompt système partagé par l'évaluation et l'application conserve l'instruction
d'entraînement, puis ajoute des garde-fous d'inférence : copie littérale des
descriptions sans lettres omises, exclusion des en-têtes et des libellés
« Sous-total » des articles. Un exemple fictif de tableau montre une description
complète et une quantité décimale, sans transformer le sous-total en article.
Le notebook d'entraînement reste inchangé ; les anciens scores ne mesurent donc
pas ce nouveau prompt.

## Validation arithmétique partagée

Les modèles Pydantic v2 de [data/schema.py](../data/schema.py) appliquent des
`@model_validator(mode="after")` : chaque produit `quantity * unit_price`
est arrondi au centime (`ROUND_HALF_UP`, comme le générateur), puis comparé au
`total_ht` de la ligne ; la somme des montants HT est comparée au `total_ht`
global. La tolérance absolue est de 0,02 inclus, calculée avec `Decimal`.
Les montants extraits ne sont pas réécrits ou arrondis pour masquer un écart ;
les valeurs non finies sont rejetées avant les calculs.

Dans le contrat historique **sans locale et sans preuves** de `data/schema.py`,
les chaînes décimales simples `7,2`, `5,1` et `95,07` sont converties en
nombres sans supprimer leur séparateur. Les formats mixtes (`1.234,56`,
`1,234.56`), les regroupements ambigus (`1,234`), les chiffres séparés par
des espaces ou underscores et les booléens sont rejetés, plutôt que devinés.
La virgule suivie de trois chiffres après une partie entière de 1 à 3 chiffres
non nulle est considérée ambiguë ; utiliser le point décimal pour lever ce doute.
Les noms de fournisseurs sont normalisés uniquement pour les espaces et le
suffixe certain `Ltd` (casse et point final). Un suffixe OCR tel que `LId`
ou `L1d` déclenche la correction ; il n'est jamais remplacé arbitrairement.

## Preuves documentaires et récupération à blâme minimal

[invoice_evidence.py](../src/invoice_evidence.py) conserve les tokens et leurs
positions (page, rectangle, confiance OCR), les cellules, chaînes brutes et
valeurs `Decimal`. PDF natif : mots PyMuPDF ; page scannée/image : Tesseract
`fra+eng`, agrandissement bicubique ×2, niveaux de gris, TSV, segmentation
`--psm 4`. Les limites de pixels sont vérifiées avant agrandissement.
Les en-têtes et coordonnées séparent description, quantité, prix net et montant ;
les pourcentages de remise sont exclus des candidats. La locale permet par
exemple `3 627,96 €`, `7,222.74`, `1 143,35` et les espaces fines.

[invoice_grounding.py](../src/invoice_grounding.py) lit les réponses modèle
directement en `Decimal`, sans passage par `float`. Le modèle Pydantic `Triplet`
valide les produits, arrondis au centime et tolérance 0,02. Les sommes et
HT + somme des TVA lues = TTC sont également calculés en `Decimal` ; plusieurs
taux sont conservés séparément, sans supposer que toute la facture est à 20 %.
Les prix unitaires du tableau sont nets : aucune remise n'est réappliquée.
La conversion en nombres JSON n'intervient qu'à l'export ; une conversion avec
perte est refusée. Les exposants et représentations de plus de 1000 chiffres
sont refusés explicitement avant les calculs.

Les candidats des passes modèle sont admissibles uniquement s'ils figurent dans
les cellules OCR du même champ. Ils ne peuvent donc pas ajouter une valeur
absente. Les solutions sont construites dans les cellules du document, dédoublonnées
par valeur et filtrées par les contraintes. Une réparation nécessite un triplet
unique **pour la ligne**, identique dans toutes les solutions globales admissibles.
Une ambiguïté sur une autre ligne n'efface donc pas une valeur corroborée.
Les propositions des deux passes ne sont jamais comptées comme deux preuves
indépendantes ; la seconde peut compléter un champ seulement si sa cellule OCR
et les contraintes le corroborent. La recherche est bornée (8 candidats/champ,
4096 combinaisons) : un dépassement ne prouve jamais l'unicité.

Les champs déjà ancrés qui sont invariants dans les solutions admissibles sont
conservés. Les trois cas « deux des trois valeurs ancrées » sont couverts :
si la quantité manque, prix et montant corroborés restent disponibles ; si le
montant manque, le produit peut vérifier la somme globale mais ne crée pas une
valeur à exporter. Seuls les champs sans preuve sont neutralisés ; aucune
division ne crée une quantité absente du document.
Un total HT lu reste disponible malgré des lignes douteuses. TTC − TVA peut
corroborer ce total, mais ne peut pas créer un nombre qui n'apparaît nulle part
dans le document (règle d'acceptation la plus stricte).

**Fidélité et validité arithmétique sont distinctes.** Une cellule documentaire
unique et lisible est conservée telle qu'imprimée, même si son calcul est faux.
La politique de conservation a été explicitement confirmée par l'utilisateur.
Les erreurs de facture sont exportées dans `validation.flags`, **hors** du
schéma standardisé et du scoring strict. `Invoice` reste inchangé : il peut
rejeter l'arithmétique d'une facture pourtant fidèlement transcrite.
Un écart mathématique ne suffit donc plus à neutraliser une lecture certaine.

Les descriptions sont comparées avec `SequenceMatcher`, casse/accents ignorés,
seuil 0,9. Une ligne clairement associée peut fournir sa description littérale,
jamais un texte issu d'un catalogue. Une description OCR de faible confiance
différente du modèle n'est pas substituée arbitrairement. Les cas non vérifiables
portent `description_non_verifiee`. Les champs numériques non ancrés ou ambigus
deviennent `null`. Une lecture OCR fausse mais cohérente et de forte confiance
reste une limite : l'ancrage prouve la présence dans l'OCR, pas la vérité physique.

L'application reçoit un `EvidenceText` (compatible `str`) portant ces preuves.
Une simple chaîne est interprétée comme du texte documentaire, **jamais** comme
une autorisation de conserver une sortie modèle sans ancrage. L'interface et
l'évaluateur utilisent désormais tous deux `run_grounded_inference` ; aucun
chemin historique ne conserve le premier JSON du modèle comme résultat final.
Les validateurs historiques restent accessibles pour tester le schéma strict,
mais ne sont pas un chemin de sortie d'inférence. Le chemin unique s'applique
quand le tableau est illisible (champs de ligne neutralisés, pas de validation
arithmétique seule présentée comme une vérification documentaire).
Si les preuves sont indisponibles, les statuts sont `non vérifié` et les valeurs
non recoupées sont `null`, sans génération GPU sur un texte vide. Le rapport
dit explicitement « Preuves OCR indisponibles : extraction non recoupée ».

Les formats de tableau non reconnus restent explicitement non vérifiés. En
particulier, un tableau affichant seulement des montants TTC par ligne ne
justifie pas de générer les HT absents du document : ces champs ne sont pas
inventés pour correspondre à la cible du dataset. Les rejeux des trois factures
à montants HT imprimés ne constituent pas une garantie d'exactitude universelle.

## Qualification OCR et stress indépendant — historique puis récupération

[ocr_quality.py](../src/ocr_quality.py) reste le contrat de qualification de
l'ancienne stratégie, et le traitement des preuves non recoupées hors consensus.
Le diagnostic figé de cette stratégie produit 416 `null` sur **612** champs :
160 FR, 128 UK, 192 US, 132 layouts ; les seuls champs image sont 480.
L'application effectue deux lectures TSV (`--psm 4` et `--psm 6`, mêmes pixels
agrandis ×2) et compare fournisseur, date, cellules, totaux, taxes et remises.
Ce sont des segmentations distinctes d'un même moteur, **pas deux capteurs
indépendants** : elles peuvent partager une erreur, comme `7.8 → 78`.
Dans cette stratégie initiale, une moyenne <85, un tableau absent, une ligne/cellule absente, des tokens de
tableau <60, un nombre de lignes différent ou une signature différente rendent
la lecture insuffisante. Une contradiction documentaire lue par OCR image
reste indiscernable d'un chiffre OCR erroné : elle demande confirmation et ne
devient pas automatiquement un drapeau d'erreur de la facture.

Dans ce mode initial prudent, tout le document douteux était neutralisé (statuts
`non vérifié`) et aucune allocation GPU supplémentaire ni anomalie documentaire
n'est affirmée. Le rapport affiche exactement
« Preuves OCR insuffisantes : extraction non recoupée ». Les PDF natifs et
textes fournis ne passent pas par ce filtre image ; leurs véritables erreurs
imprimées restent dans `validation.flags` avec valeurs fidèlement conservées.
Les PDF mixtes propagent les problèmes des pages OCR, sans réétiqueter leurs
tokens comme du texte natif certain.

Le parseur accepte des milliers explicitement groupés dans les deux conventions
(`1.234,56` ou `1,234.56`), même à locale inverse, mais pas `1.23,56`.
Les en-têtes `P.U.`, `Qté`, `Qty`, colonnes inversées, devises en préfixe/suffixe
et tableaux PDF sur deux pages sont couverts. Une remise monétaire **séparée
et imprimée** est incluse dans les contrôles de somme, sans créer un champ
de schéma et sans réappliquer une remise annoncée déjà incluse dans les prix.

Le jeu [tests/stress/](../tests/stress/) n'utilise pas le générateur du dataset.
Les références des trois exemples sont transcrites séparément ; les autres
documents sont rédigés à la main. Les images altérées sont produites en mémoire
avec seed fixe, le PDF dans un répertoire temporaire nettoyé. Les métriques
comptent les champs absents comme `null` à tort et les valeurs ajoutées fausses
comme valeurs fausses conservées ; aucune réussite partielle n'efface un crash.
Les drapeaux d'information (équivalences de noms) ne sont pas des anomalies
fantômes, les erreurs/avertissements non attendus le sont.

Le rapport de référence avant cette qualification a exposé 8 valeurs fausses,
6 anomalies fantômes et 293 `null` à tort sur 34 cas. Le filtrage réduit les
faux succès mais augmente volontairement les `null` ; le rapport final inclut
également une onzième variante de remise séparée. Ces mesures ne couvrent pas
tous les scans réels, ni les défauts cohérents partagés par les deux lectures.
La latence OCR et la forte neutralisation restent des risques produit
à examiner avant publication, même lorsque les critères de sûreté du stress passent.

### Récupération régionale mesurée (stratégie actuelle)

Le diagnostic causal `results/null-breakdown.md` est écrit **avant** tout
abaissement de seuil et attribue chaque refus global à une cause principale.
Les 35 mêmes cas sont séparés en 21 réglage / 14 validation, seed 62419.
Les données déjà observées lors d'une passe antérieure ne constituent pas un
vrai test externe aveugle ; la partition fige seulement les nouveaux réglages.

[ocr_preprocessing.py](../src/ocr_preprocessing.py) détecte le biais géométrique
par Hough (angles de −5 à +5°, sans connaissance de facture), redresse puis
applique CLAHE, médiane, ×3 ou recadrage tableau. Otsu et seuil local sont
implémentés pour les essais mais rejetés pour le chemin d'exécution. La limite
de 25 MP est vérifiée avant chaque image agrandie. ×3 est omis si cette limite
est dépassée, avec log explicite.

[ocr_recovery.py](../src/ocr_recovery.py) fige cinq familles choisies sur réglage,
sans importer aucun rapport de score, référence ou split. Deux segmentations
d'une seule famille ne sont pas deux votes. Deux familles concordantes sont
nécessaires pour une valeur ; les descriptions et métadonnées textuelles sont
neutralisées si une famille apporte une graphie contradictoire. Le premier
consensus gardait `Ciavier sans fil` sur JPEG FR : il a été rejeté, et le
débruitage a été retenu comme veto contradictoire malgré son score isolé médiocre.

La courbe de seuils 95/85/75/60/50/40 porte **uniquement sur le réglage** :
le plus bas seuil testé sans valeur fausse est 40. Ce n'est pas une garantie
statistique ni une preuve que 39 serait dangereux ; aucune recherche n'est
faite après les résultats de validation. Le seuil moyen global n'est pas
simplement baissé : le chemin consensus exige plusieurs prétraitements et
neutralise les champs discordants, même s'ils satisfont les calculs.

Le rapport permet d'afficher une proposition modèle non fiable séparément :
« Valeur non recoupée : à vérifier manuellement », dans une section explicitement
hors du JSON. Si plus de la moitié des champs sont douteux, un avertissement
en tête demande un scan 300 dpi, droit et net. La proposition modèle reste
absente du schéma standardisé. Le diagnostic n'invente pas une proposition
si aucune génération n'a eu lieu.

Les trois profils d'erreurs supplémentaires par cas sont des reconstructions
des valeurs Qwen signalées par l'utilisateur, plus des erreurs permutées et
arithmétiquement cohérentes. Ils ne sont pas qualifiés de captures réelles locales.
Le notebook Colab et `--stress` du script réel servent à réaliser ce test ultérieur,
sans aucun téléchargement dans l'environnement actuel.

### Politique unique et latence (passe suivante)

`src/ocr_config.py` contient l'unique configuration gelée et versionnée :
confiance du consensus 40, cinq familles, deux segmentations, critères plus
stricts du chemin rapide et budget total 45 s. Les alias historiques lisent
cet objet ; tests, empreinte JSON et version permettent de tracer un changement.
Les scripts d'expérience peuvent explorer des seuils explicitement, mais
l'application n'importe jamais leurs résultats comme paramètres.

Le redressement géométrique n'est calculé qu'une fois par image. Le chemin
rapide compare **gris redressé PSM 4 et CLAHE PSM 6** : même image, mais deux
familles distinctes, sans prétendre à une indépendance statistique des moteurs.
Moyenne minimale 85 et tokens de tableau 60 sont contrôlés sur les deux lectures ;
les seuils du consensus ne sont pas relâchés. Si le tableau, les valeurs,
les noms/dates ou les contrôles ne concordent pas, les lectures sont réutilisées
dans l'escalade complète au lieu d'être recalculées.

Le délai est partagé entre toutes les pages du document. Avant/après chaque
prétraitement, page et appel, le temps restant est contrôlé ; Tesseract peut
être interrompu à ce temps restant. Les appels CPU natifs ne sont pas préemptés
au milieu d'une fonction : la mesure publiée inclut la surcharge d'arrêt/rendu,
pas une garantie stricte au milliseconde. Après épuisement, l'état est
« preuves OCR insuffisantes », sans conversion d'une lecture partielle en valeur sûre.
La fusion des pages conserve les **preuves sélectionnées**, sans reparsing
des tokens OCR bruts qui recréerait une quantité rejetée.

Le script de scans réels est OCR/contraintes seulement, sans modèle ni téléchargement.
Avec vérité terrain il mesure les null à tort et valeurs fausses ; sans référence,
il ne revendique aucun taux d'exactitude. Protocole : `docs/REAL_SCANS.md`.
Les mesures avant/après et le résultat de la comparaison figée sont publiés
dans `results/LATENCY.md` ; les dépendances locales ne prouvent pas un déploiement
HF complet (Pillow local 12.3 contre manifeste <11 notamment).

## Seconde passe, ZeroGPU et rapport

Une seule seconde passe est autorisée ; elle ne contient que les lignes en échec
et leur OCR, avec des index explicites. Les autres lignes ne sont pas renvoyées
au modèle. Les totaux documentaires sont ajoutés uniquement si le contrôle global
échoue. Le modèle ne peut pas réécrire une ligne non demandée. Si les preuves
résolvent déjà tous les champs, cette seconde allocation GPU est évitée.

`@spaces.GPU(duration=gpu_duration)` demande 15–60 s pour la première passe et
8–30 s pour la correction (512 nouveaux tokens maximum), selon la taille du
prompt. Il ne s'agit pas d'une interrogation du solde de quota : le service
peut toujours refuser. Le type, le message complet et la trace sont journalisés ;
un refus de quota affiche « Quota ZeroGPU épuisé, réessayer dans … » avec le délai
fourni par HF, ou explicitement non communiqué. L'OCR et les contraintes
fonctionnent sans GPU, y compris si la première inférence est indisponible.
Une réponse tronquée/malformée n'est jamais utilisée comme preuve.

Le rapport visible donne d'abord les anomalies **de la facture**, puis les
contrôles réussis et un unique résumé des lectures récupérées/neutralisées.
[invoice_validation.py](../src/invoice_validation.py) ne reçoit **aucune
prédiction du modèle**. Les anomalies incluent localisation, valeur imprimée,
valeur attendue, écart et justification ; un écart de sous-total découlant
d'une ligne fausse est présenté comme conséquence et non comme erreur de lecture.

L'accordéon Gradio « Détails techniques », fermé par défaut, conserve les
statuts par champ (`valide`, `récupéré (corroboré par OCR)`,
`neutralisé (non fiable)`, `non vérifié`), messages techniques et
`validation.flags`. Les traces complètes restent dans les logs. Le rapport
visible n'expose ni jargon Pydantic ni traceback. Les erreurs de lecture du
modèle n'y sont jamais présentées comme des anomalies de la facture.
Un changement de fichier efface également les détails de l'ancienne facture.
Le pied de page affiche le hash et « OCR : disponible/indisponible » ;
la disponibilité de Tesseract `fra+eng` est journalisée au démarrage.

Les contrôles documentaires couvrent les produits de ligne, somme HT, HT + TVA,
TVA par taux ou base imprimée, doublons, variantes de noms fournisseur/client,
dates et conditions de paiement, pays/devise/vocabulaire/taux et identifiants.
Les formats Ltd/Limited ou SAS/S.A.S., casse et accents équivalents sont des
informations, pas des erreurs. Les tests d'identifiants ont été calibrés sur
les 60 références **avant activation** :

| Contrôle | Références contrôlées | Échecs | Sévérité d'un échec |
| --- | ---: | ---: | --- |
| SIRET / Luhn | 20 | 18 | information |
| TVA FR / clé et SIREN | 20 | 0 | erreur |
| IBAN / modulo 97 | 20 | 0 | erreur |
| VAT UK / format | 20 | 0 | erreur |
| EIN US / format | 20 | 0 | erreur |

Cette calibration ne certifie pas fiscalement les identifiants synthétiques.
Un test vérifie les comptes et l'absence de drapeau de niveau erreur sur les
60 textes de référence. Une lecture OCR à faible confiance ne justifie pas
de qualifier une erreur de la facture. Une OCR fausse à haute confiance
reste néanmoins une limite explicitement non résolue par les seules contraintes.

## Évaluation du modèle brut et correction

Le mode `--base-only` de `src/evaluate_finetuned.py` charge uniquement
`unsloth/Qwen2.5-3B-Instruct` avec une quantification NF4 sur CUDA. Il n'importe
pas PEFT pour ce chargement et rejette un modèle auquel des adaptateurs
auraient été attachés automatiquement. Le tokenizer provient du modèle de
base ; le modèle LoRA garde son tokenizer sauvegardé. Le contexte, le
décodage sans échantillonnage et le scoring sont communs aux deux modes.

`--self-correct` est facultatif pour préserver les évaluations à une passe.
L'export `payload` de l'évaluateur est ancré même sans `--self-correct` ; ce
drapeau n'autorise que l'inférence supplémentaire, pas le contournement des
contrôles documentaires. Il utilise le même ancrage et la même
correction ciblée que l'application, sans le dialogue few-shot supplémentaire
de l'interface. La validation arithmétique seule ne constitue jamais un repli
pour produire le JSON final. Aucun résultat cible n'est transmis au modèle
ou au moteur d'ancrage.

Le rapport conserve les réponses brutes de chaque passe, les erreurs et
avertissements, les compteurs de tokens cumulés, ainsi que
`initial_json_validity_rate`, `initial_exact_match_rate` et `correction_rate`.
Le timing comprend la correction lorsqu'elle est activée. Une seconde passe
tronquée ou en erreur est signalée explicitement. Les prédictions contiennent
un `payload` sécurisé distinct de `raw`, et `warnings` conserve l'historique ;
`arithmetic_warnings` décrit les anomalies restantes de la réponse retenue.
Les écarts arithmétiques restent des échecs de schéma dans les métriques :
le scoring porte sur la réponse brute, jamais sur le modèle nullable.
`truncation_rate` décrit la réponse retenue,
`truncated_attempt_rate` signale aussi les troncatures des tentatives écartées.

[evaluate_grounding.py](../src/evaluate_grounding.py) mesure séparément l'export
final par champ : exactitude, taux de `null` à tort, taux de valeur fausse conservée,
avec pour dénominateur tous les champs attendus et les éventuels champs ajoutés.
Il accepte des JSONL `id`, `ocr_text`, `locale`, `model_responses`, `expected`,
ou `ocr_tsv` à la place de `ocr_text` pour conserver les positions des images.
Le mode `--replays` utilise explicitement les trois erreurs synthétiques de
régression et le texte du dataset ; il ne prétend pas appeler le vrai modèle.
Les tests incluent aussi le véritable OCR des images avec réponses LLM simulées.
Le test `slow` est exclu par défaut et nécessite des poids locaux explicites ;
il couvre les trois images et exige deux vraies générations identiques sur
chaque image, pas seulement une récupération OCR de repli.

### Diagnostic reproductible sans simulation

`python src/diagnose_pipeline.py --require-real-model` exécute réellement
les appels de l'application sur les trois images et affiche HEAD, SHA-256
des sources (fichiers locaux non commités inclus), disponibilité OCR/nombre
de lignes, tentatives et générations terminées, JSON bruts et final, rapport
et erreurs techniques. Il échoue explicitement si le modèle n'a pas réellement
généré sur les trois images ; il n'injecte aucune sortie simulée.

Le diagnostic initial de cette révision locale a trouvé les adaptateurs LoRA
(115 Mo) mais pas PyTorch, Transformers, PEFT ni GPU NVIDIA. Les trois appels
réels ont échoué sur `ModuleNotFoundError: No module named 'torch'` ; **aucun
JSON brut réel n'a été obtenu**. L'OCR réel a lu 20/20/22 lignes (FR/UK/US),
soit 4/3/5 articles. Les sorties finales correctes de ce diagnostic sont des
replis OCR, pas des succès du LLM. Les erreurs non ancrées ont été testées
séparément par rejeu ; le déterminisme brut du vrai modèle reste à vérifier
sur un environnement d'inférence fonctionnel.

Le code d'ancrage local était non suivi/non commité, absent du HEAD
`137b643`. Cela explique la possibilité d'un ancien chemin sur un déploiement,
mais ne prouve pas la révision réellement servie par le Space : son hash ou
ses logs sont nécessaires. Aucun commit ou push n'est effectué automatiquement.

Les résultats mesurés ne sont écrits qu'après évaluation des entrées demandées.
Un lancement bloqué avant génération ne produit pas de métriques et ne doit
pas être remplacé par les résultats des modèles simulés des tests logiciels.

## Organisation détaillée des artefacts

Inventaire de conception déplacé depuis le README ; les poids et archives
peuvent être locaux et ignorés par Git. Les noms de run sont paramétrés ici.

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
│   ├── loss_curve.png
│   ├── qwen25-3b-international-v3-<timestamp>.zip
│   └── qwen25-3b-international-v3-<timestamp>/
│       ├── lora_adapter/              # poids et tokenizer
│       ├── checkpoints/
│       ├── training_manifest.json
│       ├── train_log.json
│       └── loss_curve.png
├── requirements.txt
└── requirements-inference.txt
```

## Références de la stack

Liens conservés depuis les badges retirés pour alléger le README :

- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Hugging Face](https://huggingface.co/)
- [Google Colab](https://colab.research.google.com/)
- [ReportLab](https://docs.reportlab.com/)
- [GGUF / llama.cpp](https://github.com/ggml-org/llama.cpp)
