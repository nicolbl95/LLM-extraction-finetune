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

## Évaluation du modèle brut et correction

Le mode `--base-only` de `src/evaluate_finetuned.py` charge uniquement
`unsloth/Qwen2.5-3B-Instruct` avec une quantification NF4 sur CUDA. Il n'importe
pas PEFT pour ce chargement et rejette un modèle auquel des adaptateurs
auraient été attachés automatiquement. Le tokenizer provient du modèle de
base ; le modèle LoRA garde son tokenizer sauvegardé. Le contexte, le
décodage sans échantillonnage et le scoring sont communs aux deux modes.

`--self-correct` est facultatif pour préserver les évaluations à une passe.
Il utilise les mêmes fonctions de validation et de construction du message
correctif que l'application, ainsi que les nouveaux garde-fous du prompt partagé,
sans le dialogue few-shot supplémentaire de l'interface. Une seule correction est autorisée après un
échec JSON/Pydantic ; aucun résultat cible n'est transmis au modèle.
Une incohérence de ligne ou de somme déclenche cette seconde passe lorsque
`--self-correct` est activé. Le message transmet l'erreur Pydantic exacte et,
pour une ligne incohérente, la quantité, le prix unitaire, le produit calculé,
le montant HT extrait et le sous-total global. Le diagnostic nomme également
la description de l'article et l'écart absolu ; une erreur de somme affiche
la somme calculée, le total global attendu et leur écart, avec la tolérance.
Le prompt correctif commence par « La précédente extraction a échoué à la
validation du schéma Pydantic pour les raisons suivantes : », suivi de
l'erreur intégrale, sans troncature. Il se termine par une demande de relecture
attentive des décimales, quantités et totaux de lignes, puis d'un JSON
rigoureusement valide. Il demande une relecture de la
ligne source, sans inventer une quantité par division ni confondre le total
d'une ligne avec celui de la facture.

Le rapport conserve les réponses brutes de chaque passe, les erreurs et
avertissements, les compteurs de tokens cumulés, ainsi que
`initial_json_validity_rate`, `initial_exact_match_rate` et `correction_rate`.
Le timing comprend la correction lorsqu'elle est activée. Une seconde passe
tronquée ou en erreur est signalée explicitement ; le premier JSON
structurellement valide est conservé s'il existe. Les écarts arithmétiques
restent des échecs de schéma dans les métriques, même si l'interface permet
leur affichage avec avertissement. `truncation_rate` décrit la réponse retenue,
`truncated_attempt_rate` signale aussi les troncatures des tentatives écartées.

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
