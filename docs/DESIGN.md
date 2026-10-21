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
Le prompt d'évaluation, identique à celui de l'entraînement, reste inchangé.

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
