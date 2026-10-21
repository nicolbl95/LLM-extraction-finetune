# llm-extraction-finetune

## Overview

Fine-tuning a small LLM to extract structured JSON from invoices. The target schema is defined with Pydantic v2 in [data/schema.py](data/schema.py) (supplier name, invoice date, totals HT/TTC, and line items).

## Project structure

- `data/`: datasets and the extraction schema (`schema.py`)
- `notebooks/`: exploration and training notebooks
- `src/`: training and inference code
- `results/`: evaluation outputs and metrics
- `app/`: application entry point

## Setup

```bash
pip install -r requirements.txt
```

## Results comparison

| Model | Supplier name | Invoice date | Total HT | Total TTC | Line items F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Base model (zero-shot) | TBD | TBD | TBD | TBD | TBD | TBD |
| Base model (few-shot) | TBD | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned model | TBD | TBD | TBD | TBD | TBD | TBD |