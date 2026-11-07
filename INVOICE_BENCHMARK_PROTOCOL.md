# Invoice Extraction Benchmark Protocol

## Status and scope

This is a local, offline scoring protocol. The evaluator reads a manifest and
recorded JSON/JSONL outputs; it does not import the application, call OCR or
model code, access the network, or load model weights.

The inventory is in `INVOICE_BENCHMARK_MANIFEST.json`. Its `synthetic_test` set
is a regression cohort, not evidence of real-world generalization. There is no
currently proven, untouched genuine-invoice holdout.

## Eligibility and ground truth

A document may support an independent-generalization claim only when all of the
following are documented before inference:

1. It is a genuine issued invoice, not a template, mock-up, crop, or altered
   fixture.
2. Its source bytes and SHA-256 are frozen, and the document was not used for
   training, fine-tuning, prompt development, debugging, regression testing,
   or previous model evaluation.
3. Training-data membership and prior use can be checked; filename novelty is
   not evidence of being unseen.
4. Two annotators independently record printed values and evidence locations.
   Disagreements are explicitly adjudicated before any model result is viewed.
5. Printed values, absent values, derived-only values, and ambiguities are
   distinguished. An unsupported calculation is never treated as a printed
   amount.

Each scoreable `ground_truth` uses explicit field states:

- `present`: an adjudicated printed value is supplied.
- `absent`: the field is explicitly absent and the expected output is `null`.
- `derived_only`: only a calculated value is available; the expected printed
  value is `null`.
- `ambiguous`: multiple documentary readings remain; the field is excluded
  until adjudicated.
- `unscorable`: no reliable label is available; the field is excluded.

The same states apply to each line-item field. If a row description is
ambiguous or unavailable, the evaluator does not positionally align its other
fields.

## Output contract

Each successful output is a JSON object with exactly these top-level keys:

```json
{
  "supplier_name": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "total_ht": "finite number or null",
  "total_ttc": "finite number or null",
  "line_items": [
    {
      "description": "string or null",
      "quantity": "finite number or null",
      "unit_price": "finite number or null",
      "total_ht": "finite number or null"
    }
  ]
}
```

The quoted type descriptions above are explanatory, not JSON string values.
Numeric fields must be JSON numbers (not booleans or currency-formatted
strings); `line_items` must be an array, and each row must have exactly the
four listed keys. Missing/extra keys, malformed JSON, non-finite numbers, and
invalid dates are schema failures. Currency is inventory metadata, not an
extracted field in this output schema.

## Normalization and row alignment

- Dates compare as ISO calendar dates. Non-ISO output is a schema failure.
- Monetary values compare after decimal parsing and cent rounding with
  `ROUND_HALF_UP`; currencies are never converted.
- Quantities compare as decimals within `0.000001`.
- Strict text comparison applies Unicode NFC and collapses/strips whitespace;
  it preserves case, accents, and punctuation.
- Substantive text comparison additionally folds accents and case. This is
  reported separately and never replaces strict accuracy.
- Row matching normalizes accents/case/whitespace and ignores punctuation only
  for candidate matching. Exact normalized descriptions are preferred;
  fuzzy matches require `SequenceMatcher >= 0.88` and unique best matches.
  There is no positional fallback. Tied or weak matches remain unaligned.
- Line-item count and order are scored separately from the aligned field
  values. Extra-row non-null values count as unsupported acceptances.

## Metrics

Metrics are calculated over adjudicated `present`, `absent`, and
`derived_only` fields. Ambiguous/unscorable fields are reported but excluded.
Invalid or missing output counts as a false null for expected-present fields;
it does not earn correct-null credit.

Let:

- `TP` = correct non-null value for a `present` field.
- `FN` = null/missing output for a `present` field.
- `TN` = null output for `absent` or `derived_only`.
- `FP` = incorrect non-null output for a `present`, `absent`, or
  `derived_only` field, including unsupported/extra-row values.
- `N` = all scorable fields, including extra-row non-null values.

The evaluator reports:

- **Accepted-value precision** = `TP / all scorable non-null outputs`. This
  denominator includes wrong values, derived-only claims, and extra-row claims.
- **Field recall** = `TP / expected-present fields`.
- **Exact-field accuracy** = `(TP + TN) / N`.
- **Correct-null rate** = `TN / (TN + FN)` over output nulls on scorable fields.
- **False-null rate** = `FN / expected-present fields`.
- **Incorrect non-null acceptance rate** = `FP / all scorable non-null outputs`.
- **Substantive accuracy** uses the separately documented accent/case-folded
  text comparison; the report also exposes substantive-only matches.
- **Line-item alignment accuracy** = matched rows divided by the greater of
  gold and predicted row counts.
- **Line-item count accuracy** and **order accuracy** are reported separately.
- **Complete-invoice accuracy** is computed only for fully adjudicated
  documents; every field, row count, and row order must be exact.
- **Document coverage** includes the mean per-document fraction of expected
  present fields correctly extracted and the fraction of documents with at
  least one correct non-null value.
- **Model-inference success rate** = successful runs where the model was
  actually used divided by documents selected. It is not applicable to the
  OCR-only system. OCR-only fallback frequency is recorded separately.
- OCR, generation, and end-to-end latency use median and nearest-rank p95;
  unavailable values are not imputed.

Returning null for every field therefore yields zero field recall, a 100% false
null rate for present fields, no complete invoices, and no document coverage.
Correct nulls can help only for explicitly absent or derived-only fields.

## Comparison systems and fairness

The planned systems are:

1. `ocr_only`: OCR, evidence extraction, and grounding without Qwen.
2. `base_qwen`: base Qwen2.5-3B-Instruct without the LoRA adapter.
3. `qwen_lora`: the same base model with the fine-tuned adapter.
4. `production_grounded`: the deployed OCR + Qwen+LoRA + documentary
   reconciliation pipeline.

No base-Qwen-only capture currently exists. A base-versus-LoRA comparison is
marked `paired` only when both outputs exist for each document, actually used
the model, and match on input SHA-256, OCR-evidence SHA-256, prompt hash,
generation-settings hash, software revision, and recorded/live execution mode.
The base model commit must be the same immutable 40-character revision in both
runs; the LoRA adapter must also have an immutable 40-character revision.
Simulated outputs can validate the harness but can never establish model
quality or LoRA contribution.

Model-only output and the final grounded output are reported separately.
Historical captures from different revisions, missing evidence hashes,
unknown revisions, and simulated proposals are descriptive only, not fair
controlled comparisons.

## Response JSONL and command

Use one JSON object per document and model variant. Required fields are
`document_id`, `model_variant`, `software_revision`, `execution_mode`
(`recorded`, `simulated`, or `live`), `inference_status` (`succeeded`,
`failed`, or `not_run`), booleans `model_used` and `ocr_fallback`, the
manifest-matching `input_sha256`, and `output`. Supply `null` output when no
valid structured result was captured. For paired model comparison, also
record `model_revision`, `adapter_revision` where applicable,
`ocr_evidence_sha256`, `prompt_sha256`, `generation_settings_sha256`, and
`latency` with `ocr_seconds`, `generation_seconds`, and
`end_to_end_seconds` when known.

Example on Windows:

```powershell
.venv\Scripts\python.exe scripts\evaluate_invoice_benchmark.py `
  --manifest INVOICE_BENCHMARK_MANIFEST.json `
  --responses path\to\recorded-responses.jsonl `
  --set synthetic_test `
  --output path\to\benchmark-results.json `
  --summary path\to\benchmark-summary.txt
```

Before scoring, the harness checks every selected source file SHA-256, rejects
duplicate documents and duplicate responses, validates the JSON schema, and
records missing responses. It performs no inference. Keep real invoice files
and raw captures in approved, access-controlled storage; do not put sensitive
customer data in a public repository or report.
