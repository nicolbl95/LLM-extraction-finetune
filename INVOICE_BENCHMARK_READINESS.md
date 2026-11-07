# Invoice Benchmark Readiness

## Verdict

- **Offline scoring harness:** ready for recorded responses.
- **Independent generalization benchmark:** not ready; zero documents have proven untouched, genuine-invoice status.
- **First public inference:** **NOT READY** pending separate review/deployment of the two locally fixed callback error paths and collection of an approved test batch. No Space was changed in this task.

No hosted inference, ZeroGPU allocation, model download, deployment, remote write, or OCR run was performed for benchmark preparation. The full offline pytest run did invoke existing local OCR tests.

## Inventory and eligibility

`INVOICE_BENCHMARK_MANIFEST.json` inventories 637 image assets. These are asset paths, not 637 independent invoices; variants and copies are linked where known.

| Inventory class | Assets | Ground truth / use |
|---|---:|---|
| Synthetic invoices | 600 | Complete deterministic labels; 480 train, 60 validation/development, 60 test/regression |
| Public app examples | 3 | Byte-identical copies of three synthetic test documents; not independent |
| Blank example template | 1 | Not an invoice |
| Local test fixtures | 8 | Previously used regression variants/layout fixture |
| Diagnostics assets | 19 | Historical scans/variants, templates, development examples, and one partial row crop |
| Full historical scans | 6 | Five manual records plus Pure Cleaning; used in development/regression |

The three synthetic split IDs are disjoint, each split contains all nine template families, and the 600 synthetic invoices are evenly distributed across `fr-FR`, `en-GB`, and `en-US`. Suppliers and template families recur across splits. The adapter training manifest and complete prior-use log are unavailable, so the 60-document synthetic test split is **contamination-unknown**, not a proven held-out test set.

There are ten historically evaluated web-template/mock-up documents: four in the frozen four-template cohort and six in the new-layout cohort. Three of the four were marked independent within that original cohort, while one was regression-only; none is proven to be a fresh, genuine business invoice today. The six-layout set contains five templates and one authenticity-unconfirmed document; Service and Travel share a layout family. Two additional development examples lack authoritative standalone truth. No genuinely untouched genuine-invoice documents are present.

Ground truth is reliable for the synthetic generation targets. Historical real-scan and template annotations are useful for regression analysis but have prior use, incomplete field semantics, single-annotator or authenticity limitations. `null` values in those legacy annotations must not be reinterpreted as confirmed absence without adjudication. Currency is populated only where an existing annotation explicitly records it.

### Duplicate analysis

SHA-256 confirmed three byte-identical example/test pairs:

- `app/examples/france.png` / `data/invoices/invoice-0406.png`
- `app/examples/uk.png` / `data/invoices/invoice-0110.png`
- `app/examples/us.png` / `data/invoices/invoice-0435.png`

A 64-bit DCT perceptual-hash screen found 2,115 non-identical pairs at Hamming distance ≤6, including 673 synthetic cross-split candidates and 398 cross-template candidates. This heuristic is too coarse to prove duplicate documents; those pairs are not labeled duplicates. The manifest records the method and caveat.

All 637 asset hashes, all linked annotation-source hashes, all 600 embedded synthetic annotation hashes, and all document IDs were independently checked after manifest generation.

## Recorded outputs and fair comparison

- **Base Qwen-only captures:** none found.
- Six recorded Qwen+LoRA outputs report model use at old revision `907db76a1ae24d4d95552d7d06c1289eee11183a`. Two additional four-template captures explicitly report model use at that revision; two others have unknown model-use/revision status.
- The legacy six-document score reports raw-model 73/104 strict-correct fields, 27 wrong non-null outputs, and 4 missing fields (73% precision among 100 non-null outputs; 70.19% strict accuracy). Grounding retained 8 supported non-null values, rejected 0 unsupported non-null values, correctly abstained on 3 fields, and missed 93 fields (8% printed-field coverage). This is descriptive legacy scoring on templates, not a controlled comparison or independent generalization result.
- The saved Mirada capture reports model use, 5/16 strict fields, and three documentary rows; its exact revision is not recorded.
- The nine-invoice matrix and 38-case/114-profile stress results use synthetic proposals or simulated faults, not genuine Qwen generations.

No existing results fairly quantify the contribution of LoRA over base Qwen. Historical software/model revisions, missing evidence/prompt hashes, mutable model references, and incomplete capture metadata prevent a controlled A/B claim. Latency and success rates for the current public revision also remain unmeasured.

## Harness and validation

The evaluator is `scripts/evaluate_invoice_benchmark.py`; metric and CLI tests are in `tests/test_evaluate_invoice_benchmark.py`. It validates the strict JSON shape, selected image hashes, ground-truth source and annotation hashes, duplicate documents/responses, and missing outputs. It reports accepted precision, recall, exact and substantive accuracy, correct/false null rates, line-item matching/count/order, complete-invoice accuracy, coverage, inference/fallback rates, and latency. It does not import the application or make network/model calls.

Run it with a recorded JSONL file:

```powershell
.venv\Scripts\python.exe scripts\evaluate_invoice_benchmark.py `
  --manifest INVOICE_BENCHMARK_MANIFEST.json `
  --responses path\to\recorded-responses.jsonl `
  --set synthetic_test `
  --output path\to\benchmark-results.json `
  --summary path\to\benchmark-summary.txt
```

Validation results:

| Check | Result |
|---|---|
| New metric and CLI tests | **22 passed** |
| Benchmark + app tests excluding the French-OCR-dependent cache replay | **70 passed, 2 skipped, 1 deselected, 25 subtests passed** |
| Manifest smoke evaluation | 60 source images and their ground-truth hashes verified; zero responses correctly produce zero recall/coverage and zero complete invoices |
| Full offline `pytest -q` | **699 passed, 2 failed, 41 errors, 11 skipped, 3 deselected, 113 passing subtests** (run before the final four evaluator cases; those cases pass in the final 22-test run) |

The full-suite errors are the stress-test gates requiring Tesseract `fra+eng`; the local Windows Tesseract installation exposes only `eng` and `osd`. The two OCR-dependent failures were the France example cache replay (TTC was null) and the Joanna Binet layout reference (`FR-001` read as `FR-0O1`). No language data was installed, no OCR policy was changed, and these results do not establish the hosted Linux OCR status. The three real-model tests remain deselected by `pytest.ini`; no model weights were loaded.

For comparison, PRODUCT-REVIEW-50 previously recorded 729 offline tests passed, 120 passing subtests, and three slow-model tests deselected. That is historical evidence, not a substitute for the current full-suite result.

## Local safety finding

Fault injection reproduced two pre-existing user-facing error-path defects in the local app source:

1. A malformed image preview included the server-side temporary file path in its message.
2. A JSON-download write error escaped the streaming Gradio callback with the raw `OSError`.

The local changes in `app/main.py` redact those messages, log details server-side, preserve the displayed result, and show a safe download warning. Focused tests pass. The previously verified public baseline was `ac2913303c7f6401c0a27688027bacdc0ae69a61`; the current public revision was not re-read or modified in this benchmark task, and the local fix has not been deployed. Treat the public callback issue as a blocker until separately reviewed and addressed through an explicitly authorized release.

## Missing evidence and collection plan

No current file is suitable as the primary independent batch. The next batch is **12 new, permissioned or rights-cleared genuine invoices**—not selected from the current assets—with double annotation before any inference. Use these collection slots:

| Slot IDs | Locale | Required document profile |
|---|---|---|
| FR-01–FR-04 | France / French | Clear standard VAT invoice; multi-row invoice with wrapped descriptions; mixed-rate VAT/discount with printed totals; low-quality or partially missing-field scan |
| UK-01–UK-04 | United Kingdom / English | Clear VAT invoice; decimal quantities and several rows; multi-page or shipping/tax layout; low-quality or ambiguous-field scan |
| US-01–US-04 | United States / English | Clear sales-tax invoice; multi-row/wrapped descriptions; shipping or discount with printed totals; low-quality or partially missing-field scan |

Require at least three supplier/layout families per locale and no supplier or template reused from the synthetic corpus or historical cases. Record country, language, and currency only when explicitly supported by the document. Freeze image hashes and both independent annotations before inference; adjudicate disagreements without viewing model output. Do not upload confidential customer invoices to a public repository or Space.

When ZeroGPU quota and separate authorization are available, record one JSONL entry per document/system with `document_id`, exact input SHA-256, system variant, execution mode (`live` or `recorded`), whether the model actually ran, inference status, software revision, immutable base-model and adapter revisions, OCR-evidence hash, prompt/settings hashes, OCR/generation/end-to-end latency, and strict output. Save raw model responses only in approved access-controlled storage. Run all four systems on the same frozen bytes and OCR evidence; keep base-model, LoRA-only, and final grounded outputs separate. Simulated responses may test the harness but must never be reported as model results.

## Next steps

1. Review the two local error-path changes; no deployment is authorized by this report.
2. Restore/verify local `fra+eng` through an approved environment change, then rerun the full offline suite and stress gate. Do not install language data without approval.
3. Collect and double-annotate the 12-document batch above; prove non-contamination and freeze its manifest.
4. Pin and record immutable model/adapter revisions and populate response JSONL for all comparison systems.
5. Run the offline evaluator and report the LoRA contribution only if the paired-comparison checks pass.
