# PR110 — Owner guide: collecting and annotating twelve independent invoices

This guide is the whole workflow. It assumes no knowledge of the pipeline internals and needs no code edits:
you drop files into a folder, answer questions in a terminal, and run one command that says whether the
benchmark may be measured.

**Target: 12 documents the system has never seen.** Each needs (a) the invoice file and (b) a sealed reference
JSON that says what the invoice actually prints. Nine rounds of this campaign were blocked on exactly this
missing; the tooling is finished, the data is not.

Everything you supply stays on this machine. `benchmark_prep/invoices/`, `benchmark_prep/references/` and
`results/product-review110/` are in `.gitignore`, so no invoice file, no printed text and no supplier name can
be committed. Nothing is uploaded anywhere: the scripts never open a network connection.

---

## 0. Preparing a new machine

A `git clone` delivers the whole toolchain: the six `scripts/pr110_*.py` entry points and the ten research
harnaises they import, `benchmark_prep/` (this guide, the annotation protocol, the two schemas, the two
reference templates, `annotation_tools.py` and its tests) and the frozen PR103 source manifest that
`pr110_ready.py` defaults to. It does **not** deliver the five inputs below, and never will — four of them are
private data, one is a machine-level dependency.

| What | Where it comes from | In Git? | Needed to |
|---|---|---|---|
| Invoice files you collect | you copy them in | no — `benchmark_prep/invoices/` is ignored | run the intake and the gate |
| Sealed reference JSONs | `pr110_annotate.py seal` writes them | no — `benchmark_prep/references/` is ignored | pass the annotation gate |
| Development truth (`data/invoices/`, 600 rendered pages) | restore from the owner's machine | no — ignored since PR119 | run `pytest tests` at full strength; 3 tests deselect without it |
| PR103's frozen prior-use inventory (`results/product-review103/pr103-dataset-manifest.json`, 640 hashes) | restore from the owner's machine — it cannot be republished | no — it names real parties, measured in PR147 | decide `previous_use`, which both the eligibility screen and `pr110_ready.py` consult; 11 of the 51 tests below |
| Tesseract with `fra.traineddata` | system install + `TESSDATA_PREFIX` | no — machine-level | any OCR run; without `fra` the reader degrades to English silently (PR81) |

```
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt pytest
# pytest is installed separately on purpose: requirements.txt is mounted as-is by Hugging Face to
# build the public demo image, so test-only dependencies must not be added to it.
export TESSDATA_PREFIX="/chemin/vers/tessdata"    # doit contenir fra.traineddata
mkdir -p benchmark_prep/invoices benchmark_prep/references
```

Then check the toolchain before touching any invoice — `pr110_status.py` reads no invoice file and writes only
`results/product-review110/pr110-phase1-state.json`:

```
.venv/Scripts/python.exe -m pytest tests/test_pr110_benchmark_prep.py \
    benchmark_prep/tests/test_annotation_tools.py tests/test_pr103_reference_validator.py -q
.venv/Scripts/python.exe scripts/pr110_status.py
```

On a clone that has none of the private inputs above, the expected result is measured, not guessed: the three
test modules give **40 passed, 11 failed**, every failure being
`FileNotFoundError: results/product-review103/pr103-dataset-manifest.json` — the prior-use authority the
eligibility screen refuses to invent. Copying that one file from the owner's machine turns all 51 green; PR147
measured 14 unqualified real-party names inside it, which is why it stays out of Git. `pr110_status.py` itself
does run on an empty clone and prints `intake: 0 invoice file(s), 0 reference(s) | target 12 | shortage 12`
with `verdict CHECK REQUIRED`. That is the tool working, not a broken install. The real gate is section 7
(`pr110_ready.py`), and it stays BLOCKED until twelve annotated documents exist.

## 1. Where to put the invoices

```
benchmark_prep/invoices/     <- the invoice files, one file per invoice
benchmark_prep/references/   <- the annotated references (created for you by the CLI)
```

If the folders are missing:

```
mkdir -p benchmark_prep/invoices benchmark_prep/references
```

Name files however you like — a name proves nothing and nothing in the workflow reads it for eligibility. The
reference is keyed by the **document id** you choose in step 3, and it records the invoice's SHA-256, so the
pairing survives a rename.

One invoice per file. If a PDF contains two invoices on two pages, split it first: the intake refuses a file
whose page map does not add up, and the screen reports it as *pending_review*, not eligible.

## 2. Accepted formats

| Accepted | Notes |
|---|---|
| `.pdf` | multipage is fine; declare which pages the invoice occupies |
| `.png` | screenshots of a PDF viewer are accepted and are classified as `scanned_image`-quality input |
| `.jpg` / `.jpeg` | photographs of a printed invoice are accepted, declare `photograph` |
| `.tif` / `.tiff` / `.webp` / `.bmp` | the screen opens these too, but no round of this campaign has ever been measured on them, so a PDF/PNG/JPEG copy is the safer submission |

Anything the reader cannot open (`.docx`, `.eml`, `.zip`) is reported as not readable and the document is
**excluded**. Do not convert something yourself to get it in — a converted file loses the acquisition quality
you would otherwise declare, and the screen would then describe a document you no longer have. Ask instead.

A native PDF (real text layer) and a scanned PDF are different test cases. Both are wanted; declare which one
it is in the guide, because the results are reported separately.

## 3. How "previously used" and "duplicate" are detected

Independence is decided from bytes and pixels plus the campaign's own inventory — never from a filename.

| What the screen computes | Against what |
|---|---|
| SHA-256 of the file | the PR103 inventory of 640 files, which records each file's *purpose* (synthetic training render, development real scan, regression fixture, derivative, demo asset) |
| SHA-256 of the text layer of every page of a PDF | the same inventory, so a re-exported PDF that lost its bytes is still caught |
| perceptual hashes (`ahash`, `dhash`) within 10 bits | every image the campaign has ever run the pipeline on |
| SHA-256 inside your own intake folder | so the same invoice cannot be counted twice |

What you will see, and what it means:

* `excluded — exact_duplicate: <path> (<purpose>)` — this file is byte-identical to something already used. The
  message names the file it matched, so you can check the judgement yourself.
* `excluded — visually_similar_duplicate: within 10 bits of <path> (<distance>)` — same page, different bytes
  (recompressed, re-exported, resized). Re-scan at a different resolution does not make it independent.
* `excluded — previous_use: PR103 classifies this hash as <purpose>` — a document the campaign already used.
* `excluded — duplicate_inside_the_intake: the same bytes are also present as <file>` — you dropped the same
  invoice twice.
* `pending_review` — not a duplicate and not a problem: something is still missing (no annotation, an unsealed
  one, permission blank).

Known limits, stated rather than hidden: the perceptual screen is unreliable on near-empty pages (a blank
white page and a template screenshot can sit within 10 bits of an unrelated example image) — if the screen
excludes a document you *know* is new, tell me and I will re-measure it; and a genuinely new invoice from the
*same supplier* as a development document is allowed, because independence is per document, but it will look
similar and the finding will be printed for you to confirm.

## 4. The 12 documents: what a useful cohort looks like

`pr110_ready.py` reports coverage gaps against these categories (a category with fewer than two documents is
`underrepresented`). You do not have to satisfy all of them, but a cohort that is 12 × "one-page French PDF
with a bordered table" measures one thing.

French · British · American · Canadian · genuinely issued (not a template) · bordered table · borderless table
· multi-line descriptions · extra reference columns · explicit tax breakdown (HT and TTC differ) · no tax
breakdown (single total) · two or more line items · small text or low-contrast acquisition.

## 5. Recording the correct values — the guided CLI

This section is the process to follow. `ANNOTATION_PROTOCOL.md` in the same folder is PR103's earlier
two-annotator protocol for a different schema (`invoice-blind-annotation-v1`, with bounding boxes); PR110
scores against the PR103/PR110 reference schema instead, so follow this guide and record a second reviewer in
`document.second_review` rather than producing two parallel forms.

For each invoice, four commands (`python` below means the project interpreter — see the note under the block;
`pr110_ready.py` prints its launch command with the absolute interpreter path for exactly that reason):

```
# 1. create a blank reference, recording the file's hash, page count and DPI
python scripts/pr110_annotate.py new inv01 --invoice benchmark_prep/invoices/facture-2026-07.pdf

# 2. answer the questions (provenance first, then the header, then each line item)
python scripts/pr110_annotate.py guide inv01

# 3. check, then seal
python scripts/pr110_annotate.py check inv01 --invoice benchmark_prep/invoices/facture-2026-07.pdf
python scripts/pr110_annotate.py seal inv01 --invoice benchmark_prep/invoices/facture-2026-07.pdf
```

Use the project interpreter (`.venv/Scripts/python.exe`, Python 3.12) — the system Python has no Tesseract
bindings and the evaluation would then refuse to run.

Two more commands are useful at any point in the flow:

```
# print the reference JSON and say where the invoice file is
python scripts/pr110_annotate.py show inv01

# print how many fields will be scored, how many stay unverified, what is still open
python scripts/pr110_annotate.py summary inv01
```

`guide inv01 --redo` asks every question again. `guide` skips what you already answered, so you can stop
mid-document and finish later.

The order of the guide is deliberate:

1. **provenance** — where the invoice came from, that a business really issued it, that you permit local
   evaluation and what that permission rests on, language, VAT convention, how it was acquired, which pages,
   image quality, who annotated it, who reviewed it;
2. **header** — `supplier_name`, `invoice_date`, `currency`, `total_ht`, `total_ttc`, `tax.rate`,
   `tax.amount`, `shipping`;
3. **line items** — `description`, `quantity`, `unit_price`, `total_ht`, plus an optional printed reference
   number per line. Answer `n` to "add line item" when there are no more.

Two rules the tool enforces, because they are where previous rounds went wrong:

* **Copy what is printed, exactly.** Not what you believe the total should be: what the document prints, in the
  printed order, with the printed separators. If the invoice prints `1 234,56 €`, enter that.
* **Shipping, discounts and tax are not line items.** They have their own fields. A row for "Livraison" makes
  the row count wrong and the table score meaningless.

**Do not open the reference file after the extraction output exists for that document.** The reference is blind
by construction: the annotation tool never imports the pipeline, OCR or a model, so it cannot show you a
prediction, and there is no way to un-see one. Annotate first, evaluate later.

## 6. Provenance for every value

Every field is asked as *value + state*. The state decides how the value is scored — this is the part that
matters most.

| State | You type | Meaning | Scored as |
|---|---|---|---|
| `printed` | `p` | the document prints this value | a normal expectation: correct, or **incorrect** if extraction disagrees |
| `derived` | `d` | the document does not print it; it follows by arithmetic from printed values — you are then asked *which* values and *what* arithmetic | still scored, and the derivation is kept in the file so a later agent cannot mistake it for printed truth |
| `absent` | `a` | the document has no such line at all | **unverified** — excluded from every denominator |
| `illegible` | `u` | there is ink there but it cannot be read | **unverified** |
| `ambiguous` | `m` | readable but undecidable (e.g. `0` and `O` are indistinguishable at this resolution) — you are then asked *what* is undecidable | **unverified** |
| skip | `s` | nothing recorded | the validator reports missing provenance and the document stays `pending_review` |

`absent`, `illegible` and `ambiguous` force the value to `null`. That is the point: "genuinely unknown" is
recorded as unverified instead of guessed, so an extraction that happens to match a guess earns no credit and a
human guess costs nothing.

Example: an invoice prints a subtotal `1 000,00 HT`, a VAT line `200,00` and a grand total `1 200,00 TTC`.
`total_ht` = `p` `1000,00`; `total_ttc` = `p` `1200,00`; `tax.amount` = `p` `200,00`. If there were no printed
VAT amount but it follows as `total_ttc - total_ht`, use `d` and write exactly that. If the document prints one
combined total with no tax line, `total_ht` = `a` → null, and extraction is not penalised for not inventing it.

A second reviewer is not mandatory, but `second_review.performed = y` with the disagreements resolved is what
turns a single-person annotation into evidence I can quote; when two of you disagree, record the state as
`ambiguous` rather than picking a winner.

## 7. Launching validation

Run from the repository root, with the French language data visible (without `TESSDATA_PREFIX`, Tesseract
silently reads French documents in English only — PR81/PR82 showed that produces numbers that look like
pipeline failures):

```
export TESSDATA_PREFIX="$PWD/.venv/tessdata"          # Windows PowerShell: $env:TESSDATA_PREFIX="$PWD\.venv\tessdata"
```

Then, in order:

| Command | What it does | Makes OCR calls |
|---|---|---|
| `python scripts/pr110_status.py` | re-verifies the 29 production sources against the PR103/PR109 freezes, checks every experimental switch is off, prints how many invoices and references exist and the reuse map | 0 |
| `python scripts/pr103_dataset_manifest.py --output-name <name>.json` | re-hashes every file in the repository and records what the campaign used each one for; hand the result to the next line with `--inventory` so "already used" covers files added after PR103's freeze | 0 |
| `python scripts/pr110_eligibility.py --self-test` | classifies every file in the intake as **eligible / excluded / pending_review** with the reason for each | 0 |
| `python scripts/pr110_ready.py` | the gate: needs ≥ 12 eligible **and** every reference valid + sealed; prints coverage gaps; writes the cohort freeze only when it passes; exits non-zero otherwise | 0 |
| `python scripts/pr110_evaluate.py --dry-run` | re-verifies the freeze from the bytes on disk and lists what would run | 0 |
| `python scripts/pr110_evaluate.py` | extracts each frozen document once through the production path and scores it | one full production run per document (PR102 measured several seconds per document, OCR-only, single process) |

The eligibility screen's authority for "this document was already used" is a manifest of the repository taken
when PR103 ran, so it cannot name a file created after that snapshot. The byte and perceptual screens still
apply to such a file; rebuilding the inventory and passing it with `--inventory` closes the interval to the
time between the two runs. PR129 measured it on the 8 files the frozen manifest classifies outside the "used"
purposes: 2/8 excluded against the frozen manifest, 8/8 against a rebuilt one
(`results/product-review110/pr129-phase2-purpose-gap.json`).

`pr110_ready.py` refuses to prepare a run when either gate fails and never freezes a partial cohort: nine
documents is not a smaller benchmark, it is a different one. If the gate says `NOT_READY`, the printed reasons
are the work list — there is no flag that overrides it.

`pr110_evaluate.py` also refuses, before the first pipeline call, if the freeze does not recompute (a reference
edited after sealing, a document swapped on disk, a production source that changed since the freeze) or if the
cohort contains French invoices and `fra` cannot be loaded here.

## 8. Reading the results

The evaluation prints four lines and writes the full JSON under `results/product-review110/`.

* `header` — the four scored header fields (`supplier_name`, `invoice_date`, `total_ht`, `total_ttc`):
  `correct`, `incorrect`, `missing`, `unverified`.
* `rows` — `expected` line items in the reference vs `exported` by the pipeline, and how many were matched
  `correctly`, `missing`, or `misassociated` (right values, wrong row).
* `fields` — cell level: `correct`, `incorrect`, `incorrect_association`, `unverified`.
* `position` — cells scored by column position, the fallback when row association is not decidable.

How to judge them:

1. **`incorrect` is the number that matters, not `correct`.** An abstention (missing field) costs coverage; a
   wrong value claims something that is not printed. A run whose `incorrect` is 0 with modest coverage is a
   trustworthy system; high coverage with non-zero `incorrect` is not.
2. **Denominators are `correct + incorrect + missing`.** Fields the reference recorded as `absent`, `illegible`
   or `ambiguous` are `unverified` and excluded, so the score reads "correct out of scored", not "correct out of
   printed". `reference_projection` per document tells you how many values each reference contributed.
3. **Per-document, never only aggregate.** Twelve documents averaged together hide that four of them produced
   nothing; read the `documents` list for where the loss is.
4. **`runtime` is reported per document** (`seconds`, `ocr_calls`, `preprocessing_used`, `source_kind`) — a
   result that cannot say how it was produced is not a result.
5. A model is **not** part of this run. The evaluation is OCR-only through the production entry point; if you
   are asked to measure the fine-tuned model on this cohort, that is a separate, explicit instruction with its
   own hardware budget.

---

### If something looks wrong

Say what you saw rather than editing the tooling: the printed reason from the gate, the document id, and the
SHA-256 prefix. Do not adjust thresholds, consensus rules, or a sealed reference to make a number move — those
are the three ways this campaign has previously produced a result that meant nothing.
