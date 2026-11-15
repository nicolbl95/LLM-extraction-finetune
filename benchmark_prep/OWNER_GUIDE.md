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
| The document authority **and the variants it names** (`diagnostics/holdout/`, `diagnostics/anonymized/`, `real-scans/`, `real-scans-supplementary/`) | built on the owner's machine by `scripts/pr110_authority.py`, from `$PR110_DOCUMENT_AUTHORITY` or `benchmark_prep/document-authority.private.json`; the images come from the same machine | no — PR167 took the registry out of Git, and the images never were in it | decide `registered_variant`, and hold `previous_use` to what the PR103 inventory covers. Without the registry **nothing is admitted** (section 3b); without its images the registry is incomplete, which is the same refusal |
| The demonstration registry (`benchmark_prep/document-authority.demo.json`) | comes with the clone | yes | show the format and let the tests run without private data. It declares `scope: "demo"` and admits nothing |
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

Independence is decided from the document — its bytes, its variants and its recorded history — never from a
filename and never from a resemblance alone. The four steps run in this order and the first that fires
decides:

| Step | What is compared | What it establishes |
|---|---|---|
| 1 | SHA-256 against the PR103 inventory of 640 files, each with the purpose it was used for | this exact file was already used |
| 1b | SHA-256 against every variant in the document authority (section 3b), including the original downloads that live outside the repository | this exact file is a known variant of a document the campaign used |
| 2 | the page's geometry against those registered variants: one homography on which at least 60 features agree, covering at least 70 % of the surviving correspondences | a re-rendered copy — resized, recompressed, cropped, screenshotted — of a known document |
| 3 | perceptual hashes within 10 bits of an image the campaign already ran | a *resemblance* nobody has confirmed: manual review, never an exclusion |
| 4 | SHA-256 inside your own intake folder | the same file was dropped twice |

Step 3 no longer excludes, because PR163 measured two genuinely different invoices inside that 10-bit
radius: a British template at 2 bits from a US scan, a pro forma at 10 bits from another one. A visual
resemblance is evidence of *something* and proof of nothing, so it holds the document for a human.

What you will see, and what it means:

* `excluded — exact_duplicate: <path> (<purpose>)` — byte-identical to a file already used, or to a recorded
  variant of one. The message names the file it matched.
* `excluded — registered_variant: the page is a registered variant of <document> (<role>): N of M features
  agree on one homography` — same page, different bytes: recompressed, re-rendered, resized, screenshotted,
  or converted from the original download. Re-scanning at a different resolution does not make it
  independent.
* `excluded — previous_use: PR103 classifies this hash as <purpose>` — a document the campaign already used.
* `excluded — duplicate_inside_the_intake: the same bytes are also present as <file>` — you dropped the same
  invoice twice.
* `pending_review` with `visually_similar_duplicate: resemblance only, not identity: within 10 bits of …` —
  the page looks like a document the campaign used, but nothing confirms that it *is* that document. It is
  not admitted; tell me and I will measure it again.
* `pending_review` with `registered_variant: … the correspondence is inconclusive` — the page shares enough
  geometry to be related, not enough to be the same rendering. Two invoices issued from one template look
  like this, which is why the screen asks instead of deciding.
* `pending_review` with `document_authority` — the authority is missing, unreadable, one of its variants
  cannot be found or no longer matches its recorded hash, **or** a development image sits in a folder the
  authority claims to have covered without being named in it, **or** it names no document and **or** it
  declares a scope other than this repository's (section 3b). The message names the files. **Nothing is
  admitted in this state**: a screen without its authority is not evidence of independence. Fix it by
  rebuilding the authority (section 3b) rather than by editing the report.
* `pending_review` with no duplicate finding — not a duplicate and not a problem: something is still missing
  (no annotation, an unsealed one, permission blank).

### 3b. The document authority is a private file

Steps 1b and 2 need a record of what the campaign already used, and PR167 moved that record out of Git: the
authority names development documents and says where each variant is kept, which is precisely what a public
clone must not carry. Three paths, one behaviour each:

| Where | In Git | What a screen does with it |
|---|---|---|
| `$PR110_DOCUMENT_AUTHORITY`, else `benchmark_prep/document-authority.private.json` | no — ignored | **the** authority: `registered_variant` is decided from it, and the development corpus behind every `exact_duplicate` |
| `benchmark_prep/document-authority.demo.json` | yes | a format demonstration holding only repository-shipped paths. It declares `scope: "demo"`, and a screen that loads it holds every document |
| `benchmark_prep/document-authority.json` | no longer tracked | the path PR164–PR166 published the registry under. Nothing reads it; if it is still on disk, move it to the private path |

What the authority holds is a document id, a category, the declared region and per-variant `path`, `sha256`,
role and `expect_present` — no party name as a field, nothing printed on a page, no absolute path. Two
consequences: a supplier's name can survive inside a *filename*, which is why the file stays local, and the
registry must never be copied into the Hugging Face demo package, which needs none of it.

**A clone without the authority admits nothing, and that is the intended result.** The absence of a record is
silence, not evidence that a document is new. So the screen answers `pending_review` with `document_authority`
for every file, `pr110_ready.py` prints the absence as a named blocking finding and freezes nothing, and
`pr110_evaluate.py` refuses a freeze whose independence it cannot re-establish — before importing the
pipeline. The same refusal covers a registry that is corrupt, one whose variants are missing or no longer
hash to what was recorded, one whose `built_from` manifests have moved, one that names no document at all, and
the demonstration file loaded by mistake: an empty match list against an empty registry proves nothing. The fix
is always the authority, never a flag — there is none.

**To add a document to the authority**, let it enter development use (a corpus the registry covers, or the
recorded hold location), then rebuild and verify before anything is frozen:

```
python scripts/pr110_authority.py            # rebuild the private registry, and record the manifests read
python scripts/pr110_authority.py --check    # verify: named variants present and unchanged, nothing covered unnamed
python scripts/pr110_eligibility.py --self-test
```

The rebuild covers the corpora that exist on the owner's machine, so it cannot be run on a clone; there the
registry is copied, like `data/invoices/`. Freeze the cohort only after it, because the freeze copies the
authority's state — path, scope, version, how many documents and variants verified — so a later run can be
tied to the authority that certified it.

### Independence and admissibility are two different questions

* **Independent** is what the four steps above decide, from the document's bytes and geometry: this is not a
  copy of something the campaign already used.
* **Admissible** is what your answers decide, in `pr110_annotate.py guide`: a genuinely issued business
  invoice, of a region and language the cohort scores, in a condition you can read, with your permission to
  use it locally. A document can be perfectly independent and inadmissible — one of the eleven pages PR163
  measured was genuinely unseen by the campaign and, on the owner's answers, not a document the cohort
  scores; the gate certified in PR165 does not count it.

Both layers have to agree before a document reaches the cohort. `pr110_ready.py` counts and freezes only the
references whose document the screen classifies `eligible`; a reference that validates and seals on a document
the screen excludes or holds is listed under `intake_gate.annotated_not_admitted` with the screen's reason, and
is not one of the twelve. An unanswered admissibility question cannot even reach that point: `seal` refuses to
freeze a reference whose language, region or permission is blank.

Three things the screen cannot do for you, and you should not ask it to:

1. **Provenance is a declaration.** The screen recognises documents; it cannot know where a file came from.
   `pr110_annotate.py guide` asks for the origin and for the permission, and both are recorded next to the
   hash. An unverified origin is what `pending_review` is for.
2. **A demo invoice is not an invoice.** A downloaded template, a mock-up, a pro forma or a credit note is
   recognised as a document and can still be excluded on `genuine_invoice` even when it is brand new. The
   cohort counts genuinely issued invoices only; `guide` asks the question explicitly and refuses to seal
   `genuine_business_invoice = false`.
3. **Two copies are one document.** The same invoice saved twice, or once as PNG and once as WebP, is one
   document: it contributes one row to the cohort and is caught by steps 1, 1b and 4.

Known limits, stated rather than hidden: identity is decided from the page's geometry, never from its words
(PR164 makes no OCR call), so a document re-rendered beyond recognition could still pass — the similarity
finding stays alongside for that reason; and a genuinely new invoice from the *same supplier and the same
template* as a development document is allowed, because independence is per document, but it will look
similar and will be held for review.

## 4. The 12 documents: four under each of three conventions

Twelve is not the requirement — twelve of one kind measures one billing convention and reports it as three.
The protocol wants **4 invoices declared `FR`, 4 `UK` and 4 `US`**, and PR167 turned that into the third gate:
`pr110_ready.py` now needs the intake gate (12 eligible) **and** the annotation gate (12 sealed and valid)
**and** the coverage gate (4/4/4 of the *selected* cohort) before it writes a freeze. A shortfall is printed
with the composition it measured, not described in prose:

```
FR: 3/4, UK: 4/4, US: 4/4 — NOT_READY, the selected cohort leaves a quota short: missing FR: 1
```

* **Which twelve are selected is deterministic, not whatever the disk returned first.** Within a country,
  eligible references are ordered by (document id, document sha256) and the first four are taken, so the same
  intake and the same sealed references always select the same cohort; the freeze records the ids, the counts
  and a `selection_sha256` a later run is diffed against.
* **A Canadian invoice does not become a French one.** Candidates beyond a quota, and documents whose region
  carries none, are named in `reserve` and never evaluated — that is how a full intake with the wrong shape
  reads, rather than a shortage.
* **Two references on one document id are one invoice in two slots**, and an unsealed reference is not a slot
  at all; the gate counts documents, not files.

Collect more than twelve if you can: the reserve is what lets a document be withdrawn after the freeze without
re-running the gate.

These categories are still reported, and stay advisory where the quota is not — a category with fewer than two
documents is `underrepresented`:

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
| `python scripts/pr110_authority.py --check` | verifies the document authority against the disk: every variant it names is present and still hashes to what was recorded, its `built_from` manifests still hash, it names at least one document and declares this repository's scope, and no development image sits in a folder it covers without being named in it; prints the offenders and exits non-zero otherwise | 0 |
| `python scripts/pr110_authority.py` | rebuilds the private authority (section 3b) from the development corpora and the recorded holds, and is the fix when the line above reports a stale authority. Re-run it after any document enters development use, before any cohort is frozen. `--demo` writes the tracked format demonstration instead and admits nothing | 0 |
| `python scripts/pr110_eligibility.py --self-test` | classifies every file in the intake as **eligible / excluded / pending_review** with the reason for each | 0 |
| `python scripts/pr110_ready.py` | the gate: needs ≥ 12 eligible, **and** every reference valid + sealed, **and** a 4/4/4 selection (section 4); prints the composition it measured and the coverage gaps; writes the cohort freeze only when all three pass; exits non-zero otherwise | 0 |
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

The document authority is a snapshot of the development corpora, so it ages the moment a document enters
development use. A registry that has not been rebuilt does not know that document, and an unknown document is
the one thing the authority exists to refuse: PR165 measured that three of the ten attachments it already
recognises turn `eligible` with no open question once their identity is deleted from the loaded registry. That
is why `--check` now compares the folders a registry covers against the images it names, why an incomplete
authority holds every document at the screen, and why it is a named blocking finding at the gate rather than a
silent shortage. The order that cannot be walked past is: rebuild the authority, re-run `--check`, then run
`pr110_ready.py`.

`pr110_evaluate.py` also refuses, before the first pipeline call, if the freeze does not recompute (a reference
edited after sealing, a document swapped on disk, a production source that changed since the freeze) or if the
cohort contains French invoices and `fra` cannot be loaded here.

### What the freeze holds, and what the evaluation re-checks

The freeze `pr110_ready.py` writes is the contract the run is measured against: the twelve document ids with
the region each declares, the document and sealed-reference digests, the selection (quota, counts, ids,
`selection_sha256`, reserve), the authority the documents were certified against, the production source hashes
and the three gate results. Before extracting anything, `pr110_evaluate.py` re-runs those conditions against
the bytes on disk and returns `verdict: REFUSED` with 0 documents when:

* a reference was edited after sealing, or the file on disk is not the document that was annotated;
* an invoice was added to the intake and takes a quota slot, or a frozen one was withdrawn, so these bytes no
  longer select the frozen twelve — the message names who was withdrawn and who was newly taken;
* a frozen slot was relabelled to another country (`declares region 'US' where the freeze certified 'UK'`),
  which is how a shortfall would otherwise be patched after the fact;
* the composition no longer satisfies 4/4/4, or the freeze records no selection at all;
* the document authority is absent, corrupt, incomplete, or the demonstration file (section 3b): a run that
  cannot re-establish independence does not get to report one;
* a production source changed since the freeze, or the cohort has French invoices and `fra` cannot load here.

Preparing the evaluation is then this order, and none of it costs an OCR call until the last line:

```
python scripts/pr110_authority.py --check          # the authority is current, or nothing is admissible
python scripts/pr110_eligibility.py --self-test    # per file: eligible / excluded / pending_review + why
python scripts/pr110_ready.py                      # three gates, then the freeze
python scripts/pr110_evaluate.py --dry-run         # the freeze re-computes on these bytes
python scripts/pr110_evaluate.py                   # one production run per frozen document
```

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
