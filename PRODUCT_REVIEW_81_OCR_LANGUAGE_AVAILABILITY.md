# PRODUCT-REVIEW-81 — OCR language availability and reproducible French validation

Branch `wip/layouts-reprise`, HEAD `9d7a641`, no commit, push, stash, reset, clean, branch change or deployment was performed.
Scope: `app/main.py`, four test files, one new test file, this report, `QODER_HANDOVER.md`, project memory. No model inference, no GPU, no network, no real invoice upload, no Hugging Face Space change.

## 1. Root cause

`app/main.py` requested `lang="fra+eng"` as a literal string inside `ocr_image()`, and nothing on the read path ever asked Tesseract which languages it could actually load.

Tesseract 5.4 behaviour, verified directly on this machine:

| Request | `fra` installed | Result |
|---|---|---|
| `lang="fra+eng"` | yes | French + English recognition |
| `lang="fra+eng"` | **no** | **silently continues with `eng` only** — exit 0, no warning to the caller |
| `lang="fra"` | no | `TesseractError: Failed loading language 'fra'` |

So the only case that fails loudly is the one nobody used. An installation that lacks `fra` still returns readable text — French diacritics are simply absent from the evidence (`marinière` → `mariniere`, `Facile à` → `Facile a`), which is exactly the PR80 Joanna failure.

Two existing mechanisms failed to make this visible:

- `ocr_availability()` did detect the missing language, but it was never consulted by `ocr_image()`; its only per-extraction caller discarded the reason string (`engine_available, _ = ocr_availability()`), so the report said "Outil de lecture de texte : indisponible" without ever naming the missing language.
- The frozen OCR policy (`src/ocr_config.py`, `version="ocr-policy-5-bounded-fastpath-quota"`) has **no language field**. Historical artifacts such as `results/real-scans-review.json` store `ocr_configuration` and `ocr_configuration_sha256` (`6ec4771b…`) and therefore cannot prove which languages the run actually loaded.

This is an availability/observability defect, not a consensus defect. The accent-preference logic that PR80 relied on was already correct; it only ever sees text the engine was able to produce.

## 2. Exact changes

### Product code — [app/main.py](./app/main.py)

| Location | Change |
|---|---|
| [app/main.py:59-60](./app/main.py) | `OCR_REQUESTED_LANGUAGES = ("fra", "eng")` and `_OCR_LANGUAGES_CACHE` (module constants). The requested language set is now declared once instead of being embedded in a call string. |
| [app/main.py:228-237](./app/main.py) | `installed_ocr_languages(*, refresh=False)` — one `pytesseract.get_languages()` query, cached by `(tesseract_cmd, TESSDATA_PREFIX)`; a failing listing is logged and cached as the empty set instead of raising. Never called per image. |
| [app/main.py:240-253](./app/main.py) | `ocr_language_plan(*, requested=…, refresh=False)` — the four-way distinction the objective asks for: `requested`, `effective` (requested ∩ installed, in requested order), `missing`, `fallback` (some requested missing **and** some still usable), plus `installed` and `tessdata_prefix`. |
| [app/main.py:256-272](./app/main.py) | `require_ocr_languages(*, requested=…, require_all=False, refresh=False)` — general-purpose prerequisite gate. Default: refuse only when **nothing** is usable (public fallback policy unchanged). `require_all=True`: refuse when any required language is missing — this is the strict benchmark mode. |
| [app/main.py:280-295](./app/main.py) | `ocr_image()` resolves the plan once, logs a WARNING naming requested/effective/missing/tessdata when a fallback occurs, and passes `lang="+".join(plan["effective"])`. The literal `"fra+eng"` is gone. |
| [app/main.py:1429-1440](./app/main.py) | `ocr_availability()` now derives its answer from the cached plan. Boolean semantics are unchanged (`False` whenever a requested language is absent); the reason string gained the effective read: `"langues absentes : fra ; lecture effective : eng"`. |
| [app/main.py:642-651](./app/main.py) | `run_extraction_ui()` adds one line to the existing "### Lecture du document" block for OCR/consensus sources: `- Langues de lecture : fra+eng.` when complete, or `- Langues de lecture : fra+eng demandées ; eng effectivement utilisées ; absentes : fra.` |

`src/` is untouched: no change to `ocr_config.py`, `ocr_consensus.py` (including the PR80 identifier consensus), `invoice_grounding.py`, `invoice_evidence.py`, monetary or quantity handling, or model inference. `packages.txt` is byte-identical to `HEAD`.

### Tests

| File | Change | Why |
|---|---|---|
| [tests/test_ocr_language_availability.py](./tests/test_ocr_language_availability.py) | **New.** 8 deterministic tests, all language conditions produced by patching `installed_ocr_languages` / `pytesseract.image_to_data`; no OCR subprocess, no network, no GPU. | Phase 3 requirement. |
| [tests/test_grounding.py:435-449](./tests/test_grounding.py) | `test_image_ocr_is_x2_grayscale_tsv` pins `installed_ocr_languages` to `{fra, eng}` before asserting `lang == "fra+eng"`. | The assertion itself is unchanged and still exact; it no longer passes or fails depending on the host installation. |
| [tests/test_document_report.py:257-265](./tests/test_document_report.py) | The Joanna French-layout acceptance test now skips with an explicit reason when `fra` is absent, instead of failing on accents under an English-only read. | "Do not allow a test to pass or fail under English-only OCR while claiming to have validated the French OCR configuration." The executable-absent case still `pytest.fail`s as before. |
| [tests/test_grounding_slow.py:19-25](./tests/test_grounding_slow.py) | The `france.png` end-to-end case skips when `fra` is absent. | Same rule for the real-model French path (still `slow`-marked and deselected by default). **Not executed here:** with no `INVOICE_SLOW_ADAPTER_PATH` configured the pre-existing adapter skip at line 15 fires first, so `-m slow` gave `3 skipped` in both environments; the new guard was only syntax-verified. |

The existing skip convention (`"Tesseract fra/eng languages are missing"` in five files) and the existing strict convention (`RuntimeError("Isolated Tesseract data lacks required languages: …")` in `scripts/capture_product_review64_ocr_baseline.py`) were kept as they are; no expected invoice value, threshold, or validation rule was altered.

## 3. Before / after

| Situation | Before PR81 | After PR81 |
|---|---|---|
| `fra` + `eng` installed | Reads with `fra+eng`, report says "disponible". | Identical: `effective=["fra","eng"]`, `missing=[]`, `fallback=False`, reads with `fra+eng`, `- Langues de lecture : fra+eng.` |
| `fra` missing, `eng` present (this machine's default) | Reads with `eng`, **while every call and every artifact still said `fra+eng`**; report said "indisponible" with no explanation; the Joanna test failed on two accents. | Reads with `eng`, WARNING `OCR language fallback: requested fra+eng, effective eng, missing fra (tessdata prefix: …)`, report states requested/effective/absent, `ocr_availability()` reason names `fra` and the effective read, Joanna test skips explicitly. Public extraction still completes — fallback policy unchanged. |
| No usable language at all | `TesseractError` surfaced unpredictably depending on the call. | Deterministic `RuntimeError("OCR indisponible : installez Tesseract et les langues fra/eng (voir packages.txt).")` — the message `tests/test_app.py::test_ocr_failure_is_explicit` already asserts. |
| Benchmark/test requiring French | Guards existed in 5 test files and 1 script, but nothing in the product expressed a strict mode; 2 tests were unprotected. | `require_ocr_languages(require_all=True)` raises naming the missing language, the requested set, the installed set and the tessdata path; the two unprotected tests are now guarded. |
| Cost of the check | — | 1 `get_languages()` subprocess per `(tesseract_cmd, TESSDATA_PREFIX)`, not per image: measured **1 listing for 7 plan queries** and **1 listing across 3 full `ocr_image()` reads (36 Tesseract calls)**. |

## 4. Phase 4 — measured results in both environments

Same binary in both: `C:\Program Files\Tesseract-OCR\tesseract.exe`, `tesseract v5.4.0.20240606`, leptonica-1.84.1.

| Fact | Environment A (`fra` available) | Environment B (shell default) |
|---|---|---|
| Effective tessdata directory | `…\.venv\product-review64\tessdata\` (set through `TESSDATA_PREFIX`) | `C:\Program Files\Tesseract-OCR/tessdata/` (`TESSDATA_PREFIX` unset) |
| Languages reported by Tesseract | `eng`, `fra` (2) | `eng`, `osd` (2) |
| `main.ocr_language_plan()` | `requested=[fra, eng]`, `effective=[fra, eng]`, `missing=[]`, `fallback=False` | `requested=[fra, eng]`, `effective=[eng]`, `missing=[fra]`, `fallback=True` |
| `main.ocr_availability()` | `(True, "fra+eng disponibles")` | `(False, "langues absentes : fra ; lecture effective : eng")` |
| Languages actually passed to Tesseract | `fra+eng` | `eng` only |

The `fra.traineddata` used for Environment A is the hash-verified PR64/PR72 pin (1 130 365 bytes, SHA-256 `ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca`); `tests/test_product_review64_ocr_baseline.py:409-431` re-verifies it and is green in both environments (it sets its own prefix).

| Suite | Environment A | Environment B |
|---|---|---|
| `tests/test_ocr_language_availability.py` alone (8 tests) | 8 passed (10.16 s) | 8 passed (9.74 s) |
| Joanna acceptance + `tests/test_ocr_language_availability.py` + `tests/test_grounding.py` | **79 passed**, 0 skipped, 0 failed (41.44 s) | **77 passed, 2 skipped**, 0 failed (15.65 s) — skips: Joanna (new explicit reason), `tests/test_grounding.py:495` (pre-existing) |
| PR79 expanded fallback suite (`test_app`, `test_grounding`, `test_real_runtime_regressions`, `test_product_review64_ocr_baseline`, `test_french_book_table`, `test_uk_layout_regressions`, `test_reliability_generalization`) | **221 passed, 34 subtests passed, 0 failed, 0 skipped** (177.88 s) — identical to PR80 | **212 passed, 9 skipped, 28 subtests passed, 0 failed** (52.91 s) — identical to PR80 |
| Adjacent suite (`test_invoices`, `test_document_report`, `test_report_quality_warnings`) | **80 passed**, 0 failed, 0 skipped (53.86 s) | 77 passed, **3 skipped, 0 failed** (20.58 s) — PR80 measured 77 passed / **1 failed** / 2 skipped here; the failure is now an explicit, reasoned skip |
| Description trio (`test_grounding`, `test_description_grounding`, `test_ocr_text_integrity`) | 85 passed, 0 skipped (32.61 s) | 84 passed, 1 skipped (11.98 s) — PR80 recorded 84 passed / 1 skipped in both |
| PR80 accent + identifier subset (`-k "accented or accent or ligature or invoice_number or identifier"`) | 22 passed, 110 deselected (0.95 s) | 22 passed, 110 deselected (0.86 s) — **PR80 behaviour preserved, byte-for-byte same counts** |
| Review + budget + config (`test_app_review_ui`, `test_app_review_persistence`, `test_invoice_review`, `test_invoice_review_edit`, `test_ocr_budget`) | 62 passed, 4 subtests (23.73 s) | 62 passed, 4 subtests (21.27 s) |
| `tests/stress/test_null_recovery.py -k "accent or description"` | 4 passed | 4 passed |
| Combined final check (`test_ocr_language_availability` + `test_grounding` + `test_document_report` + `test_app`) | not run as one command | **190 passed, 4 skipped, 28 subtests passed, 0 failed** (61.51 s) |
| `-m slow tests/test_grounding_slow.py` | 3 skipped (adapter prerequisite, as before) | 3 skipped (adapter prerequisite, as before) |

**Newly introduced failures: none in either environment.** The nine OCR-dependent skips in Environment B are pre-existing language guards; they execute and pass in Environment A (212 + 9 = 221, matching PR80 exactly). Skips were produced with `-rs` and are reported above as skips; none is presented as a pass.

Expected differences between the two environments, all accounted for: the Joanna acceptance test (passes in A, skips in B), `tests/test_grounding.py:495` (passes in A, skips in B), the 9-skip/0-skip split in the expanded fallback suite, and the description trio's 85 vs 84+1.

## 5. Effect on the reliability of historical French benchmarks

- Runs whose artifacts positively record the languages are trustworthy for French: `results/ocr-french-smoke-2026-11-01.json` records `tessdata_prefix` and `languages: ["eng","fra","osd"]` with `app_ocr_availability: available=true`, and `scripts/capture_product_review64_ocr_baseline.py` refuses to start without `eng` + `fra` and hash-verifies both `traineddata` files.
- Runs recorded only through `ocr_configuration` / `ocr_configuration_sha256` (e.g. `results/real-scans-review.json`, and the independent real-scan and six-layout evaluations built on the same harness) **cannot be shown to have used `fra`** — the policy fingerprint has no language field, so an English-only degraded run and a full `fra+eng` run produce the same fingerprint. Their French figures must be treated as "language environment unproven" and re-measured before being cited as French OCR or French extraction results.
- Cache hazard found but **not** changed (out of scope): `scripts/experiment_ocr.py:48-67` requests `fra+eng` and keys its TSV cache on `image_hash / method / psm / v1` — the language environment is not part of the key, so a degraded run can serve cached English-only text to a later run that does have `fra`. Any re-measurement must delete or version that cache first. `scripts/diagnose_independent_benchmark.py:103` and `scripts/shadow_fast_path.py:43` also still hardcode `fra+eng`.
- The PR80 conclusion is unchanged and now mechanically enforced: the Joanna diacritic gap is the degradation itself, not a recovery gap.

## 6. Remaining risks and blockers

1. Historical French benchmark numbers remain unproven (see §5); re-measuring them costs OCR time and was not authorized here.
2. `scripts/experiment_ocr.py`'s language-blind TSV cache can still poison future research runs.
3. The report's language line reads the ambient plan at render time rather than the plan captured during the read. They are identical in every realistic path (same process, same environment) but are not bound together in `diagnostics`; binding them would mean changing the diagnostics contract.
4. `ocr_availability()` still returns `False` during a permitted fallback, so the report can pair "Outil de lecture de texte : indisponible" with a line explaining that a usable language was read anyway. This is pre-existing semantics, deliberately preserved; changing the boolean would alter an existing public status.
5. Unchanged project blockers: no independent real-invoice validation set (15 invoices), six zero-row layouts, Microspective rows, 19 modified + 99 untracked paths still uncommitted (committing requires explicit authorization), and the full repository suite was **not** run — `pytest.ini` keeps `-m "not slow"`, so nothing here exercised real-model inference.

## 7. Staging readiness decision

**For this change: YES to keep staging as it is.** `packages.txt` installs `tesseract-ocr`, `tesseract-ocr-fra`, `tesseract-ocr-eng` and is byte-identical to `HEAD`, so on the Space the plan resolves to `effective=[fra, eng]`, `fallback=False` — the exact pre-PR81 read. The only user-visible differences there are one added report line and a WARNING that never fires. No Space modification was made and none is needed.

**For the release as a whole: NO**, for the reasons in §6, unchanged by this review.

## 8. Reproduction

```powershell
# Environment B (shell default): eng + osd only
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
.\.venv\Scripts\python.exe -m pytest -q -rs tests\test_ocr_language_availability.py
.\.venv\Scripts\python.exe -m pytest -q -rs tests\test_document_report.py::test_real_ocr_equivalent_invoice_with_unusual_french_layout_is_grounded   # skipped

# Environment A: fra available
$env:TESSDATA_PREFIX = (Resolve-Path .venv\product-review64\tessdata).Path
.\.venv\Scripts\python.exe -m pytest -q -rs tests\test_ocr_language_availability.py "tests\test_document_report.py::test_real_ocr_equivalent_invoice_with_unusual_french_layout_is_grounded"   # 9 passed
.\.venv\Scripts\python.exe -m pytest -q tests\test_app.py tests\test_grounding.py tests\test_real_runtime_regressions.py tests\test_product_review64_ocr_baseline.py tests\test_french_book_table.py tests\test_uk_layout_regressions.py tests\test_reliability_generalization.py   # 221 passed
```

Strict mode for a future benchmark:

```python
from app import main
main.require_ocr_languages(require_all=True)   # raises naming fra + the tessdata path when fra is absent
```
