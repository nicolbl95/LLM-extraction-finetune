# PRODUCT-REVIEW-80 — Joanna Invoice Regression: Diacritic Root Cause

**Date:** 2026-11-07
**Scope:** Local, CPU-only, offline. No model inference, no GPU allocation, no weight download, no real-invoice upload, no deployment, no commit, no push, no branch switch, no stash/reset/clean.

## Result

The remaining Joanna failure is **not a product defect and not a diacritic-recovery gap in the pipeline**. It is a **local environment defect**: on this Windows machine the system Tesseract installation exposes only `eng` and `osd`, so the `fra+eng` request in `app/main.py` silently degrades to English-only recognition. English-only reads drop `è` and `à` (`marinière` → `mariniere`, `Facile à` → `Facile a`) and mis-read `FR-001` as `FR-0O1`/`FR-00O1`.

With `TESSDATA_PREFIX` pointed at the hash-verified project-local `fra.traineddata`, the same command on the same fixture passes: `1 passed in 14.54s`. The accented text is genuinely recoverable from reliable existing evidence, so no accent-restoration logic was needed and none was added.

**PR80 changed no product code.** It preserved the PR80-predecessor `FR-001` identifier work, added four deterministic (no-OCR) diacritic regressions to the focused suite, and records the environment condition that earlier reports had left implicit.

**Staging readiness for this defect: YES for the code paths involved** — the deployed Space installs `tesseract-ocr-fra` through [packages.txt](./packages.txt), which is byte-identical to `HEAD`. **Readiness of the release as a whole remains NO** for the reasons listed in §7 (no independent real-invoice evidence, full repository suite not run).

## Baseline and preserved work

- Branch: `wip/layouts-reprise`, HEAD `9d7a641` (`feat: add invoice extraction review and local draft persistence`), 2 ahead / 0 behind `origin/wip/layouts-reprise`.
- Working tree at PR80 start and end: **19 modified tracked paths, 95 untracked paths**. Nothing was cleaned, reverted, staged, or committed. All 19 modified paths are still modified; the untracked count is unchanged.
- PR80 additions were made only to `tests/test_grounding.py` (already modified), plus three new untracked documents (`PRODUCT_REVIEW_80_JOANNA_REGRESSION.md`, `QODER_HANDOVER.md`, one memory file).
- The pre-existing uncommitted PR80 identifier work was verified and kept intact: `src/ocr_consensus.py` `_confident_invoice_number` / `_agreed_invoice_number` ([src/ocr_consensus.py:117-143](./src/ocr_consensus.py)) and the five new identifier tests in [tests/test_document_report.py:528-586](./tests/test_document_report.py). No `O→0` normalization exists anywhere.

## Phase 1 — OCR evidence by independent family

Bounded, CPU-only probe on `tests/fixtures/layout-joanna-binet.png` (1200×1600), one read per configured preprocessing family per segmentation, `lang="fra+eng"`, the frozen policy of `src/ocr_config.py` untouched. Script kept outside the repository (`/tmp/pr80_families.py`, `/tmp/pr80_grave.py`); the probe is reproducible with the command in §5.

### Default local environment (`TESSDATA_PREFIX` unset)

`tesseract --list-langs` → `eng`, `osd` (2 languages; **no `fra`**).

| Family / psm | description tokens | identifier token |
|---|---|---|
| deskew / 4 | `escargot`, `mariniere`, `Facile`, `accordéon` | `FR-001` |
| deskew / 6 | same | `FR-0O1` |
| clahe / 4 | same | `FR-001` |
| clahe / 6 | same | `FR-0O1` |
| crop / 4 | same | `FR-001` |
| crop / 6 | same | `FR-0O1` |
| denoise / 4 | same | `FR-001` |
| denoise / 6 | same | `FR-0O1` |
| scale3 / 4 | same | `FR-001` |
| scale3 / 6 | same | `FR-00O1` |

Token confidences for the `Facile …` line in this environment: `Facile(95-96) a(95-96) jouer(92-93) accordéon(88-91)`. **Zero of 10 reads produce `marinière` or `à`.** The three-vote identifier rule still resolves `FR-001` because `psm 4` is unanimous, but the two accented graphemes are simply absent from the evidence.

### Same probe with the frozen local `fra` model

`TESSDATA_PREFIX=.venv/product-review64/tessdata` → `--list-langs` returns `eng`, `fra`.

| Family / psm | `marini…` | `Facile …` | identifier |
|---|---|---|---|
| deskew / 4, 6 | `marinière` | `a(96)` | `FR-001` |
| clahe / 4, 6 | `marinière` | `à(96)` then `jouer(95) accordéon(95)` | `FR-001` |
| crop / 4, 6 | `marinière` | `a(96)` | `FR-001` |
| denoise / 4, 6 | `mariniere` | `a(96)` | `FR-001` |
| scale3 / 4, 6 | `mariniere` | `a(95)` | `FR-001` |

Three of five independent families read `marinière`; one family (`clahe`) reads `à`. `FR-0O1` disappears entirely: with `fra` loaded, all 10 reads give `FR-001`. `accordéon` survives in both environments, which is why the earlier reports saw only two missing diacritics — `é` is inside the ASCII-adjacent vocabulary of the English model, `è` and `à` are not.

## Phase 1 — Where the description is chosen (trace)

1. [app/main.py](./app/main.py) `ocr_image()` requests `lang="fra+eng"` for every document, image or PDF page. Tesseract 5.4 does not abort on the unavailable `fra` in a multi-language request; it proceeds with `eng` only. `app.ocr_availability()` ([app/main.py:1361-1370](./app/main.py)) does detect this and returns `(False, "langues absentes : fra")`, but it is used for diagnostics/consumers, not as a hard gate on the read path.
2. [src/invoice_evidence.py:18-20](./src/invoice_evidence.py) `folded()` = NFKD normalization with combining marks stripped. This is accent-**insensitive comparison**, not accent **erasure**: it never rewrites the stored text.
3. [src/ocr_consensus.py:242-249](./src/ocr_consensus.py) selects the row description across families with `value=folded`, `unanimous=True`, `prefer=_accent_count`. Reads that fold to the same string vote as one candidate; among the candidates the original with the most accented characters is retained ([src/ocr_consensus.py:34-35](./src/ocr_consensus.py)). Folded disagreement yields `""` and `description_confident=False`. This logic is **already committed at `HEAD`**, unchanged by PR80; the continuation/second-line path at [src/ocr_consensus.py:264-271](./src/ocr_consensus.py) behaves the same way.
4. [src/invoice_grounding.py:453-494](./src/invoice_grounding.py) compares the model description with the OCR row description through `folded`; when they fold equal it exports `row.description` — the OCR text — with status `valide`. It never copies the model's accented or unaccented spelling into the payload.

**Conclusion:** the grounding stage is the messenger, not the cause. When the OCR stage is fed a French document with the French model absent, the correct accented string never reaches consensus, so there is nothing for the accent preference to select.

## Phase 2 — Minimal correction

Evidence supported exactly one kind of change: prove the required behaviour with deterministic tests that do not depend on the local Tesseract installation. No product code was modified, because:

- adding an accent-restoration heuristic would be a linguistic guess, explicitly forbidden ("Never invent accents using linguistic guesses or hardcoded invoice-specific substitutions");
- the authentic text is already recoverable from reliable existing evidence — the pinned `fra.traineddata` (SHA-256 `ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca`, verified in PR64/PR72) — so no fallback substitution is justified;
- the committed consensus preference already satisfies "preserve authentic diacritics when independently supported" and "do not replace reliable accented text with an inferior OCR variant merely because their accent-insensitive forms match".

### Files changed

| File | Change |
|---|---|
| [tests/test_grounding.py:18](./tests/test_grounding.py) | Import `build_consensus` so the description path is tested across families, not only inside `reconcile`. |
| [tests/test_grounding.py:212-242](./tests/test_grounding.py) | Add the `JOANNA_LAYOUT` fixture text plus `joanna_readings()` / `joanna_model()` helpers: a French row with `FACTURE FR-001`, `Joanna Binet`, `29/01/2019`, `2 × 15,00 = 30,00`. |
| [tests/test_grounding.py:245-278](./tests/test_grounding.py) | Add the four focused regressions described below. |
| [PRODUCT_REVIEW_80_JOANNA_REGRESSION.md](./PRODUCT_REVIEW_80_JOANNA_REGRESSION.md) | This report. |
| [QODER_HANDOVER.md](./QODER_HANDOVER.md) | New authoritative handover; `VSCODE_HANDOVER.md` left untouched. |

No change to `src/`, `app/`, model files, weights, dependencies, OCR thresholds, `ocr_config` policy, JSON schema, expected invoice payloads, or test expectations of existing tests. `packages.txt` and `README.md`/`requirements.txt` modifications in the tree predate PR80 and were not touched.

### The four new regressions

| Test | Requirement covered |
|---|---|
| `test_supported_accented_description_survives_an_unaccented_family_read` | Correct retention of supported French diacritics. Both family orders; consensus keeps `Petit marinière uniforme en bleu`, stays `description_confident`, and `FR-001` is still resolved in the same fixture. |
| `test_unaccented_model_echo_does_not_downgrade_the_accented_ocr_description` | Reliable accented text is not replaced by an accent-insensitive equivalent; the exported description stays `marinière` and remains `valide`. |
| `test_model_accent_without_an_accented_read_is_never_invented` | No invented diacritics. When every family reads `mariniere`, the payload keeps `mariniere` even though the model proposed `marinière`. |
| `test_conflicting_accented_reads_leave_the_description_unverified_and_keep_the_identifier` | Safe behaviour when OCR sources disagree: `bleu` vs `blond` folds differ → description `""` / not confident → exported `None` with status `non vérifié`, while `quantity`, `unit_price`, `total_ht` and `FR-001` are all preserved. |

The tests build `DocumentEvidence` from text (`from_text` → token confidence 100), so they are deterministic and independent of the local Tesseract language data. They were verified against the committed behaviour, not written to match an implementation change — no product code changed, so they pass by construction on `HEAD` logic; they failed only during authoring, before two fixture/expectation corrections: an initial fixture typo (`marini{accent}` instead of `marini{accent}re`) and one wrong status constant (`NULL`; consensus evidence relabels `NULL` → `UNVERIFIED` at [src/invoice_grounding.py:666-670](./src/invoice_grounding.py)). Both corrections are fixture/assertion accuracy fixes in the new code, not weakened expectations.

### Behaviour before and after

| Situation | Before PR80 | After PR80 |
|---|---|---|
| Joanna fixture, correct environment | Test passed, nothing asserted the reason | Same pass, now explained and re-verified with per-family evidence |
| Joanna fixture, default local environment | Fails on two missing diacritics; recorded as "unresolved OCR gap" | Fails identically, now attributed to absent `fra` with reproducible evidence |
| Accent handling in consensus/grounding | Implemented at `HEAD`, tested only in `tests/stress/test_null_recovery.py:235-257` | Unchanged, plus four focused tests inside the fallback suite |
| `FR-001` identifier extraction | 16 invoice-number/identifier tests green | Same 16 green, plus identifier assertions inside the two new diacritic tests |

## Phase 3 — Test results

Environment A is the documented PR72 local OCR environment; Environment B is the shell default. Both were measured for every suite.

```powershell
$env:TESSDATA_PREFIX = (Resolve-Path .venv\product-review64\tessdata).Path   # Environment A only
.\.venv\Scripts\python.exe -m pytest -q <suite>
```

| Suite | Environment A (`fra` loaded) | Environment B (`eng` only) |
|---|---|---|
| `tests/test_document_report.py::test_real_ocr_equivalent_invoice_with_unusual_french_layout_is_grounded` | **1 passed** in 14.54 s | **1 failed** in 12.74 s — `line_items[2]` `Facile a jouer accordéon` ≠ `Facile à jouer accordéon`; the `invoice_number == "FR-001"` assertion at line 262 passes first |
| `tests/test_grounding.py` + `tests/test_description_grounding.py` + `tests/test_ocr_text_integrity.py` | 84 passed, 1 skipped (same in B) | 84 passed, 1 skipped |
| `tests/test_document_report.py -k "invoice_number or identifier"` | 16 passed | 16 passed |
| `tests/stress/test_null_recovery.py -k "accent or description"` | 4 passed | 4 passed |
| PR79 expanded fallback suite (`test_app`, `test_grounding`, `test_real_runtime_regressions`, `test_product_review64_ocr_baseline`, `test_french_book_table`, `test_uk_layout_regressions`, `test_reliability_generalization`) | **221 passed, 34 subtests passed, 0 failed, 0 skipped** in 123.04 s | 212 passed, 9 skipped, 28 subtests passed, 0 failed in 38.53 s |
| Adjacent suite (`test_invoices`, `test_document_report`, `test_report_quality_warnings`) | **80 passed, 0 failed, 0 skipped** in 45.19 s | 77 passed, **1 failed**, 2 skipped in 27.11 s |

Reconciliation against the PR79 baseline: the fallback suite went from `208 passed, 9 skipped, 28 subtests` to `212 passed, 9 skipped, 28 subtests` in Environment B (+4 = the new PR80 tests, no other delta) and to `221 passed, 0 skipped, 34 subtests` in Environment A (+9 = the nine OCR-dependent skips now execute and pass, +6 subtests). The adjacent suite went from `74 passed, 1 failed, 2 skipped` to `77 passed, 1 failed, 2 skipped` in Environment B (+3 = identifier tests added by the PR80 predecessor) and to `80 passed, 0 failed, 0 skipped` in Environment A.

**Newly introduced failures: none.** The only failure observed anywhere in Phase 3 is the pre-existing Joanna accent test in Environment B, which passes in Environment A.

The focused new tests are OCR-independent, so they are green in both environments; that is deliberate, so that this behaviour stays covered on machines without `fra`.

## Phase 4 — Reproduction commands

```powershell
# 1. Show the defect's cause
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs          # eng, osd
$env:TESSDATA_PREFIX = (Resolve-Path .venv\product-review64\tessdata).Path
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs          # eng, fra

# 2. Joanna fixture test, both environments
.\.venv\Scripts\python.exe -m pytest -q tests\test_document_report.py::test_real_ocr_equivalent_invoice_with_unusual_french_layout_is_grounded
# with $env:TESSDATA_PREFIX cleared -> fails on accents; with it set -> passes

# 3. Focused diacritic + identifier regressions (environment independent)
.\.venv\Scripts\python.exe -m pytest -q tests\test_grounding.py tests\test_document_report.py -k "accented or accent or ligature or invoice_number or identifier"
# 22 passed, 110 deselected — verified identical in Environment A and Environment B
```

## Remaining limitations and blockers

1. **Silent language degradation is still silent on the read path.** `app.ocr_availability()` detects the missing `fra`, but `ocr_image()` does not consult it before reading. Any Windows session without `TESSDATA_PREFIX` — including this repository's own default shell — produces English-only reads of French invoices: dropped `è`/`à`, `FR-0O1`, `Net Aa payer`. Every local French benchmark recorded before this report inherits that caveat.
2. **A single-family accent can become the exported value.** `clahe` alone read `à`, and because all families fold to the same string the preference kept it. The word itself is corroborated by every family, so this satisfies "independently supported" only in the accent-insensitive sense. Worth an explicit decision, not a silent change.
3. **`Net à payer TTC` label recognition depends on the same environment.** PR79 fixed the `Net Aa payer TTC` variant, which was a degraded-OCR artifact; with `fra` loaded that variant does not appear for this fixture.
4. **No model or GPU inference was run.** No new live evidence about Qwen+LoRA accuracy. `pytest.ini` still excludes the `slow` marker, and the full repository suite was not run.
5. **No independent real-invoice validation.** The five development scans and the six frozen layouts remain dev/regression material only.

## Staging readiness

- **This defect: resolved as an environment requirement, no code change needed.** The Space installs `tesseract-ocr-fra`, so the production OCR path already has the language data this machine lacks.
- **The release: still NOT ready to deploy.** Blockers 1, 4 and 5 above are unresolved, and the required full offline suite has not been run in either environment. Deploying would not fix or worsen anything measured here.

## Next highest-priority improvement

Make the missing-language condition impossible to miss, without touching the frozen OCR policy: gate `app/main.py:ocr_image()` on `app.ocr_availability()` (or surface its result in the compact summary) so a `fra`-less environment reports degraded language coverage instead of silently producing English-only readings, and re-run the French regressions plus the PR79 fallback suite in both environments to establish the corrected baseline. Any benchmark number recorded before that gate exists should be treated as environment-suspect until reproduced with the documented `TESSDATA_PREFIX`.
