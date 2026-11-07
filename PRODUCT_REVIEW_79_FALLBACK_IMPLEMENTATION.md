# PRODUCT-REVIEW-79 — Reliable Free-Demo Fallback

**Date:** 2026-11-06  
**Scope:** Local code and offline regressions only. No model inference, GPU allocation, weight download, real-invoice upload, deployment, commit, or push.

## Result

PR79 makes the public status safer and fixes one reproduced French total-label recognition gap without changing the model, OCR thresholds, grounding policy, or JSON schema.

The public summary now calls values **cross-checked** rather than **verified**, explicitly reports quota fallback and unrecognized product lines, and treats no recognized lines as incomplete extraction. The France example's OCR line `Net Aa payer TTC : 6 860,12 [U+FFFD]` is now recognized as a labelled TTC total; the printed amount is parsed from OCR evidence, not derived from arithmetic.

The narrow fallback suite passes. **Not ready for staging yet:** an adjacent real-OCR regression still fails on the Joanna fixture's invoice number (`FR-0O1` instead of `FR-001`). No `O`-to-`0` correction was added because that would guess an identifier without sufficient evidence.

## Baseline and preserved work

At PR79 start:

- Branch: `wip/layouts-reprise`
- HEAD: `9d7a641a52c114484ee614730d15882dbaa35d66` (two commits ahead)
- Worktree: 19 tracked paths modified and 94 untracked paths already present. They were preserved; no cleanup or reversion was performed.

The exact rules and prior recommendations are in [AGENTS.md](./AGENTS.md) and [PRODUCT_REVIEW_78_PUBLIC_DEMO_RELIABILITY.md](./PRODUCT_REVIEW_78_PUBLIC_DEMO_RELIABILITY.md).

## Files changed

| File | Change |
|---|---|
| [src/invoice_evidence.py](./src/invoice_evidence.py) | Accept the narrowly observed OCR label `net aa payer ttc` as a TTC label, alongside `net a payer ttc`. Amounts still must be read from the document. |
| [app/main.py](./app/main.py) | Rename compact-summary counts to “champs recoupés”; mark no recognized product descriptions as incomplete; show an explicit OCR-only/quota explanation. |
| [tests/test_grounding.py](./tests/test_grounding.py) | Cover the French TTC OCR label, preservation of `œ`, rejection of U+FFFD descriptions, and an internally consistent but OCR-unsupported model proposal. |
| [tests/test_app.py](./tests/test_app.py) | Update UI wording expectations and cover quota/no-line messages. |
| [tests/test_real_runtime_regressions.py](./tests/test_real_runtime_regressions.py) | Update the compact-summary wording expectation. |
| [tests/test_reliability_generalization.py](./tests/test_reliability_generalization.py) | Update the compact-summary wording expectation. |
| [PRODUCT_REVIEW_79_FALLBACK_IMPLEMENTATION.md](./PRODUCT_REVIEW_79_FALLBACK_IMPLEMENTATION.md) | This implementation and validation report. |

No schema, model, weight, dependency, OCR-threshold, or deployment-configuration changes were made.

## Defects corrected and behavior

### French TTC label

Before: the current France OCR evidence contained `Net Aa payer TTC` and the value `6 860,12`, but the strict total-label parser did not recognize the label. The public-example regression returned `total_ttc: null` despite the printed TTC line.

After: the parser accepts the single observed repeated-`a` OCR variant. The France example now retains the labelled `total_ttc` from OCR evidence, and its existing regression passes. No arithmetic calculation was added to produce a missing monetary value.

### OCR corruption and product descriptions

The saved PR64 OCR reads already contain U+FFFD in `Main-d'...uvre`, before reconciliation. That locates the corruption in the OCR text/reader stage, not output formatting or grounding. The exact subcause—Tesseract recognition versus subprocess decoding—remains unproven.

The existing grounding path rejects replacement-character descriptions. New tests verify that:

- a valid French `œ` remains unchanged in an OCR-only result;
- a description containing U+FFFD is not exported and does not create a product row;
- the resulting empty-line condition is surfaced to the user.

The original glyph cannot safely be reconstructed from U+FFFD, so PR79 does not guess or silently replace it.

### Microspective and other fallback safeguards

The current worktree already contains the narrow contextual `RATE / PRICE` header handling; the saved-header regression passes. The captured Microspective tokens still do not support reliable row reconstruction, so PR79 adds no geometric or quantity/amount recovery. The UI now explicitly says when no product lines were recognized and marks the result incomplete.

Existing quota handling remains intact. The UI states when OCR-only mode was used and gives a quota-specific explanation. Tests also verify that a model proposal whose quantity and unit price multiply to a plausible printed total is not retained when those operands are unsupported by OCR; the OCR-anchored values remain authoritative. Ambiguous values remain null.

The JSON contract and expected invoice payloads were not changed.

## Regression results

### Before changes

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_app.py tests\test_grounding.py tests\test_real_runtime_regressions.py tests\test_product_review64_ocr_baseline.py tests\test_french_book_table.py tests\test_uk_layout_regressions.py
```

Result: **166 passed, 1 failed, 9 skipped, 27 subtests passed**. The failure was the France public example: `total_ttc` was `null` instead of the printed `6860.12`. The saved/current OCR text showed the unmatched `Net Aa payer TTC` label; PR79 adds a focused regression for this case.

### After changes

Focused PR79 checks: **6 passed**. One initial assertion expected a parenthetical quota phrase; it was corrected to assert the two message components independently, then rerun successfully.

Expanded fallback suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_app.py tests\test_grounding.py tests\test_real_runtime_regressions.py tests\test_product_review64_ocr_baseline.py tests\test_french_book_table.py tests\test_uk_layout_regressions.py tests\test_reliability_generalization.py
```

Result: **208 passed, 9 skipped, 28 subtests passed, 0 failed.** This includes the France, UK, and US examples, quota fallback, ambiguous-quantity and monetary-grounding controls, accent/replacement-character cases, and saved Microspective regressions. Tests used local OCR and mocked/synthetic model behavior; no model or GPU inference ran.

Adjacent invoice/report checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_invoices.py tests\test_document_report.py tests\test_report_quality_warnings.py
```

Result: **74 passed, 1 failed, 2 skipped.** The failure is `test_real_ocr_equivalent_invoice_with_unusual_french_layout_is_grounded`: current OCR reads the Joanna fixture's `FR-001` as `FR-0O1`, so `invoice_number` is not accepted. The PR79 diff does not modify invoice-number recognition; no blind character substitution was added.

The full repository suite and all `slow` real-model tests were not run. `pytest.ini` excludes slow tests by default.

## Remaining limitations and staging decision

- The OCR-only path cannot recover the original `œ` once U+FFFD is present; it now rejects the affected description rather than exporting corrupted text.
- Microspective still has no reliably reconstructed product rows from the saved evidence. The UI clearly reports this as incomplete.
- The Joanna invoice-number OCR failure remains unresolved and requires an evidence-based decision before staging.
- No new live model evidence was collected. The fallback improvements do not establish accuracy on independent real invoices.

**Staging readiness: NO.** The targeted PR79 fallback suite is green, but the adjacent Joanna OCR test is failing, and the full repository suite has not been run. Do not deploy until the identifier failure is either corrected with evidence and regression coverage or explicitly accepted, and the required offline suite is green.

## Next highest-priority improvement

Triage the Joanna `FR-0O1` recognition using its independent OCR reads and identifier-token evidence. Preserve `null` if the intended `0` cannot be distinguished reliably; do not normalize `O` to `0` globally.
