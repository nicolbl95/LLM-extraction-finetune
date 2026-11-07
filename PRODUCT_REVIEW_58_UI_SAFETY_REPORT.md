# PRODUCT-REVIEW-58 — UI Safety Validation

**Date:** 2026-11-05  
**Scope:** Read-only public/staging verification, local UI error handling, offline regression validation, and local-only release preparation. No Space writes, inference, model downloads, commits, or pushes were performed.

## Verdict

**CONDITIONAL GO** for the prepared one-file public patch. Both fixes pass focused tests and a corrected-candidate Linux startup/UI smoke; a matching rollback backup and parent/tree guards are prepared. **Do not deploy without separate explicit authorization and a fresh preflight.**

## Causes and verified fixes

1. **Malformed-image preview path disclosure.** `preview_document` returned `str(exc)` to the UI. Pillow’s `UnidentifiedImageError` includes the upload’s temporary path, so an invalid image could expose a server path.
   - Preview errors now expose only allowlisted safe validation messages or a generic safe reason.
   - Server diagnostics retain exception class and useful detail after Windows/POSIX absolute paths are redacted. A 240-character bound prevents oversized log entries.
2. **JSON download failure escaped the Gradio stream.** `_download_json` ran outside an error boundary in `run_extraction_ui`; an `OSError` aborted the generator after extraction.
   - The callback now catches only `OSError`, logs sanitized diagnostics, keeps the strict JSON result visible, and returns no download file with an explicit unavailable warning. The recruiter-facing summary carries the same warning.
   - Successful JSON serialization and download output are unchanged.

The PR54-to-PR58 candidate diff is restricted to the preview error path, a path-sanitizing diagnostic helper, JSON-download error handling, and its status propagation. No extraction, grounding, OCR thresholds, or model-inference logic changed. `run_recruiter_extraction_ui` creates diagnostics per invocation; an interleaved-session regression confirms one session’s result/download state does not carry into another.

Focused regressions in [tests/test_app.py](./tests/test_app.py) verify that malformed-image and download-failure paths do not expose paths in UI text or logs, that the extracted payload survives a failed download, that a successful download matches the strict payload, and that interleaved sessions remain separate.

## Validation results

| Gate | Result |
|---|---|
| Focused tests on the dirty local source tree (Windows) | **4 passed, 49 deselected** |
| Focused tests against the isolated PR58 candidate (Linux) | **4 passed, 49 deselected** |
| Full offline suite on a Linux copy of the dirty local source tree | **751 passed, 6 skipped, 3 deselected, 120 passing subtests** in 523.05 seconds |
| Nine-invoice recorded matrix | **9 established cases plus ordinary Cedar control unchanged**; 83 recorded OCR reads, 0 fresh reads |
| Cedar altered/control check | Altered printed line amount **369.07** retained with quantity/unit price null and mismatch warning; ordinary control retains **396.07** with no mismatch |
| Mirada saved-response replay | **PASS**; exact captured five verified fields, three documentary rows, no unsupported shipping, no new OCR/model calls |
| Stress replay | **38 cases / 648 fields; 114 simulated model-error profiles**; wrong values 0, phantom anomalies 0, crashes 0; 161 false nulls unchanged from baseline; 255 recorded OCR reads, 0 fresh |
| Frozen references | **87/87 SHA-256 hashes match** |
| Immutable PR54 candidate smoke | **PASS**; exact 29-file manifest, imports, Gradio `/` and `/config`, three example hashes, strict JSON download generation |
| Corrected PR58 candidate smoke | **PASS**; exact 29-file candidate, imports, Gradio `/` and `/config`, three example hashes, strict JSON download generation |
| Registered Gradio streaming callback fault injection | **PASS**; callback returned the JSON result, safe warning, and `None` download without raising or exposing the injected path |

The full-suite run preceded the final test-only interleaved-session assertion; that added test passed in both focused runs above. The suite was run in an ephemeral, network-disabled copy with its test prerequisites (`.pytest_cache`, temporary Git metadata, and the test-only adapter-repository setting) recreated. An initial copy missing those prerequisites reported 17 harness/environment failures; the corrected run passed. No source or test expectations were weakened.

### Environment

| Environment | Versions / status |
|---|---|
| Windows local `.venv` | Python 3.12.0; Tesseract 5.4.0; languages `eng`, `osd` (no `fra`) |
| Existing offline Linux image | Python 3.12.16; pytest 9.1.1; Gradio 6.30.0; Tesseract 5.3.0 with `eng`, `fra`, `osd`; `pip check` passed |
| Linux inference-package imports | Torch 2.13.0, Transformers 5.22.1, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2; imports only, no model instance |
| Model/network safeguards | Docker `--network none`; Hub/Transformers offline; prefetch and preload disabled; no safetensors in the test cache; slow-model tests deselected |

Thus the Windows OCR limitation is confirmed, but it does not block this baseline: the full suite and candidate smoke used the existing Linux image with French traineddata. No language data was installed.

## Public and staging preflight

Read-only Hub verification confirmed:

- Public Space is at `ac2913303c7f6401c0a27688027bacdc0ae69a61`, public, `RUNNING`, `zero-a10g`, and has no public variables.
- All **29/29** public Git blobs exactly match the immutable PR54 candidate (`f471658db2e246b41b96e95beb181c3d423bb24703a64718adfa5e26f653df56`).
- Staging remains private at `03bc5441a651718ddbbaee9f0d0cf94622bb35f3`, `RUNNING`, `zero-a10g`, with 62 files. Its tree hash `a5fd50f72acb9fa4fdeb0398f02f720091584a1fe8265701e165bdb3cd8905f5` and variable hash `77c807fee1039952f3c0aeb8b74e4fce73d929a0b92bef3d3dafe50605af37e6` match the recorded baseline.
- The prior 84-file backup is for the older `5a20d99…` parent and is not the rollback target for this one-file patch. A new exact 29-file backup of the current public PR54 tree was created and verified.

## Prepared release and rollback

The immutable PR54 candidate remains untouched. A separate local PR58 candidate was derived from those 29 files with only `app/main.py` replaced:

| Artifact | SHA-256 |
|---|---|
| PR54 deployed tree | `f471658db2e246b41b96e95beb181c3d423bb24703a64718adfa5e26f653df56` |
| PR54 `app/main.py` | `3a54fec4bd0a68e47fdf2a0f8e3bfde97f5f2a5e4ccd1979e74bfdc1591adb79` |
| PR58 `app/main.py` | `423a9565b0f2a1cb3fd766c1983c08461e175b40ddbb0b6beb3acdcbfac3fb5d` |
| PR58 candidate tree | `81ec1010015f10030ec7712dc09657a87b354f61e5142ff9ea0306dbfcb58477` |
| Corrected current-public rollback archive (29 files; PR59) | `71895925e49da55d840cd9a47cdc7893efc9cefbbc028adf4e9abcc21c4de0ba` |

The proposed manifest is [PR58_PUBLIC_PATCH_MANIFEST.json](./.venv/product-review58/PR58_PUBLIC_PATCH_MANIFEST.json). Its exact operation set is:

- **Additions:** 0
- **Replacements:** 1 — `app/main.py`
- **Deletions:** 0

PR59 found that the first PR58 archive had been sourced from the patched candidate and therefore was not a valid rollback. Its superseded archive SHA-256 was `a48c80ab2396fd0349fc55c9b099d44d044e40c755a4f661d0264065064b8d7d`. The corrected archive is now built from the immutable PR54 candidate and has SHA-256 `71895925e49da55d840cd9a47cdc7893efc9cefbbc028adf4e9abcc21c4de0ba`. The exact current-tree backup is [PUBLIC_AC291_ROLLBACK_BACKUP_MANIFEST.json](./.venv/product-review58/rollback-backup/PUBLIC_AC291_ROLLBACK_BACKUP_MANIFEST.json), with archive [public-ac2913303c7f6401c0a27688027bacdc0ae69a61-29-files.zip](./.venv/product-review58/rollback-backup/public-ac2913303c7f6401c0a27688027bacdc0ae69a61-29-files.zip). All 29 archive entries match their original PR54 SHA-256 and Git blob hashes; the archived app hash is the deployed PR54 value. Local release/rollback simulation changed only `app/main.py` and restored all 29 original file hashes.

The manifest requires the public parent to still equal `ac2913303c7f6401c0a27688027bacdc0ae69a61` immediately before any future write. Rollback is permitted only if the remote is still at the exact PR58 commit and its full tree matches the PR58 candidate; otherwise stop. The inverse operation restores the backed-up PR54 `app/main.py`. No deployment was performed.

## Remaining limitations and handover

- The public Space still runs the original PR54 application and therefore has not received these fixes. This report does not authorize deployment.
- The Windows OCR installation still lacks French traineddata; use the verified Linux environment for French-dependent regression until that local setup is separately addressed.
- The full suite was not rerun after the final test-only interleaved-session assertion; that assertion passed independently on both local and candidate environments.
- Repository branch is `wip/layouts-reprise`, HEAD is `9d7a641a52c114484ee614730d15882dbaa35d66`, and the worktree remains dirty with pre-existing changes preserved. No commit, push, or remote write occurred.

**Next step:** obtain separate deployment authorization. Immediately before a future write, repeat the parent, 29-blob, public runtime/settings, staging revision/tree/variable, and rollback-backup guards in the manifest.
