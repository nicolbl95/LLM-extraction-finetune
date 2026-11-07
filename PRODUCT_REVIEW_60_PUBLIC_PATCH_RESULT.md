# PRODUCT-REVIEW-60 — Public UI Safety Patch Result

**Verdict: PASS.** The authorized one-file patch was committed and verified on the public Space. No extraction callback, Qwen inference, explicit model download, hardware/variable change, private staging write, local Git commit, or push was performed.

## Deployment

- **Target:** `nicolbl95/LLM-extraction-finetune`
- **Previous revision:** `ac2913303c7f6401c0a27688027bacdc0ae69a61`
- **New revision:** `927a115997ef2f850255e1b14cfe37fe861bb710`
- **Commit parent:** exactly the verified previous revision above
- **Operation:** one replacement, `app/main.py`; zero additions and zero deletions
- **Old app SHA-256:** `3a54fec4bd0a68e47fdf2a0f8e3bfde97f5f2a5e4ccd1979e74bfdc1591adb79`
- **New app SHA-256:** `423a9565b0f2a1cb3fd766c1983c08461e175b40ddbb0b6beb3acdcbfac3fb5d`
- **PR58 candidate tree SHA-256:** `81ec1010015f10030ec7712dc09657a87b354f61e5142ff9ea0306dbfcb58477`

The Hugging Face SDK payload was inspected before submission. It contained the exact parent commit and one `app/main.py` file operation whose bytes matched the authorized candidate hash. No delete, extra path, or ignored operation was present. The exact commit SHA was recorded in [.venv/product-review58/DEPLOYMENT_RECEIPT.json](./.venv/product-review58/DEPLOYMENT_RECEIPT.json).

## Preflight and remote integrity

- Public parent, public visibility, `RUNNING` state, `zero-a10g` hardware, and empty public variables matched preflight.
- All 29 PR54 parent Git blobs matched the immutable release manifest.
- The candidate contained exactly 29 files, and only `app/main.py` differed from PR54.
- The corrected rollback archive SHA-256 was `71895925e49da55d840cd9a47cdc7893efc9cefbbc028adf4e9abcc21c4de0ba`. It contains all 29 original PR54 files, including the original `app/main.py`; each file’s SHA-256 and Git blob were checked.
- Private staging remains at `03bc5441a651718ddbbaee9f0d0cf94622bb35f3`, private and `RUNNING`, with its expected 62-file tree, variable hash, and `zero-a10g` hardware.
- All 87 frozen reference hashes passed immediately before the write.

After deployment, the current public revision matched the new commit and the full 29-file remote tree matched the PR58 candidate. Comparing the old and new remote trees showed exactly one changed path: `app/main.py`.

## Build, runtime, and UI

- Public Space reached **RUNNING** on the unchanged `zero-a10g` tier.
- Root and `/config` returned **HTTP 200**.
- Configuration includes the upload control, extraction button, JSON display, JSON download control, and three examples.
- France, UK, and US example files each returned **HTTP 200** and matched their expected SHA-256.
- Public configuration contained no detected private diagnostic markers or internal host paths.
- Staging remained unchanged after deployment.
- **No extraction callback was invoked.**

The build log read completed (29 lines) with no critical build failure. Runtime log inspection found three nonfatal `Exception ignored` event-loop destructor tracebacks (`ValueError: Invalid file descriptor: -1`) during Gradio hot reload. The application proceeded to RUNNING and served both health endpoints; these warnings did not meet the rollback criteria.

The application’s normal startup automatically prefetched the adapter/tokenizer (about 131 MB) and base-model files (about 2.37 GB). This was not an explicit model-download request or inference call by this task, but it is recorded here as an automatic startup behavior. No Qwen inference or GPU allocation was requested.

## Regression and recovery

- Focused UI safety tests: **4 passed, 49 deselected**.
- No rollback was needed.
- The corrected PR54 rollback archive remains intact. Rollback, if ever separately required, must use the exact new revision `927a115997ef2f850255e1b14cfe37fe861bb710` as `parent_commit` and proceed only if the remote tree still matches the PR58 candidate. No rollback operation was issued.

## Handover

The PR58 rollback defect was corrected before this deployment; the updated readiness report is [PRODUCT_REVIEW_59_PATCH_READINESS.md](./PRODUCT_REVIEW_59_PATCH_READINESS.md). Current public revision is `927a115997ef2f850255e1b14cfe37fe861bb710`. No private staging or local Git history was changed. The dirty worktree was preserved; no local commit or push occurred.
