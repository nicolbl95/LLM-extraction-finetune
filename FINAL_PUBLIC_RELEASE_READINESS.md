# Final Public Release Readiness

## Verdict: CONDITIONAL GO

The **29-file public runtime candidate passes offline, safety, import and UI
smoke checks** and contains the validated runtime changes through PRODUCT-REVIEW-49.
No demonstrated application-code defect blocks a controlled deployment.

**Do not deploy under this audit.** Before a future public write, obtain:

1. Explicit owner approval for the exact **64 deletions** from the current public
   84-file research repo (datasets, notebooks, docs, results, scripts and tests
   are among the existing public files being removed in favor of a 29-file runtime
   allowlist).
2. A decision about the mutable model revisions: pin to revisions proven to be
   the ones used for validation, or explicitly accept that adapter/base `main`
   references can move. Historical loaded model commits are not proven, so blindly
   pinning to today’s metadata SHAs would also be an unverified behavior change.
3. Separate deployment authorization. The current public Space is sleeping, so
   the candidate cannot be build/startup tested on that exact Space without a
   write/wake operation. After a later authorized deployment, perform the
   post-deploy checks below.

No source or Space changes were made for this report. No hosted inference, GPU
allocation, model-weight download, variable/hardware operation, commit, push or
deployment occurred.

## Verified state

| Target | Verified revision | Visibility/state | Hardware |
|---|---|---|---|
| Public `nicolbl95/LLM-extraction-finetune` | `f00fce7abf4fdc5f9fd8ad1f932adf3be736c023` | Public, SLEEPING | Requested `zero-a10g`; no hardware allocated |
| Private staging `nicolbl95/LLM-extraction-finetune-staging` | `03bc5441a651718ddbbaee9f0d0cf94622bb35f3` | Private, RUNNING at matching SHA | `zero-a10g` |
| Local public candidate | 29 files, tree SHA-256 `b019f78d6c409bffda2fceab23870f233de34db8c82a0298a08534763f81c395` | Built in an isolated allowlisted folder; not deployed | No inference run |

The private staging tree has **23 candidate paths byte-identical** to the public
candidate. The five staging differences are its approved private diagnostic
registration plus README/package/requirements variants; `constraints-hf.txt` is
the candidate’s release-only constraint file. There are **no missing functional
fixes and no unexplained staging drift**. The private validation/capture helper
is absent from the public candidate.

The local branch is `wip/layouts-reprise`, HEAD
`9d7a641a52c114484ee614730d15882dbaa35d66`. Existing dirty work is preserved.

## Exact candidate, public parent, and backup

Fresh read-only snapshot of all **84 files** at public parent
`f00fce7abf4fdc5f9fd8ad1f932adf3be736c023`:

- Every snapshot file’s size and Git blob were checked against Hub metadata.
- Every archived SHA-256, ZIP inventory and CRC were verified.
- No model-weight-like files were present; no weights were downloaded.
- Rollback ZIP SHA-256:
  `8f74466dc8b684fdecd6936c3fc670b91b35657564c8de3551205789e4223c51`.

The **29-file release manifest** contains a SHA-256 and size for every candidate
file. Exact current-public-to-candidate diff:

| Operation | Count |
|---|---:|
| Additions | 9 |
| Replacements | 11 |
| Deletions | 64 |
| Unchanged common files | 9 |

All exact paths and per-file old/new hashes are in
[PUBLIC_BACKUP_MANIFEST.json](.venv/final-public-release/PUBLIC_BACKUP_MANIFEST.json);
the proposed parent-guarded operations and rollback are in
[PUBLIC_DEPLOYMENT_PLAN.json](.venv/final-public-release/PUBLIC_DEPLOYMENT_PLAN.json).
Candidate SHA-256 inventory is in
[PUBLIC_RELEASE_MANIFEST.json](.venv/final-public-release/PUBLIC_RELEASE_MANIFEST.json).

The 64 removals are existing **public** development/research content, including
training/test JSONL datasets, notebooks, docs, evaluation/results files, scripts,
test suites and fixtures. The existing public tree has no paths named
`real-scans`, `diagnostics` or `private_staging`; the release candidate physically
excludes all tests/fixtures, diagnostics, caches, databases, logs and private
diagnostic callbacks. The current public tree does contain synthetic test
fixtures and research assets; those are included in the proposed deletions.
Because this is a substantial repository-scope reduction, it requires explicit
owner confirmation before any future write.

### Intended changes represented by the candidate

The candidate contains the validated current app and runtime modules for:

- strict document-grounded JSON and printed monetary-value fidelity;
- French split table headers, table reconstruction, and Mirada handling;
- conservative unknown-currency presentation;
- standard example cache retention;
- OCR-only fallback and concise, accurate model/pass/performance reporting;
- the compact recruiter-facing UI and strict JSON download.

The nine additions are `constraints-hf.txt`, review helper modules, OCR fast-path
and table geometry/reconstruction/recovery/shadow modules. The 11 replacements
are the README, `app/main.py`, package/requirements files, and the validated
evidence, grounding, validation, consensus, OCR-quality and OCR-recovery modules.
Use the manifest for the complete exact file list.

## Security, privacy and API surface

All 29 candidate files were SHA-256 checked. Python files parse. A bounded scan
found:

- no credential-shaped tokens in allowlisted text/config files;
- no `private_staging_validation` marker or private diagnostic endpoint;
- no model-weight files, fixture directories, scans, database, cache or logs.

Candidate UI smoke exposes only `load_example`, `preview_ui` and
`run_extraction_ui`; there are three standard examples and no private diagnostics.
Normal API/UI has no staging response-capture mechanism. The scan is not a
general-purpose security certification.

The adapter repository `nicolbl95/llm-extraction-weights` and base repository
`unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` are currently public and ungated.
No credentials are required by the observed public metadata.

## Runtime and compatibility

### Configuration and GPU

- Public Space variables: **none configured**.
- `GRADIO_SSR_MODE` is unset publicly. Gradio **6.30.0** resolves an unset value
  to `False`, so its effective behavior is the requested
  `GRADIO_SSR_MODE=false`; no variable change is required.
- Candidate README pins Space SDK Gradio **6.30.0** and Python **3.12**.
- Public Space requests `zero-a10g`; current public runtime is SLEEPING with no
  allocation. The private staging runtime is RUNNING on `zero-a10g`.
- Official read-only quota query returned **9.524590 free seconds**, **0
  overquota seconds**, reset at `2026-11-05T20:10:45Z`. This is not sufficient
  for meaningful hosted Qwen testing. No allocation was made.

### Dependencies and OCR

Candidate constraints pin Gradio 6.30.0, Pillow 12.3.0, pytesseract 0.3.13,
Linux PyMuPDF 2.1, Linux OpenCV 4.14.0.*, and Linux NumPy 2.5.3. Requirements
constrain Torch to 2.8–2.13, Transformers 4.49–<6, PEFT >=0.14, Accelerate >=1.2
and Linux bitsandbytes >=0.48. `spaces` and other transitive dependencies are
not fully pinned; `constraints-hf.txt` explicitly is **not a complete Linux
inference lock**.

Candidate `packages.txt` includes Tesseract plus English/French language
packages. Last available public startup logs report Tesseract 5.3.0 with
`eng,fra,osd`; the latest staging startup logs report Tesseract 5.5.0 with the
same languages. Public current logs are historical because the Space is
SLEEPING; the candidate’s exact new public build has not been started.

### Model references and mutable revisions

Current read-only metadata:

| Repository | Current metadata SHA | Visibility |
|---|---|---|
| `nicolbl95/llm-extraction-weights` | `a1e56cc8c313cb194395047332cd8df2a0e34e63` | Public, ungated |
| `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit` | `21d7790b1332196a5cf65ba72f2b46b659f74ddf` | Public, ungated |

The adapter lookup defaults to mutable `main`; the adapter config’s base revision
is null. These metadata SHAs are **not proof** that the earlier successful
staging runs loaded those exact commits. Weight references and the Qwen+LoRA/NF4
architecture were not changed. Before claiming byte-for-byte model
reproducibility, establish/pin the previously validated commits or explicitly
accept mutable model revisions. No model file was downloaded.

### Logs and fallback

Sanitized read-only log checks found no recognized quota-exceeded, GPU-abort,
OOM or timeout markers in available public/staging logs. The public log snapshot
contains two `ValueError` tracebacks whose causes cannot be established from the
saved evidence; they are not enough to claim a current-runtime error or a clean
build. The public worker is sleeping.

The offline suite exercises `GPU task aborted` and quota-failure classification,
ensuring the OCR-only result does not report successful generation or accept
unsupported model values. The public candidate smoke blocks model loading,
weight download and OCR; JSON/download parity uses a simulated model proposal,
not a real inference.

## Final regression results

| Check | Result |
|---|---|
| Full offline suite, serial run | **729 passed, 0 failed, 0 skipped; 3 slow tests deselected; 120 subtests passed** |
| OCR during full suite | 448 recorded OCR replays; 0 fresh |
| Nine-invoice matrix + ordinary Cedar control | Prior strict payloads unchanged; 83 recorded reads, 0 fresh |
| Cedar 369.07 altered-image control | **2 passed** (normal example and altered printed amount); wrong arithmetic replacement remains rejected |
| Mirada captured-response replay | Pass; saved 5/16 strict result retained; no shipping row/quantity; no new model/OCR calls |
| Stress | **38 cases / 114 simulated error profiles**; wrong values 0, phantom anomalies 0, crashes 0; false-null rate 161/648 unchanged |
| Frozen/reference hashes | **87/87 unchanged** |
| Clean public candidate import/UI smoke | Pass: 29 files, 3 normal endpoints/examples, strict JSON/download parity, model blocked |
| Public backup/release inventory | 84/84 current files backed up/verified; candidate 29/29 hashes verified |

The full suite is the authoritative final result. No test expectations or frozen
targets were changed. No hosted inference was run.

## Required variable changes

**None.** Current public variable set is empty. `GRADIO_SSR_MODE` defaults
effectively to false under the pinned Gradio 6.30.0; adapter repo/revision
defaults are already public and no secret is needed. Hardware remains
`zero-a10g` as currently requested; there is no hardware operation in the plan.

## Deployment and rollback plan (not executed)

If separately authorized after the conditions below:

1. Re-read the live public revision and visibility. Abort unless still public at
   parent `f00fce7abf4fdc5f9fd8ad1f932adf3be736c023`.
2. Verify all 29 candidate hashes and the 84-file backup ZIP hash.
3. Use `HfApi.create_commit(repo_type="space", revision="main",
   parent_commit="f00fce7abf4fdc5f9fd8ad1f932adf3be736c023",
   create_pr=False)` with exactly 9 add, 11 replace and 64 delete operations from
   the manifest. Do not retry on parent mismatch; do not overwrite any path not
   in the plan.
4. Verify new revision is RUNNING; effective variables/SSR; expected hardware;
   UI/config/three examples and cache lifetime; private diagnostic routes absent;
   OCR startup reports `fra+eng`; strict JSON/download parity; Qwen status
   fallback without submitting an invoice; exact remote file inventory.
5. On failure, verify the live SHA equals the actual new commit. Make a
   compensating parent-guarded commit that restores the **75** replaced/deleted
   old paths from the verified backup and deletes the **9** candidate additions.
   Never rewrite history or touch variables/hardware.

The fresh public rollback archive is
[public-current-file-backup.zip](.venv/final-public-release/backup/public-current-file-backup.zip);
its SHA-256 is
`8f74466dc8b684fdecd6936c3fc670b91b35657564c8de3551205789e4223c51`.

## Conditions and final action

This is **CONDITIONAL GO**, not a deployment authorization. Before an authorized
write, the owner must:

1. Confirm the 29-file runtime allowlist and explicitly accept removal of the
   64 existing public research/docs/dataset/script/test paths. The verified
   rollback backup protects every old byte.
2. Decide whether mutable adapter/base `main` references are acceptable for this
   demo. For reproducible model behavior, identify the historically tested
   commits before pinning; the present metadata SHAs alone are insufficient.
3. Obtain separate deployment authorization. The public Space is currently
   sleeping, so its candidate build/startup cannot be verified without a future
   write/start. Do not wake or deploy it during this task.

If those conditions are not accepted, **do not deploy**. Otherwise use the exact
parent-guarded plan and rollback above; no further broad audit is recommended.

## AI-to-AI handover

1. **State:** public parent `f00fce7abf4fdc5f9fd8ad1f932adf3be736c023`,
   public/SLEEPING; staging `03bc5441a651718ddbbaee9f0d0cf94622bb35f3`,
   private/RUNNING. No remote writes. Local branch `wip/layouts-reprise`,
   HEAD `9d7a641a52c114484ee614730d15882dbaa35d66`; dirty work preserved.
2. **Actions:** verified exact public and staging inventories; fresh 84-file
   public backup; candidate hashes/security scan; dependency/model/runtime
   checks; final offline regressions and clean candidate smoke.
3. **Files created:** `FINAL_PUBLIC_RELEASE_READINESS.md` plus ignored
   `.venv/final-public-release` manifests, backup, diff plan and verification logs.
   No source/test files changed.
4. **Validation:** 729 full tests/120 subtests/3 deselected; nine matrix unchanged;
   Cedar2 pass; Mirada saved replay pass; stress38/114 zero wrong/anomaly/crash;
   frozen87; clean candidate pass; backup84 and manifest29 SHA checks pass.
5. **Blockers/conditions:** owner decision on 64 deletions; mutable model revision
   accept/pin choice; public sleeping prevents candidate runtime verification
   until an authorized deployment/start.
6. **Next:** obtain those decisions and separate deployment authorization, then
   execute only the parent-guarded manifest plan. No GPU inference or deployment
   was performed here.
