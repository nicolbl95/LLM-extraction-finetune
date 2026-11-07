# PRODUCT-REVIEW-51 — Public Release Engineering

## Executive decision

**CONDITIONAL GO — prepared for a separately authorized, monitored release; not deployed.**

The original public `BUILD_ERROR` remains **unattributed** because the historical
build logs were not retained and the Hub log endpoint only exposes current
buffers. The old candidate's exact tree is available, all of its local imports
and UI construction pass the clean smoke, its Python 3.12 dependency resolver
dry-run succeeds on Windows, and the previous rollback issue is independently
fixed in the new deployment helper. However, the Linux Hugging Face build was
not reproduced: the Docker daemon is unavailable, and a Windows pip dry-run is
not a Linux container build. Do not treat this report as deployment approval.

No remote write, deployment, variable/hardware change, hosted inference, GPU
allocation, model-weight download, retraining, Git commit, or push occurred.

## 1. Original public `BUILD_ERROR`

### Confirmed evidence

- The candidate commit was `2169cf0d7f7d3ba4c8db8c085db35bd78d2f3565`; its
  Hub tree is exactly **29 files**.
- The saved deployment state records `BUILD_ERROR` for that revision, but
  contains no builder traceback or first actionable error.
- Later public commits replaced the active build/runtime log buffers. The
  installed `huggingface_hub` 2.3.0 `fetch_space_logs` API has no revision
  parameter; it reads `/api/spaces/{repo}/logs/{build|run}` for the current
  buffers. The original revision's logs are therefore unavailable through the
  supported API. See the [API reference](https://huggingface.co/docs/huggingface_hub/v2.3.0/en/package_reference/hf_api#huggingface_hub.HfApi.fetch_space_logs)
  and installed [`hf_api.py`](.venv/Lib/site-packages/huggingface_hub/hf_api.py).
- **Original phase and first actionable error:** only the Space stage
  `BUILD_ERROR` is recorded. No evidence identifies whether the Docker build,
  Python dependency install, Space metadata processing, or app startup failed.
  The first actionable error is not recoverable.
- The later `aa009...` rollback's
  `FileNotFoundError: /app/tests/fixtures/layout-joanna-binet.png` is a
  **separate rollback failure**, not evidence for the original candidate
  `2169cf0...` failure. Candidate `app/main.py` exposed only
  `REGIONAL_EXAMPLES` in Gradio; the missing fixture was not part of that
  startup example list.
- The original candidate's Space metadata uses Gradio 6.30.0 and Python 3.12,
  matching the successful private staging README metadata. That differs from
  the old public runtime's implicit Python 3.10.13. Candidate/staging dependency
  manifests differ intentionally; staging also has a private-only diagnostic
  registration. The candidate's exact file tree, requirements, and Space
  metadata are preserved in the manifests below.

### Offline reconstruction

- The 29-file candidate contains the app entry point, all locally imported
  runtime modules, `data/schema.py`, the three displayed invoice images,
  `packages.txt`, `requirements.txt`, `constraints-hf.txt`, and Gradio Space
  README metadata. The clean candidate smoke resolved its local import closure
  entirely within the candidate.
- The candidate import/UI/strict-JSON-download smoke passed with model loading,
  `snapshot_download`, and OCR calls blocked. It is a **local smoke**, not a
  hosted build.
- A Python 3.12 Windows pip `--dry-run --ignore-installed` resolved the
  candidate requirements plus the recorded Gradio 6.30.0 / spaces 0.51.3 build
  pins, including Torch 2.13.0. No packages or model weights were installed.
  This does not resolve Linux-only markers.
- A Linux cross-platform pip dry-run could not select `hf-transfer` under the
  forced `manylinux`/binary-only wheel filter. This is a limitation of that
  cross-target check, not evidence that the actual Hugging Face build fails.
- Docker is installed but its Linux daemon is unavailable
  (`dockerDesktopLinuxEngine` pipe not found), so a local container build could
  not be run.
- No production logic change is attributed to the old `BUILD_ERROR`. The only
  app change here is a separate, evidence-backed packaged-path guard: the
  candidate removes optional example fixture files, while `checked_path`
  previously hashed those absent paths before checking existence. The guard
  skips absent optional examples and retains the existing expired-upload
  `ValueError`. It does not change OCR, extraction, grounding, or field policy.

## 2. Candidate packaging and configuration

Fresh candidate directory:
`.venv/final-public-release/review51/candidate/`

Manifest: [`PUBLIC_RELEASE_MANIFEST.json`](.venv/final-public-release/review51/PUBLIC_RELEASE_MANIFEST.json)  
Tree SHA-256: `09019e32234231c9cbe49b26334de0a2599682e55c0cc3a458f2ad6c73279a4`  
File count: **29**  
Current public parent: `5a20d99f31eef764b02429721c447b0b5ca8afcf`

All Python files parse; the candidate is an exact allowlist with no test,
diagnostic, scan, experiment, database, cache, log, secret-shaped string, or
model-weight file. It includes the three regional examples and schema. Optional
legacy example path constants remain guarded against absent files.

| Candidate file | SHA-256 |
|---|---|
| `README.md` | `915c624fe39bccf7a51d9f299dbe4a72328c5b15834bb2a2a2a0c4e095e73dd3` |
| `app/examples/france.png` | `6b4909f8428cad4fa726e8eb2299d8fc9e0f9419e8e9a93a77cb1f75aabf5671` |
| `app/examples/uk.png` | `0d03a8b55fb89d0a59696059eefe45e49c4640643fb3c5fe28c492a9e448f57e` |
| `app/examples/us.png` | `5a7b9bc7e1faef3f9570e3bfccd6626d583f66da189609b9ca00482466b27ae4` |
| `app/main.py` | `3a54fec4bd0a68e47fdf2a0f8e3bfde97f5f2a5e4ccd1979e74bfdc1591adb79` |
| `constraints-hf.txt` | `785b55b861541f122bd0d0dc6bf99580483e9435c72e9dfb865256f02c25ce8c` |
| `data/schema.py` | `e12e3686abfe99b0801a4510ac36a0c5001a16bb6d7d3e54fb932bf2d474021a` |
| `packages.txt` | `4232b5f5a82e195878fefb0c2758cf3f061e2572a1308317125acb5540d5c942` |
| `requirements-inference.txt` | `f6e272cb88db9d5c0965886c12f80254f0fbf15ec001d2d3ca934e2a27b94d65` |
| `requirements.txt` | `20dbcddfbdd657d086dad208cbaabeb33e6e13d113bb7d8df7344d7c97451ee3` |
| `src/evaluate_baseline.py` | `cf975fa23ce9d2c7b7cbe75844d5a2d23cd2a665599f3810b8b3fecfcc508ba5` |
| `src/evaluate_finetuned.py` | `1b053b28074c4923fc7540e1fe755faa97ec4b6a9e8f42199891080ecef74f06` |
| `src/invoice_evidence.py` | `4cece73deb1a6f96d38a1340f93c5caa502e3d575b55a4f045f33f0ad34c2182` |
| `src/invoice_grounding.py` | `a86df0a3818d7c5b51918e467a50877c7767e262dc531b03cbeb7cc0fa6ab7c1` |
| `src/invoice_review.py` | `7e90a03dcb6486b717b3054551639f7506fa00030799ae8f298d108fef9867c2` |
| `src/invoice_review_edit.py` | `7f8d5e4a7b79e5ddf845648e11ba5605e9c53938dded4480b7a84570fcc3ad4a` |
| `src/invoice_review_store.py` | `32a062e6318e1df3fbec846d290765c0166a1d021ec9075ab9617393f4d3d381` |
| `src/invoice_validation.py` | `91dcc415c74d589d9f5461056cfe69584a9792c18cd83867f56ea31197fe4853` |
| `src/ocr_budget.py` | `a800e922e8366859baa8ea07451be6bedf86cc8a10581d0ab85ecba94177a690` |
| `src/ocr_config.py` | `5e2b796558b08c89b0c2aea9dcfdd6d11df2e37f93ee8169811ba2d1e30f4ae6` |
| `src/ocr_consensus.py` | `04b06cae1d093aa2f5ae41f3708d8a147a841c0580dde1f67d8cfe6790a1dc81` |
| `src/ocr_fast_path.py` | `048520d9c8c56ff12aca210c68662e190533f54f3925d5606ecdcc3fed5d3ed6` |
| `src/ocr_preprocessing.py` | `8040aaf5a009fbb862696ec841ee57c790836678850c486096e3044bbfb04c49` |
| `src/ocr_quality.py` | `93ead821b87fc2f017a036168e28d23f58fcb2d4959db64d7b2d1cf5143d6217` |
| `src/ocr_recovery.py` | `2bd0d21140339e96616ef2972c8d6c7f372095e5275bd7f6dedfd76fb4b6ccdc` |
| `src/table_geometry.py` | `dad45fdd498c806270028d44440dc8de0f9fd0cdb661bd1abef3bad9b3735355` |
| `src/table_reconstruction.py` | `4f741a6d0f96d5fd7751901157fa21ef71aa0c9e8fa7c62ab743f4ace9fb4027` |
| `src/table_recovery.py` | `f6d76a36234133b3695e612f6222902fc31800c2726f224c73966aae67733f4e` |
| `src/table_shadow.py` | `d2375ceca92f540578141ba1a465fc0989c5c4f61f32e9552200ad196ffaaebc` |

### Runtime and staging comparison

- Candidate and staging Space metadata both specify Gradio 6.30.0 and Python
  3.12 on `zero-a10g`. The candidate moves the old public Space from its prior
  Python 3.10 default to the Python 3.12 configuration already used in staging.
- Candidate/staging package and requirements files differ intentionally; the
  candidate pins selected staging-tested OCR/UI libraries. Full transitive
  dependency versions (including `spaces`, `datasets`, and several inference
  libraries) are not locked.
- The candidate does not include model weights. `app.main.startup_prefetch()`
  may download adapter/base artifacts when the Space environment enables it;
  that path was not executed in this task.

## 3. Git-ignore and commit/rollback reliability

The original public `.gitignore` contains `*.png`. The previous rollback's
`CommitOperationAdd` set included that `.gitignore`; the preupload service used
its content, set `_should_ignore`, and the SDK omitted the four PNGs. The next
four-add commit used the remote `.gitignore` and serialized a header-only
payload. The commits returned SHAs, but tree diffs showed no additions. This was
an SDK preupload-ignore behavior, not an LFS pointer problem; original PNG
metadata reported `lfs=None`.

The new [deploy_release.py](.venv/final-public-release/review51/deploy_release.py):

1. Builds operations only from the SHA-256 allowlist.
2. Verifies all 29 candidate and 84 backup files before any write.
3. Calls `HfApi.preupload_lfs_files` with a one-off comment-only
   `gitignore_content`; it does not change `.gitignore`.
4. Requires every Add to have `_should_ignore is False` and a supported upload
   mode.
5. Serializes with the installed client and checks the exact parent, paths,
   upload modes, and content hashes before `create_commit`.
6. Rechecks the branch parent immediately before the single commit.
7. Compares the complete post-commit remote tree against the manifest; a commit
   SHA alone is never accepted as success.
8. Applies the same preupload, payload, parent, and whole-tree checks to
   rollback. It never retries and does not automatically make a second write.

The [local release/rollback simulation](.venv/final-public-release/review51/ROLLBACK_SIMULATION.json)
reconstructs the verified 84-file baseline, applies 9 adds/11 replacements/64
deletions, then simulates restoration with 75 Adds and 9 Deletes. Default
`.gitignore` handling drops exactly the four PNG Adds (71 of 75 are serialized);
an ignored-only recovery serializes only a commit header. With the override,
all 75 Adds and 9 Deletes are serialized and the full 84-file tree round-trips
byte-for-byte. This simulation makes no Hub write.

The current-parent backup and operation plan are
[PUBLIC_BACKUP_MANIFEST.json](.venv/final-public-release/review51/PUBLIC_BACKUP_MANIFEST.json)
and [PUBLIC_DEPLOYMENT_PLAN.json](.venv/final-public-release/review51/PUBLIC_DEPLOYMENT_PLAN.json).
Current public revision `5a20d99f31eef764b02429721c447b0b5ca8afcf` was
rechecked read-only: public/RUNNING, 84/84 blobs match the backup, variables
empty, requested hardware unchanged. Private staging remains
`03bc5441a651718ddbbaee9f0d0cf94622bb35f3`, private/RUNNING.

## 4. Regression and safety results

| Gate | Result |
|---|---|
| Focused candidate-pruned-path test | **1 passed** |
| Full offline suite | **730 passed, 0 failed, 0 skipped; 3 slow tests deselected; 120 subtests** |
| Full-suite OCR source | 448 recorded replays, **0 fresh OCR calls** |
| Nine-invoice matrix + ordinary Cedar control | Prior payloads unchanged; 83 recorded reads, 0 fresh |
| Cedar 369.07 normal/altered control | **2 passed**; printed 369.07 preserved, no arithmetic substitution |
| Mirada saved-response replay | Pass; saved 5/16 result and three rows preserved; no shipping; no OCR/model calls |
| Stress | **38 cases / 114 simulated model-error profiles**; wrong values 0, phantom anomalies 0, crashes 0; false nulls 161/648 unchanged; 255 recorded reads, 0 fresh |
| Frozen/reference integrity | **87/87 hashes match** |
| Clean public candidate import/UI/JSON-download smoke | Pass; 29 files, 3 examples/normal endpoints; model, weights, and OCR calls blocked |
| Current public backup and candidate manifests | 84/84 backup blobs and 29/29 candidate hashes verified |

The first full run exposed an existing timing race in
`test_cache_protection_does_not_retain_new_uploads_or_json`: Gradio cleanup
uses strict `file_age > age`, and the test invoked `age=0` immediately after
creating files. The test now waits 1.1 seconds so both generated files are
eligible for deletion; the isolated test and final suite pass. Production cache
behavior was not changed by this test adjustment.

No frozen expectations or invoice policy changed. The new path test and app
guard only cover removed optional example assets; extraction, OCR consensus,
grounding, schema, and model behavior were not modified.

## 5. Remaining risks and recommendation

- **Original public `BUILD_ERROR` cause remains unresolved.** There is no
  archived builder log for revision `2169cf0...`; the Hub logs API has no
  revision selector and the active buffers were replaced by later builds.
  Do not attribute the later rollback's missing-fixture `RUNTIME_ERROR` to
  that earlier build.
- Docker is installed, but the Docker Desktop Linux daemon is unavailable.
  The Windows Python 3.12 dependency dry-run resolved the recorded SDK pins, but
  it is not a Linux build. The cross-platform pip attempt was blocked by the
  forced `hf-transfer` wheel tag filter, not an observed Hugging Face build
  error.
- The candidate requirements are not a complete transitive Linux lock; mutable
  model `main` references and the app's optional startup prefetch remain
  operational assumptions. No weights or model files were downloaded.
- The public release remains **not deployed**. The current public Space remains
  at restored revision `5a20d99f31eef764b02429721c447b0b5ca8afcf`.

**Recommendation: CONDITIONAL GO.** The artifact, simulation, and local checks
are ready for a separately authorized, one-time controlled deployment using
the new helper. Because the historical `BUILD_ERROR` cannot be reconstructed
and no Linux container build was possible, the deployment must stop on any
build/startup failure; capture the fresh logs, verify the exact file tree, and
do not make speculative retries or fixes without new authorization.

## Handover

- Branch `wip/layouts-reprise`; HEAD `9d7a641a52c114484ee614730d15882dbaa35d66`.
- Existing dirty work preserved. Local source changes are limited to the
  optional-example existence guard and its test; no Git commit or push.
- Remote public state and private staging were read-only checked; no deployment,
  variable/hardware changes, hosted inference, GPU allocation, or model
  downloads occurred.
- Future action: obtain explicit deployment authorization, rerun helper
  `--preflight`, then use `--deploy` once. If anything fails, stop and inspect;
  use `--rollback-parent <actual-new-sha>` only under separately authorized
  rollback procedure.
