# Final Public Deployment Report

## Verdict: DEPLOYED - public Space running and essential UI checks passed

The PR54 artifact was deployed once with the PR55 exact-parent/manifest helper.
The complete remote tree, runtime, browser UI, normal API surface, and three
example file hashes passed verification. No rollback was needed.

## Revision and operations

- **Previous public revision:** `5a20d99f31eef764b02429721c447b0b5ca8afcf`
- **New public revision:** `ac2913303c7f6401c0a27688027bacdc0ae69a61`
- **Candidate tree SHA-256:** `f471658db2e246b41b96e95beb181c3d423bb24703a64718adfa5e26f653df56`
- **Operations:** 9 additions, 10 replacements, 64 deletions; 10 common files
  unchanged. The Hub payload had 19 `file` additions/replacements and 64
  deletions.
- **Exact operation plan:** [.venv/final-public-release/review54/PUBLIC_DEPLOYMENT_PLAN.json](.venv/final-public-release/review54/PUBLIC_DEPLOYMENT_PLAN.json)
- **29-file manifest:** [.venv/final-public-release/review54/PUBLIC_RELEASE_MANIFEST.json](.venv/final-public-release/review54/PUBLIC_RELEASE_MANIFEST.json)
- **Verified 84-file backup:** [.venv/final-public-release/review54/PUBLIC_BACKUP_MANIFEST.json](.venv/final-public-release/review54/PUBLIC_BACKUP_MANIFEST.json)
- **Backup ZIP SHA-256:** `8f74466dc8b684fdecd6936c3fc670b91b35657564c8de3551205789e4223c51`

Preupload used the one-commit ignore override. The serialized payload contained
every approved Add/Delete with matching hashes, including all operations the
SDK might otherwise ignore. The post-commit Hub tree has 29 files and matches
the candidate Git blobs exactly.

## Build, runtime, and UI

- Hugging Face build completed; build log contained **0 error lines**.
- Space reached **RUNNING** at the new revision. Runtime metadata reports
  `zero-a10g`, unchanged from before deployment. Public visibility and the empty
  public variable set are unchanged.
- Root and `/config`: **HTTP 200**.
- Browser UI rendered the upload control, three regional example buttons,
  extraction button, strict JSON display, and JSON download control.
- Public API names are exactly `load_example`, `preview_ui`, and
  `run_extraction_ui`; no private diagnostic/capture API was present.
- France, UK, and US example file requests returned HTTP 200 and matched their
  manifest SHA-256 values.
- Tesseract startup reports English and French available (`eng`, `fra`, `osd`).
- No invoice extraction callback, Qwen inference, or GPU inference request was
  invoked. The extraction control was not clicked.

### Automatic model prefetch observed

The application’s **normal startup prefetch** ran because no production
variables were changed. Startup logs show it downloaded approximately **131 MB**
of adapter/tokenizer files and **2.37 GB** of base-model files. No explicit
download command was run, no model was instantiated, and no inference occurred.
This did not prevent startup, so the documented rollback condition was not met.
The existing prefetch behavior and mutable model references remain an
operational consideration for future restarts.

Other non-blocking diagnostics:

- Startup emitted three ignored asyncio destructor tracebacks with
  `ValueError: Invalid file descriptor: -1`, plus a Gradio hot-reload notice.
  The process continued, served HTTP 200, and reached RUNNING.
- Browser console recorded one unidentified 404 resource; essential UI/config
  loaded and rendered.
- Hosted dependency resolution selected `huggingface_hub 1.33.0` from the
  unbounded requirement; local PR54 Linux testing used 2.4.0. The hosted build
  and startup succeeded with the resolved hosted version.

## Security, integrity, staging, and rollback

- All **29 candidate files** match their manifest; all **84 original backup
  files** and hashes were verified before deployment.
- The new public tree exactly matches the PR54 manifest.
- Private staging remains private/RUNNING at
  `03bc5441a651718ddbbaee9f0d0cf94622bb35f3`; its full tree and recorded
  variable-value digest are unchanged.
- Frozen/reference hashes remain **87/87 unchanged**.
- No private diagnostics were exposed. No public variables or hardware settings
  were changed; no paid hardware was enabled.
- **Rollback:** not needed.
- **No hosted invoice extraction, Qwen inference, model retraining, Git commit,
  or push occurred.**

Machine-readable post-deployment checks:
[FINAL_DEPLOYMENT_VERIFICATION.json](.venv/final-public-release/review54/FINAL_DEPLOYMENT_VERIFICATION.json).
Captured build/startup logs:
[HOSTED_BUILD_LOGS.txt](.venv/final-public-release/review54/HOSTED_BUILD_LOGS.txt),
[HOSTED_RUNTIME_LOGS.txt](.venv/final-public-release/review54/HOSTED_RUNTIME_LOGS.txt).

## Handover

- Local branch: `wip/layouts-reprise`; HEAD:
  `9d7a641a52c114484ee614730d15882dbaa35d66`.
- Existing dirty worktree preserved; no local commit or push.
- The single authorized public commit is verified; no further deployment or
  rollback operation was attempted.
