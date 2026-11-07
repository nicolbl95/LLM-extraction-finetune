# PRODUCT-REVIEW-55 - Final PR54 Deployment Readiness

## Verdict: GO for separately authorized guarded deployment

Preparation and local simulation are complete. **No deployment, preupload,
remote write, inference, model download, hardware/variable change, Git commit,
or push occurred.** The current public application remains unchanged.

## Exact artifact and live parent

| Item | Verified value |
|---|---|
| Approved candidate | `.venv/final-public-release/review54/candidate/` |
| Candidate files | 29; every size and SHA-256 verified |
| Candidate tree SHA-256 | `f471658db2e246b41b96e95beb181c3d423bb24703a64718adfa5e26f653df56` |
| Release manifest SHA-256 | `8ba72c30d2be742ffec12a233ebdadffce8b6c52fb389146a4018b6ce04bcaab` |
| Current public parent | `5a20d99f31eef764b02429721c447b0b5ca8afcf` |
| Current public status | Public, RUNNING; requested `zero-a10g`; variables empty |
| Current public backup | 84/84 live Git blobs match verified backup bytes |
| Backup archive SHA-256 | `8f74466dc8b684fdecd6936c3fc670b91b35657564c8de3551205789e4223c51` |
| Private staging | Private/RUNNING, `03bc5441a651718ddbbaee9f0d0cf94622bb35f3`; complete 62-file tree unchanged |
| Frozen references | 87/87 unchanged |

The PR54 requirements retain the validated pins inline; no sibling constraints
include is needed during the isolated HF installation step. Its system package
list is LF-only. Both corrected files match their exact release hashes.

### Complete manifests

- [Release inventory: all 29 sizes and SHA-256 hashes](.venv/final-public-release/review54/PUBLIC_RELEASE_MANIFEST.json)
- [Full release and rollback operations with old/new hashes](.venv/final-public-release/review54/PUBLIC_DEPLOYMENT_PLAN.json)
- [Complete verified public backup manifest](.venv/final-public-release/review54/PUBLIC_BACKUP_MANIFEST.json)
- [Backup ZIP](.venv/final-public-release/review54/backup/public-5a20d99-84-files.zip)
- [Read-only live preflight result](.venv/final-public-release/review54/PR55_PREFLIGHT.json)

The operation-plan SHA-256 is
`3a2879df500d769c84cecc3fc19da428b7db141ed675041379aaffb5acbb318a`.

## Recalculated operation counts

| Direction | Additions | Replacements | Deletions | Unchanged |
|---|---:|---:|---:|---:|
| Current public to PR54 | 9 | 10 | 64 | 10 |
| PR54 back to original public | 64 | 10 | 9 | 10 |

The Hub represents additions and replacements as `CommitOperationAdd`:

- **Release:** 19 Add + 64 Delete = **83 operations**, producing 29 exact files.
- **Rollback:** 74 Add + 9 Delete = **83 operations**, restoring 84 exact files.

The old 11-replacement/75-path-restore plan is obsolete. PR54's LF-only
`packages.txt` matches the original public bytes and needs no replacement.
All deletion paths are individually listed and backed up; no local source file
is deleted.

## Successful Linux evidence

The saved [PR54 Linux result](.venv/final-public-release/review54/LINUX_RESULT.json)
is PASS and identifies this exact candidate tree. Evidence-file hashes are
bound into the deployment plan and rechecked by the helper.

- Real local Linux/amd64, Debian Bookworm, CPython 3.12.16.
- Gradio 6.30.0, spaces 0.51.3, Torch 2.13.0; `pip check` passed.
- Original entry point started; root and `/config` returned HTTP 200.
- Tesseract 5.3.0 with `eng`, `fra`, `osd`.
- Three standard examples delivered over HTTP with exact hashes; JSON download
  generation and UI configuration passed.
- Networking disabled during startup verification; no models loaded or inferred.

See [PR54 report](PRODUCT_REVIEW_54_LINUX_BUILD_RESULT.md),
[successful build log](.venv/final-public-release/review54/corrected-linux-build-lf.log),
[startup log](.venv/final-public-release/review54/linux-startup.log), and
[installed versions](.venv/final-public-release/review54/linux-pip-freeze.txt).
This evidence was verified, not rerun or relabeled as a hosted/ZeroGPU test.

## Canonical helper and omission protections

Use only
[review54/deploy_release.py](.venv/final-public-release/review54/deploy_release.py).
It has fixed PR54 candidate/manifest paths and pinned hashes for the artifact,
operation plan, backup manifest, and staging tree. It cannot accept the PR51
artifact. Both legacy CLI entry points now exit with an explicit obsolete-release
error before performing any action.

The helper:

1. Verifies all candidate and backup bytes, SHA-256 values, Git blob IDs, exact
   path inventories, Linux evidence hashes, and all 87 frozen references.
2. Derives the complete forward/reverse diffs and requires exact equality with
   the approved operation plan.
3. Before a separately authorized write, uses a comment-only
   `preupload_lfs_files(gitignore_content=..., free_memory=False)` override for
   only the explicit manifest Add objects. It never modifies public `.gitignore`.
4. Rejects unchecked/ignored Adds, unsupported upload modes, and Adds the SDK
   would remove as unchanged.
5. Serializes with the installed Hub SDK and verifies the exact parent,
   operation multiplicities, regular-file content hashes, and LFS SHA-256/size
   records. No missing/duplicate/extra path is tolerated.
6. Rechecks the live parent after preupload, then submits one exact-parent commit.
   An attempt record prevents blind reuse. Ambiguous writes are recorded as
   uncertain and are never retried automatically.
7. Verifies the returned revision is the live head and compares the complete
   resulting remote tree against the expected manifest. A returned SHA is not
   sufficient.

The helper deliberately reports `complete_tree_verified_startup_pending` after
delivery. **That status is not a successful deployment verdict.** Hosted build,
runtime, logs, frontend/API, and cache checks must then pass as described below.
No automatic compensating write is made.

## Local release/rollback validation

[Simulation receipt](.venv/final-public-release/review54/PR55_SIMULATION.json):
all serialized release and rollback paths and hashes verified; forward/reverse
trees match every byte, SHA-256, and Git blob ID.

The four formerly ignored PNGs are explicitly present in the rollback payload:

- `app/examples/modele-vierge.png`
- `results/loss_curve.png`
- `tests/fixtures/layout-joanna-binet.png`
- `tests/fixtures/us-line3-amount-error.png`

An ignored Add is a failing negative control, not a silently omitted operation.
The rollback serializer uses an explicitly labeled offline placeholder for the
future deployment SHA; the remote helper rejects that placeholder.

**14 offline helper tests passed**, covering round-trip integrity, refusal of
the PR51 manifest, unchecked/ignored Adds, unsupported modes, SDK no-op removal,
missing/duplicate Adds, corrupt content, missing Deletes, LFS hash/size checks,
refusal without exact-plan authorization, and disabled legacy entry points.

Commands executed:

```powershell
.venv\Scripts\python.exe -m pytest -q .venv\final-public-release\review54\test_deploy_release.py --basetemp .venv\final-public-release\review54\pr55-test-tmp
.venv\Scripts\python.exe .venv\final-public-release\review54\deploy_release.py --simulate
.venv\Scripts\python.exe .venv\final-public-release\review54\deploy_release.py --preflight
```

Simulation is offline: no preupload/Hub API calls. Live preflight uses read-only
metadata and file-tree endpoints. No full invoice regression suite was rerun for
these deployment-only changes.

## Final release instructions - NOT EXECUTED

Only after explicit approval of the PR54 artifact and exact 9/10/64 operation set:

1. Re-run the commands above. Stop if the public parent, visibility, settings,
   complete backup parity, candidate hashes, or staging tree differ. Do not
   rebase the plan silently.
2. Use the canonical helper with `--deploy` and the exact operation-plan hash:

   ```powershell
   .venv\Scripts\python.exe .venv\final-public-release\review54\deploy_release.py --deploy --authorize-plan-sha256 3a2879df500d769c84cecc3fc19da428b7db141ed675041379aaffb5acbb318a
   ```

3. Preserve the attempt receipt and actual new revision. If the commit call is
   ambiguous, inspect the branch read-only; never repeat it based on a timeout.
4. Immediately archive sanitized build/runtime logs labeled with the actual new
   revision, before any possible rollback replaces the current log buffers.
5. Confirm the new revision builds and reaches RUNNING. Use runtime metadata's
   raw revision when supplied, not a nonexistent `SpaceRuntime.sha` attribute;
   verify live head and startup revision evidence as well.
6. Verify public root/config HTTP 200 and rendered frontend; upload control,
   exactly three regional examples, extraction button, compact summary, strict
   JSON and download controls; only the three normal API names. No private
   diagnostic endpoint may appear.
7. Verify three original example file hashes, then retain their original
   Gradio URLs and verify them again after at least 1,230 seconds without first
   reselecting them. Do not confuse Hub source-file access with cache survival.
8. Reconfirm public visibility, empty variables, unchanged requested hardware,
   exact 29-file inventory, and unchanged staging tree/settings. Do not invoke
   invoice extraction, Qwen, or any GPU request.
9. Declare success only after every hosted acceptance gate passes. On failure,
   stop, retain logs and exact state, and follow only a separately authorized
   guarded rollback; no speculative fixes/redeployments.

## Final rollback instructions - NOT EXECUTED

If rollback is separately authorized:

1. Require the exact live head to equal the **actual new PR54 deployment SHA**.
   The helper also requires its tree to match the PR54 candidate. If the tree
   is unexpectedly partial or another writer has changed it, stop for a new
   explicit recovery plan rather than guessing additional operations.
2. Reverify the 84-file archive and operation plan. Run:

   ```powershell
   .venv\Scripts\python.exe .venv\final-public-release\review54\deploy_release.py --rollback-parent <ACTUAL_NEW_PR54_SHA> --authorize-plan-sha256 3a2879df500d769c84cecc3fc19da428b7db141ed675041379aaffb5acbb318a
   ```

3. The same ignore override, serialized hash checks and exact parent guard apply.
   Restore 74 old paths; delete only the 9 PR54 additions.
4. Require the full restored 84-file tree to match the backup; then verify the
   restored revision builds/starts and its original UI/examples are accessible.
   Archive logs and recheck visibility/settings/staging. No inference.
5. If rollback is ambiguous or fails, stop and report the exact remaining
   revision/inventory/runtime. Do not retry automatically.

## Remaining operational conditions

There is **no demonstrated local packaging or file-delivery blocker** remaining.
Explicit public-deployment approval and actual hosted acceptance checks are still
required. The local Linux test does not certify ZeroGPU inference/CUDA behavior
or the hosted builder's full transitive dependency environment.

The existing production model/prefetch configuration is preserved, not modified
by this task. Its normal startup prefetch can fetch missing model artifacts; the
offline Linux test disabled it. If future deployment authorization also forbids
startup weight fetching, obtain separate approval for suitable runtime controls
before deployment; do not silently change Space variables.

Model references and transitive dependency ranges remain mutable as previously
disclosed. No fresh model test or hosted inference is claimed.

## Handover

- Local branch: `wip/layouts-reprise`; HEAD:
  `9d7a641a52c114484ee614730d15882dbaa35d66`.
- Existing dirty work preserved; no application/candidate/frozen files changed.
- Added canonical PR54 helper/tests, bound backup/operation manifests, simulation/
  read-only verification records, and this report. Disabled the two obsolete
  release CLI entry points; their historical manifests/artifacts are preserved.
- Validation: 14 helper tests passed; exact release/rollback simulation passed;
  29 candidate hashes, 84-file backup parity and 87 frozen hashes passed;
  public/staging remain RUNNING at their prior revisions.
- Next: await explicit authorization for the corrected PR54 public deployment.
  Do not execute any write under PRODUCT-REVIEW-55.
