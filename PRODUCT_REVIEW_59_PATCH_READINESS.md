# PRODUCT-REVIEW-59 — PR58 Rollback Correction and Patch Readiness

**Date:** 2026-11-05  
**Scope:** Correct the local PR58 rollback package, verify parent-guarded one-file release/restore simulations, and perform the requested bounded read-only preflight. No public/private Space writes, inference, model downloads, GPU allocation, Git commits, or pushes occurred.

## Backup defect confirmed

The original PR58 rollback ZIP was incorrectly created from `public-patch-candidate`, not the immutable PR54 production candidate. Its `app/main.py` SHA-256 was `423a9565b0f2a1cb3fd766c1983c08461e175b40ddbb0b6beb3acdcbfac3fb5d`, identical to the patched release file.

| Version | `app/main.py` SHA-256 |
|---|---|
| Original deployed PR54 / current public app | `3a54fec4bd0a68e47fdf2a0f8e3bfde97f5f2a5e4ccd1979e74bfdc1591adb79` |
| Patched PR58 candidate | `423a9565b0f2a1cb3fd766c1983c08461e175b40ddbb0b6beb3acdcbfac3fb5d` |
| App in superseded PR58 ZIP | `423a9565b0f2a1cb3fd766c1983c08461e175b40ddbb0b6beb3acdcbfac3fb5d` |

This confirms the suspected rollback error.

## Corrected rollback package

The ZIP was rebuilt from `.venv/final-public-release/review54/candidate/` only. It now contains exactly the original PR54 29-file production tree:

- Every entry’s path, size, SHA-256, and Git blob SHA-1 match the immutable PR54 manifest/source.
- The archived `app/main.py` hash is `3a54fec4bd0a68e47fdf2a0f8e3bfde97f5f2a5e4ccd1979e74bfdc1591adb79`, the original deployed app, not the patch.
- The corrected archive SHA-256 is **`71895925e49da55d840cd9a47cdc7893efc9cefbbc028adf4e9abcc21c4de0ba`**.
- The patched PR58 candidate remains unchanged: app SHA-256 `423a9565b0f2a1cb3fd766c1983c08461e175b40ddbb0b6beb3acdcbfac3fb5d`; 29-file tree SHA-256 `81ec1010015f10030ec7712dc09657a87b354f61e5142ff9ea0306dbfcb58477`.
- The release manifest still specifies only **one replacement**, `app/main.py`; zero additions and zero deletions.

Updated local artifacts:

- [PR58_PUBLIC_PATCH_MANIFEST.json](./.venv/product-review58/PR58_PUBLIC_PATCH_MANIFEST.json)
- [PUBLIC_AC291_ROLLBACK_BACKUP_MANIFEST.json](./.venv/product-review58/rollback-backup/PUBLIC_AC291_ROLLBACK_BACKUP_MANIFEST.json)
- [Corrected 29-file rollback ZIP](./.venv/product-review58/rollback-backup/public-ac2913303c7f6401c0a27688027bacdc0ae69a61-29-files.zip)

## Release and rollback simulation

The offline simulation used the exact file set and parent guards from the manifest:

1. **Original state:** 29 PR54 file hashes and Git blobs, tree `f471658db2e246b41b96e95beb181c3d423bb24703a64718adfa5e26f653df56`.
2. **Release:** requires current parent `ac2913303c7f6401c0a27688027bacdc0ae69a61`; exactly one replacement of `app/main.py` yields PR58 tree `81ec1010015f10030ec7712dc09657a87b354f61e5142ff9ea0306dbfcb58477`. A wrong release parent was rejected.
3. **Rollback:** requires the exact PR59 release commit as `parent_commit` and the full remote tree to match the PR58 candidate. The original app bytes are read from the corrected archive; exactly one `app/main.py` replacement restores the PR54 tree and **29/29** original SHA-256 and Git blob values. A wrong rollback parent was rejected.
4. The Hugging Face SDK’s `_prepare_commit_payload` serialized the release with `parentCommit=ac2913303c7f6401c0a27688027bacdc0ae69a61` and the patched file hash. The rollback payload was verified with an offline-only placeholder parent; that placeholder must never be used remotely. No preupload or `create_commit` call was made.

## Final read-only checks

- Public revision remains `ac2913303c7f6401c0a27688027bacdc0ae69a61`, public, `RUNNING`, `zero-a10g`; all 29 remote Git blobs match the original PR54 tree; public variables remain empty.
- Private staging remains at `03bc5441a651718ddbbaee9f0d0cf94622bb35f3`, private and `RUNNING`; its 62-file tree hash, variable hash, and `zero-a10g` hardware match the recorded baseline.
- Focused UI regressions: **4 passed, 49 deselected**.
- Frozen references: **87/87 SHA-256 hashes match**.
- No extraction, OCR/model accuracy, benchmark, or full regression audit was repeated.

## Final verdict

**GO — the one-file patch and its corrected rollback are ready for a separately authorized deployment.** This is a readiness verdict only, not deployment authorization. Immediately before any future write, recheck the public parent and full 29-file tree; if either differs from this manifest, stop. Rollback must use the exact PR59 returned commit as its parent guard and the corrected PR54 archive.
