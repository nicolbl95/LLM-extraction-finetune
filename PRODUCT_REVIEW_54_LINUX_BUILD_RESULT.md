# PRODUCT-REVIEW-54 - Local Linux Build Verification

## Verdict

**PASS: the corrected 29-file candidate built and started in a genuine local
Linux container. The original immutable PR51 candidate failed installation.**

Two release-packaging defects were reproduced and corrected without changing
application code, extraction, OCR, grounding, schema, model references, or
dependency version limits. No Hugging Face operation was performed.

The local Linux build/startup gate is now satisfied for the **PR54** candidate.
This is not a successful hosted build, ZeroGPU test, or deployment authorization.
The old PR51 artifact must not be deployed.

## Environment and isolation

- Docker Desktop Linux daemon verified first with `docker info`: Docker 29.6.1,
  Linux/amd64, WSL2 kernel `6.18.33.1-microsoft-standard-WSL2`.
- Base: Debian Bookworm, CPython **3.12.16**, GCC 12.2.0, glibc 2.36.
- Pinned base image:
  `python:3.12-bookworm@sha256:1bc6cb484acf8d7b0393d547ab55c3f147d641bef8adafbf7dfec5ba4d2d99e0`.
- Successful image: `invoice-pr54-verified:local`, ID
  `sha256:70661284ea3143512a679db53ed4e2b14fca45777971c755d1e144094f9412a5`.
- Runtime container used `--network none`, read-only root filesystem, a temporary
  `/tmp`, 2 CPUs, 2 GiB memory, no GPU devices, no credentials, and no existing
  model cache. It was removed with `--rm` after the application was terminated.
- Model prefetch/preload disabled; Hub/Transformers offline; adapter repository
  empty; device CPU; analytics disabled; `GRADIO_SSR_MODE=false`.
- Build networking downloaded Linux/system/Python packages, including the normal
  Torch CUDA dependencies. These are software libraries, not model weights.
  No model weights, model initialization, GPU allocation, or inference occurred.
- The verified image remains local, approximately 11.56 GB. No shared Docker
  resources were pruned or unrelated containers stopped.

## Reproduced defects and minimal corrections

### 1. Sibling constraints file missing during HF-style installation

The original exact PR51 candidate passed all 29 manifest hashes before testing.
The Dockerfile reproduced the observed HF dependency-install pattern: mount
only `requirements.txt` at `/tmp/requirements.txt`, install, then copy app files.

First actionable error:

```text
ERROR: Could not open requirements file: [Errno 2] No such file or directory:
'/tmp/constraints-hf.txt'
```

The candidate's `-c constraints-hf.txt` directive resolves relative to the mounted
requirements file. The sibling constraints file is not present at that build
stage, even though it exists in the repository.

**Correction:** inline the same six validated constraints in `requirements.txt`.
All original package requirements/ranges and platform markers remain in force;
no dependency threshold was loosened. The original constraints file remains in
the 29-file artifact for provenance.

### 2. CRLF package list breaks Linux xargs/apt

After Python dependency installation passed, apt reported:

```text
E: Unable to locate package tesseract-ocr
E: Unable to locate package tesseract-ocr-fra
E: Unable to locate package tesseract-ocr-eng
```

The original candidate contained:

```text
b'tesseract-ocr\r\ntesseract-ocr-fra\r\ntesseract-ocr-eng\r\n'
```

Linux `xargs` preserves those carriage returns in the package arguments. The
saved successful staging package list already used LF-only bytes.

**Correction:** normalize `packages.txt` to LF, preserving the exact three
package names. apt installation then passed.

These are confirmed current Linux reproductions of release blockers. Because
the original hosted failure logs are unavailable, they do not prove which error
was encountered first in historical public revision `2169cf0...`.

## Build and startup results

| Check | Result |
|---|---|
| Original immutable PR51 candidate | FAIL: missing `/tmp/constraints-hf.txt` |
| Corrected Python install under isolated requirements mount | PASS |
| LF-only package list / apt install | PASS |
| Corrected Docker image build/export | PASS, exit 0 |
| `pip check` | PASS: no broken requirements |
| Every active Linux candidate requirement | Installed version satisfies specifier |
| Torch, Transformers, PEFT, Accelerate, bitsandbytes imports | PASS; no model instantiated |
| Actual entry point `python app/main.py` | Started successfully |
| Gradio root and `/config` | HTTP 200 / HTTP 200 |
| Gradio SSR | Disabled |
| Upload, extraction button, JSON display/download controls | Present |
| API set | `load_example`, `preview_ui`, `run_extraction_ui` only |
| Three regional example controls | Present |
| Three example HTTP file deliveries | HTTP 200, exact original SHA-256 values |
| Example cache cleanup protection | Files survive cleanup |
| JSON download generation | PASS using an all-null strict payload; no extraction |
| OCR system packages | Tesseract 5.3.0; `eng`, `fra`, `osd` available |
| Corrected candidate hashes after startup | 29/29 PASS |
| Original PR51 artifact | 29/29 unchanged |
| Frozen/reference hashes | 87/87 unchanged |
| Focused packaging regressions | 3 passed |

Observed installed versions:

| Package | Version |
|---|---|
| Gradio / Gradio client | 6.30.0 / 2.7.2 |
| spaces / Torch | 0.51.3 / 2.13.0 |
| Transformers / PEFT | 5.22.1 / 0.21.2 |
| Accelerate / bitsandbytes | 1.15.0 / 0.50.2 |
| Pillow / PyMuPDF | 12.3.0 / 2.1 |
| pytesseract | 0.3.13 |
| OpenCV wheel / NumPy | 4.14.0.94 / 2.5.3 |
| huggingface_hub | 2.4.0 |

Startup logs explicitly report model prefetch disabled and OCR available.
The only application warning was unavailable Git revision metadata: the
allowlisted container deliberately contains no `.git` directory or injected
commit SHA. No startup exception or frontend initialization failure occurred.
No invoice extraction request was submitted.

The first corrected install attempt was interrupted before completion and was
not counted as a pass. A bounded retry with a persistent pip download cache
completed dependency installation. Windows editing initially retained CRLF;
the new byte-level regression caught this before the final LF-normalized build.

## Artifact identity and changed files

Original PR51 tree (preserved):
`09019e32234231c9cbbe49b26334de0a2599682e55c0cc3a458f2ad6c73279a4`.

Corrected PR54 tree:
`f471658db2e246b41b96e95beb181c3d423bb24703a64718adfa5e26f653df56`.

Corrected artifact:
[candidate/](.venv/final-public-release/review54/candidate/).
Exact file sizes/hashes:
[PUBLIC_RELEASE_MANIFEST.json](.venv/final-public-release/review54/PUBLIC_RELEASE_MANIFEST.json).

Only these two of the 29 candidate files differ:

| File | New SHA-256 |
|---|---|
| `requirements.txt` | `5c391e59b10f3bf4823980db6c922344ecad063f56e899c18fbf5fd9f4d25871` |
| `packages.txt` | `be6560c5e25fbb426cb9d83fc15cb90564af01e338ff66e792dbb4240ec43803` |

Local project changes made for this task:

- [requirements.txt](requirements.txt): inline equivalent constraints.
- [packages.txt](packages.txt): LF-only system-package list.
- [test_hf_build_requirements.py](tests/test_hf_build_requirements.py): three
  regressions for self-containment, exact constraint preservation, and LF-only
  apt arguments.
- This report and ignored PR54 build/harness/evidence artifacts.

The repository's Git line-ending settings can reintroduce CRLF on checkout.
The byte-level regression and exact release hashes detect this; no Git/Windows/
WSL settings were changed.

## Evidence and limits

- [Original Linux failure log](.venv/final-public-release/review54/original-linux-build.log)
- [Corrected dependency install and apt failure](.venv/final-public-release/review54/corrected-linux-build-cached.log)
- [Successful final Linux build](.venv/final-public-release/review54/corrected-linux-build-lf.log)
- [Machine-readable Linux result](.venv/final-public-release/review54/LINUX_RESULT.json)
- [Application startup log](.venv/final-public-release/review54/linux-startup.log)
- [UI configuration](.venv/final-public-release/review54/linux-ui-config.json)
- [Installed package snapshot](.venv/final-public-release/review54/linux-pip-freeze.txt)
- [Reproduction Dockerfile](.venv/final-public-release/review54/Dockerfile)
- [Offline verification harness](.venv/final-public-release/review54/verify_linux.py)

This is a genuine **local Linux** build, not a hosted HF/ZeroGPU build.
It preserves Python 3.12, Gradio 6.30.0, candidate constraints, and the isolated
requirements-mount behavior. It also uses the observed HF extras for Gradio,
spaces, Torch upper bound, uvicorn and websockets.

It is not a bit-identical HF builder: Debian Bookworm's Node 18 is installed
(SSR disabled), and every HF infrastructure bootstrap dependency/layer is not
recreated. No CUDA execution, Qwen loading/inference, or production model
prefetch path was tested. The package snapshot records the actual resolved
versions; the candidate's transitive ranges remain mutable.

No full invoice/stress suite was rerun for these packaging-only changes.
The prior PR51 results remain historical; the three new focused regressions and
the real Linux installation/startup/file/UI checks are the validation for PR54.

## Release recommendation and handover

**GO for the local Linux build/startup gate on the corrected PR54 artifact.**
A public release still requires explicit authorization and a fresh live-parent/
backup check. No remote resource was read or modified during this task.

Do not invoke the old PR51 deployment helper against its hardcoded old candidate.
The [corrected operation plan](.venv/final-public-release/review54/CORRECTED_OPERATION_PLAN.json)
is local preparation only. Against the previously saved public backup it has
**9 additions, 10 replacements, 64 deletions, 10 unchanged files**; LF-normalized
`packages.txt` now equals the original public bytes. Rollback would restore
74 old paths and delete 9 new paths, using the established ignore override and
exact-parent/tree-parity protections. These counts must be reverified before
any separately authorized deployment.

- Branch: `wip/layouts-reprise`.
- HEAD: `9d7a641a52c114484ee614730d15882dbaa35d66`.
- Existing dirty work preserved; no Git commit or push.
- Application sources, extraction behavior, and frozen references unchanged.
- Disposable validation container removed; application terminated; no helper
  server left running.
- Next: obtain separate release approval for the corrected artifact and updated
  operation plan. Do not retry the original PR51 candidate.
