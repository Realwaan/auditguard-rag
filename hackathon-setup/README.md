# ALMA hackathon laptop setup

Prepared for October 22, 2026, Cebu. This folder hands off the development
environment, not the AuditGuard application. Build the application during the
event under the organizers' rules. Setup utilities do not imply organizer
approval of a prebuilt scaffold.

## Status

- Target: **Windows x64, standard CPython 3.13**; this PC has 3.13.5.
- 63 Python dependency versions captured, including DOCX, PPTX and HTTP testing.
- BGE small English embedding cache included, about 67 MB.
- **Offline package wheels are missing.** The sandboxed package download failed
  and the subsequent download permission request was cancelled. Exact package
  availability and fresh installation have not been verified.
- Python, Node, Git, editor installers and frontend npm packages are not included.
- ALMA endpoint, served model ID, API protocol, image support and network are
  unconfirmed. No ALMA inference or RAG accuracy test has run.
- `READINESS.json`, when present, reports local dependency checks only.

Latest run on this PC: **4 of 6 checks passed**. Windows/Python, dependency
versions, PDF rendering and Office parsers passed. The HTTP dependency check and
combined embedding/database check each exceeded 45 seconds under the execution
sandbox; their cause is unconfirmed. Rerun on the laptop before relying on this
setup. The copied model cache has checksums in `model-inventory.json`, but cached
files alone do not prove successful model inference.

## On the laptop

1. Copy this complete folder including `model-cache`, or extract the setup ZIP.
2. Install standard **Python 3.13 x64**. Recreate the environment; do not transfer
   this PC's virtual environment, Python executable or site-packages directory.
3. In PowerShell in this folder, with internet available, run:

   ```powershell
   .\setup.ps1 -Online
   ```

   Use `-Python 'C:\Path\To\Python313\python.exe'` if `python` selects another
   interpreter. If your organization's script policy blocks the script, follow
   its approved process; these are the individual setup commands:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.windows-py313.lock.txt
   .\.venv\Scripts\python.exe -m pip check
   .\.venv\Scripts\python.exe preflight.py
   ```

4. Confirm the checks pass **on the laptop** and rerun with internet disconnected.
   They exercise the cached model, parsers, BM25, LanceDB and HTTP dependencies
   using synthetic data. Each check has a 45-second timeout.
5. Select `.venv\Scripts\python.exe` as your editor's interpreter during the event.

To prepare an offline installation backup, run this on a connected Windows x64
machine with the same Python minor version, then copy `wheels` too:

```powershell
python -m pip download --only-binary=:all: -r requirements.windows-py313.lock.txt -d wheels
```

Only after that succeeds can you use `setup.ps1` without `-Online`. The lock is a
snapshot of installed package metadata, not yet a verified fresh-install lock.
If an exact version is unavailable, resolve and recheck the environment before
substituting versions.

## At the venue

Obtain the base URL, exact model ID, API protocol, Wi-Fi/network, image support,
context limit and timeout expectations from the organizers. Do not guess the
Gemma identifier or append `/v1` without confirming it.

If ALMA confirms an **OpenAI-compatible** chat-completions API:

```powershell
.\use-alma.ps1 -BaseUrl 'http://ALMA_HOST:PORT/v1' -Model 'EXACT_SERVED_MODEL_ID'
.\.venv\Scripts\python.exe check-alma.py
```

Replace both values with organizer-provided values. Settings apply to the current
PowerShell session. Google mode is disabled; FastEmbed uses the copied cache.
`alma-local` is a non-secret placeholder for SDKs requiring a nonempty API key,
not an ALMA-issued credential. The API probe sends **no authentication header**
and only `Reply with exactly READY.` It never falls back to another provider.

For another protocol, follow the organizer's client setup. A successful greeting
checks one text request, not chart reasoning or RAG accuracy. Application code
built at the event must explicitly use ALMA and reject missing configuration;
environment variables alone cannot enforce that on arbitrary application code.
Bind your development server to `127.0.0.1` unless team access is deliberate.

## Optional frontend tools

This PC has Node **22.17.0** and npm **10.9.2**. Installed Vite 8.3.4 requires
Node `^20.19.0 || >=22.12.0`, so this PC meets its declared requirement.
`frontend-versions.json` records installed frontend package versions. The kit
does not contain a frontend project, node_modules or an offline npm cache.
Install frontend dependencies with internet when building your event project.
Sign into your editor and coding tools on the laptop beforehand.

## Boundaries

No application source, built UI, private documents, original .env, API keys,
existing database or copied virtual environment is included. OCR is not supplied
or tested; start with text-based PDFs. The original code audit findings remain
open. ALMA inference still needs access to ALMA's network even when parsing and
embeddings work offline. Registration and organizer approval are not verified.

References: [Python venv](https://docs.python.org/3.13/library/venv.html),
[pip download](https://pip.pypa.io/en/stable/cli/pip_download/),
[FastEmbed cache](https://qdrant.tech/documentation/edge/edge-fastembed-embeddings/).
