# Hackathon Laptop Setup (clean machine)

## Install (do this before Thursday, on good internet)
1. Git
2. Python 3.11-3.13 (tick "Add to PATH")
3. VS Code and Antigravity (sign in once)
4. Chrome
5. Optional: Tesseract OCR (only for scanned PDFs)

## Get the code
```powershell
git clone <YOUR-BOT-REPO-URL>
cd website-associate-bot\hackathon-starter
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pre-download the embedding model (needs internet, run once)
```powershell
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```
Copy the model cache folder to a USB stick as a backup.
Offline wheels: `pip download -r requirements.txt -d wheels` then
`pip install --no-index --find-links wheels -r requirements.txt`.

## Secrets
```powershell
Copy-Item .env.example .env
```
Fill in `.env` on the day with the ALMA URL, model name and API key.
Never commit `.env`. Never share the key outside the team (disqualification).

## Bot (management tool, optional)
In the repo root: `pip install -r requirements.txt`, then create `.env` with
`DISCORD_TOKEN` and `DATABASE_URL` from your password manager.
Run `python main.py`, then `/sync-commands` and `/load-tickets hackathon <channel>`.

## Bring on USB / cloud drive
Demo documents, slides plus PDF backup, wheels, embedding model cache.

## Dry run (do this week)
Clone, install, run the API smoke test (ticket 01) on a fresh venv. Target: under 20 minutes.
