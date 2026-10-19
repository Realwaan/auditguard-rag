import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ALMA Gemma 4 Inference Server
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-or-alma-key")

# Storage & Paths
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
PREVIEWS_DIR = BASE_DIR / "app" / "static" / "previews"

for d in [DOCS_DIR, INDEX_DIR, PREVIEWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Retrieval & Guardrail Thresholds
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.58"))
TOP_K = int(os.getenv("TOP_K", "4"))
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
