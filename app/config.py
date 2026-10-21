import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env (and parent directory .env fallback if present)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

# ALMA Gemma 4 Inference Server
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-or-alma-key")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))

# Google GenAI Native Gemma 4 (31B)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GOOGLE_GENAI = os.getenv("USE_GOOGLE_GENAI", "true").lower() in ("true", "1", "yes")
_CURRENT_GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
GEMMA_MODEL = _CURRENT_GEMMA_MODEL

def get_gemma_model() -> str:
    """Returns the actively selected Gemma model name."""
    global _CURRENT_GEMMA_MODEL
    return _CURRENT_GEMMA_MODEL

def set_gemma_model(model_name: str) -> str:
    """Dynamically updates the active Gemma model name across all modules."""
    global _CURRENT_GEMMA_MODEL, GEMMA_MODEL
    _CURRENT_GEMMA_MODEL = model_name
    GEMMA_MODEL = model_name
    return _CURRENT_GEMMA_MODEL

get_active_gemma_model = get_gemma_model
set_active_gemma_model = set_gemma_model

# Storage & Paths
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
PREVIEWS_DIR = BASE_DIR / "app" / "static" / "previews"
CROPS_DIR = BASE_DIR / "app" / "static" / "crops"

for d in [DOCS_DIR, INDEX_DIR, PREVIEWS_DIR, CROPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Retrieval & Guardrail Thresholds
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.58"))
TOP_K = int(os.getenv("TOP_K", "4"))
# Bound unsearchable image-page inspection; larger scans require a page-specific query.
MAX_VISUAL_SCAN_PAGES = max(1, int(os.getenv("MAX_VISUAL_SCAN_PAGES", "8")))
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024)))
