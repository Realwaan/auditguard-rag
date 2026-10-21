import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env (and parent directory .env fallback if present)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

# ALMA Gemma 4 Inference Server
_PLACEHOLDER_KEYS = {"", "local-or-alma-key", "alma-hackathon-token", "REPLACE_ON_HACKATHON_DAY", "YOUR_API_KEY_HERE", "YOUR_GOOGLE_AI_STUDIO_API_KEY"}
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-or-alma-key").strip()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))

# Google GenAI Native Gemma 4 (31B / 26B MoE)
# Accepts GEMINI_API_KEY, GOOGLE_API_KEY, or falls back to LLM_API_KEY if a real key was placed there
_raw_gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_API_KEY = _raw_gemini_key if _raw_gemini_key not in _PLACEHOLDER_KEYS else (
    LLM_API_KEY if LLM_API_KEY not in _PLACEHOLDER_KEYS else ""
)
USE_GOOGLE_GENAI = os.getenv("USE_GOOGLE_GENAI", "true").lower() in ("true", "1", "yes")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")

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
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))


def get_active_gemma_model() -> str:
    """Returns the currently selected Gemma 4 model identifier."""
    return GEMMA_MODEL


def set_active_gemma_model(model_name: str) -> str:
    """Updates the active Gemma 4 model identifier at runtime."""
    global GEMMA_MODEL
    GEMMA_MODEL = model_name
    return GEMMA_MODEL
