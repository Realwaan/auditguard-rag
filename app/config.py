import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env (and parent directory .env fallback if present)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

_PLACEHOLDER_KEYS = {
    "", "local-or-alma-key", "alma-hackathon-token", "REPLACE_ON_HACKATHON_DAY",
    "YOUR_API_KEY_HERE", "YOUR_GOOGLE_AI_STUDIO_API_KEY", "gsk_your_groq_api_key_here",
    "AIzaSy_your_gemini_key_here"
}

# 1. Primary Fast API (Groq)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))

# 2. Optional Fallback (Google Gemini OpenAI-compatible endpoint)
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")
FALLBACK_API_KEY = (os.getenv("FALLBACK_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip())

# Legacy compat alias for GEMINI_API_KEY if needed
GEMINI_API_KEY = FALLBACK_API_KEY
GEMMA_MODEL = LLM_MODEL
USE_GOOGLE_GENAI = bool(FALLBACK_API_KEY)

# 3. Local Embeddings (FastEmbed ONNX - zero API burn)
_raw_embed = (os.getenv("EMBEDDING_MODEL", "").strip() or os.getenv("DEFAULT_EMBED_MODEL", "all-MiniLM-L6-v2").strip())
if _raw_embed in ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"):
    DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
else:
    DEFAULT_EMBED_MODEL = _raw_embed

# Storage & Paths
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
PREVIEWS_DIR = BASE_DIR / "app" / "static" / "previews"
CROPS_DIR = BASE_DIR / "app" / "static" / "crops"

for d in [DOCS_DIR, INDEX_DIR, PREVIEWS_DIR, CROPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Retrieval & Guardrail Thresholds
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))


def get_active_gemma_model() -> str:
    """Returns the primary active model identifier."""
    return LLM_MODEL


def set_active_gemma_model(model_name: str) -> str:
    """Updates the active model identifier at runtime."""
    global LLM_MODEL, GEMMA_MODEL
    LLM_MODEL = model_name
    GEMMA_MODEL = model_name
    return LLM_MODEL
