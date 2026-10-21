from typing import Optional, Tuple, Dict, Any
from openai import OpenAI
import app.config as cfg

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

_PLACEHOLDER_KEYS = {"", "local-or-alma-key", "alma-hackathon-token", "REPLACE_ON_HACKATHON_DAY"}


def has_valid_gemini_key() -> bool:
    key = (cfg.GEMINI_API_KEY or "").strip()
    return bool(_HAS_GENAI and key and key not in _PLACEHOLDER_KEYS)


def has_valid_openai_endpoint() -> bool:
    base = (cfg.LLM_BASE_URL or "").strip().rstrip("/")
    key = (cfg.LLM_API_KEY or "").strip()
    if not base or "REPLACE_WITH_ALMA_URL" in base:
        return False
    # If pointing at official OpenAI API, require a non-placeholder API key
    if "api.openai.com" in base and key in _PLACEHOLDER_KEYS:
        return False
    return True


def is_llm_configured() -> bool:
    """Returns True if either Google GenAI or a valid OpenAI-compatible endpoint is configured."""
    return (cfg.USE_GOOGLE_GENAI and has_valid_gemini_key()) or has_valid_openai_endpoint()


def get_genai_client():
    if has_valid_gemini_key():
        return genai.Client(api_key=cfg.GEMINI_API_KEY)
    return None


def get_openai_client(timeout: Optional[float] = None) -> OpenAI:
    """Returns an OpenAI SDK client configured for the ALMA / local inference server."""
    return OpenAI(
        base_url=cfg.LLM_BASE_URL,
        api_key=cfg.LLM_API_KEY or "alma-hackathon-token",
        timeout=timeout or cfg.LLM_TIMEOUT_SECONDS,
        max_retries=0
    )


def call_plain_gemma_with_telemetry(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """
    Calls Gemma 4 WITHOUT any RAG context for side-by-side baseline comparison,
    returning both the response text and real token telemetry.
    """
    concise_prompt = f"Answer concisely in 1-2 sentences without extra preamble:\n{prompt}"
    active_model = cfg.get_active_gemma_model()

    if cfg.USE_GOOGLE_GENAI and has_valid_gemini_key():
        try:
            client = get_genai_client()
            response = client.models.generate_content(
                model=active_model,
                contents=concise_prompt,
            )
            if response and response.text:
                usage = getattr(response, "usage_metadata", None)
                total_tok = getattr(usage, "total_token_count", 0) or 0
                return response.text, {
                    "baseline_tokens": int(total_tok),
                    "baseline_mode": "google_genai",
                    "baseline_model": active_model,
                }
        except Exception:
            pass

    if has_valid_openai_endpoint():
        client = get_openai_client(timeout=8.0)
        try:
            response = client.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question directly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            text = response.choices[0].message.content or "No response received."
            usage = getattr(response, "usage", None)
            total_tok = getattr(usage, "total_tokens", 0) or 0
            return text, {
                "baseline_tokens": int(total_tok),
                "baseline_mode": "openai_compatible",
                "baseline_model": cfg.LLM_MODEL,
            }
        except Exception:
            pass

    return (
        "[Plain Gemma Offline: No external LLM endpoint configured or reachable. Grounded AST verification remains active.]",
        {"baseline_tokens": 0, "baseline_mode": "offline", "baseline_model": "none"}
    )


def call_plain_gemma(prompt: str) -> str:
    """Calls Gemma 4 WITHOUT any RAG context."""
    text, _ = call_plain_gemma_with_telemetry(prompt)
    return text
