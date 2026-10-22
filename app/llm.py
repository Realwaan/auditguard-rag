from typing import Optional, Tuple, Dict, Any
from openai import OpenAI
import app.config as cfg

_PLACEHOLDER_KEYS = {
    "", "local-or-alma-key", "alma-hackathon-token", "REPLACE_ON_HACKATHON_DAY",
    "YOUR_API_KEY_HERE", "YOUR_GOOGLE_AI_STUDIO_API_KEY", "gsk_your_groq_api_key_here",
    "AIzaSy_your_gemini_key_here"
}


def has_valid_primary_key() -> bool:
    """Returns True if the primary Groq API key is populated with a real key."""
    key = (cfg.LLM_API_KEY or "").strip()
    return bool(key and key not in _PLACEHOLDER_KEYS)


def has_valid_fallback_key() -> bool:
    """Returns True if the fallback Gemini OpenAI-compatible API key is populated."""
    key = (cfg.FALLBACK_API_KEY or "").strip()
    return bool(key and key not in _PLACEHOLDER_KEYS)


def is_llm_configured() -> bool:
    """Returns True if either Primary (Groq) or Fallback (Gemini) is ready."""
    return has_valid_primary_key() or has_valid_fallback_key()


def get_primary_client(timeout: Optional[float] = None) -> OpenAI:
    """Returns an OpenAI SDK client pointed at Primary Fast API (Groq)."""
    return OpenAI(
        base_url=cfg.LLM_BASE_URL,
        api_key=cfg.LLM_API_KEY or "no-key",
        timeout=timeout or cfg.LLM_TIMEOUT_SECONDS,
        max_retries=1
    )


def get_fallback_client(timeout: Optional[float] = None) -> OpenAI:
    """Returns an OpenAI SDK client pointed at Fallback (Google Gemini OpenAI-compatible endpoint)."""
    return OpenAI(
        base_url=cfg.FALLBACK_BASE_URL,
        api_key=cfg.FALLBACK_API_KEY or "no-key",
        timeout=timeout or cfg.LLM_TIMEOUT_SECONDS,
        max_retries=1
    )


# Compatibility aliases
def get_openai_client(timeout: Optional[float] = None) -> OpenAI:
    if has_valid_primary_key():
        return get_primary_client(timeout)
    return get_fallback_client(timeout)


def has_valid_gemini_key() -> bool:
    return has_valid_fallback_key()


def has_valid_openai_endpoint() -> bool:
    return has_valid_primary_key() or has_valid_fallback_key()


def get_genai_client():
    return None


def call_plain_gemma_with_telemetry(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """
    Calls the LLM directly WITHOUT any RAG context for baseline comparison,
    trying Primary Fast API (Groq) first, then Fallback (Gemini), then offline.
    """
    # 1. Try Primary Fast API (Groq)
    if has_valid_primary_key():
        try:
            client = get_primary_client(timeout=8.0)
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
                "baseline_mode": "groq_primary",
                "baseline_model": cfg.LLM_MODEL,
            }
        except Exception:
            pass

    # 2. Try Fallback (Gemini OpenAI-compatible)
    if has_valid_fallback_key():
        try:
            client = get_fallback_client(timeout=8.0)
            response = client.chat.completions.create(
                model=cfg.FALLBACK_MODEL,
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
                "baseline_mode": "gemini_fallback",
                "baseline_model": cfg.FALLBACK_MODEL,
            }
        except Exception:
            pass

    return (
        "[Baseline Offline: No external LLM endpoint configured or reachable. Grounded AST verification remains active.]",
        {"baseline_tokens": 0, "baseline_mode": "offline", "baseline_model": "none"}
    )


def call_plain_gemma(prompt: str) -> str:
    """Calls baseline without RAG context."""
    text, _ = call_plain_gemma_with_telemetry(prompt)
    return text
