import os
from openai import OpenAI
from app.config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
    GEMINI_API_KEY, USE_GOOGLE_GENAI, get_gemma_model
)

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

def get_genai_client():
    if _HAS_GENAI and GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    return None

def get_openai_client() -> OpenAI:
    """Returns an OpenAI SDK client configured for ALMA Gemma 4 server with production-grade timeout."""
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY or "alma-hackathon-token",
        timeout=15.0,
        max_retries=1
    )

def call_plain_gemma(prompt: str) -> str:
    """
    Calls Gemma 4 WITHOUT any RAG context.
    Used for the live 'Hallucination Radar' side-by-side comparison on stage.
    Uses native Google GenAI with fast failover.
    """
    concise_prompt = f"Answer concisely in 1-2 sentences without extra preamble:\n{prompt}"
    if USE_GOOGLE_GENAI and _HAS_GENAI and GEMINI_API_KEY:
        try:
            client = get_genai_client()
            response = client.models.generate_content(
                model=get_gemma_model(),
                contents=concise_prompt,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[AuditGuard Google GenAI Error - Plain Gemma Baseline]: {e}")

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question directly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content or "No response received."
    except Exception as e:
        return "[Plain Gemma Offline: Local server unreached. Grounded defense active.]"
