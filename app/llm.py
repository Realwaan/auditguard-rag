import os
import base64
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
from openai import OpenAI
from app.config import (
    BASE_DIR, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
    GEMINI_API_KEY, USE_GOOGLE_GENAI, get_gemma_model
)

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

def is_llm_configured() -> bool:
    """Returns True if Google GenAI key or custom OpenAI endpoint is configured."""
    return bool((USE_GOOGLE_GENAI and GEMINI_API_KEY) or (LLM_API_KEY and LLM_API_KEY != "local-or-alma-key") or LLM_BASE_URL)

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

def load_image_bytes(image_url: str) -> Optional[Tuple[bytes, str]]:
    """
    Resolves an image URL (such as /static/crops/xxx.webp or /static/previews/xxx.webp)
    to disk and returns (raw_bytes, mime_type). Returns None if file does not exist.
    """
    if not image_url:
        return None
    
    clean_url = str(image_url).split("?")[0].strip()
    if clean_url.startswith("/static/"):
        rel_path = clean_url[len("/static/"):]
        file_path = BASE_DIR / "app" / "static" / rel_path
    elif clean_url.startswith("static/"):
        rel_path = clean_url[len("static/"):]
        file_path = BASE_DIR / "app" / "static" / rel_path
    else:
        file_path = Path(clean_url)
        if not file_path.is_absolute():
            file_path = BASE_DIR / clean_url

    if not file_path.exists() or not file_path.is_file():
        return None

    try:
        data = file_path.read_bytes()
        mime, _ = mimetypes.guess_type(str(file_path))
        if not mime:
            suffix = file_path.suffix.lower()
            if suffix == ".webp":
                mime = "image/webp"
            elif suffix == ".png":
                mime = "image/png"
            elif suffix in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            else:
                mime = "application/octet-stream"
        return (data, mime)
    except Exception as e:
        print(f"[AuditGuard Image Load Warning]: {e}")
        return None

def make_genai_part(data: bytes, mime_type: str):
    """Creates a Google GenAI Part from raw bytes for multimodal inference."""
    if _HAS_GENAI:
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    return None

def make_openai_image_part(data: bytes, mime_type: str) -> Dict[str, Any]:
    """Creates an OpenAI-compatible multimodal content dictionary for chat completions."""
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{b64}",
            "detail": "high"
        }
    }

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
