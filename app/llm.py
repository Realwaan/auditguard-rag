import os
from openai import OpenAI
from app.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

def get_openai_client() -> OpenAI:
    """Returns an OpenAI SDK client configured for ALMA Gemma 4 server."""
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY or "alma-hackathon-token"
    )

def call_plain_gemma(prompt: str) -> str:
    """
    Calls Gemma 4 WITHOUT any RAG context.
    Used for the live 'Hallucination Radar' side-by-side comparison on stage.
    """
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
        return f"[Plain Gemma Error: {e}]"
