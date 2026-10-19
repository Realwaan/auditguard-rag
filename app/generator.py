from typing import List, Dict, Any, Generator
from app.llm import get_openai_client
from app.config import LLM_MODEL

BASE_SYSTEM_PROMPT = """You are AuditGuard, a defense-grade document intelligence and compliance verification copilot.
Your job is to answer questions strictly and exclusively using the provided numbered source excerpts.

STRICT GROUNDING RULES:
1. Every factual statement or recommendation MUST cite its supporting excerpt using citation badges like [1], [2], etc.
2. If the provided excerpts do not explicitly contain the answer, reply ONLY with:
   "I cannot verify this information in the provided document excerpts."
3. Do NOT invent, assume, extrapolate, or use outside unverified knowledge.
4. Keep answers professional, concise, and audit-ready.
"""

def build_grounded_prompt(query: str, chunks: List[Dict[str, Any]], persona_addon: str = "") -> str:
    """Builds the context payload with numbered excerpts for Gemma 4."""
    context_lines = []
    for c in chunks:
        cid = c.get("citation_id", 1)
        page = c.get("page", 1)
        doc = c.get("filename", "Document")
        parent = c.get("parent_text", "").strip()
        context_lines.append(f"[{cid}] (File: {doc}, Page {page}):\n{parent}\n")

    joined_context = "\n".join(context_lines)

    return f"""=== VERIFIED SOURCE EXCERPTS ===
{joined_context}
================================

USER QUESTION:
{query}

REMINDER: Answer strictly from the excerpts above. Cite [1], [2] for every claim. If unverified, refuse to guess."""

def generate_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = ""
) -> str:
    """Synchronous grounded generation."""
    client = get_openai_client()
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return response.choices[0].message.content or "No response from Gemma."
    except Exception as e:
        return f"Error contacting ALMA Gemma server: {e}"

def stream_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = ""
) -> Generator[str, None, None]:
    """
    Streams tokens in real-time from Gemma 4 via SSE.
    Yields delta text strings for sub-300ms time-to-first-token.
    """
    client = get_openai_client()
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=600,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    except Exception as e:
        yield f"[Streaming error: {e}]"
