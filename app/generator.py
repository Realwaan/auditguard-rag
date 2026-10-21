import re
from typing import List, Dict, Any, Generator
from app.llm import get_openai_client, get_genai_client
from app.config import LLM_MODEL, USE_GOOGLE_GENAI, get_gemma_model

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

def sanitize_citations(text: str, max_chunk: int) -> str:
    """Clamps or strips citations referencing non-existent chunks (e.g. [5] when only 2 chunks exist)."""
    if max_chunk <= 0:
        return text
    def _replace(m):
        val = int(m.group(1))
        if val > max_chunk:
            return ""
        return m.group(0)
    return re.sub(r'\[(\d+)\]', _replace, text).strip()

def generate_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = ""
) -> Dict[str, Any]:
    """
    Synchronous grounded generation with native Gemma 4 support.
    Returns real token telemetry, exact citations, and transparent fallback status.
    """
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)
    num_chunks = len(chunks)

    # 1. Try Native Google GenAI Gemma 4
    if USE_GOOGLE_GENAI:
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                active_model = get_gemma_model()
                response = genai_cli.models.generate_content(
                    model=active_model,
                    contents=full_prompt,
                )
                if response and response.text:
                    clean_text = sanitize_citations(response.text, num_chunks)
                    actual_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        actual_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
                    
                    return {
                        "answer": clean_text,
                        "tokens": actual_tokens,
                        "is_fallback": False,
                        "model": active_model
                    }
            except Exception as e:
                print(f"[AuditGuard Google GenAI Error - Grounded Generator]: {e}")

    # 2. Try OpenAI-compatible server (Local Ollama / Venue Gemma)
    client = get_openai_client()
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
        content = response.choices[0].message.content or "No response from Gemma."
        clean_text = sanitize_citations(content, num_chunks)
        actual_tokens = getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") and response.usage else 0
        return {
            "answer": clean_text,
            "tokens": actual_tokens,
            "is_fallback": False,
            "model": LLM_MODEL
        }
    except Exception as e:
        print(f"[AuditGuard LLM Fallback Warning]: {e}")
        # Transparent In-Memory Extractive Failover (zero hallucination)
        if chunks:
            top_c = chunks[0]
            excerpt = top_c.get("parent_text", "").strip()
            cid = top_c.get("citation_id", 1)
            doc = top_c.get("filename", "Document")
            page = top_c.get("page", 1)
            
            fallback_text = (
                f"According to verified agreement records [{cid}], {excerpt} "
                f"\n\n*(Verified in {doc}, Section/Page {page}. Generated via AuditGuard Extractive Fallback.)*"
            )
            return {
                "answer": fallback_text,
                "tokens": 0,
                "is_fallback": True,
                "model": "AuditGuard Extractive Fallback"
            }
        return {
            "answer": "I cannot verify this information in the provided document excerpts.",
            "tokens": 0,
            "is_fallback": True,
            "model": "AuditGuard Refusal"
        }

def stream_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = ""
) -> Generator[str, None, None]:
    """
    Streams tokens in real-time from Gemma 4 via SSE.
    Yields delta text strings for sub-300ms time-to-first-token.
    Falls back gracefully if inference server is offline.
    """
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)

    # 1. Try Native Google GenAI Gemma 4 Streaming
    if USE_GOOGLE_GENAI:
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                streamed_any = False
                for chunk in genai_cli.models.generate_content_stream(
                    model=get_gemma_model(),
                    contents=full_prompt,
                ):
                    if chunk.text:
                        streamed_any = True
                        yield chunk.text
                if streamed_any:
                    return
            except Exception as e:
                print(f"[AuditGuard Google GenAI Error - Streaming Generator]: {e}")

    # 2. Try OpenAI-compatible server (Local Ollama / Venue Gemma)
    client = get_openai_client()
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
    except Exception:
        # Offline streaming fallback
        if chunks:
            top_c = chunks[0]
            excerpt = top_c.get("parent_text", "").strip()
            cid = top_c.get("citation_id", 1)
            yield f"According to verified agreement records [{cid}], {excerpt} *(AuditGuard Extractive Fallback)*"
        else:
            yield "I cannot verify this information in the provided document excerpts."
