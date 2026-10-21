from typing import List, Dict, Any, Generator, Tuple
import app.config as cfg
from app.llm import (
    get_openai_client,
    get_genai_client,
    has_valid_gemini_key,
    has_valid_openai_endpoint,
)
from app.discrepancy_scanner import detect_heuristic_discrepancy

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
    """Builds the context payload with numbered excerpts and linked visual table data for Gemma 4."""
    context_lines = []
    for c in chunks:
        cid = c.get("citation_id", 1)
        page = c.get("page", 1)
        doc = c.get("filename", "Document")
        parent = c.get("parent_text", "").strip()
        block = f"[{cid}] (File: {doc}, Page {page}):\n{parent}"
        if c.get("linked_visuals"):
            vis_titles = ", ".join(v.get("title", "Visual") for v in c["linked_visuals"])
            block += f"\n(Linked Visual Evidence: {vis_titles})"
        context_lines.append(block + "\n")

    joined_context = "\n".join(context_lines)

    return f"""=== VERIFIED SOURCE EXCERPTS ===
{joined_context}
================================

USER QUESTION:
{query}

REMINDER: Answer strictly from the excerpts above. Cite [1], [2] for every claim. If unverified, refuse to guess."""


def _build_extractive_fallback(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Deterministic Extractive Synthesis when no external LLM server is reachable.
    Synthesizes the highest-scoring verified excerpts with exact citation markers [1], [2]
    and highlights any cross-modal discrepancy between cited clauses and linked tables.
    """
    if not chunks:
        return "I cannot verify this information in the provided document excerpts."

    top_c = chunks[0]
    cid = top_c.get("citation_id", 1)
    doc = top_c.get("filename", "Document")
    page = top_c.get("page", 1)
    excerpt = top_c.get("parent_text", "").strip()

    parts = [f"Based on verified document records [{cid}], {excerpt}"]

    # Include second excerpt if it comes from a complementary clause or linked table on the topic
    if len(chunks) > 1 and chunks[1].get("dense_score", 0.0) >= 0.45:
        second_c = chunks[1]
        cid2 = second_c.get("citation_id", 2)
        excerpt2 = second_c.get("parent_text", "").strip()
        if excerpt2 and excerpt2 != excerpt:
            parts.append(f"\n\nAdditionally, supporting record [{cid2}] states:\n{excerpt2}")
            disc = detect_heuristic_discrepancy(excerpt, excerpt2) or detect_heuristic_discrepancy(excerpt2, excerpt)
            if disc and disc.get("status") == "CONTRADICTION":
                parts.append(
                    f"\n\n**Cross-Modal Audit Alert:** {disc.get('discrepancy_delta')} [{cid}][{cid2}]"
                )

    parts.append(
        f"\n\n*(Verified in {doc}, Page {page} • Extractive AST Failover Mode: 0 external LLM tokens used.)*"
    )
    return "".join(parts)


def generate_grounded_answer_with_telemetry(
    query: str,
    chunks: List[Dict[str, Any]],
    persona_addon: str = ""
) -> Tuple[str, Dict[str, Any]]:
    """
    Synchronous grounded generation returning (answer_text, telemetry_dict).
    Captures genuine token usage from Google GenAI or OpenAI-compatible server.
    """
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)
    active_model = cfg.get_active_gemma_model()

    # 1. Try Native Google GenAI Gemma 4
    if cfg.USE_GOOGLE_GENAI and has_valid_gemini_key():
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                response = genai_cli.models.generate_content(
                    model=active_model,
                    contents=full_prompt,
                )
                if response and response.text:
                    usage = getattr(response, "usage_metadata", None)
                    prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
                    comp_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
                    total_tok = int(getattr(usage, "total_token_count", 0) or (prompt_tok + comp_tok))
                    return response.text, {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": comp_tok,
                        "total_tokens": total_tok,
                        "generation_mode": "google_genai",
                        "model_used": active_model,
                    }
            except Exception:
                pass

    # 2. Try OpenAI-compatible server (Local Ollama / Venue Gemma)
    if has_valid_openai_endpoint():
        client = get_openai_client()
        try:
            response = client.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=600
            )
            text = response.choices[0].message.content or "No response from Gemma."
            usage = getattr(response, "usage", None)
            prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            comp_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tok = int(getattr(usage, "total_tokens", 0) or (prompt_tok + comp_tok))
            return text, {
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": total_tok,
                "generation_mode": "openai_compatible",
                "model_used": cfg.LLM_MODEL,
            }
        except Exception:
            pass

    # 3. Transparent Extractive AST Fallback (Offline Mode)
    fallback_text = _build_extractive_fallback(query, chunks)
    return fallback_text, {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "generation_mode": "extractive_fallback",
        "model_used": "extractive-ast-engine",
    }


def generate_grounded_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    persona_addon: str = ""
) -> str:
    """Synchronous grounded generation returning the answer string."""
    answer, _ = generate_grounded_answer_with_telemetry(query, chunks, persona_addon)
    return answer


def stream_grounded_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    persona_addon: str = ""
) -> Generator[str, None, None]:
    """
    Streams tokens in real-time from Gemma 4.
    Falls back transparently to extractive AST synthesis if the inference server is offline.
    """
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)
    active_model = cfg.get_active_gemma_model()

    # 1. Try Native Google GenAI Gemma 4 Streaming
    if cfg.USE_GOOGLE_GENAI and has_valid_gemini_key():
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                streamed_any = False
                for chunk in genai_cli.models.generate_content_stream(
                    model=active_model,
                    contents=full_prompt,
                ):
                    if chunk.text:
                        streamed_any = True
                        yield chunk.text
                if streamed_any:
                    return
            except Exception:
                pass

    # 2. Try OpenAI-compatible server (Local Ollama / Venue Gemma)
    if has_valid_openai_endpoint():
        client = get_openai_client()
        try:
            stream = client.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=600,
                stream=True
            )
            streamed_any = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    streamed_any = True
                    yield delta
            if streamed_any:
                return
        except Exception:
            pass

    # 3. Offline streaming fallback (includes explicit extractive mode disclosure)
    yield _build_extractive_fallback(query, chunks)
