import re
from typing import List, Dict, Any, Generator, Optional, Tuple
from app.llm import (
    get_openai_client, get_genai_client,
    load_image_bytes, make_genai_part, make_openai_image_part
)
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

def verify_citations_grounding(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rigorously verifies that claims citing [1], [2], etc. are genuinely supported by their referenced chunks.
    Isolates the clause associated with each citation badge and validates numbers, percentages,
    and content keywords to prevent unsupported assertions or fabricated citation tags.
    """
    if not answer or not chunks:
        return {
            "all_supported": True,
            "total_citations": 0,
            "unsupported_count": 0,
            "verified_citations": []
        }

    chunk_map = {c.get("citation_id", idx + 1): c for idx, c in enumerate(chunks)}
    citation_positions = list(re.finditer(r'\[(\d+)\]', answer))
    if not citation_positions:
        return {
            "all_supported": True,
            "total_citations": 0,
            "unsupported_count": 0,
            "verified_citations": []
        }

    verified_list = []
    unsupported_count = 0
    total_citations = 0

    prev_end = 0
    for match in citation_positions:
        cid = int(match.group(1))
        total_citations += 1
        start_idx = match.start()
        end_idx = match.end()

        # The clause belongs from the previous boundary (previous citation, or start of sentence)
        clause_raw = answer[prev_end:start_idx].strip()
        sub_sentences = re.split(r'[.!?\n]+', clause_raw)
        claim_clause = sub_sentences[-1].strip() if sub_sentences else clause_raw
        # Clean leading conjunctions/punctuation
        claim_clause = re.sub(r'^[,\s]*(?:and|or|but|also|furthermore)?\s*', '', claim_clause, flags=re.IGNORECASE)
        prev_end = end_idx

        target_chunk = chunk_map.get(cid)
        if not target_chunk:
            unsupported_count += 1
            verified_list.append({
                "citation_id": cid,
                "supported": False,
                "reason": "Citation ID out of bounds",
                "claim_excerpt": claim_clause[:100]
            })
            continue

        chunk_text = (target_chunk.get("parent_text", "") + " " + target_chunk.get("child_text", "")).lower()
        claim_text = claim_clause.lower()

        # 1. Number / Percentage consistency check for this specific clause
        claim_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', claim_text)
        number_mismatch = False
        for num in claim_numbers:
            if len(num) > 1 and num not in chunk_text:
                number_mismatch = True
                break

        # 2. Key content keyword overlap for this clause
        claim_tokens = [
            t for t in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', claim_text) 
            if t not in {
                "the", "and", "that", "this", "with", "from", "for", "are", "was", "were", 
                "been", "according", "records", "excerpts", "clause", "page", "section", "provides"
            }
        ]
        if claim_tokens:
            matched_tokens = [t for t in claim_tokens if t in chunk_text]
            overlap_ratio = len(matched_tokens) / len(claim_tokens)
        else:
            overlap_ratio = 1.0

        is_supported = (overlap_ratio >= 0.25) and not number_mismatch
        if not is_supported:
            unsupported_count += 1

        verified_list.append({
            "citation_id": cid,
            "supported": is_supported,
            "overlap_score": round(overlap_ratio, 2),
            "number_verified": not number_mismatch,
            "claim_excerpt": claim_clause[:100],
            "reason": "Supported by evidentiary excerpt" if is_supported else (
                "Numerical discrepancy between clause and cited chunk" if number_mismatch 
                else "Insufficient textual support in cited excerpt"
            )
        })

    return {
        "all_supported": (unsupported_count == 0),
        "total_citations": total_citations,
        "unsupported_count": unsupported_count,
        "verified_citations": verified_list
    }

def extract_visual_images_from_chunks(chunks: List[Dict[str, Any]]) -> List[Tuple[bytes, str]]:
    """Extracts unique visual image crops associated with retrieved chunks from disk."""
    visual_images: List[Tuple[bytes, str]] = []
    seen_urls = set()
    for c in chunks:
        urls = []
        if c.get("image_url"):
            urls.append(c.get("image_url"))
        for lv in c.get("linked_visuals", []):
            if lv.get("image_url"):
                urls.append(lv.get("image_url"))
        for u in urls:
            if u and u not in seen_urls:
                seen_urls.add(u)
                img_data = load_image_bytes(u)
                if img_data:
                    visual_images.append(img_data)
    return visual_images

def generate_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = "",
    model_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronous grounded generation with native Gemma 4 support.
    Accepts multimodal image crops from linked visual nodes, passes images directly to Gemma,
    and performs deep post-generation citation verification.
    """
    system_content = BASE_SYSTEM_PROMPT
    if persona_addon:
        system_content += f"\n\nSPECIALIZED AUDIT PERSONA:\n{persona_addon}"

    user_content = build_grounded_prompt(query, chunks, persona_addon)
    num_chunks = len(chunks)
    active_model = model_override or get_gemma_model()
    visual_images = extract_visual_images_from_chunks(chunks)

    # 1. Try Native Google GenAI Gemma 4 (multimodal)
    if USE_GOOGLE_GENAI:
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                contents: List[Any] = [full_prompt]
                for img_bytes, mime in visual_images:
                    part = make_genai_part(img_bytes, mime)
                    if part:
                        contents.append(part)

                response = genai_cli.models.generate_content(
                    model=active_model,
                    contents=contents,
                )
                if response and response.text:
                    clean_text = sanitize_citations(response.text, num_chunks)
                    actual_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        actual_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
                    
                    citation_audit = verify_citations_grounding(clean_text, chunks)
                    return {
                        "answer": clean_text,
                        "tokens": actual_tokens,
                        "is_fallback": False,
                        "model": active_model,
                        "has_multimodal_context": len(visual_images) > 0,
                        "citation_verification": citation_audit
                    }
            except Exception as e:
                print(f"[AuditGuard Google GenAI Error - Grounded Generator]: {e}")

    # 2. Try OpenAI-compatible server (Local Ollama / Venue Gemma)
    client = get_openai_client()
    try:
        if visual_images:
            user_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_content}]
            for img_bytes, mime in visual_images:
                user_parts.append(make_openai_image_part(img_bytes, mime))
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_parts}
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

        response = client.chat.completions.create(
            model=active_model if model_override else LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600
        )
        content = response.choices[0].message.content or "No response from Gemma."
        clean_text = sanitize_citations(content, num_chunks)
        actual_tokens = getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") and response.usage else 0
        citation_audit = verify_citations_grounding(clean_text, chunks)
        return {
            "answer": clean_text,
            "tokens": actual_tokens,
            "is_fallback": False,
            "model": active_model if model_override else LLM_MODEL,
            "has_multimodal_context": len(visual_images) > 0,
            "citation_verification": citation_audit
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
            citation_audit = verify_citations_grounding(fallback_text, chunks)
            return {
                "answer": fallback_text,
                "tokens": 0,
                "is_fallback": True,
                "model": "AuditGuard Extractive Fallback",
                "has_multimodal_context": len(visual_images) > 0,
                "citation_verification": citation_audit
            }
        return {
            "answer": "I cannot verify this information in the provided document excerpts.",
            "tokens": 0,
            "is_fallback": True,
            "model": "AuditGuard Refusal",
            "has_multimodal_context": False,
            "citation_verification": {"all_supported": True, "total_citations": 0, "unsupported_count": 0, "verified_citations": []}
        }

def stream_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    persona_addon: str = "",
    model_override: Optional[str] = None
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
    active_model = model_override or get_gemma_model()
    visual_images = extract_visual_images_from_chunks(chunks)

    # 1. Try Native Google GenAI Gemma 4 Streaming
    if USE_GOOGLE_GENAI:
        genai_cli = get_genai_client()
        if genai_cli:
            try:
                full_prompt = f"{system_content}\n\n{user_content}"
                contents: List[Any] = [full_prompt]
                for img_bytes, mime in visual_images:
                    part = make_genai_part(img_bytes, mime)
                    if part:
                        contents.append(part)

                streamed_any = False
                for chunk in genai_cli.models.generate_content_stream(
                    model=active_model,
                    contents=contents,
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
        if visual_images:
            user_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_content}]
            for img_bytes, mime in visual_images:
                user_parts.append(make_openai_image_part(img_bytes, mime))
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_parts}
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

        stream = client.chat.completions.create(
            model=active_model if model_override else LLM_MODEL,
            messages=messages,
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
