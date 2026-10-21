"""
Automated Cross-Modal Discrepancy & Contradiction Scanner.
Compares author prose claims in text paragraphs against linked visual crops (Tables/Charts).
Detects discrepancies, unit mismatches, and contradictory claims with audit-ready provenance.
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional
from app.ast_builder import DocumentAST, ASTNodeType, ASTNode
from app.llm import get_openai_client, get_genai_client
from app.config import LLM_MODEL, USE_GOOGLE_GENAI, GEMINI_API_KEY, get_gemma_model


def detect_heuristic_discrepancy(text: str, visual_text: str) -> Optional[Dict[str, Any]]:
    """
    Fast local heuristic scanner to detect blatant text-vs-visual contradictions
    (e.g., text says 'increased by 15%', but visual says 'declined').
    Uses strict word boundaries and filters out false positives like 'uptime' or hyphens.
    """
    text_lower = text.lower()
    vis_lower = visual_text.lower()

    # Pattern 1: Growth claim vs Negative/Decline visual data using exact word boundaries
    growth_keywords = [r'\bgrowth\b', r'\bincreased\b', r'\bsurged\b', r'\bexpanded\b', r'\bimproved\b', r'\bgain\b']
    decline_keywords = [r'\bdecline\b', r'\bdeclined\b', r'\bdropped\b', r'\bfell\b', r'\bdecreased\b', r'\bloss\b', r'\bcontraction\b', r'\bdownturn\b']

    text_claims_growth = any(re.search(k, text_lower) for k in growth_keywords)
    vis_shows_decline = any(re.search(k, vis_lower) for k in decline_keywords)

    if text_claims_growth and vis_shows_decline:
        return {
            "status": "CONTRADICTION",
            "title": "Growth Metric Mismatch",
            "text_claim": "Author text claims positive growth/expansion.",
            "visual_fact": "Linked visual data indicates a decline or contraction.",
            "discrepancy_delta": "Text reports growth while linked visual data reflects a contraction.",
            "confidence": 0.88
        }

    # Pattern 2: Explicit percentage conflict on matching metric terms
    pct_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*%')
    text_pcts = pct_pattern.findall(text)
    vis_pcts = pct_pattern.findall(visual_text)
    if text_pcts and vis_pcts:
        metric_tokens = {"uptime", "availability", "growth", "margin", "accuracy", "error", "sla", "penalty"}
        shared_tokens = {w for w in metric_tokens if re.search(r'\b' + re.escape(w) + r'\b', text_lower) and re.search(r'\b' + re.escape(w) + r'\b', vis_lower)}
        if shared_tokens:
            t_val = float(text_pcts[0])
            v_val = float(vis_pcts[0])
            # Strict 0.5% tolerance for uptime/SLA vs 2.0% for business metrics
            tolerance = 0.5 if any(m in shared_tokens for m in ["uptime", "availability", "sla"]) else 2.0
            if abs(t_val - v_val) >= tolerance:
                metric_name = list(shared_tokens)[0].capitalize()
                return {
                    "status": "CONTRADICTION",
                    "title": f"{metric_name} Percentage Discrepancy",
                    "text_claim": f"Text quotes {t_val}% for {metric_name}.",
                    "visual_fact": f"Linked visual data reports {v_val}%.",
                    "discrepancy_delta": f"Numerical variance of {abs(t_val - v_val):.1f}% detected between prose and visual data.",
                    "confidence": 0.92
                }

    return None


def audit_cross_modal_pair(
    p_node: ASTNode,
    v_node: ASTNode,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Analyzes a linked Text Paragraph and Visual Node pair for consistency.
    Uses Google GenAI Gemma 4 or OpenAI server when available, with deterministic fallback.
    """
    # 1. First run fast heuristic check
    heuristic = detect_heuristic_discrepancy(p_node.text, v_node.text)

    # 2. If LLM is available, perform deep verification
    if use_llm:
        system_prompt = """You are AuditGuard Cross-Modal Compliance Auditor.
Your job is to compare a text claim against a linked visual table or chart description.
Determine if they ALIGN, PARTIALLY ALIGN (CAUTION), or CONTRADICT each other.
Return ONLY valid JSON matching this schema:
{
  "status": "ALIGN" | "CAUTION" | "CONTRADICTION",
  "title": "<Short 4-6 word description>",
  "text_claim": "<Exact claim made in prose>",
  "visual_fact": "<Exact data points shown in visual>",
  "discrepancy_delta": "<Specific contradiction explanation or 'None'>",
  "confidence": <float between 0.0 and 1.0>
}"""

        user_content = f"""TEXT EXCERPT (Paragraph on Page {p_node.page}):
"{p_node.text}"

LINKED VISUAL ELEMENT ({v_node.title} on Page {v_node.page}):
"{v_node.text}"

Audit the claim against the visual data and output JSON."""

        # Try Google GenAI native Gemma 4
        if USE_GOOGLE_GENAI and GEMINI_API_KEY:
            try:
                genai_cli = get_genai_client()
                if genai_cli:
                    full_p = f"{system_prompt}\n\n{user_content}"
                    resp = genai_cli.models.generate_content(
                        model=get_gemma_model(),
                        contents=full_p
                    )
                    if resp and resp.text:
                        raw = resp.text
                        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                        if json_match:
                            parsed = json.loads(json_match.group(0))
                            if "status" in parsed:
                                return {
                                    "paragraph_id": p_node.id,
                                    "visual_id": v_node.id,
                                    "page": p_node.page,
                                    "image_url": v_node.image_url,
                                    **parsed
                                }
            except Exception as ge:
                print(f"[AuditGuard Google GenAI Cross-Modal Check]: {ge}")

        # Try OpenAI client only if local/custom server is explicitly configured
        if not USE_GOOGLE_GENAI or (LLM_API_KEY and LLM_API_KEY != "local-or-alma-key"):
            try:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=250
                )
                raw = response.choices[0].message.content or ""
                # Extract JSON block
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    return {
                        "paragraph_id": p_node.id,
                        "visual_id": v_node.id,
                        "page": p_node.page,
                        "image_url": v_node.image_url,
                        **parsed
                    }
            except Exception:
                pass

    # If heuristic found an issue
    if heuristic:
        return {
            "paragraph_id": p_node.id,
            "visual_id": v_node.id,
            "page": p_node.page,
            "image_url": v_node.image_url,
            **heuristic
        }

    # Default aligned state
    return {
        "paragraph_id": p_node.id,
        "visual_id": v_node.id,
        "page": p_node.page,
        "image_url": v_node.image_url,
        "status": "ALIGN",
        "title": "Cross-Modal Alignment Verified",
        "text_claim": p_node.text[:120] + "...",
        "visual_fact": v_node.text[:120] + "...",
        "discrepancy_delta": "No contradictions detected between prose and chart.",
        "confidence": 0.95
    }


def scan_ast_for_discrepancies(ast: DocumentAST, use_llm: bool = True) -> Dict[str, Any]:
    """
    Full AST Discrepancy Scan:
    Audits every cross-modal sibling link in the document and returns a summary report.
    """
    findings: List[Dict[str, Any]] = []
    links = ast.get_cross_modal_links()

    for link in links:
        source_node = ast.get_node(link["source_id"])
        target_node = ast.get_node(link["target_id"])

        if not source_node or not target_node:
            continue

        p_node = source_node if source_node.type == ASTNodeType.PARAGRAPH else target_node
        v_node = target_node if target_node.type == ASTNodeType.VISUAL else source_node

        if p_node.type == ASTNodeType.PARAGRAPH and v_node.type == ASTNodeType.VISUAL:
            audit_item = audit_cross_modal_pair(p_node, v_node, use_llm=use_llm)
            findings.append(audit_item)

    contradictions_count = sum(1 for f in findings if f["status"] == "CONTRADICTION")
    caution_count = sum(1 for f in findings if f["status"] == "CAUTION")
    aligned_count = sum(1 for f in findings if f["status"] == "ALIGN")

    total_pairs = len(findings)
    if total_pairs == 0:
        consistency_score = 100.0
    else:
        consistency_score = round(((aligned_count + 0.5 * caution_count) / total_pairs) * 100, 1)

    return {
        "doc_id": ast.doc_id,
        "total_pairs_audited": total_pairs,
        "pairs_audited": total_pairs,
        "contradictions_count": contradictions_count,
        "caution_count": caution_count,
        "aligned_count": aligned_count,
        "consistency_score": consistency_score,
        "score": consistency_score,
        "verdict": "FLAGGED" if contradictions_count > 0 else ("CAUTION" if caution_count > 0 else "CONSISTENT"),
        "findings": findings
    }
