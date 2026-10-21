"""
Automated Cross-Modal Discrepancy & Contradiction Scanner.
Compares author prose claims in text paragraphs against linked visual crops (Tables/Charts).
Detects discrepancies, SLA breaches, unit mismatches, and contradictory claims with audit-ready provenance.
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional
from app.ast_builder import DocumentAST, ASTNodeType, ASTNode
from app.llm import get_openai_client, get_genai_client, is_llm_configured, has_valid_openai_endpoint
from app.config import LLM_MODEL, USE_GOOGLE_GENAI, get_active_gemma_model

GROWTH_PATTERN = re.compile(
    r'\b(?:growth|grew|increased|increases|surged|expanded|improved|gain|gains|rose|risen)\b',
    re.IGNORECASE
)
DECLINE_PATTERN = re.compile(
    r'\b(?:decline|declined|dropped|fell|decreased|decreases|loss|contraction|shrank)\b|(?<!\w)-\d+(?:\.\d+)?\s*%',
    re.IGNORECASE
)
PCT_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s*%')


def _analyze_pipe_table_vs_claim(text: str, visual_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses pipe-delimited Markdown tables in visual_text and checks whether any row
    contradicts or breaches a numerical/percentage guarantee stated in the prose paragraph.
    """
    text_pcts = [float(x) for x in PCT_PATTERN.findall(text)]
    if not text_pcts:
        return None

    lines = [ln.strip() for ln in visual_text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return None

    headers = [c.strip().lower() for c in lines[0].strip("|").split("|")]
    # Identify target/SLA column vs actual/recorded column if present
    actual_col_idx = None
    for idx, h in enumerate(headers):
        if any(k in h for k in ("actual", "recorded", "observed", "measured", "result", "performance")):
            actual_col_idx = idx
            break

    claimed_pct = text_pcts[0]
    breach_rows = []

    for row_line in lines[1:]:
        if re.match(r'^\|[\s\-:|]+\|$', row_line):
            continue
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        if not cells:
            continue

        row_label = cells[0]
        if actual_col_idx is not None and actual_col_idx < len(cells):
            cell_pcts = [float(x) for x in PCT_PATTERN.findall(cells[actual_col_idx])]
            if cell_pcts:
                actual_val = cell_pcts[0]
                delta = round(abs(claimed_pct - actual_val), 2)
                if delta >= 0.5 or "breach" in row_line.lower():
                    breach_rows.append((row_label, actual_val, delta, row_line))
        else:
            row_pcts = [float(x) for x in PCT_PATTERN.findall(row_line)]
            for val in row_pcts:
                if val > 0 and abs(claimed_pct - val) >= 1.0 and val != claimed_pct:
                    breach_rows.append((row_label, val, round(abs(claimed_pct - val), 2), row_line))

    if breach_rows:
        # Select the most severe deviation row
        worst_label, worst_val, worst_delta, worst_raw = max(breach_rows, key=lambda x: x[2])
        return {
            "status": "CONTRADICTION",
            "title": f"Table Metric Variance ({worst_label})",
            "text_claim": text.strip()[:160],
            "visual_fact": f"{worst_label}: {worst_val}% recorded in table ({worst_raw})",
            "discrepancy_delta": (
                f"Prose specifies {claimed_pct}%, whereas linked table records {worst_val}% "
                f"for {worst_label} (variance of {worst_delta:.2f}%)."
            ),
            "confidence": 0.94,
        }

    return None


def detect_heuristic_discrepancy(text: str, visual_text: str) -> Optional[Dict[str, Any]]:
    """
    Deterministic cross-modal verifier to detect text-vs-visual contradictions:
    1. Directional polarity conflict (prose claims growth/surge while visual reports decline/negative %);
    2. Structured table row breach against prose SLA/target percentage;
    3. Direct percentage variance between prose claim and chart figures.
    Uses strict word-boundary regexes to prevent false positives on substrings.
    """
    if not text or not visual_text:
        return None

    # Pattern 1: Directional Growth Claim vs Negative/Decline Visual Data (whole-word matched)
    text_claims_growth = bool(GROWTH_PATTERN.search(text))
    vis_shows_decline = bool(DECLINE_PATTERN.search(visual_text))

    if text_claims_growth and vis_shows_decline:
        text_pcts = PCT_PATTERN.findall(text)
        vis_pcts = PCT_PATTERN.findall(visual_text)
        delta_detail = "Text reports positive growth/expansion while linked visual data reflects a contraction/decline."
        if text_pcts and vis_pcts:
            delta_detail = (
                f"Prose claims positive growth ({text_pcts[0]}%), whereas linked visual reports "
                f"decline/contraction ({', '.join(p + '%' for p in vis_pcts[:3])})."
            )
        return {
            "status": "CONTRADICTION",
            "title": "Growth Metric Mismatch",
            "text_claim": text.strip()[:160],
            "visual_fact": visual_text.strip().splitlines()[0][:160] if "\n" not in visual_text else visual_text.strip()[:160],
            "discrepancy_delta": delta_detail,
            "confidence": 0.88,
        }

    # Pattern 2: Structured Markdown/ASCII Table vs Prose Claim
    if "|" in visual_text:
        tbl_discrepancy = _analyze_pipe_table_vs_claim(text, visual_text)
        if tbl_discrepancy:
            return tbl_discrepancy

    # Pattern 3: Explicit Percentage Conflict on Non-Table Visuals
    text_pcts = [float(x) for x in PCT_PATTERN.findall(text)]
    vis_pcts = [float(x) for x in PCT_PATTERN.findall(visual_text)]
    if text_pcts and vis_pcts:
        t_val = text_pcts[0]
        # Check if the prose percentage matches ANY percentage in the visual within 2.0%
        closest_v = min(vis_pcts, key=lambda v: abs(t_val - v))
        if abs(t_val - closest_v) >= 2.0:
            return {
                "status": "CONTRADICTION",
                "title": "Percentage Variance Discrepancy",
                "text_claim": f"Text quotes {t_val}% ({text.strip()[:120]})",
                "visual_fact": f"Linked visual quotes {closest_v}% ({visual_text.strip()[:120]})",
                "discrepancy_delta": f"Numerical variance of {abs(t_val - closest_v):.1f}% detected between prose ({t_val}%) and visual ({closest_v}%).",
                "confidence": 0.92,
            }

    return None


def audit_cross_modal_pair(
    p_node: ASTNode,
    v_node: ASTNode,
    use_llm: bool = False
) -> Dict[str, Any]:
    """
    Analyzes a linked Text Paragraph and Visual Node pair for consistency.
    Combines deterministic numerical/table verification with optional LLM audit.
    """
    heuristic = detect_heuristic_discrepancy(p_node.text, v_node.text)
    if heuristic:
        return {
            "paragraph_id": p_node.id,
            "visual_id": v_node.id,
            "page": p_node.page,
            "image_url": v_node.image_url,
            **heuristic
        }

    has_substantive_visual_text = len(v_node.text.strip()) > len(v_node.title.strip()) + 10

    if use_llm and has_substantive_visual_text and is_llm_configured():
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

        raw = ""
        if USE_GOOGLE_GENAI:
            genai_cli = get_genai_client()
            if genai_cli:
                try:
                    resp = genai_cli.models.generate_content(
                        model=get_active_gemma_model(),
                        contents=f"{system_prompt}\n\n{user_content}",
                    )
                    raw = resp.text or ""
                except Exception:
                    raw = ""

        if not raw and has_valid_openai_endpoint():
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
            except Exception:
                raw = ""

        if raw:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if parsed.get("status") in ("ALIGN", "CAUTION", "CONTRADICTION"):
                        return {
                            "paragraph_id": p_node.id,
                            "visual_id": v_node.id,
                            "page": p_node.page,
                            "image_url": v_node.image_url,
                            **parsed
                        }
                except Exception:
                    pass

    aligned_conf = 0.90 if has_substantive_visual_text else 0.65

    return {
        "paragraph_id": p_node.id,
        "visual_id": v_node.id,
        "page": p_node.page,
        "image_url": v_node.image_url,
        "status": "ALIGN",
        "title": "Cross-Modal Alignment Verified" if has_substantive_visual_text else "Visual Link Indexed (No Conflict)",
        "text_claim": p_node.text[:140] + ("..." if len(p_node.text) > 140 else ""),
        "visual_fact": v_node.text[:140] + ("..." if len(v_node.text) > 140 else ""),
        "discrepancy_delta": "No contradictions detected between prose and visual data.",
        "confidence": aligned_conf,
    }


def scan_ast_for_discrepancies(ast: DocumentAST, use_llm: bool = False) -> Dict[str, Any]:
    """
    Full AST Discrepancy Scan:
    Audits every cross-modal sibling link between prose paragraphs and visuals in the document.
    Deduplicates identical visual-table self-mirroring nodes when a prose claim is already linked.
    """
    findings: List[Dict[str, Any]] = []
    links = ast.get_cross_modal_links()

    # Group links by visual_id so we prioritize auditing distinct prose paragraphs against the visual
    pairs_to_audit = []
    for link in links:
        source_node = ast.get_node(link["source_id"])
        target_node = ast.get_node(link["target_id"])
        if not source_node or not target_node:
            continue

        p_node = source_node if source_node.type == ASTNodeType.PARAGRAPH else target_node
        v_node = target_node if target_node.type == ASTNodeType.VISUAL else source_node

        if p_node.type == ASTNodeType.PARAGRAPH and v_node.type == ASTNodeType.VISUAL:
            pairs_to_audit.append((p_node, v_node))

    # Filter out synthetic mirror paragraphs (where p_node.text == v_node.text) if v_node is also linked to real prose
    visual_prose_counts: Dict[str, int] = {}
    for p_node, v_node in pairs_to_audit:
        if p_node.text.strip() != v_node.text.strip():
            visual_prose_counts[v_node.id] = visual_prose_counts.get(v_node.id, 0) + 1

    for p_node, v_node in pairs_to_audit:
        if p_node.text.strip() == v_node.text.strip() and visual_prose_counts.get(v_node.id, 0) > 0:
            continue
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
        "contradictions_count": contradictions_count,
        "caution_count": caution_count,
        "aligned_count": aligned_count,
        "consistency_score": consistency_score,
        "verdict": "FLAGGED" if contradictions_count > 0 else ("CAUTION" if caution_count > 0 else "CONSISTENT"),
        "findings": findings
    }
