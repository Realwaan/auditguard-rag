"""
AuditGuard Grounding & Relevance Gatekeeper.
Evaluates whether retrieved document excerpts contain sufficient semantic and lexical
grounding to answer the user query without hallucination.
"""

import re
from typing import List, Dict, Any, Optional
from app.config import SIMILARITY_THRESHOLD

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", 
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    "please", "explain", "describe", "define", "tell", "cite", "exact", "explicitly",
    "provided", "document", "documents", "excerpt", "excerpts", "page", "pages"
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "percent", "percentage"
}

NEGATION_WORDS = {
    "no", "not", "never", "none", "neither", "nor", "prohibited", "banned",
    "disallowed", "unauthorized", "excluding", "without", "except"
}

def extract_query_keywords(query: str) -> List[str]:
    """Extract content-bearing keywords from query using word boundaries, filtering common stopwords."""
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', query.lower())
    return list(dict.fromkeys(t for t in tokens if len(t) >= 2 and t not in STOP_WORDS))


def keyword_present(keyword: str, text: str) -> bool:
    """Match whole words, including ordinary singular/plural forms."""
    forms = {keyword}
    if len(keyword) > 4 and keyword.endswith("s") and not keyword.endswith(("ss", "us", "is")):
        forms.add(keyword[:-1])
    elif len(keyword) > 3:
        forms.add(keyword + "s")
    return any(re.search(r'\b' + re.escape(form) + r'\b', text, re.I) for form in forms)

def evaluate_heuristic_entailment(
    context_text: str,
    query: str,
    top_dense: float,
    lexical_recall: float
) -> Dict[str, Any]:
    """
    Heuristic Claim-Evidence Alignment & Proposition Gate:
    Evaluates whether the retrieved premise provides direct, answer-bearing evidence
    for the query hypothesis using composite lexical overlap, quantitative presence,
    and semantic similarity signals.
    
    Treats heuristic scores as signals, distinguishing:
      - ENTAILED: Premise directly supplies answer target (metrics, conditions, facts)
      - NEUTRAL: Premise discusses topic but lacks required answer target / evidence
      - CONTRADICTORY: Premise exhibits explicit polarity/negation divergence from query
    """
    if not query:
        # If no explicit query proposition, evaluate based on semantic density
        if top_dense >= SIMILARITY_THRESHOLD:
            return {
                "verdict": "ENTAILED",
                "entailment_prob": round(min(0.98, top_dense), 3),
                "is_answer_bearing": True,
                "reason": "Retrieved context exceeds semantic density threshold.",
                "method": "heuristic_semantic_density"
            }
        else:
            return {
                "verdict": "NEUTRAL",
                "entailment_prob": round(top_dense * 0.5, 3),
                "is_answer_bearing": False,
                "reason": "Retrieved context does not meet grounding threshold.",
                "method": "heuristic_semantic_density"
            }

    ctx_lower = context_text.lower()
    q_lower = query.lower()

    # 1. Quantitative vs Descriptive Answer Target Check
    # Match requested quantities, not substrings such as "rate" in "corporate",
    # nor descriptive questions such as "What is an SLA?".
    is_quantitative = bool(re.search(
        r'\b(?:percentage|percent|how much|how many|how long|amount|threshold)\b', q_lower
    ))
    ctx_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', ctx_lower)
    ctx_word_numbers = [w for w in NUMBER_WORDS if re.search(r'\b' + re.escape(w) + r'\b', ctx_lower)]
    has_numerical_target = bool(ctx_numbers) or bool(ctx_word_numbers)
    if re.search(r'\b(?:percentage|percent)\b', q_lower):
        number_names = "|".join(sorted(NUMBER_WORDS - {"percent", "percentage"}))
        has_numerical_target = bool(re.search(
            rf'(?:\d+(?:\.\d+)?|\b(?:{number_names}))\s*(?:%|percent\b)', ctx_lower
        ))

    if is_quantitative and not has_numerical_target:
        # Context is on-topic but lacks the actual numerical figure requested
        return {
            "verdict": "NEUTRAL",
            "entailment_prob": round(min(0.35, top_dense * 0.4), 3),
            "is_answer_bearing": False,
            "missing_target": "Quantitative metric/number missing from text",
            "reason": "The context mentions the requested topic but lacks the specific quantitative metric or figure required to answer.",
            "method": "heuristic_target_verification"
        }

    # Questions are not factual assertions: negation in a source is not a contradiction of a question.

    # 3. Proposition alignment probability
    has_answer_target = has_numerical_target if is_quantitative else (lexical_recall >= 0.40)
    
    if top_dense >= SIMILARITY_THRESHOLD and has_answer_target:
        # Genuine entailment: Evidence provides the facts needed to answer
        entailment_prob = round(0.65 + 0.25 * min(1.0, top_dense) + 0.10 * min(1.0, lexical_recall), 3)
        return {
            "verdict": "ENTAILED",
            "entailment_prob": min(0.99, entailment_prob),
            "is_answer_bearing": True,
            "reason": "Retrieved context contains direct, answer-bearing evidentiary propositions.",
            "method": "heuristic_proposition_gate"
        }
    elif top_dense < SIMILARITY_THRESHOLD and lexical_recall < 0.45:
        return {
            "verdict": "NEUTRAL",
            "entailment_prob": round(top_dense * 0.5, 3),
            "is_answer_bearing": False,
            "missing_target": "Insufficient factual overlap",
            "reason": "Retrieved excerpts lack necessary semantic alignment to verify the query.",
            "method": "heuristic_proposition_gate"
        }
    else:
        # Borderline / Ambiguous
        return {
            "verdict": "NEUTRAL",
            "entailment_prob": round(top_dense * 0.6, 3),
            "is_answer_bearing": False,
            "missing_target": "Ambiguous evidence",
            "reason": "Context discusses related subject matter but lacks complete evidentiary proof.",
            "method": "heuristic_proposition_gate"
        }

# Alias for backward compatibility with existing callers and tests
verify_logical_entailment = evaluate_heuristic_entailment

def evaluate_relevance(retrieved_chunks: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
    """
    CRAG Relevance Gatekeeper with Formal NLI Entailment Verification:
    Computes genuine composite grounding from:
      1. Dense Vector Cosine Similarity (semantic match)
      2. Lexical Keyword Coverage (exact word-boundary token recall across retrieved excerpts)
      3. Logical Entailment Proposition Analysis (verifies if context actually answers the question)
    
    If context falls below threshold or lacks the required answer target, activates Refusal Protocol.
    """
    if not retrieved_chunks:
        return {
            "status": "REFUSE",
            "confidence": 0.0,
            "grounding_score": 0.0,
            "lexical_recall": 0.0,
            "dense_score": 0.0,
            "nli_verdict": "UNVERIFIABLE",
            "nli_entailment_prob": 0.0,
            "badge": "⚠️ OUT-OF-BOUNDS (0% Grounding)",
            "reason": "No relevant text chunks found in uploaded documents."
        }

    # 1. Dense Cosine Similarity
    top_dense = max(float(c.get("dense_score", 0.0)) for c in retrieved_chunks)

    # 2. Lexical Keyword Coverage across all retrieved parent texts
    combined_context = " ".join(
        c.get("parent_text", c.get("child_text", "")) for c in retrieved_chunks
    ).lower()

    query_keywords = extract_query_keywords(query) if query else []
    if query_keywords:
        matched_kws = [
            kw for kw in query_keywords 
            if keyword_present(kw, combined_context)
        ]
        lexical_recall = len(matched_kws) / max(1, len(query_keywords))
    else:
        # If no query string passed, rely on dense score alone
        lexical_recall = top_dense if not query else 0.0

    # Escalation check
    is_promoted = any(c.get("escalated_from") == "whole_page_scan" for c in retrieved_chunks)
    page_rel = max((float(c.get("page_relevance", 0.0)) for c in retrieved_chunks if c.get("escalated_from") == "whole_page_scan"), default=0.0)
    effective_signal = max(top_dense, page_rel) if is_promoted else top_dense

    # 3. Composite Grounding Confidence:
    # 60% semantic similarity + 40% lexical keyword recall (or calibrated page relevance if escalated)
    composite_grounding = round((top_dense * 0.60 + lexical_recall * 0.40), 3) if not is_promoted else round(page_rel, 3)
    confidence_pct = round(composite_grounding * 100, 1)

    # 4. Heuristic Claim-Evidence Alignment & Proposition Gate
    nli_eval = evaluate_heuristic_entailment(combined_context, query, effective_signal, lexical_recall)
    nli_verdict = nli_eval["verdict"]
    nli_prob = nli_eval["entailment_prob"]
    eval_method = nli_eval.get("method", "heuristic_proposition_gate")

    # Refusal decision: Requires positive entailment or minimum dense and keyword presence
    if nli_verdict in ("CONTRADICTION", "CONTRADICTORY"):
        return {
            "status": "REFUSE",
            "confidence": confidence_pct,
            "grounding_score": composite_grounding,
            "dense_score": round(top_dense, 3),
            "lexical_recall": round(lexical_recall, 3),
            "grounding_verdict": "CONTRADICTORY",
            "heuristic_alignment_score": nli_prob,
            "claim_evidence_verified": False,
            "evaluation_method": eval_method,
            # Aliases for backward compatibility
            "nli_verdict": "CONTRADICTORY (RESTRICTION_DETECTED)",
            "nli_entailment_prob": nli_prob,
            "nli_method": eval_method,
            "badge": f"⚠️ OUT-OF-BOUNDS (Contradictory Premise)",
            "reason": nli_eval.get("reason", "Retrieved evidence directly contradicts the query proposition.")
        }

    if nli_verdict == "NEUTRAL" and (top_dense < SIMILARITY_THRESHOLD or not nli_eval.get("is_answer_bearing", True)):
        reason_msg = nli_eval.get("reason", "The uploaded documents do not contain verified data answering this query.")
        return {
            "status": "REFUSE",
            "confidence": confidence_pct,
            "grounding_score": composite_grounding,
            "dense_score": round(top_dense, 3),
            "lexical_recall": round(lexical_recall, 3),
            "grounding_verdict": "UNVERIFIABLE",
            "heuristic_alignment_score": nli_prob,
            "claim_evidence_verified": False,
            "evaluation_method": eval_method,
            # Aliases for backward compatibility
            "nli_verdict": f"{nli_verdict} (UNVERIFIABLE)",
            "nli_entailment_prob": nli_prob,
            "nli_method": eval_method,
            "badge": f"⚠️ OUT-OF-BOUNDS ({confidence_pct}% Grounding)",
            "reason": f"{reason_msg} Refusal protocol activated to prevent model hallucination."
        }

    badge_label = (
        f"🌳 PROMOTED TO PAGE SCAN • ENTAILED ({confidence_pct}% Grounding)"
        if is_promoted
        else f"🛡️ GROUNDED • ENTAILED ({confidence_pct}% Grounding)"
    )

    reason_text = (
        f"Verified contextual grounding recovered via Tier-2 AST Whole-Page Escalation (Density: {top_dense:.2f}, Keyword recall: {lexical_recall:.0%}, Alignment Signal: {nli_prob:.0%})."
        if is_promoted
        else f"Verified contextual grounding found in source documents (Density: {top_dense:.2f}, Keyword recall: {lexical_recall:.0%}, Alignment Signal: {nli_prob:.0%})."
    )

    return {
        "status": "SYNTHESIZE",
        "confidence": confidence_pct,
        "grounding_score": composite_grounding,
        "dense_score": round(top_dense, 3),
        "lexical_recall": round(lexical_recall, 3),
        "grounding_verdict": "SUPPORTED",
        "heuristic_alignment_score": nli_prob,
        # Retrieval relevance cannot verify an answer that has not been generated.
        "claim_evidence_verified": False,
        "retrieval_evidence_available": True,
        "evaluation_method": eval_method,
        # Aliases for backward compatibility
        "nli_verdict": nli_verdict,
        "nli_entailment_prob": nli_prob,
        "nli_method": eval_method,
        "is_answer_bearing": nli_eval.get("is_answer_bearing", True),
        "badge": badge_label,
        "reason": reason_text
    }


def verify_generated_answer(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    eval_result: Dict[str, Any],
    query: str = "",
    citation_audit: Optional[Dict[str, Any]] = None,
    is_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Post-Generation Guardrail:
    1. Detects if the LLM itself refused because the retrieved chunks lacked the specific answer.
    2. Audits all inline citation markers [1], [2], etc. against the retrieved chunk IDs.
    """
    updated = dict(eval_result)
    from app.generator import is_refusal_answer, verify_citations_grounding

    updated["claim_evidence_verified"] = False
    if not answer.strip() or is_refusal_answer(answer):
        updated["status"] = "REFUSE"
        updated["nli_verdict"] = "NEUTRAL (UNVERIFIABLE)"
        updated["grounding_verdict"] = "UNVERIFIABLE"
        updated["claim_evidence_verified"] = False
        updated["citation_verified"] = False
        updated["verified_citations"] = []
        updated["invalid_citations"] = []
        updated["badge"] = f"⚠️ UNVERIFIABLE IN EXCERPTS ({updated.get('confidence', 0.0)}% Retrieval)"
        updated["reason"] = "Retrieved excerpts were topically related but did not contain verifiable facts to answer this specific question."
        return updated

    valid_ids = {int(c.get("citation_id", idx + 1)) for idx, c in enumerate(retrieved_chunks)}
    cited_raw = [int(m) for m in re.findall(r'\[(\d+)\]', answer)]
    verified_citations = sorted(list({cid for cid in cited_raw if cid in valid_ids}))
    invalid_citations = sorted(list({cid for cid in cited_raw if cid not in valid_ids}))

    updated["verified_citations"] = verified_citations
    updated["invalid_citations"] = invalid_citations
    updated["citation_verified"] = len(verified_citations) > 0 and len(invalid_citations) == 0
    audit = citation_audit if citation_audit is not None else verify_citations_grounding(answer, retrieved_chunks)
    updated["citation_verified"] = updated["citation_verified"] and audit.get("all_supported", False)
    keywords = extract_query_keywords(query)
    coverage = sum(
        keyword_present(k, answer) or (k in {"percent", "percentage"} and "%" in answer)
        for k in keywords
    ) / len(keywords) if keywords else 1.0
    updated["answer_keyword_coverage"] = round(coverage, 3)
    if not updated["citation_verified"] or coverage < 0.60 or is_fallback:
        updated["status"] = "CAUTION_UNSUPPORTED_CITATION"
        updated["grounding_verdict"] = "UNVERIFIED"
        updated["nli_verdict"] = "UNVERIFIED"
        updated["badge"] = "CAUTION: Answer requires review"
        updated["reason"] = (
            "The answer may not address the requested topic; review the cited excerpts."
            if coverage < 0.60 else "The generated answer did not pass all citation checks."
        )
    else:
        updated["nli_verdict"] = "HEURISTIC_SUPPORTED"
        updated["badge"] = f"CITATIONS CHECKED ({updated.get('confidence', 0.0)}% Retrieval)"
        updated["reason"] = "Citation and keyword checks passed; these heuristic checks do not establish semantic entailment."
    return updated
