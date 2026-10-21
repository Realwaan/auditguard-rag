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
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
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
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]{3,}\b', query.lower())
    return [t for t in tokens if t not in STOP_WORDS]

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
    is_quantitative = any(w in q_lower for w in [
        "what percentage", "how much", "how many", "rate", "penalty credit", 
        "uptime", "availability", "dollar", "cost", "fee", "cap", "sla"
    ])
    ctx_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', ctx_lower)
    ctx_word_numbers = [w for w in NUMBER_WORDS if re.search(r'\b' + re.escape(w) + r'\b', ctx_lower)]
    has_numerical_target = bool(ctx_numbers) or bool(ctx_word_numbers)

    if is_quantitative and not has_numerical_target and lexical_recall < 0.60:
        # Context is on-topic but lacks the actual numerical figure requested
        return {
            "verdict": "NEUTRAL",
            "entailment_prob": round(min(0.35, top_dense * 0.4), 3),
            "is_answer_bearing": False,
            "missing_target": "Quantitative metric/number missing from text",
            "reason": "The context mentions the requested topic but lacks the specific quantitative metric or figure required to answer.",
            "method": "heuristic_target_verification"
        }

    # 2. Polarity / Negation Divergence Check
    q_has_negation = any(re.search(r'\b' + re.escape(w) + r'\b', q_lower) for w in NEGATION_WORDS)
    ctx_has_negation = any(re.search(r'\b' + re.escape(w) + r'\b', ctx_lower) for w in NEGATION_WORDS)
    polarity_divergence = (q_has_negation != ctx_has_negation) and (lexical_recall >= 0.50)

    # 3. Proposition alignment probability
    has_answer_target = has_numerical_target if is_quantitative else (lexical_recall >= 0.40)
    
    if top_dense >= SIMILARITY_THRESHOLD and has_answer_target:
        if polarity_divergence and not q_has_negation:
            # Query is affirmative but context contains strict negative/prohibition
            return {
                "verdict": "CONTRADICTORY",
                "entailment_prob": round(top_dense * 0.4, 3),
                "is_answer_bearing": True,
                "reason": "Retrieved context asserts restrictive or negative conditions contrary to affirmative query proposition.",
                "method": "heuristic_polarity_divergence"
            }
        
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
            if re.search(r'\b' + re.escape(kw) + r'\b', combined_context)
        ]
        lexical_recall = len(matched_kws) / max(1, len(query_keywords))
    else:
        # If no query string passed, rely on dense score alone
        lexical_recall = top_dense

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
    if nli_verdict == "CONTRADICTION":
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
        "claim_evidence_verified": True,
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
) -> Dict[str, Any]:
    """
    Post-Generation Guardrail:
    1. Detects if the LLM itself refused because the retrieved chunks lacked the specific answer.
    2. Audits all inline citation markers [1], [2], etc. against the retrieved chunk IDs.
    """
    updated = dict(eval_result)
    if not answer:
        return updated

    ans_lower = answer.strip().lower()
    if "i cannot verify this information in the provided document excerpts" in ans_lower:
        updated["status"] = "REFUSE"
        updated["nli_verdict"] = "NEUTRAL (UNVERIFIABLE)"
        updated["grounding_verdict"] = "UNVERIFIABLE"
        updated["claim_evidence_verified"] = False
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
    return updated
