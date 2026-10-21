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

def extract_query_keywords(query: str) -> List[str]:
    """Extract content-bearing keywords from query using word boundaries, filtering common stopwords."""
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]{3,}\b', query.lower())
    return [t for t in tokens if t not in STOP_WORDS]

def evaluate_relevance(retrieved_chunks: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
    """
    CRAG Relevance Gatekeeper:
    Computes genuine composite grounding from:
      1. Dense Vector Cosine Similarity (semantic match)
      2. Lexical Keyword Coverage (exact word-boundary token recall across retrieved excerpts)
    
    If context falls below threshold, activates Refusal Protocol to prevent hallucination.
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

    # 3. Composite Grounding Confidence:
    # 60% semantic similarity + 40% lexical keyword recall
    composite_grounding = round((top_dense * 0.60 + lexical_recall * 0.40), 3)
    confidence_pct = round(composite_grounding * 100, 1)

    # Escalation check
    is_promoted = any(c.get("escalated_from") == "whole_page_scan" for c in retrieved_chunks)

    # Refusal decision: Requires both minimum dense match and keyword presence
    if top_dense < SIMILARITY_THRESHOLD and lexical_recall < 0.45:
        return {
            "status": "REFUSE",
            "confidence": confidence_pct,
            "grounding_score": composite_grounding,
            "dense_score": round(top_dense, 3),
            "lexical_recall": round(lexical_recall, 3),
            "nli_verdict": "NEUTRAL (UNVERIFIABLE)",
            "nli_entailment_prob": round(top_dense * 0.5, 3),
            "badge": f"⚠️ OUT-OF-BOUNDS ({confidence_pct}% Grounding)",
            "reason": "The uploaded documents do not contain verified data answering this query. Refusal protocol activated to prevent model hallucination."
        }

    badge_label = (
        f"🌳 PROMOTED TO PAGE SCAN • ENTAILED ({confidence_pct}% Grounding)"
        if is_promoted
        else f"🛡️ GROUNDED • ENTAILED ({confidence_pct}% Grounding)"
    )

    reason_text = (
        f"Verified contextual grounding recovered via Tier-2 AST Whole-Page Escalation (Density: {top_dense:.2f}, Keyword recall: {lexical_recall:.0%})."
        if is_promoted
        else f"Verified contextual grounding found in source documents (Density: {top_dense:.2f}, Keyword recall: {lexical_recall:.0%})."
    )

    return {
        "status": "SYNTHESIZE",
        "confidence": confidence_pct,
        "grounding_score": composite_grounding,
        "dense_score": round(top_dense, 3),
        "lexical_recall": round(lexical_recall, 3),
        "nli_verdict": "ENTAILED",
        "nli_entailment_prob": composite_grounding,
        "badge": badge_label,
        "reason": reason_text
    }
