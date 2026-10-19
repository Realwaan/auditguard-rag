from typing import List, Dict, Any
from app.config import SIMILARITY_THRESHOLD

def evaluate_relevance(retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Corrective RAG (CRAG) Relevance Gatekeeper.
    Evaluates whether the retrieved context is mathematically confident enough to answer.
    If the top match is below SIMILARITY_THRESHOLD, triggers a Defense Refusal to prevent hallucination.
    """
    if not retrieved_chunks:
        return {
            "status": "REFUSE",
            "confidence": 0.0,
            "badge": "⚠️ OUT-OF-BOUNDS (0% Match)",
            "reason": "No relevant text chunks found in uploaded documents."
        }

    # Best dense similarity score in retrieved set
    top_dense = max(c.get("dense_score", 0.0) for c in retrieved_chunks)
    confidence_pct = round(top_dense * 100, 1)

    if top_dense < SIMILARITY_THRESHOLD:
        return {
            "status": "REFUSE",
            "confidence": confidence_pct,
            "badge": f"⚠️ OUT-OF-BOUNDS ({confidence_pct}% Grounding)",
            "reason": "The uploaded documents do not contain verified data answering this query. Refusal protocol activated to prevent model hallucination."
        }

    # Verified grounding
    badge_label = f"🛡️ GROUNDED ({confidence_pct}% Confidence)"
    return {
        "status": "SYNTHESIZE",
        "confidence": confidence_pct,
        "badge": badge_label,
        "reason": "Verified contextual grounding found in source documents."
    }
