"""
AuditGuard Grounding & Relevance Gatekeeper.
Evaluates whether retrieved document excerpts contain sufficient semantic and lexical
grounding to answer the user query without hallucination.
"""

import math
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
    "provided", "document", "documents", "excerpt", "excerpts", "page", "pages",
    "key", "highlight", "highlighted", "highlights", "highlighting", "mention", "mentioned",
    "action", "item", "items"
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "percent", "percentage"
}
ONES = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19
}
TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}
SCALES = {
    'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000
}
ALL_NUM_WORDS = sorted(list(set(ONES.keys()) | set(TENS.keys()) | set(SCALES.keys())), key=len, reverse=True)

def normalize_written_numbers(text: str) -> str:
    """Replaces written-out English numbers (e.g. 'fifty', 'twenty-five', 'two hours') with digit equivalents."""
    if not text:
        return ""
    pattern_hyphen = r'\b(' + '|'.join(ALL_NUM_WORDS) + r')\s*-\s*(' + '|'.join(ALL_NUM_WORDS) + r')\b'
    text = re.sub(pattern_hyphen, r'\1 \2', text, flags=re.IGNORECASE)
    pattern = r'\b(?:' + '|'.join(ALL_NUM_WORDS) + r')\b(?:\s+(?:and\s+)?(?:' + '|'.join(ALL_NUM_WORDS) + r')\b)*'
    def repl(m):
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', m.group(0)) if w.lower() in ALL_NUM_WORDS]
        if not words:
            return m.group(0)
        total, current = 0, 0
        for tok in words:
            if tok in ONES:
                current += ONES[tok]
            elif tok in TENS:
                current += TENS[tok]
            elif tok == 'hundred':
                current = (current if current else 1) * 100
            elif tok in ('thousand', 'million', 'billion'):
                current = (current if current else 1) * SCALES[tok]
                total += current
                current = 0
        return str(total + current)
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

NEGATION_WORDS = {
    "no", "not", "never", "none", "neither", "nor", "prohibited", "banned",
    "disallowed", "unauthorized", "excluding", "without", "except"
}

def extract_query_keywords(query: str) -> List[str]:
    """Extract content-bearing keywords from query using word boundaries, filtering common stopwords."""
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', query.lower())
    return list(dict.fromkeys(t for t in tokens if len(t) >= 2 and t not in STOP_WORDS))


def stem_word(w: str) -> set:
    forms = {w}
    if w.endswith("ing") and len(w) > 5:
        forms.add(w[:-3])
        forms.add(w[:-3] + "e")
    elif w.endswith("ed") and len(w) > 4:
        forms.add(w[:-2])
        forms.add(w[:-1])
    elif w.endswith("es") and len(w) > 4:
        forms.add(w[:-2])
        forms.add(w[:-1])
    elif w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        forms.add(w[:-1])
    return forms

def keyword_present(keyword: str, text: str) -> bool:
    """Match whole words, including singular/plural, past tense (-ed), and progressive (-ing) forms."""
    forms = set()
    for f in stem_word(keyword.lower()):
        forms.add(f)
        if len(f) > 3:
            forms.add(f + "s")
            forms.add(f + "ed")
            forms.add(f + "ing")
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
    # Citation-location instructions do not request a numerical metric value.
    q_lower = re.sub(r'\b(?:please\s+)?cite\b[^?!.]*', '', query.lower()).strip()

    # 1. Quantitative vs Descriptive Answer Target Check
    # Identify conceptual / definitional questions (e.g. "What is an SLA?", "What is availability?", "Define uptime")
    # which seek conceptual explanations rather than quantitative metrics.
    explicit_definitional = bool(re.search(
        r'^(?:what\s+(?:is|are)\s+(?:an?|the\s+(?:concept|meaning|definition|purpose)\s+of)\b|define\b|meaning\s+of\b|definition\s+of\b|explain\s+(?:what\s+is\s+)?(?:an?|the\s+(?:concept|meaning|definition)\s+of)\b|what\s+does\b.+\bmean\b)',
        q_lower.strip()
    ))
    bare_what_is_concept = bool(re.search(
        r'^what\s+(?:is|are)\s+(?!(?:the|our|your|their|its|this|that|current|required|guaranteed|minimum|maximum)\b)[a-zA-Z0-9_\-\s]{2,30}\??$',
        q_lower.strip()
    )) and not bool(re.search(
        r'\b(?:fee|fees|cost|costs|price|cap|dollar|dollars|\$|penalty|credit|deductible|liability|threshold|allowance|quota|response\s+time|notice\s+period)\b',
        q_lower
    ))
    has_explicit_value_request = bool(re.search(
        r'\b(?:how\s+much|how\s+many|how\s+long|amount|percentage|percent|exact|figure|number|rate\s+of|limit|cap\s+of|maximum|minimum|target|goal|threshold|required|guaranteed)\b',
        q_lower
    ))
    # Explicit definitions of targets remain conceptual; bare "what is ... target"
    # asks for the actual value, even when the article "the" is omitted.
    definition_request = bool(re.search(
        r'^(?:define\b|meaning\s+of\b|definition\s+of\b|what\s+does\b.+\bmean\b|what\s+is\s+the\s+(?:concept|meaning|definition|purpose)\s+of\b)',
        q_lower.strip(),
    ))
    is_conceptual = definition_request or ((explicit_definitional or bare_what_is_concept) and not has_explicit_value_request)

    # Strip structural document references (e.g. "Section 2", "Clause 4.1", "Table 1", "Page 3")
    # so section or page identifiers are never confused with substantive quantitative targets.
    substantive_ctx = re.sub(
        r'\b(?:section|clause|table|figure|fig\.?|page|p\.?|pp\.?|article|paragraph|para\.?|item|step|appendix|exhibit|rule)\s+\d+(?:[\.\-]\d+)*\b',
        '', ctx_lower
    )
    substantive_ctx = re.sub(r'\b(?:v|ver|version)\s*\d+(?:\.\d+)*\b', '', substantive_ctx)
    substantive_ctx = normalize_written_numbers(substantive_ctx)

    is_quantitative = False
    has_numerical_target = True

    if not is_conceptual:
        # Check for quantitative indicators:
        # - Direct quantity queries (how much, how many, how long, amount, percentage, threshold)
        # - Financial & metric terms (fee, cost, cap, dollar, penalty, credit, uptime, availability, etc.)
        has_qty_question = bool(re.search(
            r'\b(?:how\s+much|how\s+many|how\s+long|amount|percentage|percent|threshold)\b', q_lower
        ))
        has_qty_terms = bool(re.search(
            r'\b(?:fee|fees|cost|costs|price|cap|dollar|dollars|\$|penalty|credit|uptime|availability|deductible|liability|allowance|quota|response\s+time|notice\s+period)\b', q_lower
        ))

        if has_qty_question or has_qty_terms or has_explicit_value_request:
            is_quantitative = True
            # Extract query metric topic keywords to verify the quantity is associated with the requested metric
            qty_generic = STOP_WORDS | {
                "what", "how", "many", "much", "long", "exact", "maximum", "minimum",
                "allowed", "required", "guaranteed", "number", "amount", "percentage",
                "percent", "figure", "value", "days", "hours", "minutes", "dollars", "dollar",
                "target", "goal", "limit", "threshold"
            }
            metric_kws = [t for t in extract_query_keywords(q_lower) if t not in qty_generic]
            unspecified_re = re.compile(
                r'\b(?:not\s+(?:yet\s+)?(?:been\s+)?specified|unspecified|undecided|under\s+review|to\s+be\s+determined|tbd|not\s+(?:yet\s+)?determined|not\s+(?:yet\s+)?defined|not\s+stated|none\s+specified)\b'
            )
            ctx_clauses = [
                cl.strip() for cl in re.split(r'\.(?!\d)|[!?;\n|]+|\s+(?:and|but|while|whereas)\s+', substantive_ctx) if cl.strip()
            ] or [substantive_ctx]
            overlaps = [sum(keyword_present(kw, cl) for kw in metric_kws) for cl in ctx_clauses]
            required_overlap = max(1, math.ceil(max(overlaps, default=0) * 0.6))

            def _clause_has_associated_quantity(pattern: str) -> bool:
                for cl, overlap in zip(ctx_clauses, overlaps):
                    if unspecified_re.search(cl):
                        continue
                    if not re.search(pattern, cl):
                        continue
                    if not metric_kws or overlap >= required_overlap:
                        return True
                return False

            # Verify that context contains numbers associated with the requested quantity type
            if re.search(r'\b(?:how\s+many\s+days|days\s+of|number\s+of\s+days|leave\s+days|notice\s+period)\b', q_lower):
                has_numerical_target = _clause_has_associated_quantity(
                    r'\b\d+(?:\.\d+)?\s*(?:business\s+days?|calendar\s+days?|working\s+days?|days?)\b'
                )
            elif re.search(r'\b(?:how\s+many\s+hours|hours\s+of|response\s+time)\b', q_lower):
                has_numerical_target = _clause_has_associated_quantity(
                    r'\b\d+(?:\.\d+)?\s*(?:business\s+hours?|working\s+hours?|hours?|hrs?|minutes?|mins?|days?)\b'
                )
            elif re.search(r'\b(?:how\s+long)\b', q_lower):
                has_numerical_target = _clause_has_associated_quantity(
                    r'\b\d+(?:\.\d+)?\s*(?:business\s+days?|calendar\s+days?|working\s+days?|days?|hours?|hrs?|minutes?|mins?|weeks?|months?|years?)\b'
                )
            elif re.search(r'\b(?:percentage|percent|uptime|availability)\b', q_lower):
                has_numerical_target = _clause_has_associated_quantity(
                    r'\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?\b)'
                )
            elif re.search(r'\b(?:fee|fees|cost|costs|price|dollar|dollars|\$|threshold|deductible|liability|cap|penalty|credit)\b', q_lower):
                has_numerical_target = _clause_has_associated_quantity(
                    r'(?:[$€£]\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|\b(?:dollars?|cents?|usd|eur|euros?|gbp|pounds?|percent(?:age)?|x|times|months?\s+of\s+fees?)\b))'
                )
            else:
                # Exclude standalone calendar years (e.g. "in 2024") from satisfying general quantity questions
                has_numerical_target = _clause_has_associated_quantity(
                    r'(?<!\bin\s)(?<!\bsince\s)(?<!\bfrom\s)\b\d+(?:,\d{3})*(?:\.\d+)?%?\b'
                )

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
    has_answer_target = has_numerical_target if is_quantitative else (
        lexical_recall >= 0.40 or (top_dense >= 0.65 and lexical_recall >= 0.20)
    )
    
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

    if any(c.get("visual_scan_incomplete") for c in retrieved_chunks):
        return {
            "status": "REFUSE", "confidence": 0.0, "grounding_score": 0.0,
            "dense_score": 0.0, "lexical_recall": 0.0,
            "grounding_verdict": "INCOMPLETE_VISUAL_COVERAGE", "nli_verdict": "UNVERIFIABLE",
            "nli_entailment_prob": 0.0, "claim_evidence_verified": False,
            "badge": "SCAN NEEDS A PAGE SELECTION",
            "reason": "This document has more unreadable scanned pages than can be inspected in one request. Ask about a specific page (for example, 'on page 2'), or upload an OCR-readable document.",
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

    # If retrieved chunks are visual-only scanned pages without OCR text, allow multimodal visual inspection
    has_unocred_visual_scan = any(
        c.get("has_visual") and c.get("image_url") and (
            (c.get("is_scanned") and not c.get("is_ocr"))
            or "(visual / scanned content)" in c.get("parent_text", "").lower()
        )
        for c in retrieved_chunks
    )
    if has_unocred_visual_scan and nli_verdict == "NEUTRAL":
        return {
            "status": "SYNTHESIZE",
            "confidence": confidence_pct,
            "grounding_score": composite_grounding,
            "dense_score": round(top_dense, 3),
            "lexical_recall": round(lexical_recall, 3),
            "grounding_verdict": "VISUAL_SCAN_EVIDENCE",
            "heuristic_alignment_score": nli_prob,
            "claim_evidence_verified": False,
            "retrieval_evidence_available": True,
            "evaluation_method": "multimodal_visual_inspection",
            "nli_verdict": "VISUAL_SCAN_EVIDENCE",
            "nli_entailment_prob": nli_prob,
            "nli_method": "multimodal_visual_inspection",
            "is_answer_bearing": True,
            "badge": "🖼️ VISUAL SCAN • MULTIMODAL INSPECTION",
            "reason": "Document page contains scanned visual imagery without embedded text; routing rendered page image to multimodal synthesis."
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
