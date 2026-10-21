import re
from typing import Dict, Any, List

DOC_PROFILES = {
    "LEGAL_CONTRACT": {
        "label": "Legal Agreement / SLA",
        "icon": "⚖️",
        "keywords": [
            "agreement", "liabilities", "indemnification", "severability",
            "termination", "governing law", "sla", "service level", "penalty", "breach"
        ],
        "prompt_addon": (
            "You are acting as an Enterprise Legal & Compliance Auditor. "
            "Pay strict attention to exact section numbers, liability caps, deadlines, and binding obligations."
        )
    },
    "SOP_PROCEDURAL": {
        "label": "Standard Operating Procedure (SOP)",
        "icon": "📋",
        "keywords": [
            "standard operating procedure", "sop", "workflow", "step 1", "step 2",
            "responsibilities", "prerequisites", "escalation", "incident response"
        ],
        "prompt_addon": (
            "You are acting as an Operations Director. Format procedures into clear, numbered action items, "
            "explicitly stating which role is responsible for each step."
        )
    },
    "FINANCIAL_AUDIT": {
        "label": "Financial / Audit Report",
        "icon": "📊",
        "keywords": [
            "balance sheet", "fiscal year", "net income", "ebitda", "expenditure",
            "revenue", "assets", "liabilities", "audit", "quarterly"
        ],
        "prompt_addon": (
            "You are acting as a Forensic Financial Verifier. Ensure all numerical figures, percentages, "
            "and currency amounts are cited with 100% mathematical accuracy without rounding unless stated."
        )
    },
    "ACADEMIC_POLICY": {
        "label": "Institutional / Academic Policy",
        "icon": "🎓",
        "keywords": [
            "curriculum", "honor code", "grading", "dean", "prerequisite",
            "academic integrity", "syllabus", "faculty", "student affairs"
        ],
        "prompt_addon": (
            "You are acting as an Academic Policy Ombudsman. Clearly outline student rights, course prerequisites, "
            "and procedural appeal timelines."
        )
    },
    "TECHNICAL_MANUAL": {
        "label": "Technical Documentation",
        "icon": "⚙️",
        "keywords": [
            "installation", "configuration", "api", "endpoint", "architecture",
            "deployment", "parameters", "cli", "sdk", "runtime", "port"
        ],
        "prompt_addon": (
            "You are acting as a Senior Systems Architect. Highlight exact CLI commands, configuration parameters, "
            "and critical system prerequisites."
        )
    }
}


def classify_document(sample_text: str) -> Dict[str, Any]:
    """
    Analyzes document header / first pages text to classify the document type
    and select the appropriate adaptive persona.
    Returns genuine keyword coverage confidence (0.0% when falling back to general specification).
    """
    if not sample_text or not sample_text.strip():
        return {
            "type": "GENERAL_SPECIFICATION",
            "label": "General Enterprise Document",
            "icon": "📄",
            "confidence": 0.0,
            "matched_keywords": [],
            "prompt_addon": "You are a professional Enterprise Research Analyst. Provide clear, well-structured, cited answers."
        }

    text_lower = sample_text.lower()
    scores: Dict[str, int] = {}
    matched_by_type: Dict[str, List[str]] = {}

    for doc_type, info in DOC_PROFILES.items():
        matched = [
            kw for kw in info["keywords"]
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        ]
        scores[doc_type] = len(matched)
        matched_by_type[doc_type] = matched

    best_type = max(scores, key=scores.get) if scores else None

    if not best_type or scores[best_type] == 0:
        return {
            "type": "GENERAL_SPECIFICATION",
            "label": "General Enterprise Document",
            "icon": "📄",
            "confidence": 0.0,
            "matched_keywords": [],
            "prompt_addon": "You are a professional Enterprise Research Analyst. Provide clear, well-structured, cited answers."
        }

    matched_count = scores[best_type]
    total_profile_kws = max(1, len(DOC_PROFILES[best_type]["keywords"]))
    total_all_matches = sum(scores.values())

    # Genuine confidence calculation: combines profile keyword coverage and specificity (dominance over other profiles)
    coverage_ratio = min(1.0, matched_count / min(5, total_profile_kws))
    specificity_ratio = matched_count / max(1, total_all_matches)
    confidence = round((0.65 * coverage_ratio + 0.35 * specificity_ratio) * 100.0, 1)

    return {
        "type": best_type,
        "label": DOC_PROFILES[best_type]["label"],
        "icon": DOC_PROFILES[best_type]["icon"],
        "confidence": confidence,
        "matched_keywords": matched_by_type[best_type],
        "prompt_addon": DOC_PROFILES[best_type]["prompt_addon"]
    }
