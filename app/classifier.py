import re
from typing import Dict, Any

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
    Analyzes document header / first page text to classify the document type
    and select the appropriate adaptive persona.
    """
    text_lower = sample_text.lower()
    scores = {}

    for doc_type, info in DOC_PROFILES.items():
        score = sum(1 for kw in info["keywords"] if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
        scores[doc_type] = score

    best_type = max(scores, key=scores.get) if scores else None
    
    if not best_type or scores[best_type] == 0:
        return {
            "type": "GENERAL_SPECIFICATION",
            "label": "General Enterprise Document",
            "icon": "📄",
            "confidence": 75.0,
            "prompt_addon": "You are a professional Enterprise Research Analyst. Provide clear, well-structured, cited answers."
        }
    
    # Calculate confidence based on keyword density
    matched_count = scores[best_type]
    confidence = min(98.0, 70.0 + (matched_count * 5.0))

    return {
        "type": best_type,
        "label": DOC_PROFILES[best_type]["label"],
        "icon": DOC_PROFILES[best_type]["icon"],
        "confidence": confidence,
        "prompt_addon": DOC_PROFILES[best_type]["prompt_addon"]
    }
