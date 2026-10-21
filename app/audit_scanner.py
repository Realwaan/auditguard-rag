import re
import difflib
from typing import List, Dict, Any, Optional
from app.retriever import HybridRetriever
from app.evaluator import evaluate_relevance

DEFAULT_COMPLIANCE_CHECKS = [
    {
        "id": "SLA-01",
        "category": "Service Level Agreement",
        "title": "Minimum Service Uptime Guarantee",
        "query": "What is the guaranteed service availability or uptime percentage?",
        "criticality": "HIGH",
        "required_terms": ["uptime", "availability", "99", "sla", "guarantee", "guarantees"],
        "disqualifying_phrases": ["no uptime guarantee", "no availability guarantee", "best effort only", "as-is without"]
    },
    {
        "id": "PEN-02",
        "category": "Commercial Risk",
        "title": "Breach Penalty & Invoice Credit Clause",
        "query": "What penalty credit or invoice discount applies if availability drops?",
        "criticality": "HIGH",
        "required_terms": ["penalty", "credit", "invoice", "discount", "billing"],
        "disqualifying_phrases": ["no penalty", "no billing credit", "waives all credits", "non-refundable"]
    },
    {
        "id": "SEC-03",
        "category": "Data Security & Access",
        "title": "Zero-Trust & Hardware-Enforced MFA",
        "query": "What are the rules for credential storage, MFA, and access control?",
        "criticality": "CRITICAL",
        "required_terms": ["mfa", "sso", "credentials", "authentication", "access"],
        "disqualifying_phrases": ["without mfa", "mfa is optional", "may store credentials on local", "ban mfa", "prohibit mfa", "no mfa"]
    },
    {
        "id": "INC-04",
        "category": "Operational Resilience",
        "title": "Incident Escalation Notification Timeline",
        "query": "What are the mandatory logging and notification steps for security alerts?",
        "criticality": "HIGH",
        "required_terms": ["incident", "escalation", "minutes", "hour", "logged", "notice", "alert"],
        "disqualifying_phrases": ["no notification required", "at vendor discretion"]
    },
    {
        "id": "FIN-05",
        "category": "Financial Controls",
        "title": "Dual Executive Sign-off for Wire Transfers",
        "query": "What is the dollar threshold requiring dual authorization for wire disbursements?",
        "criticality": "MEDIUM",
        "required_terms": ["wire", "dual", "authorization", "disbursements", "sign-off", "cfo"],
        "disqualifying_phrases": ["single signature", "no dual authorization", "solely by virtual assistant"]
    },
    {
        "id": "BCP-06",
        "category": "Business Continuity",
        "title": "Disaster Recovery RTO & Off-site Failover",
        "query": "What is the disaster recovery failover timeline and recovery time objective RTO?",
        "criticality": "HIGH",
        "required_terms": ["disaster", "recovery", "rto", "failover", "standby"],
        "disqualifying_phrases": ["no disaster recovery", "no failover"]
    },
    {
        "id": "ESG-07",
        "category": "Regulatory & ESG",
        "title": "Carbon Emission Offsets & Environmental Quota",
        "query": "What are the greenhouse gas carbon emission offset metrics and environmental reporting targets?",
        "criticality": "MEDIUM",
        "required_terms": ["carbon", "emission", "emissions", "ghg", "environmental", "offset", "offsets"],
        "disqualifying_phrases": ["exempt from environmental", "no carbon reporting"]
    }
]

COMPLIANCE_REMEDIATION_PLAYBOOK: Dict[str, Dict[str, str]] = {
    "SLA-01": {
        "standard_clause": "2.1 Minimum Service Uptime: The Service Provider guarantees a monthly service availability of not less than 99.9% during client business hours, excluding scheduled maintenance.",
        "rationale": "Defines an unambiguous mathematical threshold for availability breach enforcement.",
        "action_required": "Insert formal uptime availability baseline into the Service Level Agreement section."
    },
    "PEN-02": {
        "standard_clause": "2.2 Penalty Credit: In the event monthly availability drops below 99.5%, client shall be entitled to an immediate 15% billing credit applied to the subsequent invoice cycle.",
        "rationale": "Enforces commercial accountability and eliminates vendor breach impunity.",
        "action_required": "Add contractual billing discount trigger upon SLA failure."
    },
    "SEC-03": {
        "standard_clause": "3.1 Zero-Trust & Hardware MFA: Remote contractors must access enterprise systems exclusively via hardware-enforced FIDO2/MFA and enterprise SSO. Storing credentials on personal local drives is strictly prohibited.",
        "rationale": "Satisfies ISO 27001 / SOC 2 access control and credential leakage requirements.",
        "action_required": "Mandate hardware tokens and ban local credential persistence."
    },
    "INC-04": {
        "standard_clause": "3.2 Incident Escalation: Any suspected unauthorized data access must be formally logged within 15 minutes of occurrence and written notice served to the Client Security Officer within 1 hour.",
        "rationale": "Establishes tight breach containment and regulatory disclosure windows.",
        "action_required": "Bind vendor to 15-minute logging and 1-hour executive notification."
    },
    "FIN-05": {
        "standard_clause": "4.1 Dual Wire Authorization: All financial disbursements or wire transactions exceeding $50,000 USD require dual written authorization from the Chief Financial Officer (CFO) and Managing Director.",
        "rationale": "Prevents business email compromise (BEC) fraud and unauthorized treasury transfers.",
        "action_required": "Establish dual executive signature barrier on high-value wires."
    },
    "BCP-06": {
        "standard_clause": "5.1 Business Continuity & RTO: In the event of primary infrastructure failure, failover to secondary hot-standby systems must achieve a Recovery Time Objective (RTO) of under 2 hours.",
        "rationale": "Guarantees business continuity and caps catastrophic disaster downtime.",
        "action_required": "Incorporate explicit RTO ceiling in Disaster Recovery addendum."
    },
    "ESG-07": {
        "standard_clause": "6.1 Environmental Reporting: Vendor shall submit quarterly greenhouse gas (GHG) emission assessments and maintain certified carbon offsets for all cloud compute resources.",
        "rationale": "Complies with Scope 3 supply-chain ESG emissions disclosure directives.",
        "action_required": "Attach quarterly sustainability disclosure and offset commitment."
    }
}

def generate_redline_diff(original_text: str, target_clause: str, control_id: str) -> str:
    """Generates a clean unified git-style diff showing the proposed remediation insertion."""
    from_lines = [f"{original_text}\n"] if original_text and original_text.strip() else []
    to_lines = [
        f"@@ -0,0 +1,2 @@ # [REMEDIATION INSERTION - {control_id}]\n",
        f"+ {target_clause}\n"
    ]
    diff = difflib.unified_diff(
        from_lines,
        to_lines,
        fromfile="Active Document (Current)",
        tofile=f"Compliance Standard ({control_id})"
    )
    return "".join(diff)

def verify_control_substance(excerpt: str, check: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifies that a retrieved excerpt actually addresses the compliance control substance
    rather than merely sharing superficial embedding similarity, and checks for disqualifying negations.
    """
    if not excerpt or not excerpt.strip():
        return {"substantive_pass": False, "matched_terms": [], "reason": "Empty clause excerpt."}

    excerpt_lower = excerpt.lower()
    for bad_phrase in check.get("disqualifying_phrases", []):
        if bad_phrase in excerpt_lower:
            return {
                "substantive_pass": False,
                "matched_terms": [],
                "reason": f"Clause explicitly negates control requirement ('{bad_phrase}')."
            }

    # Control specific negation heuristics
    if check["id"] == "SLA-01" and re.search(r'\b(0%|zero\s*percent|no\s+uptime|does\s+not\s+guarantee|no\s+guarantee|best\s+effort\s+only)\b', excerpt_lower):
        return {
            "substantive_pass": False,
            "matched_terms": [],
            "reason": "Document explicitly disclaims or negates uptime availability."
        }
    if check["id"] == "SEC-03" and re.search(r'\b(ban\s+mfa|prohibit\s+mfa|no\s+mfa|without\s+mfa|prohibits\s+mfa)\b', excerpt_lower):
        return {
            "substantive_pass": False,
            "matched_terms": [],
            "reason": "Document explicitly bans or excludes mandatory MFA controls."
        }
    if check["id"] == "PEN-02" and re.search(r'\b(no\s+penalty|no\s+credit|no\s+liability|penalties\s+waived)\b', excerpt_lower):
        return {
            "substantive_pass": False,
            "matched_terms": [],
            "reason": "Document waives or disclaims contractual breach credits."
        }

    req_terms = check.get("required_terms", [])
    if req_terms:
        matched = [t for t in req_terms if re.search(r'\b' + re.escape(t) + r'\b', excerpt_lower)]
        if not matched:
            return {
                "substantive_pass": False,
                "matched_terms": [],
                "reason": f"Clause lacks required control terms ({', '.join(req_terms[:3])})."
            }
        return {"substantive_pass": True, "matched_terms": matched, "reason": "Control substance verified."}

    return {"substantive_pass": True, "matched_terms": [], "reason": "Substance verified."}

def run_compliance_audit(retriever: HybridRetriever) -> Dict[str, Any]:
    """
    Executes an automated multi-point pre-flight compliance audit across the active document.
    Runs in under 40 milliseconds using in-memory hybrid search.
    Enriched with automated redline remediation diffs and playbooks.
    """
    results = []
    passed_count = 0
    gaps_count = 0

    for check in DEFAULT_COMPLIANCE_CHECKS:
        retrieved = retriever.retrieve(check["query"], top_k=2)
        eval_res = evaluate_relevance(retrieved, query=check["query"])
        remediation = COMPLIANCE_REMEDIATION_PLAYBOOK.get(check["id"], {})

        first_text = retrieved[0].get("parent_text", retrieved[0].get("child_text", "")) if retrieved else ""
        substance = verify_control_substance(first_text, check)
        is_negated = not substance["substantive_pass"]
        negation_reason = substance["reason"]

        if eval_res["status"] == "SYNTHESIZE" and not is_negated:
            best_chunk = retrieved[0]
            excerpt = best_chunk.get("parent_text", best_chunk.get("child_text", ""))
            clean_excerpt = excerpt.split("\n")[0].strip()
            if len(clean_excerpt) > 130:
                clean_excerpt = clean_excerpt[:127] + "..."

            passed_count += 1
            results.append({
                "id": check["id"],
                "control": check["title"],
                "category": check["category"],
                "title": check["title"],
                "criticality": check.get("criticality", "MEDIUM"),
                "status": "PASS",
                "badge": "✅ PASS",
                "confidence": eval_res["confidence"],
                "matched_terms": substance["matched_terms"],
                "clause_excerpt": clean_excerpt,
                "excerpt": clean_excerpt,
                "evidence": clean_excerpt,
                "full_clause": excerpt,
                "page": best_chunk.get("page", 1),
                "image_url": best_chunk.get("image_url", ""),
                "citation_id": best_chunk.get("citation_id", 1),
                "finding": f"Verified in Section/Page {best_chunk.get('page', 1)} with {eval_res['confidence']}% confidence (matched: {', '.join(substance['matched_terms'][:3])}).",
                "remediation_clause": "Compliant with enterprise standard. No remediation required.",
                "redline_diff": None,
                "diff": None,
                "action_plan": "Verified active in document.",
                "rationale": remediation.get("rationale", "Meets compliance requirements.")
            })
        else:
            gaps_count += 1
            std_clause = remediation.get("standard_clause", "Standard enterprise clause required.")
            action = remediation.get("action_required", "Remediation insertion required.")
            rationale = remediation.get("rationale", "Mandatory compliance clause.")
            redline = generate_redline_diff("", std_clause, check["id"])
            finding_text = f"NON-COMPLIANCE DETECTED: {negation_reason}" if (is_negated and first_text) else f"HIGH AUDIT RISK: {action}"

            gap_clause_desc = negation_reason if (is_negated and first_text) else "Clause missing from active document or fails compliance verification."
            results.append({
                "id": check["id"],
                "control": check["title"],
                "category": check["category"],
                "title": check["title"],
                "criticality": check.get("criticality", "MEDIUM"),
                "status": "GAP",
                "badge": "❌ GAP DETECTED",
                "confidence": eval_res["confidence"],
                "matched_terms": substance["matched_terms"],
                "excerpt": gap_clause_desc,
                "clause_excerpt": gap_clause_desc,
                "evidence": gap_clause_desc,
                "full_clause": "",
                "page": None,
                "image_url": "",
                "citation_id": None,
                "finding": finding_text,
                "remediation_clause": std_clause,
                "redline_diff": redline,
                "diff": redline,
                "action_plan": action,
                "rationale": rationale
            })

    total_checks = len(DEFAULT_COMPLIANCE_CHECKS)
    score_pct = round((passed_count / total_checks) * 100, 1)

    if score_pct >= 80:
        verdict = "AUDIT PASSED (CONDITIONAL)"
        verdict_color = "emerald"
    elif score_pct >= 50:
        verdict = "WARNING: DEFICIENCIES DETECTED"
        verdict_color = "amber"
    else:
        verdict = "FAILED AUDIT (CRITICAL GAPS)"
        verdict_color = "red"

    return {
        "overall_score": score_pct,
        "total_checks": total_checks,
        "total_controls": total_checks,
        "controls_checked": total_checks,
        "passed_count": passed_count,
        "gaps_count": gaps_count,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "checklist": results,
        "controls": results,
        "remediation_count": gaps_count,
        "remediation_playbook_available": True
    }

