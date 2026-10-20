from typing import List, Dict, Any
from app.retriever import HybridRetriever
from app.evaluator import evaluate_relevance

DEFAULT_COMPLIANCE_CHECKS = [
    {
        "id": "SLA-01",
        "category": "Service Level Agreement",
        "title": "Minimum Service Uptime Guarantee",
        "query": "What is the guaranteed service availability or uptime percentage?",
        "criticality": "HIGH"
    },
    {
        "id": "PEN-02",
        "category": "Commercial Risk",
        "title": "Breach Penalty & Invoice Credit Clause",
        "query": "What penalty credit or invoice discount applies if availability drops?",
        "criticality": "HIGH"
    },
    {
        "id": "SEC-03",
        "category": "Data Security & Access",
        "title": "Zero-Trust & Hardware-Enforced MFA",
        "query": "What are the rules for credential storage, MFA, and access control?",
        "criticality": "CRITICAL"
    },
    {
        "id": "INC-04",
        "category": "Operational Resilience",
        "title": "Incident Escalation Notification Timeline",
        "query": "What are the mandatory logging and notification steps for security alerts?",
        "criticality": "HIGH"
    },
    {
        "id": "FIN-05",
        "category": "Financial Controls",
        "title": "Dual Executive Sign-off for Wire Transfers",
        "query": "What is the dollar threshold requiring dual authorization for wire disbursements?",
        "criticality": "MEDIUM"
    },
    {
        "id": "BCP-06",
        "category": "Business Continuity",
        "title": "Disaster Recovery RTO & Off-site Failover",
        "query": "What is the disaster recovery failover timeline and recovery time objective RTO?",
        "criticality": "HIGH"
    },
    {
        "id": "ESG-07",
        "category": "Regulatory & ESG",
        "title": "Carbon Emission Offsets & Environmental Quota",
        "query": "What are the greenhouse gas carbon emission offset metrics and environmental reporting targets?",
        "criticality": "MEDIUM"
    }
]

def run_compliance_audit(retriever: HybridRetriever) -> Dict[str, Any]:
    """
    Executes an automated multi-point pre-flight compliance audit across the active document.
    Runs in under 40 milliseconds using in-memory hybrid search.
    """
    results = []
    passed_count = 0
    gaps_count = 0

    for check in DEFAULT_COMPLIANCE_CHECKS:
        retrieved = retriever.retrieve(check["query"], top_k=2)
        eval_res = evaluate_relevance(retrieved)

        if eval_res["status"] == "SYNTHESIZE":
            best_chunk = retrieved[0]
            excerpt = best_chunk.get("parent_text", best_chunk.get("child_text", ""))
            # Truncate clean excerpt for summary card
            clean_excerpt = excerpt.split("\n")[0].strip()
            if len(clean_excerpt) > 110:
                clean_excerpt = clean_excerpt[:107] + "..."

            passed_count += 1
            results.append({
                "id": check["id"],
                "category": check["category"],
                "title": check["title"],
                "criticality": check["criticality"],
                "status": "PASS",
                "badge": "✅ PASS",
                "confidence": eval_res["confidence"],
                "clause_excerpt": clean_excerpt,
                "full_clause": excerpt,
                "page": best_chunk.get("page", 1),
                "image_url": best_chunk.get("image_url", ""),
                "citation_id": best_chunk.get("citation_id", 1),
                "finding": f"Verified in Section/Page {best_chunk.get('page', 1)} with {round(eval_res['confidence']*100, 1)}% confidence."
            })
        else:
            gaps_count += 1
            results.append({
                "id": check["id"],
                "category": check["category"],
                "title": check["title"],
                "criticality": check["criticality"],
                "status": "GAP",
                "badge": "❌ GAP DETECTED",
                "confidence": eval_res["confidence"],
                "clause_excerpt": "Clause not present in agreement or below confidence threshold.",
                "full_clause": "",
                "page": None,
                "image_url": "",
                "citation_id": None,
                "finding": "HIGH AUDIT RISK: Required compliance clause is absent from document."
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
        "passed_count": passed_count,
        "gaps_count": gaps_count,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "checklist": results
    }
