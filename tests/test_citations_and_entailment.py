import pytest
from app.evaluator import evaluate_relevance, verify_logical_entailment
from app.generator import verify_citations_grounding

def test_nli_detects_topic_match_without_answer_target():
    """
    Issue 1 Test:
    Verifies that a passage mentioning the right topic (e.g. penalty, SLA)
    but lacking the actual required answer (e.g. percentage/metric)
    is identified as NEUTRAL rather than falsely claiming ENTAILED.
    """
    chunks = [{
        "parent_text": "The contract discusses general service level agreements and review of penalties by management.",
        "child_text": "general service level agreements and review of penalties",
        "dense_score": 0.65
    }]
    query = "What is the exact penalty credit percentage for an uptime breach?"
    
    res = evaluate_relevance(chunks, query=query)
    # Context does not contain the percentage/metric requested
    assert "NEUTRAL" in res["nli_verdict"]
    assert res["status"] == "REFUSE"

def test_nli_entails_when_answer_target_is_present():
    """
    Issue 1 Test:
    Verifies that when the passage actually contains the numerical answer target,
    it receives an ENTAILED verdict.
    """
    chunks = [{
        "parent_text": "In the event of an uptime breach below 99.9%, a 15% penalty credit applies to the invoice.",
        "child_text": "15% penalty credit applies to the invoice",
        "dense_score": 0.75
    }]
    query = "What is the penalty credit percentage for an uptime breach?"
    
    res = evaluate_relevance(chunks, query=query)
    assert res["nli_verdict"] == "ENTAILED"
    assert res["status"] == "SYNTHESIZE"
    assert res["nli_entailment_prob"] >= 0.70

def test_citation_verification_detects_unsupported_claims():
    """
    Issue 6 Test:
    Verifies that citation verification flags claims that cite a chunk
    whose content contradicts or does not support the claim.
    """
    chunks = [
        {
            "citation_id": 1,
            "parent_text": "We provide a 99.9% uptime SLA for cloud hosting.",
            "child_text": "99.9% uptime SLA"
        },
        {
            "citation_id": 2,
            "parent_text": "Standard support response time is 4 business hours.",
            "child_text": "4 business hours"
        }
    ]

    # Valid answer citing correct facts
    valid_answer = "The cloud hosting platform provides a 99.9% uptime SLA [1], and support response time is 4 business hours [2]."
    audit_valid = verify_citations_grounding(valid_answer, chunks)
    assert audit_valid["all_supported"] is True
    assert audit_valid["unsupported_count"] == 0

    # Unsupported answer claiming numbers/terms not in chunk 1
    hallucinated_answer = "The vendor provides a 95.0% uptime guarantee [1] and guarantees a 30-minute response [2]."
    audit_invalid = verify_citations_grounding(hallucinated_answer, chunks)
    assert audit_invalid["all_supported"] is False
    assert audit_invalid["unsupported_count"] >= 1
