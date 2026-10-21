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

def test_sanitize_citations_removes_invalid_and_out_of_range():
    from app.generator import sanitize_citations
    text = "Uptime is 99.9% [1], penalty is 15% [2], invalid zero is [0], out of range is [99]."
    clean = sanitize_citations(text, max_chunk=2)
    assert "[1]" in clean
    assert "[2]" in clean
    assert "[0]" not in clean
    assert "[99]" not in clean


def test_evaluator_honest_claim_evidence_signals():
    """Verifies that evaluator returns explicit grounding_verdict, heuristic_alignment_score, and claim_evidence_verified."""
    chunks = [{
        "parent_text": "Clause 4.1: Wire transfers exceeding $50,000 require written sign-off from CFO and Managing Director.",
        "child_text": "Wire transfers exceeding $50,000 require CFO sign-off",
        "dense_score": 0.88
    }]
    query = "What is the authorization threshold for wire transfers?"

    res = evaluate_relevance(chunks, query=query)
    assert res["status"] == "SYNTHESIZE"
    assert res["grounding_verdict"] == "SUPPORTED"
    # A retrieval gate cannot certify claims in an answer that does not exist yet.
    assert res["claim_evidence_verified"] is False
    assert res["retrieval_evidence_available"] is True
    assert res["heuristic_alignment_score"] > 0.70
    assert res["evaluation_method"] == "heuristic_proposition_gate"
    # Backward compatibility
    assert res["nli_verdict"] == "ENTAILED"

    # Out of bounds / unverifiable
    ungrounded_chunks = [{
        "parent_text": "Company holidays include Thanksgiving and Christmas.",
        "child_text": "Company holidays include Thanksgiving",
        "dense_score": 0.15
    }]
    ref_res = evaluate_relevance(ungrounded_chunks, query=query)
    assert ref_res["status"] == "REFUSE"
    assert ref_res["grounding_verdict"] == "UNVERIFIABLE"
    assert ref_res["claim_evidence_verified"] is False
    assert "NEUTRAL" in ref_res["nli_verdict"]



def test_citation_verification_detects_unit_mismatch_days_vs_hours():
    chunks = [{
        "citation_id": 1,
        "parent_text": "Support response time is 8 days under standard SLA tier.",
        "child_text": "Support response time is 8 days"
    }]
    answer = "Support response time is 8 hours [1]."
    audit = verify_citations_grounding(answer, chunks)
    assert audit["all_supported"] is False
    assert audit["verdict"] == "UNSUPPORTED_CLAIM"
    assert audit["unsupported_count"] >= 1
    assert "unit" in audit["verified_citations"][0]["reason"].lower()


def test_citation_verification_detects_written_number_mismatch_fifty_vs_fifteen():
    chunks = [{
        "citation_id": 1,
        "parent_text": "Penalty credit is fifty percent for complete outages.",
        "child_text": "Penalty credit is fifty percent"
    }]
    answer = "Penalty credit is fifteen percent [1]."
    audit = verify_citations_grounding(answer, chunks)
    assert audit["all_supported"] is False
    assert audit["verdict"] == "UNSUPPORTED_CLAIM"
    assert audit["unsupported_count"] >= 1


def test_citation_verification_accepts_normalized_written_numbers():
    chunks = [{
        "citation_id": 1,
        "parent_text": "Penalty credit is fifty percent for complete outages.",
        "child_text": "Penalty credit is fifty percent"
    }]
    answer = "Penalty credit is 50% [1]."
    audit = verify_citations_grounding(answer, chunks)
    assert audit["all_supported"] is True
    assert audit["verdict"] == "VERIFIED"


def test_citation_verification_flags_uncited_preceding_sentence():
    chunks = [{
        "citation_id": 1,
        "parent_text": "Backups are retained for 30 days in off-site encrypted storage.",
        "child_text": "Backups are retained for 30 days"
    }]
    answer = "Data is encrypted at rest using AES-256. Backups are retained for 30 days [1]."
    audit = verify_citations_grounding(answer, chunks)
    assert audit["all_supported"] is False
    assert audit["verdict"] == "UNSUPPORTED_CLAIM"
    assert any(c["citation_id"] is None for c in audit["verified_citations"])


def test_evaluator_rejects_structural_section_number_for_duration_query():
    chunks = [{
        "parent_text": "Section 2 discusses annual leave eligibility. All employees in good standing are entitled to leave as approved by HR.",
        "child_text": "Section 2 discusses annual leave eligibility",
        "dense_score": 0.70
    }]
    query = "How many days of annual leave are allowed?"
    res = evaluate_relevance(chunks, query=query)
    assert res["status"] == "REFUSE"
    assert "NEUTRAL" in res["nli_verdict"]
    assert "quantitative metric" in res["reason"].lower()


def test_evaluator_accepts_valid_duration_quantity():
    chunks = [{
        "parent_text": "Section 2: Full-time employees are entitled to 25 days of annual leave per calendar year.",
        "child_text": "25 days of annual leave per calendar year",
        "dense_score": 0.85
    }]
    query = "How many days of annual leave are allowed?"
    res = evaluate_relevance(chunks, query=query)
    assert res["status"] == "SYNTHESIZE"
    assert res["nli_verdict"] == "ENTAILED"


def test_evaluator_conceptual_question_not_blocked_by_numerical_check():
    chunks = [{
        "parent_text": "A Service Level Agreement (SLA) is a documented commitment between a service provider and a client specifying service standards.",
        "child_text": "Service Level Agreement (SLA) is a documented commitment",
        "dense_score": 0.82
    }]
    query = "What is an SLA?"
    res = evaluate_relevance(chunks, query=query)
    assert res["status"] == "SYNTHESIZE"
    assert res["nli_verdict"] == "ENTAILED"


def test_evaluator_financial_cap_detection():
    chunks_no_cap = [{
        "parent_text": "Section 5 discusses dispute arbitration and governing law in the State of Delaware.",
        "child_text": "Section 5 discusses dispute arbitration",
        "dense_score": 0.65
    }]
    query = "What is the maximum liability cap?"
    res_no = evaluate_relevance(chunks_no_cap, query=query)
    assert res_no["status"] == "REFUSE"

    chunks_with_cap = [{
        "parent_text": "Section 5: The aggregate liability cap shall not exceed $1,000,000 for any single claim.",
        "child_text": "aggregate liability cap shall not exceed $1,000,000",
        "dense_score": 0.85
    }]
    res_with = evaluate_relevance(chunks_with_cap, query=query)
    assert res_with["status"] == "SYNTHESIZE"
    assert res_with["nli_verdict"] == "ENTAILED"


def test_citation_verification_rejects_unsupported_units_items_vs_hours():
    """Finding 1 regression: An unclassified/different source unit ('items') must not support 'hours'."""
    chunks = [{
        "citation_id": 1,
        "parent_text": "The SLA threshold is 8 items.",
        "child_text": "The SLA threshold is 8 items."
    }]
    answer = "The SLA threshold is 8 hours [1]."
    audit = verify_citations_grounding(answer, chunks)
    assert audit["all_supported"] is False
    assert audit["verdict"] == "UNSUPPORTED_CLAIM"
    assert audit["unsupported_count"] == 1
    assert "unit mismatch" in audit["verified_citations"][0]["reason"].lower()


@pytest.mark.parametrize("source_text", [
    "Response time is 4 hours. Backups are retained for 30 days.",
    "4 hours response time; 30 days backup retention.",
    "Response time is 4 hours, and backups are retained for 30 days.",
])
def test_citation_verification_rejects_quantities_from_unrelated_statements(source_text):
    """Finding 4 regression: A quantity from a different statement/clause cannot support the wrong metric claim."""
    chunks = [{
        "citation_id": 1,
        "parent_text": source_text,
        "child_text": source_text
    }]
    answer = "Response time is 30 days [1]."
    audit = verify_citations_grounding(answer, chunks)
    assert audit["all_supported"] is False
    assert audit["verdict"] == "UNSUPPORTED_CLAIM"
    assert audit["unsupported_count"] == 1
    assert "statement mismatch" in audit["verified_citations"][0]["reason"].lower()


def test_evaluator_rejects_unrelated_financial_numbers():
    """Finding 3 regression: An unrelated year ('2024') must not satisfy a missing liability cap query."""
    chunks = [{
        "parent_text": "The contract begins in 2024. The liability cap has not been specified.",
        "child_text": "The contract begins in 2024. The liability cap has not been specified.",
        "dense_score": 0.80
    }]
    query = "What is the maximum liability cap?"
    res = evaluate_relevance(chunks, query=query)
    assert res["status"] == "REFUSE"
    assert "NEUTRAL" in res["nli_verdict"]


def test_citation_verification_verifies_correct_sentence_without_overwriting():
    """Finding 2 regression: Verify the selected sentence for [1] is preserved while flagging the preceding uncited sentence."""
    chunks = [{
        "citation_id": 1,
        "parent_text": "Backups are retained for 30 days in off-site encrypted storage.",
        "child_text": "Backups are retained for 30 days"
    }]
    answer = "The system uses AES-256 encryption. Backups are retained for 30 days [1]."
    audit = verify_citations_grounding(answer, chunks)
    assert audit["all_supported"] is False
    assert audit["verdict"] == "UNSUPPORTED_CLAIM"
    uncited_entries = [c for c in audit["verified_citations"] if c["citation_id"] is None]
    cited_entries = [c for c in audit["verified_citations"] if c["citation_id"] == 1]
    assert len(uncited_entries) == 1
    assert uncited_entries[0]["supported"] is False
    assert "AES-256" in uncited_entries[0]["claim_excerpt"]
    assert len(cited_entries) == 1
    assert cited_entries[0]["supported"] is True
    assert cited_entries[0]["claim_excerpt"] == "Backups are retained for 30 days"


@pytest.mark.parametrize("query,source_text", [
    (
        "What is availability?",
        "Availability refers to the ability of a system to remain operational and accessible when required by authorized users."
    ),
    (
        "What is uptime?",
        "Uptime is a measure of system reliability representing continuous operational status without service interruption."
    ),
])
def test_evaluator_allows_conceptual_metric_questions_without_articles(query, source_text):
    """Conceptual-question regression: Definitional queries without 'a'/'an' must not be blocked by percentage checks."""
    chunks = [{
        "parent_text": source_text,
        "child_text": source_text,
        "dense_score": 0.82
    }]
    res = evaluate_relevance(chunks, query=query)
    assert res["status"] == "SYNTHESIZE"
    assert res["nli_verdict"] == "ENTAILED"

    # Value-seeking queries with 'the' still require the actual numerical metric
    res_value = evaluate_relevance(chunks, query="What is the uptime?")
    assert res_value["status"] == "REFUSE"

