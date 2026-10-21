"""Regressions for misleading answers reported against commit 3978500.

These checks use explicit evidence and fake provider responses, never a live API.
"""
from types import SimpleNamespace

import pytest

from app.evaluator import evaluate_relevance, extract_query_keywords, verify_generated_answer
from app.generator import generate_grounded_answer, verify_citations_grounding


def evidence(text, **extra):
    return {"citation_id": 1, "parent_text": text, "child_text": text,
            "dense_score": 0.8, "filename": "test.txt", "page": 1, **extra}


def test_two_letter_technical_terms_are_not_discarded():
    assert extract_query_keywords("What is AI?") == ["ai"]
    result = evaluate_relevance([evidence("Our office opens on Monday.")], "AI")
    assert result["status"] == "REFUSE"


def test_plural_definition_and_instruction_words_do_not_hide_answer():
    result = evaluate_relevance(
        [evidence("A microprocessor is a CPU on a single integrated circuit.")],
        "What are microprocessors? Please cite the exact page.",
    )
    assert result["status"] == "SYNTHESIZE"


def test_quantitative_question_without_number_is_refused_even_with_all_keywords():
    result = evaluate_relevance(
        [evidence("The penalty credit percentage for an uptime breach is under review.")],
        "What is the penalty credit percentage for an uptime breach?",
    )
    assert result["status"] == "REFUSE"


def test_unrelated_number_does_not_verify_missing_percentage():
    result = evaluate_relevance(
        [evidence("Section 2 discusses the penalty credit percentage for an uptime breach; the amount is undecided.")],
        "What is the penalty credit percentage for an uptime breach?",
    )
    assert result["status"] == "REFUSE"


@pytest.mark.parametrize("answer,source", [
    ("Support response time is 8 hours [1].", "Support response time is 4 hours."),
    ("Uptime is 95.9% [1].", "Uptime is 99.9%."),
    ("Uptime is 95.9%. [1]", "Uptime is 99.9%."),
    ("The fee is 15% [1].", "The fee is 115%."),
    ("The fee is 15% [1].", "The fee is 15 dollars."),
    ("Uptime is 99.9% [1]. Support is free forever.", "Uptime is 99.9%."),
    ("I cannot verify the date. Support response is 8 hours [1].", "Support response is 4 hours."),
])
def test_citation_audit_rejects_known_false_support(answer, source):
    assert verify_citations_grounding(answer, [evidence(source)])["all_supported"] is False


@pytest.mark.parametrize("answer", ["", "I CANNOT VERIFY THIS INFORMATION IN THE PROVIDED DOCUMENT EXCERPTS."])
def test_refusal_clears_positive_metadata(answer):
    result = verify_generated_answer(answer, [evidence("Microprocessor overview.")], {
        "status": "SYNTHESIZE", "confidence": 79.3, "nli_verdict": "ENTAILED",
        "grounding_verdict": "SUPPORTED", "claim_evidence_verified": True,
    })
    assert result["status"] == "REFUSE"
    assert result["claim_evidence_verified"] is False
    assert result["grounding_verdict"] == "UNVERIFIABLE"
    assert "ENTAILED" not in result["nli_verdict"]


def test_outage_does_not_present_unrelated_paragraph_as_answer(monkeypatch):
    import app.generator as generator
    monkeypatch.setattr(generator, "USE_GOOGLE_GENAI", False)
    def unavailable():
        raise RuntimeError("Provider unavailable")
    monkeypatch.setattr(generator, "get_openai_client", unavailable)
    result = generate_grounded_answer("Explain market growth", [evidence(
        "Looking ahead to 2027, our focus is AI-driven workflows. The market is moving faster than ever."
    )])
    assert result["is_fallback"] is True
    assert "Looking ahead" not in result["answer"]
    assert result["citation_verification"]["verdict"] == "REFUSAL"


def test_unanswered_query_is_not_verified(monkeypatch):
    result = verify_generated_answer(
        "The market is moving faster than ever [1].",
        [evidence("The market is moving faster than ever.")],
        {"status": "SYNTHESIZE", "confidence": 80.0},
        query="Explain market growth",
    )
    assert result["status"] != "SYNTHESIZE"
    assert result["claim_evidence_verified"] is False


def test_valid_cited_answer_remains_usable():
    result = verify_generated_answer(
        "A microprocessor is a CPU on a single integrated circuit [1].",
        [evidence("A microprocessor is a CPU on a single integrated circuit.")],
        {"status": "SYNTHESIZE", "confidence": 80.0}, query="What are microprocessors?",
    )
    assert result["status"] == "SYNTHESIZE"
    assert result["citation_verified"] is True


def test_empty_provider_response_is_not_success(monkeypatch):
    import app.generator as generator
    monkeypatch.setattr(generator, "USE_GOOGLE_GENAI", False)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
    monkeypatch.setattr(generator, "get_openai_client", lambda: client)
    result = generate_grounded_answer("What is uptime?", [evidence("Uptime measures availability.")])
    assert result["is_fallback"] is True
    assert "No response from Gemma" not in result["answer"]


def test_negated_source_can_answer_a_question_about_restrictions():
    result = evaluate_relevance([evidence("Employees must not store credentials on personal devices.")],
                                "Explain restrictions on storing credentials on personal devices.")
    assert result["status"] == "SYNTHESIZE"


def test_percentage_written_as_words_is_answer_bearing():
    assert evaluate_relevance([evidence("The penalty credit is fifteen percent.")],
                              "What is the penalty credit percentage?")["status"] == "SYNTHESIZE"


def test_retrieval_keeps_distinct_pages_and_handles_punctuation(monkeypatch):
    import numpy as np
    import app.retriever as retrieval
    monkeypatch.setattr(retrieval, "embed_query", lambda _: np.array([1.0, 0.0]))
    chunks = [evidence("Microprocessors: an introduction.", page=1),
              evidence("Microprocessors: an introduction.", page=2),
              evidence("A different paragraph.", page=3),
              evidence("Employee leave policy.", page=4),
              evidence("Office parking rules.", page=5)]
    # Keep the search term rare enough for BM25 to assign a positive IDF.
    r = retrieval.HybridRetriever(chunks, np.array([[1., 0.], [.9, 0.], [.5, 0.], [.2, 0.], [.1, 0.]]))
    assert r.retrieve("microprocessors?", top_k=0) == []
    results = r.retrieve("microprocessors?", top_k=3)
    assert {c["page"] for c in results} == {1, 2, 3}
    assert any(c["bm25_score"] > 0 for c in results if c["page"] in (1, 2))
