import pytest
import numpy as np
from app.ingest import extract_pages_from_text, hierarchical_chunk
from app.retriever import HybridRetriever
from app.evaluator import evaluate_relevance

def test_hierarchical_chunking():
    sample_text = """Paragraph one has multiple sentences. It outlines clause 1.1 with specific details.

Paragraph two defines the penalty credit. Any violation results in a $5,000 fine."""
    pages = extract_pages_from_text(sample_text, "test_doc.txt")
    chunks, classification = hierarchical_chunk(pages)

    assert len(chunks) >= 2
    # Verify child sentence is distinct from parent paragraph
    assert "child_text" in chunks[0]
    assert "parent_text" in chunks[0]
    assert len(chunks[0]["parent_text"]) >= len(chunks[0]["child_text"])

def test_evaluator_crag_refusal():
    # Low score chunks should trigger REFUSE
    dummy_chunks = [
        {"child_text": "irrelevant text", "dense_score": 0.25}
    ]
    eval_res = evaluate_relevance(dummy_chunks)
    assert eval_res["status"] == "REFUSE"
    assert "OUT-OF-BOUNDS" in eval_res["badge"]

def test_evaluator_crag_acceptance():
    # High score chunks should trigger SYNTHESIZE
    dummy_chunks = [
        {"child_text": "relevant policy text", "dense_score": 0.78}
    ]
    eval_res = evaluate_relevance(dummy_chunks)
    assert eval_res["status"] == "SYNTHESIZE"
    assert "GROUNDED" in eval_res["badge"]
