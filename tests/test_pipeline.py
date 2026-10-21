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

def test_compliance_audit_scanner():
    from app.audit_scanner import run_compliance_audit
    sample_text = """1. UPTIME SLA GUARANTEE: We guarantee 99.9% availability during business hours.
2. PENALTY CREDIT: In the event of a breach, a 15% billing penalty credit applies to the invoice."""
    pages = extract_pages_from_text(sample_text, "test_sla.txt")
    chunks, _ = hierarchical_chunk(pages)
    embeddings = np.random.randn(len(chunks), 384).astype(np.float32)
    # Unit normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    retriever = HybridRetriever(chunks, embeddings)
    
    report = run_compliance_audit(retriever)
    assert "overall_score" in report
    assert "checklist" in report
    assert len(report["checklist"]) == 7
    assert "verdict" in report
    assert report["remediation_playbook_available"] is True
    assert "remediation_count" in report
    
    # Verify Robin/Ironclad automated redline diff integration
    gap_items = [item for item in report["checklist"] if item["status"] == "GAP"]
    assert len(gap_items) > 0
    first_gap = gap_items[0]
    assert first_gap["remediation_clause"] is not None
    assert first_gap["redline_diff"] is not None
    assert "+ " in first_gap["redline_diff"]
    assert "action_plan" in first_gap

def test_nli_entailment_evaluator():
    from app.evaluator import evaluate_relevance
    # Entailed chunk
    grounded_chunks = [{"dense_score": 0.85, "child_text": "Clause 1 verified."}]
    res = evaluate_relevance(grounded_chunks)
    assert res["status"] == "SYNTHESIZE"
    assert res["nli_verdict"] == "ENTAILED"
    assert res["nli_entailment_prob"] >= 0.80
    assert "ENTAILED" in res["badge"]

    # Refusal chunk
    ungrounded_chunks = [{"dense_score": 0.20, "child_text": "Unrelated topic."}]
    ref_res = evaluate_relevance(ungrounded_chunks)
    assert ref_res["status"] == "REFUSE"
    assert "NEUTRAL" in ref_res["nli_verdict"]
    assert "OUT-OF-BOUNDS" in ref_res["badge"]
