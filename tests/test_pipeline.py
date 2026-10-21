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

    # Verify a clause that explicitly negates uptime guarantee fails substance verification
    from app.audit_scanner import verify_control_substance, DEFAULT_COMPLIANCE_CHECKS
    sla_check = DEFAULT_COMPLIANCE_CHECKS[0]
    negated = verify_control_substance("The vendor provides no uptime guarantee and operates best effort only.", sla_check)
    assert negated["substantive_pass"] is False

def test_post_generation_guardrail_verification():
    from app.evaluator import verify_generated_answer
    retrieved = [{"citation_id": 1, "parent_text": "Uptime is 99.9%"}, {"citation_id": 2, "parent_text": "Penalty is 15%"}]
    initial = {"status": "SYNTHESIZE", "confidence": 82.0, "badge": "🛡️ GROUNDED"}

    # Valid citation [1] and [2]
    v_ok = verify_generated_answer("The uptime is 99.9% [1] with a 15% credit [2].", retrieved, initial)
    assert v_ok["status"] == "SYNTHESIZE"
    assert v_ok["verified_citations"] == [1, 2]
    assert v_ok["invalid_citations"] == []
    assert v_ok["citation_verified"] is True

    # Hallucinated citation [9]
    v_bad = verify_generated_answer("The uptime is 99.9% [1] and [9].", retrieved, initial)
    assert v_bad["invalid_citations"] == [9]
    assert v_bad["citation_verified"] is False

    # LLM explicit refusal flips status to REFUSE
    v_ref = verify_generated_answer("I cannot verify this information in the provided document excerpts.", retrieved, initial)
    assert v_ref["status"] == "REFUSE"

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


def test_crag_whole_page_escalation_promotion():
    from app.ast_builder import DocumentAST, ASTNode, ASTNodeType
    from app.evaluator import evaluate_relevance

    # 1. Low chunk score initially triggers REFUSE
    low_chunks = [{"dense_score": 0.32, "page": 1, "child_text": "Initial low match.", "parent_text": "Low match"}]
    initial_eval = evaluate_relevance(low_chunks)
    assert initial_eval["status"] == "REFUSE"

    # 2. Build AST with Page 1 containing complete context
    ast = DocumentAST("doc_escalate_test", "Escalation Test Contract")
    ast.add_node(ASTNode("p1", ASTNodeType.PARAGRAPH, "Intro", "The service provider agrees to obligations.", 1, "root/p1"))
    ast.add_node(ASTNode("p2", ASTNodeType.PARAGRAPH, "Cap", "Liability limit and cap shall not exceed 500,000 USD.", 1, "root/p2"))

    # 3. Dummy HybridRetriever with 2 chunks
    dummy_chunks = [
        {"parent_text": "The service provider agrees to obligations.", "child_text": "agrees", "page": 1},
        {"parent_text": "Liability limit and cap shall not exceed 500,000 USD.", "child_text": "Liability limit", "page": 1}
    ]
    embeddings = np.array([[0.3, 0.4] + [0.0]*382, [0.35, 0.45] + [0.0]*382], dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    retriever = HybridRetriever(dummy_chunks, embeddings)

    # 4. Escalate to whole-page scan for in-bounds query
    escalated = retriever.escalate_page_scan("What is the liability limit cap?", [1], ast=ast, min_threshold=0.50)
    assert len(escalated) == 1
    assert escalated[0]["page"] == 1
    assert escalated[0]["escalated_from"] == "whole_page_scan"
    assert "Liability limit and cap" in escalated[0]["parent_text"]

    # 5. Promoted evaluation succeeds with PROMOTED TO PAGE SCAN badge
    promoted_eval = evaluate_relevance(escalated)
    assert promoted_eval["status"] == "SYNTHESIZE"
    assert "PROMOTED TO PAGE SCAN" in promoted_eval["badge"]

    # 6. Out-of-bounds query still returns empty and triggers refusal
    oob_escalated = retriever.escalate_page_scan("Secret recipe for fried chicken", [1], ast=ast, min_threshold=0.50)
    assert len(oob_escalated) == 0
