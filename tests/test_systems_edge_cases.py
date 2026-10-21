import pytest
import numpy as np
from app.embeddings import embed_texts
from app.retriever import HybridRetriever
from app.ast_builder import DocumentAST, ASTNodeType, ASTNode
from app.discrepancy_scanner import scan_ast_for_discrepancies
from app.audit_scanner import run_compliance_audit

def test_embed_texts_empty_input():
    """Verify embed_texts handles empty list gracefully without AxisError."""
    result = embed_texts([])
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 384)

def test_hybrid_retriever_empty_corpus():
    """Verify HybridRetriever initializes without ZeroDivisionError on empty chunks."""
    retriever = HybridRetriever([], np.zeros((0, 384), dtype=np.float32))
    assert retriever.bm25 is None
    results = retriever.retrieve("any search query", top_k=4)
    assert results == []

def test_discrepancy_scanner_empty_ast():
    """Verify discrepancy scanner on empty AST produces valid report without falsely asserting 100% consistency."""
    empty_ast = DocumentAST(doc_id="empty_test")
    report = scan_ast_for_discrepancies(empty_ast, use_llm=False)
    assert report["total_pairs_audited"] == 0
    assert report["contradictions_count"] == 0
    assert report["verdict"] in ("NO_PAIRS", "UNAUDITED")
    assert report["consistency_score"] == 0.0

def test_compliance_audit_empty_corpus():
    """Verify pre-flight compliance audit handles empty retriever without crashing."""
    retriever = HybridRetriever([], np.zeros((0, 384), dtype=np.float32))
    report = run_compliance_audit(retriever)
    assert report["total_checks"] == 7
    assert report["passed_count"] == 0
    assert report["gaps_count"] == 7
    assert report["overall_score"] == 0.0
