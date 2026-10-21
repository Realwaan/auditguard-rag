import pytest
from app.ast_builder import DocumentAST, ASTNode, ASTNodeType, link_cross_modal_symbols
from app.discrepancy_scanner import scan_ast_for_discrepancies, detect_heuristic_discrepancy

def test_heuristic_contradiction_detection():
    text = "The executive summary reports revenue surged by 15% across all regions."
    visual_text = "Table 4: Regional performance shows a net decline of -4% in operating profit."
    
    result = detect_heuristic_discrepancy(text, visual_text)
    assert result is not None
    assert result["status"] == "CONTRADICTION"
    assert "Growth Metric Mismatch" in result["title"]

def test_no_false_positive_on_substrings_and_hyphens():
    """Verify words containing 'up' (e.g. 'uptime', 'support') and tables with '-' do not false-trigger growth conflict."""
    text = "2.1 Uptime Guarantee: Customer support operates during standard hours."
    visual_text = "| Tier | Hours | Response |\n|---|---|---|\n| Tier-1 | 08:00-18:00 | 15 min |"
    result = detect_heuristic_discrepancy(text, visual_text)
    assert result is None

def test_pipe_table_breach_detection():
    """Verify structured Markdown table breach against prose SLA percentage is caught."""
    text = "2.1 Uptime Guarantee: The service provider guarantees 99.9% availability."
    visual_text = (
        "Table 1: Q3 Historical Availability\n"
        "| Month | Target SLA | Actual Uptime |\n"
        "| July | 99.9% | 99.95% |\n"
        "| September | 99.9% | 98.20% |"
    )
    result = detect_heuristic_discrepancy(text, visual_text)
    assert result is not None
    assert result["status"] == "CONTRADICTION"
    assert "98.2" in result["visual_fact"]

def test_ast_discrepancy_scan_end_to_end():
    ast = DocumentAST(doc_id="audit_q4", title="Q4 Audit")
    
    p = ASTNode(
        id="p_claim",
        type=ASTNodeType.PARAGRAPH,
        title="Revenue Claim",
        text="The executive report claims our conversion rate expanded to 28% in Figure 4.",
        page=2,
        materialized_path="root/sec_1/p_claim"
    )
    ast.add_node(p)

    v = ASTNode(
        id="v_fig_4",
        type=ASTNodeType.VISUAL,
        title="Figure 4: Conversion Rate Trend",
        text="Monthly conversion rate: Jan 12%, Feb 11%, Mar 10% (decline trend).",
        page=2,
        materialized_path="root/visuals/fig_4"
    )
    ast.add_node(v)

    link_cross_modal_symbols(ast)

    # Run scan without LLM calls for deterministic testing
    report = scan_ast_for_discrepancies(ast, use_llm=False)
    assert report["total_pairs_audited"] == 1
    assert report["contradictions_count"] >= 1
    assert report["verdict"] == "FLAGGED"
    assert len(report["findings"]) == 1
    assert report["findings"][0]["status"] == "CONTRADICTION"
