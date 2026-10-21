import pytest
from app.ast_builder import (
    ASTNode, ASTNodeType, DocumentAST,
    build_document_ast, link_cross_modal_symbols
)

def test_ast_node_and_tree_hierarchy():
    ast = DocumentAST(doc_id="test_doc", title="Test Specification")
    assert ast.root_id == "test_doc_root"
    assert ast.get_node(ast.root_id) is not None

    # Add section
    sec = ASTNode(
        id="sec_1",
        type=ASTNodeType.SECTION,
        title="1. Architecture",
        text="Section on system architecture",
        page=1,
        materialized_path="root/sec_1"
    )
    ast.add_node(sec, parent_id=ast.root_id)
    assert sec.id in ast.nodes[ast.root_id].children_ids

    # Add paragraph
    p = ASTNode(
        id="p_1",
        type=ASTNodeType.PARAGRAPH,
        title="Paragraph 1",
        text="The power system is described in Figure 2.1.",
        page=1,
        materialized_path="root/sec_1/p_1"
    )
    ast.add_node(p, parent_id="sec_1")
    assert p.id in ast.nodes["sec_1"].children_ids
    assert p.parent_id == "sec_1"

    # Ancestors
    ancestors = ast.get_ancestors("p_1")
    assert len(ancestors) == 2
    assert ancestors[0].id == "sec_1"
    assert ancestors[1].id == ast.root_id

def test_tier_1_symbol_linking_exact_match():
    ast = DocumentAST(doc_id="spec_audit", title="SOP")
    
    # Paragraph referencing Figure 3
    p = ASTNode(
        id="p_10",
        type=ASTNodeType.PARAGRAPH,
        title="Paragraph 10",
        text="Notice the high-voltage threshold illustrated in Figure 3 below.",
        page=2,
        materialized_path="root/sec_1/p_10"
    )
    ast.add_node(p)

    # Visual node titled Figure 3
    v = ASTNode(
        id="v_fig_3",
        type=ASTNodeType.VISUAL,
        title="Figure 3: High-Voltage Threshold Graph",
        text="Voltage curve vs time",
        page=2,
        materialized_path="root/visuals/fig_3"
    )
    ast.add_node(v)

    link_cross_modal_symbols(ast)

    assert "v_fig_3" in p.linked_node_ids
    assert "p_10" in v.linked_node_ids
    assert p.metadata.get("link_tier") == 1
    assert p.metadata.get("link_weight") == 1.0

def test_tier_2_spatial_adjacency_fallback():
    ast = DocumentAST(doc_id="spec_audit", title="SOP")
    
    # Paragraph with NO figure mention
    p = ASTNode(
        id="p_20",
        type=ASTNodeType.PARAGRAPH,
        title="Paragraph 20",
        text="System components require standard maintenance checks quarterly.",
        page=4,
        materialized_path="root/sec_2/p_20"
    )
    ast.add_node(p)

    # Visual node on Page 4 with no matching label in text
    v = ASTNode(
        id="v_diag_9",
        type=ASTNodeType.VISUAL,
        title="System Schematic Diagram",
        text="Schematic wire layout",
        page=4,
        materialized_path="root/visuals/diag_9"
    )
    ast.add_node(v)

    link_cross_modal_symbols(ast)

    # Tier 2 fallback must wire them because they are on the same page
    assert "v_diag_9" in p.linked_node_ids
    assert "p_20" in v.linked_node_ids
    assert v.metadata.get("link_tier") == 2
    assert v.metadata.get("link_weight") == 0.75

def test_build_document_ast_pipeline():
    sample_pages = [
        {
            "page": 1,
            "text": "## 1. SCOPE & OVERVIEW\nThis document covers standard server configuration and Figure 1.\n\nAll servers must comply with 99.9% uptime."
        },
        {
            "page": 2,
            "text": "## 2. DISASTER RECOVERY\nIn the event of an outage, failover occurs within 60 seconds."
        }
    ]
    ast = build_document_ast("demo_sop", sample_pages)
    
    dict_repr = ast.to_dict()
    assert dict_repr["doc_id"] == "demo_sop"
    assert dict_repr["total_nodes"] >= 4
    assert len(ast.get_leaves()) >= 2

    # Verify RAPTOR thematic section summary edge
    sec_nodes = [n for n in ast.nodes.values() if n.type == ASTNodeType.SECTION]
    assert len(sec_nodes) >= 1
    summarized_sec = [s for s in sec_nodes if "thematic_summary" in s.metadata]
    assert len(summarized_sec) >= 1
    assert len(summarized_sec[0].metadata["thematic_summary"]) > 0


def test_table_visual_node_not_duplicated_as_paragraph():
    """Verify that Markdown tables create a VISUAL node without creating a duplicate PARAGRAPH node."""
    sample_pages = [
        {
            "page": 1,
            "text": (
                "## 2. SYSTEM UPTIME & SLA\n"
                "Uptime commitment is 99.9% as reported in Table 1.\n\n"
                "Table 1: Monthly Availability Log\n"
                "| Month | Target | Actual |\n"
                "| July | 99.9% | 99.95% |\n"
                "| Sept | 99.9% | 98.20% |\n"
            )
        }
    ]
    ast = build_document_ast("table_dedup_test", sample_pages)
    
    # Exactly one visual table node should exist
    vis_nodes = [n for n in ast.nodes.values() if n.type == ASTNodeType.VISUAL]
    assert len(vis_nodes) == 1
    assert "Table 1" in vis_nodes[0].title
    assert "98.20%" in vis_nodes[0].text

    # The paragraph nodes should NOT include a duplicate copy of the table text
    para_nodes = [n for n in ast.nodes.values() if n.type == ASTNodeType.PARAGRAPH]
    assert len(para_nodes) == 1
    assert "Uptime commitment is 99.9%" in para_nodes[0].text
    assert not any("Monthly Availability Log" in p.text and "| July |" in p.text for p in para_nodes)


def test_slide_deck_numbered_dash_sections():
    """Verify that slide decks using '1-1: Agenda' and '2-1: System Overview' branch into distinct sections."""
    sample_pages = [
        {
            "page": 1,
            "text": "1-1: Executive Overview\nWelcome to the operational review meeting."
        },
        {
            "page": 2,
            "text": "2-1: Microservice Architecture\nDetailed diagrams of the streaming ingestion cluster."
        },
        {
            "page": 3,
            "text": "3-1: Compliance Audit Results\nFinal scorecard across all seven SOC2 criteria."
        }
    ]
    ast = build_document_ast("slide_deck_test", sample_pages)

    sec_nodes = [n for n in ast.nodes.values() if n.type == ASTNodeType.SECTION]
    sec_titles = [s.title for s in sec_nodes]

    # Must produce 3 separate sections rather than collapsing under a single default section
    assert len(sec_nodes) == 3
    assert any("1-1: Executive Overview" in t for t in sec_titles)
    assert any("2-1: Microservice Architecture" in t for t in sec_titles)
    assert any("3-1: Compliance Audit Results" in t for t in sec_titles)

