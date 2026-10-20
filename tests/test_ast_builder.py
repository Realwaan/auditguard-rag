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
