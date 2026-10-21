import pytest
import numpy as np
from app.ast_builder import DocumentAST, ASTNode, ASTNodeType
from app.vector_store import LanceVectorStore

def test_lancedb_indexing_and_sibling_hydration(tmp_path):
    store = LanceVectorStore(db_path=tmp_path / "test_lance")
    
    ast = DocumentAST(doc_id="finance_q3", title="Q3 Earnings Report")
    
    # 1. Text paragraph referencing Figure 1
    p1 = ASTNode(
        id="p_1",
        type=ASTNodeType.PARAGRAPH,
        title="Revenue Analysis",
        text="Total gross margin expanded by 14% as detailed in Figure 1.",
        page=3,
        materialized_path="root/sec_financials/p_1",
        linked_node_ids=["v_fig_1"]
    )
    ast.add_node(p1)

    # 2. Visual node (Figure 1 crop)
    v1 = ASTNode(
        id="v_fig_1",
        type=ASTNodeType.VISUAL,
        title="Figure 1: Margin Growth",
        text="Bar chart showing margin by quarter",
        page=3,
        materialized_path="root/visuals/v_fig_1",
        image_url="/static/crops/finance_q3_v_fig_1.webp",
        linked_node_ids=["p_1"]
    )
    ast.add_node(v1)

    # 2 nodes indexed
    embeddings = np.random.randn(2, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    tbl_name = store.index_ast(ast, embeddings)
    assert tbl_name.startswith("ast_finance_q3")

    # Search with p1's embedding
    results = store.search("finance_q3", embeddings[0], ast=ast, top_k=2)
    assert len(results) >= 1
    top_hit = results[0]
    assert top_hit["id"] == "p_1"
    # Sibling hydration check: v_fig_1 should be attached to top_hit
    assert "linked_visuals" in top_hit
    assert len(top_hit["linked_visuals"]) == 1
    assert top_hit["linked_visuals"][0]["id"] == "v_fig_1"
    assert top_hit["image_url"] == "/static/crops/finance_q3_v_fig_1.webp"


def test_ast_hierarchy_restoration_after_restart(tmp_path):
    store = LanceVectorStore(db_path=tmp_path / "test_lance_hierarchy")
    doc_id = "hierarchy_doc"
    ast = DocumentAST(doc_id=doc_id, title="Hierarchy Preservation Spec")

    sec1 = ASTNode(
        id="sec_1",
        type=ASTNodeType.SECTION,
        title="Section 1: Security Controls",
        text="Section 1 heading",
        page=1,
        materialized_path="root/sec_1",
        parent_id=ast.root_id
    )
    ast.add_node(sec1, parent_id=ast.root_id)

    p1 = ASTNode(
        id="p_1",
        type=ASTNodeType.PARAGRAPH,
        title="Access Policy",
        text="All contractors require hardware MFA keys.",
        page=1,
        materialized_path="root/sec_1/p_1",
        parent_id="sec_1"
    )
    ast.add_node(p1, parent_id="sec_1")

    v1 = ASTNode(
        id="v_1",
        type=ASTNodeType.VISUAL,
        title="MFA Architecture",
        text="Diagram of MFA token authentication flow.",
        page=1,
        materialized_path="root/sec_1/v_1",
        parent_id="sec_1",
        image_url="/static/crops/mfa_flow.webp",
        linked_node_ids=["p_1"]
    )
    ast.add_node(v1, parent_id="sec_1")

    # Verify original AST relationships
    assert sec1.id in ast.nodes[ast.root_id].children_ids
    assert p1.id in ast.nodes["sec_1"].children_ids
    assert v1.id in ast.nodes["sec_1"].children_ids

    # Index into LanceDB
    tbl_name = store.index_ast(ast)
    assert tbl_name.startswith("ast_hierarchy_doc")

    # Reconstruct from LanceDB (simulating server restart)
    reconstructed = store.reconstruct_document_data(doc_id)
    assert reconstructed is not None
    r_ast, r_chunks, r_emb = reconstructed

    # Assert reconstructed nodes exist
    assert "sec_1" in r_ast.nodes
    assert "p_1" in r_ast.nodes
    assert "v_1" in r_ast.nodes

    # Assert parent-child hierarchy is accurately restored
    assert r_ast.nodes["sec_1"].parent_id == r_ast.root_id
    assert "sec_1" in r_ast.nodes[r_ast.root_id].children_ids

    assert r_ast.nodes["p_1"].parent_id == "sec_1"
    assert "p_1" in r_ast.nodes["sec_1"].children_ids

    assert r_ast.nodes["v_1"].parent_id == "sec_1"
    assert "v_1" in r_ast.nodes["sec_1"].children_ids

    # Ancestor resolution check
    p1_ancestors = [a.id for a in r_ast.get_ancestors("p_1")]
    assert p1_ancestors == ["sec_1", r_ast.root_id]

