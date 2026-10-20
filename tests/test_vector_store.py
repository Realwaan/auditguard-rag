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
