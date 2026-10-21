import numpy as np

from app.ast_builder import build_document_ast, ASTNodeType
from app.ingest import hierarchical_chunk
from app.vector_store import LanceVectorStore


def test_short_fact_survives_beside_table_and_long_paragraph():
    text = 'Response: 4 hours.\n\nA detailed description of the support policy.\n\n| Metric | Value |\n| Retention | 30 days |'
    pages = [{'page': 1, 'text': text, 'filename': 'policy.md'}]
    ast = build_document_ast('qa_policy', pages)
    chunks, _ = hierarchical_chunk(pages, ast=ast)
    assert any('Response: 4 hours.' in c['parent_text'] for c in chunks)
    assert any('Retention | 30 days' in c['parent_text'] for c in chunks)
    assert 'Retention | 30 days' in ast.get_page_text(1)
    assert all(c['ast_node_id'] in ast.nodes for c in chunks)
    assert len([n for n in ast.get_leaves() if n.type == ASTNodeType.VISUAL]) == 1


def test_visual_only_document_and_scan_metadata_survive_restart(tmp_path):
    pages = [
        {'page': 1, 'filename': 'policy.pdf', 'text': '| Metric | Value |\n| Retention | 30 days |'},
        {'page': 2, 'filename': 'policy.pdf', 'text': 'Page 2 (Visual / Scanned Content)', 'is_scanned': True, 'is_ocr': False},
    ]
    ast = build_document_ast('qa_restore', pages)
    chunks, _ = hierarchical_chunk(pages, {2: '/static/previews/qa_page2.webp'}, ast=ast)
    store = LanceVectorStore(tmp_path / 'qa_lance')
    nodes = [n for n in ast.nodes.values() if n.type != ASTNodeType.ROOT]
    store.index_ast(ast, np.eye(len(nodes), 384, dtype=np.float32))
    restored = store.reconstruct_document_data('qa_restore')
    assert restored is not None
    _, restored_chunks, vectors = restored
    assert len(vectors) == len(restored_chunks)
    assert {c['ast_node_id'] for c in restored_chunks} == {c['ast_node_id'] for c in chunks}
    scan = next(c for c in restored_chunks if c['page'] == 2)
    assert scan['filename'] == 'policy.pdf'
    assert scan['is_scanned'] is True and scan['is_ocr'] is False
    assert scan['image_url'] == '/static/previews/qa_page2.webp'
