import numpy as np

from app.evaluator import evaluate_relevance
from app.retriever import HybridRetriever


def test_visual_inspection_budget_requires_page_selection(monkeypatch):
    import app.retriever as retriever_module
    monkeypatch.setattr(retriever_module, 'MAX_VISUAL_SCAN_PAGES', 2)
    monkeypatch.setattr(retriever_module, 'embed_query', lambda query: np.array([1., 0.]))
    chunks = [{'page': p, 'filename': 'scan.pdf', 'parent_text': f'Page {p} (Visual / Scanned Content)',
               'child_text': f'Page {p} (Visual / Scanned Content)', 'is_scanned': True, 'is_ocr': False,
               'has_visual': True, 'image_url': f'/static/previews/p{p}.webp'} for p in range(1, 5)]
    retriever = HybridRetriever(chunks, np.tile([1., 0.], (4, 1)))
    result = retriever.retrieve('What is the support response time?', top_k=1)
    assert evaluate_relevance(result, 'What is the support response time?')['grounding_verdict'] == 'INCOMPLETE_VISUAL_COVERAGE'
    selected = retriever.retrieve('What is the support response time on page 1?', top_k=1)
    assert [c['page'] for c in selected] == [1]
    assert evaluate_relevance(selected, 'What is the support response time on page 1?')['status'] == 'SYNTHESIZE'
    assert retriever.retrieve('What is on page 99?', top_k=1) == []
