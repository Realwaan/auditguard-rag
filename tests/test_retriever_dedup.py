import numpy as np
import pytest
from app.retriever import HybridRetriever

def test_retriever_deduplicates_parents_without_dropping_top_k():
    """
    Verify that when multiple matching child sentences come from the same parent paragraph,
    the retriever continues scanning candidates until top_k distinct parent paragraphs
    are returned (Issue 7 fix).
    """
    chunks = [
        {"child_text": "Sentence 1 of paragraph A", "parent_text": "Paragraph A full text"},
        {"child_text": "Sentence 2 of paragraph A", "parent_text": "Paragraph A full text"},
        {"child_text": "Sentence 3 of paragraph A", "parent_text": "Paragraph A full text"},
        {"child_text": "Sentence 1 of paragraph B", "parent_text": "Paragraph B full text"},
        {"child_text": "Sentence 1 of paragraph C", "parent_text": "Paragraph C full text"},
    ]
    # Synthetic embeddings matrix with 5 rows matching 384 dimensions
    embeddings = np.random.randn(5, 384).astype(np.float32)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    retriever = HybridRetriever(chunks, embeddings)
    
    # Query matching chunk 0, 1, 2 heavily
    results = retriever.retrieve("sentence", top_k=2)
    
    # We requested top_k=2 unique parent paragraphs
    assert len(results) == 2
    parent_texts = [r["parent_text"] for r in results]
    assert len(set(parent_texts)) == 2
    assert "Paragraph A full text" in parent_texts
    assert parent_texts[0] != parent_texts[1]
