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
    from app.embeddings import embed_query

    q_vec = embed_query("sentence")
    # Set Paragraph A chunks to rank at the very top (slots 1, 2, 3)
    embeddings = np.zeros((5, 384), dtype=np.float32)
    embeddings[0] = q_vec
    embeddings[1] = q_vec
    embeddings[2] = q_vec
    embeddings[3] = q_vec * 0.8
    embeddings[4] = q_vec * 0.5

    retriever = HybridRetriever(chunks, embeddings)
    
    # Query matching chunk 0, 1, 2 heavily
    results = retriever.retrieve("sentence", top_k=2)
    
    # We requested top_k=2 unique parent paragraphs
    assert len(results) == 2
    parent_texts = [r["parent_text"] for r in results]
    assert len(set(parent_texts)) == 2
    assert "Paragraph A full text" in parent_texts
    assert parent_texts[0] != parent_texts[1]
