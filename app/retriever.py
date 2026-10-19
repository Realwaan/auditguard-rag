import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from app.embeddings import embed_query

class HybridRetriever:
    """
    Hybrid Retriever combining Sparse Lexical Search (BM25) and Dense Vector Cosine Similarity
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings  # Shape: (N, 384), pre-normalized unit vectors
        
        # Tokenize child sentences for BM25
        tokenized_corpus = [c["child_text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 4, k_rrf: int = 60) -> List[Dict[str, Any]]:
        if not self.chunks or len(self.chunks) == 0:
            return []

        # 1. Sparse BM25 Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_rank_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]

        # 2. Dense Vector Dot Product Search
        q_vec = embed_query(query)
        dense_scores = np.dot(self.embeddings, q_vec)
        dense_rank_indices = np.argsort(dense_scores)[::-1][:top_k * 3]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score = sum(1.0 / (k_rrf + rank))
        rrf_map = {}

        for rank, idx in enumerate(bm25_rank_indices):
            if bm25_scores[idx] > 0:  # only score positive lexical matches
                rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (k_rrf + rank))

        for rank, idx in enumerate(dense_rank_indices):
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (k_rrf + rank))

        # Sort indices by highest RRF score
        sorted_indices = sorted(rrf_map.keys(), key=lambda i: rrf_map[i], reverse=True)[:top_k]

        # Deduplicate parent paragraphs if multiple child sentences hit the same parent
        seen_parents = set()
        results = []
        citation_num = 1

        for idx in sorted_indices:
            chunk = self.chunks[idx].copy()
            parent = chunk["parent_text"]
            if parent in seen_parents:
                continue
            seen_parents.add(parent)

            chunk["citation_id"] = citation_num
            chunk["dense_score"] = float(dense_scores[idx])
            chunk["bm25_score"] = float(bm25_scores[idx])
            chunk["rrf_score"] = float(rrf_map[idx])
            results.append(chunk)
            citation_num += 1

        return results
