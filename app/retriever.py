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
        if not chunks:
            self.bm25 = None
            return

        tokenized_corpus = [c["child_text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 4, k_rrf: int = 60) -> List[Dict[str, Any]]:
        if not self.chunks or len(self.chunks) == 0 or self.bm25 is None:
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

    def escalate_page_scan(
        self,
        query: str,
        candidate_pages: List[int],
        ast: Any = None,
        min_threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Tier-2 CRAG Hierarchical Tree Escalation:
        When chunk-level confidence falls below threshold (< 0.50), promotes the search
        to Whole-Page scanning across candidate pages to recover multi-paragraph context.
        Returns escalated whole-page chunks if verified; otherwise empty list.
        """
        if not self.chunks:
            return []

        # 1. Determine candidate pages to scan
        pages_to_check = candidate_pages
        if not pages_to_check:
            if ast and hasattr(ast, "get_all_pages"):
                pages_to_check = ast.get_all_pages()
            else:
                pages_to_check = sorted(list({c.get("page", 1) for c in self.chunks}))

        if not pages_to_check:
            return []

        import re
        q_vec = embed_query(query)
        stop_words = {"what", "is", "the", "are", "how", "for", "and", "with", "does", "can", "in", "of", "to", "a"}
        raw_tokens = [re.sub(r'[^a-zA-Z0-9]', '', t) for t in query.lower().split()]
        query_tokens = [t for t in raw_tokens if len(t) > 2 and t not in stop_words]
        if not query_tokens:
            query_tokens = [t for t in raw_tokens if len(t) > 2]

        best_page = None
        best_score = 0.0
        best_text = ""

        for p in pages_to_check:
            if ast and hasattr(ast, "get_page_text"):
                page_text = ast.get_page_text(p)
            else:
                page_chunks = [c["parent_text"] for c in self.chunks if c.get("page") == p]
                page_text = "\n\n".join(dict.fromkeys(page_chunks))

            if not page_text.strip():
                continue

            # Dense similarity against chunks on this page
            page_dense_scores = [
                float(np.dot(self.embeddings[i], q_vec))
                for i, c in enumerate(self.chunks)
                if c.get("page") == p
            ]
            max_dense = max(page_dense_scores) if page_dense_scores else 0.0

            # Lexical coverage of query tokens across the whole page text
            page_lower = page_text.lower()
            matching_tokens = sum(1 for t in query_tokens if t in page_lower)
            lexical_ratio = matching_tokens / max(1, len(query_tokens))

            # Composite whole-page relevance score
            # If key query concepts span across paragraphs on this page, boost confidence to ground synthesis
            boosted_lexical = round(0.55 + (lexical_ratio * 0.35), 3) if lexical_ratio >= 0.50 else round(lexical_ratio * 0.50, 3)
            page_score = max(max_dense, boosted_lexical)

            if page_score > best_score:
                best_score = page_score
                best_page = p
                best_text = page_text

        # If whole-page context satisfies threshold, promote to whole-page synthesis
        if best_page is not None and best_score >= min_threshold:
            filename = ast.title if (ast and hasattr(ast, "title")) else self.chunks[0].get("filename", "Document")
            return [{
                "citation_id": 1,
                "child_text": f"Whole-Page Scan (Page {best_page}): {best_text[:280]}...",
                "parent_text": best_text,
                "filename": filename,
                "page": best_page,
                "dense_score": round(best_score, 3),
                "bm25_score": 1.0,
                "rrf_score": 1.0,
                "escalated_from": "whole_page_scan"
            }]

        return []
