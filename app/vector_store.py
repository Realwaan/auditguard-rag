"""
Embedded LanceDB Vector & Columnar Store for Multimodal AST Nodes.
Powered by Apache Arrow and in-process Rust engine.
Enables sub-5ms vector retrieval with materialized path filtering
and automatic cross-modal sibling hydration.
"""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import lancedb

from app.config import INDEX_DIR
from app.ast_builder import DocumentAST, ASTNode, ASTNodeType
from app.embeddings import embed_texts

LANCE_DB_PATH = INDEX_DIR / "lancedb"


class LanceVectorStore:
    """Embedded LanceDB vector store for AST nodes."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or LANCE_DB_PATH
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    def is_ready(self) -> bool:
        """Verifies that the underlying LanceDB connection and directory are operational."""
        try:
            if hasattr(self.db, "list_tables"):
                _ = self.db.list_tables()
            else:
                _ = self.db.table_names()
            return self.db_path.exists()
        except Exception:
            return False

    def _get_table_name(self, doc_id: str) -> str:
        safe_id = "".join(c if c.isalnum() else "_" for c in doc_id)
        return f"ast_{safe_id}"

    def has_document(self, doc_id: str) -> bool:
        try:
            self.db.open_table(self._get_table_name(doc_id))
            return True
        except Exception:
            return False

    def index_ast(
        self,
        ast: DocumentAST,
        embeddings_matrix: Optional[np.ndarray] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Indexes all leaf nodes (paragraphs) and visual nodes into a LanceDB table.
        If embeddings_matrix is omitted or has a different row count than AST leaves,
        computes exact embeddings for the AST leaves so vectors strictly match node texts.
        """
        leaves = [
            n for n in ast.nodes.values()
            if n.type in (ASTNodeType.PARAGRAPH, ASTNodeType.VISUAL)
        ]
        if not leaves:
            return ""

        if embeddings_matrix is None or len(embeddings_matrix) != len(leaves):
            started = time.perf_counter()
            leaf_texts = [f"{n.title}: {n.text}" if n.title else n.text for n in leaves]
            embeddings_matrix = embed_texts(leaf_texts)
            if timings is not None:
                timings["leaf_embeddings_ms"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        table_name = self._get_table_name(ast.doc_id)
        records = []
        for idx, node in enumerate(leaves):
            vec = embeddings_matrix[idx].tolist() if idx < len(embeddings_matrix) else [0.0] * 384
            records.append({
                "id": str(node.id),
                "vector": vec,
                "doc_id": str(ast.doc_id),
                "node_type": str(node.type.value),
                "title": str(node.title),
                "text": str(node.text),
                "page": int(node.page),
                "materialized_path": str(node.materialized_path),
                "image_url": str(node.image_url or ""),
                "visual_type": str(node.visual_type or ""),
                "parent_id": str(node.parent_id or ""),
                "linked_node_ids": json.dumps(node.linked_node_ids),
                "metadata": json.dumps(node.metadata),
            })

        self.db.create_table(table_name, data=records, mode="overwrite")
        if timings is not None:
            timings["database_write_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return table_name

    def search(
        self,
        doc_id: str,
        query_vector: np.ndarray,
        ast: Optional[DocumentAST] = None,
        top_k: int = 4,
        path_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search over LanceDB with sibling hydration:
        If a text node is retrieved, automatically pulls its linked visual crop.
        """
        table_name = self._get_table_name(doc_id)
        try:
            tbl = self.db.open_table(table_name)
        except Exception:
            return []
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else list(query_vector)

        search_builder = tbl.search(query_list).limit(max(1, top_k * 2))
        if path_prefix:
            safe_prefix = re.sub(r"[^a-zA-Z0-9_/\-.]", "", path_prefix)
            if safe_prefix:
                search_builder = search_builder.where(f"materialized_path LIKE '{safe_prefix}%'")

        raw_results = search_builder.to_list()

        results = []
        for r in raw_results[:top_k]:
            linked_ids = json.loads(r.get("linked_node_ids", "[]"))
            node_id = r.get("id")

            # Calculate similarity score from LanceDB squared L2 distance on unit-normalized vectors:
            # For unit vectors u, v: ||u - v||^2 = 2 - 2*cos(u, v) => cos(u, v) = 1 - (dist / 2)
            dist = float(r.get("_distance", 0.0))
            sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            item: Dict[str, Any] = {
                "id": node_id,
                "doc_id": r.get("doc_id"),
                "node_type": r.get("node_type"),
                "title": r.get("title"),
                "text": r.get("text"),
                "page": r.get("page"),
                "materialized_path": r.get("materialized_path"),
                "image_url": r.get("image_url") or None,
                "visual_type": r.get("visual_type") or None,
                "parent_id": r.get("parent_id"),
                "linked_node_ids": linked_ids,
                "dense_score": round(sim_score, 4),
                "has_visual": bool(r.get("image_url")),
            }

            # Sibling Resolution: pull linked visual node data from AST if available
            if ast and linked_ids:
                linked_visuals = []
                for lid in linked_ids:
                    linked_node = ast.get_node(lid)
                    if linked_node and linked_node.type == ASTNodeType.VISUAL:
                        linked_visuals.append({
                            "id": linked_node.id,
                            "title": linked_node.title,
                            "image_url": linked_node.image_url,
                            "visual_type": linked_node.visual_type,
                            "bbox": linked_node.bbox,
                            "page": linked_node.page,
                        })
                if linked_visuals:
                    item["linked_visuals"] = linked_visuals
                    if not item["image_url"]:
                        item["image_url"] = linked_visuals[0]["image_url"]
                        item["has_visual"] = True

            results.append(item)

        return results
