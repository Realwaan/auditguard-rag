"""Search-ready ingestion, versioned disk reuse, and optional background indexing."""
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock

import numpy as np

import app.config as cfg
from app.ast_builder import ASTNode, ASTNodeType, DocumentAST, build_document_ast
from app.embeddings import embed_texts
from app.ingest import extract_pages_from_pdf, extract_pages_from_text, hierarchical_chunk, render_page_previews
from app.retriever import HybridRetriever

logger = logging.getLogger(__name__)
# Bump when extraction, chunking, classification, or visual metadata changes.
PIPELINE_VERSION = "2-lazy-visuals"
CACHE_DIR = cfg.INDEX_DIR / "ingestion-cache"


def document_cache_key(contents: bytes, filename: str) -> str:
    kind = "pdf" if filename.lower().endswith(".pdf") else "text"
    signature = json.dumps([PIPELINE_VERSION, cfg.DEFAULT_EMBED_MODEL, kind]).encode()
    return hashlib.sha256(signature + b"\0" + contents).hexdigest()


def _load_cache(key: str):
    try:
        with np.load(CACHE_DIR / f"{key}.npz", allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            vectors = saved["vectors"].copy()
        if metadata["key"] != key:
            return None
        chunks = metadata["chunks"]
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            return None
        ast_data = metadata["ast"]
        ast = DocumentAST(ast_data["doc_id"], ast_data["title"])
        ast.nodes = {}
        for node_id, fields in ast_data["nodes"].items():
            fields["type"] = ASTNodeType(fields["type"])
            ast.nodes[node_id] = ASTNode(**fields)
        if ast.root_id not in ast.nodes or ast.doc_id != f"doc_{key[:32]}":
            return None
        if metadata["is_pdf"] and (
            not (cfg.DOCS_DIR / f"{ast.doc_id}.pdf").exists()
            or not (cfg.PREVIEWS_DIR / f"{ast.doc_id}_page_1.webp").exists()
        ):
            return None
        return ast, chunks, metadata["classification"], vectors, metadata["pages"]
    except FileNotFoundError:
        return None
    except Exception:
        # A partial/old/corrupt cache must never prevent a fresh attachment.
        logger.warning("Ignoring unreadable ingestion cache %s", key, exc_info=True)
        return None


def _save_cache(key, ast, chunks, classification, vectors, pages, is_pdf):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({
        "key": key, "ast": ast.to_dict(), "chunks": chunks,
        "classification": classification, "pages": pages, "is_pdf": is_pdf,
    })
    # One atomic file: readers cannot see metadata without its matching vectors.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=".npz", delete=False) as output:
            tmp_path = output.name
            np.savez_compressed(output, metadata=metadata, vectors=vectors)
        os.replace(tmp_path, CACHE_DIR / f"{key}.npz")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_cached_ast(doc_id: str):
    """Allow saved citation URLs to resolve after restart, without selecting a document."""
    if not re.fullmatch(r"doc_[0-9a-f]{32}", doc_id):
        return None
    for path in CACHE_DIR.glob(f"{doc_id[4:]}*.npz"):
        cached = _load_cache(path.stem)
        if cached and cached[0].doc_id == doc_id:
            return cached[0]
    return None


def prepare_document(contents: bytes, filename: str) -> dict:
    """Perform only work required for chat search; no leaf embedding pass here."""
    started = time.perf_counter()
    timings = dict.fromkeys((
        "cache_lookup_ms", "extraction_ms", "previews_ms", "document_tree_ms",
        "chunking_ms", "child_embeddings_ms", "cache_write_ms", "retriever_ms",
    ), 0.0)

    def measure(stage, function, *args, **kwargs):
        before = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings[stage] = round((time.perf_counter() - before) * 1000, 2)

    key_started = time.perf_counter()
    key = document_cache_key(contents, filename)
    cached = _load_cache(key)
    timings["cache_lookup_ms"] = round((time.perf_counter() - key_started) * 1000, 2)
    is_pdf = filename.lower().endswith(".pdf")
    cache_warning = None
    if cached:
        ast, chunks, classification, vectors, page_count = cached
        # Content reuse also supports renames; citations retain the uploaded name.
        ast.title = filename
        ast.nodes[ast.root_id].title = filename
        ast.nodes[ast.root_id].text = f"Document Root: {filename}"
        for chunk in chunks:
            chunk["filename"] = filename
    else:
        doc_id = f"doc_{key[:32]}"
        if is_pdf:
            pages = measure("extraction_ms", extract_pages_from_pdf, contents, filename)
        else:
            pages = measure("extraction_ms", extract_pages_from_text, contents.decode("utf-8", errors="replace"), filename)
        if not pages or not any(p["text"].strip() for p in pages):
            raise ValueError("Could not extract readable text from document.")
        page_images = measure("previews_ms", render_page_previews, contents, doc_id) if is_pdf else {}
        ast = measure("document_tree_ms", build_document_ast, doc_id, pages,
                      pdf_bytes=contents if is_pdf else None, doc_title=filename, lazy_visuals=True)
        chunks, classification = measure("chunking_ms", hierarchical_chunk, pages, page_images, ast=ast)
        vectors = measure("child_embeddings_ms", embed_texts, [c["child_text"] for c in chunks])
        page_count = len(pages)
        try:
            measure("cache_write_ms", _save_cache, key, ast, chunks, classification, vectors, page_count, is_pdf)
        except Exception:
            cache_warning = "Search is ready, but the reuse cache could not be saved."
            logger.warning("Could not save ingestion cache %s", key, exc_info=True)
    retriever = measure("retriever_ms", HybridRetriever, chunks, vectors)
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    logger.info("Ingestion doc=%s cache_hit=%s ready_ms=%s stages=%s", ast.doc_id, bool(cached), elapsed, timings)
    return {
        "doc_id": ast.doc_id, "filename": filename, "pages": page_count,
        "total_chunks": len(chunks), "total_ast_nodes": len(ast.nodes),
        "cross_modal_links": len(ast.get_cross_modal_links()),
        "indexing_duration_ms": elapsed, "stage_timings_ms": timings,
        "cache_hit": bool(cached), "cache_warning": cache_warning,
        "classification": classification, "retriever": retriever, "ast": ast, "chunks": chunks,
    }


class BackgroundIndexer:
    """One worker; jobs never control the active document or chat readiness."""

    def __init__(self, store):
        self.store = store
        self._lock = Lock()
        self._executor = None
        self._jobs = {}

    def submit(self, ast):
        with self._lock:
            previous = self._jobs.get(ast.doc_id)
            if previous and previous["status"] in ("queued", "indexing", "ready"):
                return previous
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lance-index")
            state = {"status": "queued", "error": None, "stage_timings_ms": {}}
            self._jobs[ast.doc_id] = state
            self._executor.submit(self._run, ast, state)
            return state

    def _run(self, ast, state):
        started = time.perf_counter()
        with self._lock:
            state["status"] = "indexing"
        result = {"stage_timings_ms": {}}
        try:
            # Content/model/version are encoded in doc_id, so a prior table is reusable.
            if self.store.has_document(ast.doc_id):
                result["reused"] = True
            else:
                self.store.index_ast(ast, timings=result["stage_timings_ms"])
                result["reused"] = False
            result["status"] = "ready"
        except Exception:
            result["error"] = "Optional vector storage failed; chat search remains available. Reattach to retry."
            result["status"] = "failed"
            logger.exception("Background LanceDB indexing failed for %s", ast.doc_id)
        finally:
            result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            with self._lock:
                state.update(result)

    def snapshot(self, state):
        with self._lock:
            return deepcopy(state)

    def shutdown(self):
        with self._lock:
            executor, self._executor = self._executor, None
        if executor:
            executor.shutdown(wait=True)
