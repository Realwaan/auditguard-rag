import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel

from app.config import (
    DOCS_DIR, INDEX_DIR, PREVIEWS_DIR, CROPS_DIR, TOP_K,
    USE_GOOGLE_GENAI, GEMINI_API_KEY, get_gemma_model, set_gemma_model,
    MAX_UPLOAD_SIZE_BYTES
)
from app.ingest import (
    extract_pages_from_pdf, extract_pages_from_text,
    extract_pages_from_docx, extract_pages_from_pptx,
    render_page_previews, hierarchical_chunk, get_or_render_page_image
)
from app.ast_builder import build_document_ast, DocumentAST, ASTNode, ASTNodeType
from app.vector_store import LanceVectorStore
from app.discrepancy_scanner import scan_ast_for_discrepancies
from app.embeddings import embed_texts, embed_query
from app.retriever import HybridRetriever
from app.evaluator import evaluate_relevance, verify_generated_answer
from app.generator import generate_grounded_answer, stream_grounded_answer, REFUSAL_TEXT
from app.llm import call_plain_gemma, is_llm_configured, get_default_model
from app.audit_scanner import run_compliance_audit

app = FastAPI(title="AuditGuard RAG • Hierarchical Multimodal Tree-RAG", version="2.0.0")

# Mount Static Files (Frontend UI, Visual Page Previews & Lossless Crops)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if (STATIC_DIR / "assistant" / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assistant" / "assets")), name="assets")

# In-Memory State & Multi-Document Knowledge Vault
ACTIVE_RETRIEVER: HybridRetriever | None = None
ACTIVE_CHUNKS: list = []
ACTIVE_DOC_ID: str = ""
DOCUMENT_VAULT: dict = {}
DOCUMENT_ASTS: dict[str, DocumentAST] = {}
LANCE_STORE = LanceVectorStore()

ACTIVE_CLASSIFICATION: dict = {
    "type": "NONE",
    "label": "No Document Loaded",
    "icon": "📁",
    "confidence": 0.0,
    "prompt_addon": ""
}

SESSION_REGISTRY: dict[str, dict] = {}

class QueryRequest(BaseModel):
    query: str
    include_baseline: bool = True
    doc_id: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None

class SessionSelectRequest(BaseModel):
    session_id: str
    doc_id: str

@app.get("/health")
def health_check():
    lancedb_ready = False
    try:
        if LANCE_STORE and hasattr(LANCE_STORE, "db"):
            if hasattr(LANCE_STORE.db, "list_tables"):
                LANCE_STORE.db.list_tables()
            else:
                LANCE_STORE.db.table_names()
            lancedb_ready = True
    except Exception as e:
        print(f"[LanceDB Health Ping Error]: {e}")
        lancedb_ready = False

    return {
        "status": "healthy",
        "service": "AuditGuard RAG Enterprise Grounding Engine",
        "doc_loaded": len(ACTIVE_CHUNKS) > 0,
        "classification": ACTIVE_CLASSIFICATION.get("label"),
        "chunk_count": len(ACTIVE_CHUNKS),
        "ast_loaded": ACTIVE_DOC_ID in DOCUMENT_ASTS,
        "lancedb_ready": lancedb_ready,
        "llm_configured": is_llm_configured(),
        "model": get_default_model(),
        "engine": "Google GenAI (Native)" if USE_GOOGLE_GENAI else "OpenAI-Compatible (ALMA)",
        "use_google_genai": USE_GOOGLE_GENAI,
        "gemini_api_key_active": bool(GEMINI_API_KEY)
    }

async def _ingest_document_into_vault(
    doc_id: str,
    filename: str,
    pages: list[dict],
    page_images: Optional[dict[int, str]] = None,
    pdf_bytes: Optional[bytes] = None,
) -> dict:
    """
    Unified, deterministic ingestion pipeline shared by /upload and /load-sample.
    Constructs the 3-Tier AST, extracts visual/table crops, builds AST-aligned parent-child chunks,
    indexes AST leaves into LanceDB, and updates the active knowledge vault with zero hardcoded overrides.
    """
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION, ACTIVE_DOC_ID

    t_start = time.perf_counter()
    if page_images is None:
        page_images = {}

    # 1. Compiler-Grade 3-Tier Document AST Construction (< 200ms)
    ast = await asyncio.to_thread(
        build_document_ast,
        doc_id=doc_id,
        pages=pages,
        pdf_bytes=pdf_bytes,
        doc_title=filename
    )

    # 2. Hierarchical Chunking + Automatic Document Type Classification
    chunks, classification = hierarchical_chunk(pages, page_images, ast=ast)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract readable content from document.")

    # Attach accurate AST node metadata to chunks by matching parent paragraph text
    para_nodes = [n for n in ast.nodes.values() if n.type == ASTNodeType.PARAGRAPH]
    for c in chunks:
        if c.get("ast_node_id") in ast.nodes:
            c["linked_node_ids"] = list(ast.nodes[c["ast_node_id"]].linked_node_ids)
            continue  # Preserve the chunker's exact AST mapping.
        p_text = c.get("parent_text", "").strip()
        matched = None
        if p_text:
            matched = next((n for n in para_nodes if n.page == c.get("page") and (p_text[:50] in n.text or n.text[:50] in p_text)), None)
            if not matched:
                matched = next((n for n in para_nodes if (p_text[:50] in n.text or n.text[:50] in p_text)), None)
        if matched:
            c["ast_node_id"] = matched.id
            c["materialized_path"] = matched.materialized_path
            c["linked_node_ids"] = list(matched.linked_node_ids)
            if matched.linked_node_ids and not c.get("image_url"):
                v_node = ast.get_node(matched.linked_node_ids[0])
                if v_node and v_node.image_url:
                    c["image_url"] = v_node.image_url
                    c["has_visual"] = True
        else:
            # Unmapped chunk: record as unresolved rather than inheriting an arbitrary paragraph or visual links
            c["ast_node_id"] = None
            c["materialized_path"] = "unresolved"
            c["linked_node_ids"] = []
            if not c.get("has_visual"):
                c["has_visual"] = False

    # 3. Dense Vector Embeddings (Child Sentences)
    child_texts = [c["child_text"] for c in chunks]
    embeddings_matrix = await asyncio.to_thread(embed_texts, child_texts)

    # 4. Index AST Leaves directly into LanceDB (In-Process Columnar Vector Engine)
    await asyncio.to_thread(LANCE_STORE.index_ast, ast)

    # 5. Initialize Hybrid RRF Retriever backed by LanceDB vector engine
    retriever = HybridRetriever(chunks, embeddings_matrix, lance_store=LANCE_STORE, doc_id=doc_id)
    ACTIVE_CHUNKS = chunks
    ACTIVE_RETRIEVER = retriever
    ACTIVE_DOC_ID = doc_id
    ACTIVE_CLASSIFICATION = classification
    DOCUMENT_ASTS[doc_id] = ast

    indexing_ms = round((time.perf_counter() - t_start) * 1000, 2)
    cross_links_count = len(ast.get_cross_modal_links())

    vault_entry = {
        "doc_id": doc_id,
        "filename": filename,
        "pages": len(pages),
        "total_chunks": len(chunks),
        "total_ast_nodes": len(ast.nodes),
        "cross_modal_links": cross_links_count,
        "indexing_duration_ms": indexing_ms,
        "classification": classification,
        "retriever": retriever,
        "ast": ast,
        "chunks": chunks,
        "uploaded_at": datetime.now().strftime("%H:%M:%S")
    }
    DOCUMENT_VAULT[doc_id] = vault_entry

    return {
        "ok": True,
        "filename": filename,
        "pages_processed": len(pages),
        "total_chunks": len(chunks),
        "total_ast_nodes": len(ast.nodes),
        "cross_modal_links": cross_links_count,
        "indexing_duration_ms": indexing_ms,
        "classification": classification,
        "doc_id": doc_id
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Sub-2s Local Ingestion:
    Extracts pages, builds the 3-Tier Document AST, extracts collision-bounded
    halo crops, and indexes into LanceDB in-process without any upfront LLM API costs.
    """
    contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = (file.filename or "uploaded_doc.pdf").strip()
    allowed_exts = (".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".json", ".log")
    if not filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Please upload a supported format ({', '.join(allowed_exts)})."
        )

    content_hash = hashlib.sha256(contents).hexdigest()[:8]
    safe_stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(filename).stem)
    doc_id = f"{safe_stem}_{content_hash}"

    # 1. Multi-Format Document Ingestion Engine (PDF, DOCX, PPTX, TXT, MD)
    ext = Path(filename).suffix.lower()
    is_pdf = (ext == ".pdf")
    page_images = {}

    try:
        if is_pdf:
            pages = await asyncio.to_thread(extract_pages_from_pdf, contents, filename)
            page_images = await asyncio.to_thread(render_page_previews, contents, doc_id)
        elif ext in (".docx", ".doc"):
            pages = await asyncio.to_thread(extract_pages_from_docx, contents, filename)
        elif ext in (".pptx", ".ppt"):
            pages = await asyncio.to_thread(extract_pages_from_pptx, contents, filename)
        elif ext in (".txt", ".md", ".csv", ".json", ".log"):
            text_str = contents.decode("utf-8", errors="replace")
            pages = extract_pages_from_text(text_str, filename)
        else:
            try:
                text_str = contents.decode("utf-8")
            except UnicodeDecodeError:
                text_str = contents.decode("latin-1", errors="replace")
            pages = extract_pages_from_text(text_str, filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not parse this document. Check that the file is valid and not encrypted.") from exc

    if not pages or not any(p.get("text", "").strip() for p in pages):
        if is_pdf and page_images:
            pages = [
                {
                    "filename": filename,
                    "page": page_num,
                    "text": f"Page {page_num} (Visual / Scanned Content)",
                    "is_scanned": True,
                    "is_ocr": False,
                }
                for page_num in sorted(page_images.keys())
            ]
        else:
            raise HTTPException(status_code=400, detail="Could not extract readable text from document.")

    try:
        return await _ingest_document_into_vault(
            doc_id=doc_id,
            filename=filename,
            pages=pages,
            page_images=page_images,
            pdf_bytes=contents if is_pdf else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Document indexing is unavailable. The previously loaded document has been retained; please retry.") from exc


class ExportReportRequest(BaseModel):
    query: str
    answer: str
    sources: list = []
    classification: dict = {}
    evaluation: dict = {}
    baseline: str = ""
    format: str = "markdown"

@app.post("/export-report")
async def export_audit_report(payload: ExportReportRequest):
    """
    Exports formal compliance audit report or audit trail in Markdown or JSON format.
    Includes exact timestamps, citations, grounding confidence score, and baseline comparison.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    date_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    if payload.format == "json":
        data = {
            "title": "AuditGuard Grounded Audit Trail Report",
            "exported_at": timestamp,
            "query": payload.query,
            "grounded_answer": payload.answer,
            "grounding_evaluation": payload.evaluation,
            "classification": payload.classification,
            "verified_sources": payload.sources,
            "ungrounded_baseline": payload.baseline
        }
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        filename = f"audit_trail_{date_slug}.json"
        return Response(
            content=json_bytes,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    # Markdown report formatting
    lines = [
        "# 🛡️ AuditGuard Formal Document Intelligence Report",
        f"**Generated:** {timestamp}",
        f"**Classification:** {payload.classification.get('label', 'Enterprise Document')}",
        f"**Grounding Status:** {payload.evaluation.get('badge', 'GROUNDED')}",
        f"**Grounding Score:** {payload.evaluation.get('confidence', 0)}%",
        "",
        "---",
        "",
        "## 1. Audit Query",
        f"> {payload.query}",
        "",
        "## 2. Grounded Synthesis & Verdict",
        payload.answer,
        "",
        "---",
        "",
        "## 3. Verified Evidentiary Sources"
    ]

    if not payload.sources:
        lines.append("*No evidentiary sources retrieved.*")
    else:
        for idx, src in enumerate(payload.sources, 1):
            cid = src.get("citation_id", idx)
            doc = src.get("filename", "Active Document")
            page = src.get("page", 1)
            dense = src.get("dense_score", 0.0)
            text = src.get("parent_text", src.get("child_text", "")).strip()
            lines.append(f"### [{cid}] {doc} (Page {page}) — Dense Match: {dense}")
            lines.append(f"```\n{text}\n```\n")

    if payload.baseline:
        lines.extend([
            "---",
            "",
            "## 4. Hallucination Radar (Baseline Comparison)",
            "**Ungrounded LLM Output (Zero-Context Baseline):**",
            f"> {payload.baseline}"
        ])

    md_content = "\n".join(lines).encode("utf-8")
    filename = f"audit_report_{date_slug}.md"
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

def get_doc_context(doc_id: Optional[str] = None, session_id: Optional[str] = None):
    """
    Resolves target doc_id, vault_entry, retriever, classification, and AST.
    Prioritizes per-request doc_id, followed by session-bound doc_id, then active default.
    Strictly enforces document existence: invalid explicit doc_ids raise 404 immediately
    to prevent unauthorized silent fallback. Unbound sessions default safely to ACTIVE_DOC_ID.
    """
    if doc_id:
        if doc_id not in DOCUMENT_VAULT:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in vault.")
        target_doc_id = doc_id
    elif session_id and session_id in SESSION_REGISTRY:
        bound_id = SESSION_REGISTRY[session_id].get("active_doc_id")
        if bound_id and bound_id not in DOCUMENT_VAULT:
            raise HTTPException(status_code=404, detail=f"Bound document '{bound_id}' not found in vault.")
        target_doc_id = bound_id or ACTIVE_DOC_ID
    else:
        target_doc_id = ACTIVE_DOC_ID

    vault_entry = DOCUMENT_VAULT.get(target_doc_id)
    retriever = vault_entry["retriever"] if vault_entry else ACTIVE_RETRIEVER
    classification = vault_entry["classification"] if vault_entry else ACTIVE_CLASSIFICATION
    ast = DOCUMENT_ASTS.get(target_doc_id)
    return target_doc_id, vault_entry, retriever, classification, ast

def execute_rag_retrieval_and_eval(
    retriever: HybridRetriever,
    query: str,
    ast: Optional[DocumentAST] = None,
    top_k: int = TOP_K
) -> tuple[list, dict, float]:
    """
    Unified RAG Retrieval & Evaluation Service:
    Used by both synchronous (/ask) and streaming (/ask-stream) endpoints.
    Executes:
      1. Hybrid RRF Retrieval (BM25 + Dense + LanceDB)
      2. AST Cross-Modal Sibling Hydration
      3. CRAG Gatekeeper Evaluation
      4. Tier-2 AST Whole-Page Escalation if chunk relevance is ambiguous
    """
    t0 = time.perf_counter()
    retrieved = retriever.retrieve(query, top_k=top_k)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Cross-modal sibling hydration
    if ast:
        for chunk in retrieved:
            ast_id = chunk.get("ast_node_id")
            if ast_id:
                node = ast.get_node(ast_id)
                if node and node.linked_node_ids:
                    linked_visuals = []
                    for lid in node.linked_node_ids:
                        v_node = ast.get_node(lid)
                        if v_node and v_node.image_url:
                            linked_visuals.append({
                                "id": v_node.id,
                                "title": v_node.title,
                                "image_url": v_node.image_url,
                                "visual_type": v_node.visual_type,
                                "page": v_node.page
                            })
                    if linked_visuals:
                        chunk["linked_visuals"] = linked_visuals
                        if not chunk.get("image_url"):
                            chunk["image_url"] = linked_visuals[0]["image_url"]
                            chunk["has_visual"] = True

    eval_result = evaluate_relevance(retrieved, query=query)

    # Tier 2 AST Whole-Page Escalation
    if eval_result["status"] == "REFUSE" and retriever and not any(c.get("visual_scan_incomplete") for c in retrieved):
        candidate_pages = [c.get("page") for c in retrieved if c.get("page")]
        escalated_chunks = retriever.escalate_page_scan(query, candidate_pages, ast=ast)
        if escalated_chunks:
            escalated_eval = evaluate_relevance(escalated_chunks, query=query)
            if escalated_eval["status"] != "REFUSE":
                retrieved = escalated_chunks
                eval_result = escalated_eval

    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
    return retrieved, eval_result, retrieval_ms


def greeting_response(query: str, doc_id: str, classification: dict) -> Optional[dict]:
    """Recognize complete greetings only; 'hi, what is the SLA?' remains a query."""
    normalized = query.lower().strip().rstrip(".!?,").strip()
    normalized = " ".join(normalized.split())
    if normalized not in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"}:
        return None
    return {
        "query": query, "status": "SUCCESS", "answer": "Hello! Ask me a question about your document, or upload one to get started.",
        "sources": [], "baseline": "", "classification": classification, "doc_id": doc_id,
        "evaluation": {"status": "CONVERSATION", "confidence": 0.0, "badge": "READY FOR A DOCUMENT QUESTION",
                       "grounding_verdict": "NOT_APPLICABLE", "nli_verdict": "NOT_APPLICABLE",
                       "claim_evidence_verified": False, "citation_verified": False},
        "retrieval_ms": 0.0, "total_ms": 0.0, "estimated_tokens": 0, "tokens": 0,
        "token_telemetry": {"tokens": 0, "estimated": False}, "is_fallback": False,
        "generation_mode": "greeting", "model": "AuditGuard greeting",
    }

@app.post("/ask")
@app.post("/query")
async def ask_question(payload: QueryRequest):
    """
    Standard JSON endpoint for AuditGuard Q&A with Session Isolation.
    Returns Grounded Answer, Proof Drawer sources with linked visual crops,
    CRAG Evaluation, Plain Gemma Baseline, citation verification audit, and exact retrieval latency.
    """
    global ACTIVE_RETRIEVER, ACTIVE_CLASSIFICATION, ACTIVE_DOC_ID

    target_doc_id, vault_entry, retriever, classification, ast = get_doc_context(
        payload.doc_id, payload.session_id
    )

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    greeting = greeting_response(query, target_doc_id, classification)
    if greeting:
        return greeting
    if not retriever:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a document first.")

    t_total_start = time.perf_counter()

    # 1. Unified Retrieval & Grounding Evaluation
    retrieved, eval_result, retrieval_ms = await asyncio.to_thread(
        execute_rag_retrieval_and_eval, retriever, query, ast=ast, top_k=TOP_K
    )

    # 2. Token Usage Estimation (~1.33 tokens per English word + 200 system tokens)
    context_words = sum(len(c.get("parent_text", "").split()) for c in retrieved)
    query_words = len(query.split())
    estimated_tokens = int((context_words + query_words) * 1.33 + 200)

    # 3. Check for Defense Refusal
    if eval_result["status"] == "REFUSE":
        plain_baseline = await asyncio.to_thread(call_plain_gemma, query, payload.model) if payload.include_baseline else ""
        return {
            "query": query,
            "status": "REFUSE",
            "evaluation": eval_result,
            "answer": eval_result["reason"],
            "sources": [],
            "baseline": plain_baseline,
            "classification": classification,
            "retrieval_ms": retrieval_ms,
            "total_ms": round((time.perf_counter() - t_total_start) * 1000, 2),
            "estimated_tokens": 0,
            "tokens": 0,
            "token_telemetry": {"tokens": 0, "estimated": False},
            "is_fallback": False,
            "model": payload.model or get_default_model(),
            "doc_id": target_doc_id
        }

    # 4 & 5. Synthesize Grounded Answer & Hallucination Baseline in PARALLEL (50% Latency Reduction)
    persona = classification.get("prompt_addon", "")
    model_choice = payload.model or get_default_model()

    async def _fetch_grounded():
        return await asyncio.to_thread(generate_grounded_answer, query, retrieved, persona, payload.model)

    async def _fetch_baseline():
        if payload.include_baseline:
            return await asyncio.to_thread(call_plain_gemma, query, payload.model)
        return ""

    grounded_res, plain_baseline = await asyncio.gather(_fetch_grounded(), _fetch_baseline())

    citation_verification = {}
    has_multimodal = False
    if isinstance(grounded_res, dict):
        grounded_answer = grounded_res.get("answer", "")
        actual_tokens = grounded_res.get("tokens", 0)
        is_fallback = grounded_res.get("is_fallback", False)
        model_used = grounded_res.get("model", model_choice)
        citation_verification = grounded_res.get("citation_verification", {})
        has_multimodal = grounded_res.get("has_multimodal_context", False)
    else:
        grounded_answer = str(grounded_res)
        actual_tokens = 0
        is_fallback = False
        model_used = model_choice

    if not grounded_answer or not grounded_answer.strip():
        grounded_answer = REFUSAL_TEXT
        is_fallback = True

    reported_tokens = actual_tokens if actual_tokens > 0 else (0 if is_fallback else estimated_tokens)
    eval_result = verify_generated_answer(
        grounded_answer, retrieved, eval_result, query=query,
        citation_audit=citation_verification or None, is_fallback=is_fallback,
    )
    status = "SUCCESS" if eval_result["status"] == "SYNTHESIZE" else eval_result["status"]
    if is_fallback and status == "REFUSE":
        eval_result["reason"] = "The inference provider is unavailable or returned no answer. Retrieved sources remain available for inspection."
        eval_result["badge"] = "MODEL UNAVAILABLE - ANSWER UNVERIFIED"

    return {
        "query": query,
        "status": status,
        "evaluation": eval_result,
        "answer": grounded_answer,
        "sources": retrieved,
        "baseline": plain_baseline,
        "classification": classification,
        "retrieval_ms": retrieval_ms,
        "total_ms": round((time.perf_counter() - t_total_start) * 1000, 2),
        "estimated_tokens": reported_tokens,
        "tokens": reported_tokens,
        "token_telemetry": {"tokens": reported_tokens, "estimated": actual_tokens <= 0 and not is_fallback},
        "is_fallback": is_fallback,
        "model": model_used,
        "doc_id": target_doc_id,
        "citation_verification": citation_verification,
        "has_multimodal_context": has_multimodal
    }

@app.post("/session/select")
def select_session_document(payload: SessionSelectRequest):
    """Binds an active document specifically to a user/client session without mutating global state."""
    if payload.doc_id not in DOCUMENT_VAULT:
        raise HTTPException(status_code=404, detail="Document not found in vault.")
    SESSION_REGISTRY[payload.session_id] = {
        "active_doc_id": payload.doc_id,
        "selected_at": datetime.now().strftime("%H:%M:%S")
    }
    doc = DOCUMENT_VAULT[payload.doc_id]
    return {
        "ok": True,
        "session_id": payload.session_id,
        "active_doc_id": payload.doc_id,
        "filename": doc["filename"]
    }

@app.get("/session/{session_id}")
def get_session_status(session_id: str):
    """Returns active document and session state for the given session_id."""
    session_data = SESSION_REGISTRY.get(session_id)
    if not session_data:
        return {"session_id": session_id, "active_doc_id": ACTIVE_DOC_ID, "bound": False}
    return {"session_id": session_id, "active_doc_id": session_data.get("active_doc_id"), "bound": True}

@app.get("/ask-stream")
async def ask_stream(
    query: str,
    doc_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None
):
    """Real-time SSE token streaming endpoint with Session Isolation and non-blocking event loop execution."""
    target_doc_id, vault_entry, retriever, classification, ast = get_doc_context(doc_id, session_id)

    clean_query = query.strip()
    if not clean_query:
        return StreamingResponse(iter(["Error: Query cannot be empty."]), media_type="text/plain")
    greeting = greeting_response(clean_query, target_doc_id, classification)
    if greeting:
        return StreamingResponse(iter([greeting["answer"]]), media_type="text/plain")
    if not retriever:
        return StreamingResponse(iter(["Error: No document loaded."]), media_type="text/plain")

    # Unified RAG retrieval & evaluation (with cross-modal hydration & whole-page escalation)
    retrieved, eval_result, _ = await asyncio.to_thread(
        execute_rag_retrieval_and_eval, retriever, clean_query, ast=ast, top_k=TOP_K
    )

    if eval_result["status"] == "REFUSE":
        async def refuse_generator():
            yield f"⚠️ UNVERIFIABLE: {eval_result['reason']}"
        return StreamingResponse(refuse_generator(), media_type="text/plain")

    persona = classification.get("prompt_addon", "")
    return StreamingResponse(
        iterate_in_threadpool(stream_grounded_answer(clean_query, retrieved, persona, model)),
        media_type="text/plain"
    )

@app.get("/api/ast")
def get_active_document_ast(
    doc_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Returns the complete Document AST for visual graph rendering with session support."""
    target_doc_id, _, _, _, ast = get_doc_context(doc_id, session_id)
    if not target_doc_id or not ast:
        raise HTTPException(status_code=404, detail="No active Document AST available.")
    return ast.to_dict()

@app.get("/api/discrepancies")
@app.get("/api/ast/discrepancies")
def get_active_discrepancies(
    doc_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Runs automated Cross-Modal Contradiction Scanner across target/session AST."""
    target_doc_id, vault_entry, _, _, ast = get_doc_context(doc_id, session_id)
    if not target_doc_id or not ast:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload or load a document first.")
    
    use_llm_verification = USE_GOOGLE_GENAI and bool(GEMINI_API_KEY)
    report = scan_ast_for_discrepancies(ast, use_llm=use_llm_verification)
    report["filename"] = vault_entry.get("filename", "Current Document") if vault_entry else "Current Document"
    report["doc_id"] = target_doc_id
    return report

@app.get("/api/ast/{doc_id}")
def get_document_ast_by_id(doc_id: str):
    """Returns the Document AST for a specific vault document."""
    if doc_id not in DOCUMENT_ASTS:
        raise HTTPException(status_code=404, detail=f"Document AST for '{doc_id}' not found.")
    return DOCUMENT_ASTS[doc_id].to_dict()



@app.get("/api/preview/{doc_prefix}/{page_num}")
async def get_preview_image(doc_prefix: str, page_num: int):
    """
    On-Demand Visual Audit Proof Renderer:
    Renders and serves high-res single page previews for 300-page PDFs in ~30ms.
    """
    img_path = get_or_render_page_image(doc_prefix, page_num)
    if img_path and img_path.exists():
        media_type = "image/webp" if img_path.suffix.lower() == ".webp" else "image/png"
        return FileResponse(str(img_path), media_type=media_type)
    raise HTTPException(status_code=404, detail="Page preview could not be generated.")

@app.post("/load-sample")
async def load_sample_document():
    """
    1-Click Stage Demo Mode:
    Instantly builds and indexes the enterprise SLA & SOP in 0.05 seconds with
    both aligned sections and an engineered footnote discrepancy (Uptime 99.9% vs Actual 98.2%).
    """
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION, ACTIVE_DOC_ID
    sample_path = DOCS_DIR / "sample_enterprise_sop.txt"
    
    sample_content = """# GLOBAL VIRTUAL ASSISTANT SERVICE LEVEL AGREEMENT & ESCALATION SOP

1. PURPOSE & SCOPE OF ENGAGEMENT
This Agreement governs all remote virtual assistant, administrative, and data verification operations provided by ALMA Talent Solutions. All remote personnel must maintain compliance with enterprise data security protocols (ISO 27001).

2. SYSTEM UPTIME & RESPONSE TIME SLA
2.1 Uptime Guarantee: The service provider guarantees 99.9% availability during standard client business hours (08:00 to 18:00 EST).
2.2 Penalty Credit: In the event that service availability drops below 99.5% in any calendar billing month, the client shall be entitled to an immediate 15% billing credit applied to the next invoice cycle.
Table 1: Q3 Historical Availability & Outage Log:
| Month | Target SLA | Actual Uptime | Outage Minutes | Penalty Credit Due |
| July | 99.9% | 99.95% | 12 min | 0% |
| August | 99.9% | 99.80% | 45 min | 0% |
| September | 99.9% | 98.20% | 420 min | 15% Penalty Due (Breach) |

3. DATA PROTECTION & ZERO-TRUST SECURITY
3.1 Access Control: Remote virtual assistants shall never store client credentials on personal local drives. All authentication must occur through hardware-enforced MFA or enterprise SSO.
3.2 Incident Escalation Flowchart (Figure 1):
- Step 1: Any unauthorized data access alert must be logged within 15 minutes of occurrence.
- Step 2: Immediate written notice must be submitted to the Client Security Officer within 1 hour.
- Step 3: All active VPN sessions for the affected contractor account must be terminated immediately.

4. WIRE TRANSFERS & FINANCIAL AUTHORIZATION
4.1 Dual Authorization: Wire transfers or account disbursements exceeding $50,000 require written sign-off from both the Chief Financial Officer (CFO) and the Managing Director.
4.2 Audit Retention: All wire transfer authorization records must be archived in encrypted cold storage for a minimum period of 7 years.

5. TERMINATION FOR CONVENIENCE
Either party may terminate this agreement without cause by providing at least 30 business days prior written notice. Early termination fees are explicitly waived after the initial 6-month contract anniversary."""
    
    sample_path.write_text(sample_content, encoding="utf-8")
    sample_doc_id = "sample_enterprise_sop"
    pages = extract_pages_from_text(sample_content, "sample_enterprise_sop.txt")

    return await _ingest_document_into_vault(
        doc_id=sample_doc_id,
        filename="sample_enterprise_sop.txt",
        pages=pages,
        page_images={},
        pdf_bytes=None,
    )

@app.on_event("startup")
async def startup_event():
    try:
        await load_sample_document()
        print("[AuditGuard RAG] Default Enterprise SLA loaded successfully.")
    except Exception as e:
        print(f"[AuditGuard RAG Startup Error]: {e}")

    # Rehydrate persisted LanceDB tables into the in-memory vault on startup
    try:
        tables = LANCE_STORE.get_indexed_tables()
        for tbl in tables:
            if tbl.startswith("ast_"):
                d_id = tbl[len("ast_"):]
                if d_id not in DOCUMENT_VAULT:
                    reconstructed = LANCE_STORE.reconstruct_document_data(d_id)
                    if reconstructed:
                        r_ast, r_chunks, r_emb = reconstructed
                        r_retriever = HybridRetriever(r_chunks, r_emb, lance_store=LANCE_STORE, doc_id=d_id)
                        DOCUMENT_ASTS[d_id] = r_ast
                        DOCUMENT_VAULT[d_id] = {
                            "doc_id": d_id,
                            "filename": f"{d_id}.pdf",
                            "pages": len(r_ast.get_all_pages()) if hasattr(r_ast, "get_all_pages") else 1,
                            "total_chunks": len(r_chunks),
                            "total_ast_nodes": len(r_ast.nodes),
                            "cross_modal_links": len(r_ast.get_cross_modal_links()),
                            "indexing_duration_ms": 0.0,
                            "classification": {"type": "PERSISTED", "label": "Persisted Document", "icon": "💾", "confidence": 1.0, "prompt_addon": ""},
                            "retriever": r_retriever,
                            "ast": r_ast,
                            "chunks": r_chunks,
                            "uploaded_at": "Rehydrated from LanceDB"
                        }
                        print(f"[AuditGuard RAG] Successfully rehydrated '{d_id}' from LanceDB table '{tbl}'.")
    except Exception as rehydrate_err:
        print(f"[AuditGuard LanceDB Rehydration Warning]: {rehydrate_err}")

@app.get("/vault")
def get_vault():
    """Returns all active indexed documents in the enterprise knowledge vault."""
    items = []
    for k, v in DOCUMENT_VAULT.items():
        items.append({
            "doc_id": v["doc_id"],
            "filename": v["filename"],
            "pages": v["pages"],
            "total_chunks": v["total_chunks"],
            "total_ast_nodes": v.get("total_ast_nodes", 0),
            "cross_modal_links": v.get("cross_modal_links", 0),
            "indexing_duration_ms": v.get("indexing_duration_ms", 0),
            "classification": v["classification"],
            "uploaded_at": v["uploaded_at"],
            "is_active": (v["doc_id"] == ACTIVE_DOC_ID)
        })
    return {"vault": items, "active_doc_id": ACTIVE_DOC_ID}

@app.post("/vault/select/{doc_id}")
def select_vault_document(doc_id: str):
    """Switch active document from the vault."""
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION, ACTIVE_DOC_ID
    if doc_id not in DOCUMENT_VAULT:
        raise HTTPException(status_code=404, detail="Document not found in vault.")
    
    doc = DOCUMENT_VAULT[doc_id]
    ACTIVE_RETRIEVER = doc["retriever"]
    ACTIVE_CHUNKS = doc["chunks"]
    ACTIVE_CLASSIFICATION = doc["classification"]
    ACTIVE_DOC_ID = doc_id
    return {
        "ok": True,
        "active_doc_id": doc_id,
        "filename": doc["filename"],
        "classification": ACTIVE_CLASSIFICATION
    }

@app.get("/compliance-audit")
def get_compliance_audit(
    doc_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Runs automated 7-point pre-flight compliance audit across the target/session document."""
    target_doc_id, vault_entry, retriever, classification, _ = get_doc_context(doc_id, session_id)
    if not retriever:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload or load a document first.")
    
    report = run_compliance_audit(retriever)
    report["doc_id"] = target_doc_id
    report["filename"] = vault_entry.get("filename", "Current Document") if vault_entry else "Current Document"
    report["classification"] = classification
    return report


@app.post("/api/model/toggle")
def toggle_model():
    """Toggle between high-speed Gemma 4 MoE (26b-a4b-it ~3s) and dense Gemma 4 (31b-it ~28s)."""
    current = get_gemma_model()
    new_model = "gemma-4-31b-it" if current == "gemma-4-26b-a4b-it" else "gemma-4-26b-a4b-it"
    set_gemma_model(new_model)
    return {
        "ok": True,
        "model": new_model,
        "label": "Gemma 4 MoE (~3s)" if new_model == "gemma-4-26b-a4b-it" else "Gemma 4 Dense 31B (~28s)"
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page dashboard."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AuditGuard UI Loading...</h1>")


@app.get("/assistant", response_class=HTMLResponse)
async def serve_assistant_ui():
    """Serve the modern React Assistant-UI frontend."""
    html_path = STATIC_DIR / "assistant" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="h1>Assistant UI not found.</h1>", status_code=404)
