import asyncio
import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel

import app.config as cfg
from app.config import (
    DOCS_DIR, TOP_K, USE_GOOGLE_GENAI, MAX_UPLOAD_SIZE_BYTES,
    get_active_gemma_model, set_active_gemma_model
)
from app.ingest import get_or_render_page_image
from app.ast_builder import DocumentAST, ASTNodeType, get_or_render_visual_crop
from app.vector_store import LanceVectorStore
from app.discrepancy_scanner import scan_ast_for_discrepancies
from app.embeddings import get_embedding_model
from app.indexing import BackgroundIndexer, prepare_document, load_cached_ast
from app.retriever import HybridRetriever
from app.evaluator import evaluate_relevance, verify_generated_answer
from app.generator import (
    generate_grounded_answer_with_telemetry,
    stream_grounded_answer
)
from app.llm import call_plain_gemma_with_telemetry, is_llm_configured
from app.audit_scanner import run_compliance_audit
from app.exporter import build_pdf_bytes, build_docx_bytes, build_csv_string

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Warm up before accepting attachments; a transient model failure remains retryable.
    started = time.perf_counter()
    try:
        await asyncio.to_thread(get_embedding_model)
        app.state.embedding_preload_succeeded = True
    except Exception:
        app.state.embedding_preload_succeeded = False
        logger.exception("Embedding model preload failed; attachment will retry initialization")
    app.state.embedding_preload_ms = round((time.perf_counter() - started) * 1000, 2)
    try:
        yield
    finally:
        await asyncio.to_thread(BACKGROUND_INDEXER.shutdown)


app = FastAPI(title="AuditGuard RAG • Hierarchical Multimodal Tree-RAG", version="2.0.0", lifespan=lifespan)

# Mount Static Files (Frontend UI, Visual Page Previews & Lossless Crops)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if (STATIC_DIR / "assistant" / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assistant" / "assets")), name="assets")

# In-Memory State & Multi-Document Knowledge Vault (Protected by Async Lock)
STATE_LOCK = asyncio.Lock()
ACTIVE_RETRIEVER: HybridRetriever | None = None
ACTIVE_CHUNKS: list = []
ACTIVE_DOC_ID: str = ""
DOCUMENT_VAULT: dict = {}
DOCUMENT_ASTS: dict[str, DocumentAST] = {}
LANCE_STORE = LanceVectorStore()
BACKGROUND_INDEXER = BackgroundIndexer(LANCE_STORE)

ACTIVE_CLASSIFICATION: dict = {
    "type": "NONE",
    "label": "No Document Loaded",
    "icon": "📁",
    "confidence": 0.0,
    "prompt_addon": ""
}


class QueryRequest(BaseModel):
    query: str
    include_baseline: bool = False


class ExportReportRequest(BaseModel):
    query: str
    answer: str
    sources: list = []
    classification: dict = {}
    evaluation: dict = {}
    baseline: str = ""
    format: str = "markdown"


async def _attach_document(contents: bytes, filename: str, request_started: float) -> dict:
    """Publish search-ready state before the optional LanceDB embedding pass."""
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION, ACTIVE_DOC_ID
    try:
        entry = await asyncio.to_thread(prepare_document, contents, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    doc_id = entry["doc_id"]
    entry["uploaded_at"] = datetime.now().strftime("%H:%M:%S")
    entry["background_indexing"] = BACKGROUND_INDEXER.submit(entry["ast"])
    async with STATE_LOCK:
        DOCUMENT_ASTS[doc_id] = entry["ast"]
        DOCUMENT_VAULT[doc_id] = entry
        ACTIVE_RETRIEVER = entry["retriever"]
        ACTIVE_CHUNKS = entry["chunks"]
        ACTIVE_CLASSIFICATION = entry["classification"]
        ACTIVE_DOC_ID = doc_id
    entry["attachment_duration_ms"] = round((time.perf_counter() - request_started) * 1000, 2)
    return {
        "ok": True, "status": "ready", "filename": filename, "doc_id": doc_id,
        "pages_processed": entry["pages"], "total_chunks": entry["total_chunks"],
        "total_ast_nodes": entry["total_ast_nodes"], "cross_modal_links": entry["cross_modal_links"],
        "indexing_duration_ms": entry["indexing_duration_ms"],
        "attachment_duration_ms": entry["attachment_duration_ms"],
        "stage_timings_ms": entry["stage_timings_ms"], "cache_hit": entry["cache_hit"],
        "cache_warning": entry["cache_warning"], "classification": entry["classification"],
        "background_indexing": BACKGROUND_INDEXER.snapshot(entry["background_indexing"]),
    }


def _hydrate_ast_siblings(retrieved: list[dict], ast: DocumentAST | None) -> None:
    """Attaches linked visual sibling nodes from the AST onto retrieved text chunks."""
    if not ast:
        return
    for chunk in retrieved:
        ast_id = chunk.get("ast_node_id")
        if not ast_id:
            continue
        node = ast.get_node(ast_id)
        if not node or not node.linked_node_ids:
            continue
        linked_visuals = []
        for lid in node.linked_node_ids:
            v_node = ast.get_node(lid)
            if v_node and v_node.type == ASTNodeType.VISUAL:
                linked_visuals.append({
                    "id": v_node.id,
                    "title": v_node.title,
                    "image_url": v_node.image_url,
                    "visual_type": v_node.visual_type,
                    "page": v_node.page,
                })
        if linked_visuals:
            chunk["linked_visuals"] = linked_visuals
            if not chunk.get("image_url") and linked_visuals[0].get("image_url"):
                chunk["image_url"] = linked_visuals[0]["image_url"]
            chunk["has_visual"] = True


@app.get("/health")
def health_check():
    active_model = get_active_gemma_model()
    return {
        "status": "healthy",
        "embedding_preload_succeeded": getattr(app.state, "embedding_preload_succeeded", False),
        "embedding_preload_ms": getattr(app.state, "embedding_preload_ms", None),
        "doc_loaded": len(ACTIVE_CHUNKS) > 0,
        "classification": ACTIVE_CLASSIFICATION.get("label"),
        "chunk_count": len(ACTIVE_CHUNKS),
        "ast_loaded": ACTIVE_DOC_ID in DOCUMENT_ASTS,
        "lancedb_ready": LANCE_STORE.is_ready(),
        "llm_configured": is_llm_configured(),
        "model": active_model,
        "engine": "Google GenAI (Native)" if USE_GOOGLE_GENAI else "OpenAI-Compatible (ALMA)",
        "use_google_genai": USE_GOOGLE_GENAI
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Local Document Ingestion:
    Validates file size & extension, extracts pages, builds the 3-Tier Document AST,
    detects visual bounds and prepares HybridRetriever. LanceDB runs in the background.
    """
    request_started = time.perf_counter()
    filename = (file.filename or "uploaded_doc.pdf").strip()
    allowed_exts = (".pdf", ".txt", ".md")
    if not filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Please upload {', '.join(allowed_exts)}."
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    return await _attach_document(contents, filename, request_started)


@app.post("/ask")
@app.post("/query")
async def ask_question(payload: QueryRequest):
    """
    Standard JSON endpoint for AuditGuard Q&A.
    Returns Grounded Answer, Proof Drawer sources with linked visual crops,
    CRAG Evaluation, Plain Gemma Baseline, genuine token telemetry, and exact latency.
    """
    # Take an atomic snapshot of active document state
    doc_id = ACTIVE_DOC_ID
    doc_entry = DOCUMENT_VAULT.get(doc_id, {})
    retriever = doc_entry.get("retriever") or ACTIVE_RETRIEVER
    ast = doc_entry.get("ast") or DOCUMENT_ASTS.get(doc_id)
    classification = doc_entry.get("classification") or ACTIVE_CLASSIFICATION

    if not retriever:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a document first.")

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t_total_start = time.perf_counter()

    # 1. Hybrid RRF Retrieval with Microsecond Benchmarking
    t0 = time.perf_counter()
    retrieved = await asyncio.to_thread(retriever.retrieve, query, TOP_K)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 2. AST Cross-Modal Sibling Hydration
    _hydrate_ast_siblings(retrieved, ast)

    # 3. CRAG Relevance Gatekeeper (Tier 1: Chunk-Level Evaluation)
    eval_result = evaluate_relevance(retrieved, query=query)

    # Tier 2 AST Whole-Page Escalation:
    if eval_result["status"] == "REFUSE":
        candidate_pages = [c.get("page") for c in retrieved if c.get("page")]
        escalated_chunks = await asyncio.to_thread(
            retriever.escalate_page_scan, query, candidate_pages, ast
        )
        if escalated_chunks:
            escalated_eval = evaluate_relevance(escalated_chunks, query=query)
            if escalated_eval["status"] != "REFUSE":
                retrieved = escalated_chunks
                _hydrate_ast_siblings(retrieved, ast)
                eval_result = escalated_eval

    # 4. Check for Defense Refusal
    total_ast_nodes = len(ast.nodes) if ast else doc_entry.get("total_ast_nodes", 0)
    cross_modal_links = len(ast.get_cross_modal_links()) if ast else doc_entry.get("cross_modal_links", 0)
    indexing_duration_ms = doc_entry.get("indexing_duration_ms", 0.0)

    if eval_result["status"] == "REFUSE":
        plain_baseline = ""
        baseline_tokens = 0
        if payload.include_baseline:
            plain_baseline, base_meta = await asyncio.to_thread(call_plain_gemma_with_telemetry, query)
            baseline_tokens = base_meta.get("baseline_tokens", 0)

        total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)
        return {
            "query": query,
            "status": "REFUSE",
            "evaluation": eval_result,
            "answer": eval_result["reason"],
            "sources": [],
            "baseline": plain_baseline,
            "classification": classification,
            "retrieval_ms": retrieval_ms,
            "total_ms": total_ms,
            "estimated_tokens": baseline_tokens,
            "token_telemetry": {
                "grounded_prompt_tokens": 0,
                "grounded_completion_tokens": 0,
                "grounded_total_tokens": 0,
                "baseline_tokens": baseline_tokens,
                "total_tokens": baseline_tokens,
                "generation_mode": "refusal_gatekeeper",
                "model_used": get_active_gemma_model(),
                "total_ast_nodes": total_ast_nodes,
                "cross_modal_links": cross_modal_links,
                "indexing_duration_ms": indexing_duration_ms,
            },
            "generation_mode": "refusal_gatekeeper",
        }

    # 5. Synthesize Grounded Answer
    persona = classification.get("prompt_addon", "")

    async def _fetch_grounded():
        return await asyncio.to_thread(generate_grounded_answer_with_telemetry, query, retrieved, persona)

    async def _fetch_baseline():
        if payload.include_baseline:
            return await asyncio.to_thread(call_plain_gemma_with_telemetry, query)
        return "", {"baseline_tokens": 0, "baseline_mode": "disabled"}

    (grounded_answer, gen_meta), (plain_baseline, base_meta) = await asyncio.gather(
        _fetch_grounded(), _fetch_baseline()
    )

    # 6. Post-Generation Guardrail Verification (audits inline citations and LLM-level refusals)
    eval_result = verify_generated_answer(grounded_answer, retrieved, eval_result)
    final_status = "REFUSE" if eval_result["status"] == "REFUSE" else "SUCCESS"
    total_tokens = int(gen_meta.get("total_tokens", 0) + base_meta.get("baseline_tokens", 0))
    total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

    return {
        "query": query,
        "status": final_status,
        "evaluation": eval_result,
        "answer": grounded_answer,
        "sources": [] if final_status == "REFUSE" else retrieved,
        "baseline": plain_baseline,
        "classification": classification,
        "retrieval_ms": retrieval_ms,
        "total_ms": total_ms,
        "estimated_tokens": total_tokens,
        "token_telemetry": {
            "grounded_prompt_tokens": gen_meta.get("prompt_tokens", 0),
            "grounded_completion_tokens": gen_meta.get("completion_tokens", 0),
            "grounded_total_tokens": gen_meta.get("total_tokens", 0),
            "baseline_tokens": base_meta.get("baseline_tokens", 0),
            "total_tokens": total_tokens,
            "generation_mode": gen_meta.get("generation_mode", "unknown"),
            "model_used": gen_meta.get("model_used", ""),
            "total_ast_nodes": total_ast_nodes,
            "cross_modal_links": cross_modal_links,
            "indexing_duration_ms": indexing_duration_ms,
        },
        "generation_mode": gen_meta.get("generation_mode", "unknown"),
    }


@app.get("/ask-stream")
async def ask_stream(query: str):
    """Real-time token streaming endpoint with full CRAG & Tier-2 AST escalation parity."""
    doc_id = ACTIVE_DOC_ID
    doc_entry = DOCUMENT_VAULT.get(doc_id, {})
    retriever = doc_entry.get("retriever") or ACTIVE_RETRIEVER
    ast = doc_entry.get("ast") or DOCUMENT_ASTS.get(doc_id)
    classification = doc_entry.get("classification") or ACTIVE_CLASSIFICATION

    if not retriever:
        return StreamingResponse(iter(["Error: No document loaded."]), media_type="text/plain")

    retrieved = await asyncio.to_thread(retriever.retrieve, query, TOP_K)
    _hydrate_ast_siblings(retrieved, ast)
    eval_result = evaluate_relevance(retrieved, query=query)

    if eval_result["status"] == "REFUSE":
        candidate_pages = [c.get("page") for c in retrieved if c.get("page")]
        escalated_chunks = await asyncio.to_thread(
            retriever.escalate_page_scan, query, candidate_pages, ast
        )
        if escalated_chunks:
            escalated_eval = evaluate_relevance(escalated_chunks, query=query)
            if escalated_eval["status"] != "REFUSE":
                retrieved = escalated_chunks
                _hydrate_ast_siblings(retrieved, ast)
                eval_result = escalated_eval

    if eval_result["status"] == "REFUSE":
        return StreamingResponse(
            iter([f"⚠️ UNVERIFIABLE: {eval_result['reason']}"]),
            media_type="text/plain"
        )

    persona = classification.get("prompt_addon", "")
    return StreamingResponse(
        stream_grounded_answer(query, retrieved, persona),
        media_type="text/event-stream"
    )


@app.post("/export-report")
async def export_report(payload: ExportReportRequest):
    """
    Exports an audited compliance finding as a downloadable PDF (.pdf), Word (.docx),
    CSV (.csv), Markdown (.md), or JSON (.json) report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_label = payload.classification.get("label", "General Enterprise Document")
    doc_type = payload.classification.get("type", "GENERAL_SPECIFICATION")
    eval_badge = payload.evaluation.get("badge", "Verified")
    confidence = payload.evaluation.get("confidence", 0.0)
    fmt = payload.format.lower().strip()

    if fmt == "json":
        report_obj = {
            "generated_at": timestamp,
            "engine": "AuditGuard Tree-RAG v2.0",
            "model": get_active_gemma_model(),
            "classification": payload.classification,
            "evaluation": payload.evaluation,
            "query": payload.query,
            "grounded_answer": payload.answer,
            "baseline_comparison": payload.baseline,
            "sources": payload.sources,
        }
        return Response(
            content=json.dumps(report_obj, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="audit_trail_{file_timestamp}.json"'}
        )

    if fmt == "pdf":
        pdf_bytes = build_pdf_bytes(
            title="AuditGuard Verification Report",
            timestamp=timestamp,
            doc_label=doc_label,
            confidence=confidence,
            query=payload.query,
            answer=payload.answer,
            sources=payload.sources,
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="audit_report_{file_timestamp}.pdf"'}
        )

    if fmt in ("docx", "word", "doc"):
        docx_bytes = build_docx_bytes(
            title="AuditGuard Verification Report",
            timestamp=timestamp,
            doc_label=doc_label,
            confidence=confidence,
            query=payload.query,
            answer=payload.answer,
            sources=payload.sources,
        )
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="audit_report_{file_timestamp}.docx"'}
        )

    if fmt == "csv":
        csv_text = build_csv_string(
            timestamp=timestamp,
            query=payload.query,
            answer=payload.answer,
            sources=payload.sources,
        )
        return Response(
            content=csv_text.encode("utf-8-sig"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="audit_report_{file_timestamp}.csv"'}
        )

    md_lines = [
        "# 🛡️ AuditGuard RAG — Compliance Verification Report",
        f"**Generated At:** {timestamp}  ",
        f"**Document Classification:** {doc_label} (`{doc_type}`)  ",
        f"**Grounding Verdict:** {eval_badge} (`{confidence}%` Confidence)  ",
        f"**Active Model:** `{get_active_gemma_model()}`  ",
        "",
        "---",
        "",
        "## 1. Compliance Audit Query",
        f"> {payload.query}",
        "",
        "## 2. Verified Grounded Finding",
        f"{payload.answer}",
        "",
    ]

    if payload.baseline:
        md_lines.extend([
            "## 3. Hallucination Radar (Ungrounded Baseline Comparison)",
            "**Plain Gemma 4 (Zero AST Context):**",
            f"> {payload.baseline}",
            "",
        ])

    md_lines.extend([
        "## 4. Source Evidence & AST Provenance Trail",
        "| Citation | Document | Page | AST Path | Similarity | Clause Excerpt |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for s in payload.sources:
        cid = s.get("citation_id", 1)
        doc = s.get("filename", "Document")
        page = s.get("page", 1)
        ast_path = s.get("materialized_path", f"root/page_{page}")
        score = round(float(s.get("dense_score", 0.0)) * 100, 1)
        excerpt = s.get("parent_text", s.get("child_text", "")).replace("\n", " ").strip()
        if len(excerpt) > 120:
            excerpt = excerpt[:117] + "..."
        excerpt = excerpt.replace("|", "\\|")
        md_lines.append(f"| `[{cid}]` | {doc} | Page {page} | `{ast_path}` | {score}% | {excerpt} |")

    md_lines.extend([
        "",
        "---",
        "*Generated by AuditGuard Tree-RAG Verification Engine*"
    ])

    md_content = "\n".join(md_lines)
    return Response(
        content=md_content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="audit_report_{file_timestamp}.md"'}
    )


@app.get("/api/ast")
def get_active_document_ast():
    """Returns the complete Document AST for visual graph rendering."""
    if not ACTIVE_DOC_ID or ACTIVE_DOC_ID not in DOCUMENT_ASTS:
        raise HTTPException(status_code=404, detail="No active Document AST available.")
    return DOCUMENT_ASTS[ACTIVE_DOC_ID].to_dict()


@app.get("/api/discrepancies")
@app.get("/api/ast/discrepancies")
async def get_active_discrepancies(use_llm: bool = False):
    """Runs automated Cross-Modal Contradiction Scanner across active AST."""
    if not ACTIVE_DOC_ID or ACTIVE_DOC_ID not in DOCUMENT_ASTS:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload or load a document first.")

    ast = DOCUMENT_ASTS[ACTIVE_DOC_ID]
    report = await asyncio.to_thread(scan_ast_for_discrepancies, ast, use_llm)
    report["filename"] = DOCUMENT_VAULT.get(ACTIVE_DOC_ID, {}).get("filename", "Current Document")
    return report


@app.get("/api/ast/{doc_id}")
def get_document_ast_by_id(doc_id: str):
    """Returns the Document AST for a specific vault document."""
    if doc_id not in DOCUMENT_ASTS:
        raise HTTPException(status_code=404, detail=f"Document AST for '{doc_id}' not found.")
    return DOCUMENT_ASTS[doc_id].to_dict()


@app.get("/api/crop/{doc_id}/{node_id}")
async def get_visual_crop(doc_id: str, node_id: str):
    ast = DOCUMENT_ASTS.get(doc_id) or await asyncio.to_thread(load_cached_ast, doc_id)
    node = ast.get_node(node_id) if ast else None
    if node is None or node.type != ASTNodeType.VISUAL:
        raise HTTPException(status_code=404, detail="Visual not found.")
    image_path = await asyncio.to_thread(get_or_render_visual_crop, ast, node)
    if image_path and image_path.exists():
        return FileResponse(str(image_path), media_type="image/webp")
    raise HTTPException(status_code=404, detail="Visual image unavailable.")


@app.get("/api/preview/{doc_prefix}/{page_num}")
async def get_preview_image(doc_prefix: str, page_num: int):
    """
    On-Demand Visual Audit Proof Renderer:
    Renders and serves high-res single page previews for 300-page PDFs in ~30ms.
    """
    img_path = await asyncio.to_thread(get_or_render_page_image, doc_prefix, page_num)
    if img_path and img_path.exists():
        media_type = "image/webp" if img_path.suffix.lower() == ".webp" else "image/png"
        return FileResponse(str(img_path), media_type=media_type)
    raise HTTPException(status_code=404, detail="Page preview could not be generated.")


SAMPLE_SOP_CONTENT = """# GLOBAL VIRTUAL ASSISTANT SERVICE LEVEL AGREEMENT & ESCALATION SOP

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


@app.post("/load-sample")
async def load_sample_document():
    """
    1-Click Stage Demo Mode:
    Passes sample_enterprise_sop.txt through the exact same AST, visual table extraction,
    and LanceDB indexing pipeline used for uploaded documents (zero hardcoded AST node overrides).
    """
    request_started = time.perf_counter()
    sample_filename = "sample_enterprise_sop.txt"
    sample_path = DOCS_DIR / sample_filename
    if not sample_path.exists():
        sample_path.write_text(SAMPLE_SOP_CONTENT, encoding="utf-8")
    sample_content = sample_path.read_text(encoding="utf-8")

    return await _attach_document(sample_content.encode("utf-8"), sample_filename, request_started)



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
            "attachment_duration_ms": v.get("attachment_duration_ms", 0),
            "stage_timings_ms": v.get("stage_timings_ms", {}),
            "cache_hit": v.get("cache_hit", False),
            "cache_warning": v.get("cache_warning"),
            "background_indexing": BACKGROUND_INDEXER.snapshot(v.get("background_indexing", {})),
            "classification": v["classification"],
            "uploaded_at": v["uploaded_at"],
            "is_active": (v["doc_id"] == ACTIVE_DOC_ID)
        })
    return {"vault": items, "active_doc_id": ACTIVE_DOC_ID}


@app.post("/vault/select/{doc_id}")
async def select_vault_document(doc_id: str):
    """Switch active document from the vault atomically."""
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION, ACTIVE_DOC_ID
    if doc_id not in DOCUMENT_VAULT:
        raise HTTPException(status_code=404, detail="Document not found in vault.")

    async with STATE_LOCK:
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
async def get_compliance_audit():
    """Runs automated 7-point pre-flight compliance audit across the active document."""
    doc_id = ACTIVE_DOC_ID
    doc_entry = DOCUMENT_VAULT.get(doc_id, {})
    retriever = doc_entry.get("retriever") or ACTIVE_RETRIEVER
    classification = doc_entry.get("classification") or ACTIVE_CLASSIFICATION

    if not retriever:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload or load a document first.")

    report = await asyncio.to_thread(run_compliance_audit, retriever)
    report["filename"] = doc_entry.get("filename", "Current Document")
    report["classification"] = classification
    return report


@app.post("/api/model/toggle")
def toggle_model():
    """Toggle between high-speed Gemma 4 MoE (26b-a4b-it) and dense Gemma 4 (31b-it)."""
    current = get_active_gemma_model()
    if current == "gemma-4-26b-a4b-it":
        updated = set_active_gemma_model("gemma-4-31b-it")
    else:
        updated = set_active_gemma_model("gemma-4-26b-a4b-it")
    return {
        "ok": True,
        "model": updated,
        "label": "Gemma 4 MoE (~3s)" if updated == "gemma-4-26b-a4b-it" else "Gemma 4 Dense 31B (~28s)"
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
    return HTMLResponse(content="<h1>Assistant UI not found.</h1>", status_code=404)
