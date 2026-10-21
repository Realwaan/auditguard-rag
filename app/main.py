import asyncio
import json
import time
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
from app.ingest import (
    extract_pages_from_pdf, extract_pages_from_text,
    render_page_previews, hierarchical_chunk, get_or_render_page_image,
    sanitize_doc_id
)
from app.ast_builder import build_document_ast, DocumentAST, ASTNodeType
from app.vector_store import LanceVectorStore
from app.discrepancy_scanner import scan_ast_for_discrepancies
from app.embeddings import embed_texts
from app.retriever import HybridRetriever
from app.evaluator import evaluate_relevance, verify_generated_answer
from app.generator import (
    generate_grounded_answer_with_telemetry,
    stream_grounded_answer
)
from app.llm import call_plain_gemma_with_telemetry, is_llm_configured
from app.audit_scanner import run_compliance_audit

app = FastAPI(title="AuditGuard RAG • Hierarchical Multimodal Tree-RAG", version="2.0.0")

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

ACTIVE_CLASSIFICATION: dict = {
    "type": "NONE",
    "label": "No Document Loaded",
    "icon": "📁",
    "confidence": 0.0,
    "prompt_addon": ""
}


class QueryRequest(BaseModel):
    query: str
    include_baseline: bool = True


class ExportReportRequest(BaseModel):
    query: str
    answer: str
    sources: list = []
    classification: dict = {}
    evaluation: dict = {}
    baseline: str = ""
    format: str = "markdown"


async def _ingest_document_into_vault(
    doc_id: str,
    filename: str,
    pages: list[dict],
    page_images: dict[int, str],
    pdf_bytes: bytes | None = None,
) -> dict:
    """
    Unified, deterministic ingestion pipeline shared by /upload and /load-sample.
    Constructs the 3-Tier AST, extracts visual/table crops, builds AST-aligned parent-child chunks,
    embeds chunks and AST leaves into LanceDB, and updates the active knowledge vault atomically.
    """
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION, ACTIVE_DOC_ID

    t_start = time.perf_counter()

    # 1. Build 3-Tier Document AST (automatically extracts PDF crops and Markdown/ASCII tables)
    ast = await asyncio.to_thread(
        build_document_ast,
        doc_id=doc_id,
        pages=pages,
        pdf_bytes=pdf_bytes,
        doc_title=filename,
    )

    # 2. Hierarchical Chunking synchronized with AST nodes + Automatic Classification
    chunks, classification = hierarchical_chunk(pages, page_images, ast=ast)

    # 3. Dense Vector Embeddings for Child Units
    child_texts = [c["child_text"] for c in chunks]
    embeddings_matrix = await asyncio.to_thread(embed_texts, child_texts)

    # 4. Index AST Leaves into LanceDB with true leaf embeddings
    try:
        await asyncio.to_thread(LANCE_STORE.index_ast, ast)
    except Exception as e:
        print(f"LanceDB indexing warning: {e}")

    # 5. Initialize Hybrid RRF Retriever
    retriever = HybridRetriever(chunks, embeddings_matrix)
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

    async with STATE_LOCK:
        DOCUMENT_ASTS[doc_id] = ast
        DOCUMENT_VAULT[doc_id] = vault_entry
        ACTIVE_RETRIEVER = retriever
        ACTIVE_CHUNKS = chunks
        ACTIVE_CLASSIFICATION = classification
        ACTIVE_DOC_ID = doc_id

    return {
        "ok": True,
        "filename": filename,
        "doc_id": doc_id,
        "pages_processed": len(pages),
        "total_chunks": len(chunks),
        "total_ast_nodes": len(ast.nodes),
        "cross_modal_links": cross_links_count,
        "indexing_duration_ms": indexing_ms,
        "classification": classification,
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
    extracts collision-bounded halo crops, and indexes into LanceDB & HybridRetriever.
    """
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

    doc_id = sanitize_doc_id(filename)
    is_pdf = filename.lower().endswith(".pdf")

    if is_pdf:
        pages = await asyncio.to_thread(extract_pages_from_pdf, contents, filename)
        page_images = await asyncio.to_thread(render_page_previews, contents, doc_id)
    else:
        text_str = contents.decode("utf-8", errors="replace")
        pages = extract_pages_from_text(text_str, filename)
        page_images = {}

    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract readable text from document.")

    return await _ingest_document_into_vault(
        doc_id=doc_id,
        filename=filename,
        pages=pages,
        page_images=page_images,
        pdf_bytes=contents if is_pdf else None,
    )


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
    if eval_result["status"] == "REFUSE":
        plain_baseline = ""
        baseline_tokens = 0
        if payload.include_baseline:
            plain_baseline, base_meta = await asyncio.to_thread(call_plain_gemma_with_telemetry, query)
            baseline_tokens = base_meta.get("baseline_tokens", 0)

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
            "estimated_tokens": baseline_tokens,
            "generation_mode": "refusal_gatekeeper",
        }

    # 5. Synthesize Grounded Answer & Hallucination Baseline in PARALLEL
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

    return {
        "query": query,
        "status": final_status,
        "evaluation": eval_result,
        "answer": grounded_answer,
        "sources": [] if final_status == "REFUSE" else retrieved,
        "baseline": plain_baseline,
        "classification": classification,
        "retrieval_ms": retrieval_ms,
        "total_ms": round((time.perf_counter() - t_total_start) * 1000, 2),
        "estimated_tokens": total_tokens,
        "token_telemetry": {
            "grounded_prompt_tokens": gen_meta.get("prompt_tokens", 0),
            "grounded_completion_tokens": gen_meta.get("completion_tokens", 0),
            "grounded_total_tokens": gen_meta.get("total_tokens", 0),
            "baseline_tokens": base_meta.get("baseline_tokens", 0),
            "total_tokens": total_tokens,
            "generation_mode": gen_meta.get("generation_mode", "unknown"),
            "model_used": gen_meta.get("model_used", ""),
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
    Exports an audited compliance finding as a downloadable Markdown (.md) or JSON (.json) report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_label = payload.classification.get("label", "General Enterprise Document")
    doc_type = payload.classification.get("type", "GENERAL_SPECIFICATION")
    eval_badge = payload.evaluation.get("badge", "Verified")
    confidence = payload.evaluation.get("confidence", 0.0)

    if payload.format.lower() == "json":
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
def get_active_discrepancies():
    """Runs automated Cross-Modal Contradiction Scanner across active AST."""
    if not ACTIVE_DOC_ID or ACTIVE_DOC_ID not in DOCUMENT_ASTS:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload or load a document first.")

    ast = DOCUMENT_ASTS[ACTIVE_DOC_ID]
    report = scan_ast_for_discrepancies(ast, use_llm=True)
    report["filename"] = DOCUMENT_VAULT.get(ACTIVE_DOC_ID, {}).get("filename", "Current Document")
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
    sample_filename = "sample_enterprise_sop.txt"
    sample_path = DOCS_DIR / sample_filename
    if not sample_path.exists():
        sample_path.write_text(SAMPLE_SOP_CONTENT, encoding="utf-8")
    sample_content = sample_path.read_text(encoding="utf-8")

    sample_doc_id = "sample_enterprise_sop"
    pages = extract_pages_from_text(sample_content, sample_filename)

    return await _ingest_document_into_vault(
        doc_id=sample_doc_id,
        filename=sample_filename,
        pages=pages,
        page_images={},
        pdf_bytes=None,
    )


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
