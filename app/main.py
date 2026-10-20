import json
import os
import time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from app.config import (
    DOCS_DIR, INDEX_DIR, PREVIEWS_DIR, TOP_K
)
from app.ingest import (
    extract_pages_from_pdf, extract_pages_from_text,
    render_page_previews, hierarchical_chunk, get_or_render_page_image
)
from app.embeddings import embed_texts
from app.retriever import HybridRetriever
from app.evaluator import evaluate_relevance
from app.generator import generate_grounded_answer, stream_grounded_answer
from app.llm import call_plain_gemma
from app.audit_scanner import run_compliance_audit

app = FastAPI(title="AuditGuard RAG", version="1.0.0")

# Mount Static Files (Frontend UI & Visual Page Previews)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-Memory State & Multi-Document Knowledge Vault
ACTIVE_RETRIEVER: HybridRetriever | None = None
ACTIVE_CHUNKS: list = []
ACTIVE_DOC_ID: str = ""
DOCUMENT_VAULT: dict = {}

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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "doc_loaded": len(ACTIVE_CHUNKS) > 0,
        "classification": ACTIVE_CLASSIFICATION.get("label"),
        "chunk_count": len(ACTIVE_CHUNKS)
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload any general PDF or text document and build the Hybrid RAG index."""
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION

    contents = await file.read()
    filename = file.filename or "uploaded_doc.pdf"
    doc_id = Path(filename).stem.replace(" ", "_")

    # 1. Page extraction & visual preview rendering
    if filename.lower().endswith(".pdf"):
        pages = extract_pages_from_pdf(contents, filename)
        page_images = render_page_previews(contents, doc_id)
    else:
        text_str = contents.decode("utf-8", errors="replace")
        pages = extract_pages_from_text(text_str, filename)
        page_images = {}

    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract readable text from document.")

    # 2. Hierarchical Chunking + Automatic Document Type Classification
    chunks, classification = hierarchical_chunk(pages, page_images)
    ACTIVE_CLASSIFICATION = classification

    # 3. Dense Vector Embeddings (Child Sentences)
    child_texts = [c["child_text"] for c in chunks]
    embeddings_matrix = embed_texts(child_texts)

    # 4. Initialize In-Memory Hybrid RRF Retriever
    ACTIVE_CHUNKS = chunks
    ACTIVE_RETRIEVER = HybridRetriever(chunks, embeddings_matrix)
    ACTIVE_DOC_ID = doc_id

    DOCUMENT_VAULT[doc_id] = {
        "doc_id": doc_id,
        "filename": filename,
        "pages": len(pages),
        "total_chunks": len(chunks),
        "classification": classification,
        "retriever": ACTIVE_RETRIEVER,
        "chunks": chunks,
        "uploaded_at": datetime.now().strftime("%H:%M:%S")
    }

    # Save to disk for quick stage reloads
    try:
        with open(INDEX_DIR / "current_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f)
    except Exception as e:
        print(f"Index save warning: {e}")

    return {
        "ok": True,
        "filename": filename,
        "pages_processed": len(pages),
        "total_chunks": len(chunks),
        "classification": ACTIVE_CLASSIFICATION,
        "doc_id": doc_id
    }

class ExportReportRequest(BaseModel):
    query: str
    answer: str
    sources: list = []
    classification: dict = {}
    evaluation: dict = {}
    baseline: str = ""
    format: str = "markdown"

@app.post("/ask")
async def ask_question(payload: QueryRequest):
    """
    Standard JSON endpoint for AuditGuard Q&A.
    Returns Grounded Answer, Proof Drawer sources, CRAG Evaluation, Plain Gemma Baseline,
    exact retrieval latency in milliseconds, and estimated token usage.
    """
    global ACTIVE_RETRIEVER, ACTIVE_CLASSIFICATION
    if not ACTIVE_RETRIEVER:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a document first.")

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Hybrid RRF Retrieval with Microsecond Benchmarking
    t0 = time.perf_counter()
    retrieved = ACTIVE_RETRIEVER.retrieve(query, top_k=TOP_K)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Token Usage Estimation (~1.33 tokens per English word + 200 system tokens)
    context_words = sum(len(c.get("parent_text", "").split()) for c in retrieved)
    query_words = len(query.split())
    estimated_tokens = int((context_words + query_words) * 1.33 + 200)

    # 3. CRAG Relevance Gatekeeper
    eval_result = evaluate_relevance(retrieved)

    # 4. Check for Defense Refusal
    if eval_result["status"] == "REFUSE":
        plain_baseline = call_plain_gemma(query) if payload.include_baseline else ""
        return {
            "query": query,
            "status": "REFUSE",
            "evaluation": eval_result,
            "answer": eval_result["reason"],
            "sources": [],
            "baseline": plain_baseline,
            "classification": ACTIVE_CLASSIFICATION,
            "retrieval_ms": retrieval_ms,
            "estimated_tokens": estimated_tokens
        }

    # 5. Synthesize Grounded Answer with Gemma 4
    persona = ACTIVE_CLASSIFICATION.get("prompt_addon", "")
    grounded_answer = generate_grounded_answer(query, retrieved, persona)

    # 6. Baseline for Side-by-Side Comparison (Hallucination Radar)
    plain_baseline = call_plain_gemma(query) if payload.include_baseline else ""

    return {
        "query": query,
        "status": "SUCCESS",
        "evaluation": eval_result,
        "answer": grounded_answer,
        "sources": retrieved,
        "baseline": plain_baseline,
        "classification": ACTIVE_CLASSIFICATION,
        "retrieval_ms": retrieval_ms,
        "estimated_tokens": estimated_tokens
    }

@app.get("/ask-stream")
async def ask_stream(query: str):
    """Real-time SSE token streaming endpoint."""
    global ACTIVE_RETRIEVER, ACTIVE_CLASSIFICATION
    if not ACTIVE_RETRIEVER:
        return StreamingResponse(iter(["Error: No document loaded."]), media_type="text/plain")

    retrieved = ACTIVE_RETRIEVER.retrieve(query, top_k=TOP_K)
    eval_result = evaluate_relevance(retrieved)

    if eval_result["status"] == "REFUSE":
        return StreamingResponse(
            iter([f"⚠️ UNVERIFIABLE: {eval_result['reason']}"]),
            media_type="text/plain"
        )

    persona = ACTIVE_CLASSIFICATION.get("prompt_addon", "")
    return StreamingResponse(
        stream_grounded_answer(query, retrieved, persona),
        media_type="text/event-stream"
    )

@app.get("/api/preview/{doc_prefix}/{page_num}")
async def get_preview_image(doc_prefix: str, page_num: int):
    """
    On-Demand Visual Audit Proof Renderer:
    Renders and serves high-res single page previews for 300-page PDFs in ~30ms (supports WebP & PNG).
    """
    img_path = get_or_render_page_image(doc_prefix, page_num)
    if img_path and img_path.exists():
        media_type = "image/webp" if img_path.suffix.lower() == ".webp" else "image/png"
        return FileResponse(str(img_path), media_type=media_type)
    raise HTTPException(status_code=404, detail="Page preview could not be generated.")

@app.post("/export-report")
async def export_report(payload: ExportReportRequest):
    """
    Downloads audited compliance analysis as .md or .json.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_label = payload.classification.get("label", "General Document")
    doc_type = payload.classification.get("type", "UNKNOWN")
    confidence = payload.evaluation.get("confidence", 0.0)
    badge = payload.evaluation.get("badge", "UNVERIFIED")

    if payload.format.lower() == "json":
        export_data = {
            "title": "AuditGuard Compliance Verification Report",
            "timestamp": timestamp,
            "document_type": doc_type,
            "document_classification": doc_label,
            "confidence_score": confidence,
            "status_badge": badge,
            "query": payload.query,
            "grounded_answer": payload.answer,
            "unverified_baseline": payload.baseline,
            "sources": payload.sources
        }
        return JSONResponse(
            content=export_data,
            headers={"Content-Disposition": f'attachment; filename="audit_trail_{file_timestamp}.json"'}
        )

    # Markdown format
    md_lines = [
        f"# 🛡️ AuditGuard Compliance Verification Report",
        f"**Generated:** {timestamp} | **Document Type:** {doc_label} (`{doc_type}`)",
        f"**Audit Status:** `{badge}` | **Confidence:** `{round(confidence * 100, 1)}%`",
        f"",
        f"---",
        f"",
        f"## 1. Compliance Audit Query",
        f"> {payload.query}",
        f"",
        f"## 2. Verified Grounded Finding (Gemma 4)",
        f"{payload.answer}",
        f"",
    ]

    if payload.baseline:
        md_lines.extend([
            f"## 3. Hallucination Radar (Baseline Comparison)",
            f"**Plain Gemma 4 (No RAG):**",
            f"> {payload.baseline}",
            f"",
        ])

    md_lines.extend([
        f"## 4. Source Evidence & Visual Provenance Trail",
        f"| Citation | Document | Page | Confidence | Clause Excerpt |",
        f"| :--- | :--- | :--- | :--- | :--- |"
    ])

    for s in payload.sources:
        cid = s.get("citation_id", 1)
        doc = s.get("filename", "Doc")
        page = s.get("page", 1)
        score = round(s.get("dense_score", 0.0) * 100, 1)
        excerpt = s.get("parent_text", s.get("child_text", "")).replace("\n", " ").strip()
        if len(excerpt) > 120:
            excerpt = excerpt[:117] + "..."
        md_lines.append(f"| `[{cid}]` | {doc} | Page {page} | {score}% | {excerpt} |")

    md_lines.extend([
        f"",
        f"---",
        f"*Report certified by AuditGuard RAG Engine • Built for ALMA Future Forge Hackathon 2026*"
    ])

    md_content = "\n".join(md_lines)
    return HTMLResponse(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="audit_report_{file_timestamp}.md"'}
    )

@app.post("/load-sample")
async def load_sample_document():
    """
    1-Click Stage Demo Mode:
    Instantly builds/loads an enterprise SLA & SOP in 0.05 seconds for seamless presentations.
    """
    global ACTIVE_RETRIEVER, ACTIVE_CHUNKS, ACTIVE_CLASSIFICATION
    sample_path = DOCS_DIR / "sample_enterprise_sop.txt"
    
    if not sample_path.exists():
        # Auto-create sample document
        sample_content = """# GLOBAL VIRTUAL ASSISTANT SERVICE LEVEL AGREEMENT & ESCALATION SOP

1. PURPOSE & SCOPE OF ENGAGEMENT
This Agreement governs all remote virtual assistant, administrative, and data verification operations provided by ALMA Talent Solutions. All remote personnel must maintain compliance with enterprise data security protocols (ISO 27001).

2. SYSTEM UPTIME & RESPONSE TIME SLA
2.1 Uptime Guarantee: The service provider guarantees 99.9% availability during standard client business hours (08:00 to 18:00 EST).
2.2 Penalty Credit: In the event that service availability drops below 99.5% in any calendar billing month, the client shall be entitled to an immediate 15% billing credit applied to the next invoice cycle. If availability drops below 99.0%, a 25% credit shall apply.

3. DATA PROTECTION & ZERO-TRUST SECURITY
3.1 Access Control: Remote virtual assistants shall never store client credentials on personal local drives. All authentication must occur through hardware-enforced MFA or enterprise SSO.
3.2 Incident Escalation Flowchart (Figure 1):
- Step 1: Any unauthorized data access alert must be logged within 15 minutes of occurrence.
- Step 2: Immediate written notice must be submitted to the Client Security Officer (security@client.corp) within 1 hour.
- Step 3: All active VPN sessions for the affected contractor account must be terminated immediately.

4. WIRE TRANSFERS & FINANCIAL AUTHORIZATION
4.1 Dual Authorization: Wire transfers or account disbursements exceeding $50,000 require written sign-off from both the Chief Financial Officer (CFO) and the Managing Director.
4.2 Audit Retention: All wire transfer authorization records must be archived in encrypted cold storage for a minimum period of 7 years.

5. TERMINATION FOR CONVENIENCE
Either party may terminate this agreement without cause by providing at least 30 business days prior written notice. Early termination fees are explicitly waived after the initial 6-month contract anniversary."""
        sample_path.write_text(sample_content, encoding="utf-8")

    text_data = sample_path.read_text(encoding="utf-8")
    pages = extract_pages_from_text(text_data, "sample_enterprise_sop.txt")
    chunks, classification = hierarchical_chunk(pages)
    
    ACTIVE_CLASSIFICATION = classification
    ACTIVE_CHUNKS = chunks
    child_texts = [c["child_text"] for c in chunks]
    embeddings_matrix = embed_texts(child_texts)
    ACTIVE_RETRIEVER = HybridRetriever(chunks, embeddings_matrix)
    
    sample_doc_id = "sample_enterprise_sop"
    ACTIVE_DOC_ID = sample_doc_id
    DOCUMENT_VAULT[sample_doc_id] = {
        "doc_id": sample_doc_id,
        "filename": "sample_enterprise_sop.txt",
        "pages": len(pages),
        "total_chunks": len(chunks),
        "classification": classification,
        "retriever": ACTIVE_RETRIEVER,
        "chunks": chunks,
        "uploaded_at": datetime.now().strftime("%H:%M:%S")
    }

    return {
        "ok": True,
        "filename": "sample_enterprise_sop.txt",
        "doc_id": sample_doc_id,
        "pages_processed": len(pages),
        "total_chunks": len(chunks),
        "classification": ACTIVE_CLASSIFICATION
    }

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
def get_compliance_audit():
    """Runs automated 6-point pre-flight compliance audit across the active document."""
    if not ACTIVE_RETRIEVER:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload or load a document first.")
    
    report = run_compliance_audit(ACTIVE_RETRIEVER)
    report["filename"] = DOCUMENT_VAULT.get(ACTIVE_DOC_ID, {}).get("filename", "Current Document")
    report["classification"] = ACTIVE_CLASSIFICATION
    return report

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page dashboard."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AuditGuard UI Loading...</h1>")
