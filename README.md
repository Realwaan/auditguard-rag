# 🛡️ AuditGuard RAG
### Hierarchical Multimodal Tree-RAG & Cross-Modal Compliance Verification Copilot
*Built for the ALMA Future Forge Hackathon 2026 (Cebu City)*  
*Powered by **Gemma 4**, In-Process **LanceDB (Apache Arrow/Rust)** & C++ **PDFium***

---

## 🎯 Executive Summary
Enterprises and remote operations teams waste hundreds of hours cross-referencing US client SOPs, vendor contracts, financial reports, and technical manuals. General-purpose LLMs hallucinate clauses and fail audit reviews. Furthermore, standard RAG pipelines suffer from **context severance** (separating figures from the paragraphs explaining them) and **caption lossiness** (summarizing charts into blurry text).

**AuditGuard RAG** solves this with a **Compiler-Grade Hierarchical Multimodal Tree-RAG architecture**:
1. **3-Tier Document AST**: Parses document hierarchy (`Document Root (H0) ➔ Section (H1/H2) ➔ Paragraph Leaf (H3) ⟷ Visual Leaf (Figure/Table)`) with materialized path prefix lookups (`root/doc_sec_2/doc_para_3`).
2. **Collision-Bounded Halo Cropper**: Expands visual bounding boxes by 12% outward padding to capture axis units and chart legends without clipping, bounded by page margins and saved as lossless WebP crops.
3. **3-Tier Cross-Modal Linker**: Wires bidirectional sibling pointers between citing text and chart visuals using exact regex symbols (`Figure 1`, `Table 1`), same-section metric alignment, and spatial adjacency fallback.
4. **Cross-Modal Contradiction Hunter**: Detects discrepancies between author prose claims (*"Uptime guarantee of 99.9%"*) and linked visual evidence (*"Table 1 shows 98.20% in September"*).
5. **2-Tier Epistemic Humility (CRAG Gatekeeper with Whole-Page Promotion)**: When leaf-chunk relevance falls below threshold, AuditGuard climbs the Document AST to **Whole-Page Scanning** to recover multi-paragraph context. If even the full page cannot verify the query, it strictly triggers a defense refusal (`OUT-OF-BOUNDS`) rather than fabricating facts.
6. **Embedded In-Process LanceDB Engine**: Columnar vector storage powered by Rust and Apache Arrow with sub-5ms retrieval and zero external Docker or database dependencies.

---

## Attachment indexing and performance

An attachment is ready when text extraction, document structure, child embeddings,
and hybrid search are available. The additional paragraph/visual embeddings and
LanceDB write run in a single background worker. Chat, audits, and citations use
the ready document immediately. Graceful shutdown waits for that worker to finish.
Failed background jobs are logged and exposed through `/vault`; reattaching retries
them without rebuilding an otherwise valid search cache.

The embedding model preloads during application startup, moving initialization out
of the first attachment request. `/health` reports `embedding_preload_succeeded`
and `embedding_preload_ms`. A preload failure is logged; later embedding requests
can retry initialization.

Processed documents are cached under `data/index/ingestion-cache/` as atomic NPZ
files containing JSON metadata and numeric vectors (no pickle). Cache keys include
file contents, PDF/text format, embedding model, and `PIPELINE_VERSION` in
`app/indexing.py`. Bump that version when extraction, chunking, classification, or
visual metadata changes. Cache reuse survives server restarts and filename changes;
citations use the newly uploaded filename. Document IDs are opaque content-based
identifiers: identical contents reuse a vault entry, while changed contents under
the same filename receive a new entry. Invalid caches rebuild automatically.

PDF attachments render only the first page preview. All figure/table detection and
text links still run before readiness, but later page previews and visual crops
render on demand and are saved for reuse. `/api/crop/{doc_id}/{node_id}` can restore
the cached document tree to serve saved citation links after a restart. Keep the
cached metadata and original PDFs to retain this behavior.

`/upload`, `/load-sample`, and `/vault` expose `cache_hit`, `cache_warning`, and
`stage_timings_ms` for cache lookup, extraction, previews, document-tree building,
chunking, child embeddings, cache writes, and retriever setup.
`indexing_duration_ms` now measures the complete search-preparation pipeline,
including extraction and previews. `attachment_duration_ms` includes endpoint work
after multipart parsing; neither measures browser-to-server transfer time.
`background_indexing` separately reports status, errors, duration, and completed
`leaf_embeddings_ms` / `database_write_ms`. A reused LanceDB table has no embedding
or write timings because those stages were skipped.

Regression checks:

```powershell
python -m pytest tests -q -p no:cacheprovider --basetemp data/index/test-run
```

In a local three-run benchmark on the bundled text SOP with a warm embedding model,
median preparation fell from 414 ms for the previous synchronous pipeline to
210 ms for search readiness; cached preparation took 10 ms. This excludes HTTP
overhead and background work and is not an estimate for larger PDFs.

## 🏗️ Architecture: Cross-Linked Multimodal AST

```
[ Upload PDF / SOP ]
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
[ pypdfium2 C++ Extraction ]                   [ Collision-Bounded Halo Cropper ]
  - Hierarchy & Headings                         - 12% Outward Margin Expansion
  - Paragraph Leaf Blocks                        - Lossless WebP Rasterization
        │                                                 │
        ▼                                                 ▼
[ 3-Tier Document AST ] ◄─── Bi-Directional Linker ───► [ Visual Leaves ]
  (Root ➔ Section ➔ Para)  (Symbols, Adjacency, Section)   (/static/crops/*.webp)
        │
        ▼
[ In-Process LanceDB (Rust/Arrow) + BM25 ]
  - Dense Embeddings: BAAI/bge-small-en-v1.5 (ONNX)
  - Lexical Sparse: BM25Okapi
        │
        ▼
[ Query Execution & Reciprocal Rank Fusion ]
        │
        ▼
[ AST Sibling Hydration ]
  - Automatically pulls linked Visual Crop into citation payload
        │
        ▼
[ CRAG Relevance Gatekeeper ]
  Tier 1: Chunk Confidence < Threshold?
   ├── NO  ──► Send Grounded Context + Domain Persona (Direct Match)
   └── YES ──► 🔄 Escalate to Whole-Page / Parent-Section Scan
                     │
                     ▼
               Tier 2: Whole-Page Relevance < Threshold?
                ├── NO  ──► 🛡️ PROMOTED TO PAGE SCAN (Synthesize with Full Page)
                └── YES ──► ⚠️ OUT-OF-BOUNDS REFUSAL (Defense Refusal Triggered!)
                                    │
                                    ▼
                        [ Gemma 4 LLM Inference ]
                       │
                       ▼
             [ Split-Screen UI ]
         - Traversed AST Path Highlight
         - Side-by-Side Hallucination Radar
         - Expandable Proof Drawer & Visual Lightbox
```

---

## ⚡ Quickstart (Under 2 Minutes)

### 1. Requirements
- Python 3.11+
- Chrome or modern browser

### 2. Start Application
Simply double-click `run.bat` or run:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 3. Load Sample SOP (1-Click)
Click **"1-Click Demo"** in the top navigation bar to index a sample Enterprise IT & Security SLA agreement with a built-in table discrepancy (`99.9%` SLA vs `98.20%` September actual) for contradiction testing.

---

## 🧪 Testing
Run the complete automated test suite:
```powershell
python -m pytest tests -v
```

---

## 🔑 Gemma 4 Configuration (`.env`)
```ini
GEMINI_API_KEY=your-gemini-api-key
USE_GOOGLE_GENAI=true
GEMMA_MODEL=gemma-4-26b-a4b-it
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma-4
LLM_API_KEY=alma-hackathon-token
SIMILARITY_THRESHOLD=0.58
TOP_K=4
DEFAULT_EMBED_MODEL=BAAI/bge-small-en-v1.5
```
