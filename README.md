# 🛡️ AuditGuard RAG
### Hierarchical Multimodal Tree-RAG & Cross-Modal Compliance Verification Copilot
*Built for the ALMA Future Forge Hackathon 2026 (Cebu City)*  
*Powered by **Gemma 4**, In-Process **LanceDB (Apache Arrow/Rust)** & C++ **PDFium***

---

## 🎯 Executive Summary
Enterprises and remote operations teams waste hundreds of hours cross-referencing US client SOPs, vendor contracts, financial reports, and technical manuals. General-purpose LLMs hallucinate clauses and fail audit reviews. Furthermore, standard RAG pipelines suffer from **context severance** (separating figures from the paragraphs explaining them) and **caption lossiness** (summarizing charts into blurry text).

**AuditGuard RAG** solves this with a **Compiler-Grade Hierarchical Multimodal Tree-RAG architecture**:
1. **3-Tier Document AST**: Parses document hierarchy (`Document Root (H0) ➔ Section (H1/H2) ➔ Paragraph Leaf (H3) ⟷ Visual Leaf (Figure/Table)`) with materialized path prefix lookups (`root/01_scope/02_sla/para_1`).
2. **Collision-Bounded Halo Cropper**: Expands visual bounding boxes by 12% outward padding to capture axis units (`$ in Millions`) and chart legends without clipping, bounded by page margins and saved as lossless WebP crops.
3. **3-Tier Cross-Modal Linker**: Wires bidirectional sibling pointers between citing text and chart visuals using exact regex symbols (`Figure 1`, `Table II`), spatial adjacency fallback, and parent section association.
4. **Cross-Modal Contradiction Hunter**: Detects discrepancies between author prose claims (*"Uptime guarantee of 99.9%"*) and linked visual evidence (*"Table 1 shows 98.2% breach"*).
5. **2-Tier Epistemic Humility (CRAG Gatekeeper with Whole-Page Promotion)**: When leaf-chunk relevance is ambiguous or falls below threshold (tau = 0.50), AuditGuard climbs the Document AST to **Whole-Page Scanning** to recover multi-paragraph context. If even the full page cannot verify the query (< 0.50), it strictly triggers a defense refusal (OUT-OF-BOUNDS) rather than fabricating facts.
6. **Embedded In-Process LanceDB Engine**: Columnar vector storage powered by Rust and Apache Arrow with sub-5ms retrieval and zero external Docker or database dependencies.

---

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
  Tier 1: Chunk Confidence < 0.50?
   ├── NO  ──► Send Grounded Context + Domain Persona (Direct Match)
   └── YES ──► 🔄 Escalate to Whole-Page / Parent-Section Scan
                     │
                     ▼
               Tier 2: Whole-Page Relevance < 0.50?
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
Click **"Load Sample SOP"** in the top navigation bar to instantly index a sample Enterprise IT & Security SLA agreement with built-in footnote discrepancy for contradiction testing.

---

## 🧪 Testing
Run the complete automated test suite (27/27 passing):
```powershell
pytest tests -v
```

---

## 🔑 Gemma 4 Configuration (`.env`)
```ini
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma-4
LLM_API_KEY=alma-hackathon-token
SIMILARITY_THRESHOLD=0.50
TOP_K=4
DEFAULT_EMBED_MODEL=BAAI/bge-small-en-v1.5
```
On hackathon morning, simply configure `LLM_BASE_URL` with ALMA's private inference endpoint.
- Chrome or modern browser

### 2. Start Application
Simply double-click `run.bat` or run:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 3. Load Sample SOP (1-Click)
Click **"Load Sample SOP"** in the top navigation bar to instantly index a sample Enterprise IT & Security SLA agreement.

---

## 🧪 Testing
Run the comprehensive test suite verifying the classifier and hybrid retrieval pipeline:
```powershell
pytest tests -v
```

---

## 🔑 Gemma 4 Configuration (`.env`)
```ini
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma-4
LLM_API_KEY=alma-hackathon-token
SIMILARITY_THRESHOLD=0.58
TOP_K=4
DEFAULT_EMBED_MODEL=BAAI/bge-small-en-v1.5
```
On hackathon morning, simply configure `LLM_BASE_URL` with ALMA's private inference endpoint.
