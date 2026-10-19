# 🛡️ AuditGuard RAG
### Defense-Grade Document Intelligence & Compliance Verification Copilot
*Built for the ALMA Future Forge Hackathon 2026 (Cebu City)*  
*Powered by **Gemma 4** & Local Hybrid Neural-Lexical Search*

---

## 🎯 Executive Summary
Enterprises and remote operations teams waste hundreds of hours cross-referencing US client SOPs, vendor contracts, and security audits. General-purpose LLMs hallucinate clauses and fail audit reviews.

**AuditGuard RAG** solves this with a **zero-trust, epistemic retrieval architecture**:
1. **Epistemic Humility (Refusal over Hallucination)**: When query relevance drops below threshold ($\tau = 0.58$), the system rigorously refuses to answer (`⚠️ OUT-OF-BOUNDS`) rather than fabricating facts.
2. **Visual Proof & Provenance**: Every claim includes citation badges (`[1]`, `[2]`) that open the **Visual Audit Proof Drawer**, rendering high-resolution PDF page captures directly from disk.
3. **Optimized for Gemma 4**: Formatted with strict instruction grounding, zero-shot persona adaptation, and low-latency SSE streaming.

---

## 🏗️ Architecture: Hybrid Small-to-Big Engine

```
[ Upload PDF / SOP ]
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
[ Page Text Extraction ]                 [ pypdfium2 Visual Render ]
        │                                         │
        ▼                                         ▼
[ Small-to-Big Chunking ]               [ /static/previews/page_N.png ]
  - Child sentences (~150 chars)                  │
  - Parent paragraphs (~600 chars)               │
        │                                         │
        ▼                                         │
[ In-Process Indexing ]                           │
  - Dense Vector: BAAI/bge-small (ONNX)          │
  - Lexical Sparse: BM25Okapi                     │
        │                                         │
        ▼                                         │
[ Query: "What is the penalty?" ]                 │
        │                                         │
        ├──────────────────────┐                  │
        ▼                      ▼                  │
 [ Vector Search ]      [ BM25 Lexical ]          │
        └──────────┬───────────┘                  │
                   ▼                              │
      [ Reciprocal Rank Fusion ]                  │
                   │                              │
                   ▼                              │
        [ CRAG Relevance Gate ]                   │
          Confidence < 0.58?                      │
           ├── YES ──► ⚠️ OUT-OF-BOUNDS REFUSAL   │
           └── NO  ──► Send Parent Context        │
                               │                  │
                               ▼                  │
                   [ Gemma 4 LLM Inference ]      │
                               │                  │
                               ▼                  │
                     [ Streamed Response ]        │
                     "Under clause 9 [1]..."      │
                               │                  │
                               ▼                  │
                   [ Clickable Citation Badge ] ──┘
                    (Opens Visual Page Proof)
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
