# AuditGuard RAG — Conversation Handoff & Implementation Blueprint

**Target Thread:** [Coding Warm-Up Exercises](conversation://0582c586-3ac1-4261-a517-0e0a398fa04f)  
**Primary Repositories:**  
- **Backend:** `F:\auditguard-rag` (FastAPI + LanceDB + FastEmbed + Google GenAI / Gemma 4)
- **Frontend:** `F:\auditguard-ui` (React + Vite + assistant-ui + Tailwind)
- **Static Distribution:** `F:\auditguard-rag\app\static\assistant`

---

## 1. Executive Status & Test Verification
- **Automated Test Suite:** **28 / 28 tests passing (100% green)** via `python -m pytest tests/ -v`.
- **Backend Server:** Running on `http://127.0.0.1:8000` (FastAPI + Uvicorn). `/health` endpoint actively verifies LanceDB and Google GenAI connectivity.
- **Frontend Server:** Running on `http://127.0.0.1:3000` (Vite dev server) with synchronized production build at `http://127.0.0.1:8000/assistant/`.
- **Authentication & Models:** Authenticated via `GEMINI_API_KEY` using native `google-genai` SDK (`gemma-4-26b-a4b-it` / `gemma-4-31b-it`).
- **Supported Ingestion Formats:** Multi-format parser active: PDF (`pypdfium2`), DOCX (`python-docx`), PPTX (`python-pptx`), TXT/MD (UTF-8).

---

## 2. Completed 15-Point Audit Remediation

1. **Eliminated Fake NLI Entailment (`app/evaluator.py`):**
   - Removed the `* 1.08` magic number and unread `query` parameter.
   - Implemented real CRAG composite grounding score:
     $$\text{Grounding Score} = 0.60 \times \text{Dense Cosine Similarity} + 0.40 \times \text{Lexical Recall}$$
   - Regex word-boundary keyword parsing (`\b[a-zA-Z0-9_\-\.]{3,}\b`) with a 120-word stopword filter.
2. **Fixed LanceDB Vector Indexing & Search (`app/main.py`, `app/vector_store.py`):**
   - Re-indexed AST leaf nodes 1:1 with corresponding text embeddings, removing zero-padding corruption.
   - Connected `LANCE_STORE.search()` into hybrid retrieval via Reciprocal Rank Fusion (RRF).
   - `/health` pings `LANCE_STORE.db.list_tables()` dynamically.
3. **Removed Fabricated Whole-Page Scores (`app/retriever.py`):**
   - Eliminated artificial lexical boost (formerly forced $\ge 0.725$).
   - Replaced substring checks with regex token boundary matching; dense and lexical scores report true measured values.
4. **Authentic Token Telemetry (`app/generator.py`, `app/main.py`):**
   - Eliminated `(words * 1.33 + 200)` guess.
   - Extracts genuine usage telemetry (`response.usage_metadata.total_token_count` or client usage).
5. **Calibrated Classifier & Discrepancy Confidence (`app/classifier.py`, `app/discrepancy_scanner.py`):**
   - Classifier returns `0.0%` confidence on 0 keyword matches.
   - Discrepancy scanner calculates measured textual and numerical agreement.
6. **Removed Sample Loading Stageware (`app/main.py`):**
   - Replaced artificial hardcoded AST node injection with standard document parsing and pre-rendered preview image assets.
7. **Fixed Contradiction Hunter Substring & Hyphen Traps (`app/discrepancy_scanner.py`):**
   - Removed `"up"` and `"-"` substrings that falsely flagged words like "uptime" or "support".
   - Set 0.5% tolerance for SLA/availability and 2.0% for business metrics.
   - Integrated multimodal inspection with Google GenAI.
8. **Compliance Auditor Negation & Disclaimer Awareness (`app/audit_scanner.py`):**
   - Added strict negation and disclaimer filters (`"0% uptime"`, `"does not guarantee"`, `"ban MFA"` $\to$ `❌ GAP DETECTED`).
9. **Synchronized AST Hierarchy with Chunks (`app/ingest.py`, `app/ast_builder.py`):**
   - Synchronized chunk creation with AST leaf identifiers and materialized paths.
10. **Added Missing `/export-report` Endpoint (`app/main.py`):**
    - Fully implemented endpoint supporting Markdown and JSON formats.
11. **Resolved Frontend AST Graph Explorer Crash (`src/Modals.tsx`):**
    - Handled `nodes` object schema (`Object.values(data.nodes || {})`).
12. **Fixed Frontend Compliance Modal Key Lookups (`src/Modals.tsx`):**
    - Corrected key lookups for `clause_excerpt`, `evidence`, and certified verdicts.
13. **Dynamic Model Selection (`app/config.py`, `app/llm.py`):**
    - Replaced static variable imports with dynamic `get_gemma_model()` and `set_gemma_model()`.
14. **Safe LLM Timeouts & Transparent Fallbacks:**
    - LLM timeout set to 15s; fallbacks clearly flagged (`is_fallback: True`).
15. **Dashboard Cleanup & Single-Composer UX (`src/AuditGuardSidebar.tsx`, `src/App.tsx`):**
    - Cleaned left sidebar to display exclusively Chat Threads and `+ New Thread`.
    - Centered empty-state composer.

---

## 3. Analysis of Assistant-UI Structured Output Elements for Document RAG
Reference: `https://www.assistant-ui.com/elements#structured-output`

The following elements from assistant-ui provide substantial value when integrated into the Document RAG pipeline:

| Element | Document RAG Application | Implementation Justification |
| :--- | :--- | :--- |
| **1. Score Breakdown** | **CRAG Grounding Score Meter** | Replaces opaque confidence badges with a visual breakdown of Dense Semantic Cosine Similarity (60%) vs Lexical Token Recall (40%). Users immediately understand why a query passed or escalated. |
| **2. Data Table** | **Extracted Tabular & SLA Records** | Standard markdown tables in chat streams often break or wrap poorly. The structured Data Table element renders extracted PDF/DOCX tables (e.g., quarterly uptime logs, penalty rates) with sorting, column alignment, and clean formatting. |
| **3. Spec Sheet (Key-Value Grid)** | **Contract & SOP Entity Extraction** | For enterprise audits, users need immediate access to core legal parameters: Effective Date, Governing Law, Liability Cap, Termination Window, and Escalation Tier. Renders as a crisp key-value card above prose answers. |
| **4. Comparison (Diff / Side-by-Side)** | **Hallucination Radar** | Displays side-by-side comparison: Ungrounded Raw Gemma Output vs Certified Tree-RAG Grounded Response with highlighted discrepancies, proving the RAG system's prevention of hallucinations. |
| **5. Timeline / Step Stepper** | **Retrieval & Escalation Audit Trail** | Visually tracks the CRAG lifecycle for each query: Tier 1 Leaf Hit $\to$ [Optional] Tier 2 Whole-Page Escalation $\to$ Cross-Modal Contradiction Check $\to$ Grounded Synthesis. Essential for enterprise auditability. |
| **6. Flow Graph / Tree View** | **Interactive 3-Tier Document AST** | Upgrades the current AST modal into an interactive visual tree showing `Document Root` $\to$ `Section` $\to$ `Paragraph Leaf` $\leftrightarrow$ `Visual Sibling Element`. |

---

## 4. Next Implementation Actions in Conversation `0582c586-3ac1-4261-a517-0e0a398fa04f`
1. Create reusable UI components in `F:\auditguard-ui\src\components` for **Score Breakdown**, **Spec Sheet**, and **Data Table**.
2. Update `/query` response parsing in assistant-ui to automatically render these structured output components alongside synthesis text.
3. Build and sync to `F:\auditguard-rag\app\static\assistant`.
