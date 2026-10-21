# AuditGuard RAG bug-fix review

Reviewed local HEAD: `3978500` — `feat: implement RAG pipeline, AST parser, discrepancy scanner, and API endpoints for contract auditing`.

The changes are local working-tree edits. No setup, installation, deployment, or model configuration change was performed during this fix pass.

## Reported answer mismatch

The screenshot text showed three distinct cases:

- **“What are microprocessors”** produced a refusal alongside a positive grounding indicator. In the reviewed commit, the API still retained positive entailment and claim-verification fields after a generated refusal. The post-generation verifier existed but was not called by `/ask`.
- **“Explain market growth”** produced a CEO announcement through extractive failover. The latest commit's fallback wording differs from the screenshot, but it still returned the first retrieved paragraph as the answer without checking whether it answered the question.
- **“hi”** entered document retrieval. The latest commit had no greeting route, and its keyword filter discarded two-letter terms, including technical terms such as AI and IP.

These failure patterns were reproduced with controlled evidence/provider responses against the reviewed code. The exact `comporglesson1_handout.pdf` from the screenshot was not among the local source documents, so its original extraction/retrieval failure has not been reproduced end to end.

## Changes

| Area | Confirmed issue and correction |
| --- | --- |
| Final answer evaluation | `/ask` now runs post-generation checks. Refusals and empty responses clear positive verification fields. Citation failures or weak question alignment produce a caution result. |
| Provider failure | Provider construction failures and empty output are handled. An outage returns an explicit unverified result with inspectable retrieved sources instead of presenting an unrelated paragraph as the answer. |
| Greetings and retrieval | Complete greetings bypass embeddings and inference. Two-letter technical terms, ordinary singular/plural forms, and punctuation are handled more consistently. Identical text on distinct pages retains distinct source locations. |
| Quantitative evidence | Missing requested percentages cannot pass solely because the query words or a section number occur in the source. Numeric and written percentage values remain supported. |
| Citation checking | Single-digit changes, changed decimals, substring number matches, percentage/unit mismatches, citations after periods, mixed refusal/claim text, and trailing uncited claims are checked. An invalid UI citation no longer opens an unrelated first source. |
| ALMA model selection | An omitted request model no longer forces the Google-specific default onto an OpenAI-compatible endpoint. Explicit model overrides also reach the comparison baseline. Configured timeout is used by both clients. |
| Streaming | Empty usage frames are skipped. Partial-stream failures are marked unverified and cannot be followed by a second provider's answer. Completed streams receive a review warning when answer checks fail. |
| Uploads and state | Size limits are enforced; whitespace-only and malformed files receive client errors. Unsupported binary `.doc`/`.ppt` uploads are rejected rather than passed to XML-format parsers. Indexing failure returns 503 and preserves prior active document state. Exact AST mappings are retained. |
| Visual evidence | Lazy `/api/preview/...` URLs can be resolved for model input. Image loading is restricted to generated preview/crop assets, and invalid preview identifiers are rejected. |
| UI and telemetry | The dashboard distinguishes caution, greeting, refusal, and provider outage. Retrieval timing includes escalation. Actual token counts are not mislabeled as estimates merely because they equal the estimate. |

Retrieval relevance does **not** establish that an answer is correct. `claim_evidence_verified` is no longer asserted by these heuristic checks. Successful final checks report `citation_verified` and `HEURISTIC_SUPPORTED`, with retrieval confidence labeled as retrieval rather than answer accuracy.

## Verification

Command: `python -m pytest tests -q -p no:cacheprovider`

Result: **96 passed, 3 deprecation warnings, 18.07 seconds**.

Tests used separate temporary document/index directories, the existing cached embedding model, real LanceDB operations, and simulated/unavailable inference providers. They did not consume live LLM calls or replace the user's document vault. Windows sandbox temporary-folder permissions required running the isolated tests outside that sandbox.

Also passed: `node --check app/static/app.js` and `git diff --check`.

Existing tests were updated where the former expectation contradicted the fix: retrieval alone must not assert verified claims; a citation fixture now explicitly states the claimed credit; successful streaming uses a deterministic provider instead of relying on the misleading offline fallback.

## Limits for live testing

- This is a regression-test pass rate, **not** a measured RAG answer-success rate. Real Gemma/ALMA output quality, venue model ID/API compatibility, network latency, and laptop performance remain untested.
- Keyword and numeric citation checks are conservative heuristics. They can flag valid paraphrases and cannot prove semantic entailment, resolve all contradictions, or guarantee resistance to document prompt injection.
- Streamed text is provisional until completion; it cannot be retracted after transmission. Failures and failed final checks append an explicit warning.
- The separately maintained React frontend source was not changed. The repository's dashboard JavaScript received the targeted display/citation fixes and a syntax check; no interactive browser validation was performed.
- Large-document indexing still performs child-chunk and AST-node embedding work. No unmeasured upload-speed claim or optimization was introduced.
- Three existing dependency/lifecycle deprecation warnings remain; no dependency upgrade was attempted.
