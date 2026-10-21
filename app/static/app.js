// AuditGuard RAG - Hierarchical Multimodal Tree-RAG Client Logic

let ACTIVE_SOURCES = [];
let LAST_QUERY_DATA = null;

const ALLOWED_HTML_ELEMENTS = [
    "p", "br", "strong", "b", "em", "i", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "hr", "span", "button", "a"
];
const ALLOWED_HTML_ATTRIBUTES = ["class", "type", "value", "title", "href", "target", "rel"];

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

/**
 * Safely sanitizes and renders HTML into a target element using the native Sanitizer API
 * (`Element.prototype.setHTML`) when available, with a strict DOMParser allowlist fallback.
 */
function safeSetHTML(el, rawHtml) {
    if (!el) return;
    if ("setHTML" in Element.prototype) {
        try {
            el.setHTML(rawHtml, {
                sanitizer: {
                    elements: ALLOWED_HTML_ELEMENTS,
                    attributes: ALLOWED_HTML_ATTRIBUTES
                }
            });
            return;
        } catch (e) {
            // Fall through to DOMParser allowlist sanitizer if custom config signature differs
        }
    }

    const parser = new DOMParser();
    const doc = parser.parseFromString(rawHtml, "text/html");
    const allowedTags = new Set(ALLOWED_HTML_ELEMENTS.map(t => t.toUpperCase()));
    const allowedAttrs = new Set(ALLOWED_HTML_ATTRIBUTES.map(a => a.toLowerCase()));

    function cleanNode(node) {
        const children = Array.from(node.childNodes);
        for (const child of children) {
            if (child.nodeType === Node.ELEMENT_NODE) {
                if (!allowedTags.has(child.tagName)) {
                    child.remove();
                    continue;
                }
                for (const attr of Array.from(child.attributes)) {
                    const name = attr.name.toLowerCase();
                    const val = attr.value.trim().toLowerCase();
                    if (!allowedAttrs.has(name) || (name === "href" && val.startsWith("javascript:"))) {
                        child.removeAttribute(attr.name);
                    }
                }
                cleanNode(child);
            } else if (child.nodeType !== Node.TEXT_NODE) {
                child.remove();
            }
        }
    }

    cleanNode(doc.body);
    el.replaceChildren(...doc.body.childNodes);
}

document.addEventListener("DOMContentLoaded", () => {
    setupUploadHandlers();
    setupCompareToggle();
    setupKeyboardShortcuts();
    setupDelegatedCitationClicks();
    checkInitialHealth();
    loadVault();
    if (window.marked) {
        marked.setOptions({ gfm: true, breaks: true });
    }
});

function setupDelegatedCitationClicks() {
    const answerContainer = document.getElementById("answer-container");
    if (answerContainer) {
        answerContainer.addEventListener("click", (e) => {
            const btn = e.target.closest(".citation-anchor");
            if (btn && btn.value) {
                openCitationByIndex(Number(btn.value));
            }
        });
    }
}

// ----------------------------------------------------
// KEYBOARD SHORTCUTS & GENERAL UX
// ----------------------------------------------------
function setupKeyboardShortcuts() {
    const queryInput = document.getElementById("query-input");
    if (queryInput) {
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                submitQuery();
            }
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeProofDrawer();
            closeASTModal();
            closeContradictionsModal();
            closeComplianceAuditModal();
        }
    });
}

function showToast(message, type = "success") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    const bgClass = type === "error" ? "bg-rose-950/90 border-rose-800 text-rose-200" :
                    type === "info"  ? "bg-slate-900/90 border-cyan-800/80 text-cyan-200" :
                                       "bg-slate-900/90 border-emerald-500/50 text-emerald-200";

    const icon = type === "error" ? "alert-circle" :
                 type === "info"  ? "info" : "check-circle";

    toast.className = `flex items-center gap-2.5 px-4 py-3 rounded-xl border shadow-xl backdrop-blur-md text-xs font-medium pointer-events-auto transition-all duration-300 transform translate-y-2 opacity-0 ${bgClass}`;
    toast.innerHTML = `<i data-lucide="${icon}" class="w-4 h-4 shrink-0"></i> <span>${escapeHtml(message)}</span>`;

    container.appendChild(toast);
    if (window.lucide) lucide.createIcons();

    requestAnimationFrame(() => {
        toast.classList.remove("translate-y-2", "opacity-0");
    });

    setTimeout(() => {
        toast.classList.add("translate-y-2", "opacity-0");
        setTimeout(() => toast.remove(), 350);
    }, 3500);
}

// ----------------------------------------------------
// INITIAL HEALTH & CLASSIFICATION
// ----------------------------------------------------
function checkInitialHealth() {
    fetch("/health")
        .then(r => r.json())
        .then(data => {
            if (data.model) {
                const engineBadge = document.getElementById("model-engine-badge");
                if (engineBadge) {
                    engineBadge.innerText = data.model;
                }
            }
            if (data.doc_loaded) {
                updateDocBadge(data.classification || "General Enterprise Document", "📄");
            }
            loadVault();
        })
        .catch(() => {});
}

function setupUploadHandlers() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("border-emerald-500", "bg-slate-950/80");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("border-emerald-500", "bg-slate-950/80");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-emerald-500", "bg-slate-950/80");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    const statusBox = document.getElementById("upload-status");
    const filenameEl = document.getElementById("upload-filename");
    const chunksEl = document.getElementById("upload-chunks");

    statusBox.classList.remove("hidden");
    filenameEl.innerText = file.name;
    chunksEl.innerText = "Indexing AST...";

    const formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(async r => {
        const payload = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(payload.detail || "Upload failed");
        return payload;
    })
    .then(data => {
        chunksEl.innerText = `${data.total_chunks} chunks (${data.cross_modal_links || 0} links)`;
        if (data.classification) {
            updateDocBadge(data.classification.label, data.classification.icon);
        }
        if (data.indexing_duration_ms !== undefined) {
            showIngestSpeed(data.indexing_duration_ms);
        }
        showToast(`Indexed ${file.name} in ${(data.indexing_duration_ms / 1000).toFixed(2)}s`, "success");
        loadVault();
    })
    .catch(err => {
        chunksEl.innerText = "Error indexing file";
        showToast(err.message || "Failed to index file", "error");
        console.error(err);
    });
}

function loadSampleDoc() {
    const btn = document.getElementById("btn-load-sample");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span class="animate-spin inline-block mr-1">⏳</span> Slicing AST...`;
    btn.disabled = true;

    fetch("/load-sample", { method: "POST" })
        .then(r => {
            if (!r.ok) throw new Error("Failed to load sample");
            return r.json();
        })
        .then(data => {
            btn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5"></i> Tree Ready`;
            if (window.lucide) lucide.createIcons();

            if (data.classification) {
                updateDocBadge(data.classification.label, data.classification.icon);
            }
            const statusBox = document.getElementById("upload-status");
            statusBox.classList.remove("hidden");
            document.getElementById("upload-filename").innerText = data.filename;
            document.getElementById("upload-chunks").innerText = `${data.total_chunks} chunks (${data.cross_modal_links || 0} visuals linked)`;
            if (data.indexing_duration_ms !== undefined) {
                showIngestSpeed(data.indexing_duration_ms);
            }
            showToast("Sample SOP & SLA indexed into LanceDB", "success");
            loadVault();
            setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; if (window.lucide) lucide.createIcons(); }, 2000);
        })
        .catch(err => {
            btn.innerHTML = `❌ Failed`;
            setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 2000);
            console.error(err);
        });
}

function showIngestSpeed(ms) {
    const badge = document.getElementById("ingest-speed-badge");
    const val = document.getElementById("ingest-speed-val");
    if (badge && val) {
        val.innerText = `${(ms / 1000).toFixed(2)}s`;
        badge.classList.remove("hidden");
        badge.classList.add("flex");
    }
}

function updateDocBadge(label, icon = "📄") {
    const container = document.getElementById("doc-badge-container");
    if (!container) return;
    container.classList.remove("hidden");
    container.classList.add("sm:flex");
    document.getElementById("doc-badge-icon").innerText = icon;
    document.getElementById("doc-badge-label").innerText = label;
}

function setQuery(text) {
    const cleanText = text.replace(/\(Refusal Test\)/i, "").trim();
    const input = document.getElementById("query-input");
    input.value = cleanText;
    input.focus();
    submitQuery();
}

function setupCompareToggle() {
    const toggle = document.getElementById("compare-toggle");
    const groundedCard = document.getElementById("grounded-card");
    const baselineCard = document.getElementById("baseline-card");

    toggle.addEventListener("change", () => {
        if (toggle.checked) {
            groundedCard.classList.remove("md:col-span-12");
            groundedCard.classList.add("md:col-span-6");
            baselineCard.classList.remove("hidden");
        } else {
            groundedCard.classList.remove("md:col-span-6");
            groundedCard.classList.add("md:col-span-12");
            baselineCard.classList.add("hidden");
        }
    });
}

// ----------------------------------------------------
// MARKDOWN RENDERING & CITATION FORMATTING
// ----------------------------------------------------
function renderAuditAnswerInto(container, markdownText) {
    if (!container) return;
    if (!markdownText) {
        container.textContent = "";
        return;
    }
    let html = "";
    if (window.marked && typeof window.marked.parse === "function") {
        html = window.marked.parse(markdownText);
    } else {
        html = escapeHtml(markdownText).replace(/\n/g, "<br>");
    }

    // Convert [1], [2], [10] citation tokens into semantic buttons (handled via delegated event listener)
    html = html.replace(/\[(\d+)\]/g, (match, p1) => {
        return `<button type="button" value="${p1}" class="citation-anchor" title="View Verified AST Proof for Citation [${p1}]"><span>[${p1}]</span></button>`;
    });

    safeSetHTML(container, html);
}

function renderBaselineAnswerInto(container, markdownText) {
    if (!container) return;
    if (!markdownText) {
        safeSetHTML(container, `<p class="text-slate-500 italic py-4">No baseline response received.</p>`);
        return;
    }
    let html = "";
    if (window.marked && typeof window.marked.parse === "function") {
        html = window.marked.parse(markdownText);
    } else {
        html = escapeHtml(markdownText).replace(/\n/g, "<br>");
    }
    safeSetHTML(container, html);
}

// ----------------------------------------------------
// QUERY EXECUTION & VERIFICATION ENGINE
// ----------------------------------------------------
async function submitQuery() {
    const queryInput = document.getElementById("query-input");
    const query = queryInput.value.trim();
    if (!query) return;

    const btn = document.getElementById("btn-submit");
    btn.disabled = true;
    const btnOriginalHtml = `<i data-lucide="check-circle" class="w-4 h-4"></i> <span>Verify</span>`;
    let startTime = Date.now();
    btn.innerHTML = `<span class="animate-spin inline-block">⏳</span> <span>Verifying...</span>`;
    const timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        btn.innerHTML = `<span class="animate-spin inline-block">⏳</span> <span>Verifying (${elapsed}s)...</span>`;
    }, 200);

    const answerContainer = document.getElementById("answer-container");
    const baselineContainer = document.getElementById("baseline-container");
    const badge = document.getElementById("grounding-badge");
    const chipsContainer = document.getElementById("citation-chips");
    const chipsList = document.getElementById("chips-list");
    const pathContainer = document.getElementById("ast-path-container");
    const pathDisplay = document.getElementById("ast-path-display");
    const visualStrip = document.getElementById("linked-visuals-strip");

    // Skeleton loading state
    answerContainer.innerHTML = `
      <div class="space-y-3 py-3 animate-pulse">
        <div class="h-4 bg-emerald-500/20 rounded w-3/4"></div>
        <div class="h-3.5 bg-slate-800 rounded w-full"></div>
        <div class="h-3.5 bg-slate-800 rounded w-5/6"></div>
        <p class="text-[11px] font-mono text-emerald-400 mt-2">Traversing Document AST & Resolving Sibling Pointers...</p>
      </div>`;

    baselineContainer.innerHTML = `
      <div class="space-y-2 py-3 animate-pulse">
        <div class="h-3.5 bg-slate-800 rounded w-4/5"></div>
        <div class="h-3 bg-slate-800/80 rounded w-full"></div>
        <div class="h-3 bg-slate-800/80 rounded w-2/3"></div>
        <p class="text-[11px] font-mono text-amber-400/80 mt-2">Querying ungrounded parametric baseline...</p>
      </div>`;

    badge.innerText = "Evaluating...";
    badge.className = "text-[11px] px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono font-medium border border-slate-700";
    chipsContainer.classList.add("hidden");
    chipsList.innerHTML = "";
    pathContainer.classList.add("hidden");
    visualStrip.classList.add("hidden");

    const compareActive = document.getElementById("compare-toggle").checked;

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, include_baseline: compareActive })
        });

        const data = await response.json();
        clearInterval(timerInterval);
        btn.disabled = false;
        btn.innerHTML = btnOriginalHtml;
        if (window.lucide) lucide.createIcons();
        LAST_QUERY_DATA = data;

        const metricsBadge = document.getElementById("metrics-badge");
        if (metricsBadge && data.retrieval_ms !== undefined) {
            const totalStr = data.total_ms !== undefined
                ? (data.total_ms < 1000 ? `${data.total_ms}ms` : `${(data.total_ms / 1000).toFixed(1)}s`)
                : `${data.retrieval_ms}ms`;
            const tokCount = data.estimated_tokens || 0;
            const modeLabel = data.generation_mode === "extractive_fallback" ? " (Extractive)" : "";
            metricsBadge.innerText = `⚡ Retrieval: ${data.retrieval_ms}ms · Total: ${totalStr} | ${tokCount} tok${modeLabel}`;
            metricsBadge.classList.remove("hidden");
        }

        if (!response.ok) {
            answerContainer.innerHTML = `<div class="p-4 rounded-xl bg-rose-950/40 border border-rose-900 text-rose-200 text-xs">
                <p class="font-bold flex items-center gap-1.5 mb-1"><i data-lucide="alert-circle" class="w-4 h-4"></i> Query Error</p>
                <p>${escapeHtml(data.detail || "Error querying document intelligence engine")}</p>
            </div>`;
            if (window.lucide) lucide.createIcons();
            return;
        }

        // 1. Render Grounding Confidence Badge
        badge.innerText = data.evaluation.badge;
        if (data.status === "REFUSE") {
            badge.className = "text-[11px] px-2.5 py-0.5 rounded-full bg-rose-950 text-rose-300 font-mono font-bold border border-rose-800 shadow-sm";
            answerContainer.innerHTML = `
              <div class="p-4 bg-rose-950/40 border border-rose-900/80 rounded-xl text-rose-200">
                <div class="flex items-center gap-2 mb-2 font-bold text-rose-300">
                  <i data-lucide="shield-alert" class="w-5 h-5 text-rose-400"></i>
                  <span>DEFENSE REFUSAL ACTIVATED (CRAG Gatekeeper)</span>
                </div>
                <p class="text-xs leading-relaxed mb-2">${escapeHtml(data.answer)}</p>
                <div class="p-2 rounded bg-rose-950/80 border border-rose-900 text-[11px] font-mono text-rose-300">
                  Epistemic Humility Rule: When context relevance falls below threshold (τ = 0.58) or lacks verifiable facts, AuditGuard strictly refuses rather than hallucinating.
                </div>
              </div>`;
        } else {
            badge.className = "text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-300 font-mono font-bold border border-emerald-700/80 shadow-sm flex items-center gap-1";
            renderAuditAnswerInto(answerContainer, data.answer);
        }

        // 2. Traversed AST Path Display (Interactive Breadcrumb using real AST metadata)
        ACTIVE_SOURCES = data.sources || [];
        if (ACTIVE_SOURCES.length > 0) {
            const topSource = ACTIVE_SOURCES[0];
            const pageNum = topSource.page || 1;
            const citId = topSource.citation_id || 1;
            const secTitle = topSource.section_title || `Section (P${pageNum})`;
            const astNodeId = topSource.ast_node_id || `Para_${citId}`;
            const matPath = topSource.materialized_path || `root/${astNodeId}`;
            const visualTitle = (topSource.linked_visuals && topSource.linked_visuals.length > 0)
                ? topSource.linked_visuals[0].title
                : null;

            pathDisplay.innerHTML = `
              <button onclick="openASTModal()" title="${escapeHtml(matPath)}" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-purple-300 border border-purple-800/60 transition flex items-center gap-1 font-mono">
                <i data-lucide="folder" class="w-3 h-3"></i> root
              </button>
              <span class="text-slate-600">➔</span>
              <button onclick="openASTModal()" title="${escapeHtml(matPath)}" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-blue-300 border border-blue-800/60 transition flex items-center gap-1 font-mono">
                <i data-lucide="file-text" class="w-3 h-3"></i> ${escapeHtml(secTitle.slice(0, 32))}
              </button>
              <span class="text-slate-600">➔</span>
              <button onclick="openCitationByIndex(${Number(citId)})" title="Node ID: ${escapeHtml(astNodeId)}" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition flex items-center gap-1 font-mono">
                <i data-lucide="align-left" class="w-3 h-3"></i> ${escapeHtml(astNodeId)} [${Number(citId)}]
              </button>
              ${visualTitle ? `
                <span class="text-slate-600">⟷</span>
                <button onclick="openProofDrawer(ACTIVE_SOURCES[0])" class="px-1.5 py-0.5 rounded bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-700 transition flex items-center gap-1 font-mono font-bold">
                  <i data-lucide="image" class="w-3 h-3"></i> [${escapeHtml(visualTitle.slice(0, 36))}]
                </button>` : ""}`;

            pathContainer.classList.remove("hidden");

            // 3. Linked Visual Sibling Banner
            if (topSource.linked_visuals && topSource.linked_visuals.length > 0) {
                const vis = topSource.linked_visuals[0];
                if (vis.image_url) {
                    document.getElementById("linked-visual-thumb").src = vis.image_url;
                }
                document.getElementById("linked-visual-title").innerText = `Resolved Sibling: ${vis.title}`;
                visualStrip.classList.remove("hidden");
            }

            // 4. Proof Drawer Chips
            chipsContainer.classList.remove("hidden");
            ACTIVE_SOURCES.forEach(s => {
                const chipBtn = document.createElement("button");
                chipBtn.className = "px-2.5 py-1 rounded-lg bg-slate-950 hover:bg-slate-800 text-slate-200 border border-slate-700/80 text-[11px] font-mono flex items-center gap-1.5 transition shadow-sm hover:border-emerald-500/50";
                chipBtn.innerHTML = `<span class="font-bold text-emerald-400">[${Number(s.citation_id)}]</span> <span class="text-slate-400">Page ${Number(s.page)}</span>${s.has_visual ? ' <span title="Linked Visual Table/Figure">🖼️</span>' : ''}`;
                chipBtn.onclick = () => openProofDrawer(s);
                chipsList.appendChild(chipBtn);
            });
        }

        // 5. Plain Baseline Container (Hallucination Radar with Sanitized Markdown Rendering)
        if (compareActive && data.baseline) {
            renderBaselineAnswerInto(baselineContainer, data.baseline);
        } else if (!compareActive) {
            safeSetHTML(baselineContainer, `<p class="text-slate-500 italic py-4">Hallucination Radar comparison is currently toggled off.</p>`);
        }

        if (window.lucide) lucide.createIcons();

    } catch (err) {
        clearInterval(timerInterval);
        btn.disabled = false;
        btn.innerHTML = btnOriginalHtml;
        if (window.lucide) lucide.createIcons();
        answerContainer.innerHTML = `<p class="text-rose-400 text-xs">Failed to connect to backend: ${escapeHtml(err.message || err)}</p>`;
    }
}

function openCitationByIndex(idx) {
    const match = ACTIVE_SOURCES.find(s => s.citation_id === Number(idx));
    if (match) {
        openProofDrawer(match);
    } else if (ACTIVE_SOURCES.length > 0) {
        openProofDrawer(ACTIVE_SOURCES[0]);
    }
}

function openProofDrawer(source) {
    if (!source) return;
    document.getElementById("drawer-doc-name").innerText = source.filename || "Document";
    document.getElementById("drawer-page-badge").innerText = `Page ${source.page || 1}`;
    document.getElementById("drawer-parent-text").innerText = source.parent_text || source.child_text;

    const visualSection = document.getElementById("drawer-visual-section");
    const visualImg = document.getElementById("drawer-visual-image");

    if (source.image_url) {
        visualImg.src = source.image_url;
        visualSection.classList.remove("hidden");
    } else {
        visualSection.classList.add("hidden");
    }

    const drawer = document.getElementById("proof-drawer");
    drawer.classList.remove("translate-x-full");
    if (window.lucide) lucide.createIcons();
}

function closeProofDrawer() {
    const drawer = document.getElementById("proof-drawer");
    drawer.classList.add("translate-x-full");
}

// ----------------------------------------------------
// MODAL 1: AST GRAPH EXPLORER
// ----------------------------------------------------
function openASTModal() {
    const modal = document.getElementById("ast-modal");
    modal.classList.remove("hidden");
    loadASTGraph();
    if (window.lucide) lucide.createIcons();
}

function closeASTModal() {
    document.getElementById("ast-modal").classList.add("hidden");
}

function loadASTGraph() {
    const container = document.getElementById("ast-tree-container");
    container.innerHTML = `<p class="text-slate-500 animate-pulse text-center py-4">Traversing In-Memory AST Structure...</p>`;

    fetch("/api/ast")
        .then(r => {
            if (!r.ok) throw new Error("Could not load AST");
            return r.json();
        })
        .then(data => {
            document.getElementById("ast-total-nodes").innerText = data.total_nodes || 0;
            const linksCount = Array.isArray(data.cross_modal_links)
                ? data.cross_modal_links.length
                : (data.cross_modal_links_count || 0);
            document.getElementById("ast-total-links").innerText = linksCount;
            renderASTTree(data);
            if (window.lucide) lucide.createIcons();
        })
        .catch(err => {
            console.error("AST load error:", err);
            container.innerHTML = `<p class="text-amber-400 text-center py-4">⚠️ No active document loaded. Please upload a document or click <strong>1-Click Demo</strong> first.</p>`;
        });
}

function renderASTTree(data) {
    const container = document.getElementById("ast-tree-container");
    const rawNodes = data.nodes || {};
    const nodes = Array.isArray(rawNodes) ? rawNodes : Object.values(rawNodes);
    if (nodes.length === 0) {
        container.innerHTML = `<p class="text-slate-500 italic text-center py-4">Tree is empty. Upload a document to build AST hierarchy.</p>`;
        return;
    }

    const sections = nodes.filter(n => String(n.type || "").toLowerCase() === "section");
    const paragraphs = nodes.filter(n => String(n.type || "").toLowerCase() === "paragraph");
    const visuals = nodes.filter(n => String(n.type || "").toLowerCase() === "visual");

    let html = `<div class="space-y-3 font-sans">
      <!-- Root Node -->
      <div class="p-2.5 rounded-xl bg-purple-950/40 border border-purple-800 text-purple-200 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="px-2 py-0.5 rounded bg-purple-900 text-[10px] font-mono font-bold">H0: ROOT</span>
          <span class="font-bold text-xs">${escapeHtml(data.title || data.filename || "Document AST Root")}</span>
        </div>
        <span class="text-[10px] text-purple-400 font-mono">${nodes.length} total nodes</span>
      </div>
    `;

    sections.forEach(sec => {
        const secParas = paragraphs.filter(p => p.parent_id === sec.id);
        const summary = sec.metadata && sec.metadata.thematic_summary ? sec.metadata.thematic_summary : "";
        html += `
        <div class="ml-4 pl-3 border-l-2 border-slate-700/80 space-y-2">
          <!-- Section Node -->
          <div class="p-2 rounded-lg bg-blue-950/50 border border-blue-800/80 text-blue-200 flex flex-col gap-1">
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <span class="px-1.5 py-0.5 rounded bg-blue-900 text-[10px] text-blue-200 font-mono font-bold">H1/H2: SECTION</span>
                <span class="font-semibold text-xs">${escapeHtml(sec.title)}</span>
              </div>
              <span class="text-[10px] text-blue-400 font-mono">P${Number(sec.page || 1)}</span>
            </div>
            ${summary ? `<p class="text-[10px] text-slate-400 italic truncate">Summary: ${escapeHtml(summary)}</p>` : ""}
          </div>
        `;

        secParas.slice(0, 4).forEach(p => {
            const hasLink = (p.linked_node_ids || []).length > 0;
            html += `
            <div class="ml-4 pl-3 border-l border-slate-800">
              <div class="p-1.5 rounded-lg bg-slate-900 border ${hasLink ? 'border-cyan-700/80 bg-cyan-950/30' : 'border-slate-800'} text-slate-300 flex items-center justify-between gap-2 text-[11px]">
                <span class="truncate max-w-[460px]" title="${escapeHtml(p.materialized_path)}">${escapeHtml(p.text)}</span>
                ${hasLink ? '<span class="px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 text-[10px] border border-cyan-700/80 font-mono shrink-0">⟷ Sibling Linked</span>' : ''}
              </div>
            </div>`;
        });
        if (secParas.length > 4) {
            html += `<p class="ml-8 text-[10px] text-slate-500 italic">+ ${secParas.length - 4} more paragraphs in this section...</p>`;
        }
        html += `</div>`;
    });

    if (visuals.length > 0) {
        html += `
        <div class="pt-3 border-t border-slate-800">
          <p class="text-[10px] uppercase font-bold text-emerald-400 tracking-wider mb-2 flex items-center gap-1.5">
            <i data-lucide="image" class="w-3.5 h-3.5"></i> Resolved Visual Leaves (Halo-Cropped & Table Artifacts):
          </p>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
        `;
        visuals.forEach(v => {
            html += `
            <div class="p-2 rounded-lg bg-emerald-950/30 border border-emerald-800/60 flex items-center gap-2.5">
              ${v.image_url ? `<img src="${escapeHtml(v.image_url)}" class="w-12 h-9 object-cover rounded border border-slate-700 shrink-0">` : ''}
              <div class="overflow-hidden">
                <p class="text-xs font-bold text-emerald-300 truncate">${escapeHtml(v.title)}</p>
                <p class="text-[10px] text-slate-400">Page ${Number(v.page || 1)} • ${escapeHtml(v.visual_type || 'figure')} • ${(v.linked_node_ids || []).length} linked</p>
              </div>
            </div>`;
        });
        html += `</div></div>`;
    }

    html += `</div>`;
    container.innerHTML = html;
}

// ----------------------------------------------------
// MODAL 2: CONTRADICTION HUNTER
// ----------------------------------------------------
function openContradictionsModal() {
    const modal = document.getElementById("contradiction-modal");
    modal.classList.remove("hidden");
    loadContradictions();
    if (window.lucide) lucide.createIcons();
}

function closeContradictionsModal() {
    document.getElementById("contradiction-modal").classList.add("hidden");
}

function loadContradictions() {
    const container = document.getElementById("discrepancy-findings-container");
    container.innerHTML = `<p class="text-slate-500 animate-pulse text-center py-4">Running Cross-Modal AST Contradiction Scan...</p>`;

    fetch("/api/discrepancies")
        .then(r => {
            if (!r.ok) throw new Error("Could not load discrepancies");
            return r.json();
        })
        .then(report => {
            document.getElementById("discrepancy-score-val").innerText = `${report.consistency_score}%`;
            document.getElementById("discrepancy-pairs-text").innerText = `${report.total_pairs_audited} Cross-Modal Pairs Audited`;

            const badge = document.getElementById("discrepancy-verdict-badge");
            badge.innerText = report.verdict;
            if (report.verdict === "FLAGGED") {
                badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-950 text-rose-300 border border-rose-800";
            } else if (report.verdict === "CAUTION") {
                badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-950 text-amber-300 border border-amber-800";
            } else {
                badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800";
            }

            renderDiscrepancyFindings(report.findings || []);
            if (window.lucide) lucide.createIcons();
        })
        .catch(err => {
            container.innerHTML = `<p class="text-amber-400 text-center py-4">⚠️ No active document to audit. Please click <strong>1-Click Demo</strong> first.</p>`;
        });
}

function renderDiscrepancyFindings(findings) {
    const container = document.getElementById("discrepancy-findings-container");
    if (!findings || findings.length === 0) {
        container.innerHTML = `<div class="p-6 rounded-xl bg-slate-950 border border-slate-800 text-center text-slate-400">
            <i data-lucide="shield-check" class="w-8 h-8 mx-auto text-emerald-400 mb-2"></i>
            <p class="font-semibold text-slate-200">No cross-modal pairs or discrepancies found.</p>
            <p class="text-xs text-slate-500 mt-1">Upload a document containing tables or figures to audit cross-modal alignment.</p>
        </div>`;
        return;
    }

    container.innerHTML = "";
    findings.forEach(f => {
        const isConflict = f.status === "CONTRADICTION";
        const card = document.createElement("div");
        card.className = `p-4 rounded-xl border ${isConflict ? 'bg-rose-950/20 border-rose-900/60' : 'bg-slate-950 border-slate-800'} flex flex-col gap-2.5`;
        card.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-base">${isConflict ? '🚨' : '✅'}</span>
            <span class="font-bold text-xs ${isConflict ? 'text-rose-300' : 'text-slate-200'}">${escapeHtml(f.title)}</span>
          </div>
          <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold ${isConflict ? 'bg-rose-950 text-rose-400 border border-rose-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'}">
            ${escapeHtml(f.status)} (${Math.round((f.confidence || 0.9) * 100)}% Conf)
          </span>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-1 text-xs">
          <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <p class="text-[10px] uppercase font-bold text-slate-400 tracking-wider mb-1">Author Prose Claim:</p>
            <p class="text-slate-200 italic">"${escapeHtml(f.text_claim)}"</p>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <p class="text-[10px] uppercase font-bold text-slate-400 tracking-wider mb-1">Raw Visual Chart Fact:</p>
            <p class="text-emerald-300 font-mono font-semibold">"${escapeHtml(f.visual_fact)}"</p>
          </div>
        </div>

        ${f.discrepancy_delta && f.discrepancy_delta !== 'None' ? `
        <div class="p-2 rounded-lg bg-rose-950/40 border border-rose-900/40 text-[11px] text-rose-200 font-mono">
          <strong>Audit Delta:</strong> ${escapeHtml(f.discrepancy_delta)}
        </div>` : ''}
        `;
        container.appendChild(card);
    });
}

// ----------------------------------------------------
// MODAL 3: COMPLIANCE PRE-FLIGHT AUDIT
// ----------------------------------------------------
function openComplianceAuditModal() {
    const modal = document.getElementById("compliance-audit-modal");
    modal.classList.remove("hidden");
    loadComplianceAudit();
    if (window.lucide) lucide.createIcons();
}

function closeComplianceAuditModal() {
    document.getElementById("compliance-audit-modal").classList.add("hidden");
}

function loadComplianceAudit() {
    const container = document.getElementById("audit-checklist-container");
    container.innerHTML = `<p class="text-slate-500 animate-pulse py-6 text-center text-xs">Auditing 7 mandatory controls across AST & LanceDB...</p>`;

    fetch("/compliance-audit")
        .then(r => {
            if (!r.ok) throw new Error("Could not load audit");
            return r.json();
        })
        .then(data => {
            document.getElementById("audit-score-value").innerText = `${data.overall_score}%`;
            document.getElementById("audit-ratio-text").innerText = `${data.passed_count}/${data.total_controls} Controls Verified`;

            const verdictBadge = document.getElementById("audit-verdict-badge");
            verdictBadge.innerText = data.verdict;
            if (data.verdict_color === "emerald" || (data.verdict || "").includes("PASSED")) {
                verdictBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800";
            } else if (data.verdict_color === "red" || (data.verdict || "").includes("FAILED")) {
                verdictBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-950 text-rose-300 border border-rose-800";
            } else {
                verdictBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-950 text-amber-300 border border-amber-800";
            }

            container.innerHTML = "";
            data.checklist.forEach(item => {
                const row = document.createElement("div");
                row.className = "p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2";
                const excerptText = item.clause_excerpt || item.excerpt || "Control missing from active document.";

                row.innerHTML = `
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2.5">
                    <span class="text-base">${item.status === 'PASS' ? '✅' : '⚠️'}</span>
                    <div>
                      <p class="font-bold text-white">${escapeHtml(item.control || item.title)}</p>
                      <p class="text-[11px] text-slate-400">${escapeHtml(item.category)}</p>
                    </div>
                  </div>
                  <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold ${item.status === 'PASS' ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-amber-950 text-amber-300 border border-amber-800'}">
                    ${escapeHtml(item.status)} (${Number(item.confidence || 0)}% Conf)
                  </span>
                </div>
                <div class="p-2 rounded bg-slate-900 border border-slate-800 text-slate-300 italic text-[11px]">
                  "${escapeHtml(excerptText)}"
                </div>`;

                if (item.status === "GAP" && item.remediation_clause) {
                    const remWrapper = document.createElement("div");
                    remWrapper.className = "mt-2 pt-2 border-t border-slate-800/80";
                    remWrapper.innerHTML = `
                      <details class="group cursor-pointer">
                        <summary class="text-[11px] font-semibold text-amber-400 hover:text-amber-300 flex items-center justify-between">
                          <span class="flex items-center gap-1.5"><i data-lucide="git-pull-request" class="w-3.5 h-3.5"></i> Automated Enterprise Remediation Diff</span>
                          <span class="text-[10px] text-slate-500 group-open:rotate-180 transition-transform">▼</span>
                        </summary>
                        <div class="mt-2 space-y-2 bg-slate-900/90 p-3 rounded-lg border border-slate-800 text-[11px]">
                          <p class="text-slate-300"><strong class="text-slate-100 font-semibold">Legal Rationale:</strong> ${escapeHtml(item.rationale || 'Mandatory compliance control.')}</p>
                          <div class="p-2.5 rounded bg-slate-950 font-mono text-[10px] text-emerald-400 overflow-x-auto border border-emerald-950 shadow-inner">
                            <pre class="whitespace-pre-wrap">${escapeHtml(item.redline_diff || '')}</pre>
                          </div>
                          <div class="flex items-center justify-between pt-1">
                            <span class="text-[10px] text-slate-400 italic">Pre-approved enterprise fallback clause</span>
                            <button type="button" class="copy-clause-btn px-2.5 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-[10px] transition flex items-center gap-1">
                              <i data-lucide="copy" class="w-3 h-3"></i> Copy Approved Clause
                            </button>
                          </div>
                        </div>
                      </details>`;
                    const copyBtn = remWrapper.querySelector(".copy-clause-btn");
                    if (copyBtn) {
                        copyBtn.addEventListener("click", () => {
                            navigator.clipboard.writeText(item.remediation_clause);
                            showToast("Pre-approved clause copied to clipboard!", "success");
                        });
                    }
                    row.appendChild(remWrapper);
                }

                container.appendChild(row);
            });
            if (window.lucide) lucide.createIcons();
        })
        .catch(err => {
            container.innerHTML = `<p class="text-amber-400 text-center py-4">⚠️ No active document loaded. Please click <strong>1-Click Demo</strong> first.</p>`;
        });
}

// ----------------------------------------------------
// KNOWLEDGE VAULT & MULTI-DOC SWITCHING
// ----------------------------------------------------
function loadVault() {
    fetch("/vault")
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById("vault-list");
            const badge = document.getElementById("vault-count-badge");
            const items = data.vault || [];
            badge.innerText = `${items.length} Docs`;

            if (items.length === 0) {
                list.innerHTML = `<p class="text-[11px] text-slate-500 italic py-3 text-center">No documents in vault yet.</p>`;
                return;
            }

            list.innerHTML = "";
            items.forEach(doc => {
                const isActive = doc.is_active;
                const row = document.createElement("div");
                row.className = `p-2.5 rounded-xl border flex items-center justify-between text-xs transition-all ${
                    isActive
                    ? 'bg-emerald-950/40 border-emerald-500/60 shadow-sm shadow-emerald-950/40'
                    : 'bg-slate-950/80 hover:bg-slate-850 border-slate-800 hover:border-slate-700 cursor-pointer'
                }`;

                row.onclick = () => {
                    if (!isActive) selectVaultDoc(doc.doc_id, doc.filename);
                };

                row.innerHTML = `
                <div class="flex items-center gap-2.5 truncate">
                  <div class="h-6 w-6 rounded-md flex items-center justify-center shrink-0 ${isActive ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-400'}">
                    <i data-lucide="${isActive ? 'check-circle' : 'file-text'}" class="w-3.5 h-3.5"></i>
                  </div>
                  <div class="truncate">
                    <p class="truncate font-semibold ${isActive ? 'text-emerald-200' : 'text-slate-300'}">${escapeHtml(doc.filename)}</p>
                    <p class="text-[10px] text-slate-500 font-mono">${Number(doc.pages || 1)} pages • ${Number(doc.total_chunks || 0)} chunks</p>
                  </div>
                </div>
                ${isActive
                  ? `<span class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono font-bold shrink-0">ACTIVE</span>`
                  : `<button type="button" class="text-[10px] px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition shrink-0">Switch</button>`
                }`;

                list.appendChild(row);
            });
            if (window.lucide) lucide.createIcons();
        })
        .catch(() => {});
}

function selectVaultDoc(docId, filename = "document") {
    fetch(`/vault/select/${encodeURIComponent(docId)}`, { method: "POST" })
        .then(r => r.json())
        .then(data => {
            loadVault();
            if (data.classification) {
                updateDocBadge(data.classification.label, data.classification.icon);
            }
            showToast(`Switched active document to ${filename}`, "info");
        });
}

// ----------------------------------------------------
// EXPORT AUDIT REPORTS (.MD / .JSON)
// ----------------------------------------------------
async function downloadReport(format = "markdown") {
    if (!LAST_QUERY_DATA) {
        showToast("Please run a verification query first before exporting an audit report.", "info");
        return;
    }

    try {
        const payload = {
            query: LAST_QUERY_DATA.query,
            answer: LAST_QUERY_DATA.answer,
            sources: LAST_QUERY_DATA.sources || [],
            classification: LAST_QUERY_DATA.classification || {},
            evaluation: LAST_QUERY_DATA.evaluation || {},
            baseline: LAST_QUERY_DATA.baseline || "",
            format: format
        };

        const response = await fetch("/export-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Export failed");

        const blob = await response.blob();
        const disposition = response.headers.get("Content-Disposition");
        let filename = format === "json" ? "audit_trail.json" : "audit_report.md";
        if (disposition && disposition.includes("filename=")) {
            filename = disposition.split("filename=")[1].replace(/["']/g, "");
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        showToast(`Downloaded ${filename}`, "success");
    } catch (err) {
        console.error("Export error:", err);
        showToast("Failed to export report", "error");
    }
}

function toggleModel() {
    fetch("/api/model/toggle", { method: "POST" })
        .then(r => r.json())
        .then(data => {
            const badge = document.getElementById("model-engine-badge");
            if (badge) badge.innerText = data.model;
            showToast(`Switched active LLM to ${data.label}`, "info");
            if (window.lucide) lucide.createIcons();
        })
        .catch(() => {});
}
