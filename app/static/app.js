// AuditGuard RAG - Hierarchical Multimodal Tree-RAG Client Logic

let ACTIVE_SOURCES = [];
let LAST_QUERY_DATA = null;

document.addEventListener("DOMContentLoaded", () => {
    setupUploadHandlers();
    setupCompareToggle();
    checkInitialHealth();
    loadVault();
});

function checkInitialHealth() {
    fetch("/health")
        .then(r => r.json())
        .then(data => {
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
        dropzone.classList.add("border-emerald-500");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("border-emerald-500");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-emerald-500");
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
    .then(r => {
        if (!r.ok) throw new Error("Upload failed");
        return r.json();
    })
    .then(data => {
        chunksEl.innerText = `${data.total_chunks} nodes ready`;
        if (data.classification) {
            updateDocBadge(data.classification.label, data.classification.icon);
        }
        if (data.indexing_duration_ms !== undefined) {
            showIngestSpeed(data.indexing_duration_ms);
        }
        loadVault();
    })
    .catch(err => {
        chunksEl.innerText = "Error indexing file";
        console.error(err);
    });
}

function loadSampleDoc() {
    const btn = document.getElementById("btn-load-sample");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span class="animate-spin">⏳</span> Slicing AST...`;
    btn.disabled = true;

    fetch("/load-sample", { method: "POST" })
        .then(r => r.json())
        .then(data => {
            btn.innerHTML = `✅ Tree Ready`;
            if (data.classification) {
                updateDocBadge(data.classification.label, data.classification.icon);
            }
            const statusBox = document.getElementById("upload-status");
            statusBox.classList.remove("hidden");
            document.getElementById("upload-filename").innerText = data.filename;
            document.getElementById("upload-chunks").innerText = `${data.total_chunks} nodes (${data.cross_modal_links || 1} visuals linked)`;
            if (data.indexing_duration_ms !== undefined) {
                showIngestSpeed(data.indexing_duration_ms);
            }
            loadVault();
            setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 2000);
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
    document.getElementById("doc-badge-container").classList.remove("hidden");
    document.getElementById("doc-badge-icon").innerText = icon;
    document.getElementById("doc-badge-label").innerText = label;
}

function setQuery(text) {
    const cleanText = text.replace(/\(Refusal Test\)/, "").trim();
    document.getElementById("query-input").value = cleanText;
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

async function submitQuery() {
    const queryInput = document.getElementById("query-input");
    const query = queryInput.value.trim();
    if (!query) return;

    const btn = document.getElementById("btn-submit");
    btn.disabled = true;

    const answerContainer = document.getElementById("answer-container");
    const baselineContainer = document.getElementById("baseline-container");
    const badge = document.getElementById("grounding-badge");
    const chipsContainer = document.getElementById("citation-chips");
    const chipsList = document.getElementById("chips-list");
    const pathContainer = document.getElementById("ast-path-container");
    const pathDisplay = document.getElementById("ast-path-display");
    const visualStrip = document.getElementById("linked-visuals-strip");

    answerContainer.innerHTML = `<p class="text-emerald-400 animate-pulse">Traversing AST Hierarchy & Sibling Pointers...</p>`;
    baselineContainer.innerHTML = `<p class="text-slate-400 animate-pulse">Requesting plain baseline...</p>`;
    badge.innerText = "Evaluating...";
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
        btn.disabled = false;
        LAST_QUERY_DATA = data;

        const metricsBadge = document.getElementById("metrics-badge");
        if (metricsBadge && data.retrieval_ms !== undefined) {
            metricsBadge.innerText = `⚡ ${data.retrieval_ms}ms | ~${data.estimated_tokens || 0} tok`;
        }

        if (!response.ok) {
            answerContainer.innerHTML = `<p class="text-red-400">❌ ${data.detail || "Error querying system"}</p>`;
            return;
        }

        // 1. Render Grounding Confidence Badge
        badge.innerText = data.evaluation.badge;
        if (data.status === "REFUSE") {
            badge.className = "text-[11px] px-2.5 py-1 rounded-full bg-red-950 text-red-400 font-mono font-medium border border-red-800";
            answerContainer.innerHTML = `<div class="p-3.5 bg-red-950/40 border border-red-900/60 rounded-lg text-red-200">
                <p class="font-bold mb-1">🛡️ DEFENSE REFUSAL ACTIVATED</p>
                <p>${data.answer}</p>
            </div>`;
        } else {
            badge.className = "text-[11px] px-2.5 py-1 rounded-full bg-emerald-950 text-emerald-300 font-mono font-medium border border-emerald-800";
            
            let formattedAnswer = data.answer.replace(/\[(\d+)\]/g, (match, p1) => {
                return `<button onclick="openCitationByIndex(${p1})" class="inline-flex items-center px-1.5 py-0.2 mx-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono font-bold hover:bg-emerald-500 hover:text-white transition cursor-pointer">[${p1}]</button>`;
            });

            answerContainer.innerHTML = `<div class="whitespace-pre-wrap">${formattedAnswer}</div>`;
        }

        // 2. Traversed AST Path Display
        ACTIVE_SOURCES = data.sources || [];
        if (ACTIVE_SOURCES.length > 0) {
            const topSource = ACTIVE_SOURCES[0];
            const matPath = topSource.materialized_path || `root/section_${topSource.page}/p_${topSource.citation_id || 1}`;
            pathDisplay.innerHTML = `<span class="text-purple-400">root</span> ➔ <span class="text-blue-400">Section (P${topSource.page})</span> ➔ <span class="text-slate-300">Paragraph [${topSource.citation_id || 1}]</span>${topSource.linked_visuals ? ' ⟷ <span class="text-emerald-400 font-bold">[' + topSource.linked_visuals[0].title + ']</span>' : ''}`;
            pathContainer.classList.remove("hidden");

            // 3. Linked Visual Sibling Banner
            if (topSource.linked_visuals && topSource.linked_visuals.length > 0) {
                const vis = topSource.linked_visuals[0];
                document.getElementById("linked-visual-thumb").src = vis.image_url;
                document.getElementById("linked-visual-title").innerText = `Resolved: ${vis.title}`;
                visualStrip.classList.remove("hidden");
            }

            // 4. Proof Drawer Chips
            chipsContainer.classList.remove("hidden");
            ACTIVE_SOURCES.forEach(s => {
                const chipBtn = document.createElement("button");
                chipBtn.className = "px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[11px] font-mono flex items-center gap-1 transition";
                chipBtn.innerHTML = `<span>[${s.citation_id}]</span> <span class="text-slate-400">Page ${s.page}</span>${s.has_visual ? ' 🖼️' : ''}`;
                chipBtn.onclick = () => openProofDrawer(s);
                chipsList.appendChild(chipBtn);
            });
        }

        // 5. Plain Baseline Container (Hallucination Radar)
        if (compareActive && data.baseline) {
            baselineContainer.innerHTML = `<div class="whitespace-pre-wrap text-slate-300">${data.baseline}</div>`;
        }

    } catch (err) {
        btn.disabled = false;
        answerContainer.innerHTML = `<p class="text-red-400">Failed to connect to backend: ${err}</p>`;
    }
}

function openCitationByIndex(idx) {
    const match = ACTIVE_SOURCES.find(s => s.citation_id === Number(idx));
    if (match) {
        openProofDrawer(match);
    }
}

function openProofDrawer(source) {
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

    document.getElementById("proof-drawer").classList.remove("translate-x-full");
}

function closeProofDrawer() {
    document.getElementById("proof-drawer").classList.add("translate-x-full");
}

// ----------------------------------------------------
// MODAL 1: AST GRAPH EXPLORER
// ----------------------------------------------------
function openASTModal() {
    const modal = document.getElementById("ast-modal");
    modal.classList.remove("hidden");
    loadASTGraph();
}

function closeASTModal() {
    document.getElementById("ast-modal").classList.add("hidden");
}

function loadASTGraph() {
    const container = document.getElementById("ast-tree-container");
    container.innerHTML = `<p class="text-slate-500 animate-pulse">Loading Document Abstract Syntax Tree...</p>`;

    fetch("/api/ast")
        .then(r => {
            if (!r.ok) throw new Error("No active AST loaded");
            return r.json();
        })
        .then(ast => {
            document.getElementById("ast-total-nodes").innerText = ast.total_nodes || 0;
            document.getElementById("ast-total-links").innerText = (ast.cross_modal_links || []).length;
            renderASTGraph(ast);
        })
        .catch(err => {
            container.innerHTML = `<p class="text-amber-400">⚠️ No AST currently active. Please click <strong>⚡ 1-Click Demo</strong> or upload a document first.</p>`;
        });
}

function renderASTGraph(ast) {
    const container = document.getElementById("ast-tree-container");
    container.innerHTML = "";

    const nodes = ast.nodes || {};
    const root = nodes[ast.root_id];

    let html = `
    <div class="space-y-3">
      <!-- Root Node -->
      <div class="p-2.5 rounded-lg bg-purple-950/60 border border-purple-800 text-purple-200 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="px-1.5 py-0.5 rounded bg-purple-800 text-[10px] text-white font-bold">H0: ROOT</span>
          <span class="font-bold">${root ? root.title : "Document Root"}</span>
        </div>
        <span class="text-[10px] text-purple-400 font-mono">path: root</span>
      </div>
    `;

    // Group children into sections and visuals
    const sections = Object.values(nodes).filter(n => n.type === "section");
    const visuals = Object.values(nodes).filter(n => n.type === "visual");
    const paragraphs = Object.values(nodes).filter(n => n.type === "paragraph");

    sections.forEach(sec => {
        const secParas = paragraphs.filter(p => p.parent_id === sec.id);
        html += `
        <div class="ml-4 pl-3 border-l-2 border-slate-700 space-y-2">
          <!-- Section Node -->
          <div class="p-2 rounded-lg bg-blue-950/60 border border-blue-800 text-blue-200 flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="px-1.5 py-0.5 rounded bg-blue-800 text-[10px] text-white font-bold">H1/H2: SECTION</span>
              <span class="font-semibold text-xs">${sec.title}</span>
            </div>
            <span class="text-[10px] text-blue-400 font-mono">P${sec.page}</span>
          </div>
        `;

        secParas.slice(0, 3).forEach(p => {
            const hasLink = (p.linked_node_ids || []).length > 0;
            html += `
            <div class="ml-4 pl-3 border-l border-slate-800">
              <div class="p-1.5 rounded bg-slate-900 border ${hasLink ? 'border-cyan-700/80 bg-cyan-950/20' : 'border-slate-800'} text-slate-300 flex items-center justify-between text-[11px]">
                <span class="truncate max-w-[400px]">${p.text}</span>
                ${hasLink ? '<span class="px-1.5 py-0.2 rounded bg-cyan-900/80 text-cyan-300 text-[10px] border border-cyan-700">⟷ Sibling Linked</span>' : ''}
              </div>
            </div>`;
        });
        if (secParas.length > 3) {
            html += `<p class="ml-8 text-[10px] text-slate-500 italic">+ ${secParas.length - 3} more paragraphs in this section...</p>`;
        }
        html += `</div>`;
    });

    if (visuals.length > 0) {
        html += `
        <div class="pt-2 border-t border-slate-800">
          <p class="text-[10px] uppercase font-bold text-emerald-400 tracking-wider mb-2">Resolved Visual Crops (Halo-Cropped):</p>
          <div class="grid grid-cols-2 gap-2">
        `;
        visuals.forEach(v => {
            html += `
            <div class="p-2 rounded-lg bg-emerald-950/40 border border-emerald-800 flex items-center gap-2">
              ${v.image_url ? `<img src="${v.image_url}" class="w-10 h-8 object-cover rounded border border-slate-700">` : ''}
              <div class="overflow-hidden">
                <p class="text-xs font-bold text-emerald-300 truncate">${v.title}</p>
                <p class="text-[10px] text-slate-400">Page ${v.page} • ${v.visual_type || 'figure'}</p>
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
}

function closeContradictionsModal() {
    document.getElementById("contradiction-modal").classList.add("hidden");
}

function loadContradictions() {
    const container = document.getElementById("discrepancy-findings-container");
    container.innerHTML = `<p class="text-slate-500 animate-pulse">Running Cross-Modal AST Audit Scan...</p>`;

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
                badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-red-950 text-red-300 border border-red-800";
            } else {
                badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800";
            }

            renderDiscrepancyFindings(report.findings || []);
        })
        .catch(err => {
            container.innerHTML = `<p class="text-amber-400">⚠️ No active document to audit. Please click <strong>⚡ 1-Click Demo</strong> first.</p>`;
        });
}

function renderDiscrepancyFindings(findings) {
    const container = document.getElementById("discrepancy-findings-container");
    if (!findings || findings.length === 0) {
        container.innerHTML = `<div class="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center text-slate-400">
            <p>✅ All cross-modal pairs align. No discrepancies detected between prose claims and visual tables.</p>
        </div>`;
        return;
    }

    container.innerHTML = "";
    findings.forEach(f => {
        const isConflict = f.status === "CONTRADICTION";
        const card = document.createElement("div");
        card.className = `p-4 rounded-xl border ${isConflict ? 'bg-red-950/20 border-red-900/60' : 'bg-slate-950 border-slate-800'} flex flex-col gap-2.5`;
        card.innerHTML = `
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-base">${isConflict ? '🚨' : '✅'}</span>
            <span class="font-bold text-xs ${isConflict ? 'text-red-300' : 'text-slate-200'}">${f.title}</span>
          </div>
          <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold ${isConflict ? 'bg-red-950 text-red-400 border border-red-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800'}">
            ${f.status} (${Math.round((f.confidence || 0.9) * 100)}% Conf)
          </span>
        </div>

        <div class="grid grid-cols-2 gap-3 mt-1 text-xs">
          <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <p class="text-[10px] uppercase font-bold text-slate-400 tracking-wider mb-1">Author Prose Claim:</p>
            <p class="text-slate-200 italic">"${f.text_claim}"</p>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
            <p class="text-[10px] uppercase font-bold text-slate-400 tracking-wider mb-1">Raw Visual Chart Fact:</p>
            <p class="text-emerald-300 font-mono">"${f.visual_fact}"</p>
          </div>
        </div>

        ${f.discrepancy_delta && f.discrepancy_delta !== 'None' ? `
        <div class="p-2 rounded-lg bg-red-950/40 border border-red-900/40 text-[11px] text-red-200 font-mono">
          <strong>Audit Delta:</strong> ${f.discrepancy_delta}
        </div>` : ''}
        `;
        container.appendChild(card);
    });
}

// ----------------------------------------------------
// MODAL 3: COMPLIANCE AUDIT
// ----------------------------------------------------
function openComplianceAuditModal() {
    const modal = document.getElementById("compliance-audit-modal");
    modal.classList.remove("hidden");
    loadComplianceAudit();
}

function closeComplianceAuditModal() {
    document.getElementById("compliance-audit-modal").classList.add("hidden");
}

function loadComplianceAudit() {
    const container = document.getElementById("audit-checklist-container");
    container.innerHTML = `<p class="text-slate-500 animate-pulse py-4 text-center">Auditing 7 mandatory controls across AST...</p>`;

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
            if (data.verdict === "CERTIFIED") {
                verdictBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800";
            } else {
                verdictBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-950 text-amber-300 border border-amber-800";
            }

            container.innerHTML = "";
            data.checklist.forEach(item => {
                const row = document.createElement("div");
                row.className = "p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between text-xs";
                row.innerHTML = `
                <div class="flex items-center gap-2.5">
                  <span class="text-base">${item.status === 'PASS' ? '✅' : '⚠️'}</span>
                  <div>
                    <p class="font-bold text-white">${item.control}</p>
                    <p class="text-[11px] text-slate-400">${item.finding}</p>
                  </div>
                </div>
                <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold ${item.status === 'PASS' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-amber-950 text-amber-400 border border-amber-800'}">
                  ${item.status} (${Math.round(item.confidence * 100)}%)
                </span>`;
                container.appendChild(row);
            });
        })
        .catch(err => {
            container.innerHTML = `<p class="text-amber-400 py-4 text-center">⚠️ No active document loaded. Please click <strong>⚡ 1-Click Demo</strong> first.</p>`;
        });
}

// ----------------------------------------------------
// VAULT MANAGEMENT
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
                list.innerHTML = `<p class="text-[11px] text-slate-500 italic py-2 text-center">No documents in vault yet.</p>`;
                return;
            }

            list.innerHTML = "";
            items.forEach(doc => {
                const isActive = doc.is_active;
                const row = document.createElement("div");
                row.className = `p-2 rounded-lg border flex items-center justify-between text-xs cursor-pointer transition ${isActive ? 'bg-emerald-950/40 border-emerald-700/60 text-emerald-200' : 'bg-slate-950 hover:bg-slate-800/60 border-slate-800 text-slate-300'}`;
                row.onclick = () => selectVaultDoc(doc.doc_id);
                row.innerHTML = `
                <div class="flex items-center gap-2 truncate">
                  <span>${isActive ? '🟢' : '📄'}</span>
                  <span class="truncate font-medium">${doc.filename}</span>
                </div>
                <span class="text-[10px] text-slate-500 font-mono">${doc.pages}p</span>`;
                list.appendChild(row);
            });
        })
        .catch(() => {});
}

function selectVaultDoc(docId) {
    fetch(`/vault/select/${docId}`, { method: "POST" })
        .then(r => r.json())
        .then(data => {
            loadVault();
            if (data.classification) {
                updateDocBadge(data.classification.label, data.classification.icon);
            }
        });
}

async function downloadReport(format = "markdown") {
    if (!LAST_QUERY_DATA) {
        alert("Please ask a question first before exporting an audit report.");
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
    } catch (err) {
        console.error("Export error:", err);
    }
}
