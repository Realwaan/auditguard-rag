// AuditGuard RAG - Frontend Client Logic

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
    chunksEl.innerText = "Indexing...";

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
        chunksEl.innerText = `${data.total_chunks} chunks ready`;
        if (data.classification) {
            updateDocBadge(data.classification.label, data.classification.icon);
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
    btn.innerHTML = `<span class="animate-spin">⏳</span> Loading...`;
    btn.disabled = true;

    fetch("/load-sample", { method: "POST" })
        .then(r => r.json())
        .then(data => {
            btn.innerHTML = `✅ Demo Ready`;
            if (data.classification) {
                updateDocBadge(data.classification.label, data.classification.icon);
            }
            const statusBox = document.getElementById("upload-status");
            statusBox.classList.remove("hidden");
            document.getElementById("upload-filename").innerText = data.filename;
            document.getElementById("upload-chunks").innerText = `${data.total_chunks} chunks ready`;
            loadVault();
            setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 2000);
        })
        .catch(err => {
            btn.innerHTML = `❌ Failed`;
            setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 2000);
            console.error(err);
        });
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

    answerContainer.innerHTML = `<p class="text-emerald-400 animate-pulse">Running Hybrid Search & Synthesizing Grounded Proof...</p>`;
    baselineContainer.innerHTML = `<p class="text-slate-400 animate-pulse">Requesting plain baseline...</p>`;
    badge.innerText = "Evaluating...";
    chipsContainer.classList.add("hidden");
    chipsList.innerHTML = "";

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
            
            // Format citations in answer text: replace [1] with interactive badge
            let formattedAnswer = data.answer.replace(/\[(\d+)\]/g, (match, p1) => {
                return `<button onclick="openCitationByIndex(${p1})" class="inline-flex items-center px-1.5 py-0.2 mx-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono font-bold hover:bg-emerald-500 hover:text-white transition cursor-pointer">[${p1}]</button>`;
            });

            answerContainer.innerHTML = `<div class="whitespace-pre-wrap">${formattedAnswer}</div>`;
        }

        // 2. Proof Drawer Source Chips
        ACTIVE_SOURCES = data.sources || [];
        if (ACTIVE_SOURCES.length > 0) {
            chipsContainer.classList.remove("hidden");
            ACTIVE_SOURCES.forEach(s => {
                const chipBtn = document.createElement("button");
                chipBtn.className = "px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[11px] font-mono flex items-center gap-1 transition";
                chipBtn.innerHTML = `<span>[${s.citation_id}]</span> <span class="text-slate-400">Page ${s.page}</span>`;
                chipBtn.onclick = () => openProofDrawer(s);
                chipsList.appendChild(chipBtn);
            });
        }

        // 3. Plain Baseline Container (Hallucination Radar)
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
        alert("Error exporting audit report: " + err);
    }
}

// Enterprise Document Vault Functions
function loadVault() {
    fetch("/vault")
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById("vault-list");
            const badge = document.getElementById("vault-count-badge");
            if (!list) return;
            const items = data.vault || [];
            if (badge) badge.innerText = `${items.length} Docs`;
            if (items.length === 0) {
                list.innerHTML = `<p class="text-[11px] text-slate-500 italic py-2 text-center">No documents in vault yet. Upload or load sample.</p>`;
                return;
            }
            list.innerHTML = "";
            items.forEach(doc => {
                const row = document.createElement("div");
                row.className = `p-2 rounded-lg border text-xs flex items-center justify-between cursor-pointer transition ${
                    doc.is_active ? 'bg-emerald-950/40 border-emerald-500/60 text-emerald-200' : 'bg-slate-950 hover:bg-slate-800 border-slate-800 text-slate-300'
                }`;
                row.onclick = () => selectVaultDoc(doc.doc_id);
                row.innerHTML = `
                    <div class="truncate max-w-[180px]">
                        <p class="font-medium truncate">${doc.filename}</p>
                        <p class="text-[10px] text-slate-400 font-mono">${doc.total_chunks} chunks • ${doc.uploaded_at}</p>
                    </div>
                    <span class="text-[10px] px-1.5 py-0.5 rounded ${doc.is_active ? 'bg-emerald-500 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400 font-mono'}">
                        ${doc.is_active ? 'ACTIVE' : 'SELECT'}
                    </span>
                `;
                list.appendChild(row);
            });
        })
        .catch(() => {});
}

function selectVaultDoc(docId) {
    fetch(`/vault/select/${docId}`, { method: "POST" })
        .then(r => r.json())
        .then(data => {
            if (data.ok) {
                if (data.classification) {
                    updateDocBadge(data.classification.label, data.classification.icon);
                }
                loadVault();
            }
        });
}

// Enterprise Compliance Audit Scanner Modal Logic
let CURRENT_AUDIT_DATA = null;

function openComplianceAuditModal() {
    const modal = document.getElementById("compliance-modal");
    modal.classList.remove("hidden");
    
    const valueEl = document.getElementById("audit-score-value");
    const badgeEl = document.getElementById("audit-verdict-badge");
    const ratioEl = document.getElementById("audit-ratio-text");
    const container = document.getElementById("audit-checklist-container");

    valueEl.innerText = "--%";
    badgeEl.innerText = "Scanning in Memory...";
    badgeEl.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-slate-800 text-slate-300 border border-slate-700 animate-pulse";
    ratioEl.innerText = "Checking 6 controls...";
    container.innerHTML = `<p class="text-xs text-slate-400 animate-pulse py-4 text-center">Executing Vector & BM25 Multi-Point Compliance Audit in RAM...</p>`;

    fetch("/compliance-audit")
        .then(r => {
            if (!r.ok) throw new Error("Please upload or load a document first.");
            return r.json();
        })
        .then(data => {
            CURRENT_AUDIT_DATA = data;
            document.getElementById("modal-audit-subtitle").innerText = `Audit for: ${data.filename} (${data.classification ? data.classification.label : 'Enterprise Contract'})`;
            valueEl.innerText = `${data.overall_score}%`;
            badgeEl.innerText = data.verdict;
            ratioEl.innerText = `${data.passed_count} of ${data.total_checks} Controls Verified`;

            if (data.overall_score >= 80) {
                badgeEl.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-950 text-emerald-300 border border-emerald-800";
                valueEl.className = "text-3xl font-black text-emerald-400 font-mono mt-0.5";
            } else {
                badgeEl.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-950 text-amber-300 border border-amber-800";
                valueEl.className = "text-3xl font-black text-amber-400 font-mono mt-0.5";
            }

            container.innerHTML = "";
            data.checklist.forEach(item => {
                const isPass = item.status === "PASS";
                const row = document.createElement("div");
                row.className = `p-3 rounded-xl border flex items-start justify-between gap-3 transition ${
                    isPass ? 'bg-slate-950/80 border-slate-800 hover:border-emerald-500/50' : 'bg-red-950/20 border-red-900/40'
                }`;
                row.innerHTML = `
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-1">
                            <span class="text-[10px] font-mono px-1.5 py-0.2 rounded ${isPass ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-red-950 text-red-400 border border-red-800'}">${item.id}</span>
                            <span class="text-xs font-bold text-white">${item.title}</span>
                            <span class="text-[10px] text-slate-400">(${item.category})</span>
                        </div>
                        <p class="text-[11px] ${isPass ? 'text-slate-300' : 'text-red-300 font-medium'}">${item.finding}</p>
                        ${isPass ? `<p class="text-[10px] text-slate-500 mt-1 italic font-mono truncate">"${item.clause_excerpt}"</p>` : ''}
                    </div>
                    <div class="text-right flex flex-col items-end gap-1">
                        <span class="text-[10px] px-2 py-0.5 rounded font-bold font-mono ${isPass ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-red-500/20 text-red-300 border border-red-500/40'}">${item.badge}</span>
                        ${isPass ? `
                            <button onclick='inspectAuditItem(${JSON.stringify(item)})' class="text-[10px] text-emerald-400 hover:text-emerald-300 font-semibold underline mt-1 flex items-center gap-1 cursor-pointer">
                                <span>Inspect Page ${item.page}</span> →
                            </button>
                        ` : `
                            <span class="text-[9px] text-red-400 font-mono">RISK: ${item.criticality}</span>
                        `}
                    </div>
                `;
                container.appendChild(row);
            });
            if (window.lucide) lucide.createIcons();
        })
        .catch(err => {
            badgeEl.innerText = "Error";
            badgeEl.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-red-950 text-red-400 border border-red-800";
            container.innerHTML = `<p class="text-xs text-red-400 py-4 text-center">❌ ${err.message}</p>`;
        });
}

function closeComplianceAuditModal() {
    document.getElementById("compliance-modal").classList.add("hidden");
}

function inspectAuditItem(item) {
    closeComplianceAuditModal();
    openProofDrawer({
        filename: document.getElementById("modal-audit-subtitle") ? document.getElementById("modal-audit-subtitle").innerText : "Document",
        page: item.page,
        parent_text: item.full_clause || item.clause_excerpt,
        image_url: item.image_url
    });
}
