// AuditGuard RAG - Frontend Client Logic

let ACTIVE_SOURCES = [];

document.addEventListener("DOMContentLoaded", () => {
    setupUploadHandlers();
    setupCompareToggle();
    checkInitialHealth();
});

function checkInitialHealth() {
    fetch("/health")
        .then(r => r.json())
        .then(data => {
            if (data.doc_loaded) {
                updateDocBadge(data.classification || "General Enterprise Document", "📄");
            }
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
