// AuditGuard - Minimal Glassmorphism Client Logic

let ACTIVE_SOURCES = [];
let LAST_QUERY_DATA = null;
let LAST_INGEST_MS = null;
let PENDING_DELETE_CONV_ID = null;

// Conversation History & UI State Keys
const CONV_STORAGE_KEY = "auditguard_conversations_v2";
const PERF_ENABLED_KEY = "auditguard_perf_notif_enabled";
const SIDEBAR_COLLAPSED_KEY = "auditguard_sidebar_collapsed";
const LAST_GREETING_IDX_KEY = "auditguard_last_greeting_idx";

let CONVERSATIONS = [];
let ACTIVE_CONVERSATION_ID = null;
let PERF_NOTIF_ENABLED = true;
let PERF_POPUP_EXPANDED = false;

// Voice Dictator State
let SPEECH_RECOGNITION = null;
let IS_RECORDING_VOICE = false;

// Randomized Center Landing Greetings (Different on every refresh)
const LANDING_GREETINGS = [
    { title: "Let's Get Started", subtitle: "Attach a document or ask anything below." },
    { title: "Hello there, User", subtitle: "Drop a file or type a question to begin." },
    { title: "What can I help you find?", subtitle: "Attach a document on the left or speak your question." },
    { title: "Ready when you are", subtitle: "Upload a document or start typing below." },
    { title: "Welcome back", subtitle: "Pick up a conversation on the left or ask something new." },
    { title: "How can I help today?", subtitle: "Attach your file and ask any question." }
];

const ALLOWED_HTML_ELEMENTS = [
    "p", "br", "strong", "b", "em", "i", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "hr", "span", "button", "a",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "div"
];
const ALLOWED_HTML_ATTRIBUTES = [
    "class", "type", "value", "title", "href", "target", "rel",
    "role", "tabindex", "aria-label", "aria-labelledby", "data-turn-idx"
];

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
            // Fall through to DOMParser allowlist sanitizer
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
    randomizeLandingGreeting();
    setupIdleBackgroundState();
    initColorSchemeControls();
    initSidebarState();
    initPerfNotificationState();
    initPerfPopupResizeHandle();
    initConversations();
    setupUploadHandlers();
    setupKeyboardShortcuts();
    setupDelegatedCitationClicks();
    resetVoiceDictatorUI();
    loadVault();
    if (window.marked) {
        marked.setOptions({ gfm: true, breaks: true });
    }
});

// ----------------------------------------------------
// RANDOMIZED LANDING GREETING & IDLE AMBIENT SPHERES
// ----------------------------------------------------
function randomizeLandingGreeting() {
    const titleEl = document.getElementById("random-greeting-title");
    const subEl = document.getElementById("random-greeting-subtitle");
    if (!titleEl || !subEl) return;

    let lastIdx = -1;
    try {
        const stored = localStorage.getItem(LAST_GREETING_IDX_KEY);
        if (stored !== null) lastIdx = Number(stored);
    } catch (e) {}

    let nextIdx = Math.floor(Math.random() * LANDING_GREETINGS.length);
    if (LANDING_GREETINGS.length > 1 && nextIdx === lastIdx) {
        nextIdx = (nextIdx + 1) % LANDING_GREETINGS.length;
    }

    try {
        localStorage.setItem(LAST_GREETING_IDX_KEY, String(nextIdx));
    } catch (e) {}

    const chosen = LANDING_GREETINGS[nextIdx];
    titleEl.textContent = chosen.title;
    subEl.textContent = chosen.subtitle;
}

function setupIdleBackgroundState() {
    let idleTimer = null;
    const markActive = () => {
        const landingHero = document.getElementById("landing-hero");
        const onLanding = landingHero && !landingHero.classList.contains("hidden");
        if (!onLanding) {
            document.body.classList.remove("is-idle");
        }
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
            document.body.classList.add("is-idle");
        }, 4000);
    };

    window.addEventListener("mousemove", markActive, { passive: true });
    window.addEventListener("keydown", markActive, { passive: true });
}

// ----------------------------------------------------
// LIGHT MODE & DARK MODE CHOICES (modern-web-guidance)
// ----------------------------------------------------
function getSystemScheme() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function getEffectiveScheme() {
    const meta = document.querySelector('meta[name="color-scheme"]');
    const content = meta ? meta.content : "light dark";
    if (content === "light" || content === "dark") return content;
    return getSystemScheme();
}

function setColorScheme(targetScheme) {
    const meta = document.querySelector('meta[name="color-scheme"]');
    const systemScheme = getSystemScheme();
    let storedValue = null;
    try {
        storedValue = localStorage.getItem("color-scheme");
    } catch (e) {}

    if (storedValue === targetScheme || (storedValue && targetScheme === systemScheme)) {
        try {
            localStorage.removeItem("color-scheme");
        } catch (e) {}
        if (meta) meta.content = "light dark";
        syncColorSchemeUI();
        showToast(`Theme synced with system (${systemScheme})`, "info");
        return;
    }

    try {
        localStorage.setItem("color-scheme", targetScheme);
    } catch (e) {}
    if (meta) meta.content = targetScheme;

    syncColorSchemeUI();
    showToast(`${targetScheme === "dark" ? "Dark" : "Light"} mode active`, "info");
}

function syncColorSchemeUI() {
    const effective = getEffectiveScheme();
    const lightBtn = document.getElementById("theme-btn-light");
    const darkBtn = document.getElementById("theme-btn-dark");
    if (lightBtn && darkBtn) {
        lightBtn.classList.toggle("active", effective === "light");
        darkBtn.classList.toggle("active", effective === "dark");
    }
}

function initColorSchemeControls() {
    syncColorSchemeUI();
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    mql.addEventListener("change", () => {
        syncColorSchemeUI();
    });
    window.addEventListener("storage", (e) => {
        if (e.key === "color-scheme") {
            const meta = document.querySelector('meta[name="color-scheme"]');
            if (meta) meta.content = e.newValue ?? "light dark";
            syncColorSchemeUI();
        }
    });
}

// ----------------------------------------------------
// COLLAPSIBLE LEFT SIDEBAR & CONVERSATIONS TRACKER
// ----------------------------------------------------
function initSidebarState() {
    try {
        const collapsed = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true";
        const sidebar = document.getElementById("left-sidebar");
        if (sidebar && collapsed) {
            sidebar.classList.add("collapsed");
        }
    } catch (e) {}
}

function toggleSidebar() {
    const sidebar = document.getElementById("left-sidebar");
    if (!sidebar) return;
    const isCollapsed = sidebar.classList.toggle("collapsed");
    try {
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(isCollapsed));
    } catch (e) {}
}

function syncNewChatButtonState() {
    const btn = document.getElementById("btn-new-chat");
    if (!btn) return;
    const activeConv = CONVERSATIONS.find(c => c.id === ACTIVE_CONVERSATION_ID);
    const isEmptyNewChat = !activeConv || !Array.isArray(activeConv.turns) || activeConv.turns.length === 0;
    btn.classList.toggle("is-empty-chat", isEmptyNewChat);
    btn.setAttribute("aria-current", isEmptyNewChat ? "page" : "false");
}

function initConversations() {
    try {
        const raw = localStorage.getItem(CONV_STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed) && parsed.length > 0) {
                CONVERSATIONS = parsed.filter(c => c && Array.isArray(c.turns) && c.turns.length > 0);
            }
        }
    } catch (e) {
        CONVERSATIONS = [];
    }

    // Start each page refresh on a clean New Chat landing screen
    const freshConv = createBlankConversation();
    CONVERSATIONS.unshift(freshConv);
    ACTIVE_CONVERSATION_ID = freshConv.id;
    saveConversations();

    renderConversationList();
    restoreActiveConversationView();
}

function createBlankConversation() {
    return {
        id: "conv_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6),
        title: "New Chat",
        createdAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        turns: []
    };
}

function saveConversations() {
    try {
        localStorage.setItem(CONV_STORAGE_KEY, JSON.stringify(CONVERSATIONS.slice(0, 25)));
    } catch (e) {}
}

function startNewConversation() {
    const active = CONVERSATIONS.find(c => c.id === ACTIVE_CONVERSATION_ID);
    if (active && active.turns.length === 0) {
        randomizeLandingGreeting();
        syncNewChatButtonState();
        document.getElementById("query-input")?.focus();
        return;
    }

    const existingBlank = CONVERSATIONS.find(c => Array.isArray(c.turns) && c.turns.length === 0);
    if (existingBlank) {
        ACTIVE_CONVERSATION_ID = existingBlank.id;
    } else {
        const conv = createBlankConversation();
        CONVERSATIONS.unshift(conv);
        ACTIVE_CONVERSATION_ID = conv.id;
    }

    saveConversations();
    randomizeLandingGreeting();
    renderConversationList();
    restoreActiveConversationView();
    document.getElementById("query-input")?.focus();
}

function selectConversation(convId) {
    if (!CONVERSATIONS.some(c => c.id === convId)) return;
    ACTIVE_CONVERSATION_ID = convId;
    renderConversationList();
    restoreActiveConversationView();
}

// ----------------------------------------------------
// DELETE CONVERSATION CONFIRMATION MODAL
// ----------------------------------------------------
function requestDeleteConversation(convId, event) {
    if (event) event.stopPropagation();
    const targetConv = CONVERSATIONS.find(c => c.id === convId);
    if (!targetConv) return;

    PENDING_DELETE_CONV_ID = convId;
    const titleSpan = document.getElementById("delete-dialog-conv-title");
    if (titleSpan) {
        titleSpan.textContent = targetConv.title || "this conversation";
    }

    const dialog = document.getElementById("delete-conv-dialog");
    if (dialog && typeof dialog.showModal === "function") {
        dialog.showModal();
    } else if (window.confirm(`Are you sure you want to remove "${targetConv.title || "this conversation"}"?`)) {
        confirmDeleteConversation();
    }
}

function cancelDeleteConversation() {
    PENDING_DELETE_CONV_ID = null;
    const dialog = document.getElementById("delete-conv-dialog");
    if (dialog && typeof dialog.close === "function") {
        dialog.close();
    }
}

function confirmDeleteConversation() {
    const convId = PENDING_DELETE_CONV_ID;
    PENDING_DELETE_CONV_ID = null;

    const dialog = document.getElementById("delete-conv-dialog");
    if (dialog && typeof dialog.close === "function" && dialog.open) {
        dialog.close();
    }

    if (!convId) return;

    CONVERSATIONS = CONVERSATIONS.filter(c => c.id !== convId);
    if (CONVERSATIONS.length === 0) {
        const fresh = createBlankConversation();
        CONVERSATIONS.push(fresh);
        ACTIVE_CONVERSATION_ID = fresh.id;
    } else if (ACTIVE_CONVERSATION_ID === convId) {
        ACTIVE_CONVERSATION_ID = CONVERSATIONS[0].id;
    }

    saveConversations();
    renderConversationList();
    restoreActiveConversationView();
    showToast("Conversation removed", "info");
}

function recordTurnInConversation(queryData) {
    let conv = CONVERSATIONS.find(c => c.id === ACTIVE_CONVERSATION_ID);
    if (!conv) {
        conv = createBlankConversation();
        CONVERSATIONS.unshift(conv);
        ACTIVE_CONVERSATION_ID = conv.id;
    }

    if (conv.turns.length === 0 && queryData.query) {
        conv.title = queryData.query.length > 38
            ? queryData.query.slice(0, 36) + "…"
            : queryData.query;
    }

    conv.updatedAt = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    conv.turns.unshift({
        timestamp: conv.updatedAt,
        data: queryData
    });

    conv.turns = conv.turns.slice(0, 25);
    saveConversations();
    renderConversationList();
    renderSessionHistoryTurns(conv);
    syncNewChatButtonState();
}

function renderConversationList() {
    const listEl = document.getElementById("conversation-list");
    if (!listEl) return;

    syncNewChatButtonState();
    listEl.innerHTML = "";

    const startedConversations = CONVERSATIONS.filter(c => Array.isArray(c.turns) && c.turns.length > 0);
    if (startedConversations.length === 0) {
        listEl.innerHTML = `<p class="text-[11px] text-theme-muted py-2 text-center">No saved conversations yet.</p>`;
        if (window.lucide) lucide.createIcons();
        return;
    }

    startedConversations.forEach(conv => {
        const isActive = conv.id === ACTIVE_CONVERSATION_ID;
        const row = document.createElement("div");
        row.className = `p-2 rounded-xl flex items-center justify-between gap-2 text-xs cursor-pointer transition ${
            isActive
                ? "glass-elevated border border-cyan-500/50 text-theme-primary shadow-sm"
                : "glass-subtle hover:border-cyan-500/30 text-theme-secondary"
        }`;
        row.onclick = () => selectConversation(conv.id);

        row.innerHTML = `
          <div class="flex items-center gap-2 min-w-0 flex-1">
            <i data-lucide="message-square" class="w-3.5 h-3.5 ${isActive ? 'text-accent-cyan' : 'text-theme-muted'} shrink-0"></i>
            <div class="truncate">
              <p class="truncate font-medium text-[11px] ${isActive ? 'text-accent-cyan font-semibold' : 'text-theme-primary'}">${escapeHtml(conv.title)}</p>
              <p class="text-[10px] text-theme-muted font-mono">${escapeHtml(conv.updatedAt || conv.createdAt)} • ${conv.turns.length} ${conv.turns.length === 1 ? 'msg' : 'msgs'}</p>
            </div>
          </div>
          <button type="button" class="delete-conv-btn p-1 rounded-lg hover:bg-rose-500/20 text-theme-muted hover:text-accent-coral transition shrink-0" title="Remove conversation">
            <i data-lucide="trash-2" class="w-3 h-3"></i>
          </button>
        `;

        const delBtn = row.querySelector(".delete-conv-btn");
        if (delBtn) {
            delBtn.onclick = (e) => requestDeleteConversation(conv.id, e);
        }
        listEl.appendChild(row);
    });

    if (window.lucide) lucide.createIcons();
}

function restoreActiveConversationView() {
    const conv = CONVERSATIONS.find(c => c.id === ACTIVE_CONVERSATION_ID);
    const landingHero = document.getElementById("landing-hero");
    const activeView = document.getElementById("active-response-view");

    syncNewChatButtonState();

    if (!conv || conv.turns.length === 0) {
        landingHero?.classList.remove("hidden");
        activeView?.classList.add("hidden");
        document.body.classList.add("is-idle");
        if (window.lucide) lucide.createIcons();
        return;
    }

    landingHero?.classList.add("hidden");
    activeView?.classList.remove("hidden");
    const latestTurn = conv.turns[0];
    renderSessionHistoryTurns(conv);
    displayQueryResultOnCanvas(latestTurn.data, false);
}

/**
 * Renders all previous messages in the active conversation in chronological order
 * above the latest response card so the user can smoothly scroll back through the entire chat.
 */
function renderSessionHistoryTurns(conv) {
    const container = document.getElementById("session-history-container");
    const list = document.getElementById("session-history-list");
    if (!container || !list) return;

    if (!conv || conv.turns.length <= 1) {
        container.classList.add("hidden");
        list.innerHTML = "";
        return;
    }

    container.classList.remove("hidden");
    list.innerHTML = "";

    // conv.turns[0] is the latest turn; conv.turns.slice(1).reverse() gives oldest -> second-newest
    const olderTurnsChronological = conv.turns.slice(1).slice().reverse();

    olderTurnsChronological.forEach((turn, idx) => {
        const d = turn.data;
        const card = document.createElement("div");
        card.className = "p-5 rounded-3xl glass-panel space-y-3 transition";

        const isRefuse = d.status === "REFUSE";
        const conf = d.evaluation?.confidence ? `${d.evaluation.confidence}% Verified` : "Verified";

        card.innerHTML = `
          <div class="flex items-start justify-between gap-3 pb-2.5 border-b border-white/10">
            <div class="space-y-0.5 min-w-0">
              <span class="text-[10px] font-mono text-theme-muted uppercase tracking-wider">Earlier Message • ${escapeHtml(turn.timestamp || "")}</span>
              <h3 class="text-sm font-semibold text-theme-primary leading-snug">${escapeHtml(d.query)}</h3>
            </div>
            <span class="text-[10px] font-mono px-2.5 py-0.5 rounded-full ${isRefuse ? 'bg-rose-500/15 text-accent-coral border border-rose-500/30' : 'bg-emerald-500/15 text-accent-mint border border-emerald-500/30'} shrink-0">
              ${isRefuse ? 'Not in Document' : conf}
            </span>
          </div>
          <div class="hist-answer-body text-sm text-theme-primary leading-relaxed space-y-2"></div>
          <div class="hist-chart-slot hidden"></div>
          <div class="hist-sources-bar pt-2 border-t border-white/10 flex flex-wrap items-center gap-1.5 hidden"></div>
        `;

        const answerBody = card.querySelector(".hist-answer-body");
        if (isRefuse) {
            answerBody.innerHTML = `<p class="text-xs text-theme-secondary">${escapeHtml(d.answer)}</p>`;
        } else {
            renderAuditAnswerInto(answerBody, d.answer, idx);
        }

        const histChartSlot = card.querySelector(".hist-chart-slot");
        renderVisualDataChart(d, histChartSlot);

        const sourcesBar = card.querySelector(".hist-sources-bar");
        if (Array.isArray(d.sources) && d.sources.length > 0 && sourcesBar) {
            sourcesBar.classList.remove("hidden");
            d.sources.forEach(s => {
                const chipBtn = document.createElement("button");
                chipBtn.type = "button";
                chipBtn.className = "px-2 py-0.5 rounded-lg glass-subtle hover:border-cyan-500/50 text-theme-primary text-[10px] font-mono flex items-center gap-1 transition";
                chipBtn.innerHTML = `<span class="font-bold text-accent-mint">[${Number(s.citation_id)}]</span> <span class="text-theme-muted">Page ${Number(s.page)}</span>`;
                chipBtn.onclick = () => openProofDrawer(s);
                sourcesBar.appendChild(chipBtn);
            });
        }

        // Allow clicking citation anchors inside historical turns
        answerBody.addEventListener("click", (e) => {
            const btn = e.target.closest(".citation-anchor");
            if (btn && btn.value && Array.isArray(d.sources)) {
                const match = d.sources.find(s => s.citation_id === Number(btn.value)) || d.sources[0];
                if (match) openProofDrawer(match);
            }
        });

        list.appendChild(card);
    });

    if (window.lucide) lucide.createIcons();
}

// ----------------------------------------------------
// TOP-RIGHT PERFORMANCE TELEMETRY NOTIFICATION POPUP
// ----------------------------------------------------
function initPerfNotificationState() {
    try {
        const stored = localStorage.getItem(PERF_ENABLED_KEY);
        PERF_NOTIF_ENABLED = stored === null ? true : stored === "true";
    } catch (e) {
        PERF_NOTIF_ENABLED = true;
    }
    syncPerfNotificationButtonUI();
}

function togglePerfNotifications() {
    PERF_NOTIF_ENABLED = !PERF_NOTIF_ENABLED;
    try {
        localStorage.setItem(PERF_ENABLED_KEY, String(PERF_NOTIF_ENABLED));
    } catch (e) {}
    syncPerfNotificationButtonUI();

    if (!PERF_NOTIF_ENABLED) {
        hidePerfNotificationPopup();
        showToast("Performance pop-ups turned off", "info");
    } else {
        showToast("Performance pop-ups turned on", "success");
        if (LAST_QUERY_DATA) {
            openPerfNotificationPopup(false);
        }
    }
}

function syncPerfNotificationButtonUI() {
    const btn = document.getElementById("perf-notif-toggle-btn");
    const dot = document.getElementById("perf-notif-dot");
    const label = document.getElementById("perf-notif-label");
    if (!btn || !dot || !label) return;

    btn.setAttribute("aria-pressed", String(PERF_NOTIF_ENABLED));
    dot.className = `w-2 h-2 rounded-full ${PERF_NOTIF_ENABLED ? "bg-[#00d68f] animate-pulse" : "bg-slate-500"}`;
    label.innerText = PERF_NOTIF_ENABLED ? "Stats: ON" : "Stats: OFF";
}

function populatePerfTelemetryPopup(data) {
    if (!data) return;
    const tel = data.token_telemetry || {};
    const evalObj = data.evaluation || {};

    const retrievalMs = data.retrieval_ms !== undefined ? `${data.retrieval_ms} ms` : "0 ms";
    const totalMs = data.total_ms !== undefined
        ? (data.total_ms < 1000 ? `${data.total_ms} ms` : `${(data.total_ms / 1000).toFixed(2)} s`)
        : retrievalMs;
    const ingestMsVal = tel.indexing_duration_ms || LAST_INGEST_MS;
    const ingestStr = ingestMsVal ? `${Number(ingestMsVal).toFixed(1)} ms` : "-- ms";

    const summaryPill = document.getElementById("perf-summary-pill");
    if (summaryPill) {
        summaryPill.innerText = `${retrievalMs} • ${totalMs}`;
    }

    const setEl = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.innerText = text;
    };

    setEl("perf-stat-retrieval", retrievalMs);
    setEl("perf-stat-total", totalMs);
    setEl("perf-stat-ingest", ingestStr);

    setEl("perf-stat-verdict", evalObj.nli_verdict || data.status || "VERIFIED");
    setEl("perf-stat-confidence", `${evalObj.confidence ?? 0}%`);
    setEl("perf-stat-lexical", `${Math.round((evalObj.lexical_overlap ?? 0) * 100)}%`);

    const verifiedCits = Array.isArray(evalObj.verified_citations)
        ? evalObj.verified_citations.length
        : (data.sources ? data.sources.length : 0);
    setEl("perf-stat-citations", `${verifiedCits}`);

    const promptTok = tel.grounded_prompt_tokens ?? 0;
    const compTok = tel.grounded_completion_tokens ?? 0;
    const totalTok = tel.total_tokens ?? data.estimated_tokens ?? 0;
    const mode = tel.generation_mode || data.generation_mode || "active";

    setEl("perf-stat-mode", mode);
    setEl("perf-stat-tokens-split", `${promptTok} / ${compTok}`);
    setEl("perf-stat-tokens-total", `${totalTok} tok`);
    setEl("perf-stat-ast", `${tel.total_ast_nodes ?? 0} / ${tel.cross_modal_links ?? 0}`);
}

function openPerfNotificationPopup(forceExpand = false) {
    const popover = document.getElementById("perf-telemetry-popover");
    if (!popover) return;

    if (forceExpand && !PERF_POPUP_EXPANDED) {
        setPerfPopupExpanded(true);
    }

    try {
        if (typeof popover.showPopover === "function") {
            if (!popover.matches(":popover-open")) {
                popover.showPopover();
            }
        } else {
            popover.classList.add(":popover-open");
            popover.style.display = "block";
        }
    } catch (e) {
        popover.style.display = "block";
    }
    if (window.lucide) lucide.createIcons();
}

function hidePerfNotificationPopup() {
    const popover = document.getElementById("perf-telemetry-popover");
    if (!popover) return;
    try {
        if (typeof popover.hidePopover === "function" && popover.matches(":popover-open")) {
            popover.hidePopover();
        } else {
            popover.classList.remove(":popover-open");
            popover.style.display = "";
        }
    } catch (e) {
        popover.style.display = "";
    }
}

function setPerfPopupExpanded(expanded) {
    PERF_POPUP_EXPANDED = expanded;
    const popover = document.getElementById("perf-telemetry-popover");
    const details = document.getElementById("perf-details-body");
    const icon = document.getElementById("perf-expand-icon");
    if (!popover || !details) return;

    popover.classList.toggle("perf-expanded", expanded);
    details.classList.toggle("hidden", !expanded);
    if (!expanded) {
        popover.style.height = "";
    }
    if (icon) {
        icon.setAttribute("data-lucide", expanded ? "minimize-2" : "maximize-2");
        if (window.lucide) lucide.createIcons();
    }
}

function togglePerfPopupExpanded() {
    setPerfPopupExpanded(!PERF_POPUP_EXPANDED);
}

function initPerfPopupResizeHandle() {
    const popover = document.getElementById("perf-telemetry-popover");
    const handleBL = document.getElementById("perf-resize-bl");
    if (!popover || !handleBL) return;

    let startX = 0;
    let startY = 0;
    let startW = 0;
    let startH = 0;

    handleBL.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!PERF_POPUP_EXPANDED) {
            setPerfPopupExpanded(true);
        }
        const rect = popover.getBoundingClientRect();
        startX = e.clientX;
        startY = e.clientY;
        startW = rect.width;
        startH = rect.height;
        popover.classList.add("is-resizing");
        handleBL.setPointerCapture(e.pointerId);
    });

    handleBL.addEventListener("pointermove", (e) => {
        if (!popover.classList.contains("is-resizing")) return;
        const deltaW = startX - e.clientX;
        const deltaH = e.clientY - startY;
        const newW = Math.max(280, Math.min(window.innerWidth - 32, startW + deltaW));
        const newH = Math.max(110, Math.min(window.innerHeight - 90, startH + deltaH));
        popover.style.width = `${newW}px`;
        popover.style.height = `${newH}px`;
    });

    const stopResize = (e) => {
        if (popover.classList.contains("is-resizing")) {
            popover.classList.remove("is-resizing");
            try {
                handleBL.releasePointerCapture(e.pointerId);
            } catch (_) {}
        }
    };

    handleBL.addEventListener("pointerup", stopResize);
    handleBL.addEventListener("pointercancel", stopResize);
}

// ----------------------------------------------------
// VOICE DICTATOR (DYNAMIC ISLAND RIGHT SIDE)
// ----------------------------------------------------
function setIslandStatus(text, isRecording = false) {
    const pill = document.getElementById("island-status-pill");
    const statusText = document.getElementById("island-status-text");
    const dot = document.getElementById("island-status-dot");
    if (!pill || !statusText || !dot) return;

    if (!text) {
        pill.classList.add("hidden");
        return;
    }

    statusText.innerText = text;
    dot.className = `w-2 h-2 rounded-full ${isRecording ? "bg-[#ff4d4f] animate-ping" : "bg-[#00e5ff] animate-pulse"}`;
    pill.classList.remove("hidden");
}

/**
 * Guarantees the voice dictator button returns to its normal microphone state
 * whenever dictation is not actively recording (never stuck on a mute icon).
 */
function resetVoiceDictatorUI() {
    IS_RECORDING_VOICE = false;
    const island = document.getElementById("dynamic-island");
    const btn = document.getElementById("btn-voice-dictate");
    island?.classList.remove("voice-recording");
    if (btn) {
        btn.classList.remove("bg-rose-500/20", "text-accent-coral", "border-rose-500/40");
        btn.setAttribute("aria-pressed", "false");
        btn.setAttribute("title", "Voice Dictation");
        btn.innerHTML = `<i id="voice-dictate-icon" data-lucide="mic" class="w-4 h-4"></i>`;
    }
    setIslandStatus(null);
    if (window.lucide) lucide.createIcons();
}

function setVoiceDictatorActiveUI() {
    IS_RECORDING_VOICE = true;
    const island = document.getElementById("dynamic-island");
    const btn = document.getElementById("btn-voice-dictate");
    island?.classList.add("voice-recording");
    if (btn) {
        btn.classList.add("bg-rose-500/20", "text-accent-coral", "border-rose-500/40");
        btn.setAttribute("aria-pressed", "true");
        btn.setAttribute("title", "Listening... Click to finish");
        btn.innerHTML = `<i id="voice-dictate-icon" data-lucide="mic" class="w-4 h-4 animate-pulse"></i>`;
    }
    setIslandStatus("Listening...", true);
    if (window.lucide) lucide.createIcons();
}

function toggleVoiceDictation() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
        showToast("Voice dictation is not supported in this browser.", "error");
        resetVoiceDictatorUI();
        return;
    }

    const input = document.getElementById("query-input");

    if (IS_RECORDING_VOICE && SPEECH_RECOGNITION) {
        try {
            SPEECH_RECOGNITION.stop();
        } catch (e) {}
        resetVoiceDictatorUI();
        return;
    }

    SPEECH_RECOGNITION = new SpeechRec();
    SPEECH_RECOGNITION.lang = "en-US";
    SPEECH_RECOGNITION.continuous = false;
    SPEECH_RECOGNITION.interimResults = true;

    const baseText = input && input.value.trim() ? input.value.trim() + " " : "";

    SPEECH_RECOGNITION.onstart = () => {
        setVoiceDictatorActiveUI();
    };

    SPEECH_RECOGNITION.onresult = (event) => {
        let interim = "";
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interim += transcript;
            }
        }
        if (input) {
            input.value = (baseText + finalTranscript + interim).trimStart();
            input.focus();
        }
    };

    SPEECH_RECOGNITION.onerror = (event) => {
        resetVoiceDictatorUI();
        if (event.error && event.error !== "aborted" && event.error !== "no-speech") {
            showToast(`Voice input: ${event.error}`, "info");
        }
    };

    SPEECH_RECOGNITION.onend = () => {
        resetVoiceDictatorUI();
    };

    try {
        SPEECH_RECOGNITION.start();
    } catch (e) {
        resetVoiceDictatorUI();
        showToast("Could not start microphone.", "error");
    }
}

// ----------------------------------------------------
// DOCUMENT ATTACHMENT (DYNAMIC ISLAND LEFT SIDE & DRAG-DROP)
// ----------------------------------------------------
function triggerDocumentAttach() {
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.click();
}

function setupUploadHandlers() {
    const fileInput = document.getElementById("file-input");
    const island = document.getElementById("dynamic-island");

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
                fileInput.value = "";
            }
        });
    }

    window.addEventListener("dragover", (e) => {
        e.preventDefault();
        island?.classList.add("drag-active");
    });

    window.addEventListener("dragleave", (e) => {
        if (e.relatedTarget === null) {
            island?.classList.remove("drag-active");
        }
    });

    window.addEventListener("drop", (e) => {
        e.preventDefault();
        island?.classList.remove("drag-active");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

function handleFileUpload(file) {
    const statusBox = document.getElementById("upload-status");
    const filenameEl = document.getElementById("upload-filename");
    const chunksEl = document.getElementById("upload-chunks");

    if (statusBox && filenameEl && chunksEl) {
        statusBox.classList.remove("hidden");
        filenameEl.innerText = file.name;
        chunksEl.innerText = "Processing...";
    }
    setIslandStatus(`Attaching ${file.name}...`, false);

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
        setIslandStatus(null);
        LAST_INGEST_MS = data.indexing_duration_ms;
        if (chunksEl) {
            chunksEl.innerText = "Ready";
        }
        showToast(`Attached ${file.name}`, "success");
        loadVault();
    })
    .catch(err => {
        setIslandStatus(null);
        if (chunksEl) chunksEl.innerText = "Error";
        showToast(err.message || "Failed to attach file", "error");
        console.error(err);
    });
}

// ----------------------------------------------------
// KEYBOARD SHORTCUTS, CITATION CLICKS & TOASTS
// ----------------------------------------------------
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
        }
    });
}

function showToast(message, type = "success") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.setAttribute("popover", "manual");
    const borderClass = type === "error" ? "border-[#ff4d4f]/60 text-accent-coral" :
                        type === "info"  ? "border-cyan-500/60 text-accent-cyan" :
                                           "border-emerald-500/60 text-accent-mint";

    const icon = type === "error" ? "alert-circle" :
                 type === "info"  ? "info" : "check-circle";

    toast.className = `glass-elevated flex items-center gap-2.5 px-4 py-2.5 rounded-2xl border shadow-xl text-xs font-medium pointer-events-auto transition-all duration-300 ${borderClass}`;
    toast.innerHTML = `
      <i data-lucide="${icon}" class="w-4 h-4 shrink-0"></i>
      <span class="text-theme-primary">${escapeHtml(message)}</span>
      <button type="button" popovertargetaction="hide" class="ml-1 text-theme-muted hover:text-theme-primary">
        <i data-lucide="x" class="w-3 h-3"></i>
      </button>
    `;

    container.appendChild(toast);
    try {
        if (typeof toast.showPopover === "function") toast.showPopover();
    } catch (e) {}

    const closeBtn = toast.querySelector("button");
    if (closeBtn) {
        closeBtn.onclick = () => toast.remove();
    }
    if (window.lucide) lucide.createIcons();

    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// ----------------------------------------------------
// MARKDOWN RENDERING, TABLES & VISUAL CHARTING
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

    // Wrap Markdown tables in accessible, responsive scrollable containers (modern-web-guidance: responsive-table)
    html = html
        .replace(/<table>/gi, '<div tabindex="0" role="region" aria-label="Structured Data Table" class="table-wrapper"><table>')
        .replace(/<\/table>/gi, "</table></div>");

    // Convert [1] citation references into interactive pills
    html = html.replace(/\[(\d+)\]/g, (match, p1) => {
        return `<button type="button" value="${p1}" class="citation-anchor" title="View Source [${p1}]"><span>[${p1}]</span></button>`;
    });

    safeSetHTML(container, html);
}

/**
 * Extracts structured numerical/percentage metrics from the AI response or retrieved table chunks
 * and renders a clean visual bar chart / comparison graph when numerical data is present.
 */
function extractChartMetrics(data) {
    if (!data || data.status === "REFUSE") return [];
    const text = String(data.answer || "");
    const items = [];
    const seenLabels = new Set();

    // 1. Parse Markdown table rows for numeric or percentage values
    const lines = text.split("\n");
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("|") || /^\|[\s\-:|]+\|$/.test(trimmed)) continue;
        const cells = trimmed.split("|").map(c => c.trim()).filter(Boolean);
        if (cells.length >= 2) {
            const label = cells[0].replace(/\*\*/g, "").replace(/\[\d+\]/g, "").trim();
            if (!label || /^(metric|item|category|parameter|clause|section|feature|aspect)$/i.test(label)) continue;

            for (let i = 1; i < cells.length; i++) {
                const cellText = cells[i].replace(/\[\d+\]/g, "").trim();
                const pctMatch = cellText.match(/(\d+(?:\.\d+)?)\s*%/);
                const numMatch = cellText.match(/\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(ms|s|min|minutes|hours|hrs|days|x|k|m|gb|tb|tok)?/i);

                if (pctMatch) {
                    const val = parseFloat(pctMatch[1]);
                    const colSuffix = cells.length > 2 ? ` (Col ${i})` : "";
                    const key = `${label}${colSuffix}`;
                    if (!seenLabels.has(key) && !isNaN(val)) {
                        seenLabels.add(key);
                        items.push({ label: key, value: val, display: `${val}%`, unit: "%" });
                    }
                } else if (numMatch && !/^(page|section|table|fig)/i.test(cellText)) {
                    const rawNum = parseFloat(numMatch[1].replace(/,/g, ""));
                    const unit = (numMatch[2] || "").trim();
                    const hasDollar = cellText.includes("$");
                    const colSuffix = cells.length > 2 ? ` (${i === 1 ? "Claim/Target" : "Actual/Req"})` : "";
                    const key = `${label}${colSuffix}`;
                    if (!seenLabels.has(key) && !isNaN(rawNum) && rawNum > 0) {
                        seenLabels.add(key);
                        items.push({
                            label: key,
                            value: rawNum,
                            display: `${hasDollar ? "$" : ""}${numMatch[1]}${unit ? " " + unit : ""}`,
                            unit: unit || (hasDollar ? "$" : "")
                        });
                    }
                }
            }
        }
    }

    // 2. Fallback: extract labeled percentages or key figures from prose if no table metrics found
    if (items.length < 2) {
        const pctRegex = /([A-Za-z][A-Za-z0-9\s\-/()]{2,28}?)\s*(?:is|of|at|:|-|–)?\s*(\d+(?:\.\d+)?)\s*%/g;
        let m;
        while ((m = pctRegex.exec(text)) !== null && items.length < 6) {
            const label = m[1].trim().replace(/^(and|or|the|a|an|with|to|for)\s+/i, "");
            const val = parseFloat(m[2]);
            if (label.length >= 3 && !seenLabels.has(label) && !isNaN(val) && val <= 100) {
                seenLabels.add(label);
                items.push({ label, value: val, display: `${val}%`, unit: "%" });
            }
        }
    }

    return items.slice(0, 6);
}

function renderVisualDataChart(data, targetEl) {
    const container = targetEl || document.getElementById("visual-chart-container");
    if (!container) return;

    const metrics = extractChartMetrics(data);
    if (metrics.length < 2) {
        container.classList.add("hidden");
        container.innerHTML = "";
        return;
    }

    const maxVal = Math.max(...metrics.map(m => m.value), 1);
    const allPercentages = metrics.every(m => m.unit === "%" && m.value <= 100);
    const scaleMax = allPercentages ? 100 : maxVal * 1.08;

    let barsHtml = `
      <div class="visual-chart-card">
        <div class="flex items-center justify-between gap-2 mb-3">
          <div class="flex items-center gap-2">
            <span class="text-[11px] font-bold uppercase tracking-wider text-accent-cyan">Visual Metric Comparison</span>
          </div>
          <span class="text-[10px] font-mono text-theme-muted">${metrics.length} data points extracted</span>
        </div>
        <div class="space-y-2.5">
    `;

    metrics.forEach((m, idx) => {
        const widthPct = Math.max(6, Math.min(100, Math.round((m.value / scaleMax) * 100)));
        const barColorClass = idx % 2 === 0
            ? "from-blue-600 to-cyan-400"
            : "from-emerald-600 to-teal-400";

        barsHtml += `
          <div class="space-y-1">
            <div class="flex items-center justify-between text-xs gap-2">
              <span class="font-medium text-theme-primary truncate">${escapeHtml(m.label)}</span>
              <span class="font-mono font-semibold text-accent-cyan shrink-0">${escapeHtml(m.display)}</span>
            </div>
            <div class="w-full h-2.5 rounded-full bg-slate-500/15 overflow-hidden">
              <div class="h-full rounded-full bg-gradient-to-r ${barColorClass}" style="width: ${widthPct}%"></div>
            </div>
          </div>
        `;
    });

    barsHtml += `
        </div>
      </div>
    `;

    container.classList.remove("hidden");
    container.innerHTML = barsHtml;
}

// ----------------------------------------------------
// QUERY EXECUTION
// ----------------------------------------------------
async function submitQuery() {
    const queryInput = document.getElementById("query-input");
    const query = queryInput.value.trim();
    if (!query) return;

    if (IS_RECORDING_VOICE && SPEECH_RECOGNITION) {
        try {
            SPEECH_RECOGNITION.stop();
        } catch (e) {}
    }
    resetVoiceDictatorUI();

    const landingHero = document.getElementById("landing-hero");
    const activeView = document.getElementById("active-response-view");
    landingHero?.classList.add("hidden");
    activeView?.classList.remove("hidden");

    const btn = document.getElementById("btn-submit");
    btn.disabled = true;
    const btnOriginalHtml = `<i data-lucide="arrow-up" class="w-4 h-4"></i>`;
    btn.innerHTML = `<span class="animate-spin inline-block text-xs">⏳</span>`;

    const answerContainer = document.getElementById("answer-container");
    const chartContainer = document.getElementById("visual-chart-container");
    const badge = document.getElementById("grounding-badge");
    const heading = document.getElementById("active-query-heading");
    const chipsContainer = document.getElementById("citation-chips");
    const chipsList = document.getElementById("chips-list");
    const visualStrip = document.getElementById("linked-visuals-strip");

    if (heading) heading.innerText = query;
    answerContainer.innerHTML = `
      <div class="space-y-3 py-4 animate-pulse">
        <div class="h-4 bg-cyan-500/20 rounded w-3/4"></div>
        <div class="h-3.5 bg-slate-500/20 rounded w-full"></div>
        <div class="h-3.5 bg-slate-500/20 rounded w-5/6"></div>
      </div>`;

    if (chartContainer) {
        chartContainer.classList.add("hidden");
        chartContainer.innerHTML = "";
    }
    badge.innerText = "Thinking...";
    badge.className = "text-[11px] px-2.5 py-0.5 rounded-full glass-subtle text-theme-secondary font-mono font-semibold shrink-0";
    chipsContainer.classList.add("hidden");
    chipsList.innerHTML = "";
    visualStrip.classList.add("hidden");

    // Scroll latest turn wrapper into view smoothly
    document.getElementById("latest-turn-wrapper")?.scrollIntoView({ behavior: "smooth", block: "nearest" });

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, include_baseline: false })
        });

        const data = await response.json();
        btn.disabled = false;
        btn.innerHTML = btnOriginalHtml;
        if (window.lucide) lucide.createIcons();

        if (!response.ok) {
            badge.innerText = "Notice";
            answerContainer.innerHTML = `<div class="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/40 text-accent-coral text-xs">
                <p>${escapeHtml(data.detail || "Please attach a document first using the paperclip button below.")}</p>
            </div>`;
            if (window.lucide) lucide.createIcons();
            return;
        }

        queryInput.value = "";
        recordTurnInConversation(data);
        displayQueryResultOnCanvas(data, true);

    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = btnOriginalHtml;
        if (window.lucide) lucide.createIcons();
        answerContainer.innerHTML = `<p class="text-accent-coral text-xs">Connection error: ${escapeHtml(err.message || err)}</p>`;
    }
}

function displayQueryResultOnCanvas(data, triggerNotification = true) {
    if (!data) return;
    LAST_QUERY_DATA = data;

    const landingHero = document.getElementById("landing-hero");
    const activeView = document.getElementById("active-response-view");
    landingHero?.classList.add("hidden");
    activeView?.classList.remove("hidden");

    const answerContainer = document.getElementById("answer-container");
    const chartContainer = document.getElementById("visual-chart-container");
    const badge = document.getElementById("grounding-badge");
    const heading = document.getElementById("active-query-heading");
    const chipsContainer = document.getElementById("citation-chips");
    const chipsList = document.getElementById("chips-list");
    const visualStrip = document.getElementById("linked-visuals-strip");

    if (heading && data.query) {
        heading.innerText = data.query;
    }

    if (data.status === "REFUSE") {
        badge.innerText = "Not in Document";
        badge.className = "text-[11px] px-2.5 py-0.5 rounded-full bg-rose-500/15 text-accent-coral font-mono font-semibold border border-rose-500/30 shrink-0";
        answerContainer.innerHTML = `
          <div class="p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl text-theme-primary">
            <p class="text-xs leading-relaxed">${escapeHtml(data.answer)}</p>
          </div>`;
        if (chartContainer) {
            chartContainer.classList.add("hidden");
            chartContainer.innerHTML = "";
        }
    } else {
        const conf = data.evaluation?.confidence ? `${data.evaluation.confidence}% Verified` : "Verified";
        badge.innerText = conf;
        badge.className = "text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-accent-mint font-mono font-semibold border border-emerald-500/35 shrink-0";
        renderAuditAnswerInto(answerContainer, data.answer);
        renderVisualDataChart(data, chartContainer);
    }

    ACTIVE_SOURCES = data.sources || [];
    chipsList.innerHTML = "";

    if (ACTIVE_SOURCES.length > 0) {
        const topSource = ACTIVE_SOURCES[0];
        if (topSource.linked_visuals && topSource.linked_visuals.length > 0) {
            const vis = topSource.linked_visuals[0];
            if (vis.image_url) {
                document.getElementById("linked-visual-thumb").src = vis.image_url;
            }
            document.getElementById("linked-visual-title").innerText = vis.title || "Linked Table / Figure";
            visualStrip.classList.remove("hidden");
        } else {
            visualStrip.classList.add("hidden");
        }

        chipsContainer.classList.remove("hidden");
        ACTIVE_SOURCES.forEach(s => {
            const chipBtn = document.createElement("button");
            chipBtn.className = "px-2.5 py-1 rounded-xl glass-subtle hover:border-cyan-500/50 text-theme-primary text-[11px] font-mono flex items-center gap-1.5 transition";
            chipBtn.innerHTML = `<span class="font-bold text-accent-mint">[${Number(s.citation_id)}]</span> <span class="text-theme-muted">Page ${Number(s.page)}</span>`;
            chipBtn.onclick = () => openProofDrawer(s);
            chipsList.appendChild(chipBtn);
        });
    } else {
        visualStrip.classList.add("hidden");
        // Still show export bar even if refused or 0 sources so user can export any response
        chipsContainer.classList.remove("hidden");
    }

    populatePerfTelemetryPopup(data);
    if (triggerNotification && PERF_NOTIF_ENABLED) {
        openPerfNotificationPopup(false);
    }

    if (window.lucide) lucide.createIcons();
    if (triggerNotification) {
        document.getElementById("latest-turn-wrapper")?.scrollIntoView({ behavior: "smooth", block: "end" });
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
// DOCUMENTS LIST (SIDEBAR)
// ----------------------------------------------------
function loadVault() {
    fetch("/vault")
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById("vault-list");
            const badge = document.getElementById("vault-count-badge");
            const items = data.vault || [];
            if (badge) badge.innerText = `${items.length}`;
            if (!list) return;

            if (items.length === 0) {
                list.innerHTML = `<p class="text-[11px] text-theme-muted py-2 text-center">No documents attached yet.</p>`;
                return;
            }

            list.innerHTML = "";
            items.forEach(doc => {
                const isActive = doc.is_active;
                if (isActive && doc.indexing_duration_ms) {
                    LAST_INGEST_MS = doc.indexing_duration_ms;
                }
                const row = document.createElement("div");
                row.className = `p-2 rounded-xl flex items-center justify-between text-xs transition-all ${
                    isActive
                        ? "bg-emerald-500/15 border border-emerald-500/45 shadow-sm"
                        : "glass-subtle hover:border-cyan-500/30 cursor-pointer"
                }`;

                row.onclick = () => {
                    if (!isActive) selectVaultDoc(doc.doc_id, doc.filename);
                };

                row.innerHTML = `
                <div class="flex items-center gap-2 truncate">
                  <i data-lucide="file-text" class="w-3.5 h-3.5 ${isActive ? 'text-accent-mint' : 'text-theme-muted'} shrink-0"></i>
                  <span class="truncate font-medium text-[11px] ${isActive ? 'text-accent-mint font-semibold' : 'text-theme-primary'}">${escapeHtml(doc.filename)}</span>
                </div>
                ${isActive
                  ? `<span class="w-2 h-2 rounded-full bg-[#00d68f] shrink-0" title="Active"></span>`
                  : `<span class="text-[10px] text-theme-muted shrink-0">Select</span>`
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
        .then(() => {
            loadVault();
            showToast(`Selected ${filename}`, "info");
        });
}

// ----------------------------------------------------
// MULTI-FORMAT EXPORT REPORT (PDF, WORD, CSV, MD, JSON)
// ----------------------------------------------------
async function downloadReport(format = "markdown") {
    if (!LAST_QUERY_DATA) {
        showToast("Ask a question first before exporting.", "info");
        return;
    }

    const defaultFilenames = {
        pdf: "audit_report.pdf",
        docx: "audit_report.docx",
        word: "audit_report.docx",
        csv: "audit_report.csv",
        json: "audit_trail.json",
        markdown: "audit_report.md"
    };

    try {
        const payload = {
            query: LAST_QUERY_DATA.query,
            answer: LAST_QUERY_DATA.answer,
            sources: LAST_QUERY_DATA.sources || [],
            classification: LAST_QUERY_DATA.classification || {},
            evaluation: LAST_QUERY_DATA.evaluation || {},
            baseline: "",
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
        let filename = defaultFilenames[format] || "audit_report.md";
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
        showToast("Failed to export report", "error");
    }
}
