/* ═══════════════════════════════════════════════════════════
   SkinAI Dashboard — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

const API = "";  // Same origin

// ── State ─────────────────────────────────────────────────
let currentPage = 1;
let totalPages = 1;
let selectedNodeIdx = null;

// ── DOM References ────────────────────────────────────────
const patientGrid = document.getElementById("patient-grid");
const filterDiagnosis = document.getElementById("filter-diagnosis");
const filterSplit = document.getElementById("filter-split");
const prevPageBtn = document.getElementById("prev-page");
const nextPageBtn = document.getElementById("next-page");
const pageInfo = document.getElementById("page-info");
const emptyState = document.getElementById("empty-state");
const loadingState = document.getElementById("loading-state");
const resultsContent = document.getElementById("results-content");

// ── Label Colors ──────────────────────────────────────────
const LABEL_COLORS = {
    "akiec": "#ef4444",
    "bcc":   "#f97316",
    "bkl":   "#eab308",
    "df":    "#22c55e",
    "mel":   "#ef4444",
    "nv":    "#3b82f6",
    "vasc":  "#a855f7",
};

// ══════════ Initialisation ══════════

document.addEventListener("DOMContentLoaded", () => {
    loadStats();
    loadPatients();

    filterDiagnosis.addEventListener("change", () => { currentPage = 1; loadPatients(); });
    filterSplit.addEventListener("change", () => { currentPage = 1; loadPatients(); });
    prevPageBtn.addEventListener("click", () => { if (currentPage > 1) { currentPage--; loadPatients(); } });
    nextPageBtn.addEventListener("click", () => { if (currentPage < totalPages) { currentPage++; loadPatients(); } });
});


// ══════════ API Calls ══════════

async function loadStats() {
    try {
        const resp = await fetch(`${API}/api/stats`);
        const data = await resp.json();
        document.querySelector("#stat-patients .stat-val").textContent = data.total_patients.toLocaleString();
        document.querySelector("#stat-classes .stat-val").textContent = data.num_classes;
    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

async function loadPatients() {
    const diagnosis = filterDiagnosis.value;
    const split = filterSplit.value;

    let url = `${API}/api/patients?page=${currentPage}&page_size=48`;
    if (diagnosis) url += `&diagnosis=${diagnosis}`;
    if (split) url += `&split=${split}`;

    try {
        const resp = await fetch(url);
        const data = await resp.json();
        totalPages = data.total_pages;
        renderPatientGrid(data.patients);
        updatePagination(data);
    } catch (err) {
        console.error("Failed to load patients:", err);
        patientGrid.innerHTML = `<p style="color: var(--text-muted); grid-column: 1/-1; text-align: center; padding: 40px;">Failed to load patients. Is the server running?</p>`;
    }
}

async function predictAndExplain(nodeIdx) {
    showLoading();
    try {
        const resp = await fetch(`${API}/api/predict/${nodeIdx}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const result = await resp.json();
        renderResults(result);
    } catch (err) {
        console.error("Prediction failed:", err);
        showEmpty();
        alert("Failed to get prediction. Check the console for details.");
    }
}


// ══════════ Rendering ══════════

function renderPatientGrid(patients) {
    patientGrid.innerHTML = "";

    if (patients.length === 0) {
        patientGrid.innerHTML = `<p style="color: var(--text-muted); grid-column: 1/-1; text-align: center; padding: 40px;">No patients found matching filters.</p>`;
        return;
    }

    patients.forEach((p, i) => {
        const thumb = document.createElement("div");
        thumb.className = `patient-thumb${p.node_idx === selectedNodeIdx ? " selected" : ""}`;
        thumb.dataset.nodeIdx = p.node_idx;
        thumb.style.animationDelay = `${i * 20}ms`;

        thumb.innerHTML = `
            <img src="${API}/api/image/${p.image_id}"
                 alt="Lesion ${p.image_id}"
                 loading="lazy"
                 onerror="this.style.background='var(--bg-elevated)'; this.alt='No image';">
            <div class="thumb-overlay">
                <span class="thumb-dx">${p.diagnosis}</span>
                <span class="thumb-idx">#${p.node_idx}</span>
            </div>
        `;

        thumb.addEventListener("click", () => {
            // Deselect previous
            document.querySelectorAll(".patient-thumb.selected").forEach(el => el.classList.remove("selected"));
            thumb.classList.add("selected");
            selectedNodeIdx = p.node_idx;
            predictAndExplain(p.node_idx);
        });

        patientGrid.appendChild(thumb);
    });
}

function updatePagination(data) {
    pageInfo.textContent = `Page ${data.page} of ${data.total_pages} (${data.total} patients)`;
    prevPageBtn.disabled = data.page <= 1;
    nextPageBtn.disabled = data.page >= data.total_pages;
}


function renderResults(result) {
    // ── Status Header ──
    const statusEl = document.getElementById("pred-status");
    if (result.is_correct) {
        statusEl.className = "pred-status correct";
        statusEl.innerHTML = `✓ Correct Prediction — Patient #${result.node_idx}`;
    } else {
        statusEl.className = "pred-status incorrect";
        statusEl.innerHTML = `✗ Incorrect Prediction — Patient #${result.node_idx}`;
    }

    // ── Input Image ──
    const inputImg = document.getElementById("input-image");
    if (result.image_id) {
        inputImg.src = `${API}/api/image/${result.image_id}`;
        inputImg.alt = `Lesion ${result.image_id}`;
    }

    const inputMeta = document.getElementById("input-meta");
    const info = result.patient_info || {};
    inputMeta.innerHTML = `
        <strong>Age:</strong> ${info.age || "?"} &nbsp;|&nbsp;
        <strong>Sex:</strong> ${info.sex || "?"} &nbsp;|&nbsp;
        <strong>Site:</strong> ${info.localization || "?"}<br>
        <strong>Split:</strong> ${info.split || "?"}
    `;

    // ── Prediction Cards ──
    document.getElementById("pred-value").textContent = result.predicted_label;
    document.getElementById("pred-confidence").textContent = `Confidence: ${result.confidence}%`;
    document.getElementById("true-value").textContent = result.true_label;

    const verdictEl = document.getElementById("pred-verdict");
    if (result.is_correct) {
        verdictEl.className = "pred-verdict correct";
        verdictEl.textContent = "✓ Model is CORRECT";
    } else {
        verdictEl.className = "pred-verdict incorrect";
        verdictEl.innerHTML = `✗ Model is WRONG<br><span style="font-size: 0.75rem; color: var(--text-secondary);">Predicted ${result.predicted_label} instead of ${result.true_label}</span>`;
    }

    // ── Confidence Bars ──
    renderConfidenceBars(result);

    // ── Feature Importance ──
    renderFeatureImportance(result);

    // ── Similar Patients ──
    renderNeighbors(result);

    showResults();
}


function renderConfidenceBars(result) {
    const container = document.getElementById("confidence-bars");
    container.innerHTML = "";

    const scores = result.confidence_scores;
    const classes = Object.keys(scores);

    // Sort by confidence descending
    classes.sort((a, b) => scores[b] - scores[a]);

    classes.forEach(cls => {
        const val = scores[cls];
        const fullName = getFullName(cls);
        const isPredicted = cls === result.predicted_class;
        const isTrue = cls === result.true_class;

        let fillClass = "other";
        if (isPredicted) fillClass = "predicted";
        else if (isTrue) fillClass = "true-class";

        const row = document.createElement("div");
        row.className = "confidence-row";
        row.innerHTML = `
            <span class="conf-label">${fullName}</span>
            <div class="conf-bar-track">
                <div class="conf-bar-fill ${fillClass}" style="width: 0%"></div>
            </div>
            <span class="conf-val">${val.toFixed(1)}%</span>
        `;
        container.appendChild(row);

        // Animate width after render
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                row.querySelector(".conf-bar-fill").style.width = `${Math.min(val, 100)}%`;
            });
        });
    });
}


function renderFeatureImportance(result) {
    const container = document.getElementById("feature-bars");
    const summary = result.feature_summary;

    container.innerHTML = `
        <div class="feat-bar-group">
            <div class="feat-bar-icon">🖼️</div>
            <div class="feat-bar-name">Visual (ViT) Features</div>
            <div class="feat-bar-outer">
                <div class="feat-bar-inner visual" style="width: 0%"></div>
            </div>
            <div class="feat-bar-pct visual">${summary.visual_features_pct}%</div>
        </div>
        <div class="feat-bar-group">
            <div class="feat-bar-icon">📋</div>
            <div class="feat-bar-name">Clinical Metadata<br>(Age, Sex, Body Site)</div>
            <div class="feat-bar-outer">
                <div class="feat-bar-inner clinical" style="width: 0%"></div>
            </div>
            <div class="feat-bar-pct clinical">${summary.metadata_features_pct}%</div>
        </div>
    `;

    // Animate
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            container.querySelector(".feat-bar-inner.visual").style.width = `${summary.visual_features_pct}%`;
            container.querySelector(".feat-bar-inner.clinical").style.width = `${summary.metadata_features_pct}%`;
        });
    });
}


function renderNeighbors(result) {
    const container = document.getElementById("neighbors-grid");
    const descEl = document.getElementById("neighbors-desc");
    container.innerHTML = "";

    const neighbors = result.influential_neighbors || [];
    if (neighbors.length === 0) {
        descEl.textContent = "No influential neighbors found for this patient.";
        return;
    }

    // Count how many neighbors share the predicted diagnosis
    const sameCount = neighbors.filter(n => n.diagnosis === result.predicted_class).length;
    descEl.innerHTML = `The model predicted <strong>${result.predicted_label}</strong> because this lesion is most similar to these past patients. ` +
        `<strong>${sameCount} of ${neighbors.length}</strong> similar patients had the same diagnosis.`;

    neighbors.forEach((nb, i) => {
        const card = document.createElement("div");
        card.className = "neighbor-card";

        const imgSrc = nb.image_id ? `${API}/api/image/${nb.image_id}` : "";
        const dxFull = nb.diagnosis_full || nb.diagnosis || "Unknown";

        card.innerHTML = `
            <div class="neighbor-img-wrap">
                <img src="${imgSrc}" alt="Neighbor lesion"
                     onerror="this.style.background='var(--bg-surface)';"
                     loading="lazy">
                <span class="neighbor-rank">#${i + 1}</span>
                <span class="neighbor-similarity">Sim: ${nb.importance.toFixed(3)}</span>
            </div>
            <div class="neighbor-info">
                <div class="neighbor-dx">${dxFull}</div>
                <div class="neighbor-meta">
                    Age: ${nb.age || "?"} · Sex: ${nb.sex || "?"}<br>
                    Site: ${nb.localization || "?"}
                </div>
            </div>
        `;

        container.appendChild(card);
    });
}


// ══════════ State Toggles ══════════

function showEmpty() {
    emptyState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    resultsContent.classList.add("hidden");
}

function showLoading() {
    emptyState.classList.add("hidden");
    loadingState.classList.remove("hidden");
    resultsContent.classList.add("hidden");
}

function showResults() {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultsContent.classList.remove("hidden");
}


// ══════════ Helpers ══════════

function getFullName(shortCode) {
    const map = {
        "akiec": "Actinic Keratoses",
        "bcc":   "Basal Cell Carcinoma",
        "bkl":   "Benign Keratosis",
        "df":    "Dermatofibroma",
        "mel":   "Melanoma",
        "nv":    "Melanocytic Nevi",
        "vasc":  "Vascular Lesion",
    };
    return map[shortCode] || shortCode;
}
