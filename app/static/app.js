// Krypto Forensic Workbench - Streamlined Client-side Logic

let currentDataset = "unseen";
let windowsData = [];
let currentWindowIndex = 0;
let provenanceCache = {};

document.addEventListener("DOMContentLoaded", () => {
  initWorkbench();
  bindEvents();
});

async function initWorkbench() {
  try {
    await loadWindows(currentDataset);
  } catch (err) {
    showToast(`Initialization failed: ${err.message}`, "error");
  }
}

function bindEvents() {
  document.getElementById("select-dataset").addEventListener("change", async (e) => {
    currentDataset = e.target.value;
    await loadWindows(currentDataset);
  });

  document.getElementById("select-window").addEventListener("change", (e) => {
    selectWindow(parseInt(e.target.value, 10));
  });

  document.getElementById("btn-prev-window").addEventListener("click", () => {
    if (currentWindowIndex > 0) selectWindow(currentWindowIndex - 1);
  });

  document.getElementById("btn-next-window").addEventListener("click", () => {
    if (currentWindowIndex < windowsData.length - 1) selectWindow(currentWindowIndex + 1);
  });

  document.getElementById("btn-run-eval").addEventListener("click", runEvaluation);
  document.getElementById("artifact-search").addEventListener("input", filterEvidence);

  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("provenance-modal").addEventListener("click", (e) => {
    if (e.target.id === "provenance-modal") closeModal();
  });

  document.getElementById("btn-export").addEventListener("click", exportReport);
}

async function loadWindows(dataset) {
  const selectEl = document.getElementById("select-window");
  selectEl.innerHTML = `<option>Loading...</option>`;

  try {
    const res = await fetch(`/api/windows?dataset=${dataset}`);
    const data = await res.json();
    windowsData = data.windows;

    selectEl.innerHTML = "";
    windowsData.forEach((w, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `#${w.index + 1} · ${w.title || "Window"} (${w.event_count} events)`;
      selectEl.appendChild(opt);
    });

    if (windowsData.length > 0) {
      selectWindow(0);
    }
  } catch (err) {
    showToast(`Failed to load dataset: ${err.message}`, "error");
  }
}

function selectWindow(index) {
  if (index < 0 || index >= windowsData.length) return;
  currentWindowIndex = index;
  const windowObj = windowsData[index];

  document.getElementById("select-window").value = index;
  document.getElementById("artifact-count").textContent = `${windowObj.event_count} events`;

  // Render left evidence pane
  renderEvidence(windowObj.user_prompt);

  // If window has ground truth target, render it initially, otherwise reset
  if (windowObj.target_answer) {
    renderReconstruction(windowObj.target_answer, windowObj.user_prompt, null);
  } else {
    resetReconstruction();
  }
}

function renderEvidence(userPrompt) {
  const container = document.getElementById("artifacts-container");
  container.innerHTML = "";

  const lines = userPrompt.split("\n");
  const provenanceMap = {};
  let inProvenance = false;

  lines.forEach(line => {
    if (line.startsWith("PROVENANCE:")) {
      inProvenance = true;
      return;
    }
    if (inProvenance && line.startsWith("EVT-")) {
      const parts = line.split(" -> ");
      if (parts.length === 2) provenanceMap[parts[0].trim()] = parts[1].trim();
    }
  });

  let count = 0;
  lines.forEach(line => {
    const m = line.match(/\[EVT-([a-f0-9]+)\]\s+(\d{2}:\d{2})\s+(\w+)\s+(.+)/);
    if (m) {
      count++;
      const [_, eid, time, app, payload] = m;
      const card = document.createElement("div");
      card.className = "evidence-card";
      card.id = `evidence-${eid}`;
      card.dataset.eid = eid;

      const prov = provenanceMap[`EVT-${eid}`] || `data\\com.${app.toLowerCase()}\\... :: row_id :: ${eid}`;

      card.innerHTML = `
        <div class="card-top">
          <div class="card-tags">
            <span class="evt-id-tag">[EVT-${eid}]</span>
            <span class="app-tag">${app}</span>
          </div>
          <span class="card-time">${time}</span>
        </div>
        <div class="card-payload">${escapeHtml(payload.substring(0, 160))}</div>
        <div class="card-provenance">→ ${escapeHtml(prov)}</div>
      `;

      card.addEventListener("click", () => inspectProvenance(eid));
      container.appendChild(card);
    }
  });

  if (count === 0) {
    container.innerHTML = `<div class="empty-state"><p>No artifact records found in this window.</p></div>`;
  }
}

function filterEvidence(e) {
  const q = e.target.value.toLowerCase().trim();
  document.querySelectorAll(".evidence-card").forEach(card => {
    card.style.display = card.textContent.toLowerCase().includes(q) ? "block" : "none";
  });
}

async function runEvaluation() {
  const windowObj = windowsData[currentWindowIndex];
  if (!windowObj) return;

  const btn = document.getElementById("btn-run-eval");
  const runText = document.getElementById("run-text");
  const modelSelect = document.getElementById("select-model");
  const selectedModel = modelSelect.value;

  btn.disabled = true;
  runText.textContent = "Running Inference...";

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name: selectedModel,
        system_prompt: windowObj.system_prompt,
        user_prompt: windowObj.user_prompt,
        max_new_tokens: 512,
        temperature: 0.1,
        chat_template: selectedModel.includes("gemma") ? "gemma" : "chatml",
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Inference failed");
    }

    const genData = await res.json();
    const prediction = genData.prediction;

    // Evaluate
    const evalRes = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction: prediction,
        user_prompt: windowObj.user_prompt,
        target_answer: windowObj.target_answer,
        window_date: windowObj.date,
      }),
    });

    const evalData = await evalRes.json();
    renderReconstruction(prediction, windowObj.user_prompt, evalData.metrics);
    showToast("Evaluation complete!", "success");

  } catch (err) {
    console.error(err);
    showToast(`Error: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    runText.textContent = "Evaluate Window";
  }
}

function renderReconstruction(text, userPrompt, metrics) {
  const timelineBody = document.getElementById("timeline-body");
  const absenceBody = document.getElementById("absence-body");
  const validPromptIds = new Set((userPrompt.match(/EVT-([a-f0-9]+)/g) || []).map(s => s.replace("EVT-", "")));

  const sections = text.split(/Documented activity without direct artifact support:/i);
  const directPart = sections[0].replace(/Reconstructed activity with artifact support:/i, "").trim();
  const absencePart = sections.length > 1 ? sections[1].trim() : "";

  // Render Direct Timeline
  timelineBody.innerHTML = "";
  const lines = directPart.split("\n").filter(l => l.trim().startsWith("-") || l.trim().length > 0);

  if (lines.length === 0) {
    timelineBody.innerHTML = `<div class="empty-state"><p>No direct activity reported.</p></div>`;
  } else {
    lines.forEach(l => {
      const row = document.createElement("div");
      row.className = "timeline-entry-row";

      const formatted = l.replace(/\[EVT-([a-f0-9]+)\]/g, (match, eid) => {
        const isValid = validPromptIds.has(eid);
        return `<span class="citation-badge ${isValid ? '' : 'invalid-badge'}" data-eid="${eid}">[EVT-${eid.substring(0, 8)}]</span>`;
      });

      row.innerHTML = formatted;
      timelineBody.appendChild(row);
    });
  }

  // Render Absence Section
  absenceBody.innerHTML = "";
  const absenceLines = absencePart.split("\n").filter(l => l.trim().startsWith("-"));

  if (absenceLines.length === 0) {
    absenceBody.innerHTML = `<p class="text-muted">No unrecoverable activity recorded in this window.</p>`;
  } else {
    absenceLines.forEach(l => {
      const clean = l.replace(/^-/, "").trim();
      const item = document.createElement("div");
      item.className = "absence-item";

      const rMatch = clean.match(/\[(.*?)\]/);
      const reason = rMatch ? rMatch[1] : "Unrecoverable Store";
      const desc = clean.replace(/\[(.*?)\]/, "").trim();

      item.innerHTML = `
        <div class="absence-item-title">${escapeHtml(desc)}</div>
        <div class="absence-item-reason">Reason: ${escapeHtml(reason)}</div>
      `;
      absenceBody.appendChild(item);
    });
  }

  // Update Metrics
  if (metrics) {
    document.getElementById("metric-precision").textContent = `${(metrics.citation_precision * 100).toFixed(0)}%`;
    document.getElementById("metric-citations").textContent = `${metrics.valid_citations} / ${metrics.total_citations}`;
    document.getElementById("metric-absence").textContent = metrics.pred_has_absence_reasoning ? "Flagged" : "None";
    document.getElementById("metric-hallucinations").textContent = metrics.hallucinated_citations;
  } else {
    const cited = (text.match(/EVT-([a-f0-9]+)/g) || []).map(s => s.replace("EVT-", ""));
    const valid = cited.filter(e => validPromptIds.has(e));
    const prec = cited.length > 0 ? (valid.length / cited.length) * 100 : 100;
    
    document.getElementById("metric-precision").textContent = `${prec.toFixed(0)}%`;
    document.getElementById("metric-citations").textContent = `${valid.length} / ${cited.length}`;
    document.getElementById("metric-absence").textContent = absenceLines.length > 0 ? "Flagged" : "None";
    document.getElementById("metric-hallucinations").textContent = cited.length - valid.length;
  }

  // Attach hover and click interactions
  setupCitationListeners();
}

function setupCitationListeners() {
  document.querySelectorAll(".citation-badge").forEach(badge => {
    const eid = badge.dataset.eid;

    badge.addEventListener("mouseenter", () => {
      const el = document.getElementById(`evidence-${eid}`);
      if (el) {
        el.classList.add("active-target");
        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });

    badge.addEventListener("mouseleave", () => {
      const el = document.getElementById(`evidence-${eid}`);
      if (el) el.classList.remove("active-target");
    });

    badge.addEventListener("click", () => inspectProvenance(eid));
  });
}

async function inspectProvenance(eid) {
  const modal = document.getElementById("provenance-modal");
  const modalTitle = document.getElementById("modal-evt-id");
  const modalContent = document.getElementById("modal-content");

  modalTitle.textContent = `[EVT-${eid}] SQLite Provenance`;
  modalContent.innerHTML = `<span class="text-muted">Loading SQLite provenance...</span>`;
  modal.classList.add("active");

  try {
    if (provenanceCache[eid]) {
      renderModalBody(provenanceCache[eid]);
      return;
    }

    const res = await fetch(`/api/provenance/${eid}`);
    if (!res.ok) {
      modalContent.innerHTML = `<span style="color: var(--rose);">Event ID not found in database.</span>`;
      return;
    }

    const data = await res.json();
    provenanceCache[eid] = data;
    renderModalBody(data);
  } catch (err) {
    modalContent.innerHTML = `<span style="color: var(--rose);">Query error: ${err.message}</span>`;
  }
}

function renderModalBody(data) {
  document.getElementById("modal-content").innerHTML = `
    <div style="display:flex; flex-direction:column; gap:6px;">
      <div><strong>App:</strong> ${data.app || "System / App"}</div>
      <div><strong>Database:</strong> <span style="color: #A5B4FC;">${data.db_path || "N/A"}</span></div>
      <div><strong>Table:</strong> ${data.table || "N/A"}</div>
      <div><strong>Row ID:</strong> ${data.row_id || "N/A"}</div>
      <div><strong>Raw Timestamp:</strong> ${data.raw_timestamp} (${data.epoch_type || "auto"})</div>
      <div><strong>Normalized Time:</strong> ${data.timestamp || "N/A"}</div>
    </div>
  `;
}

function closeModal() {
  document.getElementById("provenance-modal").classList.remove("active");
}

function resetReconstruction() {
  document.getElementById("timeline-body").innerHTML = `
    <div class="empty-state">
      <p>Select a window and click <strong>Evaluate Window</strong> to generate timeline reconstruction.</p>
    </div>
  `;
  document.getElementById("absence-body").innerHTML = `
    <p class="text-muted">No missing or unrecoverable activity evaluated yet.</p>
  `;
  document.getElementById("metric-precision").textContent = "--";
  document.getElementById("metric-citations").textContent = "--";
  document.getElementById("metric-absence").textContent = "--";
  document.getElementById("metric-hallucinations").textContent = "0";
}

function exportReport() {
  const windowObj = windowsData[currentWindowIndex];
  if (!windowObj) return;

  const data = {
    window: windowObj.title,
    date: windowObj.date,
    precision: document.getElementById("metric-precision").textContent,
    citations: document.getElementById("metric-citations").textContent,
    timeline: document.getElementById("timeline-body").innerText,
    absence: document.getElementById("absence-body").innerText,
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `audit_${windowObj.date || "window"}.json`;
  a.click();
  showToast("Audit report exported as JSON", "success");
}

function showToast(msg, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function escapeHtml(s) {
  if (!s) return "";
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
