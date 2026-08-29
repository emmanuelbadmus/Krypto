// Krypto Forensic Reconstruction Workbench - Polished Interactive Client

let currentDataset = "val";
let windowsData = [];
let currentWindowIndex = 0;
let provenanceCache = {};
let activeTab = "pred";

document.addEventListener("DOMContentLoaded", () => {
  initWorkbench();
  bindEvents();
});

async function initWorkbench() {
  try {
    const statusRes = await fetch("/api/status");
    if (statusRes.ok) {
      const statusData = await statusRes.json();
      document.getElementById("status-db-info").textContent = `SQLite: ${statusData.indexed_events_count.toLocaleString()} events | ${statusData.ground_truth_count} GT`;
    }
    await loadWindows(currentDataset);
  } catch (err) {
    showToast(`Initialization error: ${err.message}`, "error");
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

  // Tab switching
  document.querySelectorAll(".view-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".view-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeTab = tab.dataset.tab;

      document.querySelectorAll(".view-container").forEach(c => c.classList.remove("active"));
      if (activeTab === "pred") document.getElementById("view-pred-container").classList.add("active");
      else if (activeTab === "target") document.getElementById("view-target-container").classList.add("active");
      else if (activeTab === "split") document.getElementById("view-split-container").classList.add("active");
    });
  });

  // Modal
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("provenance-modal").addEventListener("click", (e) => {
    if (e.target.id === "provenance-modal") closeModal();
  });

  document.getElementById("btn-export").addEventListener("click", exportReport);

  // Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

    if (e.key === "ArrowLeft") {
      if (currentWindowIndex > 0) selectWindow(currentWindowIndex - 1);
    } else if (e.key === "ArrowRight") {
      if (currentWindowIndex < windowsData.length - 1) selectWindow(currentWindowIndex + 1);
    } else if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      runEvaluation();
    } else if (e.key === "Escape") {
      closeModal();
    }
  });
}

async function loadWindows(dataset) {
  const selectEl = document.getElementById("select-window");
  selectEl.innerHTML = `<option>Loading windows...</option>`;

  try {
    const res = await fetch(`/api/windows?dataset=${dataset}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    windowsData = data.windows;

    selectEl.innerHTML = "";
    windowsData.forEach((w, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `#${w.index + 1} · ${w.date || w.title} (${w.event_count} events)`;
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

  // Render left evidence records
  renderEvidence(windowObj.user_prompt);

  // Render target answer if present
  if (windowObj.target_answer) {
    renderTargetView(windowObj.target_answer, windowObj.user_prompt);
  } else {
    document.getElementById("target-body").innerHTML = `<div class="empty-state"><p>No ground-truth target for this unlabelled window.</p></div>`;
    document.getElementById("split-target-body").innerHTML = `<div class="empty-state"><p>No target available.</p></div>`;
  }

  // Reset or initialize prediction view
  resetReconstruction();
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

      const appClass = getAppBadgeClass(app);
      const prov = provenanceMap[`EVT-${eid}`] || `data\\com.${app.toLowerCase()}\\... :: row_id :: ${eid}`;

      card.innerHTML = `
        <div class="card-top">
          <div class="card-badges">
            <span class="evt-badge">[EVT-${eid.substring(0, 8)}]</span>
            <span class="app-badge ${appClass}">${app}</span>
          </div>
          <span class="card-time">${time}</span>
        </div>
        <div class="card-payload">${escapeHtml(payload.substring(0, 180))}</div>
        <div class="card-sqlite-meta">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
          <span>${escapeHtml(prov)}</span>
        </div>
      `;

      card.addEventListener("click", () => inspectProvenance(eid));
      container.appendChild(card);
    }
  });

  if (count === 0) {
    container.innerHTML = `<div class="empty-state"><p>No artifact records found in this window.</p></div>`;
  }
}

function getAppBadgeClass(app) {
  const a = app.toLowerCase();
  if (a.includes("twitter")) return "app-twitter";
  if (a.includes("snapchat")) return "app-snapchat";
  if (a.includes("sms") || a.includes("telephony")) return "app-sms";
  if (a.includes("power") || a.includes("screen") || a.includes("system")) return "app-system";
  return "app-default";
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
  const selectedModel = document.getElementById("select-model").value;

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
        max_new_tokens: 350,
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
  const splitPredBody = document.getElementById("split-pred-body");
  const validPromptIds = new Set((userPrompt.match(/EVT-([a-f0-9]+)/gi) || []).map(s => s.replace(/EVT-/i, "").toLowerCase()));

  const sections = text.split(/Documented activity without direct artifact support:/i);
  const directPart = sections[0].replace(/Reconstructed activity with artifact support:/i, "").trim();
  const absencePart = sections.length > 1 ? sections[1].trim() : "";

  // Render Direct Timeline
  timelineBody.innerHTML = "";
  splitPredBody.innerHTML = "";
  const lines = directPart.split("\n").filter(l => l.trim().length > 0);

  if (lines.length === 0) {
    timelineBody.innerHTML = `<div class="empty-state"><p>No direct activity reported.</p></div>`;
    splitPredBody.innerHTML = `<div class="empty-state"><p>No direct activity reported.</p></div>`;
  } else {
    lines.forEach(l => {
      const formatted = formatTimelineLine(l, validPromptIds);

      const row = document.createElement("div");
      row.className = "timeline-row";
      row.innerHTML = formatted;
      timelineBody.appendChild(row);

      const splitRow = document.createElement("div");
      splitRow.className = "timeline-row";
      splitRow.innerHTML = formatted;
      splitPredBody.appendChild(splitRow);
    });
  }

  // Render Absence Section
  absenceBody.innerHTML = "";
  const absenceLines = absencePart.split("\n").filter(l => l.trim().startsWith("-"));

  if (absenceLines.length === 0) {
    absenceBody.innerHTML = `<p class="text-muted">No unrecoverable or encrypted activity documented in this window.</p>`;
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

  // Update Scorecard
  if (metrics) {
    document.getElementById("metric-precision").textContent = `${(metrics.citation_precision * 100).toFixed(0)}%`;
    document.getElementById("metric-prec-tag").textContent = `${metrics.valid_citations}/${metrics.total_citations} Valid`;
    document.getElementById("metric-absence").textContent = metrics.pred_has_absence_reasoning ? "Detected" : "None";
    document.getElementById("metric-absence").className = `metric-val ${metrics.pred_has_absence_reasoning ? 'val-emerald' : 'val-amber'}`;
    document.getElementById("metric-hallucinations").textContent = metrics.hallucinated_citations;
    document.getElementById("metric-hallucinations").className = `metric-val ${metrics.hallucinated_citations > 0 ? 'val-rose' : ''}`;
  }

  setupCitationListeners();
}

function renderTargetView(targetText, userPrompt) {
  const targetBody = document.getElementById("target-body");
  const splitTargetBody = document.getElementById("split-target-body");
  const validPromptIds = new Set((userPrompt.match(/EVT-([a-f0-9]+)/gi) || []).map(s => s.replace(/EVT-/i, "").toLowerCase()));

  targetBody.innerHTML = "";
  splitTargetBody.innerHTML = "";

  const lines = targetText.split("\n").filter(l => l.trim().length > 0);
  lines.forEach(l => {
    const formatted = formatTimelineLine(l, validPromptIds);

    const row = document.createElement("div");
    row.className = "timeline-row";
    row.innerHTML = formatted;
    targetBody.appendChild(row);

    const splitRow = document.createElement("div");
    splitRow.className = "timeline-row";
    splitRow.innerHTML = formatted;
    splitTargetBody.appendChild(splitRow);
  });
}

function formatTimelineLine(line, validPromptIds) {
  return line.replace(/\[EVT-([a-f0-9]+)\]/gi, (match, eid) => {
    const clean = eid.toLowerCase();
    const isValid = validPromptIds.has(clean);
    return `<span class="citation-chip ${isValid ? '' : 'invalid-chip'}" data-eid="${clean}">[EVT-${clean.substring(0, 8)}]</span>`;
  });
}

function setupCitationListeners() {
  document.querySelectorAll(".citation-chip").forEach(badge => {
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

  modalTitle.textContent = `[EVT-${eid.toUpperCase()}]`;
  modalContent.innerHTML = `<div class="empty-state"><p>Querying master SQLite index...</p></div>`;
  modal.classList.add("active");

  try {
    if (provenanceCache[eid]) {
      renderModalBody(provenanceCache[eid]);
      return;
    }

    const res = await fetch(`/api/provenance/${eid}`);
    if (!res.ok) {
      modalContent.innerHTML = `<div style="color: var(--rose); padding: 20px 0;">Event ID not found in master SQLite database index.</div>`;
      return;
    }

    const data = await res.json();
    provenanceCache[eid] = data;
    renderModalBody(data);
  } catch (err) {
    modalContent.innerHTML = `<div style="color: var(--rose);">Query error: ${err.message}</div>`;
  }
}

function renderModalBody(data) {
  document.getElementById("modal-content").innerHTML = `
    <div class="provenance-field-grid">
      <div class="prov-cell">
        <div class="prov-label">Application</div>
        <div class="prov-value" style="color: var(--cyan); font-weight:700;">${data.app}</div>
      </div>
      <div class="prov-cell">
        <div class="prov-label">Event Type</div>
        <div class="prov-value">${data.event_type}</div>
      </div>
      <div class="prov-cell prov-cell-full">
        <div class="prov-label">Database File Path</div>
        <div class="prov-value" style="color: #A5B4FC;">${data.db_path}</div>
      </div>
      <div class="prov-cell">
        <div class="prov-label">Table Name</div>
        <div class="prov-value">${data.table}</div>
      </div>
      <div class="prov-cell">
        <div class="prov-label">Row ID</div>
        <div class="prov-value">${data.row_id}</div>
      </div>
      <div class="prov-cell">
        <div class="prov-label">Normalized Local Time</div>
        <div class="prov-value">${data.timestamp || "N/A"}</div>
      </div>
      <div class="prov-cell">
        <div class="prov-label">Raw Epoch Timestamp</div>
        <div class="prov-value">${data.raw_timestamp} (${data.epoch_type})</div>
      </div>
      <div class="prov-cell prov-cell-full">
        <div class="prov-label">Decoded Payload / Content</div>
        <div class="prov-payload-box">${escapeHtml(data.content || "(empty payload)")}</div>
      </div>
    </div>
  `;
}

function closeModal() {
  document.getElementById("provenance-modal").classList.remove("active");
}

function resetReconstruction() {
  document.getElementById("timeline-body").innerHTML = `
    <div class="empty-state">
      <p>Click <strong>Evaluate Window</strong> to run model inference.</p>
    </div>
  `;
  document.getElementById("absence-body").innerHTML = `
    <p class="text-muted">No unrecoverable or encrypted activity documented in this window.</p>
  `;
  document.getElementById("metric-precision").textContent = "--";
  document.getElementById("metric-prec-tag").textContent = "0/0 Cited";
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
    timeline: document.getElementById("timeline-body").innerText,
    absence: document.getElementById("absence-body").innerText,
    target: windowObj.target_answer,
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `forensic_audit_${windowObj.date || "window"}.json`;
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
