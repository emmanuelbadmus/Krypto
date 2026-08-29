// Forensic Activity Reconstruction Workbench - Interactive Frontend Logic

let currentDataset = "unseen";
let windowsData = [];
let currentWindowIndex = 0;
let provenanceCache = {};

document.addEventListener("DOMContentLoaded", () => {
  initSystem();
  setupEventListeners();
});

async function initSystem() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    
    document.getElementById("stat-events").textContent = `${data.indexed_events_count.toLocaleString()} events`;
    document.getElementById("stat-gt").textContent = `${data.ground_truth_count} events`;

    await loadWindows(currentDataset);
  } catch (err) {
    console.error("Initialization error:", err);
  }
}

function setupEventListeners() {
  document.getElementById("select-dataset").addEventListener("change", async (e) => {
    currentDataset = e.target.value;
    await loadWindows(currentDataset);
  });

  document.getElementById("btn-run-eval").addEventListener("click", runEvaluation);

  document.getElementById("artifact-search").addEventListener("input", filterArtifacts);

  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("provenance-modal").addEventListener("click", (e) => {
    if (e.target.id === "provenance-modal") closeModal();
  });

  document.getElementById("btn-export").addEventListener("click", exportAuditReport);
}

async function loadWindows(dataset) {
  const pillsContainer = document.getElementById("window-pills");
  pillsContainer.innerHTML = `<span style="color: var(--text-muted); font-size: 12px;">Loading windows...</span>`;

  try {
    const res = await fetch(`/api/windows?dataset=${dataset}`);
    const data = await res.json();
    windowsData = data.windows;

    renderWindowPills();
    if (windowsData.length > 0) {
      selectWindow(0);
    }
  } catch (err) {
    pillsContainer.innerHTML = `<span style="color: var(--rose-hallucinated);">Failed to load windows</span>`;
    console.error(err);
  }
}

function renderWindowPills() {
  const pillsContainer = document.getElementById("window-pills");
  pillsContainer.innerHTML = "";

  windowsData.forEach((w, idx) => {
    const pill = document.createElement("div");
    pill.className = `window-pill ${idx === currentWindowIndex ? "active" : ""}`;
    pill.innerHTML = `<strong>#${w.index + 1}</strong> ${w.title || "Window"} <span style="opacity: 0.6; font-size: 11px;">(${w.event_count})</span>`;
    pill.addEventListener("click", () => selectWindow(idx));
    pillsContainer.appendChild(pill);
  });
}

function selectWindow(index) {
  currentWindowIndex = index;
  const windowObj = windowsData[index];
  if (!windowObj) return;

  // Update active pill
  document.querySelectorAll(".window-pill").forEach((el, i) => {
    el.classList.toggle("active", i === index);
  });

  document.getElementById("current-window-date").textContent = `Window: ${windowObj.title} (${windowObj.date || "Unknown date"})`;
  document.getElementById("artifact-count-chip").textContent = `${windowObj.event_count} events`;

  // Parse and render extracted artifacts in left pane
  renderArtifacts(windowObj.user_prompt);

  // If window already has target reconstruction, display it or reset
  if (windowObj.target_answer) {
    displayReconstruction(windowObj.target_answer, windowObj.user_prompt, true);
  } else {
    resetReconstructionPanel();
  }
}

function renderArtifacts(userPrompt) {
  const container = document.getElementById("artifacts-list");
  container.innerHTML = "";

  const artifactRegex = /\[EVT-([a-f0-9]+)\]\s+(\d{2}:\d{2})\s+(\w+)\s+(.+)/g;
  let match;
  let count = 0;

  const lines = userPrompt.split("\n");
  const extractedArtifacts = [];
  const provenanceMap = {};

  // Extract provenance lines
  let inProvenance = false;
  lines.forEach(line => {
    if (line.startsWith("PROVENANCE:")) {
      inProvenance = true;
      return;
    }
    if (inProvenance && line.startsWith("EVT-")) {
      const parts = line.split(" -> ");
      if (parts.length === 2) {
        provenanceMap[parts[0].trim()] = parts[1].trim();
      }
    }
  });

  lines.forEach(line => {
    const artMatch = line.match(/\[EVT-([a-f0-9]+)\]\s+(\d{2}:\d{2})\s+(\w+)\s+(.+)/);
    if (artMatch) {
      const [_, eid, time, app, payload] = artMatch;
      count++;
      
      const card = document.createElement("div");
      card.className = "artifact-card";
      card.id = `art-card-${eid}`;
      card.dataset.eid = eid;
      card.dataset.app = app.toLowerCase();

      const provText = provenanceMap[`EVT-${eid}`] || `data\\com.${app.toLowerCase()}\\databases... :: row_id :: ${eid}`;

      card.innerHTML = `
        <div class="artifact-header">
          <div class="artifact-meta">
            <span class="evt-id-badge">[EVT-${eid}]</span>
            <span class="app-badge">${app}</span>
          </div>
          <span class="artifact-time">${time}</span>
        </div>
        <div class="artifact-payload">${escapeHtml(payload.substring(0, 180))}</div>
        <div class="provenance-info">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7V4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v3M4 17v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3M4 7h16M4 17h16"></path></svg>
          <span>${escapeHtml(provText)}</span>
        </div>
      `;

      card.addEventListener("click", () => inspectProvenance(eid));
      container.appendChild(card);
    }
  });

  if (count === 0) {
    container.innerHTML = `<p style="color: var(--text-muted); padding: 12px;">No artifact events found in this window.</p>`;
  }
}

function filterArtifacts(e) {
  const query = e.target.value.toLowerCase().trim();
  document.querySelectorAll(".artifact-card").forEach(card => {
    const text = card.textContent.toLowerCase();
    card.style.display = text.includes(query) ? "block" : "none";
  });
}

async function runEvaluation() {
  const windowObj = windowsData[currentWindowIndex];
  if (!windowObj) return;

  const btn = document.getElementById("btn-run-eval");
  const statusChip = document.getElementById("eval-status-chip");
  const modelSelect = document.getElementById("select-model");
  const selectedModel = modelSelect.value;

  btn.disabled = true;
  btn.innerHTML = `<svg class="spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="10"></circle></svg> <span>Evaluating with ${selectedModel.split("/").pop()}...</span>`;
  statusChip.textContent = "Inferring...";
  statusChip.style.color = "var(--cyan-primary)";

  try {
    // 1. Call model generation
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name: selectedModel,
        system_prompt: windowObj.system_prompt,
        user_prompt: windowObj.user_prompt,
        max_new_tokens: 1024,
        temperature: 0.1,
        chat_template: selectedModel.includes("gemma") ? "gemma" : "chatml",
      }),
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Model generation failed");
    }

    const genData = await res.json();
    const prediction = genData.prediction;

    // 2. Call metric evaluation
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
    const metrics = evalData.metrics;

    // 3. Display results
    displayReconstruction(prediction, windowObj.user_prompt, false, metrics);
    statusChip.textContent = "Evaluation Done";
    statusChip.style.color = "var(--emerald-valid)";

  } catch (err) {
    console.error("Evaluation error:", err);
    alert(`Evaluation Error: ${err.message}`);
    statusChip.textContent = "Error";
    statusChip.style.color = "var(--rose-hallucinated)";
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>Run Evaluation</span>`;
  }
}

function displayReconstruction(text, userPrompt, isGroundTruth = false, metrics = null) {
  const timelineContent = document.getElementById("timeline-content");
  const absenceContent = document.getElementById("absence-content");

  // Valid prompt event IDs
  const validPromptIds = new Set((userPrompt.match(/EVT-([a-f0-9]+)/g) || []).map(s => s.replace("EVT-", "")));

  // Split into direct activity and absence section
  const sections = text.split(/Documented activity without direct artifact support:/i);
  const directPart = sections[0].replace(/Reconstructed activity with artifact support:/i, "").trim();
  const absencePart = sections.length > 1 ? sections[1].trim() : "";

  // Render Direct Timeline Entries
  timelineContent.innerHTML = "";
  const directLines = directPart.split("\n").filter(l => l.trim().startsWith("-"));

  if (directLines.length === 0) {
    timelineContent.innerHTML = `<p style="color: var(--text-muted); font-size: 13px;">No direct artifact-supported activities recorded in this window.</p>`;
  } else {
    directLines.forEach(line => {
      const entry = document.createElement("div");
      entry.className = "timeline-entry";

      // Parse time and text
      let lineText = line.replace(/^-/, "").trim();
      
      // Highlight and format [EVT-xxxx] chips
      const formattedHTML = lineText.replace(/\[EVT-([a-f0-9]+)\]/g, (match, eid) => {
        const isValid = validPromptIds.has(eid);
        return `<span class="citation-chip ${isValid ? '' : 'invalid'}" data-eid="${eid}" title="${isValid ? 'Click to inspect SQLite provenance' : 'HALLUCINATION: ID not in extracted provenance'}">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
          [EVT-${eid.substring(0, 8)}...]
        </span>`;
      });

      entry.innerHTML = `<span class="timeline-desc">${formattedHTML}</span>`;
      timelineContent.appendChild(entry);
    });
  }

  // Render Absence Section
  absenceContent.innerHTML = "";
  const absenceLines = absencePart.split("\n").filter(l => l.trim().startsWith("-"));

  if (absenceLines.length === 0) {
    absenceContent.innerHTML = `<p style="color: var(--text-muted); font-size: 12px;">No unrecoverable activities documented for this window.</p>`;
  } else {
    absenceLines.forEach(line => {
      const cleanLine = line.replace(/^-/, "").trim();
      const card = document.createElement("div");
      card.className = "absence-entry";
      
      // Parse reason brackets e.g. [reason]
      const reasonMatch = cleanLine.match(/\[(.*?)\]/);
      const reason = reasonMatch ? reasonMatch[1] : "Unrecoverable Activity";
      const desc = cleanLine.replace(/\[(.*?)\]/, "").trim();

      card.innerHTML = `
        <div class="absence-header">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          <span>${escapeHtml(desc)}</span>
        </div>
        <div class="absence-reason">REASON: ${escapeHtml(reason)}</div>
      `;
      absenceContent.appendChild(card);
    });
  }

  // Update Scorecard Metrics
  if (metrics) {
    document.getElementById("metric-precision").textContent = `${(metrics.citation_precision * 100).toFixed(1)}%`;
    document.getElementById("metric-citations").textContent = `${metrics.valid_citations} / ${metrics.total_citations}`;
    document.getElementById("metric-absence").textContent = metrics.pred_has_absence_reasoning ? "Detected" : "None";
    document.getElementById("metric-hallucinations").textContent = metrics.hallucinated_citations;
  } else {
    // Quick calculate from prompt
    const citedEids = (text.match(/EVT-([a-f0-9]+)/g) || []).map(s => s.replace("EVT-", ""));
    const validCitations = citedEids.filter(e => validPromptIds.has(e));
    const prec = citedEids.length > 0 ? (validCitations.length / citedEids.length) * 100 : 100;
    
    document.getElementById("metric-precision").textContent = `${prec.toFixed(1)}%`;
    document.getElementById("metric-citations").textContent = `${validCitations.length} / ${citedEids.length}`;
    document.getElementById("metric-absence").textContent = absenceLines.length > 0 ? "Detected" : "None";
    document.getElementById("metric-hallucinations").textContent = citedEids.length - validCitations.length;
  }

  // Setup interactive citation chip hover/click
  setupCitationInteractions();
}

function setupCitationInteractions() {
  document.querySelectorAll(".citation-chip").forEach(chip => {
    const eid = chip.dataset.eid;

    chip.addEventListener("mouseenter", () => {
      const artCard = document.getElementById(`art-card-${eid}`);
      if (artCard) {
        artCard.classList.add("highlighted");
        artCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });

    chip.addEventListener("mouseleave", () => {
      const artCard = document.getElementById(`art-card-${eid}`);
      if (artCard) artCard.classList.remove("highlighted");
    });

    chip.addEventListener("click", () => inspectProvenance(eid));
  });
}

async function inspectProvenance(eid) {
  const modal = document.getElementById("provenance-modal");
  const modalTitle = document.getElementById("modal-evt-id");
  const modalBody = document.getElementById("modal-body");

  modalTitle.textContent = `[EVT-${eid}] Database Provenance`;
  modalBody.innerHTML = `<span style="color: var(--text-muted);">Querying SQLite provenance registry...</span>`;
  modal.classList.add("active");

  try {
    if (provenanceCache[eid]) {
      renderProvenanceModal(provenanceCache[eid]);
      return;
    }

    const res = await fetch(`/api/provenance/${eid}`);
    if (!res.ok) {
      modalBody.innerHTML = `<span style="color: var(--rose-hallucinated);">Event ID not found in master database registry.</span>`;
      return;
    }

    const data = await res.json();
    provenanceCache[eid] = data;
    renderProvenanceModal(data);
  } catch (err) {
    modalBody.innerHTML = `<span style="color: var(--rose-hallucinated);">Failed to fetch provenance: ${err.message}</span>`;
  }
}

function renderProvenanceModal(data) {
  const modalBody = document.getElementById("modal-body");
  modalBody.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 8px;">
      <div><strong style="color: var(--cyan-primary);">Application:</strong> ${data.app || "Unknown"}</div>
      <div><strong style="color: var(--cyan-primary);">Artifact Type:</strong> ${data.artifact_type || "Generic Event"}</div>
      <div><strong style="color: var(--cyan-primary);">SQLite DB Path:</strong> <span style="color: #A5B4FC;">${data.db_path || "N/A"}</span></div>
      <div><strong style="color: var(--cyan-primary);">Table:</strong> ${data.table || "N/A"}</div>
      <div><strong style="color: var(--cyan-primary);">Row ID:</strong> ${data.row_id || "N/A"}</div>
      <div><strong style="color: var(--cyan-primary);">Raw Epoch:</strong> ${data.raw_timestamp} (${data.epoch_type})</div>
      <div><strong style="color: var(--cyan-primary);">Normalized Timestamp:</strong> ${data.timestamp}</div>
      <div style="margin-top: 8px;"><strong style="color: var(--cyan-primary);">Raw Payload:</strong></div>
      <pre style="background: rgba(0,0,0,0.4); padding: 8px; border-radius: 4px; overflow-x: auto; color: var(--text-secondary);">${JSON.stringify(data.raw_data, null, 2)}</pre>
    </div>
  `;
}

function closeModal() {
  document.getElementById("provenance-modal").classList.remove("active");
}

function resetReconstructionPanel() {
  document.getElementById("timeline-content").innerHTML = `
    <p style="color: var(--text-muted); font-style: italic;">Click "Run Evaluation" above to generate grounded forensic timeline using Gemma-4-E2B-it.</p>
  `;
  document.getElementById("absence-content").innerHTML = `
    <p style="color: var(--text-muted); font-size: 12px;">No unrecoverable activities evaluated yet.</p>
  `;
  document.getElementById("metric-precision").textContent = "--%";
  document.getElementById("metric-citations").textContent = "--";
  document.getElementById("metric-absence").textContent = "--";
  document.getElementById("metric-hallucinations").textContent = "0";
}

function exportAuditReport() {
  const windowObj = windowsData[currentWindowIndex];
  if (!windowObj) return;

  const report = {
    case: "Joshua Hickman Pixel 3 Android 9 Corpus",
    window: windowObj.title,
    date: windowObj.date,
    metrics: {
      precision: document.getElementById("metric-precision").textContent,
      citations: document.getElementById("metric-citations").textContent,
      absence: document.getElementById("metric-absence").textContent,
      hallucinations: document.getElementById("metric-hallucinations").textContent,
    },
    reconstructed_timeline: document.getElementById("timeline-content").innerText,
    absence_analysis: document.getElementById("absence-content").innerText,
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `forensic_audit_${windowObj.date || "window"}.json`;
  a.click();
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
