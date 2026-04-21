const form = document.querySelector("#boost-form");
const rawPrompt = document.querySelector("#raw_prompt");
const harnessSelect = document.querySelector("#target_harness");
const modelSelect = document.querySelector("#target_model");
const statusPill = document.querySelector("#status-pill");
const factPill = document.querySelector("#fact-check-pill");
const detectedTaskType = document.querySelector("#detected-task-type");
const selectedAdapters = document.querySelector("#selected-adapters");
const evidenceLevel = document.querySelector("#evidence-level");
const readinessScore = document.querySelector("#readiness-score");
const factCheckText = document.querySelector("#fact-check-text");
const warningsBox = document.querySelector("#warnings");
const boostedOutput = document.querySelector("#boosted-output");
const copyOutput = document.querySelector("#copy-output");
const taskSpecDetail = document.querySelector("#task-spec-detail");
const harnessDetail = document.querySelector("#harness-detail");
const modelDetail = document.querySelector("#model-detail");
const failureDetail = document.querySelector("#failure-detail");
const runsList = document.querySelector("#runs-list");
const garageList = document.querySelector("#garage-list");
const garageQuery = document.querySelector("#garage-query");
const garageSearch = document.querySelector("#garage-search");
const matrixCells = document.querySelectorAll(".matrix-cell");

const readinessDisclaimer =
  "Diese Prüfung bewertet Prompt-Struktur und Guardrails. Sie misst keine reale Modellqualität, keine Output-Faithfulness und keine Laufzeitkosten.";

function setStatus(text, state = "") {
  statusPill.textContent = text;
  statusPill.className = `status-pill ${state}`.trim();
}

function setFactStatus(passed, hasRun = true) {
  if (!hasRun) {
    factPill.textContent = "nicht gelaufen";
    factPill.className = "status-pill muted";
    return;
  }
  factPill.textContent = passed ? "bestanden" : "prüfen";
  factPill.className = passed ? "status-pill" : "status-pill warn";
}

function renderWarnings(warnings) {
  warningsBox.innerHTML = "";
  const disclaimer = document.createElement("div");
  disclaimer.className = "warning meta";
  disclaimer.textContent = readinessDisclaimer;
  warningsBox.append(disclaimer);

  for (const warning of warnings) {
    const item = document.createElement("div");
    item.className = "warning";
    item.textContent = warning;
    warningsBox.append(item);
  }
}

async function boostPrompt(event) {
  event.preventDefault();
  setStatus("läuft", "muted");
  form.querySelector("button[type='submit']").disabled = true;

  try {
    const response = await fetch("/boost", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_prompt: rawPrompt.value,
        target_harness: harnessSelect.value,
        target_model: modelSelect.value
      })
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed: ${response.status}`);
    }
    const payload = await response.json();
    const result = payload.result;
    renderResult(result);
    setStatus("gespeichert");
    await loadRuns();
  } catch (error) {
    setStatus("Fehler", "bad");
    factCheckText.textContent = error.message;
  } finally {
    form.querySelector("button[type='submit']").disabled = false;
  }
}

function renderResult(result) {
  detectedTaskType.textContent = result.detected_task_type;
  selectedAdapters.textContent = `${result.target_harness} / ${result.target_model}`;
  evidenceLevel.textContent = describeEvidence(result.evidence_summary);
  readinessScore.textContent = describeReadiness(result.readiness_score);
  boostedOutput.textContent = result.candidate_prompt || result.boosted_prompt;
  factCheckText.textContent = result.no_new_facts_passed
    ? "Der Guard hat keine verdächtigen neuen Domain-Begriffe gefunden."
    : "Prüfe die Warnungen, bevor du den Kandidaten-Prompt nutzt.";
  setFactStatus(result.no_new_facts_passed);
  renderWarnings(result.warnings || []);
  renderTaskSpec(result.task_spec || {});
  renderRuleBlock(harnessDetail, result.harness_policy || result.adapter_rules?.harness || {});
  renderRuleBlock(modelDetail, result.model_adapter || result.adapter_rules?.model || {});
  renderFailures(result.failure_modes || []);
  highlightMatrix(result.target_harness, result.target_model);
}

function describeEvidence(summary) {
  if (!summary) return "Keine Evidence-Metadaten.";
  const count = summary.sources?.length || 0;
  return `${summary.level} · ${count} Quellen · ${summary.eval_run_ids?.length || 0} Eval-Runs`;
}

function describeReadiness(score) {
  if (!score) return "Kein lokaler Score.";
  return `${score.label}: ${score.value}/${score.max_value} · kein Modellbenchmark`;
}

function renderTaskSpec(spec) {
  taskSpecDetail.innerHTML = "";
  appendDetail(taskSpecDetail, "Goal", spec.goal || "Noch nicht erzeugt.");
  appendDetail(taskSpecDetail, "Task Type", spec.task_type || "n/a");
  appendList(taskSpecDetail, "Deliverables", spec.deliverables || []);
  appendList(taskSpecDetail, "Constraints", spec.constraints || []);
  appendList(taskSpecDetail, "Quality Gates", spec.quality_gates || []);
  appendList(taskSpecDetail, "Missing Information", spec.missing_information || []);
}

function renderRuleBlock(target, block) {
  target.innerHTML = "";
  appendDetail(target, "Label", block.label || block.id || "n/a");
  appendDetail(target, "Evidence", block.evidence_level || "hypothesis");
  appendDetail(target, "Summary", block.summary || "Keine Zusammenfassung.");
  appendList(target, "Rules", block.rules || []);
  appendList(target, "Verification Gates", block.verification_gates || []);
  appendList(target, "Claim Notes", block.claim_notes || []);
  appendSources(target, block.sources || []);
}

function renderFailures(failures) {
  failureDetail.innerHTML = "";
  if (!failures.length) {
    appendDetail(failureDetail, "Status", "Keine profilspezifischen Risiko-Hypothesen.");
    return;
  }
  appendList(failureDetail, "Risk hypotheses", failures);
}

function appendDetail(target, label, value) {
  const item = document.createElement("article");
  item.className = "detail-item";
  item.innerHTML = `<strong>${escapeHtml(label)}</strong><p>${escapeHtml(value)}</p>`;
  target.append(item);
}

function appendList(target, label, items) {
  const normalized = items.length ? items : ["Keine Einträge."];
  const item = document.createElement("article");
  item.className = "detail-item";
  item.innerHTML = `<strong>${escapeHtml(label)}</strong><ul>${normalized
    .map((entry) => `<li>${escapeHtml(entry)}</li>`)
    .join("")}</ul>`;
  target.append(item);
}

function appendSources(target, sources) {
  if (!sources.length) {
    appendDetail(target, "Sources", "Keine Quellen hinterlegt.");
    return;
  }
  const item = document.createElement("article");
  item.className = "detail-item";
  item.innerHTML = `<strong>Sources</strong><ul>${sources
    .map((source) => `<li><a href="${escapeHtml(source.url)}" rel="noreferrer">${escapeHtml(source.title)}</a>${source.publisher ? ` · ${escapeHtml(source.publisher)}` : ""}</li>`)
    .join("")}</ul>`;
  target.append(item);
}

function highlightMatrix(harness, model) {
  matrixCells.forEach((cell) => {
    cell.classList.toggle(
      "is-active",
      cell.dataset.harness === harness && cell.dataset.model === model
    );
  });
}

async function loadRuns() {
  const response = await fetch("/api/runs?limit=8");
  const payload = await response.json();
  runsList.innerHTML = "";
  if (!payload.runs.length) {
    runsList.innerHTML = '<p class="meta">Noch keine Übersetzungsläufe.</p>';
    return;
  }
  for (const run of payload.runs) {
    const card = document.createElement("article");
    card.className = "run-card";
    card.innerHTML = `
      <strong>${escapeHtml(run.detected_task_type)} · ${escapeHtml(run.target_harness)} / ${escapeHtml(run.target_model)}</strong>
      <p>${escapeHtml(run.raw_prompt.slice(0, 180))}${run.raw_prompt.length > 180 ? "..." : ""}</p>
      <div class="meta">#${run.id} · ${escapeHtml(run.created_at)} · Guard ${run.no_new_facts_passed ? "bestanden" : "prüfen"}</div>
    `;
    runsList.append(card);
  }
}

async function loadGarageSamples() {
  const q = garageQuery.value.trim();
  const response = await fetch(`/api/promptgarage/samples?limit=6${q ? `&q=${encodeURIComponent(q)}` : ""}`);
  const payload = await response.json();
  garageList.innerHTML = "";
  if (!payload.samples.length) {
    garageList.innerHTML = '<p class="meta">Keine PromptGarage-Samples verfügbar.</p>';
    return;
  }
  for (const sample of payload.samples) {
    const card = document.createElement("article");
    card.className = "sample-card";
    card.innerHTML = `
      <strong>${escapeHtml(sample.title)}</strong>
      <p>${escapeHtml(sample.content)}</p>
      <div class="meta">#${sample.id}${sample.tags ? ` · ${escapeHtml(sample.tags)}` : ""}${sample.model ? ` · ${escapeHtml(sample.model)}` : ""}</div>
    `;
    garageList.append(card);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", boostPrompt);
harnessSelect.addEventListener("change", () => {
  selectedAdapters.textContent = `${harnessSelect.value} / ${modelSelect.value}`;
  highlightMatrix(harnessSelect.value, modelSelect.value);
});
modelSelect.addEventListener("change", () => {
  selectedAdapters.textContent = `${harnessSelect.value} / ${modelSelect.value}`;
  highlightMatrix(harnessSelect.value, modelSelect.value);
});
copyOutput.addEventListener("click", async () => {
  await navigator.clipboard.writeText(boostedOutput.textContent);
});
garageSearch.addEventListener("click", loadGarageSamples);
garageQuery.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadGarageSamples();
  }
});
matrixCells.forEach((cell) => {
  cell.addEventListener("click", () => {
    harnessSelect.value = cell.dataset.harness;
    modelSelect.value = cell.dataset.model;
    selectedAdapters.textContent = `${harnessSelect.value} / ${modelSelect.value}`;
    highlightMatrix(harnessSelect.value, modelSelect.value);
  });
});

loadRuns();
loadGarageSamples();
setFactStatus(false, false);
highlightMatrix(harnessSelect.value, modelSelect.value);
