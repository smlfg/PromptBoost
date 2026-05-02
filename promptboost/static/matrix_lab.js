const matrixPrompt = document.querySelector("#matrix-prompt");
const harnessSelect = document.querySelector("#matrix-harness");
const modelSelect = document.querySelector("#matrix-model");
const runDeterministic = document.querySelector("#run-deterministic");
const runFullCell = document.querySelector("#run-full-cell");
const matrixStatus = document.querySelector("#matrix-status");
const matrixSummary = document.querySelector("#matrix-summary");
const matrixNote = document.querySelector("#matrix-note");
const matrixFindings = document.querySelector("#matrix-findings");
const matrixQuestion = document.querySelector("#matrix-question");
const matrixGrid = document.querySelector("#matrix-grid");
const cellDetail = document.querySelector("#cell-detail");
const deterministicPreview = document.querySelector("#deterministic-preview");
const agenticPreview = document.querySelector("#agentic-preview");
const metricTabs = document.querySelectorAll(".metric-tab");

const state = {
  metric: "outlier",
  harnesses: window.PROMPTBOOST_MATRIX.harnesses,
  models: window.PROMPTBOOST_MATRIX.models,
  rows: [],
  selectedKey: null
};

function setStatus(text, status = "muted") {
  matrixStatus.textContent = text;
  matrixStatus.className = `status-pill ${status}`.trim();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function loadDeterministicMatrix() {
  setStatus("berechnet", "muted");
  runDeterministic.disabled = true;
  runFullCell.disabled = true;
  try {
    const payload = await postJson("/api/matrix/deterministic", {
      raw_prompt: matrixPrompt.value
    });
    state.harnesses = payload.harnesses;
    state.models = payload.models;
    state.rows = payload.rows.map((row) => ({
      ...row,
      key: keyFor(row.harness, row.model),
      deterministic: row,
      agentic_full: null,
      diff: null
    }));
    state.selectedKey = state.rows.find((row) => row.harness === harnessSelect.value && row.model === modelSelect.value)?.key || state.rows[0]?.key;
    setStatus("Matrix bereit");
    matrixNote.textContent = "Deterministische Matrix geladen. Wähle eine Zelle und starte Full-Agentic für einen gezielten MiniMax-Vergleich.";
    renderAll();
  } catch (error) {
    setStatus("Fehler", "bad");
    matrixNote.textContent = error.message;
  } finally {
    runDeterministic.disabled = false;
    runFullCell.disabled = false;
  }
}

async function runSelectedFullCell() {
  const harness = harnessSelect.value;
  const model = modelSelect.value;
  state.selectedKey = keyFor(harness, model);
  setStatus("MiniMax läuft", "muted");
  runDeterministic.disabled = true;
  runFullCell.disabled = true;
  try {
    const payload = await postJson("/api/matrix/full-cell", {
      raw_prompt: matrixPrompt.value,
      target_harness: harness,
      target_model: model
    });
    upsertFullCell(payload);
    setStatus("Zelle aktualisiert");
    matrixNote.textContent = `${harness} × ${model}: Full-Agentic-Lauf abgeschlossen. Guard ${payload.agentic_full.no_new_facts_passed ? "bestanden" : "prüfen"}.`;
    renderAll();
  } catch (error) {
    setStatus("Fehler", "bad");
    matrixNote.textContent = error.message;
  } finally {
    runDeterministic.disabled = false;
    runFullCell.disabled = false;
  }
}

function upsertFullCell(payload) {
  const key = keyFor(payload.harness, payload.model);
  let row = state.rows.find((item) => item.key === key);
  if (!row) {
    row = {
      key,
      harness: payload.harness,
      model: payload.model,
      deterministic: payload.deterministic,
      score: payload.deterministic.score,
      grade: payload.deterministic.grade,
      input_adequacy: null,
      no_new_facts_passed: payload.deterministic.no_new_facts_passed,
      structure_score: payload.deterministic.structure_score,
      metrics: payload.deterministic.metrics,
      preview: payload.deterministic.preview,
      agentic_full: null,
      diff: null
    };
    state.rows.push(row);
  }
  row.deterministic = payload.deterministic;
  row.score = payload.deterministic.score;
  row.grade = payload.deterministic.grade;
  row.no_new_facts_passed = payload.deterministic.no_new_facts_passed;
  row.structure_score = payload.deterministic.structure_score;
  row.metrics = payload.deterministic.metrics;
  row.preview = payload.deterministic.preview;
  row.agentic_full = payload.agentic_full;
  row.diff = payload.diff;
  state.selectedKey = key;
}

function renderAll() {
  renderQuestion();
  renderSummary();
  renderFindings();
  renderMatrix();
  renderDetails();
}

function renderSummary() {
  const rows = state.rows;
  const analytics = analyzeRows(rows);
  const fullRows = rows.filter((row) => row.agentic_full);
  const guardPass = fullRows.filter((row) => row.agentic_full.no_new_facts_passed).length;
  const deterministicGuardPass = rows.filter((row) => row.no_new_facts_passed).length;
  const guardLabel = fullRows.length ? `${guardPass}/${fullRows.length}` : `${deterministicGuardPass}/${rows.length}`;
  matrixSummary.innerHTML = `
    <article><strong>${rows.length}</strong><span>Zellen</span></article>
    <article><strong>${guardLabel}</strong><span>Guard Pass</span></article>
    <article><strong>${Math.round(analytics.avgWords)}</strong><span>Ø Wörter</span></article>
    <article><strong>${fullRows.length}/${rows.length}</strong><span>Agentic gemessen</span></article>
  `;
}

function renderQuestion() {
  if (!matrixQuestion) return;
  const questions = {
    outlier: "Welche Zellen weichen stark ab?",
    words: "Wo entstehen lange Scaffolds?",
    structure: "Welche Kombis halten Struktur?",
    guard: "Wo ist Guard-Risiko?",
    delta: "Wo verändert Agentic stark?"
  };
  matrixQuestion.textContent = questions[state.metric] || questions.outlier;
}

function renderFindings() {
  if (!matrixFindings) return;
  const rows = state.rows;
  if (!rows.length) {
    matrixFindings.innerHTML = `<article class="finding-card"><span>Status</span><strong>Keine Matrix</strong><p>Deterministische Matrix starten.</p></article>`;
    return;
  }
  const analytics = analyzeRows(rows);
  const top = analytics.topOutlier;
  const highModel = analytics.modelStats[0];
  const lowModel = analytics.modelStats[analytics.modelStats.length - 1];
  const fullRows = rows.filter((row) => row.agentic_full);
  const risky = fullRows.find((row) => !row.agentic_full.no_new_facts_passed);
  const biggestDelta = fullRows
    .slice()
    .sort((a, b) => Math.abs(b.diff?.word_delta || 0) - Math.abs(a.diff?.word_delta || 0))[0];
  const cards = [];

  cards.push({
    label: "Auffälligste Zelle",
    value: cellName(top),
    meta: `${formatSigned(percentFrom(wordCount(top), analytics.avgWords))}% vs Ø · ${wordCount(top)} Wörter`,
    tone: wordCount(top) > analytics.avgWords ? "warn" : "info"
  });

  if (highModel && lowModel) {
    cards.push({
      label: "Modellmuster",
      value: `${highModel.name} ist am längsten`,
      meta: `Ø ${Math.round(highModel.avgWords)} Wörter · ${formatSigned(percentFrom(highModel.avgWords, lowModel.avgWords))}% vs ${lowModel.name}`,
      tone: "info"
    });
  }

  if (risky) {
    cards.push({
      label: "Agentic Risiko",
      value: cellName(risky),
      meta: `${wordCount(risky.agentic_full)} Wörter · Guard prüfen · Δ ${formatSigned(risky.diff?.word_delta || 0)}`,
      tone: "danger"
    });
  } else if (biggestDelta) {
    cards.push({
      label: "Agentic Evidenz",
      value: cellName(biggestDelta),
      meta: `stärkste Änderung: ${formatSigned(biggestDelta.diff?.word_delta || 0)} Wörter · Similarity ${biggestDelta.diff?.similarity || "n/a"}`,
      tone: "ok"
    });
  } else {
    cards.push({
      label: "Nächster Lauf",
      value: cellName(top),
      meta: `${fullRows.length}/${rows.length} agentisch gemessen · diese Outlier-Zelle zuerst gegen MiniMax prüfen.`,
      tone: "action"
    });
  }

  matrixFindings.innerHTML = cards.map(findingCardHtml).join("");
}

function renderMatrix() {
  const rowsByKey = new Map(state.rows.map((row) => [row.key, row]));
  const analytics = analyzeRows(state.rows);
  const maxDelta = Math.max(1, ...state.rows.map((row) => Math.abs(row.diff?.word_delta || 0)));
  matrixGrid.style.setProperty("--matrix-columns", state.models.length);
  matrixGrid.innerHTML = "";
  appendGridLabel("Harness / Model", "corner");
  for (const model of state.models) appendGridLabel(model, "head");
  for (const harness of state.harnesses) {
    appendGridLabel(harness, "head row");
    for (const model of state.models) {
      const row = rowsByKey.get(keyFor(harness, model));
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "matrix-lab-cell";
      cell.classList.toggle("is-selected", row?.key === state.selectedKey);
      cell.dataset.harness = harness;
      cell.dataset.model = model;
      if (!row) {
        cell.textContent = "-";
        cell.style.background = "rgba(80, 90, 110, 0.25)";
      } else {
        const { label, color } = cellMetric(row, { ...analytics, maxDelta });
        const badges = badgesFor(row, analytics).slice(0, 2);
        cell.innerHTML = `
          <span>${escapeHtml(label)}</span>
          ${badges.length ? `<small class="cell-badges">${badges.map((badge) => `<b class="${badge.tone}">${escapeHtml(badge.label)}</b>`).join("")}</small>` : ""}
        `;
        cell.style.background = color;
        cell.title = `${harness} x ${model}: ${wordCount(row)} Wörter, Struktur ${row.structure_score}/8`;
      }
      cell.addEventListener("click", () => selectCell(harness, model));
      matrixGrid.append(cell);
    }
  }
}

function cellMetric(row, context) {
  const words = wordCount(row);
  if (state.metric === "outlier") {
    const delta = words - context.avgWords;
    const pct = percentFrom(words, context.avgWords);
    return { label: `${formatSigned(pct)}%`, color: diverging(delta, context.maxOutlier) };
  }
  if (state.metric === "structure") {
    const score = row.agentic_full?.structure_score ?? row.structure_score ?? 0;
    return { label: `${score}/8`, color: heat(score, 8, "structure") };
  }
  if (state.metric === "guard") {
    if (!row.agentic_full) {
      return row.no_new_facts_passed
        ? { label: "PASS", color: "rgba(34, 197, 94, .48)" }
        : { label: "RISK", color: "rgba(244, 63, 94, .58)" };
    }
    return row.agentic_full.no_new_facts_passed
      ? { label: "PASS", color: "rgba(35, 197, 94, .76)" }
      : { label: "RISK", color: "rgba(244, 63, 94, .62)" };
  }
  if (state.metric === "delta") {
    const delta = row.diff?.word_delta || 0;
    return { label: row.diff ? formatSigned(delta) : "n/a", color: row.diff ? heat(Math.abs(delta), context.maxDelta, "delta") : "rgba(80, 90, 110, 0.25)" };
  }
  return { label: `${words}`, color: heat(words, context.maxWords, "words") };
}

function renderDetails() {
  const row = state.rows.find((item) => item.key === state.selectedKey);
  if (!row) {
    cellDetail.innerHTML = "";
    deterministicPreview.textContent = "Noch keine Zelle ausgewählt.";
    agenticPreview.textContent = "Noch kein Full-Agentic-Lauf.";
    return;
  }
  const analytics = analyzeRows(state.rows);
  harnessSelect.value = row.harness;
  modelSelect.value = row.model;
  const full = row.agentic_full;
  cellDetail.innerHTML = "";
  appendVerdict(cellDetail, verdictFor(row, analytics));
  appendDetail(cellDetail, "Kombination", `${row.harness} x ${row.model}`);
  appendDetail(cellDetail, "Matrix-Lage", `${formatSigned(percentFrom(wordCount(row), analytics.avgWords))}% vs Ø · ${wordCount(row)} Wörter · ${outlierLabel(row, analytics)}`);
  appendDetail(cellDetail, "Deterministisch", `${wordCount(row)} Wörter · Struktur ${row.structure_score}/8 · Score ${row.score}`);
  appendDetail(cellDetail, "Guard", row.no_new_facts_passed ? "Deterministisch bestanden" : "Deterministisch prüfen");
  if (full) {
    appendComparison(cellDetail, row);
    appendDetail(cellDetail, "Full-Agentic", `${wordCount(full)} Wörter · Struktur ${full.structure_score}/8 · ${Math.round(full.elapsed_ms / 100) / 10}s`);
    appendDetail(cellDetail, "Delta", `${formatSigned(row.diff.word_delta)} Wörter · Similarity ${row.diff.similarity}`);
    appendList(cellDetail, "Suspected New Facts", full.suspected_new_facts || []);
  } else {
    appendDetail(cellDetail, "Full-Agentic", "Für diese Zelle noch nicht gelaufen.");
  }
  deterministicPreview.textContent = row.preview || row.deterministic?.preview || "";
  agenticPreview.textContent = full?.preview || "Noch kein Full-Agentic-Lauf für diese Zelle.";
}

function selectCell(harness, model) {
  state.selectedKey = keyFor(harness, model);
  harnessSelect.value = harness;
  modelSelect.value = model;
  renderAll();
}

function appendGridLabel(text, className) {
  const item = document.createElement("div");
  item.className = `matrix-lab-label ${className}`;
  item.textContent = text;
  matrixGrid.append(item);
}

function appendDetail(target, label, value) {
  const item = document.createElement("article");
  item.className = "detail-item";
  item.innerHTML = `<strong>${escapeHtml(label)}</strong><p>${escapeHtml(value)}</p>`;
  target.append(item);
}

function appendVerdict(target, verdict) {
  const item = document.createElement("article");
  item.className = `verdict-card is-${verdict.tone}`;
  item.innerHTML = `
    <span>Empfehlung</span>
    <strong>${escapeHtml(verdict.label)}</strong>
    <p>${escapeHtml(verdict.reason)}</p>
  `;
  target.append(item);
}

function appendComparison(target, row) {
  const full = row.agentic_full;
  const item = document.createElement("article");
  item.className = "detail-item comparison-item";
  item.innerHTML = `
    <strong>Deterministisch vs Agentic</strong>
    <div class="comparison-grid">
      <span>Wörter</span><b>${wordCount(row)}</b><b>${wordCount(full)}</b><em>${formatSigned(row.diff?.word_delta || 0)}</em>
      <span>Struktur</span><b>${row.structure_score}/8</b><b>${full.structure_score}/8</b><em>${formatSigned(row.diff?.structure_delta || 0)}</em>
      <span>Guard</span><b>${row.no_new_facts_passed ? "PASS" : "RISK"}</b><b>${full.no_new_facts_passed ? "PASS" : "RISK"}</b><em>${full.no_new_facts_passed ? "ok" : "prüfen"}</em>
    </div>
  `;
  target.append(item);
}

function appendList(target, label, values) {
  const items = values.length ? values : ["Keine Einträge."];
  const item = document.createElement("article");
  item.className = "detail-item";
  item.innerHTML = `<strong>${escapeHtml(label)}</strong><ul>${items.map((value) => `<li>${escapeHtml(value)}</li>`).join("")}</ul>`;
  target.append(item);
}

function keyFor(harness, model) {
  return `${harness}::${model}`;
}

function analyzeRows(rows) {
  const avgWords = average(rows.map(wordCount));
  const maxWords = Math.max(1, ...rows.map(wordCount));
  const maxOutlier = Math.max(1, ...rows.map((row) => Math.abs(wordCount(row) - avgWords)));
  const harnessMaxWords = new Map();
  for (const row of rows) {
    const current = harnessMaxWords.get(row.harness);
    if (!current || wordCount(row) > wordCount(current)) harnessMaxWords.set(row.harness, row);
  }
  const topOutlier = rows.reduce((best, row) => {
    if (!best) return row;
    return Math.abs(wordCount(row) - avgWords) > Math.abs(wordCount(best) - avgWords) ? row : best;
  }, null);
  return {
    avgWords,
    maxWords,
    maxOutlier,
    harnessMaxWords,
    topOutlier,
    modelStats: groupStats(rows, "model"),
    harnessStats: groupStats(rows, "harness")
  };
}

function badgesFor(row, analytics) {
  const badges = [];
  if (row.key === analytics.topOutlier?.key) badges.push({ label: "OUT", tone: "warn" });
  if (analytics.harnessMaxWords.get(row.harness)?.key === row.key) badges.push({ label: "MAX", tone: "info" });
  if (row.agentic_full && !row.agentic_full.no_new_facts_passed) badges.push({ label: "RISK", tone: "danger" });
  if ((row.diff?.structure_delta || 0) < -1) badges.push({ label: "DROP", tone: "danger" });
  if (state.metric === "guard" && !row.agentic_full && !row.no_new_facts_passed) badges.push({ label: "RISK", tone: "danger" });
  return badges;
}

function verdictFor(row, analytics) {
  if (!row.agentic_full) {
    const isOutlier = Math.abs(wordCount(row) - analytics.avgWords) > analytics.avgWords * 0.2;
    if (isOutlier || analytics.harnessMaxWords.get(row.harness)?.key === row.key) {
      return {
        label: "Agentic testen",
        reason: "Deterministisch auffällig; diese Zelle ist ein guter Kandidat für den nächsten Full-Agentic-Vergleich.",
        tone: "action"
      };
    }
    return {
      label: "Deterministisch nutzen",
      reason: "Keine Agentic-Messung und kein starker Outlier. Der deterministische Scaffold ist der belastbare Baseline-Prompt.",
      tone: "ok"
    };
  }
  if (!row.agentic_full.no_new_facts_passed || (row.agentic_full.suspected_new_facts || []).length) {
    return {
      label: "Agentic meiden",
      reason: "Full-Agentic verletzt den Guard oder bringt verdächtige neue Fakten ein.",
      tone: "danger"
    };
  }
  if ((row.diff?.structure_delta || 0) < -1) {
    return {
      label: "Deterministisch nutzen",
      reason: "Agentic reduziert die Struktur deutlich. Der deterministische Prompt ist hier robuster.",
      tone: "warn"
    };
  }
  if ((row.diff?.word_delta || 0) <= 0 && (row.diff?.structure_delta || 0) >= 0) {
    return {
      label: "Agentic Kandidat",
      reason: "Agentic bleibt guard-konform, wird nicht länger und verliert keine Struktur.",
      tone: "ok"
    };
  }
  return {
    label: "Mehr Samples nötig",
    reason: "Agentic ist guard-konform, zeigt aber keinen klaren Qualitätsgewinn in dieser Einzelmessung.",
    tone: "info"
  };
}

function groupStats(rows, keyName) {
  const groups = new Map();
  for (const row of rows) {
    const key = row[keyName];
    const current = groups.get(key) || { name: key, count: 0, words: 0 };
    current.count += 1;
    current.words += wordCount(row);
    groups.set(key, current);
  }
  return [...groups.values()]
    .map((group) => ({ ...group, avgWords: group.words / group.count }))
    .sort((a, b) => b.avgWords - a.avgWords);
}

function wordCount(row) {
  return row?.metrics?.words || 0;
}

function cellName(row) {
  return row ? `${row.harness} × ${row.model}` : "n/a";
}

function percentFrom(value, base) {
  if (!base) return 0;
  return Math.round(((value - base) / base) * 100);
}

function outlierLabel(row, analytics) {
  const delta = wordCount(row) - analytics.avgWords;
  if (Math.abs(delta) < analytics.avgWords * 0.08) return "nahe am Matrix-Mittel";
  return delta > 0 ? "überdurchschnittlich lang" : "unterdurchschnittlich kurz";
}

function findingCardHtml(card) {
  return `
    <article class="finding-card ${card.tone ? `is-${card.tone}` : ""}">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
      <p>${escapeHtml(card.meta)}</p>
    </article>
  `;
}

function heat(value, max, paletteName) {
  const ratio = Math.min(1, Math.max(0, value / max));
  const palettes = {
    words: [[62, 95, 124], [50, 107, 95], [184, 79, 55]],
    structure: [[95, 109, 105], [62, 95, 124], [50, 107, 95]],
    delta: [[95, 109, 105], [184, 79, 55], [132, 46, 35]]
  };
  const palette = palettes[paletteName] || palettes.words;
  const scaled = ratio * (palette.length - 1);
  const index = Math.min(palette.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const [r1, g1, b1] = palette[index];
  const [r2, g2, b2] = palette[index + 1];
  const r = Math.round(r1 + (r2 - r1) * mix);
  const g = Math.round(g1 + (g2 - g1) * mix);
  const b = Math.round(b1 + (b2 - b1) * mix);
  return `rgb(${r}, ${g}, ${b})`;
}

function diverging(value, max) {
  const ratio = Math.min(1, Math.abs(value) / max);
  const neutral = [95, 109, 105];
  const negative = [62, 95, 124];
  const positive = [184, 79, 55];
  const target = value < 0 ? negative : positive;
  const mix = 0.18 + 0.82 * ratio;
  const r = Math.round(neutral[0] + (target[0] - neutral[0]) * mix);
  const g = Math.round(neutral[1] + (target[1] - neutral[1]) * mix);
  const b = Math.round(neutral[2] + (target[2] - neutral[2]) * mix);
  return `rgb(${r}, ${g}, ${b})`;
}

function average(values) {
  const real = values.filter((value) => Number.isFinite(value));
  if (!real.length) return 0;
  return real.reduce((sum, value) => sum + value, 0) / real.length;
}

function formatSigned(value) {
  return value > 0 ? `+${value}` : String(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

runDeterministic.addEventListener("click", loadDeterministicMatrix);
runFullCell.addEventListener("click", runSelectedFullCell);
metricTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    metricTabs.forEach((item) => item.classList.remove("is-active"));
    tab.classList.add("is-active");
    state.metric = tab.dataset.metric;
    renderAll();
  });
});
harnessSelect.addEventListener("change", () => selectCell(harnessSelect.value, modelSelect.value));
modelSelect.addEventListener("change", () => selectCell(harnessSelect.value, modelSelect.value));

loadDeterministicMatrix();
initMatrixTraceCanvas();

function initMatrixTraceCanvas() {
  const canvas = document.querySelector("#matrix-trace-canvas");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;

  const points = Array.from({ length: 22 }, (_, index) => ({
    x: Math.random(),
    y: Math.random(),
    dx: (Math.random() - 0.5) * 0.0007,
    dy: (Math.random() - 0.5) * 0.0007,
    radius: index % 4 === 0 ? 2.2 : 1.4
  }));

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * ratio;
    canvas.height = rect.height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function draw() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    context.clearRect(0, 0, width, height);

    for (const point of points) {
      point.x += point.dx;
      point.y += point.dy;
      if (point.x < 0 || point.x > 1) point.dx *= -1;
      if (point.y < 0 || point.y > 1) point.dy *= -1;
    }

    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const a = points[i];
        const b = points[j];
        const ax = a.x * width;
        const ay = a.y * height;
        const bx = b.x * width;
        const by = b.y * height;
        const distance = Math.hypot(ax - bx, ay - by);
        if (distance < 190) {
          context.strokeStyle = "rgba(50, 107, 95, 0.18)";
          context.lineWidth = 1;
          context.beginPath();
          context.moveTo(ax, ay);
          context.lineTo(bx, by);
          context.stroke();
        }
      }
    }

    for (const point of points) {
      context.fillStyle = point.radius > 2 ? "#b84f37" : "#326b5f";
      context.beginPath();
      context.arc(point.x * width, point.y * height, point.radius, 0, Math.PI * 2);
      context.fill();
    }

    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(draw);
}
