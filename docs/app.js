const CONFIG = {
  owner: "gitdhirajs",
  repo: "Azalyst-Crypto-Intelligence",
  runtimeRef: "runtime-data",
  refreshMs: 5 * 60 * 1000,
  bootstrapUrl: "./data/bootstrap.json",
};

const API_BASE = `https://api.github.com/repos/${CONFIG.owner}/${CONFIG.repo}`;
const WORKFLOW_CARDS = [
  { label: "Main Scanner", names: ["Main Scanner"] },
  { label: "Hourly Candle Patterns", names: ["Hourly Candle Patterns"] },
  { label: "Pages Deploy", names: ["pages build and deployment", "Deploy Dashboard"] },
];

const els = {
  liveStatus: document.getElementById("liveStatus"),
  clockValue: document.getElementById("clockValue"),
  repoVisibility: document.getElementById("repoVisibility"),
  headlineText: document.getElementById("headlineText"),
  footerStatus: document.getElementById("footerStatus"),
  tickerTape: document.getElementById("tickerTape"),
  workflowCards: document.getElementById("workflowCards"),
  runtimeSourceTag: document.getElementById("runtimeSourceTag"),
  runtimeStatusCard: document.getElementById("runtimeStatusCard"),
  mainModelSource: document.getElementById("mainModelSource"),
  hourlyModelSource: document.getElementById("hourlyModelSource"),
  mainModelCard: document.getElementById("mainModelCard"),
  hourlyModelCard: document.getElementById("hourlyModelCard"),
  hourlySignalsTable: document.getElementById("hourlySignalsTable"),
  fusedSignalsTable: document.getElementById("fusedSignalsTable"),
  summaryPreview: document.getElementById("summaryPreview"),
  mainFeatures: document.getElementById("mainFeatures"),
  hourlyFeatures: document.getElementById("hourlyFeatures"),
  mainClusters: document.getElementById("mainClusters"),
  hourlyClusters: document.getElementById("hourlyClusters"),
  mainBuckets: document.getElementById("mainBuckets"),
  hourlyBuckets: document.getElementById("hourlyBuckets"),
  workflowTable: document.getElementById("workflowTable"),
  sourceMatrix: document.getElementById("sourceMatrix"),
  runtimeFiles: document.getElementById("runtimeFiles"),
  issuesPanel: document.getElementById("issuesPanel"),
};

function updateClock() {
  const now = new Date();
  els.clockValue.textContent = now.toLocaleString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "Asia/Calcutta",
    timeZoneName: "short",
    hour12: false,
  });
}

function setLivePill(label, tone = "warn") {
  els.liveStatus.className = `pill pill-${tone}`;
  els.liveStatus.textContent = label;
}

function formatRelative(value) {
  if (!value) {
    return "n/a";
  }
  const stamp = new Date(value);
  const diffMs = Date.now() - stamp.getTime();
  if (Number.isNaN(diffMs)) {
    return "n/a";
  }
  const diffMin = Math.round(diffMs / 60000);
  if (Math.abs(diffMin) < 1) {
    return "just now";
  }
  if (Math.abs(diffMin) < 60) {
    return `${diffMin}m ago`;
  }
  const diffHr = Math.round(diffMin / 60);
  if (Math.abs(diffHr) < 24) {
    return `${diffHr}h ago`;
  }
  const diffDay = Math.round(diffHr / 24);
  return `${diffDay}d ago`;
}

function formatDate(value) {
  if (!value) {
    return "n/a";
  }
  const stamp = new Date(value);
  if (Number.isNaN(stamp.getTime())) {
    return "n/a";
  }
  return stamp.toLocaleString("en-IN", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Calcutta",
    timeZoneName: "short",
    hour12: false,
  });
}

function formatPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatDecimal(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

function statusTone(status) {
  const value = `${status || ""}`.toLowerCase();
  if (["healthy", "trained", "success", "completed"].includes(value)) {
    return "good";
  }
  if (["blocked", "failed", "error"].includes(value)) {
    return "bad";
  }
  if (["degraded", "no_data", "no_symbols", "queued", "in_progress", "requested"].includes(value)) {
    return "warn";
  }
  return "info";
}

function decodeGitHubContent(content) {
  const bytes = Uint8Array.from(atob(content.replace(/\n/g, "")), (char) =>
    char.charCodeAt(0)
  );
  return new TextDecoder("utf-8").decode(bytes);
}

async function fetchJSON(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (err) {
    console.warn(`Fetch failed for ${url}:`, err);
    return null;
  }
}

async function getRuntimeData() {
  const isLocal = window.location.protocol === 'file:' || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  
  // If local, try to fetch from the local reports directory first
  if (isLocal) {
    console.log("Local environment detected, fetching from local reports/");
    const fused = await fetchJSON('../reports/latest_fused_signals.json');
    if (fused) {
        return { 
          fused_signals: fused.signals || [],
          timestamp: fused.generated_at
        };
    }
  }

  // Fallback to GitHub API (Production/Remote)
  const repo = "gitdhirajs/Azalyst-Crypto-Intelligence";
  const branch = "runtime-data";
  const url = `https://api.github.com/repos/${repo}/contents/reports/latest_dashboard_payload.json?ref=${branch}`;
  
  const data = await fetchJSON(url);
  if (data?.content) {
    const payload = JSON.parse(atob(data.content));
    return payload;
  }
  return null;
}

async function fetchRepoJson(path, ref = "main") {
  // Use raw.githubusercontent.com to bypass API rate limits and complexity
  const url = `https://raw.githubusercontent.com/${CONFIG.owner}/${CONFIG.repo}/${ref}/${path}`;
  return await fetchJSON(url);
}

async function fetchRepoListing(path, ref = "main") {
  // For listings, we still need the API, but we can fallback gracefully
  const response = await fetchJSON(`${API_BASE}/contents/${path}?ref=${ref}`);
  return Array.isArray(response) ? response : [];
}

function latestRunByName(runs) {
  const latest = new Map();
  for (const run of runs || []) {
    if (!latest.has(run.name)) {
      latest.set(run.name, run);
    }
  }
  return latest;
}

function findLatestRun(latest, names) {
  for (const name of names) {
    if (latest.has(name)) {
      return latest.get(name);
    }
  }
  return null;
}

function mergePayloads(bootstrap, runtime) {
  const hasRuntime = Boolean(runtime);
  const runtimeMain = runtime?.main_report;
  const runtimeHourly = runtime?.hourly_report;
  const bootstrapMain = bootstrap?.main_report;
  const bootstrapHourly = bootstrap?.hourly_report;

  const mainReport = hasRuntime ? runtimeMain || null : bootstrapMain || null;
  const hourlyReport = hasRuntime ? runtimeHourly || null : bootstrapHourly || null;

  const hourlySignals = hasRuntime
    ? runtime?.hourly_live_signals || runtimeHourly?.top_live_signals || []
    : bootstrap?.hourly_live_signals || bootstrapHourly?.top_live_signals || [];

  return {
    runtime_status: runtime?.runtime_status || bootstrap?.runtime_status || {},
    main_report: mainReport,
    hourly_report: hourlyReport,
    latest_scan_signals:
      hasRuntime ? runtime?.latest_scan_signals || [] : bootstrap?.latest_scan_signals || [],
    hourly_live_signals: hourlySignals,
    fused_signals: hasRuntime ? runtime?.fused_signals || [] : bootstrap?.fused_signals || [],
    summary_markdown:
      runtime?.summary_markdown || bootstrap?.summary_markdown || "No summary published yet.",
    mainReportSource:
      hasRuntime ? "runtime-data" : bootstrapMain ? "local bootstrap" : "missing",
    hourlyReportSource:
      hasRuntime ? "runtime-data" : bootstrapHourly ? "local bootstrap" : "missing",
  };
}

function renderTicker(repo, runtimeStatus, latestWorkflows) {
  const scanner = runtimeStatus?.scanner || {};
  const main = runtimeStatus?.main_model || {};
  const hourly = runtimeStatus?.hourly_model || {};
  const items = [
    `REPO ${repo?.full_name || `${CONFIG.owner}/${CONFIG.repo}`}`,
    `SCANNER ${scanner.status || "unknown"}`,
    `ROWS ${scanner.symbols_scanned ?? "n/a"}`,
    `FAILED ${scanner.symbols_failed ?? "n/a"}`,
    `MAIN ${main.status || "missing"}`,
    `MAIN ACC ${main.accuracy != null ? formatPct(main.accuracy) : "n/a"}`,
    `HOURLY ${hourly.status || "missing"}`,
    `HOURLY AUC ${hourly.roc_auc != null ? formatDecimal(hourly.roc_auc, 3) : "n/a"}`,
  ];

  for (const item of WORKFLOW_CARDS) {
    const run = findLatestRun(latestWorkflows, item.names);
    if (run) {
      items.push(`${item.label.toUpperCase()} ${run.conclusion || run.status}`);
    }
  }

  els.tickerTape.innerHTML = items.map((item) => `<span>${item}</span>`).join("");
}

function renderWorkflowCards(runs) {
  const latest = latestRunByName(runs);
  const cards = WORKFLOW_CARDS.map((item) => {
    const run = findLatestRun(latest, item.names);
    if (!run) {
      return `
        <div class="metric-card">
          <span class="metric-label">${item.label}</span>
          <div class="metric-value">No runs yet</div>
          <div class="metric-subtle">Waiting for first execution.</div>
        </div>
      `;
    }

    const state = run.conclusion || run.status || "unknown";
    return `
      <div class="metric-card">
        <div class="meta-line">
          <strong>${item.label}</strong>
          <span class="pill pill-${statusTone(state)}">${state}</span>
        </div>
        <div class="metric-subtle">
          Run #${run.run_number} &middot; ${formatRelative(run.updated_at)} &middot; ${formatDate(run.updated_at)}
        </div>
        <div class="metric-subtle">
          <a class="runtime-link" href="${run.html_url}" target="_blank" rel="noreferrer">View run</a>
        </div>
      </div>
    `;
  });
  els.workflowCards.innerHTML = cards.join("");
  return latest;
}

function renderRuntimeStatus(runtimeStatus, usedRuntimePayload) {
  const scanner = runtimeStatus?.scanner || {};
  const requestErrors = scanner.request_errors || {};
  const status = scanner.status || "unknown";
  els.runtimeSourceTag.textContent = usedRuntimePayload ? "RUNTIME-DATA LIVE" : "BOOTSTRAP FALLBACK";

  els.runtimeStatusCard.innerHTML = `
    <div class="stats-grid">
      <div class="stat-box">
        <span class="metric-label">Status</span>
        <div class="metric-value">${status}</div>
        <div class="metric-subtle">${scanner.headline || "No scanner headline available."}</div>
      </div>
      <div class="stat-box">
        <span class="metric-label">Last Update</span>
        <div class="metric-value">${formatRelative(scanner.timestamp || runtimeStatus?.generated_at)}</div>
        <div class="metric-subtle">${formatDate(scanner.timestamp || runtimeStatus?.generated_at)}</div>
      </div>
      <div class="stat-box">
        <span class="metric-label">Rows</span>
        <div class="metric-value">${formatNumber(scanner.symbols_scanned || 0)}</div>
        <div class="metric-subtle">attempted ${formatNumber(scanner.symbols_attempted || 0)}</div>
      </div>
      <div class="stat-box">
        <span class="metric-label">Failures</span>
        <div class="metric-value">${formatNumber(scanner.symbols_failed || 0)}</div>
        <div class="metric-subtle">403s ${requestErrors.status_counts?.["403"] || 0} / 451s ${requestErrors.status_counts?.["451"] || 0}</div>
      </div>
    </div>
    <div class="meta-list">
      <div class="meta-line">
        <span>Min 24h volume filter</span>
        <strong>${formatNumber(scanner.min_volume_24h_usdt || 0)}</strong>
      </div>
      <div class="meta-line">
        <span>Known issue signal</span>
        <strong>${requestErrors.status_counts?.["403"] || requestErrors.status_counts?.["451"] ? "Provider block observed" : "No provider block captured"}</strong>
      </div>
      <div class="meta-line">
        <span>Endpoints with errors</span>
        <strong>${Object.keys(requestErrors.endpoint_counts || {}).length || 0}</strong>
      </div>
    </div>
  `;
}

function renderModelCard(target, sourceTarget, report, sourceLabel, type) {
  sourceTarget.textContent = sourceLabel.toUpperCase();

  if (!report) {
    target.innerHTML = `<div class="empty-state">No ${type} model snapshot available yet.</div>`;
    return;
  }

  const topFeature = report.top_features?.[0];
  const extraLabel =
    type === "main"
      ? `Label horizon ${report.label_horizon_minutes ?? "n/a"}m`
      : `Continuation window ${report.continuation_window_hours ?? "n/a"}h`;

  target.innerHTML = `
    <div class="stats-grid">
      <div class="stat-box">
        <span class="metric-label">Status</span>
        <div class="metric-value">${report.status || "unknown"}</div>
        <div class="metric-subtle">${formatDate(report.timestamp)}</div>
      </div>
      <div class="stat-box">
        <span class="metric-label">Accuracy</span>
        <div class="metric-value">${report.accuracy != null ? formatPct(report.accuracy) : "n/a"}</div>
        <div class="metric-subtle">baseline ${report.baseline_accuracy != null ? formatPct(report.baseline_accuracy) : "n/a"}</div>
      </div>
      <div class="stat-box">
        <span class="metric-label">ROC AUC</span>
        <div class="metric-value">${report.roc_auc != null ? formatDecimal(report.roc_auc, 3) : "n/a"}</div>
        <div class="metric-subtle">F1 ${report.f1_score != null ? formatDecimal(report.f1_score, 3) : "n/a"}</div>
      </div>
      <div class="stat-box">
        <span class="metric-label">Samples</span>
        <div class="metric-value">${formatNumber(report.n_samples || 0)}</div>
        <div class="metric-subtle">symbols ${formatNumber(report.n_symbols || 0)}</div>
      </div>
    </div>
    <div class="meta-list">
      <div class="meta-line">
        <span>Evaluation split</span>
        <strong>${report.evaluation_split || "time-ordered"}</strong>
      </div>
      <div class="meta-line">
        <span>Lead feature</span>
        <strong>${topFeature ? `${topFeature[0]} (${formatDecimal(topFeature[1], 3)})` : "n/a"}</strong>
      </div>
      <div class="meta-line">
        <span>Training note</span>
        <strong>${extraLabel}</strong>
      </div>
    </div>
  `;
}

function renderSignals(signals) {
  if (!signals?.length) {
    els.hourlySignalsTable.innerHTML = `<div class="empty-state">No hourly live signals available.</div>`;
    return;
  }

  const rows = signals.slice(0, 8).map((signal) => {
    const probability = signal.continuation_probability ?? signal.ml_probability ?? 0;
    return `
      <div class="signal-row">
        <strong>${signal.symbol}</strong>
        <span>${formatDecimal(probability * 100, 1)}%</span>
        <span>${formatDecimal(signal.rsi_1h, 1)}</span>
        <span>${formatDecimal(signal.body_pct, 2)}</span>
        <span>${formatDecimal(signal.oi_change_pct_1h, 2)}</span>
      </div>
    `;
  });

  els.hourlySignalsTable.innerHTML = `
    <div class="signal-table">
      <div class="signal-head">
        <span>Symbol</span>
        <span>Prob</span>
        <span>RSI 1H</span>
        <span>Body %</span>
        <span>OI 1H %</span>
      </div>
      ${rows.join("")}
    </div>
  `;
}

function renderFusedSignals(signals) {
  if (!signals?.length) {
    els.fusedSignalsTable.innerHTML = `<div class="empty-state">No institutional fused signals available.</div>`;
    return;
  }

  els.fusedSignalsTable.innerHTML = signals.map((signal) => {
    const tone = signal.consensus_tier === 'A' ? 'good' : (signal.consensus_tier === 'B' ? 'info' : 'warn');
    const eduImg = signal.edu_frame ? `
      <div class="edu-frame-container">
        <div class="edu-title">Methodology Evidence</div>
        <img src="${signal.edu_frame}" class="edu-thumb" alt="Educational Frame" onclick="window.open(this.src)">
        <div class="edu-caption">Blueprint v7 methodology visual alignment</div>
      </div>
    ` : '';

    return `
      <div class="metric-card" style="margin-bottom: 1rem; border-left: 3px solid var(--pill-${tone}-color, #222);">
        <div class="signal-row" style="border-bottom: 1px solid #111; margin-bottom: 8px;">
          <strong>${signal.symbol}</strong>
          <span class="pill pill-${statusTone(signal.direction === 'LONG' ? 'healthy' : 'error')}">${signal.direction}</span>
          <span class="pill pill-${tone}">Tier ${signal.consensus_tier}</span>
          <strong>${formatDecimal(signal.fused_score, 1)}</strong>
        </div>
        <div class="micro-copy" style="margin-bottom: 8px;">
          Agreement: ${formatDecimal(signal.metrics?.agreement_factor || 1, 2)} | 
          Engines: ${signal.engines_long.length + signal.engines_short.length}
        </div>
        ${eduImg}
      </div>
    `;
  }).join("");
}

function renderFeatures(target, report) {
    const features = report?.top_features || report?.top_features_permutation || [];
    if (features.length === 0) {
      target.innerHTML = `<div class="empty-state">No feature importance data available.</div>`;
      return;
    }

  const maxValue = Math.max(...features.map((item) => Number(item[1]) || 0), 0.0001);
  target.innerHTML = features.slice(0, 10).map(([name, value]) => `
    <div class="feature-row">
      <span>${name}</span>
      <div class="bar-rail">
        <div class="bar-fill" style="width:${(Number(value) / maxValue) * 100}%"></div>
      </div>
      <strong>${formatDecimal(value, 3)}</strong>
    </div>
  `).join("");
}

function renderClusters(target, clusters, positiveKey, negativeKey, labelKey) {
  if (!clusters) {
    target.innerHTML = `<div class="empty-state">No cluster analysis available.</div>`;
    return;
  }

  const positives = clusters[positiveKey] || [];
  const negatives = clusters[negativeKey] || [];

  const renderSet = (title, items, metricLabel) => `
    <div class="cluster-card">
      <h4>${title}</h4>
      ${
        items.length
          ? items
              .slice(0, 4)
              .map((item) => `
                <div class="list-row">
                  <span>${item.cluster}</span>
                  <strong>${formatNumber(item.count || 0)}</strong>
                  <span>${formatDecimal(item[labelKey], 3)} ${metricLabel}</span>
                </div>
              `)
              .join("")
          : `<p class="muted">No regimes met the minimum sample threshold.</p>`
      }
    </div>
  `;

  target.innerHTML = `
    <div class="stack">
      ${renderSet("Positive clusters", positives, labelKey === "win_rate" ? "win rate" : "continue")}
      ${renderSet("Negative clusters", negatives, labelKey === "win_rate" ? "win rate" : "continue")}
    </div>
  `;
}

function renderBuckets(target, bucketAnalysis, mode) {
  if (!bucketAnalysis) {
    target.innerHTML = `<div class="empty-state">No bucket analysis available.</div>`;
    return;
  }

  const sections = Object.entries(bucketAnalysis)
    .filter(([key]) => !key.includes("clusters"))
    .slice(0, 4)
    .map(([feature, payload]) => {
      const topKey = mode === "main" ? "top_bullish_ranges" : "top_continuation_ranges";
      const items = payload[topKey] || [];
      return `
        <div class="cluster-card">
          <h4>${feature}</h4>
          ${
            items.length
              ? items
                  .slice(0, 4)
                  .map((item) => `
                    <div class="list-row">
                      <span>${item.bucket}</span>
                      <strong>${formatNumber(item.count || 0)}</strong>
                      <span>${formatDecimal(item.win_rate ?? item.continuation_rate, 3)}</span>
                    </div>
                  `)
                  .join("")
              : `<p class="muted">No qualifying range buckets.</p>`
          }
        </div>
      `;
    });

  target.innerHTML = sections.length ? sections.join("") : `<div class="empty-state">No bucket analysis available.</div>`;
}

function renderWorkflowTable(runs) {
  if (!runs?.length) {
    els.workflowTable.innerHTML = `<div class="empty-state">No workflow history available.</div>`;
    return;
  }

  els.workflowTable.innerHTML = `
    <div class="table">
      <div class="table-head">
        <span>Workflow</span>
        <span>State</span>
        <span>Event</span>
        <span>Updated</span>
      </div>
      ${runs.slice(0, 8).map((run) => `
        <div class="table-row">
          <span><a class="runtime-link" href="${run.html_url}" target="_blank" rel="noreferrer">${run.name}</a></span>
          <span class="${statusTone(run.conclusion || run.status) === "good" ? "positive" : statusTone(run.conclusion || run.status) === "bad" ? "negative" : "warning"}">${run.conclusion || run.status}</span>
          <span>${run.event}</span>
          <span>${formatRelative(run.updated_at)}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function renderSourceMatrix(bootstrap, runtimePayload, runs, repo) {
  const sources = [
    {
      name: "Local bootstrap snapshot",
      state: bootstrap ? "available" : "missing",
      detail: bootstrap?.generated_at ? `Generated ${formatDate(bootstrap.generated_at)}` : "Bundled with dashboard",
    },
    {
      name: "Runtime payload branch",
      state: runtimePayload ? "available" : "missing",
      detail: runtimePayload?.generated_at ? `Runtime generated ${formatDate(runtimePayload.generated_at)}` : "Waiting for reports/latest_dashboard_payload.json",
    },
    {
      name: "GitHub Actions API",
      state: runs?.length ? "available" : "missing",
      detail: runs?.length ? `${runs.length} recent workflow runs loaded` : "Could not read workflow history",
    },
    {
      name: "Repository metadata",
      state: repo ? "available" : "missing",
      detail: repo?.pushed_at ? `Last push ${formatDate(repo.pushed_at)}` : "Could not read repo metadata",
    },
  ];

  els.sourceMatrix.innerHTML = `
    <div class="source-table">
      <div class="source-head">
        <span>Source</span>
        <span>State</span>
        <span>Detail</span>
      </div>
      ${sources.map((item) => `
        <div class="source-row">
          <span>${item.name}</span>
          <span class="pill pill-${item.state === "available" ? "good" : "warn"}">${item.state}</span>
          <span>${item.detail}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function renderRuntimeFiles(files) {
  if (!files?.length) {
    els.runtimeFiles.innerHTML = `<div class="empty-state">No runtime report files are visible yet.</div>`;
    return;
  }

  els.runtimeFiles.innerHTML = `
    <div class="file-list">
      ${files.map((file) => `
        <div class="file-row">
          <span>${file.name}</span>
          <a class="runtime-link" href="${file.html_url}" target="_blank" rel="noreferrer">Open</a>
        </div>
      `).join("")}
    </div>
  `;
}

function renderIssues(runtimeStatus, latestWorkflows, payload) {
  const notes = [...(runtimeStatus?.notes || [])];
  const scanner = runtimeStatus?.scanner || {};

  if (scanner.status === "blocked") {
    notes.unshift(
      "GitHub-hosted runners are succeeding as jobs, but the market-data provider is rejecting the data requests."
    );
  }

  if (payload.mainReportSource === "local bootstrap") {
    notes.push("Main model metrics are coming from the last local training snapshot because the hosted runner is not producing fresh market data.");
  }

  if (payload.hourlyReportSource === "local bootstrap") {
    notes.push("Hourly candle-pattern metrics are also being shown from the local bootstrap snapshot.");
  }

  const deployRun = findLatestRun(latestWorkflows, ["pages build and deployment", "Deploy Dashboard"]);
  if (!deployRun) {
    notes.push("GitHub Pages deployment history has not been observed yet; the site may still be bootstrapping.");
  }

  els.issuesPanel.innerHTML = notes.length
    ? notes.map((note, index) => `
        <div class="issue-card">
          <h4>Note ${index + 1}</h4>
          <p>${note}</p>
        </div>
      `).join("")
    : `<div class="empty-state">No open operational notes.</div>`;
}

function renderShell(repo, payload, runs, runtimeFiles, bootstrap, runtimePayload) {
  const runtimeStatus = payload.runtime_status || {};
  const scanner = runtimeStatus.scanner || {};
  const headline =
    scanner.headline ||
    runtimeStatus.notes?.[0] ||
    "Dashboard is connected to repository telemetry.";

  els.repoVisibility.textContent = repo?.private ? "PRIVATE REPO" : "PUBLIC REPO";
  els.headlineText.textContent = headline;
  els.summaryPreview.textContent = payload.summary_markdown;
  els.footerStatus.textContent = `${headline} | Last repo push ${formatRelative(repo?.pushed_at)}`;

  if (scanner.status === "healthy") {
    setLivePill("LIVE", "good");
  } else if (scanner.status === "blocked") {
    setLivePill("BLOCKED", "bad");
  } else {
    setLivePill("DEGRADED", "warn");
  }

  renderRuntimeStatus(runtimeStatus, Boolean(runtimePayload));
  renderModelCard(els.mainModelCard, els.mainModelSource, payload.main_report, payload.mainReportSource, "main");
  renderModelCard(
    els.hourlyModelCard,
    els.hourlyModelSource,
    payload.hourly_report,
    payload.hourlyReportSource,
    "hourly"
  );
  renderFusedSignals(payload.fused_signals);
  renderSignals(payload.hourly_live_signals);
  renderFeatures(els.mainFeatures, payload.main_report);
  renderFeatures(els.hourlyFeatures, payload.hourly_report);
  renderClusters(
    els.mainClusters,
    payload.main_report?.bucket_analysis?.scanner_clusters,
    "top_gainer_clusters",
    "top_loser_clusters",
    "win_rate"
  );
  renderClusters(
    els.hourlyClusters,
    payload.hourly_report?.bucket_analysis?.pattern_clusters,
    "top_gainer_clusters",
    "top_failure_clusters",
    "continuation_rate"
  );
  renderBuckets(els.mainBuckets, payload.main_report?.bucket_analysis, "main");
  renderBuckets(els.hourlyBuckets, payload.hourly_report?.bucket_analysis, "hourly");
  renderWorkflowTable(runs);
  renderSourceMatrix(bootstrap, runtimePayload, runs, repo);
  renderRuntimeFiles(runtimeFiles);
  const latestWorkflows = renderWorkflowCards(runs);
  renderTicker(repo, runtimeStatus, latestWorkflows);
  renderIssues(runtimeStatus, latestWorkflows, payload);
}

async function loadDashboard() {
  const [bootstrap, runtimePayload, workflowResponse, repo, runtimeFiles] = await Promise.all([
    fetchJSON(CONFIG.bootstrapUrl),
    fetchRepoJson("reports/latest_dashboard_payload.json", CONFIG.runtimeRef),
    fetchJSON(`${API_BASE}/actions/runs?per_page=8`),
    fetchJSON(API_BASE),
    fetchRepoListing("reports", CONFIG.runtimeRef),
  ]);

  const runs = workflowResponse?.workflow_runs || [];
  const payload = mergePayloads(bootstrap, runtimePayload);
  renderShell(repo, payload, runs, runtimeFiles, bootstrap, runtimePayload);
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((node) => node.classList.remove("active"));
      panels.forEach((panel) => panel.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`tab-${tab.dataset.tab}`)?.classList.add("active");
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  updateClock();
  loadDashboard();
  setInterval(updateClock, 1000);
  setInterval(loadDashboard, CONFIG.refreshMs);
});
