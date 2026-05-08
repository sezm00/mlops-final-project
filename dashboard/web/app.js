const PAGES = {
  overview: renderOverview,
  monitoring: renderMonitoring,
  drift: renderDrift,
  models: renderModels,
  features: renderFeatures,
  dvc: renderDvc,
  registry: renderRegistry,
  reports: renderReports,
};

function fmt(v, d = 3) {
  if (v === null || v === undefined || v === '') return '-';
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : String(v);
}
function fmtInt(v) {
  if (v === null || v === undefined || v === '') return '-';
  const n = Number(v);
  return Number.isFinite(n) ? String(Math.round(n)) : String(v);
}
function pct(v, d = 1) {
  if (v === null || v === undefined || v === '') return '-';
  const n = Number(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(d)}%` : String(v);
}
function shortText(v, n = 28) {
  const s = String(v ?? '-');
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}
function exists(data, key) { return data.availability?.[key]?.exists; }
function badge(text, type = 'info') { return `<span class="badge ${type}">${text}</span>`; }
function safeJson(x) { return JSON.stringify(x ?? {}, null, 2); }
function page() { return document.body.dataset.page || 'overview'; }
function sourcePill(src) { return `<span class="badge ${src === 'live_repo_file' ? 'good' : src === 'not_found' ? 'bad' : 'warn'}">${src || 'unknown'}</span>`; }
function arr(v){ return Array.isArray(v) ? v : []; }
function obj(v){ return v && typeof v === 'object' ? v : {}; }
function avg(nums){ const a = nums.filter(x => Number.isFinite(Number(x))).map(Number); return a.length ? a.reduce((s,x)=>s+x,0)/a.length : null; }
function sum(nums){ return nums.filter(x => Number.isFinite(Number(x))).map(Number).reduce((s,x)=>s+x,0); }
function latest(list){ return list?.length ? list[list.length - 1] : null; }

async function loadSummary() {
  const res = await fetch('/api/summary', { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to load /api/summary');
  return await res.json();
}
async function loadHistory() {
  const res = await fetch('/api/batch/history', { cache: 'no-store' });
  return res.ok ? await res.json() : [];
}

function setRoot(data) {
  document.querySelectorAll('[data-root]').forEach(el => el.textContent = data.project_root);
}

function setStatus(data) {
  const status = document.getElementById('status-strip');
  if (!status) return;
  const items = [
    ['Monitoring summary', exists(data, 'monitoring_summary')],
    ['Drift report', exists(data, 'drift_report_html')],
    ['DVC YAML', exists(data, 'dvc_yaml')],
    ['DVC lock', exists(data, 'dvc_lock')],
    ['Model file', exists(data, 'best_model_pkl')],
    ['Reports / notebook fallback', exists(data, 'metrics') || !!data.metrics],
  ];
  status.innerHTML = items.map(([label, ok]) => `<span class="pill"><span class="dot ${ok ? 'ok' : 'bad'}"></span>${label}</span>`).join('');
}

function table(headers, rows) {
  if (!rows?.length) return `<div class="json">No data found.</div>`;
  return `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function kpi(label, value, note, type = '') {
  const compact = String(value ?? '').length > 18 ? 'compact' : '';
  return `<div class="card kpi"><div class="label">${label}</div><div class="value ${type} ${compact}">${value}</div><div class="note">${note}</div></div>`;
}

function baseShell(title, subtitle) {
  return `<section class="hero"><div><div class="eyebrow">Telco Churn MLOps Pipeline</div><h2>${title}</h2><p>${subtitle}</p></div><div class="status-strip" id="status-strip"></div></section>`;
}

function finalMetrics(data) {
  return obj(data.best_model_summary?.final_metrics || data.best_model_summary?.all_features_metrics || data.metrics || {});
}

function normalizeModelRow(r) {
  return {
    model_name: r.model_name || r.model || r.name || 'model',
    f1_macro: Number(r.f1_macro ?? r.score ?? r.test_score ?? r.test_f1_macro ?? 0),
    verdict: r.quality_verdict || r.verdict || '-',
    gap: Number(r.train_test_gap ?? r.gap ?? 0),
    train_score: Number(r.train_score ?? 0),
    test_score: Number(r.test_score ?? r.f1_macro ?? 0),
  };
}

function bars(rows, labelKey, valueKey, max = 1, alt = false) {
  if (!rows?.length) return '<div class="json">No data found.</div>';
  const safeMax = max > 0 ? max : Math.max(...rows.map(r => Number(r[valueKey] ?? 0)), 1);
  return `<div class="chart">${rows.map(r => {
    const value = Number(r[valueKey] ?? 0);
    const w = Math.max(0, Math.min(100, (value / safeMax) * 100));
    return `<div class="bar-row"><div class="bar-name" title="${r[labelKey]}">${r[labelKey]}</div><div class="bar-track"><div class="bar-fill ${alt ? 'alt' : ''}" style="width:${w}%"></div></div><div class="bar-val">${fmt(value)}</div></div>`;
  }).join('')}</div>`;
}

function progressList(items) {
  const safeItems = arr(items);
  if (!safeItems.length) return '<div class="json">No metrics found.</div>';
  return `<div class="progress-list">${safeItems.map(item => {
    const val = Math.max(0, Math.min(1, Number(item.value ?? 0)));
    return `<div class="progress-item"><div class="name">${item.label}</div><div class="track"><div class="fill" style="width:${val * 100}%"></div></div><div class="val">${item.display ?? fmt(val)}</div></div>`;
  }).join('')}</div>`;
}

function chips(values, type='info') {
  const list = arr(values);
  if (!list.length) return '<div class="json">No data found.</div>';
  return `<div class="chips">${list.map(v => badge(String(v), type)).join('')}</div>`;
}

function lineChart(title, subtitle, labels, series, opts = {}) {
  const safeLabels = arr(labels);
  const safeSeries = arr(series).filter(s => arr(s.values).length);
  if (!safeSeries.length) {
    return `<div class="spark-wrap"><div class="spark-title"><h4>${title}</h4><span>${subtitle}</span></div><div class="json">No trend data available yet.</div></div>`;
  }
  const width = opts.width || 820;
  const height = opts.height || 180;
  const pad = 28;
  let all = [];
  safeSeries.forEach(s => all = all.concat(arr(s.values).map(Number).filter(Number.isFinite)));
  const min = Number.isFinite(opts.min) ? opts.min : Math.min(...all);
  const max = Number.isFinite(opts.max) ? opts.max : Math.max(...all);
  const range = (max - min) || 1;
  const maxLen = Math.max(...safeSeries.map(s => arr(s.values).length), 1);
  const xFor = i => pad + ((width - pad * 2) * (maxLen === 1 ? 0.5 : i / (maxLen - 1)));
  const yFor = v => height - pad - ((Number(v) - min) / range) * (height - pad * 2);
  const grid = [0, 0.25, 0.5, 0.75, 1].map(frac => {
    const y = pad + frac * (height - pad * 2);
    return `<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="rgba(255,255,255,.08)" stroke-width="1" />`;
  }).join('');
  const seriesSvg = safeSeries.map(s => {
    const values = arr(s.values);
    const points = values.map((v, i) => `${xFor(i)},${yFor(v)}`).join(' ');
    const circles = values.map((v, i) => `<circle cx="${xFor(i)}" cy="${yFor(v)}" r="3" fill="${s.color}" />`).join('');
    return `<polyline fill="none" stroke="${s.color}" stroke-width="3" points="${points}" stroke-linecap="round" stroke-linejoin="round" />${circles}`;
  }).join('');
  const xLabels = safeLabels.slice(0, maxLen).map((label, i) => `<text x="${xFor(i)}" y="${height - 6}" font-size="11" fill="#9fb0cc" text-anchor="middle">${shortText(label, 10)}</text>`).join('');
  const yLabels = [max, min + range * 0.5, min].map((v, i) => `<text x="6" y="${pad + i * ((height - pad * 2) / 2) + 4}" font-size="11" fill="#9fb0cc">${fmt(v)}</text>`).join('');
  return `<div class="spark-wrap"><div class="spark-title"><h4>${title}</h4><span>${subtitle}</span></div><svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${grid}${seriesSvg}${xLabels}${yLabels}</svg><div class="legend">${safeSeries.map(s => `<span class="legend-item"><span class="swatch" style="background:${s.color}"></span>${s.name}</span>`).join('')}</div></div>`;
}

function metricBoxes(metrics) {
  const m = obj(metrics);
  const rows = [
    ['Accuracy', m.accuracy],
    ['Precision macro', m.precision_macro],
    ['Recall macro', m.recall_macro],
    ['F1 macro', m.f1_macro],
    ['ROC-AUC', m.roc_auc],
    ['Problem type', m.problem_type],
  ];
  return `<div class="metric-grid">${rows.map(([label, value]) => `<div class="metric-box"><div class="label">${label}</div><div class="value ${label === 'Problem type' ? 'compact' : ''}">${label === 'Problem type' ? value ?? '-' : fmt(value)}</div></div>`).join('')}</div>`;
}


function metricValueSet(metrics) {
  const m = obj(metrics);
  return {
    Accuracy: Number(m.accuracy || 0),
    Precision: Number(m.precision_macro || 0),
    Recall: Number(m.recall_macro || 0),
    "F1 Macro": Number(m.f1_macro || 0),
    "ROC-AUC": Number(m.roc_auc || 0),
  };
}

function modelMetricComparisonChart(finalMetrics, allMetrics, selectedMetrics) {
  const groups = [
    { name: "Final", values: metricValueSet(finalMetrics), color: "#22d3ee" },
    { name: "All features", values: metricValueSet(allMetrics), color: "#60a5fa" },
    { name: "Selected features", values: metricValueSet(selectedMetrics), color: "#a78bfa" },
  ];
  const metricNames = ["Accuracy", "Precision", "Recall", "F1 Macro", "ROC-AUC"];
  const width = 980;
  const height = 340;
  const pad = { left: 58, right: 28, top: 28, bottom: 58 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const groupW = chartW / metricNames.length;
  const barW = 18;
  const gap = 6;
  const max = 1;

  const grid = [0, .25, .5, .75, 1].map(v => {
    const y = pad.top + chartH - v * chartH;
    return `<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="rgba(255,255,255,.09)" />
            <text x="12" y="${y + 4}" font-size="11" fill="#9fb0cc">${v.toFixed(2)}</text>`;
  }).join("");

  const barsSvg = metricNames.map((metric, i) => {
    const center = pad.left + groupW * i + groupW / 2;
    const totalBarW = groups.length * barW + (groups.length - 1) * gap;
    const startX = center - totalBarW / 2;

    const bars = groups.map((g, j) => {
      const val = Number(g.values[metric] || 0);
      const h = val / max * chartH;
      const x = startX + j * (barW + gap);
      const y = pad.top + chartH - h;
      return `<rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="6" fill="${g.color}">
                <title>${g.name} ${metric}: ${fmt(val)}</title>
              </rect>`;
    }).join("");

    return `${bars}<text x="${center}" y="${height - 22}" font-size="12" fill="#cbd5e1" text-anchor="middle">${metric}</text>`;
  }).join("");

  const legend = groups.map((g, i) => {
    const x = pad.left + i * 150;
    return `<rect x="${x}" y="8" width="10" height="10" rx="3" fill="${g.color}" />
            <text x="${x + 16}" y="18" font-size="12" fill="#dbeafe">${g.name}</text>`;
  }).join("");

  return `<div class="spark-wrap">
    <div class="spark-title"><h4>Grouped metric comparison</h4><span>Final vs all-features vs selected-features</span></div>
    <svg class="spark tall" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${grid}${barsSvg}${legend}</svg>
  </div>`;
}

function modelMetricLineChart(finalMetrics, allMetrics, selectedMetrics) {
  const metricNames = ["Accuracy", "Precision", "Recall", "F1 Macro", "ROC-AUC"];
  const datasets = [
    { name: "Final", color: "#22d3ee", values: Object.values(metricValueSet(finalMetrics)) },
    { name: "All features", color: "#60a5fa", values: Object.values(metricValueSet(allMetrics)) },
    { name: "Selected features", color: "#a78bfa", values: Object.values(metricValueSet(selectedMetrics)) },
  ];
  return lineChart("Metric profile line chart", "Scores should stay high and stable across all core metrics.", metricNames, datasets, { min: .80, max: 1.0, height: 230 });
}

function modelMetricDeltaTable(finalMetrics, allMetrics, selectedMetrics) {
  const finalSet = metricValueSet(finalMetrics);
  const allSet = metricValueSet(allMetrics);
  const selectedSet = metricValueSet(selectedMetrics);
  const rows = Object.keys(finalSet).map(metric => {
    const finalValue = finalSet[metric];
    const allValue = allSet[metric];
    const selectedValue = selectedSet[metric];
    const selectedDelta = selectedValue - allValue;
    const finalDelta = finalValue - allValue;
    const selectedLabel = `${selectedDelta >= 0 ? "+" : ""}${selectedDelta.toFixed(4)}`;
    const finalLabel = `${finalDelta >= 0 ? "+" : ""}${finalDelta.toFixed(4)}`;
    return [
      metric,
      fmt(finalValue),
      fmt(allValue),
      fmt(selectedValue),
      `<span class="${selectedDelta >= 0 ? "good" : "warn"}">${selectedLabel}</span>`,
      `<span class="${finalDelta >= 0 ? "good" : "warn"}">${finalLabel}</span>`,
    ];
  });
  return table(["Metric", "Final", "All features", "Selected features", "Selected Δ", "Final Δ"], rows);
}


function availabilityTable(data) {
  const keys = ['params','dvc_yaml','dvc_lock','raw_dvc_pointer','train_script','hpo_script','register_script','metrics','best_model_summary','registry_summary','best_model_pkl','monitoring_summary','drift_report_html','baseline_report_html'];
  return `<div class="file-grid">${keys.map(k => `<div class="file-item"><code>${data.availability[k]?.path}</code>${badge(data.availability[k]?.exists ? 'true' : 'false', data.availability[k]?.exists ? 'good' : 'bad')}</div>`).join('')}</div>`;
}

function stageSteps(stageResults) {
  const entries = Object.entries(obj(stageResults));
  if (!entries.length) return '<div class="json">No stage execution data found.</div>';
  return `<div class="steps">${entries.map(([name, result], i) => `<div class="step"><div class="num">${i + 1}</div><div><strong>${name}</strong><span>${result.command || '-'}</span></div>${badge(result.success ? 'success' : 'failed', result.success ? 'good' : 'bad')}</div>`).join('')}</div>`;
}

function healthChips(data) {
  const items = [
    ['Best model summary', !!data.best_model_summary, data.sources.best_model_summary],
    ['Metrics', !!data.metrics, data.sources.metrics],
    ['Registry summary', !!data.registry_summary, data.sources.registry_summary],
    ['Leakage report', !!data.leakage_report, data.sources.leakage_report],
    ['Feature importance', arr(data.feature_importance).length > 0, data.sources.feature_importance],
    ['Monitoring summary', !!data.monitoring_summary?.drift, exists(data, 'monitoring_summary') ? 'live repo file' : 'missing'],
  ];
  return `<div class="kpi-cluster">${items.map(([label, ok, note]) => `<div class="kpi-chip"><span>${label}</span><strong class="${ok ? 'good' : 'warn'}">${ok ? 'Available' : 'Missing'}</strong><span>${note || '-'}</span></div>`).join('')}</div>`;
}

function renderOverview(data) {
  const root = document.getElementById('content');
  const m = finalMetrics(data);
  const bm = obj(data.best_model_summary);
  const reg = obj(data.registry_summary);
  const leak = obj(data.leakage_report);
  const drift = obj(data.monitoring_summary?.drift);
  const baseline = obj(data.monitoring_summary?.baseline);
  const modelRows = arr(data.model_comparison).map(normalizeModelRow);
  const featureRows = arr(data.feature_importance);
  const stageResults = obj(data.dvc_results?.stage_results);
  const perfItems = [
    { label: 'Accuracy', value: Number(m.accuracy || 0), display: fmt(m.accuracy) },
    { label: 'Precision', value: Number(m.precision_macro || 0), display: fmt(m.precision_macro) },
    { label: 'Recall', value: Number(m.recall_macro || 0), display: fmt(m.recall_macro) },
    { label: 'F1 Macro', value: Number(m.f1_macro || 0), display: fmt(m.f1_macro) },
    { label: 'ROC-AUC', value: Number(m.roc_auc || 0), display: fmt(m.roc_auc) },
  ];
  const gapRows = modelRows.map(r => ({ model_name: r.model_name, gap: Number(r.gap || 0) }));
  const gapMax = Math.max(...gapRows.map(r => r.gap), 0.1);
  const driftLabels = ['Baseline share', 'Current share'];
  const driftSeries = [{ name: 'Drift share', color: '#22d3ee', values: [Number(baseline.drift_share || 0), Number(drift.drift_share || 0)] }];

  root.innerHTML = baseShell('Executive Overview', 'A cleaner executive layout with the full KPI set organized across model performance, drift, leakage, DVC, MLflow, and repository readiness — all following the exact local repository structure.') +
  `<section class="grid kpis">
    ${kpi('Final Model', bm.final_model_name || 'Not found', `Source: ${data.sources.best_model_summary}`, bm.final_model_name ? 'good' : 'warn')}
    ${kpi('Final Choice', bm.final_model_choice || '-', bm.final_reason || 'No final choice found', 'info')}
    ${kpi('F1 Macro', fmt(m.f1_macro), 'Primary model metric', m.f1_macro ? 'good' : 'warn')}
    ${kpi('Accuracy', fmt(m.accuracy), 'Final/test metric', m.accuracy ? 'good' : 'warn')}
    ${kpi('Precision', fmt(m.precision_macro), 'Macro precision', m.precision_macro ? 'good' : 'warn')}
    ${kpi('Recall', fmt(m.recall_macro), 'Macro recall', m.recall_macro ? 'good' : 'warn')}
    ${kpi('ROC-AUC', fmt(m.roc_auc), 'Probability separation', m.roc_auc ? 'good' : 'warn')}
    ${kpi('Leakage Check', leak.status || 'Not found', 'Automated leakage diagnostics', leak.status === 'PASS' ? 'good' : 'warn')}
    ${kpi('Drift Share', fmt(drift.drift_share), `${drift.drifted_feature_count ?? '-'} / ${drift.total_feature_count ?? '-'} features drifted`, drift.drift_detected ? 'warn' : 'good')}
    ${kpi('MLflow Version', reg.registered_model_version || '-', reg.registered_model_name || 'Registry summary', reg.registered_model_version ? 'good' : 'warn')}
    ${kpi('DVC Repro', data.dvc_results?.dvc_repro_success ? 'Success' : 'Unknown', 'train / hpo / register pipeline', data.dvc_results?.dvc_repro_success ? 'good' : 'warn')}
    ${kpi('Batch Ready', data.batch_ready ? 'Ready' : 'Blocked', data.batch_ready ? 'model + test data found' : 'needs model + test data', data.batch_ready ? 'good' : 'warn')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Performance Scorecard</h3><p class="sub">Final evaluation metrics used in the project decision.</p>${progressList(perfItems)}</div>
    <div class="card"><h3>Repository and Output Health</h3><p class="sub">Whether the dashboard found the required outputs directly or from the notebook fallback.</p>${healthChips(data)}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Model Benchmark — F1 Macro</h3>${bars(modelRows, 'model_name', 'f1_macro', 1)}</div>
    <div class="card"><h3>Train–Test Gap by Model</h3><p class="sub">Lower is better. This helps spot possible overfitting.</p>${bars(gapRows, 'model_name', 'gap', gapMax, true)}</div>
  </section>
  <section class="grid three">
    <div class="card"><h3>Feature Importance Snapshot</h3><p class="sub">Top-ranked features from permutation importance.</p>${bars(featureRows.slice(0, 5).map(r => ({ feature: r.feature, importance_mean: Number(r.importance_mean || 0) })), 'feature', 'importance_mean', Math.max(...featureRows.map(r => Number(r.importance_mean || 0)), 0.01))}</div>
    <div class="card"><h3>DVC Execution Snapshot</h3><p class="sub">Execution status captured from the recorded DVC results.</p>${stageSteps(stageResults)}</div>
    <div class="card"><h3>Drift Monitoring Snapshot</h3>${lineChart('Drift share trend', 'Baseline vs current monitoring summary', driftLabels, driftSeries, { min: 0, max: Math.max(1, Number(drift.drift_share || 0), Number(baseline.drift_share || 0)) })}<div style="margin-top:12px">${badge(`Drift detected: ${String(!!drift.drift_detected)}`, drift.drift_detected ? 'warn' : 'good')}${badge(`Alerts: ${arr(data.drift_alerts).length}`, arr(data.drift_alerts).length ? 'warn' : 'good')}</div></div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Final Model Decision</h3><p class="sub">This combines the best-model summary with the final reasoning recorded during training.</p><div class="callout"><strong>${bm.final_model_name || 'No final model detected'}</strong><br>${bm.final_reason || 'No final reasoning found.'}</div><div style="margin-top:12px">${sourcePill(data.sources.best_model_summary)}${badge(`Run ID: ${bm.final_run_id || reg.registered_run_id || '-'}`, 'info')}</div></div>
    <div class="card"><h3>System Availability</h3><p class="sub">Exact files found in the local repository.</p>${availabilityTable(data)}</div>
  </section>`;
}

function renderMonitoring(data) {
  const root = document.getElementById('content');
  const drift = obj(data.monitoring_summary?.drift);
  root.innerHTML = baseShell('Batch Monitoring', 'Monitor local scoring batches, live batch KPIs, confidence, latency, and monitoring updates. This page keeps the runner separated from drift and model analysis for clearer organization.') +
  `<section class="grid kpis" id="monitoring-kpis"></section>
   <section class="grid two-even">
     <div class="card"><h3>Batch Runner</h3><p class="sub">Required files: <span class="mono">models/best_model.pkl</span> and <span class="mono">data/splits/test.csv</span>. The dashboard will not invent any missing production dataset.</p><div class="controls"><input class="control" id="batchSize" type="number" value="50" min="1"><button class="control button" id="runBatch">Run Next Batch</button><button class="control button" id="runMonitoring">Run Monitoring Script</button></div><div id="batchMsg" class="json" style="margin-top:14px">Waiting...</div></div>
     <div class="card"><h3>Monitoring Summary Snapshot</h3><p class="sub">Directly from monitoring/evidently_reports/monitoring_summary.json.</p>${metricBoxes({ accuracy: drift.drift_share, precision_macro: drift.threshold, recall_macro: drift.drifted_feature_count, f1_macro: drift.total_feature_count, roc_auc: arr(data.drift_alerts).length, problem_type: drift.drift_detected ? 'drift detected' : 'stable' })}</div>
   </section>
   <section class="grid two-even">
     <div id="batchTrendCard" class="card"><h3>Batch Trend</h3><div class="json">Loading batch history...</div></div>
     <div id="batchMixCard" class="card"><h3>Latest Batch Mix</h3><div class="json">Loading batch history...</div></div>
   </section>
   <section class="grid two-even">
     <div class="card"><h3>Current Requirements</h3>${availabilityTable(data)}</div>
     <div id="batchHistoryTable" class="card"><h3>Batch History Table</h3><div class="json">Loading batch history...</div></div>
   </section>`;
  setTimeout(() => bindMonitoring(data), 50);
}

async function bindMonitoring(data) {
  const msg = document.getElementById('batchMsg');
  const trendCard = document.getElementById('batchTrendCard');
  const mixCard = document.getElementById('batchMixCard');
  const histTable = document.getElementById('batchHistoryTable');
  const kpiWrap = document.getElementById('monitoring-kpis');

  async function refresh() {
    const history = await loadHistory();
    renderMonitoringHistory(history, trendCard, mixCard, histTable, kpiWrap, data);
  }

  await refresh();

  document.getElementById('runBatch').onclick = async () => {
    msg.textContent = 'Running batch...';
    const res = await fetch('/api/batch/next', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batch_size: Number(document.getElementById('batchSize').value || 50) }),
    });
    const payload = await res.json();
    msg.textContent = safeJson(payload);
    await refresh();
  };

  document.getElementById('runMonitoring').onclick = async () => {
    msg.textContent = 'Running monitoring script...';
    const res = await fetch('/api/monitoring/run', { method: 'POST' });
    const payload = await res.json();
    msg.textContent = safeJson(payload);
    await refresh();
  };
}

function renderMonitoringHistory(history, trendCard, mixCard, histTable, kpiWrap, data) {
  const safeHistory = arr(history);
  const last = latest(safeHistory);
  const avgLatency = avg(safeHistory.map(x => x.latency_ms));
  const avgConfidence = avg(safeHistory.map(x => x.avg_confidence));
  const totalRows = sum(safeHistory.map(x => x.rows));
  const latestPosRate = last ? (Number(last.positive_predictions || 0) / Math.max(1, Number(last.rows || 1))) : null;

  kpiWrap.innerHTML = [
    kpi('Batches Run', fmtInt(safeHistory.length), 'Recorded local scoring batches', safeHistory.length ? 'good' : 'warn'),
    kpi('Rows Scored', fmtInt(totalRows), 'Total rows across batches', totalRows ? 'good' : 'warn'),
    kpi('Latest Batch Size', fmtInt(last?.rows), 'Rows in most recent batch', last ? 'good' : 'warn'),
    kpi('Avg Latency', last ? `${fmt(avgLatency, 2)} ms` : '-', 'Average batch latency', last ? 'good' : 'warn'),
    kpi('Avg Confidence', last ? pct(avgConfidence, 1) : '-', 'Mean prediction confidence', last ? 'good' : 'warn'),
    kpi('Latest Positive Rate', last ? pct(latestPosRate, 1) : '-', 'Positive predictions / batch rows', last ? 'info' : 'warn'),
    kpi('Current Drift Share', fmt(data.monitoring_summary?.drift?.drift_share), 'From monitoring summary', data.monitoring_summary?.drift?.drift_detected ? 'warn' : 'good'),
    kpi('Alert Count', fmtInt(arr(data.drift_alerts).length), 'drift_alerts.log entries', arr(data.drift_alerts).length ? 'warn' : 'good'),
    kpi('Runner Ready', data.batch_ready ? 'Yes' : 'No', data.batch_ready ? 'Model and test set found' : 'Missing model or test set', data.batch_ready ? 'good' : 'warn'),
    kpi('Latest Offset', fmtInt(last?.offset), 'Resume position in test set', last ? 'info' : 'warn'),
    kpi('Latest Timestamp', last ? new Date(last.timestamp * 1000).toLocaleString() : '-', 'Most recent batch event', last ? 'good' : 'warn'),
    kpi('True Labels', last?.y_true_available ? 'Available' : 'Unknown', 'Whether the batch contained labels', last?.y_true_available ? 'good' : 'warn'),
  ].join('');

  const labels = safeHistory.map((_, i) => `B${i + 1}`);
  trendCard.innerHTML = `<h3>Batch Trend</h3><p class="sub">Confidence and latency across the generated local scoring batches.</p>${lineChart('Batch metrics', 'Confidence uses a 0–1 scale; latency is raw milliseconds.', labels, [
    { name: 'Avg confidence', color: '#22d3ee', values: safeHistory.map(x => Number(x.avg_confidence || 0)) },
    { name: 'Latency (ms)', color: '#a78bfa', values: safeHistory.map(x => Number(x.latency_ms || 0)) },
  ])}`;

  if (last) {
    const pos = Number(last.positive_predictions || 0);
    const neg = Number(last.negative_predictions || 0);
    const total = Math.max(1, pos + neg);
    mixCard.innerHTML = `<h3>Latest Batch Mix</h3><p class="sub">Distribution of predictions and batch-level operational values for the most recent run.</p><div class="grid two-even"><div>${progressList([
      { label: 'Positive predictions', value: pos / total, display: `${pos} (${pct(pos / total, 1)})` },
      { label: 'Negative predictions', value: neg / total, display: `${neg} (${pct(neg / total, 1)})` },
      { label: 'Confidence', value: Number(last.avg_confidence || 0), display: pct(last.avg_confidence, 1) },
    ])}</div><div class="metric-grid"><div class="metric-box"><div class="label">Rows</div><div class="value">${fmtInt(last.rows)}</div></div><div class="metric-box"><div class="label">Latency (ms)</div><div class="value">${fmt(last.latency_ms, 2)}</div></div><div class="metric-box"><div class="label">Offset</div><div class="value">${fmtInt(last.offset)}</div></div><div class="metric-box"><div class="label">Timestamp</div><div class="value compact">${new Date(last.timestamp * 1000).toLocaleTimeString()}</div></div></div></div>`;
  } else {
    mixCard.innerHTML = `<h3>Latest Batch Mix</h3><div class="json">No batch history yet. Run a batch to populate this section.</div>`;
  }

  histTable.innerHTML = `<h3>Batch History Table</h3><p class="sub">Raw batch events recorded under dashboard/runtime/batch_history.json.</p>${table(['Batch','Rows','Latency (ms)','Confidence','Positive','Negative','Timestamp'], safeHistory.map((x, i) => [String(i + 1), fmtInt(x.rows), fmt(x.latency_ms, 2), pct(x.avg_confidence, 1), fmtInt(x.positive_predictions), fmtInt(x.negative_predictions), new Date(x.timestamp * 1000).toLocaleString()]))}`;
}

function renderDrift(data) {
  const root = document.getElementById('content');
  const d = obj(data.monitoring_summary?.drift);
  const b = obj(data.monitoring_summary?.baseline);
  const alerts = arr(data.drift_alerts);
  const driftFeatures = arr(d.drifted_features);
  root.innerHTML = baseShell('Drift Alerts', 'A dedicated drift page so drift monitoring is not mixed with training KPIs. It reads the Evidently outputs directly from the monitoring folder.') +
  `<section class="grid kpis">
    ${kpi('Current drift', String(!!d.drift_detected), 'Current monitoring section', d.drift_detected ? 'warn' : 'good')}
    ${kpi('Current share', fmt(d.drift_share), `${d.drifted_feature_count || 0}/${d.total_feature_count || 0} features`, d.drift_detected ? 'warn' : 'good')}
    ${kpi('Baseline drift', String(!!b.drift_detected), 'Baseline monitoring section', b.drift_detected ? 'warn' : 'good')}
    ${kpi('Baseline share', fmt(b.drift_share), `${b.drifted_feature_count || 0}/${b.total_feature_count || 0} features`, b.drift_detected ? 'warn' : 'good')}
    ${kpi('Alert entries', fmtInt(alerts.length), 'From drift_alerts.log', alerts.length ? 'warn' : 'good')}
    ${kpi('Threshold', fmt(d.threshold || b.threshold), 'Configured alert line', 'info')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Drifted Features</h3>${bars(driftFeatures.map(f => ({ feature: f.feature, score: Number(f.score || 0) })), 'feature', 'score', Math.max(...driftFeatures.map(f => Number(f.score || 0)), 0.25), true)}</div>
    <div class="card"><h3>Drift Share Trend</h3>${lineChart('Baseline vs current drift share', 'Two-point view from monitoring_summary.json', ['Baseline', 'Current'], [{ name: 'Drift share', color: '#fbbf24', values: [Number(b.drift_share || 0), Number(d.drift_share || 0)] }], { min: 0, max: Math.max(1, Number(b.drift_share || 0), Number(d.drift_share || 0)) })}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Current Drifted Features Table</h3>${table(['Feature', 'Score', 'Test'], driftFeatures.map(f => [f.feature, fmt(f.score), f.test || '-']))}</div>
    <div class="card"><h3>Drift Alert Log</h3><div class="json">${safeJson(alerts)}</div></div>
  </section>
  <section class="grid two-even"><div class="card"><h3>Evidently Reports</h3><p><a class="control" href="/repo-file/monitoring/evidently_reports/drift_report.html" target="_blank">Open Drift Report</a> <a class="control" href="/repo-file/monitoring/evidently_reports/baseline_report.html" target="_blank">Open Baseline Report</a></p></div><div class="card"><h3>Raw Monitoring Summary</h3><div class="json">${safeJson(data.monitoring_summary)}</div></div></section>`;
}

function renderModels(data) {
  const rows = arr(data.model_comparison).map(normalizeModelRow);
  const bm = obj(data.best_model_summary);
  const hpo = obj(data.hpo_summary);
  const finalM = obj(bm.final_metrics || {});
  const allM = obj(bm.all_features_metrics || {});
  const selM = obj(bm.selected_features_metrics || {});
  const gapMax = Math.max(...rows.map(r => Number(r.gap || 0)), 0.1);
  document.getElementById('content').innerHTML = baseShell('Model Comparison', 'Training outputs and HPO are now organized on their own page with separate benchmark, gap, and decision sections.') +
  `<section class="grid kpis">
    ${kpi('Final model', bm.final_model_name || '-', data.sources.best_model_summary, bm.final_model_name ? 'good' : 'warn')}
    ${kpi('Final choice', bm.final_model_choice || '-', bm.final_reason || 'Decision note', 'info')}
    ${kpi('Final F1', fmt(bm.final_score || finalM.f1_macro), 'Chosen final score', 'good')}
    ${kpi('All-features F1', fmt(bm.all_features_score), 'Full-feature benchmark', 'good')}
    ${kpi('Selected-features F1', fmt(bm.selected_features_score), 'Feature-subset benchmark', 'good')}
    ${kpi('HPO best', hpo.best_hpo_model_name || hpo.bestSummary?.model_name || '-', data.sources.hpo_summary, hpo.best_hpo_model_name || hpo.bestSummary?.model_name ? 'good' : 'warn')}
    ${kpi('HPO score', fmt(hpo.best_score), 'Best tuning score', hpo.best_score ? 'good' : 'warn')}
    ${kpi('Registered run', data.registry_summary?.registered_run_id || '-', 'MLflow registered run ID', data.registry_summary?.registered_run_id ? 'good' : 'warn')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>F1 Macro by Model</h3>${bars(rows, 'model_name', 'f1_macro', 1)}</div>
    <div class="card"><h3>Train–Test Gap by Model</h3><p class="sub">A compact overfitting view from the diagnostics output.</p>${bars(rows.map(r => ({ model_name: r.model_name, gap: r.gap })), 'model_name', 'gap', gapMax, true)}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Core Metrics Comparison</h3><p class="sub">This replaces the previous metric cards with a grouped bar chart across the final model, all-feature model, and selected-feature model.</p>${modelMetricComparisonChart(finalM, allM, selM)}</div>
    <div class="card"><h3>Metric Profile Line Chart</h3><p class="sub">A line-chart view makes it easier to compare the stability of each model configuration across all key metrics.</p>${modelMetricLineChart(finalM, allM, selM)}</div>
  </section>
  <section class="card"><h3>Metric Difference Table</h3><p class="sub">Selected-feature and final-model differences are compared against the all-feature model. This supports the final decision to keep feature importance as analysis only.</p>${modelMetricDeltaTable(finalM, allM, selM)}</section>
  <section class="grid two-even">
    <div class="card"><h3>Model Diagnostics Table</h3>${table(['Model', 'F1 macro', 'Gap', 'Verdict'], rows.map(r => [r.model_name, fmt(r.f1_macro), fmt(r.gap), badge(r.verdict || '-', r.verdict === 'GOOD' ? 'good' : 'warn')]))}</div>
    <div class="card"><h3>Best Model Summary</h3><div class="json">${safeJson(bm)}</div></div>
  </section>`;
}

function renderFeatures(data) {
  const rows = arr(data.feature_importance);
  const sf = obj(data.selected_features);
  const bm = obj(data.best_model_summary);
  const maxImportance = Math.max(...rows.map(r => Number(r.importance_mean || r.importance || 0)), 0.01);
  const cumulative = rows.map(r => Number(r.cumulative_importance || 0));
  const importanceLine = lineChart('Cumulative importance', 'How quickly the top-ranked features explain total importance.', rows.map(r => r.feature), [{ name: 'Cumulative', color: '#22d3ee', values: cumulative }], { min: 0, max: 1 });
  document.getElementById('content').innerHTML = baseShell('Feature Importance', 'Feature importance analysis is now isolated on its own page, with both ranking and selection rationale organized separately.') +
  `<section class="grid kpis">
    ${kpi('Reference model', sf.reference_best_model_name || '-', 'Model used for importance analysis', sf.reference_best_model_name ? 'good' : 'warn')}
    ${kpi('Selection method', sf.method || '-', 'Feature selection approach', 'info')}
    ${kpi('Selected candidate', sf.selected_candidate_name || '-', 'Chosen cutoff candidate', sf.selected_candidate_name ? 'good' : 'warn')}
    ${kpi('Selected count', fmtInt(sf.selected_feature_count), 'Number of kept features', sf.selected_feature_count ? 'good' : 'warn')}
    ${kpi('Validation score', fmt(sf.feature_selection_validation_score), 'Selection validation score', sf.feature_selection_validation_score ? 'good' : 'warn')}
    ${kpi('Final decision', bm.feature_importance_kept_as_analysis ? 'Analysis only' : 'Used in final model', 'How feature importance affected the final model choice', bm.feature_importance_kept_as_analysis ? 'warn' : 'good')}
    ${kpi('Top feature', rows[0]?.feature || '-', 'Highest-ranked feature', rows.length ? 'good' : 'warn')}
    ${kpi('Top-4 cumulative', fmt(rows[3]?.cumulative_importance), 'Cumulative importance after four features', rows[3]?.cumulative_importance ? 'good' : 'warn')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Top Feature Importance</h3>${bars(rows.map(r => ({ feature: r.feature, importance_mean: Number(r.importance_mean || r.importance || 0) })), 'feature', 'importance_mean', maxImportance)}</div>
    <div class="card"><h3>Cumulative Importance Trend</h3>${importanceLine}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Selected Features</h3><p class="sub">Final selected subset from the recorded feature-selection analysis.</p>${chips(sf.selected_features || [], 'info')}<div style="margin-top:12px"><h4>Top-10 reference features</h4>${chips(sf.top_10_reference_features || [], 'good')}</div></div>
    <div class="card"><h3>Selection Summary</h3><div class="json">${safeJson(sf)}</div></div>
  </section>
  <section class="card"><h3>Feature Importance Table</h3>${table(['Feature', 'Importance', 'Ratio', 'Cumulative'], rows.map(r => [r.feature, fmt(r.importance_mean), fmt(r.importance_ratio), fmt(r.cumulative_importance)]))}</section>`;
}

function renderDvc(data) {
  const lock = obj(data.dvc_lock);
  const yaml = obj(data.dvc_yaml);
  const duplicate = arr(lock.duplicate_stage_names);
  const stages = (arr(yaml.stages).length ? arr(yaml.stages) : arr(lock.stages));
  const dvc = obj(data.dvc_results);
  const stageResults = obj(dvc.stage_results);
  document.getElementById('content').innerHTML = baseShell('DVC Pipeline', 'The DVC page now separates structure, execution, and lock-file health so it is easier to diagnose issues like duplicate stages or invalid lock content.') +
  `<section class="grid kpis">
    ${kpi('dvc.yaml', yaml.exists ? 'Found' : 'Missing', yaml.valid_yaml ? 'Valid YAML' : (yaml.error || '-'), yaml.exists ? 'good' : 'bad')}
    ${kpi('dvc.lock', lock.exists ? 'Found' : 'Missing', lock.valid_yaml ? 'Valid YAML' : (lock.parse_error || 'Parsed as text'), lock.exists ? 'good' : 'bad')}
    ${kpi('Duplicate stages', fmtInt(duplicate.length), duplicate.join(', ') || 'none', duplicate.length ? 'warn' : 'good')}
    ${kpi('Stage count', fmtInt(stages.length), 'Stages parsed from dvc.yaml / dvc.lock', 'info')}
    ${kpi('DVC repro', dvc.dvc_repro_success ? 'Success' : 'Unknown', 'Recorded DVC pipeline reproduction result', dvc.dvc_repro_success ? 'good' : 'warn')}
    ${kpi('DVC metrics', dvc.dvc_metrics_success ? 'Success' : 'Unknown', 'Recorded dvc metrics show result', dvc.dvc_metrics_success ? 'good' : 'warn')}
    ${kpi('Lock created', dvc.dvc_lock_created ? 'Yes' : 'No', 'Whether dvc.lock was generated', dvc.dvc_lock_created ? 'good' : 'warn')}
    ${kpi('Registry summary', dvc.model_registry_summary_created ? 'Created' : 'Unknown', 'Whether register stage produced summary', dvc.model_registry_summary_created ? 'good' : 'warn')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Recorded Stage Execution</h3>${stageSteps(stageResults)}</div>
    <div class="card"><h3>Configured Pipeline Stages</h3>${table(['Stage', 'Command', 'Outputs'], stages.map(s => [s.name || '-', `<code>${s.cmd || '-'}</code>`, arr(s.outs).slice(0, 6).join('<br>') || '-']))}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>DVC Lock Parse</h3><div class="json">${safeJson(lock)}</div></div>
    <div class="card"><h3>DVC Results Summary</h3><div class="json">${safeJson(dvc)}</div></div>
  </section>`;
}

function renderRegistry(data) {
  const r = obj(data.registry_summary);
  const bm = obj(data.best_model_summary);
  document.getElementById('content').innerHTML = baseShell('MLflow Registry', 'Registry and lineage details are now isolated so versioning is easier to read and present.') +
  `<section class="grid kpis">
    ${kpi('Registered model', r.registered_model_name || '-', data.sources.registry_summary, r.registered_model_name ? 'good' : 'warn')}
    ${kpi('Version', r.registered_model_version || '-', 'MLflow model version', r.registered_model_version ? 'good' : 'warn')}
    ${kpi('Metric', r.metric_name || '-', fmt(r.metric_value), r.metric_value ? 'good' : 'warn')}
    ${kpi('Run ID', r.registered_run_id || '-', 'Registered run identifier', r.registered_run_id ? 'good' : 'warn')}
    ${kpi('Model from run', r.model_name_from_run || '-', 'Model artifact name', r.model_name_from_run ? 'good' : 'warn')}
    ${kpi('Stage from run', r.model_stage_from_run || '-', 'Logical stage / alias', r.model_stage_from_run ? 'good' : 'warn')}
    ${kpi('Final model choice', r.final_model_choice || bm.final_model_choice || '-', 'Final project decision', 'info')}
    ${kpi('Feature analysis', r.feature_importance_kept_as_analysis ? 'Analysis only' : 'Used in final model', 'Feature importance contribution', r.feature_importance_kept_as_analysis ? 'warn' : 'good')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Registry Lineage</h3><div class="kpi-cluster">
      <div class="kpi-chip"><span>Registered model</span><strong>${r.registered_model_name || '-'}</strong><span>Registry object name</span></div>
      <div class="kpi-chip"><span>Version</span><strong>${r.registered_model_version || '-'}</strong><span>Recorded version number</span></div>
      <div class="kpi-chip"><span>Run ID</span><strong>${shortText(r.registered_run_id || '-', 18)}</strong><span>MLflow run used for registration</span></div>
      <div class="kpi-chip"><span>Metric value</span><strong>${fmt(r.metric_value)}</strong><span>${r.metric_name || 'metric'}</span></div>
    </div></div>
    <div class="card"><h3>Selected Features Recorded in Registry</h3>${chips(r.selected_features_analysis || bm.selected_features_analysis || [], 'info')}<div style="margin-top:12px"><h4>Top reference features</h4>${chips(r.top_10_reference_features || bm.top_10_reference_features || [], 'good')}</div></div>
  </section>
  <section class="card"><h3>Registry Summary JSON</h3><div class="json">${safeJson(r)}</div></section>`;
}

function renderReports(data) {
  const rows = Object.entries(obj(data.availability)).map(([k, v]) => [k, `<code>${v.path}</code>`, badge(v.exists ? 'exists' : 'missing', v.exists ? 'good' : 'bad'), v.size ?? '-', v.modified ? new Date(v.modified * 1000).toLocaleString() : '-']);
  document.getElementById('content').innerHTML = baseShell('Reports and Repository Map', 'This final page is the proof page: it shows exactly which files the dashboard found, where the values came from, and the full payload used by the UI.') +
  `<section class="grid two-even">
    <div class="card"><h3>Exact Repository Files</h3>${table(['Key', 'Path', 'Status', 'Size', 'Modified'], rows)}</div>
    <div class="card"><h3>Data Sources Used</h3><div class="json">${safeJson(data.sources)}</div><h3 style="margin-top:16px">Notebook Source</h3><div class="json">${safeJson(data.notebook_source)}</div></div>
  </section>
  <section class="card"><h3>Full API Payload</h3><div class="json">${safeJson(data)}</div></section>`;
}

async function init() {
  const data = await loadSummary();
  setRoot(data);
  const fn = PAGES[page()] || renderOverview;
  fn(data);
  setStatus(data);
}

init().catch(e => {
  document.getElementById('content').innerHTML = `<div class="card"><h3>Dashboard error</h3><div class="json">${e.stack || e}</div></div>`;
});
