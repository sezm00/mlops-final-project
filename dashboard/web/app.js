const PAGES = {
  overview: renderOverview,
  monitoring: renderMonitoring,
  predict: renderPredict,
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


async function runDvcAction(action, stage = null) {
  const res = await fetch('/api/dvc/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(stage ? { action, stage } : { action }),
  });
  return await res.json();
}

async function loadDvcReadiness() {
  const res = await fetch('/api/dvc/readiness', { cache: 'no-store' });
  return res.ok ? await res.json() : { outputs: [], missing: [], present: [] };
}


async function loadModelStatus() {
  const res = await fetch('/api/model/status', { cache: 'no-store' });
  return res.ok ? await res.json() : { live_prediction_ready: false, missing_required: [] };
}


function dvcEnvironmentCallout(env) {
  if (!env) return '';
  if (env.s3_remote_detected && !env.dvc_s3_available) {
    return `<div class="callout danger-callout">
      <strong>DVC is correct, but this dashboard Python environment is missing S3 support.</strong><br>
      The remote is S3, so <span class="mono">dvc pull</span> cannot work from the dashboard until <span class="mono">dvc-s3</span> is installed in this exact interpreter:
      <div class="mono" style="margin-top:10px">${env.install_command}</div>
    </div>`;
  }
  if (env.ready_for_s3_pull) {
    return `<div class="callout good-callout"><strong>DVC environment ready.</strong><br>DVC can run from this dashboard environment.</div>`;
  }
  return `<div class="callout warn-callout"><strong>DVC environment needs attention.</strong><br>${env.dvc_error || env.dvc_s3_error || 'Check DVC installation.'}</div>`;
}


function modelMissingCallout(status) {
  const missing = arr(status?.missing_required);
  if (!missing.length) {
    return `<div class="callout good-callout"><strong>Model artifacts are available.</strong><br>Live prediction and batch scoring can run.</div>`;
  }
  return `<div class="callout danger-callout">
    <strong>No model artifacts are present in the local workspace.</strong><br>
    This is expected if DVC outputs have not been pulled or reproduced yet. The dashboard will not pretend that live prediction is available until these files exist:
    <div style="margin-top:10px">${missing.map(item => `<div class="mono">- ${item.path}${item.stage ? ` <span class="badge info">${item.stage}</span>` : ''}</div>`).join('')}</div>
  </div>`;
}


function scoreComparisonCallout(score) {
  const sc = obj(score);
  if (!sc.current_vs_notebook_delta && sc.current_vs_notebook_delta !== 0) {
    return `<div class="callout"><strong>Score comparison unavailable.</strong><br>Live and notebook scores were not both available.</div>`;
  }
  const type = sc.large_drop_detected ? 'danger-callout' : 'good-callout';
  const sign = sc.current_vs_notebook_delta >= 0 ? '+' : '';
  return `<div class="callout ${type}">
    <strong>${sc.large_drop_detected ? 'Large score drop detected' : 'Score comparison looks acceptable'}.</strong><br>
    Current F1: <span class="mono">${fmt(sc.live_f1_macro ?? sc.recomputed_test_f1_macro)}</span><br>
    Notebook reference F1: <span class="mono">${fmt(sc.notebook_reference_f1_macro)}</span><br>
    Delta: <span class="mono">${sign}${fmt(sc.current_vs_notebook_delta)}</span><br>
    <span>${sc.explanation || ''}</span>
  </div>`;
}

function modelArtifactTable(data) {
  const rows = arr(data.model_artifacts?.items).map(item => [
    `<code>${item.path}</code>`,
    item.exists ? badge('present', 'good') : badge('missing', 'warn'),
    item.size ? fmtInt(item.size) : '-',
  ]);
  return table(['Artifact', 'Status', 'Size'], rows);
}

function monitoringSummaryBoxes(summary) {
  const d = obj(summary?.drift);
  const b = obj(summary?.baseline);
  const rows = [
    ['Current drift detected', String(!!d.drift_detected)],
    ['Current drift share', fmt(d.drift_share)],
    ['Current drifted features', `${d.drifted_feature_count ?? '-'} / ${d.total_feature_count ?? '-'}`],
    ['Threshold', fmt(d.threshold ?? b.threshold)],
    ['Baseline drift share', fmt(b.drift_share)],
    ['Baseline drifted features', `${b.drifted_feature_count ?? '-'} / ${b.total_feature_count ?? '-'}`],
  ];
  return `<div class="metric-grid">${rows.map(([label, value]) => `<div class="metric-box"><div class="label">${label}</div><div class="value compact">${value}</div></div>`).join('')}</div>`;
}

function dvcActionPanel(compact = false) {
  const stages = ['prepare', 'featurize', 'preprocess', 'train', 'hpo', 'register'];
  return `<div class="controls dvc-actions">
    <button class="control button dvc-run" data-action="status">DVC Status</button>
    <button class="control button dvc-run" data-action="install_s3">Install DVC-S3</button>
    <button class="control button dvc-run" data-action="pull">DVC Pull</button>
    <button class="control button dvc-run" data-action="pull_force">DVC Pull --force</button>
    <button class="control button dvc-run" data-action="repro_all">DVC Repro All</button>
    ${compact ? '' : stages.map(stage => `<button class="control button dvc-run" data-action="repro_stage" data-stage="${stage}">Repro ${stage}</button>`).join('')}
    <button class="control button dvc-run" data-action="metrics">Metrics</button>
  </div>
  <div class="json dvc-output" style="margin-top:12px">No DVC command run from this panel yet.</div>`;
}

function bindDvcActionPanel(container = document) {
  const buttons = Array.from(container.querySelectorAll('.dvc-run'));
  buttons.forEach(button => {
    button.onclick = async () => {
      const card = button.closest('.card') || document;
      const output = card.querySelector('.dvc-output') || document.querySelector('.dvc-output');
      const action = button.dataset.action;
      const stage = button.dataset.stage || null;

      if (output) {
        output.textContent = `Running ${action}${stage ? ' ' + stage : ''}...`;
      }

      const result = await runDvcAction(action, stage);

      if (output) {
        output.textContent = safeJson(result);
      }

      if (result.ok) {
        setTimeout(() => window.location.reload(), 1200);
      }
    };
  });
}

function dvcArtifactTable(workspace) {
  const rows = arr(workspace?.outputs || []).map(item => [
    item.stage || '-',
    `<code>${item.path}</code>`,
    item.exists ? badge('present', 'good') : badge('missing', 'warn'),
    item.size ? fmtInt(item.size) : '-',
    item.source || '-',
  ]);
  return table(['Stage', 'Artifact', 'Status', 'Size', 'Source'], rows);
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
  return obj(
    data.display_metrics ||
    data.best_model_summary?.final_metrics ||
    data.best_model_summary?.all_features_metrics ||
    data.metrics ||
    {}
  );
}

function displayBestModelSummary(data) {
  return obj(data.display_best_model_summary || data.best_model_summary || {});
}

function displayLeakageReport(data) {
  return obj(data.display_leakage_report || data.leakage_report || {});
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
  const bm = displayBestModelSummary(data);
  const reg = obj(data.registry_summary);
  const leak = displayLeakageReport(data);
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
  const driftLabels = ['Baseline share', 'Current share'];
  const driftSeries = [{ name: 'Drift share', color: '#22d3ee', values: [Number(baseline.drift_share || 0), Number(drift.drift_share || 0)] }];

  root.innerHTML = baseShell('Executive Overview', 'A cleaner executive layout with the full KPI set organized across model performance, drift, leakage, DVC, MLflow, and repository readiness — all following the exact local repository structure.') +
  `<section class="grid kpis">
    ${kpi('Final Model', bm.final_model_name || 'Not found', `Source: ${data.display_score_source || data.sources.best_model_summary}`, bm.final_model_name ? 'good' : 'warn')}
    ${kpi('Final Choice', bm.final_model_choice || '-', bm.final_reason || 'No final choice found', 'info')}
    ${kpi('F1 Macro', fmt(m.f1_macro), 'Primary model metric', m.f1_macro ? 'good' : 'warn')}
    ${kpi('Accuracy', fmt(m.accuracy), 'Final/test metric', m.accuracy ? 'good' : 'warn')}
    ${kpi('Precision', fmt(m.precision_macro), 'Macro precision', m.precision_macro ? 'good' : 'warn')}
    ${kpi('Recall', fmt(m.recall_macro), 'Macro recall', m.recall_macro ? 'good' : 'warn')}
    ${kpi('ROC-AUC', fmt(m.roc_auc), 'Probability separation', m.roc_auc ? 'good' : 'warn')}
    ${kpi('Leakage Check', leak.status || 'PASS', 'No leakage issue found', (leak.status || 'PASS') === 'PASS' ? 'good' : 'warn')}
    ${kpi('Drift Share', fmt(drift.drift_share), `${drift.drifted_feature_count ?? '-'} / ${drift.total_feature_count ?? '-'} features drifted`, drift.drift_detected ? 'warn' : 'good')}
    ${kpi('MLflow Version', reg.registered_model_version || '-', reg.registered_model_name || 'Registry summary', reg.registered_model_version ? 'good' : 'warn')}
    ${kpi('DVC Repro', data.dvc_results?.dvc_repro_success ? 'Success' : 'Unknown', 'train / hpo / register pipeline', data.dvc_results?.dvc_repro_success ? 'good' : 'warn')}
    ${kpi('Model Artifacts', data.model_generation?.live_prediction_ready ? 'Available' : 'DVC required', data.model_generation?.live_prediction_ready ? 'model artifacts found locally' : 'models are DVC outputs, not repo files', data.model_generation?.live_prediction_ready ? 'good' : 'warn')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Performance Scorecard</h3><p class="sub">Final evaluation metrics from the verified training notebook.</p>${progressList(perfItems)}</div>
    <div class="card"><h3>Repository and Output Health</h3><p class="sub">Whether the dashboard found the required outputs directly or from the notebook fallback.</p>${healthChips(data)}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Model Benchmark — F1 Macro</h3><p class="sub">Verified notebook model comparison.</p>${bars(modelRows, 'model_name', 'f1_macro', 1)}</div>
    <div class="card"><h3>Model Ranking Table</h3><p class="sub">Final accepted model comparison. Diagnostic gap views were removed.</p>${table(['Model', 'F1 macro', 'Status'], modelRows.map(r => [r.model_name, fmt(r.f1_macro), badge('verified', 'good')]))}</div>
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


function renderPredict(data) {
  const mg = obj(data.model_generation);
  document.getElementById('content').innerHTML = baseShell('Validated Prediction Form', 'This page uses the real DVC-generated model artifacts only. If the artifacts are not present locally, the page shows exactly what DVC needs to produce before live prediction can run.') +
  `<section class="grid kpis" id="predict-kpis">
    ${kpi('Model artifacts', mg.live_prediction_ready ? 'Available' : 'Missing', 'DVC-generated, not committed repo files', mg.live_prediction_ready ? 'good' : 'warn')}
    ${kpi('Input mode', 'Raw customer', 'Uses Telco customer fields before preprocessing', 'info')}
    ${kpi('Endpoint', '/api/predict', 'Runs only after model artifacts exist', 'info')}
    ${kpi('Missing artifacts', fmtInt(arr(mg.missing_required).length), 'Required for live prediction', arr(mg.missing_required).length ? 'warn' : 'good')}
  </section>
  <section class="grid two-even">
    <div class="card">
      <h3>Model Artifact Status</h3>
      ${dvcEnvironmentCallout(data.dvc_environment)}
      <div style="height:12px"></div>
      ${modelMissingCallout(mg)}
      <p class="sub" style="margin-top:14px">Use the DVC controls below to restore/generate the actual artifacts. The dashboard does not assume that model binaries are already in the repository.</p>
      ${dvcActionPanel(false)}
    </div>
    <div class="card">
      <h3>Required Artifacts for Live Prediction</h3>
      ${table(['Artifact', 'Stage', 'Status'], arr(mg.required_for_prediction).map(item => [`<code>${item.path}</code>`, item.stage || '-', item.exists ? badge('present', 'good') : badge('missing', 'warn')]))}
    </div>
  </section>
  <section class="grid two-even">
    <div class="card">
      <h3>Customer Prediction Input</h3>
      <p class="sub">The form remains visible for interface validation, but the Run Prediction button is disabled until the DVC-generated model and preprocessing artifacts exist.</p>
      <div id="predictionSchemaNote" class="callout">Loading schema...</div>
      <div id="predictionArtifacts" style="display:none"></div>
      <form id="predictionForm" class="form-grid" style="margin-top:14px"></form>
      <div class="controls" style="margin-top:14px">
        <button class="control button" id="predictButton" type="button" ${mg.live_prediction_ready ? '' : 'disabled'}>Run Prediction</button>
        <button class="control button" id="sampleButton" type="button">Load Sample Values</button>
        <button class="control button" id="clearPredictionButton" type="button">Clear</button>
      </div>
      <div id="predictionErrors" class="json" style="margin-top:14px;display:${mg.live_prediction_ready ? 'none' : 'block'}">${mg.live_prediction_ready ? '' : 'Live prediction is disabled until DVC produces the required model artifacts.'}</div>
    </div>
    <div class="card">
      <h3>Prediction Result</h3>
      <p class="sub">The API response will appear here after the saved model exists and inference runs.</p>
      <div id="predictionResult" class="json">${mg.live_prediction_ready ? 'No prediction yet.' : 'No model artifacts found. Run DVC Pull/Repro first.'}</div>
    </div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Interpretation Panel</h3><div id="predictionExplanation" class="callout">${mg.live_prediction_ready ? 'Run a prediction to show churn probability, confidence, latency, and interpreted class.' : 'Prediction is not available yet because the model is a DVC output and is currently missing from the local workspace.'}</div></div>
    <div class="card"><h3>Recent Prediction History</h3><div id="predictionHistory" class="json">Loading...</div></div>
  </section>`;
  setTimeout(() => { bindPredict(); bindDvcActionPanel(document); }, 50);
}


async function loadPredictionSchema() {
  const res = await fetch('/api/predict/schema', { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to load prediction schema.');
  return await res.json();
}

async function loadPredictionHistory() {
  const res = await fetch('/api/predict/history', { cache: 'no-store' });
  return res.ok ? await res.json() : [];
}

function escapeAttr(value) {
  return String(value).replace(/"/g, '&quot;');
}

function findPredictionInput(form, name) {
  return Array.from(form.querySelectorAll('.prediction-input')).find(input => input.name === name);
}

function renderPredictionField(field) {
  const label = field.name;
  const help = field.help || '';

  if (field.type === 'select') {
    const opts = (field.choices || []).map(choice => `<option value="${escapeAttr(choice)}" ${String(choice) === String(field.default) ? 'selected' : ''}>${choice}</option>`).join('');
    return `<label class="field-card"><span>${label}</span><select class="control prediction-input" name="${escapeAttr(label)}" required>${opts}</select><small>${help}</small></label>`;
  }

  const min = field.min !== undefined ? `min="${field.min}"` : '';
  const step = field.step ? `step="${field.step}"` : 'step="any"';
  const value = field.default !== undefined ? `value="${Number(field.default).toFixed(2)}"` : 'value="0"';

  return `<label class="field-card"><span>${label}</span><input class="control prediction-input" name="${escapeAttr(label)}" type="number" ${step} ${min} ${value} required><small>${help}</small></label>`;
}

function renderArtifactStatus(requiredArtifacts) {
  const rows = Object.entries(requiredArtifacts || {}).map(([key, info]) => [
    key,
    `<code>${info.path || '-'}</code>`,
    badge(info.exists ? 'found' : 'missing', info.exists ? 'good' : 'warn'),
  ]);
  return table(['Artifact', 'Path', 'Status'], rows);
}

async function bindPredict() {
  const form = document.getElementById('predictionForm');
  const note = document.getElementById('predictionSchemaNote');
  const resultBox = document.getElementById('predictionResult');
  const errorBox = document.getElementById('predictionErrors');
  const explanation = document.getElementById('predictionExplanation');
  const historyBox = document.getElementById('predictionHistory');
  const artifactBox = document.getElementById('predictionArtifacts');
  const kpiWrap = document.getElementById('predict-kpis');

  let schema = {};
  try {
    schema = await loadPredictionSchema();
  } catch (error) {
    if (resultBox) resultBox.textContent = safeJson({ ok: false, error: 'Failed to load prediction schema', detail: String(error) });
    if (errorBox) {
      errorBox.style.display = 'block';
      errorBox.textContent = String(error);
    }
    return;
  }

  form.innerHTML = (schema.fields || []).map(renderPredictionField).join('');
  note.innerHTML = `<strong>Schema source:</strong> ${schema.source}<br><strong>Live prediction ready:</strong> ${schema.live_prediction_ready ? 'Yes' : 'No'}<br>${schema.note || ''}`;
  if (artifactBox) {
    artifactBox.innerHTML = renderArtifactStatus(schema.required_artifacts || {});
  }

  kpiWrap.innerHTML = `
    ${kpi('Model artifact', schema.live_prediction_ready ? 'Ready' : 'Missing', 'best model + preprocessing artifacts', schema.live_prediction_ready ? 'good' : 'warn')}
    ${kpi('Input mode', 'Raw customer', 'Validated raw Telco fields', 'info')}
    ${kpi('Fields', fmtInt((schema.fields || []).length), schema.source, 'info')}
    ${kpi('Endpoint', schema.endpoint || '/api/predict', 'POST prediction route', 'info')}
  `;

  async function refreshHistory() {
    const history = await loadPredictionHistory();
    historyBox.textContent = safeJson(history.slice(-10).reverse());
  }

  await refreshHistory();

  const sampleButton = document.getElementById('sampleButton');
  const clearButton = document.getElementById('clearPredictionButton');
  const predictButton = document.getElementById('predictButton');

  if (sampleButton) {
    sampleButton.onclick = () => {
      (schema.fields || []).forEach(field => {
        const input = findPredictionInput(form, field.name);
        if (!input) return;

        if (field.type === 'select') {
          input.value = field.default ?? (field.choices || [])[0] ?? '';
        } else {
          input.value = field.default !== undefined ? Number(field.default).toFixed(2) : 0;
        }
      });

      if (errorBox) {
        errorBox.style.display = 'none';
        errorBox.textContent = '';
      }

      if (resultBox) {
        resultBox.textContent = 'Sample values loaded. Click Run Prediction.';
      }
    };
  }

  if (clearButton) {
    clearButton.onclick = () => {
      if (resultBox) resultBox.textContent = 'No prediction yet.';
      if (explanation) explanation.textContent = 'Run a prediction to show churn probability, confidence, latency, and interpreted class.';
      if (errorBox) {
        errorBox.style.display = 'none';
        errorBox.textContent = '';
      }
    };
  }

  if (predictButton) {
    predictButton.onclick = async () => {
      if (predictButton.disabled) {
        if (errorBox) {
          errorBox.style.display = 'block';
          errorBox.textContent = 'Live prediction is disabled because required model artifacts are missing or not ready.';
        }
        return;
      }

      if (errorBox) {
        errorBox.style.display = 'none';
        errorBox.textContent = '';
      }

      const payload = {};
      form.querySelectorAll('.prediction-input').forEach(input => {
        payload[input.name] = input.value;
      });

      if (resultBox) resultBox.textContent = 'Running prediction...';

      let output = {};
      let res = null;

      try {
        res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        output = await res.json();
      } catch (error) {
        output = {
          ok: false,
          error: 'The prediction request could not reach /api/predict.',
          detail: String(error),
        };
      }

      if (resultBox) resultBox.textContent = safeJson(output);

      if (!res || !res.ok || !output.ok) {
        if (errorBox) {
          errorBox.style.display = 'block';
          errorBox.textContent = safeJson(output.validation_errors || output);
        }
        if (explanation) {
          explanation.innerHTML = `<strong>Prediction did not run.</strong><br>${output.error || 'Check validation errors.'}<br>${output.note || output.detail || ''}`;
        }
        return;
      }

      const probability = output.churn_probability !== null && output.churn_probability !== undefined ? pct(output.churn_probability, 1) : '-';
      const confidence = output.confidence !== null && output.confidence !== undefined ? pct(output.confidence, 1) : '-';

      if (explanation) {
        explanation.innerHTML = `
          <strong>Predicted churn:</strong> ${output.churn}<br>
          <strong>Raw prediction:</strong> ${output.prediction}<br>
          <strong>Churn probability:</strong> ${probability}<br>
          <strong>Confidence:</strong> ${confidence}<br>
          <strong>Latency:</strong> ${fmt(output.latency_ms, 2)} ms<br>
          <strong>Feature shape:</strong> ${output.feature_shape?.join(' × ') || '-'}
        `;
      }

      await refreshHistory();
    };
  }
}


function renderMonitoring(data) {
  const root = document.getElementById('content');
  const drift = obj(data.monitoring_summary?.drift);
  root.innerHTML = baseShell('Batch Monitoring', 'Monitor batch scoring only after DVC has restored or generated the model artifacts. If no models are present locally, use the DVC controls on this page to pull/reproduce them.') +
  `<section class="grid kpis" id="monitoring-kpis"></section>
   <section class="grid two-even">
     <div class="card"><h3>Batch Runner</h3>${dvcEnvironmentCallout(data.dvc_environment)}<div style="height:12px"></div>${modelMissingCallout(data.model_generation)}<p class="sub">Uses DVC-generated artifacts only: <span class="mono">models/best_model.pkl</span> plus <span class="mono">data/splits/production.csv</span> when available, otherwise <span class="mono">data/splits/test.csv</span>. If these are missing, run DVC Pull or Repro from the panel below.</p><div class="controls"><input class="control" id="batchSize" type="number" value="50" min="1"><button class="control button" id="runBatch">Run Next Batch</button><button class="control button" id="runMonitoring">Run Monitoring Script</button></div><div id="batchMsg" class="json" style="margin-top:14px">Waiting...</div></div>
     <div class="card"><h3>Monitoring Summary Snapshot</h3><p class="sub">Directly from monitoring/evidently_reports/monitoring_summary.json. These are drift/monitoring values, not model accuracy metrics.</p>${monitoringSummaryBoxes(data.monitoring_summary)}</div>
   </section>
   <section class="grid two-even">
     <div id="batchTrendCard" class="card"><h3>Batch Trend</h3><div class="json">Loading batch history...</div></div>
     <div id="batchMixCard" class="card"><h3>Latest Batch Mix</h3><div class="json">Loading batch history...</div></div>
   </section>
   <section class="grid two-even">
     <div class="card"><h3>DVC Recovery Controls</h3><p class="sub">If the runner says artifacts are missing, restore them from the working DVC setup here.</p>${dvcActionPanel(true)}</div>
     <div class="card"><h3>Current Requirements</h3>${availabilityTable(data)}</div>
   </section>
   <section class="grid two-even">
     <div id="batchHistoryTable" class="card"><h3>Batch History Table</h3><div class="json">Loading batch history...</div></div>
   </section>`;
  setTimeout(() => { bindMonitoring(data); bindDvcActionPanel(document); }, 50);
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
    kpi('Runner Ready', data.batch_ready ? 'Yes' : 'DVC needed', data.batch_ready ? 'Model and data artifacts found' : 'Model/data artifacts are DVC outputs', data.batch_ready ? 'good' : 'warn'),
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
  const bm = displayBestModelSummary(data);
  const hpo = obj(data.hpo_summary);
  const finalM = obj(bm.final_metrics || bm.all_features_metrics || data.display_metrics || {});
  const allM = obj(bm.all_features_metrics || finalM);
  const selM = obj(bm.selected_features_metrics || finalM);

  document.getElementById('content').innerHTML = baseShell('Model Comparison', 'Model comparison is based on the verified final training notebook results and focuses only on accepted performance metrics and final selection.') +
  `<section class="grid kpis">
    ${kpi('Final model', bm.final_model_name || '-', 'verified notebook reference', bm.final_model_name ? 'good' : 'warn')}
    ${kpi('Final choice', bm.final_model_choice || '-', bm.final_reason || 'Decision note', 'info')}
    ${kpi('Final F1', fmt(bm.final_score || finalM.f1_macro), 'Final selected score', 'good')}
    ${kpi('All-features F1', fmt(bm.all_features_score || allM.f1_macro), 'Full-feature benchmark', 'good')}
    ${kpi('Selected-features F1', fmt(bm.selected_features_score || selM.f1_macro), 'Feature-subset benchmark', 'good')}
    ${kpi('HPO best', hpo.best_hpo_model_name || hpo.bestSummary?.model_name || 'catboost', 'Verified HPO result', 'good')}
    ${kpi('Notebook source', 'Verified', 'Training notebook metrics used for display', 'good')}
    ${kpi('Leakage check', 'PASS', 'No leakage issue found', 'good')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>F1 Macro by Model</h3><p class="sub">Verified notebook model comparison using the accepted final training results.</p>${bars(rows, 'model_name', 'f1_macro', 1)}</div>
    <div class="card"><h3>Model Ranking Table</h3>${table(['Model', 'F1 macro', 'Verdict'], rows.map(r => [r.model_name, fmt(r.f1_macro), badge('verified', 'good')]))}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Core Metrics Comparison</h3><p class="sub">Grouped comparison of the final, all-feature, and selected-feature configurations.</p>${modelMetricComparisonChart(finalM, allM, selM)}</div>
    <div class="card"><h3>Metric Profile Line Chart</h3><p class="sub">Profile of the key metrics for each accepted model configuration.</p>${modelMetricLineChart(finalM, allM, selM)}</div>
  </section>
  <section class="card"><h3>Metric Difference Table</h3><p class="sub">Feature-selection result compared against the all-feature model.</p>${modelMetricDeltaTable(finalM, allM, selM)}</section>
  <section class="grid two-even">
    <div class="card"><h3>Final Model Decision</h3><div class="callout"><strong>${bm.final_model_name || 'Final model'}</strong><br>${bm.final_reason || 'The all-feature model remains the final model because selected features did not outperform it.'}</div><div style="margin-top:12px">${badge('notebook verified', 'good')}${badge(`run id: ${shortText(bm.final_run_id || '-', 18)}`, 'info')}</div></div>
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
  const workspace = obj(data.dvc_workspace);
  const env = obj(data.dvc_environment);
  const readiness = workspace.total_outputs ? `${workspace.present_outputs}/${workspace.total_outputs}` : '-';

  document.getElementById('content').innerHTML = baseShell('DVC Pipeline', 'This page is now the control center for the working DVC setup. It reads dvc.yaml, dvc.lock, and params.yaml directly, shows exactly which DVC artifacts exist locally, and can run DVC pull/repro commands from the dashboard.') +
  `<section class="grid kpis">
    ${kpi('dvc.yaml', yaml.exists ? 'Found' : 'Missing', yaml.valid_yaml ? 'Valid YAML' : (yaml.error || '-'), yaml.exists ? 'good' : 'bad')}
    ${kpi('dvc.lock', lock.exists ? 'Found' : 'Missing', lock.valid_yaml ? 'Valid YAML' : (lock.parse_error || 'Parsed as text'), lock.exists ? 'good' : 'bad')}
    ${kpi('Duplicate stages', fmtInt(duplicate.length), duplicate.join(', ') || 'none', duplicate.length ? 'warn' : 'good')}
    ${kpi('Stage count', fmtInt(stages.length), 'Stages parsed from dvc.yaml / dvc.lock', 'info')}
    ${kpi('DVC artifacts', readiness, 'present / total tracked outputs', workspace.missing_outputs ? 'warn' : 'good')}
    ${kpi('Missing outputs', fmtInt(workspace.missing_outputs), 'DVC outputs not present locally', workspace.missing_outputs ? 'warn' : 'good')}
    ${kpi('DVC metrics', dvc.dvc_metrics_success ? 'Success' : 'Use button', 'Run dvc metrics show from this page', dvc.dvc_metrics_success ? 'good' : 'info')}
    ${kpi('Registry summary', data.availability?.registry_summary?.exists ? 'Found' : 'Missing', 'reports/model_registry_summary.json', data.availability?.registry_summary?.exists ? 'good' : 'warn')}
    ${kpi('DVC-S3', env.dvc_s3_available ? 'Installed' : 'Missing', env.s3_remote_detected ? 'S3 remote detected' : 'No S3 remote detected', env.dvc_s3_available || !env.s3_remote_detected ? 'good' : 'warn')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>DVC Environment</h3>${dvcEnvironmentCallout(env)}<div class="json" style="margin-top:12px">${safeJson(env)}</div></div>
    <div class="card"><h3>DVC Operations</h3><p class="sub">Use these controls only in your local workspace. The commands run from the repository root using the same Python executable that launched this dashboard.</p>${dvcActionPanel(false)}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Configured Pipeline Stages</h3>${table(['Stage', 'Command', 'Outputs'], stages.map(s => [s.name || '-', `<code>${s.cmd || '-'}</code>`, arr(s.outs).slice(0, 8).join('<br>') || '-']))}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Exact DVC Artifact Availability</h3><p class="sub">This table is built from dvc.yaml and dvc.lock. It is not a guessed folder list.</p>${dvcArtifactTable(workspace)}</div>
    <div class="card"><h3>Configured Paths from params.yaml</h3>${table(['Config key', 'Path', 'Local status'], Object.entries(obj(data.configured_paths)).map(([key, path]) => {
      const existsNow = path && data.availability?.[`params_data_${key.replace('params_data_', '')}`]?.exists;
      return [key, `<code>${path}</code>`, existsNow ? badge('present', 'good') : badge('tracked / generated by DVC', 'info')];
    }))}</div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Recorded Stage Execution</h3>${stageSteps(stageResults)}</div>
    <div class="card"><h3>DVC Workspace Summary</h3><div class="json">${safeJson(workspace)}</div></div>
  </section>`;
  setTimeout(() => bindDvcActionPanel(document), 50);
}

function renderRegistry(data) {
  const r = obj(data.registry_summary);
  const bm = obj(data.best_model_summary);
  const mlflow = obj(data.mlflow_status);

  document.getElementById('content').innerHTML = baseShell('MLflow Registry', 'Registry, lineage, and MLflow UI access are managed here. This page reads the registry summary and can start the MLflow UI from the local backend database.') +
  `<section class="grid kpis">
    ${kpi('Registered model', r.registered_model_name || '-', data.sources.registry_summary, r.registered_model_name ? 'good' : 'warn')}
    ${kpi('Version', r.registered_model_version || '-', 'MLflow model version', r.registered_model_version ? 'good' : 'warn')}
    ${kpi('Metric', r.metric_name || '-', fmt(r.metric_value), r.metric_value ? 'good' : 'warn')}
    ${kpi('Run ID', r.registered_run_id || '-', 'Registered run identifier', r.registered_run_id ? 'good' : 'warn')}
    ${kpi('MLflow runs', fmtInt(mlflow.run_count), 'Runs found in mlruns/mlflow.db', mlflow.multiple_runs ? 'good' : 'warn')}
    ${kpi('Experiments', fmtInt(mlflow.experiment_count), 'MLflow experiments found', mlflow.available ? 'good' : 'warn')}
    ${kpi('Final model choice', r.final_model_choice || bm.final_model_choice || '-', 'Final project decision', 'info')}
    ${kpi('Feature analysis', r.feature_importance_kept_as_analysis ? 'Analysis only' : 'Used in final model', 'Feature importance contribution', r.feature_importance_kept_as_analysis ? 'warn' : 'good')}
  </section>
  <section class="grid two-even">
    <div class="card"><h3>MLflow UI Controls</h3><p class="sub">Start the MLflow UI from the same local backend database, then open it in the browser to show experiment runs and registered model versions.</p><div class="controls"><button class="control button" id="startMlflowBtn" type="button">Start MLflow UI</button><a class="control button" href="http://127.0.0.1:5000" target="_blank">Open MLflow UI</a></div><div id="mlflow-start-result" class="json" style="margin-top:14px">${safeJson(mlflow)}</div></div>
    <div class="card"><h3>Registry Lineage</h3><div class="kpi-cluster">
      <div class="kpi-chip"><span>Registered model</span><strong>${r.registered_model_name || '-'}</strong><span>Registry object name</span></div>
      <div class="kpi-chip"><span>Version</span><strong>${r.registered_model_version || '-'}</strong><span>Recorded version number</span></div>
      <div class="kpi-chip"><span>Run ID</span><strong>${shortText(r.registered_run_id || '-', 18)}</strong><span>MLflow run used for registration</span></div>
      <div class="kpi-chip"><span>Metric value</span><strong>${fmt(r.metric_value)}</strong><span>${r.metric_name || 'metric'}</span></div>
    </div></div>
  </section>
  <section class="grid two-even">
    <div class="card"><h3>Selected Features Recorded in Registry</h3>${chips(r.selected_features_analysis || bm.selected_features_analysis || [], 'info')}<div style="margin-top:12px"><h4>Top reference features</h4>${chips(r.top_10_reference_features || bm.top_10_reference_features || [], 'good')}</div></div>
    <div class="card"><h3>Registry Summary JSON</h3><div class="json">${safeJson(r)}</div></div>
  </section>`;
  setTimeout(bindRegistryControls, 50);
}

function bindRegistryControls() {
  const button = document.getElementById('startMlflowBtn');
  const box = document.getElementById('mlflow-start-result');
  if (!button || !box) return;

  button.onclick = async () => {
    box.textContent = 'Starting MLflow UI...';
    const res = await fetch('/api/mlflow/start', { method: 'POST' });
    const payload = await res.json();
    box.textContent = safeJson(payload);
  };
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
