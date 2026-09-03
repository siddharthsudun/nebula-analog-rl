// SILQ Dashboard -- vanilla JS, no framework, no build step. Talks to server.py's
// JSON API, which wraps the same eqrl functions eqrl.solve and the benchmarks use.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const ICONS = {
  check: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
  cross: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>',
  dash: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M5 12h14"/></svg>',
  checkCircle: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.3 2.3 5-5.3"/></svg>',
  xCircle: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/></svg>',
  alertTriangle: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 4l9 15H3z"/><path d="M12 10v4"/><circle cx="12" cy="17" r="0.4" fill="currentColor"/></svg>',
  info: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><circle cx="12" cy="8.3" r="0.4" fill="currentColor"/></svg>',
};

function el(tag, attrs = {}, html) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmt(v, decimals = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(decimals);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} ${text.slice(0, 300)}`);
  }
  return res.json();
}
function postJSON(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function bindRangeFill(input) {
  const update = () => {
    const min = Number(input.min) || 0, max = Number(input.max) || 100, val = Number(input.value);
    const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
    input.style.setProperty("--pct", `${pct}%`);
  };
  input.addEventListener("input", update);
  update();
}

function banner(kind, iconName, html) {
  return `<div class="banner ${kind}">${ICONS[iconName]}<div>${html}</div></div>`;
}

function kpiGrid(items) {
  return `<div class="kpi-grid">${items.map((it) => `
    <div>
      <div class="kpi-label">${escapeHtml(it.label)}</div>
      <div class="kpi-value${it.accent ? " accent" : ""}">${it.value}</div>
      ${it.help ? `<div class="kpi-help">${escapeHtml(it.help)}</div>` : ""}
    </div>`).join("")}</div>`;
}

function toast(message, ok) {
  let wrap = $(".toast-container");
  if (!wrap) {
    wrap = el("div", { class: "toast-container" });
    document.body.appendChild(wrap);
  }
  const t = el("div", { class: `toast ${ok ? "ok" : "fail"}` },
    `${ICONS[ok ? "check" : "cross"]}<span>${escapeHtml(message)}</span>`);
  wrap.appendChild(t);
  setTimeout(() => {
    t.style.transition = "opacity 0.25s";
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 260);
  }, 3600);
}

// -- Router -------------------------------------------------------------------

function initRouter() {
  $$(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".nav-item").forEach((b) => b.classList.remove("active"));
      $$(".view").forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      $(`#view-${btn.dataset.view}`).classList.add("active");
    });
  });
}

async function checkBackend() {
  const dot = $("#backend-dot"), label = $("#backend-status");
  try {
    await api("/api/pipeline/defaults");
    dot.classList.add("up");
    label.textContent = "backend connected";
  } catch {
    dot.classList.add("down");
    label.textContent = "backend unreachable";
  }
}

// -- Guard layer ----------------------------------------------------------------

const TIER_NAMES = {
  1: "Run integrity", 2: "Circuit sanity", 3: "Corner integrity",
  4: "Physical plausibility", 5: "Search pathology",
};

async function initGuard() {
  const data = await api("/api/guard/presets");

  const vddInput = $("#guard-vdd");
  vddInput.min = data.vdd.min;
  vddInput.max = data.vdd.max;
  vddInput.value = data.vdd.nominal;
  $("#guard-vdd-hint").textContent =
    `Spec nominal is ${data.vdd.nominal} V ±${(data.vdd.tolerance * 100).toFixed(0)}%.`;
  bindRangeFill(vddInput);
  const updateVddLabel = () => { $("#guard-vdd-val").textContent = fmt(vddInput.value, 2); };
  vddInput.addEventListener("input", updateVddLabel);
  updateVddLabel();

  const grid = $("#guard-fields");
  grid.innerHTML = data.field_meta.map((f) => `
    <label class="field">
      <span class="field-label">${escapeHtml(f.label)}</span>
      <input type="number" step="${f.step}" data-field="${f.key}" />
    </label>`).join("");

  const presetRow = $("#guard-presets");
  presetRow.innerHTML = "";
  data.presets.forEach((p) => {
    const b = el("button", { class: "btn", type: "button", title: p.description }, escapeHtml(p.label));
    b.addEventListener("click", () => applyGuardPreset(p));
    presetRow.appendChild(b);
  });
  const defaultsPreset = data.presets.find((p) => p.id === "defaults");
  if (defaultsPreset) applyGuardPreset(defaultsPreset);

  $("#guard-evaluate-btn").addEventListener("click", runGuardEvaluate);
  $("#guard-corner-check-btn").addEventListener("click", runCornerCheck);
}

function applyGuardPreset(preset) {
  for (const [key, val] of Object.entries(preset.fields)) {
    const input = $(`#guard-fields [data-field="${key}"]`);
    if (input) input.value = val;
  }
}

function readGuardFields() {
  const fields = {};
  $$("#guard-fields [data-field]").forEach((inp) => { fields[inp.dataset.field] = Number(inp.value); });
  return fields;
}

async function runGuardEvaluate() {
  const btn = $("#guard-evaluate-btn");
  const corner = $("#guard-corner").value;
  const box = $("#guard-result");
  btn.disabled = true;
  box.innerHTML = `<div class="card"><div class="spinner-line"><span class="spinner"></span>Running real ngspice on the ${escapeHtml(corner)} corner…</div></div>`;
  try {
    const data = await postJSON("/api/guard/evaluate", {
      fields: readGuardFields(), corner, vdd: Number($("#guard-vdd").value),
    });
    renderGuardResult(data);
  } catch (err) {
    box.innerHTML = `<div class="card">${banner("danger", "xCircle", `Request failed: ${escapeHtml(err.message)}`)}</div>`;
  } finally {
    btn.disabled = false;
  }
}

async function runCornerCheck() {
  const btn = $("#guard-corner-check-btn");
  btn.disabled = true;
  try {
    const data = await postJSON("/api/guard/verify-corners", { corner: $("#guard-corner").value });
    toast(data.ok
      ? "Tier 3 passed: tt and ss corner models are genuinely different."
      : `Tier 3 FAILED: ${data.reason}`, data.ok);
  } catch (err) {
    toast(`Request failed: ${err.message}`, false);
  } finally {
    btn.disabled = false;
  }
}

function renderGuardResult(data) {
  const box = $("#guard-result");
  if (data.valid) {
    const m = data.metrics;
    const hp = data.hard_pass;
    box.innerHTML = `
      <div class="card">
        ${banner("success", "checkCircle", "<strong>VALID</strong> — tiers 1, 2 and 4 all passed. This is a real, physically-plausible amplifier.")}
        ${kpiGrid([
          { label: "Boost", value: fmt(m.boost_db, 2) + " dB" },
          { label: "DC gain", value: fmt(m.dc_gain_db, 2) + " dB" },
          { label: "Peak freq", value: fmt(m.peak_freq_ghz, 2) + " GHz" },
          { label: "Power", value: fmt(m.power_w * 1e3, 2) + " mW" },
          { label: "HD3", value: fmt(m.hd3_db, 1) + " dB" },
          { label: "Noise", value: Math.round(m.noise_vrms * 1e6) + " µVrms" },
          { label: "Eye height", value: fmt(m.eye_h_ui, 2) + " UI" },
          { label: "Eye width", value: Math.round(m.eye_v_mv) + " mV" },
        ])}
        <div style="margin-top:18px; padding-top:16px; border-top:1px solid var(--border);">
          <p style="font-size:var(--fs-sm); margin-bottom:10px;"><strong>Guard-valid is not the same question as spec-passing.</strong> Here is the same design scored against <code>eqrl.specs.hard_pass</code> (<code>DEFAULT_SPEC</code>):</p>
          <div class="row wrap" style="margin-bottom:10px;">
            <span class="badge ${hp.ok ? "valid" : "warning"}">${hp.ok ? "PASSES all hard specs" : "fails the hard spec"}</span>
          </div>
          <div class="row wrap" style="font-size:var(--fs-xs); color:var(--ink-muted); gap:14px 16px;">
            ${Object.entries(hp.checks).map(([k, v]) => `<span class="row" style="gap:5px;"><span style="color:${v ? "var(--success)" : "var(--danger)"}">${ICONS[v ? "check" : "cross"]}</span>${escapeHtml(k)}</span>`).join("")}
          </div>
        </div>
        <details style="margin-top:14px;"><summary style="cursor:pointer; font-size:var(--fs-xs); color:var(--ink-faint);">Raw run artifacts</summary><pre class="code-block" style="margin-top:8px;">${escapeHtml(data.artifact_dir)}</pre></details>
      </div>`;
    return;
  }

  const tier = data.tier;
  const rows = data.reached_tiers.map((t) => {
    let cls, icon, note;
    if (t < tier) { cls = "pass"; icon = "check"; note = "passed"; }
    else if (t === tier) { cls = "fail"; icon = "cross"; note = "failed here"; }
    else { cls = "skip"; icon = "dash"; note = "not reached"; }
    return `<div class="tier-row ${cls}"><span class="tier-icon">${ICONS[icon]}</span><span class="tier-name">Tier ${t} — ${TIER_NAMES[t]}</span><span style="margin-left:auto; font-size:var(--fs-xs);">${note}</span></div>`;
  }).join("");

  box.innerHTML = `
    <div class="card">
      ${banner("danger", "xCircle", `<strong>INVALID</strong> — Tier ${tier} (${TIER_NAMES[tier]}): <code>${escapeHtml(data.check)}</code>`)}
      <p style="font-size:var(--fs-sm); color:var(--ink-muted); margin-bottom:8px;">${escapeHtml(data.reason)}</p>
      <p class="help-hint">${data.violation !== null
        ? `Violation magnitude: ${fmt(data.violation, 2)}× the failing bound's own scale (0 = exactly on the bound). Reported for context only — the guard's pass/fail decision never reads this number.`
        : "Violation magnitude: not computable for this failure (e.g. the solver never returned a number to measure a distance from)."}</p>
      <p style="font-size:var(--fs-sm); font-weight:600; margin-top:18px; margin-bottom:4px;">Tier-by-tier (evaluation stops at the first failure):</p>
      <div class="tier-list">
        ${rows}
        <div class="tier-row pending"><span class="tier-icon">${ICONS.dash}</span><span class="tier-name">Tier 3 — ${TIER_NAMES[3]}</span><span style="margin-left:auto; font-size:var(--fs-xs);">startup check, not part of this per-design path</span></div>
        <div class="tier-row pending"><span class="tier-icon">${ICONS.dash}</span><span class="tier-name">Tier 5 — ${TIER_NAMES[5]}</span><span style="margin-left:auto; font-size:var(--fs-xs);">only meaningful across a run's history</span></div>
      </div>
      <details style="margin-top:14px;"><summary style="cursor:pointer; font-size:var(--fs-xs); color:var(--ink-faint);">Raw run artifacts</summary><pre class="code-block" style="margin-top:8px;">${escapeHtml(data.artifact_dir)}</pre></details>
    </div>`;
}

// -- Results explorer --------------------------------------------------------------

let resultsRows = [];

async function initResults() {
  resultsRows = await api("/api/results");
  const sel = $("#results-picker");
  const featured = resultsRows.filter((r) => r.featured);
  const rest = resultsRows.filter((r) => !r.featured);
  sel.innerHTML = `
    <optgroup label="Featured">${featured.map((r) => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.label)}</option>`).join("")}</optgroup>
    <optgroup label="All files">${rest.map((r) => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.name)}</option>`).join("")}</optgroup>`;
  $("#results-count").textContent = `${featured.length} featured, ${resultsRows.length} total`;
  sel.addEventListener("change", () => loadResult(sel.value));
  const first = featured[0] || resultsRows[0];
  if (first) { sel.value = first.name; loadResult(first.name); }
}

async function loadResult(name) {
  const detail = $("#results-detail");
  detail.innerHTML = `<div class="spinner-line"><span class="spinner"></span>Loading…</div>`;
  const row = resultsRows.find((r) => r.name === name) || {};
  let data;
  try {
    data = await api(`/api/results/${encodeURIComponent(name)}`);
  } catch (err) {
    detail.innerHTML = banner("danger", "xCircle", `Failed to load: ${escapeHtml(err.message)}`);
    return;
  }

  const header = `
    <div class="artifact-header">
      <h2>${escapeHtml(row.label || name)}</h2>
      ${row.blurb ? `<p>${escapeHtml(row.blurb)}</p>` : ""}
      <div class="artifact-path">results/${escapeHtml(name)} — ${row.size_kb ?? "?"} KB</div>
    </div>`;

  if (name === "delivered_circuit.json") { detail.innerHTML = header + deliveredHtml(data); return; }
  if (name === "pass_vs_valid.json") {
    detail.innerHTML = header + passVsValidHtml(data);
    wirePassVsValidChart(detail, data);
    return;
  }
  if (name === "target_tracking_clean40k.json") {
    detail.innerHTML = header + targetTrackingHtml(data);
    wireTargetTrackingChart(detail, data);
    return;
  }
  detail.innerHTML = header + genericHtml(data);
}

function rawJsonBlock(d) {
  return `<details class="explainer" style="margin-top:14px;"><summary>Raw JSON</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(JSON.stringify(d, null, 2))}</pre></div></details>`;
}

function deliveredHtml(d) {
  const tt = d.pvt.tt_nominal;
  const pvt = d.pvt;
  const p = d.provenance;
  const dv = d.design;
  const wc = pvt.worst_case_by_metric;
  const wcRows = Object.entries(wc).map(([metric, v]) => `
    <tr>
      <td style="font-family:var(--font-ui);">${escapeHtml(metric)}</td>
      <td>${fmt(v.min, 3)}</td>
      <td style="font-size:10.5px; color:var(--ink-faint);">${escapeHtml(v.min_at)}</td>
      <td>${fmt(v.max, 3)}</td>
      <td style="font-size:10.5px; color:var(--ink-faint);">${escapeHtml(v.max_at)}</td>
      <td>${v.limit_lo ?? "—"}</td>
      <td>${v.limit_hi ?? "—"}</td>
    </tr>`).join("");

  return `
    <div class="section-label">Design — eqrl.circuits.ctle.DesignVars</div>
    <div class="card">${kpiGrid([
      { label: "W / L", value: `${fmt(dv.w_in * 1e6, 2)} / ${fmt(dv.l_in * 1e6, 4)} µm` },
      { label: "I_tail", value: `${fmt(dv.i_tail * 1e6, 1)} µA` },
      { label: "Rs / Cs", value: `${fmt(dv.rs / 1e3, 2)} k / ${fmt(dv.cs * 1e15, 1)} f` },
      { label: "R_load", value: `${fmt(dv.r_load, 1)} Ω` },
    ])}</div>
    <div class="section-label">Performance at tt / nominal VDD / 27°C</div>
    <div class="card">${kpiGrid([
      { label: "Boost", value: fmt(tt.boost_db, 2) + " dB" },
      { label: "DC gain", value: fmt(tt.dc_gain_db, 2) + " dB" },
      { label: "Power", value: fmt(tt.power_w * 1e3, 2) + " mW" },
      { label: "Area", value: fmt(tt.area_mm2 * 1e3, 1) + " ×10⁻³ mm²" },
    ])}</div>
    <div class="card">${kpiGrid([
      { label: "PVT corners passed", value: `${pvt.corners_passed} / ${pvt.corners_total}` },
      { label: "All corners guard-valid", value: pvt.all_guard_valid ? "yes" : "no" },
      { label: "Worst corner", value: escapeHtml(pvt.worst_corner) },
      { label: "Worst-corner target error", value: fmt(pvt.worst_corner_target_err_db, 2) + " dB" },
    ])}</div>
    <div class="section-label">Worst case across all 45 corners, per metric</div>
    <div class="table-wrap"><table class="data">
      <thead><tr><th>Metric</th><th>Min</th><th>at</th><th>Max</th><th>at</th><th>Limit lo</th><th>Limit hi</th></tr></thead>
      <tbody>${wcRows}</tbody>
    </table></div>
    <div class="section-label">Provenance</div>
    <div class="card"><p style="font-size:var(--fs-sm); color:var(--ink-muted);">Policy <code>${escapeHtml(p.policy)}</code> (sha256 <code>${escapeHtml(p.policy_sha256.slice(0, 12))}…</code>) — started from <code>${escapeHtml(p.g32_start_source)}</code>, ${escapeHtml(p.g32_reason)}. ${p.total_evals} optimizer evaluations total.</p></div>
    <details class="explainer" style="margin-top:14px;"><summary>Final schematic (SPICE netlist)</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(d.netlist)}</pre></div></details>
    ${rawJsonBlock(d)}
  `;
}

function passVsValidHtml(d) {
  const totalPass = d.pass_valid + d.pass_invalid;
  const tableRows = d.designs.map((x) => `
    <tr>
      <td>${fmt(x.target_boost_db, 2)}</td>
      <td>${fmt(x.channel_loss_db, 2)}</td>
      <td>${fmt(x.boost_db, 2)}</td>
      <td>${Math.round(x.eye_v_mv)}</td>
      <td><span class="badge ${x.guard_valid ? "valid" : "invalid"}">${x.guard_valid ? "valid" : "REJECTED"}</span></td>
      <td style="font-family:var(--font-ui); font-weight:400; color:var(--ink-muted);">${escapeHtml(x.guard_reason || "")}</td>
    </tr>`).join("");

  return `
    <div class="card">${kpiGrid([
      { label: "Passed all 8 hard specs", value: String(totalPass) },
      { label: "…and were guard-valid", value: String(d.pass_valid) },
      { label: "…but were guard-rejected", value: String(d.pass_invalid) },
      { label: "Rejection rate", value: Math.round((100 * d.pass_invalid) / totalPass) + "%", accent: true },
    ])}</div>
    <div class="section-label">Why the 24 rejected designs failed</div>
    <div class="card chart-card"><div class="chart-container" data-chart="why-invalid"></div></div>
    <div class="section-label">Every design that passed the spec (4 CMA-ES runs × up to 60 evaluations, best-per-run)</div>
    <div class="table-wrap table-scroll"><table class="data">
      <thead><tr><th>Target boost</th><th>Channel loss</th><th>Achieved boost</th><th>Eye width</th><th>Guard</th><th>Reason</th></tr></thead>
      <tbody>${tableRows}</tbody>
    </table></div>
    ${rawJsonBlock(d)}
  `;
}

function wirePassVsValidChart(detail, d) {
  const mount = detail.querySelector('[data-chart="why-invalid"]');
  if (!mount) return;
  const entries = Object.entries(d.why_invalid).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, value }));
  renderBarChart(mount, entries, { ariaLabel: "why designs were guard-rejected" });
}

function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  return num / Math.sqrt(dx2 * dy2);
}

function targetTrackingHtml(d) {
  const targets = d.map((r) => r.target);
  const boosts = d.map((r) => r.boost);
  const corr = pearson(targets, boosts);
  const mae = d.reduce((s, r) => s + Math.abs(r.boost - r.target), 0) / d.length;
  return `
    <div class="card">
      ${kpiGrid([
        { label: "Held-out specs", value: String(d.length) },
        { label: "Correlation(requested, achieved)", value: fmt(corr, 3), accent: true },
        { label: "Mean abs error", value: fmt(mae, 2) + " dB" },
      ])}
      <p class="chart-caption">A policy that ignores the requested target and always returns the same design would still score well on the 3–12 dB range check — this correlation is what tells the two cases apart.</p>
    </div>
    <div class="card chart-card" style="margin-top:14px;"><div class="chart-container" data-chart="target-tracking"></div></div>
    ${rawJsonBlock(d)}
  `;
}

function wireTargetTrackingChart(detail, d) {
  const mount = detail.querySelector('[data-chart="target-tracking"]');
  if (!mount) return;
  const points = d.map((r) => ({
    x: r.target, y: r.boost,
    tooltip: `target ${fmt(r.target, 2)} dB → achieved ${fmt(r.boost, 2)} dB`,
  }));
  renderScatterChart(mount, points, {
    refLine: true, xLabel: "Requested boost (dB)", yLabel: "Achieved boost (dB)",
    ariaLabel: "requested vs achieved boost scatter plot",
  });
}

function formatScalar(v) {
  if (v === null) return "null";
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : escapeHtml(String(Math.round(v * 10000) / 10000));
  }
  return escapeHtml(String(v));
}

function formatCell(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return escapeHtml(JSON.stringify(v));
  return escapeHtml(String(v));
}

function tableHtml(rows) {
  const keys = Object.keys(rows[0]);
  const head = keys.map((k) => `<th>${escapeHtml(k)}</th>`).join("");
  const body = rows.map((r) => `<tr>${keys.map((k) => `<td>${formatCell(r[k])}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap table-scroll"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function genericHtml(d) {
  let out = "";
  if (d && typeof d === "object" && !Array.isArray(d)) {
    const scalars = Object.entries(d).filter(([, v]) => v === null || ["number", "string", "boolean"].includes(typeof v));
    if (scalars.length) {
      out += `<div class="card">${kpiGrid(scalars.slice(0, 6).map(([k, v]) => ({ label: k, value: formatScalar(v) })))}</div>`;
      if (scalars.length > 6) {
        out += `<div class="card">${kpiGrid(scalars.slice(6, 12).map(([k, v]) => ({ label: k, value: formatScalar(v) })))}</div>`;
      }
    }
    for (const [k, v] of Object.entries(d)) {
      if (Array.isArray(v) && v.length && typeof v[0] === "object") {
        out += `<div class="section-label">${escapeHtml(k)} (${v.length} rows)</div>${tableHtml(v)}`;
      }
    }
  } else if (Array.isArray(d) && d.length && typeof d[0] === "object") {
    out += tableHtml(d);
  }
  out += rawJsonBlock(d);
  return out;
}

// -- Live pipeline ------------------------------------------------------------------

let pipelineDefaults = null;

async function initPipeline() {
  pipelineDefaults = await api("/api/pipeline/defaults");

  const targetInput = $("#pipeline-target");
  targetInput.min = pipelineDefaults.boost_db_min;
  targetInput.max = pipelineDefaults.boost_db_max;
  targetInput.value = pipelineDefaults.target_boost_db;
  bindRangeFill(targetInput);
  const updateTarget = () => { $("#pipeline-target-val").textContent = fmt(targetInput.value, 1); };
  targetInput.addEventListener("input", updateTarget);
  updateTarget();

  const channelInput = $("#pipeline-channel");
  channelInput.value = pipelineDefaults.channel_loss_db;
  bindRangeFill(channelInput);
  const updateChannel = () => { $("#pipeline-channel-val").textContent = fmt(channelInput.value, 1); };
  channelInput.addEventListener("input", updateChannel);
  updateChannel();

  $("#pipeline-parse-btn").addEventListener("click", async () => {
    const text = $("#pipeline-text").value.trim();
    if (!text) return;
    const btn = $("#pipeline-parse-btn");
    btn.disabled = true;
    try {
      const spec = await postJSON("/api/pipeline/parse-spec", { text });
      targetInput.value = spec.target_boost_db; updateTarget();
      channelInput.value = spec.channel_loss_db; updateChannel();
      bindRangeFill(targetInput); bindRangeFill(channelInput);
    } catch (err) {
      toast(`Parse failed: ${err.message}`, false);
    } finally {
      btn.disabled = false;
    }
  });

  $("#pipeline-run-btn").addEventListener("click", runPipeline);
}

async function runPipeline() {
  const btn = $("#pipeline-run-btn");
  const box = $("#pipeline-result");
  btn.disabled = true;
  box.innerHTML = `<div class="spinner-line"><span class="spinner"></span>PPO rollout, then G3.2 refinement, then independent verification — all real ngspice calls. This can take a while.</div>`;
  try {
    const result = await postJSON("/api/pipeline/run", {
      target_boost_db: Number($("#pipeline-target").value),
      channel_loss_db: Number($("#pipeline-channel").value),
      spec_index: Number($("#pipeline-spec-index").value) || 0,
      allow_fallback: $("#pipeline-fallback").checked,
    });
    renderPipelineResult(result);
  } catch (err) {
    box.innerHTML = banner("danger", "xCircle", `Run failed: ${escapeHtml(err.message)}`);
  } finally {
    btn.disabled = false;
  }
}

function renderPipelineResult(r) {
  const box = $("#pipeline-result");
  const statuses = pipelineDefaults.statuses;
  const p = r.provenance || {}, c = r.cost, v = r.verification;

  let html = r.status === statuses.fallback
    ? banner("danger", "alertTriangle", "<strong>FIXED FALLBACK DESIGN — not generated by PPO or G3.2.</strong> The architecture returned nothing guard-valid for this spec. What follows does not read the requested target and must not be reported as an AI result.")
    : banner("success", "checkCircle", `architecture: PPO (${c.stage1_evals} evals) → G3.2 (${c.stage2_evals} evals)${p.g32_reached_target === false ? " — target NOT reached" : ""}`);

  html += `<div class="card">${kpiGrid([
    { label: "Optimizer evaluations", value: String(c.optimizer_evals) },
    { label: "measure_all calls", value: String(c.measure_all_total), help: `${c.measure_all_search} search + ${c.measure_all_verification} verification` },
    { label: "SPICE analyses", value: String(c.spice_analyses_total) },
    { label: "Budget unspent", value: String(c.budget_evals_unspent) },
  ])}</div>`;

  if (!r.design) {
    html += banner("warning", "alertTriangle", "No design: the architecture produced nothing guard-valid for this spec, and fallback was not allowed.");
  } else {
    const d = r.design;
    html += `<div class="section-label">Design</div><div class="card">${kpiGrid([
      { label: "W / L", value: `${fmt(d.w_in * 1e6, 2)} / ${fmt(d.l_in * 1e6, 4)} µm` },
      { label: "I_tail", value: `${fmt(d.i_tail * 1e6, 1)} µA` },
      { label: "Rs / Cs", value: `${fmt(d.rs / 1e3, 2)} k / ${fmt(d.cs * 1e15, 1)} f` },
      { label: "R_load", value: `${fmt(d.r_load, 1)} Ω` },
    ])}`;

    if (v && v.measures) {
      const m = v.measures;
      html += `<div style="margin-top:18px; padding-top:16px; border-top:1px solid var(--border);">
        <div class="card-title" style="margin-bottom:12px;">Independent re-verification</div>
        ${kpiGrid([
          { label: "Boost", value: fmt(m.boost_db, 3) + " dB", help: `requested ${fmt(r.spec.target_boost_db, 2)} ±${fmt(r.spec.boost_tol_db, 2)}` },
          { label: "Peak freq", value: fmt(m.peak_freq_ghz, 3) + " GHz" },
          { label: "DC gain", value: fmt(m.dc_gain_db, 2) + " dB" },
          { label: "abs error", value: fmt(v.abs_err_db, 3) + " dB" },
        ])}
        <div style="margin-top:16px;">${kpiGrid([
          { label: "HD3", value: fmt(m.hd3_db, 1) + " dB" },
          { label: "Noise", value: Math.round(m.noise_vrms * 1e6) + " µVrms" },
          { label: "Power", value: fmt(m.power_w * 1e3, 3) + " mW" },
          { label: "Eye", value: `${fmt(m.eye_h_ui, 2)} UI / ${Math.round(m.eye_v_mv)} mV` },
        ])}</div>
      </div>`;
    }
    html += `</div>`;

    if (!v) html += `<p class="help-hint" style="margin:12px 2px 0;">verification: not run</p>`;
    else if (!v.guard_valid) html += banner("danger", "xCircle", `verification: GUARD REJECTED — ${escapeHtml(v.guard_check)}`);
    else if (v.passed) html += banner("success", "checkCircle", "verification: ALL TEN CHECKS PASS (independent re-measurement)");
    else html += banner("warning", "alertTriangle", `verification: fails ${(v.failing || []).join(", ")}`);
  }

  html += `<details class="explainer" style="margin-top:8px;"><summary>Full CLI-style report</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(r.describe_text)}</pre></div></details>`;
  if (r.netlist) html += `<details class="explainer" style="margin-top:10px;"><summary>Final schematic (SPICE netlist)</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(r.netlist)}</pre></div></details>`;
  html += `<details class="explainer" style="margin-top:10px;"><summary>Raw result JSON</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(JSON.stringify(r, null, 2))}</pre></div></details>`;

  box.innerHTML = html;
}

// -- Boot -----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initRouter();
  checkBackend();
  initGuard().catch((err) => { $("#guard-result").innerHTML = `<div class="card">${banner("danger", "xCircle", `Failed to load guard layer: ${escapeHtml(err.message)}`)}</div>`; });
  initResults().catch((err) => { $("#results-detail").innerHTML = banner("danger", "xCircle", `Failed to load results: ${escapeHtml(err.message)}`); });
  initPipeline().catch((err) => { $("#pipeline-result").innerHTML = banner("danger", "xCircle", `Failed to load pipeline defaults: ${escapeHtml(err.message)}`); });
});
