// SILQ Dashboard -- vanilla JS, no framework, no build step. Talks to server.py's
// JSON API, which wraps the same eqrl functions eqrl.solve and the benchmarks use.
//
// Page shape: one scrolling column, Live Pipeline first (it is the product). Guard
// Layer, Results Explorer and the benchmark record are relocated -- not rewritten --
// into collapsed developer tools at the bottom; their functions below are otherwise
// unchanged from when they lived in their own sidebar views.

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

// Fetch wrapper that never swallows a failure into a blank panel. Two error shapes are
// understood: the new structured one the backend is being upgraded to emit --
// {"error_code","title","detail","hint"} -- and plain FastAPI {"detail": "..."}. Either
// way the thrown Error carries enough for the caller to render something honest; when
// the body is neither shape (or isn't JSON at all) the raw HTTP status and body text are
// kept so nothing renders as "[object Object]" or silently disappears.
async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (netErr) {
    // fetch only rejects when the request never reached a server. Left alone this
    // surfaces as "Failed to fetch", which tells a first-time reader nothing -- and it
    // is the single most likely failure here, because a long pipeline run outlives an
    // impatient Ctrl-C in the terminal the server was started from.
    const err = new Error("Cannot reach the SILQ server");
    err.title = "Cannot reach the SILQ server";
    err.detail = `The browser could not open a connection to ${location.host} (${netErr.message}). The server process is most likely stopped.`;
    err.hint = "Restart it from the repo root with: python -m uvicorn server:app --port "
             + (location.port || "8000") + "   then reload this page.";
    err.errorCode = "server_unreachable";
    throw err;
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let parsed = null;
    try { parsed = JSON.parse(text); } catch { /* not JSON */ }

    if (parsed && typeof parsed === "object" && (parsed.title || parsed.error_code)) {
      const err = new Error(parsed.title || parsed.detail || `${res.status} ${res.statusText}`);
      err.title = parsed.title || null;
      err.detail = parsed.detail || null;
      err.hint = parsed.hint || null;
      err.errorCode = parsed.error_code || null;
      err.label = parsed.label || null;      // "Error 101" -- quotable, stable, searchable
      err.status = res.status;
      throw err;
    }
    // FastAPI's own request validation puts a LIST in `detail` -- one object per bad
    // field -- so the string test below would miss it and dump raw JSON into the banner.
    // Flatten it to "body.target_boost_db: <msg>" lines instead.
    if (parsed && Array.isArray(parsed.detail)) {
      const lines = parsed.detail.map((d) =>
        `${(d.loc || []).join(".")}: ${d.msg || "invalid"}`).join("; ");
      const err = new Error(lines);
      err.title = "The server rejected these values";
      err.detail = lines;
      err.hint = "This is a malformed request rather than a design failure -- reload the page and try again.";
      err.status = res.status;
      throw err;
    }
    const plainDetail = parsed && typeof parsed === "object" && typeof parsed.detail === "string" ? parsed.detail : "";
    const err = new Error(plainDetail || `${res.status} ${res.statusText} ${text.slice(0, 300)}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}
function postJSON(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

// Renders either error shape api() can throw as a proper banner -- title prominent,
// detail beneath it, hint as the suggested next step -- or falls back to the plain
// message (which is itself either the server's `detail` string or "<status> <text>").
function errorBannerHtml(err, fallbackTitle) {
  if (err && err.title) {
    const label = err.label ? `<span class="err-code">${escapeHtml(err.label)}</span>` : "";
    return `<div class="status-banner st-error">${ICONS.xCircle}<div>
      <div class="status-title">${label}${escapeHtml(err.title)}</div>
      ${err.detail ? `<div class="status-sub">${escapeHtml(err.detail)}</div>` : ""}
      ${err.hint ? `<div class="status-sub" style="margin-top:4px;"><strong>Next step:</strong> ${escapeHtml(err.hint)}</div>` : ""}
    </div></div>`;
  }
  const msg = (err && err.message) ? err.message : "unknown error";
  return banner("danger", "xCircle", `${escapeHtml(fallbackTitle || "Request failed")}: ${escapeHtml(msg)}`);
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

// -- Health pill ----------------------------------------------------------------------
// GET /api/health -- {ready, warming, error, policy, detail} -- backed by a background
// warm-up thread that loads the SKY130 device models and the frozen policy at server
// startup (server.py's `_warm_up`/`_startup`). Its own docstring says it is meant to be
// polled, so this polls every 2s until it reaches a terminal state (ready or error)
// rather than reading it once and leaving the pill stuck on "warming" for the ~20s the
// real warm-up takes. If the endpoint is missing entirely (404) or errors outright, this
// falls back to the old reachability probe (GET /api/pipeline/defaults) so the pill still
// resolves to a terminal state instead of being stuck on "checking..." forever.

function setHealthPill(state, text) {
  const dot = $("#health-dot"), label = $("#health-label");
  dot.className = `status-dot ${state}`;
  label.textContent = text;
}

// Warm-up and a live run both drive the one resident, non-reentrant ngspice process
// (server.py's `_warm_up` now takes `_run_lock` for exactly this reason), so a click
// during "warming" would just wait behind that lock or bounce off its 409. Disabling the
// button is the UI half of that fix -- it stops the user from ever needing to see it.
function setRunButtonWarming(warming) {
  const btn = $("#pipeline-run-btn");
  if (!btn) return;
  btn.disabled = warming;
  btn.title = warming ? "Waiting for the server to finish loading device models…" : "";
}

// Returns true when the pill has reached a terminal state (ready or error) and polling
// should stop; false to keep polling.
async function checkHealthOnce() {
  try {
    const res = await fetch("/api/health");
    if (res.status === 404) throw new Error("no-health-endpoint");
    if (!res.ok) { setHealthPill("error", `backend error (${res.status})`); return true; }
    const data = await res.json();
    if (data.error) { setHealthPill("error", data.detail || data.error); setRunButtonWarming(false); return true; }
    if (data.ready === true) { setHealthPill("ready", data.detail || "Ready"); setRunButtonWarming(false); return true; }
    setHealthPill("warming", data.detail || "Loading device models…");
    setRunButtonWarming(true);
    return false;
  } catch {
    try {
      await api("/api/pipeline/defaults");
      setHealthPill("ready", "Ready");
    } catch {
      setHealthPill("error", "backend unreachable");
    }
    setRunButtonWarming(false);
    return true;
  }
}

function pollHealth() {
  checkHealthOnce().then((done) => { if (!done) setTimeout(pollHealth, 2000); });
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
    box.innerHTML = `<div class="card">${errorBannerHtml(err, "Request failed")}</div>`;
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
    toast(`Request failed: ${(err.title || err.message)}`, false);
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
          { label: "Eye width", value: fmt(m.eye_h_ui, 2) + " UI" },
          { label: "Eye height", value: Math.round(m.eye_v_mv) + " mV" },
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
    detail.innerHTML = errorBannerHtml(err, "Failed to load");
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
      <thead><tr><th>Target boost</th><th>Channel loss</th><th>Achieved boost</th><th>Eye height (mV)</th><th>Guard</th><th>Reason</th></tr></thead>
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

// -- Live pipeline: the ask ----------------------------------------------------------

let pipelineDefaults = null;
let selectedMode = "default";

// Fixed display order (fast -> thorough), independent of whatever order the server's
// `modes` list happens to come back in.
const MODE_ORDER = ["fastest", "default", "accurate", "thinking"];
const MODE_INFO = {
  fastest: {
    name: "Fastest",
    sub: "surrogate-hedged",
    desc: "Spends one evaluation on a kNN-surrogate-guided jump before G3.2 refinement, then hands off with a smaller budget. Fewer simulator calls and a shorter wall clock — the surrogate only ever picks where to spend one evaluation, it never decides a result.",
  },
  default: {
    name: "Default",
    sub: "benchmarked",
    desc: "The pre-registered PPO → G3.2 → verification pipeline, unchanged from the published benchmark.",
  },
  accurate: {
    name: "Accurate",
    sub: "tighter tolerance",
    desc: "Same PPO → G3.2 pipeline with a larger refinement budget and a tighter internal stopping tolerance, trading more evaluations for a closer boost match.",
  },
  thinking: {
    name: "Thinking",
    sub: "multi-rollout",
    desc: "Runs PPO → G3.2 independently from 3 different seeds and keeps whichever got closest to target. Costs roughly 3× the evaluations, and can solve specs a single rollout misses.",
  },
};

// The server has already scaled these to their display unit (watts -> mW, and so on);
// all that is left is choosing decimals so 0.05 mm² and 100 mV both read cleanly.
function prettySpec(v, unit) {
  if (v === null || v === undefined) return "—";
  const mag = Math.abs(v);
  const dp = Number.isInteger(v) ? 0 : mag >= 1 ? 2 : 3;
  return `${fmt(v, dp)} ${escapeHtml(unit)}`;
}

function renderParseTable(spec) {
  const rows = spec.fields || [];
  const conflicts = rows.filter((r) => r.conflict);
  const body = rows.map((r) => {
    const asked = prettySpec(r.asked_disp, r.unit);
    // Three states, and the difference between them is the whole point of this table:
    // applied (the run uses your number), matched (your number equals the frozen
    // default, so nothing is lost), and overridden (the run ignores your number).
    let tag, cls;
    if (r.applied) { tag = "applied to this run"; cls = "pf-applied"; }
    else if (r.conflict) { tag = `run enforces ${prettySpec(r.run_disp, r.unit)}`; cls = "pf-conflict"; }
    else { tag = "matches the frozen default"; cls = "pf-same"; }
    return `<tr><td class="mono">${escapeHtml(r.field)}</td><td class="num mono">${asked}</td><td class="${cls}">${tag}</td></tr>`;
  }).join("");
  const warn = conflicts.length
    ? `<div class="parse-conflict">${conflicts.length} constraint${conflicts.length > 1 ? "s" : ""} you gave ${conflicts.length > 1 ? "are" : "is"} <strong>not</strong> applied. <code>pipeline.design()</code> is frozen to two inputs — target boost and channel loss — so the rest are scored at the benchmarked defaults shown above. The delivered circuit is not being optimised against ${conflicts.length > 1 ? "those numbers" : "that number"}.</div>`
    : "";
  // A word like "gain" has two referents in this circuit. The parser picks one to avoid
  // refusing an ordinary request, and says so here: an interpretation the user cannot
  // see and correct is not meaningfully different from an invented one.
  const assumed = Object.entries(spec.assumptions || {});
  const assumeNote = assumed.length
    ? `<div class="parse-assume">${assumed.map(([f, why]) =>
        `<strong>${escapeHtml(f)}</strong> — ${escapeHtml(why)}`).join("<br>")}</div>`
    : "";
  return `<div class="parse-head">[${escapeHtml(spec.source)}] read ${rows.length} field${rows.length === 1 ? "" : "s"} from your text. Anything you wrote that is not listed was not recognised, and anything not written at all is a default.</div>
    <table class="parse-table"><thead><tr><th>Spec field</th><th class="num">You asked</th><th>What this run does</th></tr></thead><tbody>${body}</tbody></table>${assumeNote}${warn}`;
}

function renderModeHint() {
  $("#pipeline-mode-hint").textContent = MODE_INFO[selectedMode]?.desc || "";
}

function selectMode(mode) {
  selectedMode = mode;
  $$(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  renderModeHint();
}

function renderModeRow() {
  const row = $("#pipeline-mode-row");
  const available = pipelineDefaults.modes && pipelineDefaults.modes.length
    ? pipelineDefaults.modes : MODE_ORDER;
  const ordered = MODE_ORDER.filter((m) => available.includes(m));
  row.innerHTML = ordered.map((m) => {
    const info = MODE_INFO[m] || { name: m, sub: "" };
    return `<button type="button" class="mode-btn" data-mode="${escapeHtml(m)}">
      <span class="mode-btn-name">${escapeHtml(info.name)}</span>
      <span class="mode-btn-sub">${escapeHtml(info.sub)}</span>
    </button>`;
  }).join("");
  $$(".mode-btn").forEach((b) => b.addEventListener("click", () => selectMode(b.dataset.mode)));
  selectMode(pipelineDefaults.default_mode && available.includes(pipelineDefaults.default_mode)
    ? pipelineDefaults.default_mode : "default");
}

async function initPipeline() {
  pipelineDefaults = await api("/api/pipeline/defaults");
  renderModeRow();

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
    const note = $("#pipeline-parse-note");
    if (!text) return;
    const btn = $("#pipeline-parse-btn");
    btn.disabled = true;
    note.hidden = true;
    try {
      const spec = await postJSON("/api/pipeline/parse-spec", { text });
      targetInput.value = spec.target_boost_db; updateTarget();
      channelInput.value = spec.channel_loss_db; updateChannel();
      bindRangeFill(targetInput); bindRangeFill(channelInput);
      note.className = "parse-note parse-note-ok";
      note.innerHTML = renderParseTable(spec);
      note.hidden = false;
    } catch (err) {
      // A refused parse leaves the sliders exactly as they were -- deliberately. Showing
      // a default target after an unrecognised request would look like understanding.
      note.className = "parse-note parse-note-bad";
      if (err.title) {
        const label = err.label ? `<span class="err-code">${escapeHtml(err.label)}</span>` : "";
        note.innerHTML = `<strong>${label}${escapeHtml(err.title)}</strong>${err.detail ? `<br>${escapeHtml(err.detail)}` : ""}${err.hint ? `<br><em>${escapeHtml(err.hint)}</em>` : ""}`;
      } else {
        note.textContent = err.message;
      }
      note.hidden = false;
    } finally {
      btn.disabled = false;
    }
  });

  $("#pipeline-run-btn").addEventListener("click", runPipeline);
}

// -- Live pipeline: progress tracker --------------------------------------------------
// The backend runs PPO -> G3.2 -> verification as one blocking call and reports only the
// final result -- there is no per-stage progress feed. So while the request is in flight
// every stage shows the same indeterminate "active" state (no fabricated percentages),
// and only once the result is back are stages marked done/warn/skip from what the result
// actually contains.

function setStage(id, state) {
  $(`#${id}`).className = `stage${state ? " " + state : ""}`;
}

function resetStages() {
  ["stage-search", "stage-refine", "stage-verify"].forEach((id) => setStage(id, ""));
  $("#stage-note").textContent = "";
}

function updateStagesFromResult(r) {
  const statuses = pipelineDefaults.statuses;
  const note = $("#stage-note");

  if (r.status === statuses.fallback) {
    setStage("stage-search", "warn");
    setStage("stage-refine", "warn");
    setStage("stage-verify", r.verification ? "done" : "skip");
    note.textContent = "PPO → G3.2 found nothing guard-valid for this spec; a fixed fallback design was returned instead.";
    return;
  }
  if (!r.design) {
    setStage("stage-search", "done");
    setStage("stage-refine", "warn");
    setStage("stage-verify", "skip");
    note.textContent = "The architecture ran and did not produce a guard-valid design for this spec.";
    return;
  }
  setStage("stage-search", "done");
  setStage("stage-refine", "done");
  if (!r.verification) {
    setStage("stage-verify", "skip");
    note.textContent = "Design found; independent verification did not run.";
  } else if (r.status === statuses.solved) {
    setStage("stage-verify", "done");
    note.textContent = "All three stages completed; independent verification passed.";
  } else {
    setStage("stage-verify", "warn");
    note.textContent = "Design found and re-simulated, but independent verification did not pass.";
  }
}

// -- Live pipeline: result -------------------------------------------------------------

// Server-side scale table (server.py DESIGN_FIELDS): human = SI / scale. The pipeline
// result's `design` object is in SI units (the same DesignVars the simulator takes), but
// POST /api/schematic expects the human units the guard fields use -- so this mirrors
// server.py's own conversion rather than guessing at it.
const DESIGN_FIELD_SCALE = { w_in: 1e-6, l_in: 1e-6, i_tail: 1e-6, rs: 1e3, cs: 1e-15, r_load: 1.0, w_dfe: 1.0 };
function designToHumanFields(dv) {
  const out = {};
  for (const [k, scale] of Object.entries(DESIGN_FIELD_SCALE)) {
    if (dv[k] !== undefined && dv[k] !== null) out[k] = dv[k] / scale;
  }
  return out;
}

async function fetchResultSchematic(design, r) {
  const isFallback = r.status === pipelineDefaults.statuses.fallback;
  const subtitle = r.spec ? `target ${fmt(r.spec.target_boost_db, 1)} dB boost / ${fmt(r.spec.channel_loss_db, 1)} dB channel loss` : "";
  const res = await fetch("/api/schematic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      fields: designToHumanFields(design),
      title: isFallback ? "Fixed fallback design (not AI-generated)" : "Delivered design",
      subtitle,
    }),
  });
  if (!res.ok) {
    // This endpoint returns SVG on success, so it cannot go through api(). Errors from it
    // are still the structured JSON shape, so unpack them the same way rather than
    // showing the reader a bare status code.
    let parsed = null;
    try { parsed = JSON.parse(await res.text()); } catch { /* not JSON */ }
    const err = new Error(parsed && parsed.title ? parsed.title : `server returned ${res.status}`);
    if (parsed && parsed.title) {
      err.title = parsed.title;
      err.detail = parsed.detail || null;
      err.hint = parsed.hint || null;
    }
    throw err;
  }
  return res.text();
}

function statusBannerHtml(r) {
  const statuses = pipelineDefaults.statuses;
  const v = r.verification;
  if (r.status === statuses.fallback) {
    return `<div class="status-banner st-fallback">${ICONS.alertTriangle}<div>
      <div class="status-title">Fixed fallback design — NOT AI-generated</div>
      <div class="status-sub">The architecture found nothing guard-valid for this spec. What follows is a fixed, hand-verified design that ignores your requested target — it is not an output of the PPO/G3.2 search and must not be read as one.</div>
    </div></div>`;
  }
  if (r.status === statuses.solved) {
    // Deliberately does NOT say "G3.2 closed the target". `solved` means the ten hard
    // checks pass on a fresh re-simulation, and one of those checks is boost-within-
    // tolerance -- which is a weaker statement than G3.2's own internal reached_target
    // flag. The two disagree often enough that claiming the strong one here would be
    // contradicted by the "[target NOT reached]" line in the report a few centimetres
    // below it. Say the thing that is actually verified.
    const err = v && typeof v.abs_err_db === "number"
      ? ` The delivered boost is ${fmt(v.abs_err_db, 2)} dB off the number you asked for, inside the ±${fmt(r.spec.boost_tol_db, 2)} dB tolerance.`
      : "";
    return `<div class="status-banner st-solved">${ICONS.checkCircle}<div>
      <div class="status-title">Solved</div>
      <div class="status-sub">An independent re-simulation of the delivered design passes all ten hard specs, boost included.${err}</div>
    </div></div>`;
  }
  if (r.status === statuses.closed_not_verified) {
    const fails = v && v.failing && v.failing.length ? ` — fails ${escapeHtml(v.failing.join(", "))}` : "";
    return `<div class="status-banner st-closed">${ICONS.alertTriangle}<div>
      <div class="status-title">Closed, but failed independent verification</div>
      <div class="status-sub">The solver reported it reached the target, but the fresh re-simulation disagrees${fails}.</div>
    </div></div>`;
  }
  return `<div class="status-banner st-unsolved">${ICONS.xCircle}<div>
    <div class="status-title">Unsolved</div>
    <div class="status-sub">The architecture ran and did not reach a guard-valid design for this spec.</div>
  </div></div>`;
}

function headlineHtml(d, v) {
  // Beside the numbers the drawing renders at about 0.68 of its natural size, which keeps
  // the circuit and its sizing in one screenful but puts the device labels near 8px. The
  // toggle drops the grid to one column so the schematic can be read at full size; it is
  // a layout switch, nothing is re-fetched.
  return `<div class="result-hero" id="result-hero">
    <div class="schematic-holder" id="result-schematic-holder">
      <button class="schematic-zoom" id="schematic-zoom" type="button">Enlarge</button>
      <p class="artifact-meta">rendering schematic&hellip;</p>
    </div>
    <div class="headline-grid">
      ${kpiGrid([
        { label: "W_in", value: fmt(d.w_in * 1e6, 2) + " µm" },
        { label: "L_in", value: fmt(d.l_in * 1e6, 4) + " µm" },
        { label: "I_tail", value: fmt(d.i_tail * 1e6, 1) + " µA" },
        { label: "R_s", value: fmt(d.rs / 1e3, 2) + " kΩ" },
        { label: "C_s", value: fmt(d.cs * 1e15, 1) + " fF" },
        { label: "R_load", value: fmt(d.r_load, 1) + " Ω" },
      ])}
      <div class="headline-power">${kpiGrid([
        { label: "Power", value: v && v.measures ? fmt(v.measures.power_w * 1e3, 2) + " mW" : "—", accent: true },
      ])}</div>
    </div>
  </div>`;
}

function secondaryBlockHtml(r) {
  const v = r.verification;
  if (!v || !v.measures) {
    return `<div class="secondary-block"><p class="help-hint" style="margin:0;">${v ? `verification: GUARD REJECTED — ${escapeHtml(v.guard_check || "")}` : "verification: not run"}</p></div>`;
  }
  const m = v.measures;
  return `<div class="secondary-block">
    <div class="card-title">Verification detail</div>
    ${kpiGrid([
      { label: "Boost achieved", value: fmt(m.boost_db, 3) + " dB", help: `requested ${fmt(r.spec.target_boost_db, 2)} ±${fmt(r.spec.boost_tol_db, 2)} dB` },
      { label: "Peak freq", value: fmt(m.peak_freq_ghz, 3) + " GHz" },
      // eye_h is the HORIZONTAL opening, so it is the eye's width, in UI; eye_v is the
      // VERTICAL opening, so it is its height, in mV. Naming them the other way round
      // puts a time unit on a voltage, which is the first thing an analog engineer looks
      // at here.
      { label: "Eye width", value: fmt(m.eye_h_ui, 2) + " UI" },
      { label: "Eye height", value: Math.round(m.eye_v_mv) + " mV" },
      { label: "Noise", value: Math.round(m.noise_vrms * 1e6) + " µVrms" },
      { label: "HD3", value: fmt(m.hd3_db, 1) + " dB" },
      { label: "Area", value: fmt(m.area_mm2 * 1e3, 1) + " ×10⁻³ mm²" },
    ])}
    <div class="secondary-row">
      <span>Guard verdict: <strong>${v.guard_valid ? "guard-valid" : `GUARD REJECTED — ${escapeHtml(v.guard_check || "")}`}</strong></span>
      <span>Hard-spec check: <strong>${v.passed ? "all ten checks pass" : `fails ${(v.failing || []).join(", ")}`}</strong></span>
    </div>
  </div>`;
}

function modeDetailHtml(r) {
  const detail = r.provenance && r.provenance.mode_detail;
  const mode = (detail && detail.mode) || r.mode || "default";
  if (!detail || mode === "default") return "";

  if (mode === "accurate") {
    return `<div class="secondary-block mode-detail-block">
      <div class="card-title">Accurate mode</div>
      <p class="help-hint" style="margin:0;">Ran with a larger G3.2 refinement budget and a tighter internal stopping tolerance than Default, trading more evaluations for a closer boost match.</p>
    </div>`;
  }

  if (mode === "fastest") {
    const h = detail.hedge || {};
    const verdict = !h.attempted ? "not attempted"
      : !h.evaluated ? "attempted — no candidate cleared the corpus safety radius"
      : h.accepted ? "accepted" : "attempted — not accepted";
    return `<div class="secondary-block mode-detail-block">
      <div class="card-title">Fastest mode — surrogate hedge</div>
      <div class="kv-row"><span class="kv-k">Outcome</span><span class="kv-v">${escapeHtml(verdict)}</span></div>
      ${h.reason ? `<div class="kv-row"><span class="kv-k">Reason</span><span class="kv-v">${escapeHtml(h.reason)}</span></div>` : ""}
    </div>`;
  }

  if (mode === "thinking") {
    const rollouts = detail.rollouts || [];
    const rows = rollouts.map((ro) => {
      const isWinner = ro.seed_offset === detail.winner_seed_offset;
      return `<tr><td class="mono">${ro.seed_offset}${isWinner ? " <strong>(winner)</strong>" : ""}</td><td class="num mono">${fmt(ro.best_abs_err_db, 3)} dB</td><td class="num mono">${ro.n_loose_pass}</td><td>${ro.reached_target ? "reached target" : "did not reach target"}</td></tr>`;
    }).join("");
    return `<div class="secondary-block mode-detail-block">
      <div class="card-title">Thinking mode — ${rollouts.length} independent rollouts</div>
      <table class="parse-table"><thead><tr><th>Seed offset</th><th class="num">Best abs. error</th><th class="num">Loose passes</th><th>Target</th></tr></thead><tbody>${rows}</tbody></table>
    </div>`;
  }

  return "";
}

function costStripHtml(r, elapsedS) {
  const c = r.cost;
  return `<div class="cost-strip">
    <span class="cs-item"><strong>${c.optimizer_evals}</strong> optimizer evals</span>
    <span class="cs-item"><strong>${c.measure_all_total}</strong> measure_all calls (${c.measure_all_search} search + ${c.measure_all_verification} verify)</span>
    <span class="cs-item"><strong>${c.spice_analyses_total}</strong> SPICE analyses</span>
    <span class="cs-item"><strong>${c.budget_evals_unspent}</strong> budget unspent</span>
    <span class="cs-item"><strong>${fmt(elapsedS, 1)}</strong> s wall clock (this request)</span>
  </div>`;
}

async function renderPipelineResult(r, elapsedS) {
  const box = $("#pipeline-result");
  const d = r.design;

  let html = statusBannerHtml(r);
  html += d ? headlineHtml(d, r.verification)
             : banner("warning", "alertTriangle", "No design: the architecture produced nothing guard-valid for this spec, and fallback was not allowed.");
  if (d) html += secondaryBlockHtml(r);
  html += modeDetailHtml(r);
  html += costStripHtml(r, elapsedS);
  html += `<details class="explainer" style="margin-top:4px;"><summary>Full CLI-style report</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(r.describe_text)}</pre></div></details>`;
  if (r.netlist) html += `<details class="explainer" style="margin-top:10px;"><summary>Final schematic (SPICE netlist)</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(r.netlist)}</pre></div></details>`;
  html += `<details class="explainer" style="margin-top:10px;"><summary>Raw result JSON</summary><div class="explainer-body" style="padding-left:16px;"><pre class="code-block">${escapeHtml(JSON.stringify(r, null, 2))}</pre></div></details>`;

  box.innerHTML = html;

  if (d) {
    const holder = $("#result-schematic-holder");
    try {
      const svg = await fetchResultSchematic(d, r);
      // The button lives inside the holder, so it has to be re-attached after the SVG
      // replaces the placeholder.
      holder.innerHTML = `<button class="schematic-zoom" id="schematic-zoom" type="button">Enlarge</button>${svg}`;
      const zoom = $("#schematic-zoom");
      zoom.addEventListener("click", () => {
        const wide = $("#result-hero").classList.toggle("wide");
        zoom.textContent = wide ? "Shrink" : "Enlarge";
        if (wide) holder.scrollIntoView({ block: "start", behavior: "smooth" });
      });
    } catch (err) {
      holder.innerHTML = errorBannerHtml(err, "Could not render the schematic");
    }
  }
}

// -- Live narration ---------------------------------------------------------------------
// The run is a single blocking POST, so progress is read from a second endpoint while it
// is in flight. Every line comes from the server having actually observed the frozen
// pipeline do that thing -- there are no interpolated percentages and no fake steps; when
// nothing is happening the log simply does not move.

const STAGE_OF = { load: "stage-search", search: "stage-search", refine: "stage-refine", verify: "stage-verify" };
const STAGE_LABEL = {
  load: "Loading the frozen policy",
  search: "Stage 1 of 3 · PPO search",
  refine: "Stage 2 of 3 · G3.2 refinement",
  verify: "Stage 3 of 3 · independent verification",
};
const STAGE_ORDER = ["stage-search", "stage-refine", "stage-verify"];

function appendRunEvents(events) {
  const log = $("#run-log");
  if (!log) return;
  for (const ev of events) {
    const row = document.createElement("div");
    row.className = `log-row log-${ev.kind}${ev.ok === false ? " log-rejected" : ""}`;
    row.innerHTML = `<span class="log-t">${fmt(ev.t, 1)}s</span><span class="log-text">${escapeHtml(ev.text)}</span>`;
    log.appendChild(row);
  }
  // Follow the tail: during a run the newest line is the only one worth looking at.
  log.scrollTop = log.scrollHeight;
}

function markStagesLive(stage) {
  const id = STAGE_OF[stage];
  if (!id) return;
  const at = STAGE_ORDER.indexOf(id);
  STAGE_ORDER.forEach((s, i) => setStage(s, i < at ? "done" : i === at ? "active" : ""));
}

function pollRunProgress(state) {
  return setInterval(async () => {
    if (state.stopped) return;
    try {
      const p = await api(`/api/pipeline/progress?since=${state.cursor}`);
      state.cursor = p.next;
      if (p.events.length) appendRunEvents(p.events);
      markStagesLive(p.stage);
      // The note is a running tally, not a copy of the last log line -- the trace right
      // below it already says what just happened.
      state.simulated += p.events.filter((e) => e.kind === "candidate").length;
      const label = STAGE_LABEL[p.stage] || "Running";
      $("#stage-note").textContent =
        `${label} · ${state.simulated} circuit${state.simulated === 1 ? "" : "s"} simulated · ${fmt(p.elapsed_s, 1)} s elapsed`;
    } catch { /* a dropped poll is not a failed run; the next one catches up */ }
  }, 350);
}

async function runPipeline() {
  const btn = $("#pipeline-run-btn");
  const box = $("#pipeline-result");
  btn.disabled = true;
  $$(".mode-btn").forEach((b) => { b.disabled = true; });
  box.innerHTML = "";
  $("#stage-tracker-wrap").hidden = false;
  resetStages();
  setStage("stage-search", "active");
  $("#run-log").innerHTML = "";
  $("#run-log-wrap").hidden = false;
  $("#stage-note").textContent = "Starting — every line below is a real SKY130 simulation.";

  const t0 = performance.now();
  const state = { cursor: 0, stopped: false, simulated: 0 };
  const timer = pollRunProgress(state);
  try {
    const result = await postJSON("/api/pipeline/run", {
      target_boost_db: Number($("#pipeline-target").value),
      channel_loss_db: Number($("#pipeline-channel").value),
      spec_index: Number($("#pipeline-spec-index").value) || 0,
      allow_fallback: $("#pipeline-fallback").checked,
      mode: selectedMode,
    });
    const elapsedS = (performance.now() - t0) / 1000;
    // Render FIRST, drain the log after. The answer is in hand the moment the POST
    // resolves, and the closing log lines are a nicety -- making the schematic wait on
    // one more round trip is the only latency this layer actually controls.
    updateStagesFromResult(result);
    await renderPipelineResult(result, elapsedS);
    try {
      const p = await api(`/api/pipeline/progress?since=${state.cursor}`);
      if (p.events.length) appendRunEvents(p.events);
    } catch { /* the result is already on screen; the log tail is cosmetic */ }
  } catch (err) {
    $("#stage-tracker-wrap").hidden = true;
    box.innerHTML = errorBannerHtml(err, "Run failed");
  } finally {
    state.stopped = true;
    clearInterval(timer);
    btn.disabled = false;
    $$(".mode-btn").forEach((b) => { b.disabled = false; });
  }
}

// -- Design time ------------------------------------------------------------------------
// Everything here is read from the recorded artifacts by /api/design-time. The panel leads
// with the evaluation-count result (which the permutation tests support) and shows the
// chance-matched negative on the strict column inline, rather than burying it.

function statTile(value, unit, label, sub) {
  return `<div class="stat-tile">
    <div class="stat-value">${escapeHtml(value)}<span class="stat-unit">${escapeHtml(unit || "")}</span></div>
    <div class="stat-label">${escapeHtml(label)}</div>
    ${sub ? `<div class="stat-sub">${escapeHtml(sub)}</div>` : ""}
  </div>`;
}

function pFmt(p) {
  if (p === null || p === undefined) return "&mdash;";
  if (p < 1e-3) return p.toExponential(1);
  return p.toFixed(3);
}

function renderDesignTime(d) {
  const P = d.protocol, S = d.sweep, E = d.per_eval, DL = d.delivered;
  let html = "";

  html += `<div class="stat-row">
    ${statTile(String(DL.total_evals), " evals", "to size the delivered circuit", `${DL.stage1_ppo_evals} PPO + ${DL.stage2_evals} refinement`)}
    ${statTile(S.total_points.toLocaleString(), " pts", "exhaustive sweep, actually run", `${S.elapsed_h} h · passed spec on ${S.spec_pass}`)}
    ${statTile(`${E.speedup.toFixed(0)}×`, "", "faster per evaluation", `${E.resident_median_s.toFixed(3)} s vs ${E.subprocess_median_s.toFixed(2)} s`)}
  </div>`;

  // -- The defensible claim: evaluations at a matched budget --------------------------
  html += `<div class="section-label">Evaluations to a solution — ${P.n_specs} held-out specs, ${P.budget}-evaluation budget for every arm</div>`;
  html += `<div class="card"><table class="dt-table">
    <thead><tr>
      <th>Method</th>
      <th class="num">Median evals<br/><span class="th-sub">to loose solve</span></th>
      <th class="num">Solved<br/><span class="th-sub">loose</span></th>
      <th class="num">p vs PPO<br/><span class="th-sub">permutation, loose</span></th>
      <th class="num">Sim-matched<br/><span class="th-sub">evals → loose</span></th>
      <th class="num">Solved<br/><span class="th-sub">strict</span></th>
    </tr></thead><tbody>`;
  let allStrictAtChance = true;
  for (const a of d.arms) {
    const sm = a.sim_matched;
    if (!(a.chance && a.chance.p_one_sided > 0.05)) allStrictAtChance = false;
    html += `<tr class="${a.is_silq ? "dt-silq" : ""}">
      <td><strong>${escapeHtml(a.name)}</strong>${a.is_silq ? ' <span class="dt-tag">SILQ</span>' : ""}</td>
      <td class="num">${a.loose_median}</td>
      <td class="num">${a.loose} / ${a.n_specs}</td>
      <td class="num">${a.vs_ppo_loose_p === null || a.vs_ppo_loose_p === undefined ? "&mdash;" : pFmt(a.vs_ppo_loose_p)}</td>
      <td class="num">${sm ? `${sm.evals} → ${sm.loose}` : "&mdash;"}</td>
      <td class="num dt-dim" title="${a.chance ? `chance-matched control expected ${a.chance.expected.toFixed(1)}, p=${a.chance.p_one_sided}` : ""}">${a.strict} / ${a.n_specs}</td>
    </tr>`;
  }
  html += `</tbody></table>
    <p class="help-hint" style="margin-top:12px;">
      &ldquo;Loose&rdquo; and &ldquo;strict&rdquo; are the two preregistered solve criteria (tolerance ${P.tol_db} dB).
      p-values are ${P.permutations.toLocaleString()}-permutation paired tests against the <strong>PPO</strong> arm,
      which is the reference (so PPO itself shows no p, and PPO-restart is compared to it).
      &ldquo;Sim-matched&rdquo; re-runs each arm at an equalised simulation count.
    </p></div>`;

  const ref = d.arms.find((a) => a.name === "PPO-restart");
  if (ref && ref.chance) {
    html += banner("warning", "alertTriangle",
      `Preregistered negative, reported: the strict solve <em>count</em> is greyed above because it is not distinguishable from a chance-matched control &mdash; SILQ scores ${ref.chance.observed} against an expectation of ${ref.chance.expected.toFixed(1)} (p = ${ref.chance.p_one_sided})${allStrictAtChance ? ", and the same is true of every other arm, so that column separates nothing" : ""}. The claim this panel makes is the <strong>evaluation count</strong>, not the number of specs solved.`);
  }

  // -- Versus the sweep the brief names ------------------------------------------------
  html += `<div class="section-label">Versus sweeping the parameter space</div>`;
  html += `<div class="card">
    <p>A full-factorial grid over the six design variables was <strong>actually run</strong>, not estimated:
    ${S.per_axis} points per axis = ${S.total_points.toLocaleString()} points, ${S.elapsed_h} hours of wall clock
    at ${S.s_per_point} s per point. It was guard-valid on ${S.valid} and passed spec on
    <strong>${S.spec_pass}</strong> of them; the first success came at point ${S.first_success_index.toLocaleString()}.</p>
    <p class="help-hint">${escapeHtml(S.note)}</p>
    <div class="section-label" style="margin-top:16px;">And that is the coarsest grid there is — refining it explodes</div>
    <table class="dt-table"><thead><tr><th>Points per axis</th><th class="num">Grid size</th><th class="num">Projected wall clock</th></tr></thead><tbody>`;
  for (const [k, hours] of Object.entries(S.extrapolation_hours)) {
    const n = Number(k);
    html += `<tr><td>${n}</td><td class="num">${Math.pow(n, 6).toLocaleString()}</td><td class="num">${
      hours < 48 ? `${hours.toFixed(1)} h` : `${(hours / 24).toFixed(0)} days`}</td></tr>`;
  }
  html += `</tbody></table></div>`;

  // -- Per-evaluation cost --------------------------------------------------------------
  html += `<div class="section-label">Cost per evaluation</div>`;
  html += `<div class="card"><p>Design time is evaluations &times; cost per evaluation, so the simulator loop was
    optimised too: a resident libngspice server instead of one subprocess per design.
    <strong>${E.resident_median_s.toFixed(4)} s</strong> vs <strong>${E.subprocess_median_s.toFixed(2)} s</strong>
    &mdash; a ${E.speedup.toFixed(1)}× speedup. At that rate the delivered circuit's
    ${DL.total_evals} evaluations are ${DL.wall_clock_s_at_resident_rate} s of simulation.</p>
    <p class="help-hint">${escapeHtml(E.note)}</p></div>`;

  // -- Caveats ---------------------------------------------------------------------------
  html += `<div class="section-label">What these numbers do not say</div><div class="card"><ul class="dt-caveats">`;
  for (const c of d.caveats) html += `<li>${escapeHtml(c)}</li>`;
  html += `<li>${escapeHtml(DL.unit_warning)}</li></ul></div>`;

  $("#designtime-body").innerHTML = html;
}

async function initDesignTime() {
  const res = await fetch("/api/design-time");
  if (!res.ok) throw new Error(`server returned ${res.status}`);
  renderDesignTime(await res.json());
}

// -- Schematic (guard sandbox) -----------------------------------------------------------
// The SVG is rendered server-side by eqrl.schematic from the same DesignVars the
// simulator receives, so the picture cannot drift from the measured netlist. This powers
// the "Schematic of these fields" strip inside the Guard layer sandbox dev tool.

async function showSchematic(fetcher, label) {
  const holder = $("#schematic-holder");
  holder.innerHTML = `<p class="artifact-meta">rendering ${escapeHtml(label)}&hellip;</p>`;
  try {
    const svg = await fetcher();
    holder.innerHTML = svg;
  } catch (err) {
    holder.innerHTML = banner("danger", "xCircle", `Could not render the schematic: ${escapeHtml(err.message)}`);
  }
}

async function initSchematic() {
  const delivered = async () => {
    const res = await fetch("/api/schematic");
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    return res.text();
  };
  const fromGuard = async () => {
    const res = await fetch("/api/schematic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fields: readGuardFields(),
        title: "CTLE candidate",
        subtitle: "drawn from the current Guard Layer fields — not the frozen delivered design",
      }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    return res.text();
  };

  $("#schematic-delivered-btn").addEventListener("click", () => showSchematic(delivered, "the delivered circuit"));
  $("#schematic-guard-btn").addEventListener("click", () => showSchematic(fromGuard, "the guard-field candidate"));
  await showSchematic(delivered, "the delivered circuit");
}

// -- Boot -----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  pollHealth();
  initPipeline().catch((err) => { $("#pipeline-result").innerHTML = errorBannerHtml(err, "Failed to load pipeline defaults"); });
  initGuard().catch((err) => { $("#guard-result").innerHTML = `<div class="card">${errorBannerHtml(err, "Failed to load guard layer")}</div>`; });
  initSchematic().catch((err) => { $("#schematic-holder").innerHTML = banner("danger", "xCircle", `Failed to load schematic: ${escapeHtml(err.message)}`); });
  initResults().catch((err) => { $("#results-detail").innerHTML = errorBannerHtml(err, "Failed to load results"); });
  initDesignTime().catch((err) => { $("#designtime-body").innerHTML = errorBannerHtml(err, "Failed to load design-time record"); });
});
