// silQ dashboard. Vanilla JS, no build step. Talks to server.py's JSON API, which wraps
// the same eqrl functions the CLI and the benchmarks use.
//
// Shape of the page: an intro splash, then one product surface (Design) whose spine is
// "type to circuit": the composer at the top reads the request as it is typed, the spec
// panel shows what will steer and score the run, and the stage on the right always shows
// a circuit, its sizing and one headline number. During a run the stage redraws with every
// simulated candidate. The Lab view keeps the guard sandbox, results explorer and benchmark
// record.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const ICONS = {
  check: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
  cross: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>',
  dash: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M5 12h14"/></svg>',
  checkCircle: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.3 2.3 5-5.3"/></svg>',
  xCircle: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/></svg>',
  alertTriangle: '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 4l9 15H3z"/><path d="M12 10v4"/><circle cx="12" cy="17" r="0.4" fill="currentColor"/></svg>',
  info: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><circle cx="12" cy="8.3" r="0.4" fill="currentColor"/></svg>',
  spark: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/></svg>',
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
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmt(v, decimals = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "n/a";
  return Number(v).toFixed(decimals);
}

const urlFlags = new URLSearchParams(location.search);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches || urlFlags.has("nomotion");
const hasGsap = typeof gsap !== "undefined" && !reducedMotion;

// Entrance tweens start from opacity 0. If the animation frame loop stalls (background
// tab, throttled renderer, a GSAP failure), nothing readable may ever appear, so every
// entrance also arms a timer that strips the inline styles regardless.
function animateIn(targets, vars, delayMs = 0) {
  if (!hasGsap) return;
  const nodes = typeof targets === "string" ? $$(targets) : targets;
  if (!nodes || !nodes.length) return;
  gsap.from(nodes, { ...vars, clearProps: "all" });
  setTimeout(() => gsap.set(nodes, { clearProps: "all" }), delayMs + 2500);
}

// Fetch wrapper that never swallows a failure into a blank panel. Understands the
// structured error shape {"error_code","title","detail","hint"} and plain FastAPI
// {"detail": ...}; either way the thrown Error carries enough to render something honest.
async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (netErr) {
    const err = new Error("Cannot reach the silQ server");
    err.title = "Cannot reach the silQ server";
    err.detail = `The browser could not open a connection to ${location.host} (${netErr.message}). The server process is most likely stopped.`;
    err.hint = "Restart it from the repo root with: python -m uvicorn server:app --port " + (location.port || "8000") + "   then reload this page.";
    err.errorCode = "server_unreachable";
    throw err;
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let parsed = null;
    try { parsed = JSON.parse(text); } catch { /* not JSON */ }
    if (parsed && typeof parsed === "object" && (parsed.title || parsed.error_code)) {
      const err = new Error(parsed.title || parsed.detail || `${res.status} ${res.statusText}`);
      Object.assign(err, { title: parsed.title || null, detail: parsed.detail || null, hint: parsed.hint || null,
        errorCode: parsed.error_code || null, label: parsed.label || null, status: res.status, extra: parsed });
      throw err;
    }
    if (parsed && Array.isArray(parsed.detail)) {
      const lines = parsed.detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg || "invalid"}`).join("; ");
      const err = new Error(lines);
      Object.assign(err, { title: "The server rejected these values", detail: lines,
        hint: "This is a malformed request rather than a design failure. Reload the page and try again.", status: res.status });
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
  if (!wrap) { wrap = el("div", { class: "toast-container" }); document.body.appendChild(wrap); }
  const t = el("div", { class: `toast ${ok ? "ok" : "fail"}` }, `${ICONS[ok ? "check" : "cross"]}<span>${escapeHtml(message)}</span>`);
  wrap.appendChild(t);
  setTimeout(() => { t.style.transition = "opacity 0.25s"; t.style.opacity = "0"; setTimeout(() => t.remove(), 260); }, 3800);
}

function bindRangeFill(input) {
  const update = () => {
    const min = Number(input.min) || 0, max = Number(input.max) || 100, val = Number(input.value);
    input.style.setProperty("--pct", `${max > min ? ((val - min) / (max - min)) * 100 : 0}%`);
  };
  input.addEventListener("input", update);
  update();
  return update;
}

function downloadText(name, text, type = "text/plain") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// -- Intro ----------------------------------------------------------------------------

let heroPlayed = false;
function playHeroEntrance() {
  if (heroPlayed) return;
  heroPlayed = true;
  const v = $("#hero-video");
  if (v && !reducedMotion) v.play().catch(() => {});
  animateIn(".hero-inner > *", { y: 18, opacity: 0, stagger: 0.07, duration: 0.75, ease: "power3.out" });
  animateIn(".spec-panel .panel, .stage-card", { y: 14, opacity: 0, stagger: 0.06, duration: 0.6, delay: 0.25, ease: "power2.out" }, 250);
}

function runIntro(force = false) {
  const intro = $("#intro"), video = $("#intro-video"), lockup = $("#intro-lockup"), bar = $("#intro-bar");
  let seen = false;
  try { seen = sessionStorage.getItem("silq_intro") === "1"; } catch { /* storage blocked */ }
  if ((seen && !force) || (reducedMotion && !force) || (urlFlags.has("nointro") && !force)) { intro.hidden = true; playHeroEntrance(); return; }

  intro.hidden = false;
  intro.classList.remove("leaving", "masked");
  lockup.style.opacity = "0";
  bar.style.transition = "none"; bar.style.width = "0%";
  let finished = false;
  const timers = [];
  const finish = () => {
    if (finished) return;
    finished = true;
    timers.forEach(clearTimeout);
    try { sessionStorage.setItem("silq_intro", "1"); } catch { /* storage blocked */ }
    intro.classList.add("leaving");
    setTimeout(() => { intro.hidden = true; video.pause(); intro.classList.remove("masked"); }, 720);
    playHeroEntrance();
  };
  // Timeline: the clip runs clean for ~2.3 s (signal traces, the curve forming), then the
  // mask rises and the HTML lockup carries "silQ, built for Astera Labs" to the end.
  const DURATION = 5200, MASK_AT = 2300, LOCKUP_AT = 2600;
  requestAnimationFrame(() => { bar.style.transition = `width ${DURATION}ms linear`; bar.style.width = "100%"; });
  timers.push(setTimeout(() => intro.classList.add("masked"), MASK_AT));
  if (hasGsap) {
    gsap.fromTo(lockup, { opacity: 0, y: 16, scale: 0.97 }, { opacity: 1, y: 0, scale: 1, duration: 1.0, delay: LOCKUP_AT / 1000, ease: "power3.out" });
  }
  timers.push(setTimeout(() => { lockup.style.transition = "opacity 0.9s ease"; lockup.style.opacity = "1"; }, LOCKUP_AT + (hasGsap ? 1200 : 0)));
  video.currentTime = 0;
  video.play().catch(() => { /* autoplay blocked: the lockup still carries the intro */ });
  video.onended = finish;
  setTimeout(finish, DURATION + 300);
  $("#intro-skip").onclick = finish;
}

// -- Theme, views, health -------------------------------------------------------------

function initTheme() {
  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const light = root.getAttribute("data-theme") === "light";
    if (light) root.removeAttribute("data-theme"); else root.setAttribute("data-theme", "light");
    try { localStorage.setItem("silq_theme", light ? "dark" : "light"); } catch { /* storage blocked */ }
  });
}

function showView(name) {
  $$(".view").forEach((v) => { v.hidden = v.id !== `view-${name}`; });
  $$(".nav-tab").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
}

function initViews() {
  $$(".nav-tab").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  $("#brand-link").addEventListener("click", (e) => { e.preventDefault(); showView("design"); });
  $("#replay-intro").addEventListener("click", () => { heroPlayed = true; runIntro(true); });
  if (urlFlags.get("view") === "lab") showView("lab");
}

function setHealthPill(state, text) {
  $("#health-dot").className = `status-dot ${state}`;
  $("#health-label").textContent = text;
}
let serverWarming = true;
function setRunButtonWarming(warming) {
  serverWarming = warming;
  const btn = $("#run-btn");
  btn.disabled = warming || running;
  btn.title = warming ? "Waiting for the server to finish loading device models" : "";
  const galleryBtn = $("#gallery-run");
  if (galleryBtn) galleryBtn.disabled = warming || galleryRunning;
}
async function checkHealthOnce() {
  try {
    const res = await fetch("/api/health");
    if (res.status === 404) throw new Error("no-health-endpoint");
    if (!res.ok) { setHealthPill("error", `backend error (${res.status})`); return true; }
    const data = await res.json();
    if (data.error) { setHealthPill("error", data.detail || data.error); setRunButtonWarming(false); return true; }
    if (data.ready === true) { setHealthPill("ready", "Simulator ready"); $("#health-pill").title = data.detail || ""; setRunButtonWarming(false); return true; }
    setHealthPill("warming", data.detail || "Loading device models");
    setRunButtonWarming(true);
    return false;
  } catch {
    try { await api("/api/pipeline/defaults"); setHealthPill("ready", "Ready"); }
    catch { setHealthPill("error", "backend unreachable"); }
    setRunButtonWarming(false);
    return true;
  }
}
function pollHealth() { checkHealthOnce().then((done) => { if (!done) setTimeout(pollHealth, 2000); }); }

// -- Spec state -------------------------------------------------------------------------

let defaults = null;
let selectedMode = "auto";
let updateTarget, updateChannel;
const reqInputs = {};     // field -> input element

const PARAM_UNITS = {
  w_in: { scale: 1e6, unit: "µm", dp: 2 }, l_in: { scale: 1e6, unit: "µm", dp: 3 },
  i_tail: { scale: 1e6, unit: "µA", dp: 1 }, rs: { scale: 1e-3, unit: "kΩ", dp: 2 },
  cs: { scale: 1e15, unit: "fF", dp: 1 }, r_load: { scale: 1, unit: "Ω", dp: 0 },
};
// server.py DESIGN_FIELDS: human = SI / scale. POST /api/schematic takes human units.
const DESIGN_FIELD_SCALE = { w_in: 1e-6, l_in: 1e-6, i_tail: 1e-6, rs: 1e3, cs: 1e-15, r_load: 1.0, w_dfe: 1.0 };
function designToHumanFields(dv) {
  const out = {};
  for (const [k, scale] of Object.entries(DESIGN_FIELD_SCALE)) if (dv[k] !== undefined && dv[k] !== null) out[k] = dv[k] / scale;
  return out;
}

// "15", "-30", "0.4", "1.25": the shortest exact decimal form of a display value.
function compactNum(v) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "";
  return String(Math.round(Number(v) * 1e6) / 1e6);
}

function prettyDisp(v, unit) {
  if (v === null || v === undefined) return "n/a";
  const mag = Math.abs(v);
  const dp = Number.isInteger(v) ? 0 : mag >= 100 ? 0 : mag >= 1 ? 2 : 3;
  return `${fmt(v, dp)} ${unit}`;
}

function renderRequirements() {
  const list = $("#req-list");
  list.innerHTML = "";
  for (const r of defaults.requirements) {
    const row = el("div", { class: "req-row" });
    row.innerHTML = `<div><div class="req-name">${escapeHtml(r.label)}</div><div class="req-default">${r.kind === "max" ? "at most" : "at least"} ${escapeHtml(prettyDisp(r.default_disp, r.unit))}</div></div>
      <input type="number" step="any" data-field="${r.field}" placeholder="${escapeHtml(compactNum(r.default_disp))}" aria-label="${escapeHtml(r.label)}" />
      <div><div class="req-unit">${escapeHtml(r.unit)}</div><div class="req-tag" data-tag="${r.field}"></div></div>`;
    const input = row.querySelector("input");
    reqInputs[r.field] = input;
    input.addEventListener("input", () => refreshRequirementTags());
    list.appendChild(row);
  }
  $("#req-reset").addEventListener("click", () => { Object.values(reqInputs).forEach((i) => { i.value = ""; }); refreshRequirementTags(); });
}

function requirementDirection(r, valueDisp) {
  const lowerIsTighter = ["power_w_max", "noise_vrms_max", "hd3_db_max", "area_mm2_max", "boost_db_max", "peak_freq_hi_ghz"].includes(r.field);
  const lower = valueDisp < r.default_disp;
  return lower === lowerIsTighter ? "tighter" : "looser";
}

function refreshRequirementTags() {
  let changed = 0;
  for (const r of defaults.requirements) {
    const input = reqInputs[r.field];
    const tag = $(`[data-tag="${r.field}"]`);
    const v = input.value.trim() === "" ? null : Number(input.value);
    const differs = v !== null && Number.isFinite(v) && Math.abs(v - r.default_disp) > 1e-9;
    input.classList.toggle("set", differs);
    if (differs) { changed += 1; const d = requirementDirection(r, v); tag.textContent = d; tag.className = `req-tag ${d}`; }
    else { tag.textContent = ""; tag.className = "req-tag"; }
  }
  $("#req-summary").textContent = changed ? `${changed} limit${changed === 1 ? "" : "s"} changed from the competition spec. The result is scored against both.` : "All at the competition defaults.";
}

function readRequirements() {
  const out = {};
  for (const r of defaults.requirements) {
    const raw = reqInputs[r.field].value.trim();
    if (raw === "") continue;
    const v = Number(raw);
    if (!Number.isFinite(v) || Math.abs(v - r.default_disp) < 1e-9) continue;
    out[r.field] = v / r.scale;
  }
  return Object.keys(out).length ? out : null;
}

function setRequirementSI(field, si) {
  const r = defaults.requirements.find((q) => q.field === field);
  if (!r || !reqInputs[field]) return false;
  const disp = si * r.scale;
  reqInputs[field].value = String(Math.round(disp * 1e6) / 1e6);
  return true;
}

function selectMode(mode) {
  selectedMode = mode;
  $$(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  const copy = (defaults.mode_copy || {})[mode] || { tagline: mode, detail: "" };
  $("#mode-tag").textContent = copy.tagline || "";
  $("#mode-detail").textContent = copy.detail || "";
}

function renderModes() {
  const seg = $("#mode-seg");
  const modes = defaults.modes && defaults.modes.length ? defaults.modes : ["auto", "fastest", "thinking"];
  seg.innerHTML = modes.map((m) => `<button type="button" class="mode-btn" data-mode="${escapeHtml(m)}" role="radio">${escapeHtml(((defaults.mode_copy || {})[m] || {}).label || m)}</button>`).join("");
  $$(".mode-btn").forEach((b) => b.addEventListener("click", () => selectMode(b.dataset.mode)));
  selectMode(modes.includes(defaults.default_mode) ? defaults.default_mode : modes[0]);
}

async function initSpec() {
  defaults = await api("/api/pipeline/defaults");
  const target = $("#target"), channel = $("#channel");
  target.min = defaults.boost_db_min; target.max = defaults.boost_db_max; target.value = defaults.target_boost_db;
  channel.value = defaults.channel_loss_db;
  const fillT = bindRangeFill(target), fillC = bindRangeFill(channel);
  updateTarget = () => { $("#target-val").textContent = fmt(target.value, 1); fillT(); };
  updateChannel = () => { $("#channel-val").textContent = fmt(channel.value, 1); fillC(); };
  target.addEventListener("input", updateTarget); channel.addEventListener("input", updateChannel);
  updateTarget(); updateChannel();
  renderRequirements();
  refreshRequirementTags();
  renderModes();
}

// -- Composer: type to circuit -----------------------------------------------------------
// Two readers. The keyword reader runs on every keystroke (backend "off": instant, local).
// The Claude reader runs on request or right before a run (backend "auto"), and the merge
// of the two is shown per field: which reader found it, where they disagree, what was
// assumed. Nothing is applied silently: a field lands in the spec panel only when it is
// shown as a chip.

let parseSeq = 0;
let lastParse = null;          // last successful parse (any backend)
let lastParseText = "";
let llmReadText = null;        // text that has already been read by Claude
let readingWithClaude = false;

const ROLE_HINT = { steers: "steers the search", scores: "scored at verification", fixed: "fixed by the environment" };

function setReaderBadge(state, label) {
  const b = $("#reader-badge");
  b.className = `reader-badge ${state}`;
  $("#reader-label").textContent = label;
}

function applyParse(spec) {
  const target = $("#target"), channel = $("#channel");
  for (const row of spec.fields || []) {
    if (row.field === "target_boost_db") { target.value = Math.min(Math.max(row.asked, Number(target.min)), Number(target.max)); updateTarget(); }
    else if (row.field === "channel_loss_db") { channel.value = Math.min(Math.max(row.asked, Number(channel.min)), Number(channel.max)); updateChannel(); }
    else if (row.role === "scores") setRequirementSI(row.field, row.asked);
  }
  refreshRequirementTags();
}

function renderChips(spec) {
  const strip = $("#parse-strip");
  const rows = spec.fields || [];
  const chips = rows.map((r) => {
    const cls = ["chip", r.role, r.conflict ? "conflict" : ""].join(" ");
    const dir = r.direction ? `<span class="dir ${r.direction}">${r.direction}</span>` : "";
    const src = r.source === "both" ? "both" : r.source === "llm" ? "llm" : "kw";
    const srcLabel = r.source === "both" ? "both readers" : r.source === "llm" ? "Claude" : "keyword";
    const llmAlt = r.conflict_llm && typeof r.conflict_llm.llm === "number"
      ? ` <button type="button" class="chip-btn" data-use-llm="${r.field}" data-val="${r.conflict_llm.llm}" title="The keyword reader and Claude disagree. Use Claude's ${escapeHtml(prettyDisp(r.conflict_llm.llm * (r.asked_disp / r.asked || 1), r.unit))} instead.">Claude read ${escapeHtml(prettyDisp(r.conflict_llm.llm * (r.asked_disp / r.asked || 1), r.unit))}</button>`
      : "";
    return `<span class="${cls}" title="${escapeHtml(ROLE_HINT[r.role] || "")}"><span class="chip-k">${escapeHtml(r.label)}</span><span class="chip-v">${escapeHtml(prettyDisp(r.asked_disp, r.unit))}</span>${dir}<span class="src ${src}">${srcLabel}</span>${llmAlt}</span>`;
  });
  const notes = [];
  for (const r of rows) {
    if (r.assumption) notes.push(`<div class="note warn">${ICONS.info}<div><strong>${escapeHtml(r.label)}</strong> was read from an ambiguous word: ${escapeHtml(r.assumption)}</div></div>`);
    if (r.conflict && r.conflict_note) notes.push(`<div class="note info">${ICONS.info}<div>${escapeHtml(r.conflict_note)}</div></div>`);
    if (r.conflict_llm) notes.push(`<div class="note warn">${ICONS.info}<div><strong>${escapeHtml(r.label)}</strong>: the keyword reader and Claude disagree (${escapeHtml(prettyDisp(r.conflict_llm.heuristic * (r.asked_disp / r.asked || 1), r.unit))} vs ${escapeHtml(prettyDisp(r.conflict_llm.llm * (r.asked_disp / r.asked || 1), r.unit))}). The keyword value is applied; click the chip to use Claude's.</div></div>`);
  }
  for (const w of spec.warnings || []) notes.push(`<div class="note warn">${ICONS.info}<div>${escapeHtml(w)}</div></div>`);
  const reader = spec.llm_backend ? `Claude (${spec.llm_backend}) and the keyword reader` : "keyword reader";
  const head = `<div class="parse-empty">${reader}: ${rows.length} field${rows.length === 1 ? "" : "s"} recognised. Anything you wrote that is not listed was not understood; anything not listed at all stays at its default.</div>`;
  strip.innerHTML = `<div class="chips">${chips.join("")}</div>${notes.length ? `<div class="parse-notes">${notes.join("")}</div>` : ""}${head}`;
  $$("[data-use-llm]", strip).forEach((b) => b.addEventListener("click", () => {
    const field = b.dataset.useLlm, si = Number(b.dataset.val);
    if (field === "target_boost_db") { $("#target").value = si; updateTarget(); }
    else if (field === "channel_loss_db") { $("#channel").value = si; updateChannel(); }
    else setRequirementSI(field, si);
    refreshRequirementTags();
    b.textContent = "using Claude's reading";
    b.disabled = true;
    toast(`Applied Claude's reading for ${field}.`, true);
  }));
}

function renderParseRefusal(err) {
  const strip = $("#parse-strip");
  const title = err.title || "Not understood";
  strip.innerHTML = `<div class="note bad">${ICONS.xCircle}<div><strong>${escapeHtml(title)}</strong>${err.detail ? ` ${escapeHtml(err.detail)}` : ""}${err.hint ? `<br><em>${escapeHtml(err.hint)}</em>` : ""}</div></div>`;
}

async function parseNow(text, backend) {
  const seq = ++parseSeq;
  if (!text) {
    lastParse = null; lastParseText = "";
    $("#parse-strip").innerHTML = `<div class="parse-empty">Start typing. Every quantity the readers recognise appears here, with which reader found it.</div>`;
    return null;
  }
  try {
    const spec = await postJSON("/api/pipeline/parse-spec", { text, backend });
    if (seq !== parseSeq && backend === "off") return null;   // a newer keystroke won
    lastParse = spec; lastParseText = text;
    if (backend !== "off") llmReadText = text;
    renderChips(spec);
    applyParse(spec);
    if (spec.llm_backend) setReaderBadge("llm", `Claude read it in ${fmt((spec.llm_ms || 0) / 1000, 1)} s`);
    else if (backend !== "off") setReaderBadge("", "Claude unavailable, keyword reader only");
    return spec;
  } catch (err) {
    if (seq !== parseSeq && backend === "off") return null;
    lastParse = null; lastParseText = text;
    if (backend !== "off") llmReadText = text;
    if (err.status === 422) renderParseRefusal(err);
    else $("#parse-strip").innerHTML = errorBannerHtml(err, "Reader failed");
    if (backend !== "off") setReaderBadge("", err.status === 422 ? "not a spec" : "reader error");
    return null;
  }
}

let typeTimer = null;
function initComposer() {
  const ta = $("#spec-text");
  ta.addEventListener("input", () => {
    clearTimeout(typeTimer);
    if (llmReadText !== ta.value.trim()) setReaderBadge("", "keyword reader");
    typeTimer = setTimeout(() => parseNow(ta.value.trim(), "off"), 220);
  });
  ta.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); runDesign(); }
  });
  $$(".example-chip").forEach((b) => b.addEventListener("click", () => {
    ta.value = b.dataset.text; ta.focus();
    parseNow(ta.value.trim(), "off");
    setReaderBadge("", "keyword reader");
  }));
  $("#read-btn").addEventListener("click", () => readWithClaude());
  $("#run-btn").addEventListener("click", () => runDesign());
}

async function readWithClaude() {
  const text = $("#spec-text").value.trim();
  if (!text) { toast("Type a request first.", false); return null; }
  if (readingWithClaude) return null;
  readingWithClaude = true;
  const btn = $("#read-btn");
  btn.disabled = true;
  setReaderBadge("busy", "Claude is reading the request");
  try { return await parseNow(text, "auto"); }
  finally { readingWithClaude = false; btn.disabled = false; }
}

// -- The stage --------------------------------------------------------------------------

let currentSvg = null;
let currentResult = null;
const schematicQueue = { pending: null, inflight: false, last: 0 };

function setStageTitle(title, sub) {
  $("#stage-title").textContent = title;
  $("#stage-sub").textContent = sub || "";
}
function setStateChip(cls, label) {
  $("#state-chip").className = `state-chip ${cls || ""}`;
  $("#state-label").textContent = label;
}
function setParams(dv) {
  for (const [k, u] of Object.entries(PARAM_UNITS)) {
    const node = $(`[data-p="${k}"]`);
    if (!node) continue;
    if (!dv || dv[k] === undefined || dv[k] === null) { node.textContent = "n/a"; continue; }
    const txt = `${fmt(dv[k] * u.scale, u.dp)}<span class="u">${u.unit}</span>`;
    if (node.innerHTML !== txt) {
      node.innerHTML = txt;
      node.classList.add("bump");
      setTimeout(() => node.classList.remove("bump"), 350);
    }
  }
}
function setReadout(label, valueHtml, sub) {
  $("#readout-k").textContent = label;
  $("#readout-v").innerHTML = valueHtml;
  $("#readout-sub").textContent = sub || "";
}

function placeSvg(svg) {
  const holder = $("#schematic-holder");
  holder.innerHTML = svg;
  const node = holder.querySelector("svg");
  if (node) node.classList.add("swap-in");
  currentSvg = svg;
  $("#export-svg").disabled = !svg;
}

async function fetchSchematic(fieldsHuman, title, subtitle) {
  const res = await fetch("/api/schematic", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fields: fieldsHuman, title, subtitle }),
  });
  if (!res.ok) {
    let parsed = null;
    try { parsed = JSON.parse(await res.text()); } catch { /* not JSON */ }
    const err = new Error(parsed && parsed.title ? parsed.title : `server returned ${res.status}`);
    if (parsed && parsed.title) Object.assign(err, { title: parsed.title, detail: parsed.detail || null, hint: parsed.hint || null });
    throw err;
  }
  return res.text();
}

// Coalesced: during a run candidates arrive faster than the drawing is worth refreshing,
// so only the newest pending one is fetched, at most every 450 ms.
function scheduleSchematic(fieldsHuman, title, subtitle) {
  schematicQueue.pending = { fieldsHuman, title, subtitle };
  pumpSchematic();
}
async function pumpSchematic() {
  if (schematicQueue.inflight || !schematicQueue.pending) return;
  const wait = Math.max(0, 450 - (performance.now() - schematicQueue.last));
  if (wait > 0) { setTimeout(pumpSchematic, wait); schematicQueue.inflight = true; setTimeout(() => { schematicQueue.inflight = false; pumpSchematic(); }, wait); return; }
  const job = schematicQueue.pending;
  schematicQueue.pending = null;
  schematicQueue.inflight = true;
  schematicQueue.last = performance.now();
  try { placeSvg(await fetchSchematic(job.fieldsHuman, job.title, job.subtitle)); }
  catch (err) { $("#schematic-holder").innerHTML = errorBannerHtml(err, "Could not render the schematic"); }
  finally { schematicQueue.inflight = false; if (schematicQueue.pending) pumpSchematic(); }
}

async function showDeliveredOnStage() {
  setStageTitle("Delivered CTLE", "the frozen reference design, drawn from its manifest");
  setStateChip("", "reference");
  try {
    const d = await api("/api/results/delivered_circuit.json");
    setParams(d.design);
    const tt = d.pvt && d.pvt.tt_nominal;
    if (tt) setReadout("Boost at tt", `${fmt(tt.boost_db, 2)}<span class="u">dB</span>`, `target ${fmt(d.spec && d.spec.target_boost_db, 2)} dB, ${fmt(tt.power_w * 1e3, 2)} mW, ${d.pvt.corners_passed}/${d.pvt.corners_total} PVT corners`);
  } catch { /* the drawing still loads below */ }
  try {
    const res = await fetch("/api/schematic");
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    placeSvg(await res.text());
  } catch (err) {
    $("#schematic-holder").innerHTML = errorBannerHtml(err, "Could not load the delivered schematic");
  }
}

function showFieldsOnStage(fieldsHuman, title, subtitle) {
  showView("design");
  const si = {};
  for (const [k, scale] of Object.entries(DESIGN_FIELD_SCALE)) if (fieldsHuman[k] !== undefined) si[k] = fieldsHuman[k] * scale;
  setParams(si);
  setStageTitle(title, subtitle);
  setStateChip("", "candidate");
  setReadout("Boost", "n/a", "not simulated yet; use Evaluate in the Lab");
  scheduleSchematic(fieldsHuman, title, subtitle);
}

// -- Candidate gallery --------------------------------------------------------------------
// The existing stage remains the one live, coalesced schematic. This separate gallery is a
// bounded comparison surface: each card gets its own static SVG and only the eye matrix that
// the explicitly requested guarded gallery route returns.

let galleryCatalog = [];
let galleryRunning = false;
const galleryRefresh = { pending: null, inflight: false, retryTimer: null };

function galleryMetric(value, unit) {
  return typeof value === "number" ? `${fmt(value, 2)}${unit ? ` ${unit}` : ""}` : "unavailable";
}
function galleryCardHtml(candidate) {
  const c = candidate.binding_constraint || {};
  return `<article class="gallery-card" data-gallery-id="${escapeHtml(candidate.id)}">
    <div class="gallery-card-head"><div><div class="gallery-card-title">${escapeHtml(candidate.label)}</div><div class="gallery-card-kind">${escapeHtml(candidate.kind || "candidate")}</div></div><span class="state-chip"><span class="dot"></span><span class="gallery-guard">awaiting guard</span></span></div>
    <div class="gallery-schematic"><div class="placeholder"><span class="spinner"></span>drawing schematic</div></div>
    <div class="gallery-eye"><div class="eye-empty">Run a full guarded measurement to load the eye trace.</div><div class="gallery-eye-meta"></div></div>
    <div class="gallery-metrics"><div class="gallery-metric"><span class="gallery-metric-k">Boost</span><span class="gallery-metric-v gallery-boost">unavailable</span></div><div class="gallery-metric"><span class="gallery-metric-k">Target error</span><span class="gallery-metric-v gallery-error">unavailable</span></div></div>
    <div class="gallery-details"><div><span class="gallery-detail-k">Binding constraint</span><div class="gallery-detail-v"><strong>${escapeHtml(c.label || "not recorded")}</strong><br/>${escapeHtml(c.detail || "")}</div></div><div><span class="gallery-detail-k">Scope</span><div class="gallery-detail-v gallery-scope">${escapeHtml(candidate.historical_context || "")}</div></div></div>
  </article>`;
}
function galleryCard(id) { return $(`[data-gallery-id="${CSS.escape(id)}"]`); }
function renderGalleryCards(candidates) {
  const grid = $("#gallery-grid");
  grid.innerHTML = candidates.map(galleryCardHtml).join("");
  candidates.forEach(async (candidate) => {
    const holder = $(".gallery-schematic", galleryCard(candidate.id));
    try { holder.innerHTML = await fetchSchematic(candidate.fields, candidate.label, "candidate gallery"); }
    catch (err) { holder.innerHTML = errorBannerHtml(err, "Could not render gallery schematic"); }
  });
}
function applyGalleryMeasurements(records, scope) {
  if (records.some((record) => !galleryCard(record.id))) renderGalleryCards(records);
  for (const record of records) {
    const card = galleryCard(record.id);
    if (!card) continue;
    const guard = record.guard || {};
    card.classList.toggle("guard-valid", guard.valid === true);
    card.classList.toggle("guard-invalid", guard.valid === false);
    const guardLabel = guard.valid === true ? "guard valid" : guard.valid === false ? `rejected, Tier ${guard.tier}` : "awaiting guard";
    $(".gallery-guard", card).textContent = guardLabel;
    $(".gallery-boost", card).textContent = galleryMetric(record.boost_db, "dB");
    $(".gallery-error", card).textContent = galleryMetric(record.target_error_db, "dB");
    const eyeMount = $(".gallery-eye", card);
    renderEyeChart(eyeMount, record.eye, { ariaLabel: `eye diagram for ${record.label}` });
    const eyeMeta = el("div", { class: "gallery-eye-meta" });
    const eye = record.eye || {};
    const signed = typeof eye.signed_opening_v === "number" ? `signed opening ${fmt(eye.signed_opening_v * 1e3, 1)} mV` : "signed opening unavailable";
    const ber = typeof eye.ber === "number" ? `behavioral BER ${eye.ber.toExponential(2)} (${eye.errors}/${eye.count})` : "behavioral BER unavailable";
    eyeMeta.textContent = `${signed}; ${ber}. ${eye.label || ""}`;
    eyeMount.appendChild(eyeMeta);
    const detail = $(".gallery-scope", card);
    detail.textContent = `${guard.reason || record.historical_context || ""} ${record.metric_scope || ""} ${eye.scope || ""}`.trim();
  }
  if (scope) $("#gallery-scope").textContent = scope;
}
async function runCatalogGallery() {
  if (galleryRunning || serverWarming || !galleryCatalog.length) return;
  galleryRunning = true;
  setRunButtonWarming(serverWarming);
  $("#gallery-notice").className = "gallery-notice busy";
  $("#gallery-notice").textContent = "Running full guarded measurements for the selected comparison set.";
  try {
    const result = await postJSON("/api/candidate-gallery/evaluate", { candidate_ids: galleryCatalog.map((candidate) => candidate.id) });
    applyGalleryMeasurements(result.candidates || [], result.measurement_scope);
    $("#gallery-notice").className = "gallery-notice";
    $("#gallery-notice").textContent = "Gallery measurement complete.";
  } catch (err) {
    $("#gallery-notice").className = "gallery-notice error";
    $("#gallery-notice").textContent = err.detail || err.message || "Could not measure the gallery.";
  } finally {
    galleryRunning = false;
    setRunButtonWarming(serverWarming);
    pumpLiveGallery();
  }
}
function liveGalleryCandidate(design, state, label) {
  return { id: `live-${state.simulated}-${state.galleryCandidates.length}`, label, design,
    target_boost_db: state.target, channel_loss_db: state.channel };
}
function queueLiveGallery(candidates) {
  // A run owns the shared simulator lock. Keep only its newest bounded snapshot and perform
  // one POST after the run releases the lock, instead of polling into repeated 409 responses.
  galleryRefresh.pending = candidates.slice(-6);
  pumpLiveGallery();
}
async function pumpLiveGallery() {
  if (galleryRefresh.inflight || running || galleryRunning || !galleryRefresh.pending || serverWarming) return;
  const candidates = galleryRefresh.pending;
  galleryRefresh.pending = null;
  galleryRefresh.inflight = true;
  $("#gallery-notice").className = "gallery-notice busy";
  $("#gallery-notice").textContent = "Refreshing the latest pipeline candidates after the search lock released.";
  try {
    const result = await postJSON("/api/candidate-gallery/evaluate", { live_candidates: candidates });
    applyGalleryMeasurements(result.candidates || [], result.measurement_scope);
    $("#gallery-notice").className = "gallery-notice";
    $("#gallery-notice").textContent = "Latest pipeline candidate gallery measured.";
  } catch (err) {
    if (err.status === 409) {
      galleryRefresh.pending = candidates;
      clearTimeout(galleryRefresh.retryTimer);
      galleryRefresh.retryTimer = setTimeout(pumpLiveGallery, 700);
    } else {
      $("#gallery-notice").className = "gallery-notice error";
      $("#gallery-notice").textContent = err.detail || err.message || "Could not refresh the live candidate gallery.";
    }
  } finally {
    galleryRefresh.inflight = false;
    if (galleryRefresh.pending && !running) pumpLiveGallery();
  }
}
async function initCandidateGallery() {
  const data = await api("/api/candidate-gallery");
  galleryCatalog = (data.candidates || []).slice(0, Math.min(data.max_candidates || 6, 6));
  renderGalleryCards(galleryCatalog);
  $("#gallery-scope").textContent = data.measurement_scope || "";
  $("#gallery-run").addEventListener("click", runCatalogGallery);
  setRunButtonWarming(serverWarming);
  // `?gallery=measure` is a headless-verification flag in the same family as `nointro`
  // and `nomotion`: the measured state of these cards is the one thing no unit test can
  // reach, because it needs a real browser to run the fetch and draw the eye canvases.
  // It is explicit opt-in and it waits for warm-up rather than racing it -- an early POST
  // would take a 409 from the warm-up's own hold on the simulator lock and prove nothing.
  if (urlFlags.get("gallery") === "measure") {
    const start = Date.now();
    const armed = () => {
      if (!serverWarming) { runCatalogGallery(); return; }
      if (Date.now() - start > 120000) return;   // bounded: never a permanent retry loop
      setTimeout(armed, 500);
    };
    armed();
  }
}

// -- Run ----------------------------------------------------------------------------------

let running = false;
const STAGE_LABEL = { load: "Loading the frozen policy", search: "Stage 1, PPO search", refine: "Stage 2, G3.2 refinement", verify: "Stage 3, independent verification" };
const STAGE_ORDER = ["search", "refine", "verify"];

function setStep(stage, state) {
  const s = $(`.step[data-stage="${stage}"]`);
  if (s) s.className = `step${state ? " " + state : ""}`;
}
function markStepsLive(stage) {
  const key = stage === "load" ? "search" : stage;
  const at = STAGE_ORDER.indexOf(key);
  if (at < 0) return;
  STAGE_ORDER.forEach((s, i) => setStep(s, i < at ? "done" : i === at ? "active" : ""));
}

function appendLog(events) {
  const log = $("#run-log");
  const empty = log.querySelector(".log-empty");
  if (empty) empty.remove();
  for (const ev of events) {
    const pass = ev.kind === "candidate" && ev.ok !== false && /passes all/.test(ev.text || "");
    const row = el("div", { class: `log-row log-${ev.kind}${ev.ok === false ? " log-rejected" : ""}${pass ? " log-pass" : ""}` });
    row.innerHTML = `<span class="log-t">${fmt(ev.t, 1)}s</span><span class="log-text">${escapeHtml(ev.text)}</span>`;
    log.appendChild(row);
  }
  log.scrollTop = log.scrollHeight;
}

function handleRunEvents(state, events) {
  for (const ev of events) {
    if (ev.kind === "candidate") {
      state.points.push({ boost: typeof ev.boost === "number" ? ev.boost : null, ok: ev.ok !== false,
        pass: ev.ok !== false && /passes all/.test(ev.text || ""), label: ev.text });
      state.simulated += 1;
      if (ev.design) {
        state.galleryCandidates.push(liveGalleryCandidate(ev.design, state, `Pipeline candidate ${state.simulated}`));
        if (state.galleryCandidates.length > 6) state.galleryCandidates.shift();
        setParams(ev.design);
        setStageTitle(`Candidate ${state.simulated}`, STAGE_LABEL[ev.stage] || "simulating");
        scheduleSchematic(designToHumanFields(ev.design), `Candidate ${state.simulated}`,
          ev.ok === false ? "rejected by the guard layer" : `${fmt(ev.boost, 2)} dB boost, ${STAGE_LABEL[ev.stage] || ""}`);
      }
      if (typeof ev.boost === "number") {
        const err = Math.abs(ev.boost - state.target);
        setReadout("Latest boost", `${fmt(ev.boost, 2)}<span class="u">dB</span>`, `target ${fmt(state.target, 2)} dB, off by ${fmt(err, 2)} dB${err <= state.tol ? ", inside tolerance" : ""}`);
      } else {
        setReadout("Latest circuit", `<span style="color:var(--warning)">rejected</span>`, "the guard layer refused to score it");
      }
    }
  }
  renderTraceChart($("#trace-chart"), state.points, { target: state.target, tol: state.tol });
  $("#trace-count").textContent = `${state.simulated} simulated`;
}

function pollRunProgress(state) {
  return setInterval(async () => {
    if (state.stopped) return;
    try {
      const p = await api(`/api/pipeline/progress?since=${state.cursor}`);
      state.cursor = p.next;
      if (p.events.length) { appendLog(p.events); handleRunEvents(state, p.events); }
      if (p.stage) markStepsLive(p.stage);
      $("#trace-elapsed").textContent = `${fmt(p.elapsed_s, 1)} s`;
    } catch { /* a dropped poll is not a failed run */ }
  }, 300);
}

async function runDesign() {
  if (running || serverWarming || !defaults) return;
  const text = $("#spec-text").value.trim();
  if (text && llmReadText !== text) {
    // The run is about to spend real simulator budget on this request, so give the
    // stronger reader a chance first. A refusal stops the run: nothing was understood.
    const spec = await readWithClaude();
    if (!spec && !lastParse) return;
  }
  running = true;
  const btn = $("#run-btn");
  btn.disabled = true; btn.classList.add("running");
  $$(".mode-btn").forEach((b) => { b.disabled = true; });
  currentResult = null;
  $("#result").innerHTML = "";
  ["export-netlist", "export-json"].forEach((id) => { $(`#${id}`).disabled = true; });
  const trace = $("#trace-card");
  trace.hidden = false;
  STAGE_ORDER.forEach((s) => setStep(s, ""));
  setStep("search", "active");
  $("#run-log").innerHTML = `<div class="log-empty">waiting for the first simulation</div>`;
  $("#trace-count").textContent = ""; $("#trace-elapsed").textContent = "";
  setStateChip("live", "designing");
  setStageTitle("Searching", "every candidate below is a real SKY130 simulation");

  const target = Number($("#target").value), channel = Number($("#channel").value);
  const state = { cursor: 0, stopped: false, simulated: 0, points: [], galleryCandidates: [], target, channel, tol: defaults.boost_tol_db };
  renderTraceChart($("#trace-chart"), [], { target, tol: state.tol });
  const t0 = performance.now();
  const timer = pollRunProgress(state);
  const request = {
    target_boost_db: target, channel_loss_db: channel,
    spec_index: Number($("#spec-index").value) || 0,
    allow_fallback: $("#allow-fallback").checked,
    mode: selectedMode, requirements: readRequirements(),
  };
  try {
    const result = await postJSON("/api/pipeline/run", request);
    const elapsedS = (performance.now() - t0) / 1000;
    state.stopped = true; clearInterval(timer);
    try {
      const p = await api(`/api/pipeline/progress?since=${state.cursor}`);
      if (p.events.length) { appendLog(p.events); handleRunEvents(state, p.events); }
    } catch { /* cosmetic */ }
    finishSteps(result);
    if (result.design && result.verification && typeof result.verification.abs_err_db === "number") {
      const m = result.verification.measures || {};
      state.points.push({ boost: m.boost_db, ok: true, pass: result.verification.passed, final: true, label: `delivered: ${fmt(m.boost_db, 3)} dB` });
      renderTraceChart($("#trace-chart"), state.points, { target, tol: state.tol });
    }
    await presentResult(result, elapsedS, { text, target, channel });
    if (result.design) {
      state.galleryCandidates.push(liveGalleryCandidate(result.design, state, "Pipeline final candidate"));
      if (state.galleryCandidates.length > 6) state.galleryCandidates.shift();
    }
    if (state.galleryCandidates.length) queueLiveGallery(state.galleryCandidates);
    pushHistory({ ts: Date.now(), text, target, channel, mode: selectedMode, status: result.status,
      boost: result.verification && result.verification.measures ? result.verification.measures.boost_db : null,
      elapsed: elapsedS, result });
  } catch (err) {
    state.stopped = true; clearInterval(timer);
    setStateChip("bad", "failed");
    setStageTitle("Run failed", err.title || err.message);
    $("#result").innerHTML = errorBannerHtml(err, "Run failed");
  } finally {
    running = false;
    btn.classList.remove("running");
    btn.disabled = serverWarming;
    $$(".mode-btn").forEach((b) => { b.disabled = false; });
    pumpLiveGallery();
  }
}

function finishSteps(r) {
  const st = defaults.statuses;
  if (r.status === st.fallback) { setStep("search", "warn"); setStep("refine", "warn"); setStep("verify", r.verification ? "done" : ""); return; }
  if (!r.design) { setStep("search", "done"); setStep("refine", "warn"); setStep("verify", ""); return; }
  setStep("search", "done"); setStep("refine", "done");
  setStep("verify", !r.verification ? "" : r.status === st.solved ? "done" : "warn");
}

// -- Result -------------------------------------------------------------------------------

const CHECK_META = {
  boost_range: { label: "Boost in range", m: "boost_db", unit: "dB", dp: 2 },
  boost_target: { label: "Boost on target", m: "boost_db", unit: "dB", dp: 2 },
  peak_in_band: { label: "Peak in band", m: "peak_freq_ghz", unit: "GHz", dp: 2 },
  dc_gain: { label: "DC gain floor", m: "dc_gain_db", unit: "dB", dp: 2 },
  hd3: { label: "HD3", m: "hd3_db", unit: "dB", dp: 1 },
  noise: { label: "Input noise", m: "noise_vrms", unit: "µV", dp: 0, scale: 1e6 },
  power: { label: "Power", m: "power_w", unit: "mW", dp: 2, scale: 1e3 },
  area: { label: "Area", m: "area_mm2", unit: "mm²", dp: 4 },
  eye_h: { label: "Eye width", m: "eye_h_ui", unit: "UI", dp: 2 },
  eye_v: { label: "Eye height", m: "eye_v_mv", unit: "mV", dp: 0 },
};

function checksHtml(checks, measures, title) {
  if (!checks) return "";
  const items = Object.entries(checks).map(([k, ok]) => {
    const meta = CHECK_META[k] || { label: k };
    const mv = measures && meta.m && typeof measures[meta.m] === "number" ? `${fmt(measures[meta.m] * (meta.scale || 1), meta.dp)} ${meta.unit}` : "";
    return `<div class="check ${ok ? "pass" : "fail"}"><span class="ci">${ICONS[ok ? "check" : "cross"]}</span>${escapeHtml(meta.label)}<span class="cv">${escapeHtml(mv)}</span></div>`;
  }).join("");
  return `<div class="panel"><div class="panel-head"><span class="panel-title">${escapeHtml(title)}</span><span class="panel-meta">${Object.values(checks).filter(Boolean).length} of ${Object.keys(checks).length} pass</span></div><div class="checks">${items}</div></div>`;
}

// A netlist that leaves this page loses its on-screen banner, so a failed design could
// be read later as a delivered one. Stamp the verdict into the file itself as SPICE
// comments. Verification is corner="tt" only (pipeline.py:306) -- the header says so
// rather than letting "verified" imply a PVT sign-off it never ran.
function netlistWithVerdict(r) {
  if (!r || !r.netlist) return "";
  const st = defaults.statuses, v = r.verification;
  const L = ["*", "* ---------------- SILQ provenance ----------------"];
  if (r.status === st.solved) {
    L.push("* VERIFIED at the TYPICAL CORNER ONLY (tt, 27 C, nominal Vdd), by an",
           "* independent re-simulation against every hard spec.",
           "* This is NOT a PVT sign-off: no corner sweep was run on this design.");
  } else if (r.status === st.fallback) {
    L.push("* NOT AN AI RESULT. Fixed, hand-verified fallback that ignores the requested",
           "* target. Must never be reported as an output of the SILQ search.");
  } else if (r.status === st.closed_not_verified) {
    L.push("* FAILED INDEPENDENT VERIFICATION -- DO NOT USE.",
           "* The solver reported it reached target; a fresh re-simulation disagreed.",
           "* Failing checks: " + ((v && v.failing || ["(unreported)"]).join(", ")),
           "* This design does not meet the requested specification.");
  } else {
    L.push("* UNSOLVED. No guard-valid design was found for this specification.");
  }
  if (v && v.guard_valid === false) L.push("* GUARD REJECTED on re-simulation: " + v.guard_check);
  L.push("* -------------------------------------------------", "*");
  return L.join("\n") + "\n" + r.netlist;
}

function verdictHtml(r) {
  const st = defaults.statuses;
  const v = r.verification;
  const reqs = (r.spec && r.spec.requirements) || [];
  let cls, icon, title, sub;
  if (r.status === st.fallback) {
    cls = "bad"; icon = "alertTriangle"; title = "Fixed fallback design, not an AI result";
    sub = "The architecture found nothing guard-valid for this spec. What is shown is a fixed, hand-verified design that ignores your requested target. It is not an output of the search and must not be read as one.";
  } else if (r.status === st.solved) {
    cls = "ok"; icon = "checkCircle"; title = "Verified at the typical corner";
    sub = `An independent re-simulation at the typical corner (tt, 27 C, nominal Vdd) passes every hard check, boost included. This is not a PVT sign-off — no corner sweep was run on this design.${v && typeof v.abs_err_db === "number" ? ` The delivered boost is ${fmt(v.abs_err_db, 2)} dB from the number you asked for, inside the ±${fmt(r.spec.boost_tol_db, 2)} dB tolerance.` : ""}`;
  } else if (r.status === st.closed_not_verified) {
    cls = "warn"; icon = "alertTriangle"; title = "Closed, but failed independent verification";
    sub = `The solver reported it reached the target; the fresh re-simulation disagrees${v && v.failing && v.failing.length ? `, failing ${v.failing.join(", ")}` : ""}.`;
  } else {
    cls = "neutral"; icon = "xCircle"; title = "Unsolved";
    sub = "The architecture ran and did not reach a guard-valid design for this spec.";
  }
  const bars = [];
  if (v && reqs.length) {
    bars.push(`<span class="state-chip ${v.passed ? "ok" : "bad"}"><span class="dot"></span>your limits: ${v.passed ? "pass" : "fail"}</span>`);
    if ("competition_passed" in v) bars.push(`<span class="state-chip ${v.competition_passed ? "ok" : "warn"}"><span class="dot"></span>competition spec: ${v.competition_passed ? "pass" : `fails ${(v.competition_failing || []).join(", ")}`}</span>`);
  } else if (v) {
    bars.push(`<span class="state-chip ${v.passed ? "ok" : "bad"}"><span class="dot"></span>competition spec: ${v.passed ? "pass" : "fail"}</span>`);
  }
  if (r.auto) {
    bars.push(r.auto.escalated
      ? `<span class="state-chip warn"><span class="dot"></span>auto: fast attempt ${escapeHtml(r.auto.first_status || "")}, escalated to Thinking</span>`
      : `<span class="state-chip ok"><span class="dot"></span>auto: fast search verified first time</span>`);
  } else if (r.mode) {
    bars.push(`<span class="state-chip"><span class="dot"></span>${escapeHtml(r.mode)} mode</span>`);
  }
  return `<div class="verdict ${cls}"><div class="vicon">${ICONS[icon]}</div><div><div class="verdict-title">${escapeHtml(title)}</div><div class="verdict-sub">${escapeHtml(sub)}</div>${bars.length ? `<div class="verdict-bars">${bars.join("")}</div>` : ""}</div></div>`;
}

function measuresHtml(r) {
  const v = r.verification;
  if (!v || !v.measures) return "";
  const m = v.measures;
  return `<div class="panel"><div class="panel-head"><span class="panel-title">Measured on re-simulation</span><span class="panel-meta">tt corner, ${fmt(r.spec.channel_loss_db, 1)} dB channel</span></div>${kpiGrid([
    { label: "Boost", value: fmt(m.boost_db, 3) + " dB", accent: true, help: `asked ${fmt(r.spec.target_boost_db, 2)} ±${fmt(r.spec.boost_tol_db, 2)} dB` },
    { label: "DC gain", value: fmt(m.dc_gain_db, 2) + " dB" },
    { label: "Peak freq", value: fmt(m.peak_freq_ghz, 2) + " GHz" },
    { label: "Power", value: fmt(m.power_w * 1e3, 2) + " mW" },
    { label: "Noise", value: Math.round(m.noise_vrms * 1e6) + " µVrms" },
    { label: "HD3", value: fmt(m.hd3_db, 1) + " dB" },
    { label: "Eye width", value: fmt(m.eye_h_ui, 2) + " UI" },
    { label: "Eye height", value: Math.round(m.eye_v_mv) + " mV" },
    { label: "Area", value: fmt(m.area_mm2 * 1e3, 1) + " ×10⁻³ mm²" },
  ])}</div>`;
}

function requirementsHtml(r) {
  const reqs = (r.spec && r.spec.requirements) || [];
  if (!reqs.length) return "";
  const rows = reqs.map((q) => {
    const meta = (defaults.requirements || []).find((d) => d.field === q.field) || { label: q.field, unit: "", scale: 1 };
    const val = typeof q.value === "number" ? prettyDisp(q.value * meta.scale, meta.unit) : "";
    const def = typeof q.default === "number" ? prettyDisp(q.default * meta.scale, meta.unit) : "";
    return `<div class="kv-row"><span class="kv-k">${escapeHtml(meta.label)}</span><span class="kv-v">${escapeHtml(val)} <span class="req-tag ${escapeHtml(q.direction || "")}">${escapeHtml(q.direction || "")}</span>${def ? ` <span class="help-hint" style="display:inline;margin:0;">(competition ${escapeHtml(def)})</span>` : ""}</span></div>`;
  }).join("");
  const resel = r.provenance && r.provenance.requirement_reselection;
  return `<div class="panel"><div class="panel-head"><span class="panel-title">Your acceptance limits</span><span class="panel-meta">${escapeHtml(r.spec.scored_against || "")}</span></div>${rows}${resel && resel.applied ? `<div class="help-hint" style="margin-top:8px;">The most accurate candidate failed one of these limits, so the next candidate from the search trace that passes them was re-verified and delivered instead.</div>` : ""}</div>`;
}

function guidanceHtml(r) {
  const g = r.guidance;
  if (!g || !g.headline) return "";
  const suggest = g.suggest_mode && g.suggest_mode !== r.mode && (defaults.modes || []).includes(g.suggest_mode)
    ? `<p class="help-hint" style="margin-top:6px;">Suggested next step: <button type="button" class="mode-suggest" data-mode="${escapeHtml(g.suggest_mode)}">try ${escapeHtml(g.suggest_mode)} mode</button></p>` : "";
  const auto = r.auto && r.auto.escalated && r.auto.first_guidance
    ? `<div class="kv-row"><span class="kv-k">Fast attempt stopped because</span><span class="kv-v">${escapeHtml(r.auto.first_guidance.headline || "")}</span></div>` : "";
  return `<div class="panel"><div class="panel-head"><span class="panel-title">Why the search stopped here</span></div><div style="font-weight:600;">${escapeHtml(g.headline)}</div><p class="help-hint" style="margin-top:4px;">${escapeHtml(g.detail || "")}</p>${g.reason ? `<div class="kv-row"><span class="kv-k">Solver reason</span><span class="kv-v mono">${escapeHtml(g.reason)}</span></div>` : ""}${auto}${suggest}</div>`;
}

function modeDetailHtml(r) {
  const detail = r.provenance && r.provenance.mode_detail;
  const mode = (detail && detail.mode) || r.mode || "default";
  if (!detail || mode === "default") return "";
  if (mode === "fastest") {
    const seed = detail.surrogate_seed;
    const h = detail.hedge;
    let rows = "";
    if (seed) {
      rows += `<div class="kv-row"><span class="kv-k">Corpus seed simulated at</span><span class="kv-v">${fmt(seed.candidate_boost_db, 3)} dB for a ${fmt(seed.target_boost_db, 2)} dB target</span></div>`;
      rows += `<div class="kv-row"><span class="kv-k">Seed outcome</span><span class="kv-v">${seed.candidate_guard_valid ? "guard-valid" : "guard-rejected"}${seed.candidate_loose_pass ? ", inside tolerance, so the refinement stage had nothing left to do" : ", refined by G3.2"}</span></div>`;
    } else if (h) {
      const verdict = !h.attempted ? "not attempted" : !h.evaluated ? "attempted, no candidate cleared the corpus safety radius" : h.accepted ? "accepted" : "attempted, not accepted";
      rows += `<div class="kv-row"><span class="kv-k">Outcome</span><span class="kv-v">${escapeHtml(verdict)}</span></div>${h.reason ? `<div class="kv-row"><span class="kv-k">Reason</span><span class="kv-v">${escapeHtml(h.reason)}</span></div>` : ""}`;
    } else {
      rows += `<div class="kv-row"><span class="kv-k">Corpus seed</span><span class="kv-v">no seed was close enough; the frozen policy started the search from scratch</span></div>`;
    }
    return `<div class="panel"><div class="panel-head"><span class="panel-title">Fastest mode, corpus-seeded start</span></div>${rows}</div>`;
  }
  if (mode === "thinking") {
    const rollouts = detail.rollouts || [];
    const rows = rollouts.map((ro) => {
      const isWinner = ro.start === detail.winner_start;
      const kind = String(ro.start || "").startsWith("surrogate") ? "corpus-proposed start" : "PPO rollout";
      return `<tr><td class="mono">${escapeHtml(ro.start)}${isWinner ? " <strong>(winner)</strong>" : ""}</td><td>${kind}</td><td class="num mono">${ro.best_abs_err_db === null || ro.best_abs_err_db === undefined ? "n/a" : fmt(ro.best_abs_err_db, 4) + " dB"}</td><td class="num mono">${ro.n_loose_pass}</td><td>${ro.reached_target ? "reached target" : "did not reach target"}</td></tr>`;
    }).join("");
    const gov = detail.governor_stopped ? `<p class="help-hint" style="margin-top:6px;">Stopped launching further restarts at the ~60 s budget (${detail.measure_all_spent} measure_all spent). Fewer starting points were tried than this mode's maximum.</p>` : "";
    return `<div class="panel"><div class="panel-head"><span class="panel-title">Thinking mode, ${detail.n_ppo_rollouts || 0} PPO rollout${(detail.n_ppo_rollouts || 0) === 1 ? "" : "s"}${detail.n_surrogate_starts ? ` + ${detail.n_surrogate_starts} corpus start${detail.n_surrogate_starts === 1 ? "" : "s"}` : ""}</span></div><div style="overflow-x:auto;"><table class="parse-table"><thead><tr><th>Start</th><th>Kind</th><th class="num">Best abs. error</th><th class="num">Loose passes</th><th>Target</th></tr></thead><tbody>${rows}</tbody></table></div>${gov}</div>`;
  }
  if (mode === "retarget") {
    const rt = detail.retarget || {};
    return `<div class="panel"><div class="panel-head"><span class="panel-title">Retarget mode</span></div><div class="kv-row"><span class="kv-k">Outcome</span><span class="kv-v">${rt.fired ? `resumed along ${escapeHtml(rt.accepted_axis || "another axis")}` : "did not fire"}</span></div></div>`;
  }
  return "";
}

function costStripHtml(r, elapsedS) {
  const c = r.cost || {};
  const prior = c.measure_all_prior_attempts ? `<span><strong>${c.measure_all_prior_attempts}</strong> of those in the fast attempt</span>` : "";
  return `<div class="cost-strip"><span><strong>${c.optimizer_evals ?? "n/a"}</strong> optimizer evals</span><span><strong>${c.measure_all_total ?? "n/a"}</strong> measure_all calls</span>${prior}<span><strong>${c.spice_analyses_total ?? "n/a"}</strong> SPICE analyses</span>${typeof elapsedS === "number" ? `<span><strong>${fmt(elapsedS, 1)}</strong> s wall clock</span>` : ""}</div>`;
}

async function presentResult(r, elapsedS, ctx) {
  currentResult = r;
  const st = defaults.statuses;
  const v = r.verification;
  const box = $("#result");
  let html = verdictHtml(r);
  if (!r.design) html += banner("warning", "alertTriangle", "No design: the architecture produced nothing guard-valid for this spec, and the fixed fallback was not allowed.");
  html += measuresHtml(r);
  if (v && v.checks) html += checksHtml(v.checks, v.measures, (r.spec.requirements || []).length ? "Checks against your limits" : "The ten hard checks");
  if (v && v.competition_checks && (r.spec.requirements || []).length) html += checksHtml(v.competition_checks, v.measures, "Checks against the competition spec");
  html += requirementsHtml(r);
  html += guidanceHtml(r);
  html += modeDetailHtml(r);
  html += costStripHtml(r, elapsedS);
  html += `<details class="explainer"><summary>Full report</summary><div class="explainer-body"><pre class="code-block">${escapeHtml(r.describe_text || "")}</pre></div></details>`;
  if (r.netlist) html += `<details class="explainer"><summary>SPICE netlist</summary><div class="explainer-body"><pre class="code-block">${escapeHtml(netlistWithVerdict(r))}</pre></div></details>`;
  html += `<details class="explainer"><summary>Raw result JSON</summary><div class="explainer-body"><pre class="code-block">${escapeHtml(JSON.stringify(r, null, 2))}</pre></div></details>`;
  box.innerHTML = html;
  const suggestBtn = box.querySelector(".mode-suggest");
  if (suggestBtn) suggestBtn.addEventListener("click", () => { selectMode(suggestBtn.dataset.mode); $("#mode-seg").scrollIntoView({ behavior: "smooth", block: "center" }); });
  animateIn([...box.children], { y: 12, opacity: 0, stagger: 0.05, duration: 0.5, ease: "power2.out" });

  // The stage: final drawing, sizing, headline number, exports.
  const isFallback = r.status === st.fallback;
  if (r.design) {
    setParams(r.design);
    const m = v && v.measures;
    if (m) setReadout("Delivered boost", `${fmt(m.boost_db, 2)}<span class="u">dB</span>`, `target ${fmt(r.spec.target_boost_db, 2)} dB, ${fmt(m.power_w * 1e3, 2)} mW, ${v.passed ? "verified (tt corner)" : "not verified"}`);
    else setReadout("Boost", "n/a", v ? "guard rejected the final design on re-simulation" : "verification did not run");
    const title = isFallback ? "Fixed fallback design (not AI-generated)" : "Delivered design";
    const sub = `target ${fmt(r.spec.target_boost_db, 1)} dB boost over a ${fmt(r.spec.channel_loss_db, 1)} dB channel`;
    setStageTitle(title, `${sub}${r.mode ? `, ${r.mode} mode` : ""}`);
    setStateChip(r.status === st.solved ? "ok" : isFallback ? "bad" : "warn",
      r.status === st.solved ? "verified (tt)" : isFallback ? "fallback" : r.status === st.closed_not_verified ? "not verified" : "unsolved");
    schematicQueue.pending = null;
    try { placeSvg(await fetchSchematic(designToHumanFields(r.design), title, sub)); }
    catch (err) { $("#schematic-holder").innerHTML = errorBannerHtml(err, "Could not render the schematic"); }
  } else {
    setStageTitle("No design", "nothing guard-valid came out of the search");
    setStateChip("bad", "unsolved");
    setReadout("Boost", "n/a", "");
  }
  $("#export-netlist").disabled = !r.netlist;
  $("#export-json").disabled = false;
}

function initExports() {
  $("#export-svg").addEventListener("click", () => { if (currentSvg) downloadText("silq_schematic.svg", currentSvg, "image/svg+xml"); });
  $("#export-netlist").addEventListener("click", () => { if (currentResult && currentResult.netlist) downloadText("silq_ctle.cir", netlistWithVerdict(currentResult)); });
  $("#export-json").addEventListener("click", () => { if (currentResult) downloadText("silq_result.json", JSON.stringify(currentResult, null, 2), "application/json"); });
}

// -- History ------------------------------------------------------------------------------

const HISTORY_KEY = "silq_history_v1";
function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; } }
function saveHistory(items) { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); } catch { /* quota or blocked */ } }
function pushHistory(item) {
  const items = [item, ...loadHistory()].slice(0, 8);
  saveHistory(items);
  renderHistory();
}
function renderHistory() {
  const items = loadHistory();
  const list = $("#history-list");
  if (!items.length) { list.innerHTML = `<div class="hist-empty">Runs you make on this machine are kept here, so a design can be reopened without spending simulator budget again.</div>`; return; }
  const st = defaults ? defaults.statuses : {};
  list.innerHTML = items.map((it, i) => {
    const cls = it.status === st.solved ? "ok" : it.status === st.fallback ? "bad" : "warn";
    const label = it.status === st.solved ? "verified (tt)" : it.status === st.fallback ? "fallback" : it.status === st.closed_not_verified ? "not verified" : "unsolved";
    const when = new Date(it.ts).toLocaleString(undefined, { hour: "2-digit", minute: "2-digit", month: "short", day: "numeric" });
    return `<button type="button" class="hist" data-i="${i}"><div class="hist-top"><span class="state-chip ${cls}"><span class="dot"></span>${label}</span><span class="hist-meta">${escapeHtml(when)}</span></div><div class="hist-text">${escapeHtml(it.text || `${fmt(it.target, 1)} dB over ${fmt(it.channel, 1)} dB`)}</div><div class="hist-meta"><span>target ${fmt(it.target, 1)} dB</span><span>got ${it.boost === null || it.boost === undefined ? "n/a" : fmt(it.boost, 2) + " dB"}</span><span>${escapeHtml(it.mode || "")}</span><span>${fmt(it.elapsed, 0)} s</span></div></button>`;
  }).join("");
  $$(".hist", list).forEach((b) => b.addEventListener("click", async () => {
    const it = loadHistory()[Number(b.dataset.i)];
    if (!it || !it.result || running) return;
    $("#spec-text").value = it.text || "";
    $("#target").value = it.target; updateTarget();
    $("#channel").value = it.channel; updateChannel();
    $("#trace-card").hidden = true;
    await presentResult(it.result, it.elapsed, it);
    $("#result").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
  }));
}
function initHistory() {
  $("#history-clear").addEventListener("click", () => { saveHistory([]); renderHistory(); });
  renderHistory();
}

// -- Lab: guard sandbox ------------------------------------------------------------------

const TIER_NAMES = { 1: "Run integrity", 2: "Circuit sanity", 3: "Corner integrity", 4: "Physical plausibility", 5: "Search pathology" };

async function initGuard() {
  const data = await api("/api/guard/presets");
  const vddInput = $("#guard-vdd");
  vddInput.min = data.vdd.min; vddInput.max = data.vdd.max; vddInput.value = data.vdd.nominal;
  $("#guard-vdd-hint").textContent = `Spec nominal is ${data.vdd.nominal} V ±${(data.vdd.tolerance * 100).toFixed(0)}%.`;
  const fill = bindRangeFill(vddInput);
  const updateVddLabel = () => { $("#guard-vdd-val").textContent = fmt(vddInput.value, 2); fill(); };
  vddInput.addEventListener("input", updateVddLabel);
  updateVddLabel();

  $("#guard-fields").innerHTML = data.field_meta.map((f) => `<label class="field"><span class="field-label">${escapeHtml(f.label)}</span><input type="number" step="${f.step}" data-field="${f.key}" /></label>`).join("");
  const presetRow = $("#guard-presets");
  presetRow.innerHTML = "";
  data.presets.forEach((p) => {
    const b = el("button", { class: "btn sm", type: "button", title: p.description }, escapeHtml(p.label));
    b.addEventListener("click", () => applyGuardPreset(p));
    presetRow.appendChild(b);
  });
  const defaultsPreset = data.presets.find((p) => p.id === "defaults");
  if (defaultsPreset) applyGuardPreset(defaultsPreset);
  $("#guard-evaluate-btn").addEventListener("click", runGuardEvaluate);
  $("#guard-corner-check-btn").addEventListener("click", runCornerCheck);
  $("#schematic-guard-btn").addEventListener("click", () => showFieldsOnStage(readGuardFields(), "CTLE candidate", "drawn from the Lab's guard fields, not the delivered design"));
}

function applyGuardPreset(preset) {
  for (const [key, val] of Object.entries(preset.fields)) { const input = $(`#guard-fields [data-field="${key}"]`); if (input) input.value = val; }
}
function readGuardFields() {
  const fields = {};
  $$("#guard-fields [data-field]").forEach((inp) => { fields[inp.dataset.field] = Number(inp.value); });
  return fields;
}

async function runGuardEvaluate() {
  const btn = $("#guard-evaluate-btn"), corner = $("#guard-corner").value, box = $("#guard-result");
  btn.disabled = true;
  box.innerHTML = `<div class="card"><div class="spinner-line"><span class="spinner"></span>Running real ngspice on the ${escapeHtml(corner)} corner</div></div>`;
  try {
    const data = await postJSON("/api/guard/evaluate", { fields: readGuardFields(), corner, vdd: Number($("#guard-vdd").value) });
    renderGuardResult(data);
  } catch (err) {
    box.innerHTML = `<div class="card">${errorBannerHtml(err, "Request failed")}</div>`;
  } finally { btn.disabled = false; }
}

async function runCornerCheck() {
  const btn = $("#guard-corner-check-btn");
  btn.disabled = true;
  try {
    const data = await postJSON("/api/guard/verify-corners", { corner: $("#guard-corner").value });
    toast(data.ok ? "Tier 3 passed: tt and ss corner models are genuinely different." : `Tier 3 FAILED: ${data.reason}`, data.ok);
  } catch (err) { toast(`Request failed: ${err.title || err.message}`, false); }
  finally { btn.disabled = false; }
}

function renderGuardResult(data) {
  const box = $("#guard-result");
  if (data.valid) {
    const m = data.metrics, hp = data.hard_pass;
    box.innerHTML = `<div class="card">
        ${banner("success", "checkCircle", "<strong>VALID</strong>: tiers 1, 2 and 4 all passed. This is a real, physically plausible amplifier.")}
        ${kpiGrid([
          { label: "Boost", value: fmt(m.boost_db, 2) + " dB" }, { label: "DC gain", value: fmt(m.dc_gain_db, 2) + " dB" },
          { label: "Peak freq", value: fmt(m.peak_freq_ghz, 2) + " GHz" }, { label: "Power", value: fmt(m.power_w * 1e3, 2) + " mW" },
          { label: "HD3", value: fmt(m.hd3_db, 1) + " dB" }, { label: "Noise", value: Math.round(m.noise_vrms * 1e6) + " µVrms" },
          { label: "Eye width", value: fmt(m.eye_h_ui, 2) + " UI" }, { label: "Eye height", value: Math.round(m.eye_v_mv) + " mV" },
        ])}
        <div style="margin-top:16px; padding-top:14px; border-top:1px solid var(--border);">
          <p style="font-size:var(--fs-sm); margin-bottom:10px;"><strong>Guard-valid is not the same question as spec-passing.</strong> The same design scored against the competition spec:</p>
          <div class="row wrap" style="margin-bottom:10px;"><span class="badge ${hp.ok ? "valid" : "warning"}">${hp.ok ? "PASSES all hard specs" : "fails the hard spec"}</span></div>
          <div class="checks">${Object.entries(hp.checks).map(([k, ok]) => `<div class="check ${ok ? "pass" : "fail"}"><span class="ci">${ICONS[ok ? "check" : "cross"]}</span>${escapeHtml((CHECK_META[k] || { label: k }).label)}</div>`).join("")}</div>
        </div>
        <details class="explainer" style="margin-top:14px;"><summary>Raw run artifacts</summary><div class="explainer-body"><pre class="code-block">${escapeHtml(data.artifact_dir)}</pre></div></details>
      </div>`;
    return;
  }
  const tier = data.tier;
  const rows = data.reached_tiers.map((t) => {
    let cls, icon, note;
    if (t < tier) { cls = "pass"; icon = "check"; note = "passed"; }
    else if (t === tier) { cls = "fail"; icon = "cross"; note = "failed here"; }
    else { cls = "skip"; icon = "dash"; note = "not reached"; }
    return `<div class="tier-row ${cls}"><span class="tier-icon">${ICONS[icon]}</span><span class="tier-name">Tier ${t}, ${TIER_NAMES[t]}</span><span style="margin-left:auto; font-size:var(--fs-xs);">${note}</span></div>`;
  }).join("");
  box.innerHTML = `<div class="card">
      ${banner("danger", "xCircle", `<strong>INVALID</strong>: Tier ${tier} (${TIER_NAMES[tier]}): <code>${escapeHtml(data.check)}</code>`)}
      <p style="font-size:var(--fs-sm); color:var(--ink-muted); margin-bottom:8px;">${escapeHtml(data.reason)}</p>
      <p class="help-hint">${data.violation !== null ? `Violation magnitude: ${fmt(data.violation, 2)}× the failing bound's own scale (0 = exactly on the bound). Reported for context only; the guard's pass/fail decision never reads this number.` : "Violation magnitude: not computable for this failure."}</p>
      <p style="font-size:var(--fs-sm); font-weight:600; margin-top:16px; margin-bottom:4px;">Tier by tier (evaluation stops at the first failure):</p>
      <div class="tier-list">${rows}
        <div class="tier-row pending"><span class="tier-icon">${ICONS.dash}</span><span class="tier-name">Tier 3, ${TIER_NAMES[3]}</span><span style="margin-left:auto; font-size:var(--fs-xs);">startup check, not part of this per-design path</span></div>
        <div class="tier-row pending"><span class="tier-icon">${ICONS.dash}</span><span class="tier-name">Tier 5, ${TIER_NAMES[5]}</span><span style="margin-left:auto; font-size:var(--fs-xs);">only meaningful across a run's history</span></div>
      </div>
      <details class="explainer" style="margin-top:14px;"><summary>Raw run artifacts</summary><div class="explainer-body"><pre class="code-block">${escapeHtml(data.artifact_dir)}</pre></div></details>
    </div>`;
}

// -- Lab: results explorer ---------------------------------------------------------------

let resultsRows = [];

async function initResults() {
  resultsRows = await api("/api/results");
  const sel = $("#results-picker");
  const featured = resultsRows.filter((r) => r.featured), rest = resultsRows.filter((r) => !r.featured);
  sel.innerHTML = `<optgroup label="Featured">${featured.map((r) => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.label)}</option>`).join("")}</optgroup>
    <optgroup label="All files">${rest.map((r) => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.name)}</option>`).join("")}</optgroup>`;
  $("#results-count").textContent = `${featured.length} featured, ${resultsRows.length} total`;
  sel.addEventListener("change", () => loadResult(sel.value));
  const first = featured[0] || resultsRows[0];
  if (first) { sel.value = first.name; loadResult(first.name); }
}

async function loadResult(name) {
  const detail = $("#results-detail");
  detail.innerHTML = `<div class="spinner-line"><span class="spinner"></span>Loading</div>`;
  const row = resultsRows.find((r) => r.name === name) || {};
  let data;
  try { data = await api(`/api/results/${encodeURIComponent(name)}`); }
  catch (err) { detail.innerHTML = errorBannerHtml(err, "Failed to load"); return; }
  const header = `<div class="artifact-header"><h3>${escapeHtml(row.label || name)}</h3>${row.blurb ? `<p>${escapeHtml(row.blurb)}</p>` : ""}<div class="artifact-path">results/${escapeHtml(name)}, ${row.size_kb ?? "?"} KB</div></div>`;
  if (name === "delivered_circuit.json") { detail.innerHTML = header + deliveredHtml(data); return; }
  if (name === "pass_vs_valid.json") { detail.innerHTML = header + passVsValidHtml(data); wirePassVsValidChart(detail, data); return; }
  if (name === "target_tracking_clean40k.json") { detail.innerHTML = header + targetTrackingHtml(data); wireTargetTrackingChart(detail, data); return; }
  detail.innerHTML = header + genericHtml(data);
}

function rawJsonBlock(d) {
  return `<details class="explainer" style="margin-top:14px;"><summary>Raw JSON</summary><div class="explainer-body"><pre class="code-block">${escapeHtml(JSON.stringify(d, null, 2))}</pre></div></details>`;
}

function deliveredHtml(d) {
  const tt = d.pvt.tt_nominal, pvt = d.pvt, p = d.provenance, dv = d.design;
  const wcRows = Object.entries(pvt.worst_case_by_metric).map(([metric, v]) => `<tr><td>${escapeHtml(metric)}</td><td>${fmt(v.min, 3)}</td><td style="font-size:10.5px; color:var(--ink-faint);">${escapeHtml(v.min_at)}</td><td>${fmt(v.max, 3)}</td><td style="font-size:10.5px; color:var(--ink-faint);">${escapeHtml(v.max_at)}</td><td>${v.limit_lo ?? "n/a"}</td><td>${v.limit_hi ?? "n/a"}</td></tr>`).join("");
  return `
    <div class="section-label">Design (eqrl.circuits.ctle.DesignVars)</div>
    <div class="card">${kpiGrid([
      { label: "W / L", value: `${fmt(dv.w_in * 1e6, 2)} / ${fmt(dv.l_in * 1e6, 4)} µm` }, { label: "I_tail", value: `${fmt(dv.i_tail * 1e6, 1)} µA` },
      { label: "Rs / Cs", value: `${fmt(dv.rs / 1e3, 2)} k / ${fmt(dv.cs * 1e15, 1)} f` }, { label: "R_load", value: `${fmt(dv.r_load, 1)} Ω` },
    ])}</div>
    <div class="section-label">Performance at tt, nominal VDD, 27°C</div>
    <div class="card">${kpiGrid([
      { label: "Boost", value: fmt(tt.boost_db, 2) + " dB" }, { label: "DC gain", value: fmt(tt.dc_gain_db, 2) + " dB" },
      { label: "Power", value: fmt(tt.power_w * 1e3, 2) + " mW" }, { label: "Area", value: fmt(tt.area_mm2 * 1e3, 1) + " ×10⁻³ mm²" },
    ])}</div>
    <div class="card">${kpiGrid([
      { label: "PVT corners passed", value: `${pvt.corners_passed} / ${pvt.corners_total}` }, { label: "All corners guard-valid", value: pvt.all_guard_valid ? "yes" : "no" },
      { label: "Worst corner", value: escapeHtml(pvt.worst_corner) }, { label: "Worst-corner target error", value: fmt(pvt.worst_corner_target_err_db, 2) + " dB" },
    ])}</div>
    <div class="section-label">Worst case across all 45 corners, per metric</div>
    <div class="table-wrap"><table class="data"><thead><tr><th>Metric</th><th>Min</th><th>at</th><th>Max</th><th>at</th><th>Limit lo</th><th>Limit hi</th></tr></thead><tbody>${wcRows}</tbody></table></div>
    <div class="section-label">Provenance</div>
    <div class="card"><p style="font-size:var(--fs-sm); color:var(--ink-muted);">Policy <code>${escapeHtml(p.policy)}</code> (sha256 <code>${escapeHtml(p.policy_sha256.slice(0, 12))}</code>), started from <code>${escapeHtml(p.g32_start_source)}</code>, ${escapeHtml(p.g32_reason)}. ${p.total_evals} optimizer evaluations total.</p></div>
    <details class="explainer" style="margin-top:14px;"><summary>Final schematic (SPICE netlist)</summary><div class="explainer-body"><pre class="code-block">${escapeHtml(d.netlist)}</pre></div></details>
    ${rawJsonBlock(d)}`;
}

function passVsValidHtml(d) {
  const totalPass = d.pass_valid + d.pass_invalid;
  const tableRows = d.designs.map((x) => `<tr><td>${fmt(x.target_boost_db, 2)}</td><td>${fmt(x.channel_loss_db, 2)}</td><td>${fmt(x.boost_db, 2)}</td><td>${Math.round(x.eye_v_mv)}</td><td><span class="badge ${x.guard_valid ? "valid" : "invalid"}">${x.guard_valid ? "valid" : "REJECTED"}</span></td><td style="font-family:var(--font-ui); color:var(--ink-muted);">${escapeHtml(x.guard_reason || "")}</td></tr>`).join("");
  return `
    <div class="card">${kpiGrid([
      { label: "Passed all 8 hard specs", value: String(totalPass) }, { label: "and were guard-valid", value: String(d.pass_valid) },
      { label: "but were guard-rejected", value: String(d.pass_invalid) }, { label: "Rejection rate", value: Math.round((100 * d.pass_invalid) / totalPass) + "%", accent: true },
    ])}</div>
    <div class="section-label">Why the rejected designs failed</div>
    <div class="card chart-card"><div class="chart-container" data-chart="why-invalid"></div></div>
    <div class="section-label">Every design that passed the spec (4 CMA-ES runs × up to 60 evaluations, best per run)</div>
    <div class="table-wrap table-scroll"><table class="data"><thead><tr><th>Target boost</th><th>Channel loss</th><th>Achieved boost</th><th>Eye height (mV)</th><th>Guard</th><th>Reason</th></tr></thead><tbody>${tableRows}</tbody></table></div>
    ${rawJsonBlock(d)}`;
}
function wirePassVsValidChart(detail, d) {
  const mount = detail.querySelector('[data-chart="why-invalid"]');
  if (!mount) return;
  renderBarChart(mount, Object.entries(d.why_invalid).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, value })), { ariaLabel: "why designs were guard-rejected" });
}

function pearson(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; num += dx * dy; dx2 += dx * dx; dy2 += dy * dy; }
  return num / Math.sqrt(dx2 * dy2);
}
function targetTrackingHtml(d) {
  const corr = pearson(d.map((r) => r.target), d.map((r) => r.boost));
  const mae = d.reduce((s, r) => s + Math.abs(r.boost - r.target), 0) / d.length;
  return `<div class="card">${kpiGrid([
      { label: "Held-out specs", value: String(d.length) }, { label: "Correlation(requested, achieved)", value: fmt(corr, 3), accent: true }, { label: "Mean abs error", value: fmt(mae, 2) + " dB" },
    ])}<p class="chart-caption">A policy that ignores the requested target and always returns the same design would still score well on the 3 to 12 dB range check. This correlation is what tells the two cases apart.</p></div>
    <div class="card chart-card" style="margin-top:12px;"><div class="chart-container" data-chart="target-tracking"></div></div>
    ${rawJsonBlock(d)}`;
}
function wireTargetTrackingChart(detail, d) {
  const mount = detail.querySelector('[data-chart="target-tracking"]');
  if (!mount) return;
  renderScatterChart(mount, d.map((r) => ({ x: r.target, y: r.boost, tooltip: `target ${fmt(r.target, 2)} dB, achieved ${fmt(r.boost, 2)} dB` })),
    { refLine: true, xLabel: "Requested boost (dB)", yLabel: "Achieved boost (dB)", ariaLabel: "requested vs achieved boost scatter plot" });
}

function formatScalar(v) {
  if (v === null) return "null";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : escapeHtml(String(Math.round(v * 10000) / 10000));
  return escapeHtml(String(v));
}
function formatCell(v) {
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "object") return escapeHtml(JSON.stringify(v));
  return escapeHtml(String(v));
}
function tableHtml(rows) {
  const keys = Object.keys(rows[0]);
  return `<div class="table-wrap table-scroll"><table class="data"><thead><tr>${keys.map((k) => `<th>${escapeHtml(k)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${keys.map((k) => `<td>${formatCell(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function genericHtml(d) {
  let out = "";
  if (d && typeof d === "object" && !Array.isArray(d)) {
    const scalars = Object.entries(d).filter(([, v]) => v === null || ["number", "string", "boolean"].includes(typeof v));
    if (scalars.length) {
      out += `<div class="card">${kpiGrid(scalars.slice(0, 6).map(([k, v]) => ({ label: k, value: formatScalar(v) })))}</div>`;
      if (scalars.length > 6) out += `<div class="card">${kpiGrid(scalars.slice(6, 12).map(([k, v]) => ({ label: k, value: formatScalar(v) })))}</div>`;
    }
    for (const [k, v] of Object.entries(d)) if (Array.isArray(v) && v.length && typeof v[0] === "object") out += `<div class="section-label">${escapeHtml(k)} (${v.length} rows)</div>${tableHtml(v)}`;
  } else if (Array.isArray(d) && d.length && typeof d[0] === "object") {
    out += tableHtml(d);
  }
  return out + rawJsonBlock(d);
}

// -- Lab: design time -----------------------------------------------------------------------

function statTile(value, unit, label, sub) {
  return `<div class="stat-tile"><div class="stat-value">${escapeHtml(value)}<span class="stat-unit">${escapeHtml(unit || "")}</span></div><div class="stat-label">${escapeHtml(label)}</div>${sub ? `<div class="stat-sub">${escapeHtml(sub)}</div>` : ""}</div>`;
}
function pFmt(p) { if (p === null || p === undefined) return "n/a"; return p < 1e-3 ? p.toExponential(1) : p.toFixed(3); }

function renderDesignTime(d) {
  const P = d.protocol, S = d.sweep, E = d.per_eval, DL = d.delivered;
  let html = `<div class="stat-row">
    ${statTile(String(DL.total_evals), " evals", "to size the delivered circuit", `${DL.stage1_ppo_evals} PPO + ${DL.stage2_evals} refinement`)}
    ${statTile(S.total_points.toLocaleString(), " pts", "exhaustive sweep, actually run", `${S.elapsed_h} h, passed spec on ${S.spec_pass}`)}
    ${statTile(`${E.speedup.toFixed(0)}×`, "", "faster per evaluation", `${E.resident_median_s.toFixed(3)} s vs ${E.subprocess_median_s.toFixed(2)} s`)}
  </div>`;
  html += `<div class="section-label">Evaluations to a solution: ${P.n_specs} held-out specs, ${P.budget}-evaluation budget for every arm</div>`;
  html += `<div class="card"><div style="overflow-x:auto;"><table class="dt-table"><thead><tr><th>Method</th><th class="num">Median evals<br/><span class="th-sub">to loose solve</span></th><th class="num">Solved<br/><span class="th-sub">loose</span></th><th class="num">p vs PPO<br/><span class="th-sub">permutation, loose</span></th><th class="num">Sim-matched<br/><span class="th-sub">evals to loose</span></th><th class="num">Solved<br/><span class="th-sub">strict</span></th></tr></thead><tbody>`;
  let allStrictAtChance = true;
  for (const a of d.arms) {
    const sm = a.sim_matched;
    if (!(a.chance && a.chance.p_one_sided > 0.05)) allStrictAtChance = false;
    html += `<tr class="${a.is_silq ? "dt-silq" : ""}"><td><strong>${escapeHtml(a.name)}</strong>${a.is_silq ? ' <span class="dt-tag">silQ</span>' : ""}</td><td class="num">${a.loose_median}</td><td class="num">${a.loose} / ${a.n_specs}</td><td class="num">${a.vs_ppo_loose_p === null || a.vs_ppo_loose_p === undefined ? "n/a" : pFmt(a.vs_ppo_loose_p)}</td><td class="num">${sm ? `${sm.evals} to ${sm.loose}` : "n/a"}</td><td class="num dt-dim" title="${a.chance ? `chance-matched control expected ${a.chance.expected.toFixed(1)}, p=${a.chance.p_one_sided}` : ""}">${a.strict} / ${a.n_specs}</td></tr>`;
  }
  html += `</tbody></table></div><p class="help-hint" style="margin-top:12px;">"Loose" and "strict" are the two preregistered solve criteria (tolerance ${P.tol_db} dB). p-values are ${P.permutations.toLocaleString()}-permutation paired tests against the <strong>PPO</strong> arm, which is the reference. "Sim-matched" re-runs each arm at an equalised simulation count.</p></div>`;
  const ref = d.arms.find((a) => a.name === "PPO-restart");
  if (ref && ref.chance) {
    html += banner("warning", "alertTriangle", `Preregistered negative, reported: the strict solve <em>count</em> is greyed above because it is not distinguishable from a chance-matched control. silQ scores ${ref.chance.observed} against an expectation of ${ref.chance.expected.toFixed(1)} (p = ${ref.chance.p_one_sided})${allStrictAtChance ? ", and the same is true of every other arm, so that column separates nothing" : ""}. The claim this panel makes is the <strong>evaluation count</strong>, not the number of specs solved.`);
  }
  html += `<div class="section-label">Versus sweeping the parameter space</div>`;
  html += `<div class="card"><p style="font-size:var(--fs-sm);">A full-factorial grid over the six design variables was <strong>actually run</strong>, not estimated: ${S.per_axis} points per axis = ${S.total_points.toLocaleString()} points, ${S.elapsed_h} hours of wall clock at ${S.s_per_point} s per point. It was guard-valid on ${S.valid} and passed spec on <strong>${S.spec_pass}</strong> of them; the first success came at point ${S.first_success_index.toLocaleString()}.</p><p class="help-hint">${escapeHtml(S.note)}</p>
    <div class="section-label" style="margin-top:16px;">And that is the coarsest grid there is; refining it explodes</div>
    <table class="dt-table"><thead><tr><th>Points per axis</th><th class="num">Grid size</th><th class="num">Projected wall clock</th></tr></thead><tbody>`;
  for (const [k, hours] of Object.entries(S.extrapolation_hours)) {
    const n = Number(k);
    html += `<tr><td>${n}</td><td class="num">${Math.pow(n, 6).toLocaleString()}</td><td class="num">${hours < 48 ? `${hours.toFixed(1)} h` : `${(hours / 24).toFixed(0)} days`}</td></tr>`;
  }
  html += `</tbody></table></div>`;
  html += `<div class="section-label">Cost per evaluation</div>`;
  html += `<div class="card"><p style="font-size:var(--fs-sm);">Design time is evaluations times cost per evaluation, so the simulator loop was optimised too: a resident libngspice server instead of one subprocess per design. <strong>${E.resident_median_s.toFixed(4)} s</strong> vs <strong>${E.subprocess_median_s.toFixed(2)} s</strong>, a ${E.speedup.toFixed(1)}× speedup. At that rate the delivered circuit's ${DL.total_evals} evaluations are ${DL.wall_clock_s_at_resident_rate} s of simulation.</p><p class="help-hint">${escapeHtml(E.note)}</p></div>`;
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

// -- Boot -----------------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initViews();
  runIntro(false);
  pollHealth();
  initExports();
  initSpec()
    .then(() => { initComposer(); initHistory(); })
    .catch((err) => { $("#result").innerHTML = errorBannerHtml(err, "Failed to load pipeline defaults"); });
  showDeliveredOnStage();
  initGuard().catch((err) => { $("#guard-result").innerHTML = `<div class="card">${errorBannerHtml(err, "Failed to load guard layer")}</div>`; });
  // The gallery is artifact-backed and measures nothing on load, so it boots beside the
  // other panels rather than waiting on the simulator. Its failure must land IN the grid:
  // the placeholder is a spinner, and an uncaught rejection here would leave the section
  // spinning forever while claiming to be loading something.
  initCandidateGallery().catch((err) => { $("#gallery-grid").innerHTML = errorBannerHtml(err, "Failed to load the candidate gallery"); });
  initResults().catch((err) => { $("#results-detail").innerHTML = errorBannerHtml(err, "Failed to load results"); });
  initDesignTime().catch((err) => { $("#designtime-body").innerHTML = errorBannerHtml(err, "Failed to load design-time record"); });
});
