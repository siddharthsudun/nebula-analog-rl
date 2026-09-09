// Hand-rolled SVG charts, no charting library. Three shapes: a horizontal bar chart (why
// designs get guard-rejected), a scatter plot with a y=x reference (does achieved boost
// track the requested target), and the live boost trace drawn while a design run is in
// flight (every simulated circuit's boost against the target band).

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, String(v));
  return el;
}

function textEl(attrs, content) {
  const el = svgEl("text", attrs);
  el.textContent = content;
  return el;
}

function niceStep(range, targetTicks) {
  const raw = range / targetTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm >= 5 ? 5 : norm >= 2 ? 2 : 1;
  return step * mag;
}

function niceTicks(min, max, targetTicks) {
  const step = niceStep(max - min, targetTicks) || 1;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Math.round(v * 1000) / 1000);
  return ticks;
}

function ensureTooltip(container) {
  let tip = container.querySelector(".chart-tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "chart-tooltip";
    container.appendChild(tip);
  }
  return tip;
}

function attachTip(node, container, text) {
  const tip = ensureTooltip(container.closest(".chart-container") || container);
  node.addEventListener("mousemove", (ev) => {
    const wrap = container.closest(".chart-container") || container;
    const rect = wrap.getBoundingClientRect();
    tip.style.left = `${ev.clientX - rect.left}px`;
    tip.style.top = `${ev.clientY - rect.top}px`;
    tip.textContent = text;
    tip.classList.add("show");
  });
  node.addEventListener("mouseleave", () => tip.classList.remove("show"));
}

/** Horizontal bar chart. data: [{label, value}] (caller controls order/sort). */
function renderBarChart(container, data, opts = {}) {
  const width = opts.width || 620;
  const rowH = 38;
  const padLeft = opts.padLeft || 210;
  const padRight = 46;
  const padTop = 6;
  const padBottom = 6;
  const height = padTop + padBottom + data.length * rowH;
  const maxVal = Math.max(...data.map((d) => d.value), 1);
  const plotW = width - padLeft - padRight;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`, class: "chart-svg", role: "img",
    "aria-label": opts.ariaLabel || "bar chart",
  });
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop, x2: padLeft, y2: height - padBottom, class: "axis-line" }));

  data.forEach((d, i) => {
    const y = padTop + i * rowH;
    const barH = rowH - 16;
    const barW = Math.max((d.value / maxVal) * plotW, 3);
    svg.appendChild(textEl({ x: padLeft - 12, y: y + rowH / 2 + 4, "text-anchor": "end", class: "bar-label" }, d.label));
    svg.appendChild(svgEl("rect", { x: padLeft, y: y + 8, width: plotW, height: barH, rx: 3, class: "bar-track" }));
    svg.appendChild(svgEl("rect", { x: padLeft, y: y + 8, width: barW, height: barH, rx: 3, class: "bar" }));
    svg.appendChild(textEl({ x: padLeft + barW + 9, y: y + rowH / 2 + 4, class: "bar-value" },
      opts.valueFmt ? opts.valueFmt(d.value) : d.value));
  });

  container.innerHTML = "";
  container.appendChild(svg);
}

/** Scatter plot with an optional y=x reference line and hover tooltips. points: [{x, y, tooltip?}] */
function renderScatterChart(container, points, opts = {}) {
  const width = opts.width || 640;
  const height = opts.height || 400;
  const padLeft = 46, padRight = 18, padTop = 16, padBottom = 40;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let lo = Math.min(...xs, ...ys);
  let hi = Math.max(...xs, ...ys);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;

  const ticks = niceTicks(lo, hi, 6);
  const sx = (v) => padLeft + ((v - lo) / (hi - lo)) * plotW;
  const sy = (v) => padTop + plotH - ((v - lo) / (hi - lo)) * plotH;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart-svg", role: "img", "aria-label": opts.ariaLabel || "scatter chart" });

  ticks.forEach((t) => {
    if (t < lo || t > hi) return;
    svg.appendChild(svgEl("line", { x1: sx(t), y1: padTop, x2: sx(t), y2: padTop + plotH, class: "grid-line" }));
    svg.appendChild(textEl({ x: sx(t), y: padTop + plotH + 18, "text-anchor": "middle", class: "tick-label" }, t));
    svg.appendChild(svgEl("line", { x1: padLeft, y1: sy(t), x2: padLeft + plotW, y2: sy(t), class: "grid-line" }));
    svg.appendChild(textEl({ x: padLeft - 9, y: sy(t) + 3.5, "text-anchor": "end", class: "tick-label" }, t));
  });
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop, x2: padLeft, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop + plotH, x2: padLeft + plotW, y2: padTop + plotH, class: "axis-line" }));
  if (opts.refLine) svg.appendChild(svgEl("line", { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), class: "ref-line" }));
  svg.appendChild(textEl({ x: padLeft + plotW / 2, y: height - 4, "text-anchor": "middle", class: "axis-title" }, opts.xLabel || ""));
  svg.appendChild(textEl({ x: -(padTop + plotH / 2), y: 14, "text-anchor": "middle", class: "axis-title", transform: "rotate(-90)" }, opts.yLabel || ""));

  points.forEach((p) => {
    const c = svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 5, class: "point" });
    attachTip(c, container, p.tooltip || `${p.x}, ${p.y}`);
    svg.appendChild(c);
  });

  container.innerHTML = "";
  container.appendChild(svg);
}

/**
 * Live boost trace. points: [{i, boost|null, ok, pass, final?, label}] in simulation order.
 * A rejected circuit has no boost and is drawn as a tick on the baseline so the reader still
 * sees that the simulation happened. The target band is target +/- tol.
 */
function renderTraceChart(container, points, opts = {}) {
  const width = opts.width || 560;
  const height = opts.height || 214;
  const padLeft = 40, padRight = 14, padTop = 12, padBottom = 26;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const target = Number(opts.target);
  const tol = Number(opts.tol) || 1.5;

  const boosts = points.filter((p) => typeof p.boost === "number").map((p) => p.boost);
  let lo = Math.min(target - tol - 0.5, ...boosts);
  let hi = Math.max(target + tol + 0.5, ...boosts);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 12; }
  const n = Math.max(points.length, 8);
  const sx = (i) => padLeft + ((i + 0.5) / n) * plotW;
  const sy = (v) => padTop + plotH - ((v - lo) / (hi - lo)) * plotH;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart-svg", role: "img",
    "aria-label": "boost of every simulated circuit against the target band" });

  svg.appendChild(svgEl("rect", { x: padLeft, y: sy(target + tol), width: plotW, height: Math.max(sy(target - tol) - sy(target + tol), 1), class: "band" }));
  niceTicks(lo, hi, 4).forEach((t) => {
    if (t < lo || t > hi) return;
    svg.appendChild(svgEl("line", { x1: padLeft, y1: sy(t), x2: padLeft + plotW, y2: sy(t), class: "grid-line" }));
    svg.appendChild(textEl({ x: padLeft - 7, y: sy(t) + 3.5, "text-anchor": "end", class: "tick-label" }, t));
  });
  svg.appendChild(svgEl("line", { x1: padLeft, y1: sy(target), x2: padLeft + plotW, y2: sy(target), class: "target-line" }));
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop, x2: padLeft, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop + plotH, x2: padLeft + plotW, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(textEl({ x: padLeft + plotW, y: height - 6, "text-anchor": "end", class: "chart-note" }, "simulated circuits, in order"));
  svg.appendChild(textEl({ x: padLeft + 6, y: sy(target) - 5, class: "chart-note" }, `target ${target.toFixed(2)} dB`));

  const line = [];
  points.forEach((p, i) => {
    if (typeof p.boost !== "number") {
      const t = svgEl("line", { x1: sx(i), y1: padTop + plotH - 8, x2: sx(i), y2: padTop + plotH, class: "tp rej" });
      t.style.stroke = "var(--warning)"; t.style.strokeWidth = "2";
      attachTip(t, container, p.label || `circuit ${i + 1}: rejected by the guard layer`);
      svg.appendChild(t);
      return;
    }
    line.push(`${sx(i)},${sy(p.boost)}`);
  });
  if (line.length > 1) svg.appendChild(svgEl("polyline", { points: line.join(" "), class: "trace-line" }));
  points.forEach((p, i) => {
    if (typeof p.boost !== "number") return;
    const cls = "tp" + (p.final ? " final" : p.pass ? " pass" : "");
    const c = svgEl("circle", { cx: sx(i), cy: sy(p.boost), r: p.final ? 5.5 : 3.6, class: cls });
    attachTip(c, container, p.label || `circuit ${i + 1}: ${p.boost.toFixed(2)} dB`);
    svg.appendChild(c);
  });
  if (!points.length) {
    svg.appendChild(textEl({ x: padLeft + plotW / 2, y: padTop + plotH / 2, "text-anchor": "middle", class: "chart-note" }, "no circuits simulated yet"));
  }

  container.innerHTML = "";
  container.appendChild(svg);
}

/**
 * Folded eye display from the eye engine. The roll is intentionally the same transform as
 * experiments/eyeplot.py:_centered: the selected sampling phase becomes the centre column.
 * `eye` is display data only; callers keep its guard and behavioral-measurement scope visible
 * beside this chart rather than treating a dense trace as a sign-off.
 */
function renderEyeChart(container, eye, opts = {}) {
  container.innerHTML = "";
  if (!eye || !eye.available || !Array.isArray(eye.eye_matrix) || !eye.eye_matrix.length) {
    const empty = document.createElement("div");
    empty.className = "eye-empty";
    empty.textContent = (eye && eye.reason) || "Eye trace unavailable.";
    container.appendChild(empty);
    return;
  }

  const matrix = eye.eye_matrix;
  const M = matrix[0].length;
  const phase = Number(eye.sample_phase);
  if (!Number.isInteger(M) || M < 1 || !Number.isInteger(phase) || phase < 0 || phase >= M ||
      matrix.some((row) => !Array.isArray(row) || row.length !== M)) {
    const empty = document.createElement("div");
    empty.className = "eye-empty";
    empty.textContent = "Eye trace had an invalid matrix shape.";
    container.appendChild(empty);
    return;
  }

  const width = opts.width || 330;
  const height = opts.height || 210;
  const padLeft = 42, padRight = 10, padTop = 12, padBottom = 30;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const shift = Math.floor(M / 2) - phase; // same centre-and-fold convention as eyeplot.py
  const finite = matrix.flat().filter(Number.isFinite);
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = -0.5; hi = 0.5; }
  const margin = Math.max((hi - lo) * 0.08, 0.002);
  lo -= margin; hi += margin;
  const sx = (index) => padLeft + (index / Math.max(M - 1, 1)) * plotW;
  const sy = (value) => padTop + plotH - ((value - lo) / Math.max(hi - lo, Number.EPSILON)) * plotH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`, class: "chart-svg eye-svg", role: "img",
    "aria-label": opts.ariaLabel || "folded eye diagram centered on the sampling phase",
  });
  svg.appendChild(svgEl("line", { x1: padLeft, y1: sy(0), x2: padLeft + plotW, y2: sy(0), class: "grid-line" }));
  svg.appendChild(svgEl("line", { x1: sx(Math.floor(M / 2)), y1: padTop, x2: sx(Math.floor(M / 2)), y2: padTop + plotH, class: "eye-sample-line" }));
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop + plotH, x2: padLeft + plotW, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop, x2: padLeft, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(textEl({ x: padLeft + plotW / 2, y: height - 5, "text-anchor": "middle", class: "axis-title" }, "time (UI, centred)"));
  svg.appendChild(textEl({ x: -padTop - plotH / 2, y: 13, "text-anchor": "middle", class: "axis-title", transform: "rotate(-90)" }, "differential (V)"));

  // The endpoint returns every folded row for auditability. Drawing all of them would
  // obscure the aperture and waste the browser, so uniformly sample at most 220 rows.
  const count = Math.min(matrix.length, opts.maxTraces || 220);
  const stride = matrix.length / count;
  for (let rowIndex = 0; rowIndex < count; rowIndex += 1) {
    const row = matrix[Math.floor(rowIndex * stride)];
    const points = [];
    for (let column = 0; column < M; column += 1) {
      const source = (column - shift + M) % M;
      const value = row[source];
      if (Number.isFinite(value)) points.push(`${sx(column)},${sy(value)}`);
    }
    if (points.length > 1) svg.appendChild(svgEl("polyline", { points: points.join(" "), class: "eye-trace" }));
  }
  svg.appendChild(textEl({ x: padLeft + 4, y: padTop + 12, class: "chart-note" }, "sample"));
  container.appendChild(svg);
}
