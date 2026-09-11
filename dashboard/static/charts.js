// Hand-rolled SVG charts, no charting library. A horizontal bar chart (why designs get
// guard-rejected), a scatter plot with a y=x reference (does achieved boost track the
// requested target), the live boost trace drawn while a design run is in flight, a folded
// eye display, and a line chart with a shaded infeasible region (the SNR stress test).

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
 * Line chart over an x axis, for the SNR robustness sweep.
 *
 * `shadeBelowX` is the whole point of this renderer and not decoration: left of it the eye
 * metric has no feasible point in ANY design space, so a zero there says nothing about the
 * policy. The band is drawn first, behind the data, and it is labelled on the plot rather
 * than in a caption, because the four zeros read as failure to anyone who sees the shape
 * before reading the prose.
 *
 * Label placement here is defensive, because three of this chart's labels sit exactly
 * where its data does. The band caption goes at the BOTTOM of the band: a reference line
 * near the top of the plot would otherwise strike straight through it. Horizontal-line
 * captions stop short of the last x sample rather than running to the plot edge, because
 * the line a reader most wants to check is usually the one a data point sits on top of.
 *
 * The x axis is labelled at the samples themselves, not at round numbers, because the
 * grid is deliberately uneven; labels that would touch are dropped rather than shrunk.
 *
 * series: [{points: [{x, y, lo?, hi?, censored?}], cls, label}]
 *   `censored: true` draws a downward arrow at the plot floor instead of a marker: the
 *   value is off the bottom of the axis and no y is claimed for it.
 * opts: {logY, logFloor, yMin, yMax, yTickFmt, xLabel, yLabel, width, height, ariaLabel,
 *        shadeBelowX, shadeLabel, hlines: [{y,label,cls}], vlines: [{x,label,cls}]}
 */
function renderLineChart(container, series, opts = {}) {
  // The viewBox is set to the container's own pixel width so the CSS `width: 100%` scale
  // factor is 1 and every label renders at its nominal size. Fixed viewBoxes make chart
  // text grow or shrink with the layout -- two charts of different viewBox width side by
  // side end up with visibly different type sizes, and the same pair stacked at a narrow
  // width inverts which one is bigger. The fallback matters: this panel first renders
  // while its view is still hidden, where clientWidth is 0. Callers that care re-draw on
  // view activation (see relayoutSnrCharts).
  const measured = container.clientWidth;
  const width = opts.width || (measured > 240 ? Math.round(measured) : 520);
  const height = opts.height || 260;
  const padLeft = 52, padRight = 14, padTop = 14, padBottom = 42;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const log = !!opts.logY;

  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p.x);
  const xLo = Math.min(...xs), xHi = Math.max(...xs);
  const xPad = (xHi - xLo) * 0.06 || 1;
  const x0 = xLo - xPad, x1 = xHi + xPad;

  const tf = (v) => (log ? Math.log10(Math.max(v, opts.logFloor || 1e-14)) : v);
  const ysRaw = all.map((p) => p.y).concat(all.flatMap((p) => [p.lo, p.hi]))
    .filter((v) => typeof v === "number" && Number.isFinite(v));
  let yLo = opts.yMin !== undefined ? tf(opts.yMin) : Math.min(...ysRaw.map(tf));
  let yHi = opts.yMax !== undefined ? tf(opts.yMax) : Math.max(...ysRaw.map(tf));
  (opts.hlines || []).forEach((h) => { yLo = Math.min(yLo, tf(h.y)); yHi = Math.max(yHi, tf(h.y)); });
  if (!Number.isFinite(yLo) || !Number.isFinite(yHi) || yLo === yHi) { yLo -= 1; yHi += 1; }
  const yPad = (yHi - yLo) * 0.08 || 1;
  if (opts.yMin === undefined) yLo -= yPad;
  if (opts.yMax === undefined) yHi += yPad;
  // Censoring arrows are drawn at the plot floor, so the floor has to be empty. Without
  // this the lowest real datum -- on this chart, the semi-analytic BER at the highest SNR
  // -- sits in the same few pixels as the arrow beside it, and an arrow that lands on
  // another series' marker reads as a value rather than as "off the axis". The gutter is
  // sized in pixels, not decades, so it costs a strip rather than empty octaves.
  // 22px of that is the arrow itself (see the censoring branch below); the rest is the
  // clearance between its tip and the lowest real marker.
  const CENSOR_GUTTER_PX = 40;
  if (opts.yMin === undefined && all.some((p) => p.censored && !Number.isFinite(p.y)) && ysRaw.length) {
    const lowest = Math.min(...ysRaw.map(tf));
    const need = (CENSOR_GUTTER_PX / plotH) * (yHi - yLo);
    if (lowest - yLo < need) yLo = lowest - need;
  }

  const sx = (v) => padLeft + ((v - x0) / (x1 - x0)) * plotW;
  const sy = (v) => padTop + plotH - ((tf(v) - yLo) / (yHi - yLo)) * plotH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`, class: "chart-svg", role: "img",
    "aria-label": opts.ariaLabel || "line chart",
  });

  if (typeof opts.shadeBelowX === "number" && opts.shadeBelowX > x0) {
    const edge = Math.min(sx(opts.shadeBelowX), padLeft + plotW);
    svg.appendChild(svgEl("rect", {
      x: padLeft, y: padTop, width: Math.max(edge - padLeft, 0), height: plotH,
      class: "snr-deadzone",
    }));
    svg.appendChild(svgEl("line", { x1: edge, y1: padTop, x2: edge, y2: padTop + plotH, class: "snr-deadzone-edge" }));
    if (opts.shadeLabel) {
      // Bottom of the band, right-aligned to its own edge. The top-left corner is where
      // a high reference line lands, and a line drawn later strikes through the text.
      svg.appendChild(textEl({ x: edge - 6, y: padTop + plotH - 14, "text-anchor": "end",
                               class: "snr-deadzone-label" }, opts.shadeLabel));
    }
  }

  // y ticks: decade lines under log, nice ticks otherwise
  const yTicks = log
    ? (() => { const t = []; for (let e = Math.ceil(yLo); e <= Math.floor(yHi); e += 1) t.push(e); return t; })()
    : niceTicks(yLo, yHi, 5).filter((t) => t >= yLo && t <= yHi);
  yTicks.forEach((t) => {
    const yv = log ? Math.pow(10, t) : t;
    svg.appendChild(svgEl("line", { x1: padLeft, y1: sy(yv), x2: padLeft + plotW, y2: sy(yv), class: "grid-line" }));
    svg.appendChild(textEl({ x: padLeft - 8, y: sy(yv) + 3.5, "text-anchor": "end", class: "tick-label" },
      log ? `1e${t}` : (opts.yTickFmt ? opts.yTickFmt(t) : t)));
  });
  // One label per x sample, thinned so they cannot touch. An unevenly sampled x axis
  // (this chart's grid is dense exactly where the curve bends) otherwise overlaps its own
  // labels in the interesting region and leaves them legible only where nothing happens.
  // Every kept label still sits on a real sample: none are interpolated tick positions.
  const xLabelled = [...series[0].points].sort((a, b) => a.x - b.x);
  // ~5.4px per character at the 11px tick size; only used for collision arithmetic.
  const box = (p) => { const half = (String(p.x).length * 5.4) / 2, cx = sx(p.x);
                       return { p, cx, left: cx - half, right: cx + half }; };
  const first = box(xLabelled[0]);
  const last = xLabelled.length > 1 ? box(xLabelled[xLabelled.length - 1]) : null;
  // The two ends anchor the reader's sense of the range, so they are kept first and the
  // interior is fitted between them -- a plain left-to-right greedy walk can keep an
  // interior label that then collides with the forced right-hand end.
  const kept = [first];
  let cursor = first.right;
  xLabelled.slice(1, -1).forEach((pt) => {
    const b = box(pt);
    if (b.left < cursor + 5) return;
    if (last && b.right > last.left - 5) return;
    kept.push(b);
    cursor = b.right;
  });
  if (last) kept.push(last);
  kept.forEach(({ p, cx }) => {
    svg.appendChild(textEl({ x: cx, y: padTop + plotH + 18, "text-anchor": "middle", class: "tick-label" }, p.x));
    svg.appendChild(svgEl("line", { x1: cx, y1: padTop + plotH, x2: cx, y2: padTop + plotH + 4, class: "axis-line" }));
  });
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop, x2: padLeft, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop + plotH, x2: padLeft + plotW, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(textEl({ x: padLeft + plotW / 2, y: height - 5, "text-anchor": "middle", class: "axis-title" }, opts.xLabel || ""));
  svg.appendChild(textEl({ x: -(padTop + plotH / 2), y: 13, "text-anchor": "middle", class: "axis-title", transform: "rotate(-90)" }, opts.yLabel || ""));

  // Vertical marks ride the TOP edge, horizontal marks the LEFT edge. Keeping the two
  // families on different edges is what stops them colliding with each other; putting a
  // vertical mark's caption alongside its own line reads well until the curve climbs
  // through it, which is exactly what happens on the chart this was written for.
  (opts.vlines || []).forEach((v) => {
    if (v.x < x0 || v.x > x1) return;
    const vx = sx(v.x);
    svg.appendChild(svgEl("line", { x1: vx, y1: padTop, x2: vx, y2: padTop + plotH,
                                    class: v.cls || "snr-deadzone-edge" }));
    if (!v.label) return;
    // ~5.4px per character at 11px. Only used to decide which side of the line has room.
    const fits = vx - 5 - v.label.length * 5.4 > padLeft;
    svg.appendChild(textEl({ x: vx + (fits ? -5 : 5), y: padTop + 11,
                             "text-anchor": fits ? "end" : "start", class: "chart-note" }, v.label));
  });

  (opts.hlines || []).forEach((h) => {
    svg.appendChild(svgEl("line", { x1: padLeft, y1: sy(h.y), x2: padLeft + plotW, y2: sy(h.y), class: h.cls || "ref-line" }));
    // Left edge, not right. A reference line exists to be compared against a data point,
    // and the point that matters is usually the last one -- which is where a
    // right-anchored caption lands, straight through the marker and its error bar.
    if (h.label) svg.appendChild(textEl({ x: padLeft + 6, y: sy(h.y) - 6, class: "chart-note" }, h.label));
  });

  series.forEach((s) => {
    s.points.forEach((p) => {
      if (typeof p.lo !== "number" || typeof p.hi !== "number") return;
      svg.appendChild(svgEl("line", { x1: sx(p.x), y1: sy(p.lo), x2: sx(p.x), y2: sy(p.hi), class: "snr-err" }));
    });
    const pts = s.points.filter((p) => Number.isFinite(p.y)).map((p) => `${sx(p.x)},${sy(p.y)}`);
    if (pts.length > 1) {
      svg.appendChild(svgEl("polyline", { points: pts.join(" "), class: `snr-line ${s.cls || ""}` }));
    }
    s.points.forEach((p) => {
      if (!Number.isFinite(p.y)) {
        if (!p.censored) return;
        // Off the bottom of the axis, with no y asserted for it. Drawn at the plot floor
        // rather than at a nominal detection limit, because quoting a limit here would
        // claim a bound the measurement does not support.
        const cx = sx(p.x), base = padTop + plotH - 4;
        const g = svgEl("g", { class: `snr-censor ${s.cls || ""}` });
        g.appendChild(svgEl("line", { x1: cx, y1: base - 18, x2: cx, y2: base - 5 }));
        g.appendChild(svgEl("polygon", { points: `${cx - 4},${base - 7} ${cx + 4},${base - 7} ${cx},${base}` }));
        attachTip(g, container, p.tooltip || `${p.x}: below the axis`);
        svg.appendChild(g);
        return;
      }
      const c = svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, class: `snr-dot ${s.cls || ""}` });
      attachTip(c, container, p.tooltip || `${p.x}: ${p.y}`);
      svg.appendChild(c);
    });
  });

  container.innerHTML = "";
  container.appendChild(svg);

  // Backing plates for the on-plot captions, added only once the SVG is in the document
  // because getBBox needs a laid-out text node.
  //
  // Choosing a free corner for each caption does not work here: which corner is free
  // depends on the data (an error bar rising off a zero, a curve climbing through the
  // middle, a reference line landing where a caption starts), so every rule that reads
  // well on one chart is struck through on another. A plate behind the text is
  // data-independent -- whatever crosses the caption is interrupted by it and resumes on
  // the far side, which is what a reader needs and costs nothing but a few pixels.
  svg.querySelectorAll("text.chart-note, text.snr-deadzone-label").forEach((t) => {
    const b = t.getBBox();
    const r = svgEl("rect", {
      x: b.x - 3, y: b.y - 1, width: b.width + 6, height: b.height + 2,
      rx: 2, class: "chart-note-bg",
    });
    svg.insertBefore(r, t);
  });

  if (series.some((s) => s.label)) {
    const legend = document.createElement("div");
    legend.className = "snr-legend";
    series.filter((s) => s.label).forEach((s) => {
      const item = document.createElement("span");
      item.className = "snr-legend-item";
      item.innerHTML = `<i class="${s.cls || ""}"></i>${s.label}`;
      legend.appendChild(item);
    });
    container.appendChild(legend);
  }
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
