// Hand-rolled SVG charts -- no charting library. Two shapes needed by the results
// explorer: a horizontal bar chart (why designs get guard-rejected) and a scatter
// plot with a y=x reference line (does achieved boost track the requested target).

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

/**
 * Horizontal bar chart. data: [{label, value}] (caller controls order/sort).
 */
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
    viewBox: `0 0 ${width} ${height}`,
    class: "chart-svg",
    role: "img",
    "aria-label": opts.ariaLabel || "bar chart",
  });
  svg.appendChild(svgEl("line", {
    x1: padLeft, y1: padTop, x2: padLeft, y2: height - padBottom, class: "axis-line",
  }));

  data.forEach((d, i) => {
    const y = padTop + i * rowH;
    const barH = rowH - 16;
    const barW = Math.max((d.value / maxVal) * plotW, 3);
    svg.appendChild(textEl(
      { x: padLeft - 12, y: y + rowH / 2 + 4, "text-anchor": "end", class: "bar-label" },
      d.label,
    ));
    const track = svgEl("rect", { x: padLeft, y: y + 8, width: plotW, height: barH, rx: 3 });
    track.style.fill = "var(--surface-2)";
    svg.appendChild(track);
    const bar = svgEl("rect", { x: padLeft, y: y + 8, width: barW, height: barH, rx: 3, class: "bar" });
    svg.appendChild(bar);
    svg.appendChild(textEl(
      { x: padLeft + barW + 9, y: y + rowH / 2 + 4, class: "bar-value" },
      opts.valueFmt ? opts.valueFmt(d.value) : d.value,
    ));
  });

  container.innerHTML = "";
  container.appendChild(svg);
}

/**
 * Scatter plot with an optional y=x reference line and hover tooltips.
 * points: [{x, y, tooltip?}]
 */
function renderScatterChart(container, points, opts = {}) {
  const width = opts.width || 640;
  const height = opts.height || 400;
  const padLeft = 46;
  const padRight = 18;
  const padTop = 16;
  const padBottom = 40;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let lo = Math.min(...xs, ...ys);
  let hi = Math.max(...xs, ...ys);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad;
  hi += pad;

  const xTicks = niceTicks(lo, hi, 6);
  const yTicks = niceTicks(lo, hi, 6);
  const sx = (v) => padLeft + ((v - lo) / (hi - lo)) * plotW;
  const sy = (v) => padTop + plotH - ((v - lo) / (hi - lo)) * plotH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "chart-svg",
    role: "img",
    "aria-label": opts.ariaLabel || "scatter chart",
  });

  xTicks.forEach((t) => {
    if (t < lo || t > hi) return;
    svg.appendChild(svgEl("line", { x1: sx(t), y1: padTop, x2: sx(t), y2: padTop + plotH, class: "grid-line" }));
    svg.appendChild(textEl({ x: sx(t), y: padTop + plotH + 18, "text-anchor": "middle", class: "tick-label" }, t));
  });
  yTicks.forEach((t) => {
    if (t < lo || t > hi) return;
    svg.appendChild(svgEl("line", { x1: padLeft, y1: sy(t), x2: padLeft + plotW, y2: sy(t), class: "grid-line" }));
    svg.appendChild(textEl({ x: padLeft - 9, y: sy(t) + 3.5, "text-anchor": "end", class: "tick-label" }, t));
  });

  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop, x2: padLeft, y2: padTop + plotH, class: "axis-line" }));
  svg.appendChild(svgEl("line", { x1: padLeft, y1: padTop + plotH, x2: padLeft + plotW, y2: padTop + plotH, class: "axis-line" }));

  if (opts.refLine) {
    svg.appendChild(svgEl("line", { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), class: "ref-line" }));
  }

  svg.appendChild(textEl(
    { x: padLeft + plotW / 2, y: height - 4, "text-anchor": "middle", class: "axis-title" },
    opts.xLabel || "",
  ));
  const yTitle = textEl(
    { x: -(padTop + plotH / 2), y: 14, "text-anchor": "middle", class: "axis-title", transform: "rotate(-90)" },
    opts.yLabel || "",
  );
  svg.appendChild(yTitle);

  const tip = ensureTooltip(container.closest(".chart-container") || container);

  points.forEach((p) => {
    const c = svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: 5, class: "point" });
    c.addEventListener("mousemove", (ev) => {
      const wrap = container.closest(".chart-container") || container;
      const rect = wrap.getBoundingClientRect();
      tip.style.left = `${ev.clientX - rect.left}px`;
      tip.style.top = `${ev.clientY - rect.top}px`;
      tip.textContent = p.tooltip || `${p.x}, ${p.y}`;
      tip.classList.add("show");
    });
    c.addEventListener("mouseleave", () => tip.classList.remove("show"));
    svg.appendChild(c);
  });

  container.innerHTML = "";
  container.appendChild(svg);
}
