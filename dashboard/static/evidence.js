/* Historical evidence is a static, inspectable snapshot. No simulation is launched. */
(() => {
  "use strict";
  const descriptions = {
    auto: ["Auto", "RECOMMENDED", "Starts with the fastest route and escalates to a wider search when the specification needs it."],
    fastest: ["Fastest", "LOW LATENCY", "Reuses the closest matching design already measured in history for a quick answer. Retries up to four distinct corpus seeds when guards reject a seed, then uses a small local refinement budget."],
    thinking: ["Thinking", "WIDER SEARCH", "Spends more evaluations exploring and refining candidate circuits toward the requested specification."],
    default: ["Default", "LEGACY", "Historical PPO-first search with local refinement."],
    retarget: ["Retarget", "LEGACY", "Historical retargeted search and refinement path."],
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const percent = (value) => `${(100 * value).toFixed(1)}%`;
  const error = (value) => value === null ? "n/a" : value < 0.001 ? "&lt;0.001 dB" : `${value.toPrecision(3)} dB`;
  function card(row) {
    const [name, badge, description] = descriptions[row.mode];
    return `<article class="evidence-card ${row.mode === "auto" ? "recommended" : ""}">
      <div class="evidence-card-head"><h2>${name}</h2><span class="evidence-badge">${badge}</span></div>
      <p class="evidence-description">${description}</p>
      <div class="evidence-score"><strong>${percent(row.solve_rate)}</strong><span>${row.solved} / ${row.cases} specs solved</span></div>
      <div class="evidence-bar" aria-hidden="true"><span style="width:${row.solve_rate * 100}%"></span></div>
      <dl class="evidence-stats">
        <div><dt>Not solved</dt><dd>${row.non_solved}/${row.cases} · ${percent(row.non_solved / row.cases)}</dd></div>
        <div><dt>Median boost error</dt><dd>${error(row.median_error_db)}</dd></div>
        <div><dt>Median runtime</dt><dd>${row.median_seconds.toFixed(1)} s</dd></div>
        <div><dt>Median optimizer evaluations</dt><dd>${row.median_evaluations}</dd></div>
        <div><dt>Target-hit reference*</dt><dd>${percent(row.empirical_target_hit_rate)}</dd></div>
      </dl>
      <p class="evidence-note">Boost error has ${row.error_cases}/${row.cases} measured cases. ${row.cases - row.error_cases} unmeasured. Small error on returned measurements does not mean every spec was solved. *Spec-blind chance baseline at this mode's typical budget -- see Method below.</p>
    </article>`;
  }
  async function loadEvidence() {
    const host = document.getElementById("mode-evidence");
    try {
      const response = await fetch("/static/data/mode-evidence.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.schema !== "silq.frontend.evidence.v1" || !data.modes.every((r) => descriptions[r.mode])) throw new Error("Unsupported evidence snapshot");
      const measured = data.modes.find((r) => r.mode === "auto");
      host.innerHTML = `<article class="evidence-card recommended"><div class="evidence-card-head"><h2>Measured success rate</h2><span class="evidence-badge">HISTORICAL BENCHMARK</span></div><div class="evidence-score"><strong>${percent(measured.solve_rate)}</strong><span>${measured.solved} of ${measured.cases} specifications solved with Auto</span></div><p>Measured success rate means the fraction of benchmark specifications that returned a verified nominal circuit: ${measured.solved} / ${measured.cases} = ${percent(measured.solve_rate)}. Auto combines model proposals with search and refinement. Each solved case passed the recorded circuit checks and met its boost target within &plusmn;${data.tolerance_db} dB.</p><p class="evidence-note">Measured on the historical matched sample, spec seed 137. This is a system solve rate, not standalone PPO accuracy, a guarantee for unseen specifications, or PVT or optional SNR certification. The search modes share a policy.</p></article><p class="evidence-note">${esc(data.timestamp_note)}</p><div class="evidence-grid">${data.modes.filter((r) => ["auto", "fastest", "thinking"].includes(r.mode)).map(card).join("")}</div>
        <details class="evidence-methods"><summary>Method, legacy modes and source records · ${data.matched_cases} matched specs</summary>
          <p><strong>Partial historical benchmark:</strong> ${data.matched_cases} of ${data.expected_cases} planned cases have a matched record across all five modes. Each mode is compared on the same targets and channels, spec seed ${data.spec_seed}, with a ±${data.tolerance_db} dB target tolerance. Results describe this recorded sample, not generalization to every possible specification.</p>
          <p>${esc(data.error_definition)} ${esc(data.solve_definition)} Runtime includes the recorded run; optimizer evaluation counts are a separate search-budget measure. These nominal results exclude the current PVT repair stage and optional SNR path.</p>
          <div class="evidence-table-scroll"><table><caption class="sr-only">All historical search modes and their matched comparison</caption><thead><tr><th scope="col">Mode</th><th scope="col">Solved</th><th scope="col">Boost error</th><th scope="col">Measured</th><th scope="col">Runtime</th><th scope="col">Evaluations</th><th scope="col">Target-hit reference*</th></tr></thead><tbody>${data.modes.map((r) => `<tr><th scope="row">${descriptions[r.mode][0]}</th><td>${r.solved}/${r.cases}</td><td>${error(r.median_error_db)}</td><td>${r.error_cases}/${r.cases}</td><td>${r.median_seconds.toFixed(1)} s</td><td>${r.median_evaluations}</td><td>${percent(r.empirical_target_hit_rate)}</td></tr>`).join("")}</tbody></table></div>
          <p>Default and Retarget are legacy strategies shown for context. *${esc(data.chance_definition)} It samples the historical achieved-boost distribution using each case's evaluation budget; it is not an equivalent all-constraint solve baseline.</p>
          <div class="evidence-links">${data.sources.map((s) => `<a href="${esc(s.url)}" download>${esc(s.name.split("/").pop())} ↗</a>`).join("")}<a href="/static/data/mode-evidence.json" download>Snapshot + source hashes ↗</a></div>
        </details>`;
    } catch (err) {
      host.innerHTML = `<div class="error-banner" role="alert">The historical comparison could not load. <a href="/static/data/mode-evidence.json">Open the source snapshot</a> or reload the page. <span>${esc(err.message)}</span></div>`;
    }
  }
  document.addEventListener("DOMContentLoaded", loadEvidence);
})();
