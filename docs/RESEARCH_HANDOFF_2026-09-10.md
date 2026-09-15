# Research handoff — VCM/headroom finding + novelty leads (2026-09-10)

*Written for another AI (AstraGPT-5) to pick up and execute. Every claim below is either
(a) measured directly from this repo's own guard layer and cited by file/line, or
(b) sourced to a specific paper/URL from a Firecrawl research pass. Don't take anything
here as already-implemented — it's a prioritized worklist, not a changelog.*

## CORRECTION (2026-09-10, later same day) — §1 and Lead A are both closed, both rejected

AstraGPT-5 executed on this handoff and wrote up the results in
`docs/TAIL_MIRROR_PILOT_2026-09-10.md`. Read that file for the full evidence trail. Short
version, so nobody reruns either of these:

- **§1's "free 1.3 dB at VCM=0.68" claim is retracted — it was a bug, not a finding.**
  `scratchpad/vcm_headroom_sweep.py` rebinds `ctle.VCM`, but the resident simulator actually
  reads `ctle.param_deck` / `VCM_VDD_RATIO`, which that script never touched. The sweep's
  "VCM effect" was actually just CMA-ES seed-to-seed sizing variance, mislabeled as a voltage
  effect. A corrected harness (varying the real `.param vcm_ratio` and verifying `v(cm)`
  matches the request) re-ran the same three sizings at both 0.68 and 0.72: **0.72 (shipped)
  matched or beat 0.68 on boost in 2 of 3 cases, and beat it on headroom margin in all 3**
  (e.g. 214.0 mV vs 153.3 mV at the same sizing). **Do not change `VCM_VDD_RATIO` from 0.72.**
  Step 1 and step 2 of §1's "actionable next steps" below are superseded by this — don't
  action them.
- **Lead A (wide-swing tail mirror) was implemented and is not adopted.** Screened across
  several mirror sizes; every guard-valid config trades one failure for another. The best
  guard-valid version (8x mirror width, bias ratio 0.12) is a real, physically saturated
  circuit with good current delivery, but it overshoots the spec's own boost ceiling (12.05 dB
  vs 12 dB max) while using ~6x the analytical area, more power, more noise, and a smaller eye
  than the simple mirror already shipped. Don't re-propose the wide-swing mirror without a
  materially different sizing strategy than the ones already tried there.

The rest of §1 (the raw sweep data table, the physical-validity results) is left below as a
record of what was measured and why it was misread — not as a live lead. **Leads B and C
(§2) are unaffected by this correction and are still open.**

---

## 0. Non-negotiable constraints (read before touching anything)

- The problem statement (Astera Labs / Nebula @ BITS Goa 2026, analog track) specifies
  **"1-Stage CTLE w/ source degeneration (variable Rs, Cs) + 1-Tap DFE."** A two-stage or
  cascaded CTLE is **out of scope** — this was proposed once in this project's history and
  explicitly retracted after the user caught the spec violation. Do not re-propose it.
- `w_dfe` is deliberately excluded from `ACTION_SPACE` in `src/eqrl/circuits/ctle.py` — the
  1-tap DFE is receiver-DSP applied at the slicer, not an analog knob. Don't add it back.
- Frozen artifacts — never modify: `results/seq_clean40k.zip`, `results/delivered_circuit.json`,
  reward functions/guard thresholds, `src/eqrl/sim/eye.py::compute_eye` (byte-for-byte),
  `src/eqrl/experiments/final_report.py`, `PREREG["tol"] = 1.5` dB. Historical `results/`
  files are never deleted.
- No long simulations or training runs without stating bounded cost up front (evals/time)
  before launching.
- Any claim about circuit behavior must go through the full guard layer
  (`build_evaluator(..., fast=False)` — real HD3/noise, not the fast proxy) and must report
  physical-validity fields (saturation, headroom, tail-current-delivery), not boost/gain
  numbers alone. This was an explicit user requirement this session: *"make sure the circuit
  or whatever is still fine, physically it should exist, everything should still make sense
  in reality, not just in numbers."*
- See `docs/NOVELTY.md` for the project's existing (more conservative) novelty-claims doc —
  read it first so nothing here duplicates or contradicts it. That doc's tone (sourced,
  no unqualified "first/nobody" claims) is the bar to match.

---

## 1. Measured finding: VCM sweep — the headroom theory is WRONG as originally stated

### Background
`src/eqrl/circuits/ctle.py` lines ~196-238 document that the input common-mode voltage
`VCM` trades tail-mirror headroom against input-pair/load headroom one-for-one (tail node
sits at `sp = VCM - Vgs1`). The shipped value is `VCM_VDD_RATIO = 0.72` (line ~240), chosen
because it's the only value where a monotonicity/physics-contract test suite (not a boost
search — see below) passes cleanly.

**Hypothesis tested this session:** if boost is limited by voltage headroom rather than by
topology shape, then lowering VCM should monotonically increase achievable in-band boost, up
to where the tail mirror falls out of saturation.

### What was actually run
`scratchpad/vcm_headroom_sweep.py` — CMA-ES search per VCM setting, maximizing guard-valid
`boost_db` with `peak_freq_ghz` pinned inside the spec band (1.25–2.5 GHz), at
`corner="tt"`, `fast=False` (real HD3/noise gate, not the cheap proxy). 150 evals/point.
Every candidate went through the full guard stack; the winning design at each VCM was
re-probed via `raw_eval` for per-device saturation/headroom/tail-current-delivery. Result
file: `results/vcm_headroom_sweep.json` (already committed to the repo, 6 rows, `seconds:
2076.2`). Full data:

| VCM (×VDD) | VCM (V) | Valid % | In-band survivors | Best boost | @ freq | dc_gain_db | all_saturated | worst headroom | tail delivered | load drop |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.60 | 1.080 | 71.2% | 60 | 8.73 dB | 1.42 GHz | 0.25 | True | 146.6 mV | 97.2% | 0.7106 V |
| 0.64 | 1.152 | 71.2% | 56 | 8.01 dB | 1.35 GHz | 1.22 | True | 138.8 mV | 97.1% | 0.6663 V |
| **0.68** | **1.224** | 69.9% | 57 | **14.08 dB** ← max | 1.27 GHz | 0.76 | True | 214.0 mV | 97.9% | 0.7611 V |
| 0.72 (shipped) | 1.296 | 73.9% | 64 | 12.82 dB | 1.35 GHz | 1.48 | True | 172.7 mV | 97.5% | 0.8711 V |
| 0.76 | 1.368 | 45.8% | 7 | 11.79 dB | 1.27 GHz | 0.91 | True | 132.7 mV | 97.0% | 0.7704 V |
| 0.80 | 1.440 | 60.8% | 37 | 9.19 dB | 1.35 GHz | 2.42 | True | 213.4 mV | 97.9% | 0.7108 V |

### Interpretation (do not restate the monotonic theory as fact — it's falsified)
- **All 6 points are physically valid circuits** — every device saturated, headroom well
  clear of the `SATURATION_HEADROOM_V = 0.050` (50 mV) floor in `src/eqrl/guards.py`, tail
  current delivery 97.0–97.9% of requested. The "is this a real circuit" question is fully
  answered: yes, for all 6.
- **The boost-vs-VCM relationship is NOT monotonic.** It peaks at an interior point
  (0.68×VDD, 14.08 dB), not at the lowest VCM tested. Both directions away from 0.68
  underperform: lower (0.60/0.64 → 8.0–8.7 dB) and higher (0.76/0.80 → 9.2–11.8 dB).
  `load_drop_v` (the simple `I_leg·R_load` term from the original headroom-budget formula)
  also does **not** track boost monotonically — 0.72 has the *highest* load drop of all 6
  rows (0.8711 V) yet is not the highest-boost row; 0.68 achieves more boost with a *lower*
  load drop (0.7611 V). This directly contradicts a single-variable "load drop alone
  determines boost" model.
- **Working explanation, not yet verified:** the CMA-ES search is solving two constraints
  jointly per VCM setting — headroom AND the fixed in-band peak-frequency window
  (1.25–2.5 GHz). At the extremes, designs that would boost harder likely have their peak
  drift outside the allowed band and get excluded from "best in-band," so what survives is
  not the true headroom-only optimum. This has not been separately confirmed (e.g. by
  rerunning without the band constraint and checking where the unconstrained boost optimum
  falls at each VCM) — **that's an open task**, not a settled fact.

### Actionable next steps (not yet done)
1. **There is an apparently free ~1.3 dB sitting at VCM=0.68 vs the shipped 0.72** (14.08 vs
   12.82 dB), with better headroom margin too (214.0 vs 172.7 mV). Before switching
   `VCM_VDD_RATIO` to 0.68, **re-run the existing monotonicity/physics-contract test suite**
   (referenced in `ctle.py`'s comment block, lines ~196-238 — find the actual test file, it
   is not `vcm_headroom_sweep.py`) at 0.68 to confirm it still passes cleanly, the same way
   it was checked at 0.72. The two suites measure different things (this sweep asks "how
   much boost," the other asks "is behavior sane as inputs vary") and 0.72 was chosen for
   the latter, not the former — don't assume 0.68 automatically also passes it.
2. Rerun `sweep_one()` (or a variant) with the in-band constraint removed to test whether the
   non-monotonic boost-vs-VCM curve is actually an artifact of the fixed band window, as
   hypothesized above.
3. If 0.68 is adopted, update the `ctle.py` comment block to reflect the new tradeoff data —
   don't silently change the constant without updating its rationale comment.

---

## 2. Novelty/technique research (Firecrawl pass, 20 queries + 10 full-page scrapes)

Three leads are concretely actionable; the rest is background/positioning. All sources below
were fetched and read in full this session (not just abstracts).

### Lead A — wide-swing current mirror (fixes the actual bottleneck, not just VCM)
**Source:** ECE5211 course notes, §6.3/6.3.3 "Advanced current mirrors: wide-swing" and
"Wide-swing current mirror with enhanced output impedance,"
https://www.d.umn.edu/~htang/ECE5211_doc_files/ECE5211_files/Chapter6_part2.pdf

The tail mirror needs headroom (~0.40 V of Vdsat per `ctle.py`'s own comment) — the whole
VCM sweep above is about buying that headroom by moving VCM around. A wide-swing mirror
topology instead reduces how much headroom the mirror itself *needs* (biases cascode devices
near a single Veff instead of the ~2 Veff a plain cascode mirror needs), which is a direct
fix rather than a workaround. This is "equivalent circuit B" from this project's earlier
7-variant list (varies only the tail implementation — stays legal under the "1-stage CTLE"
constraint since it doesn't change the CTLE topology shape, only how the tail current source
is built). **Not yet implemented or simulated in this repo.** Next step: add a wide-swing
tail variant alongside the existing tail implementation in `ctle.py`, run it through the same
guard-layer + health-probe methodology as `vcm_headroom_sweep.py`.

### Lead B — treat topology-variant choice as an RL action, not a human pre-selection
**Sources:**
- FALCON, https://arxiv.org/html/2505.21923v2 (layout-constrained automated analog circuit
  design via ML)
- AutoCircuit-RL, https://arxiv.org/html/2506.03122v1 ("Reinforcement Learning-Driven LLM for
  Automated Circuit Topology Generation")

Both use RL (one combined with an LLM) to choose *among* topology variants, not just size a
fixed one. This project currently has PPO size a single fixed topology; the 7 "equivalent
circuit" variants proposed earlier this session (tail type, load type — see project history)
are currently a human-curated list, picked one at a time. The novel move: expose "which tail
variant / which load variant" as a small discrete action alongside the existing continuous
sizing action space, and let the existing PPO agent learn which variant wins at which
operating point/corner, instead of a human choosing. This is additive to `docs/NOVELTY.md`'s
existing reward-integrity-guard claim, not a replacement for it — the guard layer would gate
all variants identically. **Not yet implemented.**

### Lead C — PVT robustness (the actual graded weak point)
**Sources:**
- RoSE-Opt, https://arxiv.org/html/2407.19150v1 ("Robust and Efficient Analog Circuit
  Parameter Optimization with Knowledge-infused Reinforcement Learning")
- PVTSizing, https://yibolin.com/publications/papers/ANALOG_DAC2024_Kong.pdf (TuRBO-RL
  PVT-aware batch sizing, DAC 2024)
- RobustAnalog / MIT multi-task RL,
  https://dspace.mit.edu/bitstream/handle/1721.1/146471/3551901.3556487.pdf ("Fast
  Variation-Aware Analog Circuit Design Via Multi-task RL")

Per earlier project history, only 1 of 22 audited candidate specifications was clean across the full 45-corner PVT grid
— this is a spec-graded weakness, more directly scoreable than chasing extra dB of boost.
These three papers give concrete recipes (multi-task RL across corners trained jointly,
risk-sensitive/knowledge-infused reward shaping, batched Bayesian-optimization-style sizing
across corners) rather than training once at `corner="tt"` and hoping it generalizes.
**Not yet implemented or scoped in detail** — this doc is flagging it as the highest-leverage
unexplored direction, not proposing a specific architecture change yet.

### Positioning note (not a technique, a pitch angle)
**Source:** Astera Labs' own blog, "PCIe® Retimers vs. Redrivers: Ensuring Signal Integrity
for AI Infrastructure,"
https://www.asteralabs.com/resources/blog/pcie-retimers-vs-redrivers-ensuring-signal-integrity-for-ai-infrastructure/
— their own framing is "a Redriver amplifies blindly; a Retimer actively guarantees signal
integrity." This project's guard-layer-first methodology (never trust a boost/gain number
until every device is confirmed saturated with margin — see `src/eqrl/guards.py`) is the same
philosophy applied to circuit *design* rather than signal *transmission*. Usable as a framing
line in the presentation, not a technical claim requiring further validation.

### Also scraped, lower direct relevance (context only, not actionable leads)
- "A Multi-Stage CTLE Design and Optimization for PCIe Gen6," ITESO/Intel,
  https://rei.iteso.mx/bitstreams/168ba67a-ece1-45c1-90e3-3a6a6fe444a0/download — confirms
  multi-stage/multi-band CTLE is a real published technique, but **out of scope** here per
  the 1-stage constraint in §0. Useful only to confirm we correctly excluded it, not to copy.
- KATO, https://arxiv.org/html/2404.14433v1 (knowledge-alignment transfer for transistor
  sizing across design/technology) — tangential, cross-technology transfer isn't this
  project's problem (single PDK, single spec).
- A data-driven circuit synthesizer paper (DATE conference archive,
  https://past.date-conference.com/proceedings-archive/2024/DATA/444_pdf_upload.pdf) — general
  background on ML-driven topology synthesis, overlaps with Lead B above without adding
  anything new.

---

## 3. Suggested priority order for whoever executes this

~~1. Verify the monotonicity suite at VCM=0.68~~ — **closed, rejected.** See the correction
at the top of this doc: the underlying VCM effect was a measurement bug, not a real effect.
Keep `VCM_VDD_RATIO = 0.72`.

~~2. Implement and guard-test the wide-swing tail mirror (Lead A)~~ — **closed, rejected.**
Implemented, tested, does not clear the spec's own boost ceiling without paying for it
elsewhere (6x area, more power/noise, smaller eye). See
`docs/TAIL_MIRROR_PILOT_2026-09-10.md`.

**Remaining priority order, starting here:**

1. **Highest scoring leverage:** scope a PVT-robustness pass (Lead C) — this is graded
   directly in the spec and is the project's currently-weakest measured number (1 of 22
   candidate designs cleared the full corner sweep — see the handoff-corrections note in
   `docs/TAIL_MIRROR_PILOT_2026-09-10.md` distinguishing "candidates" from "corners": the
   *delivered* artifact separately passes 45/45 corners; this 1-of-22 figure is about
   candidate designs surveyed during search, not the shipped circuit's own corner coverage).
2. **Novelty/differentiation:** topology-variant-as-RL-action (Lead B) — strong
   differentiation story, not required for spec compliance, sequence after PVT work unless
   time is tight and a presentation-ready novelty angle is needed sooner.

All of the above must go through `build_evaluator(..., fast=False)` and report the physical-
validity fields (saturation/headroom/tail-delivery), matching the methodology in
`scratchpad/vcm_headroom_sweep.py`, before any boost/gain number from them is trusted or
reported.


## PVT continuation, 2026-09-10

The live pipeline now includes bounded PVT sizing repair and independent full-grid acceptance. A full 22-case paired repair run is in progress. Current results, exact budgets, scope and corrected triage are in [PVT_CORPUS_2026-09-10.md](PVT_CORPUS_2026-09-10.md). The earlier single-case result remains a historical case study.
