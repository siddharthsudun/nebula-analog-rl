# Results

Everything below is measured through **real ngspice on the open SKY130 PDK** with
`sky130_fd_pr__nfet_01v8` transistors — not behavioral models, not hand-waving.

## Headline: RL generalizes across specs, ~7× fewer simulations than search

A single sequential RL policy was trained across randomized target-boost specs. It is then
asked to hit **held-out** target specs it never trained on. For each new spec we count the
SPICE simulations needed to reach a design that meets the spec, and compare to Bayesian
optimization (Optuna TPE) run from scratch for each spec.

| target boost | RL sims-to-spec | Bayesian sims-to-spec |
|---:|---:|---:|
| 5.0 dB | 4 | 34 |
| 5.9 dB | 5 | 22 |
| 6.7 dB | 4 | 25 |
| 7.6 dB | 2 | 16 |
| 8.4 dB | 3 | 43 |
| 9.3 dB | 5 | 5 |
| 10.1 dB | 2 | 32 |
| 11.0 dB | 3 | **did not converge (120)** |

- **RL median: 3.5 sims/spec.  Bayesian median: 25 sims/spec (~7× more).**
- **RL solved 8/8 targets; Bayesian solved 7/8.**
- The RL cost is amortized: one training run, then *every* new spec is nearly free.
  A search has no memory — it pays a full search for every spec.

Plot: `results/generalization.png`. This directly answers the problem statement's ask —
"reach near-optimal solutions with fewer search ... zero human intervention."

## The circuit the agent designs

1-stage source-degenerated CTLE (nfet_01v8 differential pair, Rs‖Cs degeneration,
resistive loads) for a PCIe Gen2 (5 Gb/s) receiver. A design the agent produced for the
9 dB target, characterized at the nominal corner:

| metric | measured | spec | |
|---|---:|---|:--:|
| HF boost | 8.4 dB | 3–12 dB | ✓ |
| peak freq | 1.26 GHz | 1.25–2.5 GHz | ✓ |
| HD3 @100 MHz | −52.6 dB | < −30 dB | ✓ |
| input noise (10 MHz–5 GHz) | 780 µVrms | < 1.5 mVrms | ✓ |
| power | 0.18 mW | < 15 mW | ✓ |
| area (est.) | 0.002 mm² | < 0.05 mm² | ✓ |

## PVT: passes the full 45-corner grid

An RL-designed CTLE was signed off across the **complete PVT grid — 5 process corners
(TT/SS/FF/SF/FS) × 3 voltages (VDD ±5%) × 3 temperatures (0/27/125 °C) = 45 corners** —
with real HD3 and noise at every corner. **All 45 pass** the poster's hard specs:

| metric | across all 45 corners | hard spec |
|---|---|---|
| HF boost | 8.9 – 10.9 dB | 3–12 dB |
| peak freq | 1.33 – 1.50 GHz | 1.25–2.5 GHz |
| HD3 | −48 to −51 dB | < −30 dB |
| input noise | 570 – 956 µVrms | < 1.5 mVrms |
| power | 0.17 – 0.19 mW | < 15 mW |
| area | 0.002 mm² | < 0.05 mm² |
| eye height | 0.50 – 0.56 UI | > 0.4 UI |
| eye opening | 172 – 199 mV | > 100 mV |

Full table: `results/final_report.json`; sized netlist: `results/final_schematic.spice`.

## Eye diagram: the equalizer actually opens the eye

The eye is measured, not assumed. The SPICE-extracted **complex** CTLE response is put in
series with a minimum-phase **PCIe-Gen2 channel** (skin + dielectric loss, 12 dB at
Nyquist) and a real **adapted 1-tap DFE**, then a random NRZ pattern is run through it and
folded (Monte-Carlo). At 5 Gb/s the raw channel eye is nearly shut; SILQ's CTLE + DFE
reopens it:

| link | eye height | eye width |
|---|---|---|
| channel only | 69 mV | 0.31 UI |
| **+ CTLE + 1-tap DFE** | **192 mV** | **0.50 UI** |

Figure: `results/eye.png` (closed vs open). The 1-tap DFE is adapted to cancel the first
post-cursor and demonstrably helps on its own.

## Describe it in English → get a circuit

`python -m eqrl.solve "PCIe Gen2 CTLE with about 8 dB of peaking, under 10 mW"` parses the
request (LLM, with a keyword fallback), has the trained policy size a CTLE in ~5 sims, and
prints the sized, fully-characterized circuit (incl. the real eye) — the bonus deliverable.

Temperature is modeled correctly (drain current 42→125 µA over 0→125 °C). Getting here
was deliberate: a *naive* nominal-only design drifts out of band at 125 °C (peak 1.19 GHz)
— the exact PVT trap the poster highlights. Two mechanisms produce robust designs:
1. **Spec-margin query** — the trained policy is asked for a design with headroom, then
   verified across the grid (the result above).
2. **PVT-aware training** (`--pvt`) — reward = worst of {nominal, hot/low-V} so the policy
   optimizes worst-case directly (implemented; use when you want the margin chosen
   automatically rather than by query).

## Speed

The enabling engineering: a **resident libngspice server**. Including the SKY130 corner
`.lib` costs ~15 s of model parsing per process launch; a simulation is ~40 ms. By keeping
ngspice resident and re-evaluating candidates with `alterparam; reset`, each AC eval is
**~44 ms** (full HD3+noise eval ~150 ms) — a measured **~350×** speedup that makes RL
training on real SPICE practical (tens of thousands of sims in minutes).

## Reproduce

```bash
# headline comparison (needs a trained agent)
python -m eqrl.experiments.generalization --model results/seq_agent_nominal.zip --targets 8
# full PVT sign-off of a design
python -m eqrl.experiments.characterize --design results/rl_design.json
```
See `SETUP.md` for environment setup and `docs/ROADMAP.md` for status.
