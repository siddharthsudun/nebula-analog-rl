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

## PVT

Temperature is modeled correctly (drain current 42→125 µA over 0→125 °C; boost
8.8→7.4 dB). The **nominal-trained** design passes all 5 process corners + VDD±5% at
27 °C, but drifts out of spec at 125 °C (boost 7.4 dB, peak 1.19 GHz) — the exact PVT
trap the poster highlights. The **PVT-aware agent** (`--pvt`, reward = worst of
{nominal, hot/low-V, cold/high-V}) is trained to hold across V×T; final sign-off across
the full 45-corner grid is produced by `characterize.py` (`results/final_report.json`).

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
