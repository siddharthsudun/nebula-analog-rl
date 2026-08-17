# EqRL — RL-Driven Analog Equalizer Design

**Nebula @ BITS Goa 2026 · Analog Track · Astera Labs**
*AI/ML for Analog Circuit Design*

A reinforcement-learning framework that takes an equalizer spec sheet as input, talks to
a SPICE simulator in a closed loop, sizes the devices, and checks the operating point of
every candidate before the measurement is allowed to count toward a reward.

Target circuit: **1-stage CTLE with source degeneration (variable Rs, Cs) + 1-tap DFE**,
for a **PCIe Gen 2 (5.0 Gbps)** receiver front-end.

---

## The problem, in one paragraph

Analog sizing is a slow, manual, expert-driven loop: tweak a transistor width, re-run
SPICE, read the plots, tweak again. The parameter space (every W/L, R, C, L) explodes
combinatorially, so brute-force sweeps are hopeless. This project replaces the human in
that loop with an RL agent: the **environment** is a SPICE testbench of the equalizer,
the **action** is a set of device sizes, the **observation** is the measured performance
(peaking, HD3, noise, power, area, eye), and the **reward** is how close we are to the
target spec. The aim of learning a *policy* rather than solving one instance is that a
policy can be pointed at a new spec without starting the search over. Whether this policy
does that is a measurement, and it is not one we currently have.

## Target specification (from the problem statement)

| Metric | Target |
|---|---|
| Signaling | NRZ, PCIe Gen 2, 5.0 Gbps (Nyquist 2.5 GHz) |
| HF peaking boost | 3–12 dB, tunable, peak between 1.25–2.5 GHz |
| Topology | 1-stage CTLE + source degeneration (var. Rs, Cs) + 1-tap DFE |
| Linearity | HD3 < −30 dB @ 100 MHz diff input |
| Input-referred noise | < 1.5 mV_rms (10 MHz – 5 GHz) |
| Power | < 15 mW |
| Area | < 0.05 mm² |
| Eye opening | > 0.4 UI (horizontal) & > 100 mV (vertical) |
| PVT | TT/SS/FF/SF/FS · VDD ±5% · 0–125 °C |
| PDK | SkyWater SKY130 (or IHP 130nm BiCMOS) |

Full spec + interpretation in [`docs/PROBLEM.md`](docs/PROBLEM.md).

## What is distinctive about this entry

The judges (Astera Labs) are not asking us to invent RL-for-analog — that lineage exists
(AutoCkt, GCN-RL Circuit Designer, DNN-Opt). They're asking for a **working flow on a real
open PDK that hits a hard spec across PVT.** That is an execution problem, and the part of
it we have executed is verification.

1. **The scoring is guarded** (`src/eqrl/guards.py`). Twenty checks in five tiers run
   against the operating point of the actual candidate, and `measure_all` is sealed so no
   number can reach a reward unvalidated. This is not decoration: **86% of the designs
   that pass all eight published specs are not valid circuits** — 24 of 28, measured
   twice with identical results (`results/pass_vs_valid.json`).
2. **The PVT grid is real and it is not being flattered.** 5 process corners × VDD ±5% ×
   {0, 27, 125} °C = 45 corners, with HD3 and input-referred noise simulated at each. The
   design currently committed in `results/` fails 10 of them
   (`results/legacy_design_recheck.json`).
3. **The simulator loop is fast enough to train on.** 77.5 ms per AC evaluation against
   6370.5 ms for a fresh ngspice subprocess — a measured 82.2× (`results/speedup.json`).
4. **LLM front-end (the bonus)** — natural-language spec → `Spec` object, with a keyword
   fallback when no API key is set. (`src/eqrl/llm/spec_parser.py`)

## Architecture

```
  natural-language spec ──(LLM wrapper)──▶ Spec object
                                             │
                                             ▼
     ┌──────────────────────────────────────────────────────┐
     │              EqualizerEnv  (Gymnasium)                │
     │                                                       │
     │   action  ─▶  device sizes (W/L, Rs, Cs, gm, DFE tap) │
     │                       │                               │
     │                       ▼                               │
     │        CTLE+DFE netlist  ──▶  ngspice / PySpice        │
     │                       │            (across PVT)        │
     │                       ▼                               │
     │        measures: peaking, HD3, noise, P, area, eye    │
     │                       │                               │
     │                       ▼                               │
     │        reward = −worst-corner distance-to-spec        │
     └──────────────────────────────────────────────────────┘
                                │
                                ▼
              RL agent (PPO / DDPG · stable-baselines3)
```

## Repo layout

| Path | What lives here |
|---|---|
| `docs/` | Problem statement, roadmap, references, design decisions |
| `src/eqrl/specs.py` | The `Spec` dataclass — target numbers as code |
| `src/eqrl/circuits/` | Parametric CTLE + DFE netlist generators |
| `src/eqrl/sim/` | ngspice/PySpice runner + measurement extraction |
| `src/eqrl/envs/` | Gymnasium environment wrapping the testbench |
| `src/eqrl/agents/` | RL training + evaluation scripts |
| `src/eqrl/guards.py` | The validation layer — 20 checks, 5 tiers, sealed measurement path |
| `src/eqrl/baselines/` | Random + Bayesian (Optuna) sweeps; CMA-ES lives in `experiments/honest_benchmark.py` |
| `src/eqrl/experiments/` | Measurement scripts — each writes its own artifact into `results/` |
| `src/eqrl/llm/` | Natural-language spec parser |
| `testbench/` | Raw SPICE testbenches (hand-written, for debugging) |

## Quickstart

```bash
# Phase 0 infra (do this FIRST — it is the real risk)
brew install ngspice
# install SKY130 PDK models (see docs/ROADMAP.md Phase 0)

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# sanity-check the simulator loop end to end
python -m eqrl.sim.ngspice_runner --selftest

# train
python -m eqrl.agents.train --algo ppo --timesteps 20000
```

## Status

Working, on real SKY130:

- `sky130_fd_pr` transistor CTLE with a current-mirror tail. AC peaking, HD3 (transient +
  FFT), input-referred noise (`.noise`), supply power, area and the channel + DFE eye are
  all simulated. HD3 and noise are only simulated when `fast=False`, which is now the
  default; with `fast=True` they are constants.
- Guard layer, sealed: `measure_all` cannot be called from a reward path without passing
  Tiers 1–4. Every rejection is logged by check, not just counted.
- Resident libngspice server: **77.5 ms** per AC evaluation against **6370.5 ms** for a
  fresh subprocess, a measured **82.2×** (`results/speedup.json`). The gain is throughput
  per candidate: the SKY130 corner `.lib` parses once instead of every launch.
- Full 45-corner PVT engine, HD3 and noise at every corner.
- Test suite: **274 passed, 3 xfailed.**

Not working, stated here because it changes how the rest of the repo reads:

- **No trained policy runs against the current code.** Every checkpoint in `results/`
  expects a 14-dimensional observation; the environment emits 18. The circuit changed
  underneath them as well — the tail mirror was found in triode, delivering 326 µA of a
  requested 1000 µA, and both the bias point and the mirror length were changed to fix it.
  The last completed guarded run did not produce a policy that yields a valid design.
- **The declared action space is 10.4% physically valid** — 26 of 250 uniform samples
  (`results/space_validity.json`). Validity survives up to roughly 1 mA of tail current
  and is zero across 90 samples above it, so most of the declared 0.05–20 mA range is
  dead space (`results/itail_profile.json`).
- **The design committed in `results/solved_design.json` fails 10 of 45 PVT corners**
  (`results/legacy_design_recheck.json`). `results/final_report.json` still asserts
  `all_pvt_pass: true` for another design of the same generation; it is stale, predates the
  mirror fix, and has not been rechecked.
- **`area` cannot fail as a constraint.** The worst design anywhere in the action space is
  0.0113 mm² against a 0.05 mm² budget. It is measured, but it is not a live constraint.

**See [`RESULTS.md`](RESULTS.md) for the measurements.** Roadmap and per-phase status in
[`docs/ROADMAP.md`](docs/ROADMAP.md); `HANDOVER.md` is the working log.

## References

See [`docs/REFERENCES.md`](docs/REFERENCES.md). Start with AutoCkt (Berkeley) — it is the
closest published analogue to what this competition is asking for.
