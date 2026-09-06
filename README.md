# EqRL: RL-Driven Analog Equalizer Design

**Nebula @ BITS Goa 2026 · Analog Track · Astera Labs**
*AI/ML for Analog Circuit Design*

A reinforcement-learning framework that takes an equalizer spec sheet as input, talks to
a SPICE simulator in a closed loop, sizes the devices, and checks the operating point of
every candidate before the measurement is allowed to count toward a reward.

Target circuit: **1-stage CTLE with source degeneration (variable Rs, Cs) + 1-tap DFE**,
for a **PCIe Gen 2 (5.0 Gbps)** receiver front-end.

## Run silQ (the product)

silQ is the front end of this repo: type the equalizer spec the way you would say it to a
colleague, and watch a verified SKY130 CTLE appear, redrawn for every simulated candidate.

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m uvicorn server:app --port 8000
# open http://127.0.0.1:8000
```

What the page does, in order: a keyword reader parses the request on every keystroke and
Claude (via the `claude` CLI or `ANTHROPIC_API_KEY`, optional) reads it on demand or right
before a run, with every disagreement shown; the frozen PPO policy plus the G3.2 refinement
stage size the circuit; an independent re-simulation scores it against the ten hard checks
and against any limits you tightened or loosened. Modes: **Auto** (fast first, Thinking
only if verification fails), **Fastest**, **Thinking**. `?nointro=1` skips the intro.

---

## The problem, in one paragraph

Analog sizing is a slow, manual, expert-driven loop: tweak a transistor width, re-run
SPICE, read the plots, tweak again. The parameter space (every W/L, R, C, L) explodes
combinatorially, so brute-force sweeps are hopeless. This project replaces the human in
that loop with an RL agent: the **environment** is a SPICE testbench of the equalizer,
the **action** is a set of device sizes, the **observation** is the measured performance
(peaking, HD3, noise, power, area, eye), and the **reward** is how close we are to the
target spec. The aim of learning a *policy* rather than solving one instance is that a
policy can be pointed at a new spec without starting the search over. We measured whether
it does: the policy retargets to a new feasible design in a median of 4 evaluations, but
hitting the *requested* boost precisely is done by a constraint-aware refinement stage
downstream, not by the policy alone. Both the win and its boundary are reported below.

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

The judges (Astera Labs) are not asking us to invent RL-for-analog. That lineage exists
(AutoCkt, GCN-RL Circuit Designer, DNN-Opt). They're asking for a **working flow on a real
open PDK that hits a hard spec across PVT.** That is an execution problem, and the part of
it we have executed is verification.

1. **The scoring is guarded** (`src/eqrl/guards.py`). Twenty checks in five tiers run
   against the operating point of the actual candidate, and `measure_all` is sealed so no
   number can reach a reward unvalidated. This is not decoration: **86% of the designs
   that pass all eight published specs are not valid circuits**: 24 of 28, measured
   twice with identical results (`results/pass_vs_valid.json`).
2. **The delivered circuit passes all 45 PVT corners, and it is not being flattered.** 5
   process corners × VDD ±5% × {0, 27, 125} °C = 45 corners, with HD3 and input-referred
   noise simulated at each, through the full guard layer and scored against the requested
   target (`results/delivered_circuit.json`). It was not hand-picked: all 22 strict-passing
   held-out candidates were swept under a rule written before any corner ran, and **1 of 22
   came back clean** (`docs/REPRODUCE.md` §21). The optimisation ran at TT only, so the
   corner result is an out-of-distribution measurement of one generated candidate, reported
   as such.
3. **The simulator loop is fast enough to train on.** 77.5 ms per AC evaluation against
   6370.5 ms for a fresh ngspice subprocess, a measured 82.2× (`results/speedup.json`).
4. **LLM front-end (the bonus)**: natural-language spec → `Spec` object, with a keyword
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
| `src/eqrl/specs.py` | The `Spec` dataclass, target numbers as code |
| `src/eqrl/circuits/` | Parametric CTLE + DFE netlist generators |
| `src/eqrl/sim/` | ngspice/PySpice runner + measurement extraction |
| `src/eqrl/envs/` | Gymnasium environment wrapping the testbench |
| `src/eqrl/agents/` | RL training + evaluation scripts |
| `src/eqrl/guards.py` | The validation layer: 20 checks, 5 tiers, sealed measurement path |
| `src/eqrl/baselines/` | Random + Bayesian (Optuna) sweeps; CMA-ES lives in `experiments/honest_benchmark.py` |
| `src/eqrl/experiments/` | Measurement scripts; each writes its own artifact into `results/` |
| `src/eqrl/llm/` | Natural-language spec parser |
| `testbench/` | Raw SPICE testbenches (hand-written, for debugging) |

## Quickstart

```bash
# Phase 0 infra (do this FIRST, it is the real risk)
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

**Delivered architecture (frozen 26 Aug 2026, `docs/REPRODUCE.md` §20–21):**

```
spec  →  PPO global feasibility search  →  G3.2 constrained target refinement
      →  independent guarded SPICE verification  →  sized netlist + measured specs
```

- **PPO learns the feasible design space.** The frozen policy `results/seq_clean40k.zip`
  reaches a valid circuit in a median of **4 evaluations**, against **2,394** for the full
  parameter sweep the poster names as the baseline. It supplies feasibility, not sizing
  precision; on its own it does not hit a *requested* boost above a matched-chance null,
  and that is reported as the boundary of the RL claim, not hidden.
- **G3.2 closes the requested spec.** A constraint-aware numerical stage that refines the
  PPO handoff. On a 40-spec held-out set it cut median target error from **1.886 dB
  (PPO→CMA-ES baseline) to 0.231 dB** at roughly half the refinement budget, and the
  coverage explanation that killed the earlier retargeting result runs the wrong way here
  (`docs/REPRODUCE.md` §20.2). **Strict solve counts do not clear their matched chance line
  for any arm and are not claimed.**
- **The delivered circuit passes all 45 PVT corners** (`results/delivered_circuit.json`):
  target 8.920 dB over a 14.83 dB channel, worst-corner error 1.081 dB, DC gain the binding
  constraint. 1 of 22 held-out candidates was PVT-clean; optimisation ran at TT only.

**Infrastructure, on real SKY130:**

- Guard layer, sealed: 20 checks in 5 tiers (incl. anti-gaming Tier 5); `measure_all`
  cannot reach a reward path without passing Tiers 1–4. Every rejection logged by check.
- Resident libngspice server: **77.5 ms** per AC evaluation vs **6370.5 ms** for a fresh
  subprocess, a measured **82.2×** (`results/speedup.json`).
- Full 45-corner PVT engine, HD3 and noise at every corner. HD3/noise simulated when
  `fast=False`, which is the default.
- **86% of the designs that pass all eight published specs are not valid circuits**: 24 of
  28, measured twice (`results/pass_vs_valid.json`). The guard is where that was found.
- **The declared action space is 20.4% physically valid**: 51 of 250 uniform samples
  (`results/space_validity.json`). The tail-current range was capped at 1 mA on measured
  physics (nothing valid above it across 90 samples); no other range was narrowed.
- `area` cannot fail as a constraint: worst design anywhere is 0.0113 mm² against a
  0.05 mm² budget. Measured, not live.

**How the record reads.** This repo kept its own retracted results in the git history
rather than deleting them: an early 32/32 retargeting number was selection bias, an early
reward was gamed, and a surrogate arm was cut on a pre-registered gate. The authoritative
account of the delivered system is `docs/REPRODUCE.md`; `RESULTS.md` carries the standalone
measurements and `HANDOVER.md` is the mid-project working log.

## References

See [`docs/REFERENCES.md`](docs/REFERENCES.md). Start with AutoCkt (Berkeley); it is the
closest published analogue to what this competition is asking for.
