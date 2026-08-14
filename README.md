# EqRL — RL-Driven Analog Equalizer Design

**Nebula @ BITS Goa 2026 · Analog Track · Astera Labs**
*AI/ML for Analog Circuit Design*

A reinforcement-learning framework that takes an equalizer spec sheet as input, talks
to a SPICE simulator in a closed loop, and sizes the devices to hit the target with
**zero human intervention** — in far fewer simulations than sweeping the parameter space.

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
target spec. The agent learns a *policy* — a strategy for reaching spec quickly — not
just one answer, so it retargets to new specs without starting over.

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

## Why we can win this

The judges (Astera Labs) are not asking us to invent RL-for-analog — that lineage exists
(AutoCkt, GCN-RL Circuit Designer, DNN-Opt). They're asking for a **working, sample-
efficient flow on a real open PDK that hits a hard spec across PVT.** That's an execution
win. Our three headline differentiators, each mapped to the problem statement's own words:

1. **Sample efficiency** — the ask is *"fewer search spaces, lowest design time."*
   Our headline result is a curve: **RL reaches spec in ~10× fewer SPICE evals than
   random / grid / Bayesian sweep.** (`src/eqrl/baselines/`)
2. **PVT robustness** — the spec demands 5 corners + VDD ±5% + 0–125 °C. Naive teams
   optimize at TT and break at SS/FF. Our reward penalizes worst-case-across-corners.
3. **LLM wrapper (the bonus)** — natural-language spec → framework config, plus
   LLM-assisted reward shaping and failure triage. (`src/eqrl/llm/`)

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
| `src/eqrl/baselines/` | Random / grid / Bayesian sweeps (the comparison baseline) |
| `src/eqrl/llm/` | Natural-language spec parser + reward-shaping helper |
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
python -m eqrl.agents.train --spec configs/pcie_gen2.yaml
```

## Status

Scaffold. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the phased plan and the current
milestone. **Phase 0 (get a CTLE simulating + measuring in ngspice) gates everything —
do not touch the RL code until that green.**

## References

See [`docs/REFERENCES.md`](docs/REFERENCES.md). Start with AutoCkt (Berkeley) — it is the
closest published analogue to what this competition is asking for.
