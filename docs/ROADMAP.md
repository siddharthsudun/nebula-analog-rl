# Roadmap

Milestones ordered by risk. **Do not move to a later phase until the earlier one is
green.** The analog testbench is the real risk, not the RL — so it goes first.

Deadlines (from poster): Abstract 6 Aug · Shortlist 12 Aug · **Final submission 15 Sep**
· Final shortlist 19 Sep · **Presentations 25 Sep**.

---

## Phase 0 — Infra & simulator loop  *(gates everything)* ✅ DONE
**Goal:** a hand-written CTLE netlist simulates in ngspice and we can read numbers back
from Python.

- [x] `brew install ngspice` — ngspice 47 installed.
- [ ] Install SKY130 SPICE models (open_pdks / volare) OR IHP sg13g2. Record path.
      *(still behavioral; real PDK models come in Phase 1)*
- [x] Hand-write a CTLE testbench in `testbench/ctle_ac.spice`, run it, see a peaking curve.
- [x] `eqrl.sim.ngspice_runner --selftest` runs the netlist and parses AC output.
- [x] Extract peak gain in dB from Python.

**Exit criterion MET:** boost responds correctly to Cs (50fF→3.0dB, 200fF→11dB,
800fF→15dB, with peak freq dropping as Cs rises — matches source-degeneration theory).

## Phase 1 — Measurement suite + Gym env  *(mostly done; PDK + HD3/noise/eye remain)*
**Goal:** a `EqualizerEnv.step(action)` returns a real observation + reward at TT.

- [x] `measures.py`: peaking, peak freq, DC gain, power, area (analytic). ✅
- [ ] **Swap behavioral gm for real SKY130 device models** (biggest remaining analog task).
- [ ] Fill in HD3 (transient + FFT) and input-referred noise (`.noise`) — currently stubbed.
- [x] `circuits/ctle.py`: parametric netlist generator from an action vector. ✅
- [x] `envs/equalizer_env.py`: Gymnasium env; reward = margins + all-pass bonus. ✅
- [x] Env passes `gymnasium.utils.env_checker`. ✅

## Phase 2 — RL that beats a sweep at TT  *(loop proven; needs real training runs)*
**Goal:** trained agent hits the TT spec, and does it in fewer sims than the baseline.

- [x] `baselines/sweep.py`: random + Bayesian (Optuna). ✅ (ran 80 real sims)
- [x] `agents/train.py`: PPO (stable-baselines3) trains through the ngspice loop. ✅
- [ ] Long training run until the agent passes TT spec (400-step smoke test done, not enough).
- [ ] **Headline plot:** best-spec-margin vs #SPICE-evals, RL vs baselines.
- [ ] Agent produces a sized schematic meeting TT specs.

## Phase 3 — Multi-objective + PVT robustness  *(the differentiator)*
**Goal:** meet all specs across all corners.

- [ ] Corner runner: TT/SS/FF/SF/FS × VDD ±5% × {0, 27, 125} °C.
- [ ] Reward = worst-corner aggregate; pass-bonus when all specs pass all corners.
- [ ] Show a naive-TT design breaking at SS/FF, then the PVT-aware agent holding.

## Phase 4 — LLM wrapper (bonus) + eye diagram + deliverables
**Goal:** the full "prompt → schematic" story and a clean submission.

- [ ] `llm/spec_parser.py`: natural language → `Spec` (Anthropic API, Claude).
- [ ] LLM-assisted reward shaping / failure triage ("why did SS fail?").
- [ ] PRBS eye-diagram generator (height V, width UI) through a channel model.
- [ ] Export final schematic (netlist + optionally a drawn schematic) + spec report.
- [ ] Report / slides: method, sample-efficiency curve, PVT table, eye, demo video.

---

## Team split (3 people)
- **Analog lead** — testbench, measurements, PDK, corner setup (Phase 0/1/3). Hardest seat.
- **RL/ML lead** — env, agents, baselines, training curves (Phase 1/2).
- **Systems/LLM lead** — glue, corner orchestration, LLM wrapper, deliverables (Phase 3/4).

## Definition of a winning submission
A working Python framework where `run(spec)` → sized, PVT-robust equalizer schematic +
spec report, with a graph proving it's ~10× more sample-efficient than a sweep, plus the
LLM natural-language front-end demoed live.
