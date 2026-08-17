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

## Phase 1 — Measurement suite + Gym env ✅ DONE
**Goal:** a `EqualizerEnv.step(action)` returns a real observation + reward at TT.

- [x] `measures.py`: peaking, peak freq, DC gain, power, area (analytic). ✅
- [x] Behavioral gm swapped for real SKY130 device models, plus a current-mirror tail. ✅
- [x] HD3 (transient + FFT) and input-referred noise (`.noise`) simulated. ✅
      *(only when `fast=False`, which is now the default; `fast=True` substitutes
      constants for these two)*
- [x] `circuits/ctle.py`: parametric netlist generator from an action vector. ✅
- [x] `envs/equalizer_env.py`: Gymnasium env; reward = margins + all-pass bonus. ✅
- [x] Env passes `gymnasium.utils.env_checker`. ✅

## Phase 2 — RL that beats a sweep at TT  *(loop proven; no working policy yet)*
**Goal:** trained agent hits the TT spec, and does it in fewer sims than the baseline.

- [x] `baselines/sweep.py`: random + Bayesian (Optuna). ✅ (ran 80 real sims)
      CMA-ES is a third baseline, in `experiments/honest_benchmark.py`. There is no grid
      baseline; do not claim one.
- [x] `agents/train.py`: PPO (stable-baselines3) trains through the ngspice loop. ✅
- [ ] A training run that ends with a policy producing valid designs. The last completed
      40k-step guarded run did not. Two reasons, one measured and one structural: the
      action space is 10.4% valid (`results/space_validity.json`), and the penalty for a
      rejected design is a flat −5.0 by default, so every rejection looks identical to the
      agent and the value function is flat. `invalid_shaping=True` ranks "nearly valid"
      above "impossible" and was not in use for that run.
- [ ] **Headline plot:** best-spec-margin vs #SPICE-evals, RL vs baselines. Blocked on the
      above. The RL leg cannot be run at all until a checkpoint matches the current
      18-dimensional observation.
- [ ] Agent produces a sized schematic meeting TT specs.

## Phase 3 — Multi-objective + PVT robustness  *(the differentiator)*
**Goal:** meet all specs across all corners.

- [x] Corner runner: TT/SS/FF/SF/FS × VDD ±5% × {0, 27, 125} °C — all 45, with HD3 and
      noise simulated at each. ✅
- [x] Reward = worst-corner aggregate; pass-bonus when all specs pass all corners. ✅
- [ ] A design that holds across the grid. The one committed in `results/` fails 10 of 45
      (`results/legacy_design_recheck.json`, `all_pvt_pass: false`); the 10 failures do
      not solve at all rather than missing a limit.
- [ ] Delete or supersede `results/final_report.json`, which still asserts
      `all_pvt_pass: true` for a different design of the same generation — never rechecked
      on the corrected circuit, and produced by a checkpoint the current env cannot load.

## Phase 3b — Validation layer ✅ DONE  *(branch `guards/validation-layer`)*
**Goal:** no measurement reaches a reward without being checked against the operating point.

- [x] 20 checks in 5 tiers; `measure_all` sealed so the guarded path cannot be bypassed. ✅
- [x] Every rejection logged by check, not just counted — a bare invalid *rate* cannot
      distinguish a bad search space from a bad guard. ✅
- [x] Caught the tail mirror sitting in triode (326 µA delivered of 1000 µA requested),
      the probe reading a different design than the one under test, and the PDK's 100 µm
      width bin aborting with no exit code. All three fixed, all three with tests. ✅
- [x] Measured what the spec sheet lets through: 24 of 28 spec-passing designs are not
      valid circuits (`results/pass_vs_valid.json`). ✅

## Phase 4 — Eye + DFE + LLM front-end + deliverables  *(front-end done; demo blocked)*
**Goal:** the full "prompt → schematic" story and a clean submission.

- [x] `llm/spec_parser.py`: natural language → `Spec` object (Anthropic API, keyword
      fallback without a key). ✅
- [ ] `eqrl.solve` end-to-end demo. The parser half works; the half that sizes the circuit
      needs a policy that runs against the current env.
- [x] Real eye: minimum-phase PCIe channel + CTLE + adapted 1-tap DFE, Monte-Carlo. ✅
- [x] Export sized netlist + spec report + eye figure. ✅
- [ ] Report / slides: method, the guard-layer findings, PVT table, eye, demo video.

---

## Team split (3 people)
- **Analog lead** — testbench, measurements, PDK, corner setup (Phase 0/1/3). Hardest seat.
- **RL/ML lead** — env, agents, baselines, training curves (Phase 1/2).
- **Systems/LLM lead** — glue, corner orchestration, LLM wrapper, deliverables (Phase 3/4).

## Definition of a winning submission
A working Python framework where `run(spec)` → sized, PVT-robust equalizer schematic +
spec report, with a measured sample-efficiency comparison against the search baselines,
plus the LLM natural-language front-end demoed live. Every number in the submission
traceable to an artifact in `results/` that a judge can re-run.

The comparison has to be measured before it can be claimed. There is no measurement of
RL-versus-search sample efficiency in this repo right now, and any figure quoted for it —
including the "~10×" that used to sit in this file — was an expectation, not a result.
