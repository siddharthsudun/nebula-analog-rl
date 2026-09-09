# Where the entry actually stands — 07 Sep 2026

**Final submission: 15 Sep.** Eight days. Architecture frozen 26 Aug; everything below
that changes behaviour is either inference-time or a separately-named arm.

This is a status list, not a pitch. Anything not measured is marked as not measured.

---

## 1. Done, and holds up

| | evidence |
|---|---|
| ngspice 47 + real SKY130 device models; parametric CTLE netlist from an action vector | `testbench/`, `src/eqrl/circuits/ctle.py` |
| Full measurement suite — AC boost/peak, DC gain, HD3 (transient+FFT), input-referred noise (`.noise`), power, area | `src/eqrl/measures.py` |
| Gym env + PPO through the real simulator; a frozen working policy at 40,960 steps | `results/seq_clean40k.zip` |
| Guard layer: 20 checks in 5 tiers, `measure_all` sealed so nothing reaches a reward unguarded, every rejection logged **by check** | `src/eqrl/guards.py` |
| 45-corner PVT sign-off runner (5 process × 3 V × 3 T), HD3 and noise simulated at each | `results/pvt_signoff_seed23.json` |
| Real eye: minimum-phase PCIe channel + CTLE + adapted 1-tap DFE, Monte-Carlo | `src/eqrl/eye.py` |
| NL → `Spec` parser, with a keyword fallback when no API key is set | `src/eqrl/llm/spec_parser.py` |
| Web dashboard; delivered netlist carries its own verification status in its header | `server.py`, `dashboard/` |
| Five inference modes, plus `g32_acceptance` as a named experimental arm | `src/eqrl/pipeline.py` |
| 541 tests green, including ratchets that check prose numbers against the code they cite | `tests/` |

### The two results that are genuinely ours

1. **The published spec set does not determine a circuit.** Of 28 designs passing all
   eight published checks, 24 were not valid circuits — 36% by a criterion nobody can
   dispute (transistor out of saturation), 86% under our full guard set. The spec set
   cannot tell an amplifier from an attenuator. `docs/PASS_VS_VALID.md`.
2. **Chance-baseline discipline.** Every solve rate in this repo carries a budget-matched
   chance line. That is not standard in this literature — of 16 RL-for-analog-sizing
   papers surveyed today, **none** reports a budget-matched random baseline.

---

## 2. Measured, and bad — state these before a judge does

* **`thinking` sits AT its chance line.** 31-spec sweep, per-spec budget-matched:

  | mode | solved | med wall | sims md/mean | chance | vs chance |
  |---|---|---|---|---|---|
  | default | 18/31 | 86.8 s | 14 / 16.0 | 29.8 | **−11.8** |
  | fastest | 24/31 | 14.8 s | 2 / 2.5 | 18.1 | **+5.9** |
  | thinking | 30/31 | 127.9 s | 32 / 40.7 | 30.5 | **−0.5** |
  | retarget | 18/31 | 84.3 s | 14 / 16.0 | 29.8 | **−11.8** |
  | auto | 30/31 | 16.8 s | 2 / 15.7 | 20.3 | **+9.7** |

  30/31 looks like the headline. It is not: at 40 simulations, spec-blind drawing from
  our own achieved-boost distribution reaches 30.5. **The margin belongs to `fastest` and
  to `auto`, not to the mode with the biggest number.** `default` and `retarget` are
  ~12 designs *below* chance — they are actively worse than not using the spec.

* **Negative-margin regime: weak, and NOT 0%.** `docs/RESULTS_MULTISEED_AUDIT.md` already
  retracted the 0% figure: on all eight runs it is 40% (2 of 5 jobs) against 57% (20 of 35)
  for positive margin — **not statistically distinguishable at n=5**, worth at most ~3 jobs
  in 40. Independently confirmed today on spec seed 137, which has three negative-margin
  specs (14, 17, 24 at -2.31, -1.21, -0.13 dB): `default`, `thinking`, `retarget` and
  `auto` each solve **2 of 3**. The one mode that goes 0 of 3 is `fastest` — so whatever
  is there looks mode-specific, not a property of the frozen policy.
  Do not quote 0%; it is a stale figure this repo has already withdrawn once.

* **The DC-gain floor is not what limits high boost.** Measured over all 374,588 corpus
  designs: with `dc_gain >= 0 dB` enforced, 251,391 designs survive and the maximum
  achieved boost is **19.17 dB** — the whole published 3-12 dB requirement sits inside
  that. The floor does thin the population as boost rises (76.5% of designs at 0-3 dB
  boost, 40.2% at 11-12 dB, 3.0% above 15 dB), but it forecloses nothing we are asked for.
  Worth knowing because IEEE 802.3 Annex 93A / PCIe reference CTLE models let DC gain go
  as low as **-15 dB**, so our 0 dB floor is stricter than standards practice and a judge
  may raise it. It is defensible as a single-stage-with-no-downstream-VGA choice; it is
  not defensible as "the standard definition".

* **PVT: ~4% of designs are 45/45 clean.** The G3.2 refinement stage does **not** improve
  corner robustness — a hypothesis we held and disproved.

* **Coverage gaps in every training config.** All nine have `pvt: False` and
  `target_range: [5.0, 11.0]` against a published 3–12 dB requirement. The 32-spec
  benchmark draws 5.01–10.99 dB, so it is in-distribution by construction.

* **No run in this project has ever had a reward curve.** `train_sequential.py` builds
  `SubprocVecEnv` with no `Monitor` wrapper, so SB3 logs zero `rollout/ep_rew_mean`.
  The only live signal overwrites its own JSON every 500 steps.

* **`retarget` fires 0/32 on a fresh seed.** An earlier n=8 win was a selection effect.

* **`seq_robust40k` stopped at 23k/40k steps.** The current trainer accepts
  `--resume <checkpoint.zip>` and loads it with `PPO.load(..., env=env)`, preserving
  `model.num_timesteps`; this corrects the stale claim that resumption is unavailable.
  `results/checkpoints/SEQ_ROBUST40K_INCOMPLETE.md` records the incomplete run, not a
  successful resumption.

* **Withdrawn:** the gap-3 correlation finding (+0.255 → +0.741) was cross-benchmark and
  is dead. On the matched holdout the frozen policy wins every metric and its correlation
  is **+0.739**. The two coverage gaps stand; the fix proposed for them does not.

---

## 3. To do, ranked by what changes the submission

### Must ship (blocks the deliverable)
1. **Finish the 32nd spec** of the mode sweep — `thinking`/`retarget`/`auto` only. The
   sweep died at 18:24 today mid-spec; 31/32 is salvaged and the conclusions are stable.
2. **The headline plot that does not exist**: best-spec-margin vs #SPICE-evals, RL vs
   CMA-ES / random / Optuna. The roadmap has promised this since Phase 2 and there is
   still **no measured RL-versus-search sample-efficiency number in this repo.** Any
   figure quoted for it would be an expectation.
3. **Report / slides**: method, the spec-set finding, the PVT table, the eye, and the
   chance-line discipline as a named contribution.
4. **Do not use `results/final_report.json` as a PVT source.** It has no
   `all_pvt_pass` field. The separate `results/solved_design_pvt.json → all_pvt_pass`
   artifact is a different pre-mirror-fix design; its schema is constructed by
   `src/eqrl/experiments/characterize.py::characterize`, whose CLI writes
   `<outdir>/final_report.json` without recording the invocation in that historical file.

### High value, inference-time only (respects the freeze)
5. **Cheap feasibility pre-filter.** MOSFET saturation and DC-gain sign are close to
   analytic in the action vector; ~42% of steps are currently spent discovering
   infeasibility *by simulation*. Highest leverage per hour of work in the whole list,
   and the survey found no analog-RL paper that even measures this.
6. **Corner screening**: find whether a 3-corner subset predicts the 45. If it does, PVT
   robustness becomes affordable inside the loop instead of a post-hoc sweep.
7. **Negative-margin hedge** — inference-time, untested.
8. **Guard-validity classifier** trained on the accumulated reject/accept history.
9. **Preregistered tightened comparison + diversify-vs-rank ablation**; `g32_acceptance`
   earns its README line only after that lands, whichever way it lands.

### Hygiene
10. `Monitor` wrapper so the next run has a reward curve at all; make the invalid-logger
    append instead of overwrite.
11. `--resume` for `train_sequential.py`.
12. `clean_requirements` silently drops `None` values.
13. Decide on `git rm -r --cached results/raw` — 15,012 tracked files defeat the ignore
    rule and every `git status` walks ~2.36 M files.

---

## 4. What the freeze forbids

No reward change, no new observation slot, no guard threshold moved, no recorded number
touched. Parser, CLI, UI, docs, CI, scripts and tests are all fair game. Every item in
§3 is written to respect that.

---

## 5. What the literature says we should steal — four surveys, 60+ papers

Surveyed 07 Sep 2026 across RL-for-analog-sizing, CTLE/SerDes equalization, surrogate/BO
and constraint handling, and PVT/yield. Ranked by value under the freeze.

### 5.1 Three findings that defend numbers we already have

1. **Our ~4% corner-clean rate is normal, not catastrophic.** PPAAS (arXiv:2507.17003,
   ICCAD 2025) evaluates an **AutoCkt baseline trained nominal-only against the same
   45-corner grid we use** and reports **0.0%, 3.2%, 0.0%, 5.2%** across four SKY130 /
   GF180 benchmarks. That is the expected failure mode of nominal-only training, not a
   defect peculiar to us. This is the single most useful citation the survey returned —
   it converts our worst-looking number into a characterisation of the paradigm.
2. **"Refinement does not improve robustness" is predicted, and nobody has published the
   ablation.** Design-centering theory (Antreich/Graeb/Wieser, TCAD 1994) says a stage
   that chases nominal tightness is doing the *opposite* of centering, which trades
   nominal tightness for margin. No paper found ran the controlled with/without ablation.
   Ours is narrow but genuinely unpublished.
3. **The chance-line discipline is not standard.** Across the RL-for-sizing papers
   surveyed, **none reports a budget-matched random baseline**; the BO literature does
   treat matched-budget random search as mandatory, and names the pathology where a
   learned method's trace is indistinguishable from random sampling for most of its
   budget. That is exactly `thinking` at its chance line. Worth stating as a contribution.

### 5.2 Adoptable at inference time — respects the freeze

| # | technique | source | why it fits |
|---|---|---|---|
| 1 | **μ-σ evaluation + failure-likelihood simulation reordering** — estimate pass probability from a few corners chosen by where failure is most likely, pay for the rest only when inconclusive | GLOVA, DAC 2025 (arXiv:2505.11208) | our two failure modes (device-guard vs spec-drift) are already separately characterised, which is exactly the input this needs |
| 2 | **Worst-case distance as a *ranking* key** — score candidates by distance-to-spec-boundary normalised by cross-corner sensitivity, pick the widest margin rather than the best nominal | Antreich/Graeb/Wieser TCAD 1994; Graeb 2007 | pure selection among candidates the frozen policy already emits |
| 3 | **SCBO** — trust-region constrained BO, region centred on the best feasible point (or least-violating point when none is feasible) | Eriksson & Poloczek, AISTATS 2021; BoTorch has it | drop-in replacement for the current local search; guard layer feeds the constraint GP |
| 4 | **Multi-fidelity GP fusion** — separate GPs for cheap and expensive fidelity plus a third for the map between them | Zhang et al., DAC 2019 (arXiv:1912.00392) | we already have exactly this split (cheap AC vs full transient) and currently allocate it by a flat budget |
| 5 | **cVAE spec → design *portfolio*** — sample a diverse candidate set conditioned on the full spec vector | arXiv:2510.05160 | trainable on the existing 374,588-design corpus with **zero new simulation**; strictly more than nearest-neighbour-on-boost |

### 5.3 Corrections the survey forced on our own to-do list

* **Item 6 ("find a 3-corner subset that predicts the 45") is the wrong shape.** No
  published result establishes a fixed predictive corner subset for analog sizing, and
  DATE 2023 (doc 10136993) reports corner-subset selection is **NP-hard** — because no
  single corner is extremal for every spec at once. Our own measured device-guard/spec-drift
  mode flip is a concrete instance of exactly that. Worse, RobustAnalog's static clustering
  is reported at **0% on 2 of 4 benchmarks** in PPAAS's evaluation. Rewrite the item as a
  *candidate-conditioned ranking* (5.2 #1), never a fixed subset.
* **The corpus has no guard labels.** `surrogate_corpus.npz` is `X` (6 params) and `Y`
  (`dc_gain_db`, `boost_db`, `peak_freq_ghz`) only. A feasibility classifier needs labels
  we would have to generate. The existing analytic check in `space_validity.py`
  (`(i_tail/2)·r_load <= VDD-0.5`) passes **85%** of the box, while the dominant rejection
  is saturation at **42.8%** — so the analytic filter we already have is nowhere near
  sufficient, and a real one needs a square-law overdrive estimate.

### 5.4 Post-freeze only — for the arm after 15 Sep

* **PPAAS's staged corner reward**: evaluate nominal first, escalate to full corners only
  once nominal is nearly satisfied. The affordable way to put PVT into a reward.
* **gm/ID electrical design space** instead of raw W/L (MLCAD 2022): reported to improve
  RL convergence, and it encodes the inversion region — attacking the 42% invalid rate at
  its source rather than filtering afterwards.
* **Goal-conditioned observation** (target and achieved margins as separate vectors) and
  **per-spec vector reward** kept un-scalarised through the critic (ORACLE preprint) — the
  surgical candidates for the policy's weak use of the target.

### 5.5 Open question the literature did not settle

Whether "requested boost > channel loss" is well-posed. No source addresses the framing.
What is established: boost difficulty is set by **gm·Rs** and does not depend on the
channel-loss number at all, so the *pairing* does not change circuit difficulty — only the
absolute boost value does. Over-equalization is a real, named, survivable regime. Combined
with §2's measurement that four of five modes solve 2 of 3 negative-margin specs, the
"blind spot" framing looks weaker than the phrase suggests.
