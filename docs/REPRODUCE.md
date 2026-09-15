# Reproducing the final benchmark

Everything the submission claims traces to one checkpoint, `results/seq_clean40k.zip`, and
to a small set of scripts run against it. This file records exactly what produced it and
exactly how to re-derive every number. Audited 23 Aug 2026; every result below was re-run
from scratch and compared against the original artifact.

## 1. The checkpoint

`results/seq_clean40k.zip` — the model behind every current RL number.

| | |
|---|---|
| trained | 21 Aug 19:18 → 22 Aug 02:24 (425 min wall) |
| steps | 40,960 (SB3 rounds 40,000 up to a multiple of `n_steps`) |
| restarts | 1, absorbed by `scripts/train_supervised.py`; 0 stalls |
| seed | 0 |
| n_envs | 8 (`SubprocVecEnv` — libngspice is a process singleton, so parallelism needs processes) |
| horizon | 20 |
| step_size | 0.18 |
| target_range | (5.0, 11.0) |
| channel_range | (8.0, 16.0) |
| fast | **true** (HD3 and noise stubbed during training) |
| guarded | **true** |
| shaped_invalid | false |
| anchor_baseline | false |
| feasible_decode | false |
| simulations | 44,984, of which 20,284 invalid (45.1%) |

Config as written by the trainer: `results/seq_clean40k_train.json`.
Supervisor log: `results/supervisor_clean.log`. Full training log:
`results/seq_clean40k_supervised.log`.

## 2. The environment

```
git           2f3ec52b3c93a598bc0448c6caf019cf5076dd2d
python        3.12.10
numpy         2.5.2
torch         2.13.0+cpu
stable-baselines3  2.9.0
gymnasium     1.3.0
ngspice       41 (libngspice, resident server)
PDK           SKY130, sky130_fd_pr__nfet_01v8
```

Toolchain environment variables (`scripts/train_supervised.py:toolchain_env`):

```
NGSPICE_LIBRARY_PATH  %USERPROFILE%/silq-ngspice/Library/bin/ngspice{}.dll
SPICE_LIB_DIR         %USERPROFILE%/silq-ngspice/Library/share/ngspice
PDK_ROOT              %USERPROFILE%/pdk
PATH                  prepend  silq-ngspice/shim ; silq-ngspice/Library/bin
PYTHONPATH            src      (silq is NOT pip-installed into .venv)
```

### Known provenance gap

**The working tree was dirty when this checkpoint was trained, and the training code is not
recoverable from git.** Nine source files are modified against `2f3ec52`, and three of them
— `envs/sequential_env.py`, `envs/equalizer_env.py`, `specs.py` — were edited at
22 Aug 08:41–08:42, *after* this model finished training at 02:24. No commit contains the
code that produced it.

This was checked rather than assumed, and the risk is empirically closed: the later edits
are gated on `Spec.boost_target_tol_db is not None` and `SequentialEqualizerEnv.boost_tol
is not None`, both defaulting to `None`, so the default path is unchanged. Re-running all
three headline measurements against the current tree reproduces the originals bit-for-bit
(section 3). **Commit and tag before the final benchmark** so this gap does not reopen.

## 3. Reproducing the headline numbers

`bash scripts/audit_reproduce.sh` re-runs all three into `repro_*` files, leaving the
originals intact. Verified 23 Aug:

| Measurement | Original | Re-run | Per-spec identical |
|---|---|---|---|
| Loose 32-spec rollout | 26/32, median 4.0 | 26/32, median 4.0 | 32/32, design diff 0.000e+00 |
| Strict 32-spec rollout (±1.5 dB) | 10/32, median 4.5 | 10/32, median 4.5 | 32/32, design diff 0.000e+00 |
| Reward audit | +4.355 / −0.047 / −5.000 | identical | JSON byte-identical |

All three are fully deterministic. Sources of determinism: specs from
`np.random.default_rng(0)`, env constructed with `seed=123`, per-spec `reset(seed=1000+i)`,
and `model.predict(deterministic=True)`.

```bash
export PYTHONPATH=src
./.venv/Scripts/python.exe -m silq.experiments.policy_rollout \
    --model results/seq_clean40k.zip --specs 32 --out results/repro_loose.json
```

Add `--boost-tol 1.5` for the strict variant. `--boost-tol` is opt-in; omitted, `hard_pass`
does not reference `target_boost_db` at all and boost is judged only against the 3–12 dB
range, which is how every artifact before 22 Aug was scored.

## 4. Known protocol inconsistencies

These are recorded, not fixed. Fixing any of them changes what a number means, which is a
decision for the team.

1. **`honest_benchmark` does not use one protocol.** Random search, CMA-ES and TPE run at
   `--budget 150` on the first `--search-specs` specs × `--seeds` seeds; RL runs one
   deterministic rollout at `env.horizon` = 20 over `--rl-specs` specs. The published
   "RL 26/32 vs random 30/30" therefore compares different specs at different budgets.
   The *validity criterion* is genuinely shared — every method routes through one
   `evaluate()`.
2. **RL's simulation cost is under-reported by about 2×.** Each RL step calls
   `env.step()`, which reaches the simulator through `_measure`
   (`sequential_env.py:266-267`), and then `evaluate()`, which
   simulates again. `solve_rl` counts one. A reported "median 4 sims" is ~8 real
   simulations, and any amortization break-even doubles accordingly.
3. **Train/test mismatch in the env.** Training ran `guarded=True, fast=True`;
   `policy_rollout` constructs `SequentialEqualizerEnv(fast=False)` and takes the
   `guarded=False` constructor default. The policy is rolled out under different
   observation dynamics than it trained on. The success test is independently guard-checked,
   so solve counts stand.
4. **Model selection overlaps the test set.** The 8-spec checkpoint curve uses
   `--specs 8`, which is a *prefix* of the same `default_rng(0)` stream as `--specs 32`.
   Specs 0–7 are shared, so a checkpoint chosen on the curve has been selected on a quarter
   of the set it is then reported against.
5. **`policy_rollout` records `design: None` for unsolved specs.** Any correlation computed
   from its rows is conditioned on success. Under `--boost-tol`, success *means*
   |achieved − requested| ≤ tol, so such a correlation is selection on the dependent
   variable — this produced a reported +0.962 for a policy whose unconditioned correlation
   is +0.114. Use `silq.experiments.target_audit` instead, which records every spec.

## 5. Audit tooling added

Measurement only. Neither changes a reward, a bound, a hyperparameter or a criterion.

- `silq.experiments.target_audit` — unbiased target tracking. Records every spec's full
  trajectory whether or not it succeeds, and runs PPO and random search on the **same**
  specs, budget and validity test.
- `silq.experiments.chance_baseline` — empirical spec-blind expectation, computed from the
  boosts the system has demonstrably achieved. The reference any retargeting claim must
  beat.
- `scripts/audit_reproduce.sh` — re-runs the three headline measurements into `repro_*`.

## 6. Files needed for the final benchmark

| File | Role |
|---|---|
| `results/seq_clean40k.zip` | the model; every RL number comes from this |
| `results/seq_clean40k_train.json` | its config, as written by the trainer |
| `results/target_tracking_clean40k.json` | 26 re-simulated designs; the chance-baseline pool |
| `docs/PROBLEM.md` | the authority for design bounds (`l_in` 0.15–1 µm, line 59) |
| `src/silq/guards.py` | thresholds and PDK bounds, owned by the project |
| `scripts/audit_reproduce.sh` | determinism check before publishing any number |

## 7. `l_in` lower bound — verified, unchanged

Asked whether the bound is a design constraint or an arbitrary implementation choice. It is
**derived from the design constraints**, and all three sources agree exactly:

```
docs/PROBLEM.md line 59   L_in  0.15 - 1 um     the specified design range
ACTION_SPACE["l_in"]      0.15 - 1 um           what the code searches
guards.PDK_BOUNDS         0.15 um minimum       the SKY130 device limit
```

Of the twelve action-space endpoints, exactly one touches a process limit — this one —
because the specified range begins at the process minimum. The `T2.8` rejections (9.4% of
simulation) come from action clipping: `decode_action` clips to `[0, 1]`, so every policy
output at or below −1 lands on exactly the same value. It is a policy-saturation artifact,
not a badly chosen bound, and the guard is correct to flag a device pinned at a process
limit. **Not changed.**

## 8. Matched-protocol target audit — results

`silq.experiments.target_audit`, 32 specs, budget 20 evaluations per method, identical
validity test, strict tolerance ±1.5 dB. Artifact: `results/target_audit_clean40k.json`.

| | PPO | Random search |
|---|---|---|
| loose solves | **26 / 32** | 8 / 32 |
| median evaluations to first loose solve | 4 | 13 |
| strict solves (±1.5 dB) | 6 / 32 | 4 / 32 |
| guard-valid evaluations per spec (of 20) | 12.2 | 4.7 |
| loose-passing evaluations per spec | 5.2 | 0.3 |
| unfiltered corr(requested, achieved) | **+0.114** (n=26) | +0.338 (n=8) |
| median \|achieved − requested\| | 2.81 dB | 1.81 dB |

The loose result is a real, large win for the policy and reproduces `policy_rollout`'s
26/32 exactly on an independent code path. The strict result is not a win: 6 versus 4 out
of 32, and the policy's unfiltered correlation with the requested boost is statistically
indistinguishable from none.

**The same data, filtered to the specs that passed the strict test, correlates at +0.966.**
That filtered figure was previously reported as evidence of retargeting. It is selection on
the dependent variable and it is not evidence of anything.

### Known limitation of the PPO arm in this tool

The env sets `terminated = passed` on the first *loose* pass. A deterministic policy
restarted from the same seed replays the identical trajectory, so under this protocol PPO
spends its 20 evaluations on ~7 distinct designs (mean distinct-boost fraction 0.36) while
random search gets 20 (0.77). PPO is therefore *under*-credited here.

This also explains the gap between this tool's 6/32 strict and `policy_rollout --boost-tol`'s
10/32. With `--boost-tol` set, `passed` gates on the target as well, so the episode does not
terminate at the loose pass and the trajectory keeps running toward the target. Up to that
point the two produce identical action sequences — the observation vector does not depend on
`boost_tol` (`sequential_env._obs` reads measurements and the target, never the margins).

`--fresh-restarts` re-seeds each restart, giving PPO 20 genuinely distinct attempts. Off by
default so the run above reproduces. Variant artifact: `results/target_audit_freshrestart.json`.

### The restart arm, and what it does and does not show — both spec sets

| | seed 0 replay | seed 0 restarts | seed 1 replay | seed 1 restarts |
|---|---|---|---|---|
| loose solves | 26 / 32 | 26 / 32 | 22 / 32 | 26 / 32 |
| strict solves (±1.5 dB) | 6 / 32 | **16 / 32** | 8 / 32 | **20 / 32** |
| median \|achieved − requested\| | 2.81 dB | 1.10 dB | 2.33 dB | 0.91 dB |
| mean distinct designs per spec | 3.2 | 10.0 | 1.6 | 3.7 |
| paired exact McNemar, strict, vs replay | — | p = 0.0020 (16 gained, 0 lost) | — | p = 0.0005 (12 gained, 0 lost) |

The strict gain replicates on the held-out spec set and is paired-significant on both. It is
also fully accounted for by the extra distinct designs: against the per-method matched chance
line (section 11) the restart arm scores 16 vs 16.9 expected on seed 0 (p = 0.75) and 20 vs
19.2 on seed 1 (p = 0.46). **More attempts, not better aim.** Restarts buy coverage; nothing
here shows the policy steering toward a requested boost.

The "closest loose-passing design" correlation for this arm is +0.553 against a null mean of
+0.584 on seed 0 (p = 0.63, *below* its null) and +0.774 against a null band of [−0.386, +0.382]
on seed 1 (p < 0.0001, above it). A statistic that flips sign relative to its own null between
two draws of the same distribution is not reportable in either direction. The matched chance
line, which controls for the number of distinct designs, is consistent across both and is the
one to quote.

## 9. Random search as a named baseline

Usable, with one caveat to state in the writeup. Under the matched protocol it shares the
spec list, the budget, the evaluator and the validity test with PPO, and its draws are
reproducible (`np.random.default_rng(20260823)`). Uniform draws in the normalized
`[0, 1]^6` action cube, decoded by the same `decode_action`.

Caveat: only 4.7 of 20 random draws survive the guard and only 0.3 loose-pass, so its
strict score of 4/32 is measured on very few valid designs and carries a wide interval.
It is a fair baseline for "can you do this by chance", not a tuned optimiser — CMA-ES and
TPE remain the strong baselines, and neither has yet been run under this protocol.

## 10. What the chance baseline does and does not assume

`chance_baseline` draws from the 26 boosts the system has demonstrably achieved, all of
which already loose-pass. It answers "given that you can produce a valid design, how often
does its boost land within ±1.5 dB of a target you never read". It is therefore conditional
on producing a valid design, which is why random search (0.3 loose-passes per spec) scores
far below its own budget-20 line of 31.0 — random search mostly fails the prior step.

The line to compare a policy against is the one for the number of *independent* attempts it
gets. PPO's attempts are a correlated trajectory, median 4 to first solve, so the relevant
references are budget 1 = 9.6/32 and budget 4 = 22.1/32. Its 10/32 sits at the budget-1
line.

## 11. The frozen matched benchmark protocol

Every method reported in the final benchmark runs exactly this. Anything that deviates is
a different measurement and must be labelled as one.

| | |
|---|---|
| specs | 32, from `target_audit.make_specs` — `np.random.default_rng(0)`, per spec `uniform(5,11)` target then `uniform(8,16)` channel, in that order |
| budget | 20 optimizer evaluations per spec per method |
| evaluator | `silq.evaluator.build_evaluator(DEFAULT_SPEC, corner="tt", fast=False, channel_loss_db=<spec's channel>)`, one guard cached per channel |
| validity | guard-valid (Tiers 1–4). A rejected candidate consumes its evaluation and records nothing |
| loose criterion (A: feasibility) | `specs.hard_pass` with the target **unscored** — boost judged against the 3–12 dB range |
| strict criterion (B: retargeting) | loose **and** \|achieved − requested\| ≤ 1.5 dB |
| `*_solved_at` | index of the FIRST evaluation meeting that criterion, 1-based; `None` if never |
| `best_*` | among the loose-passing evaluations, the one closest to target — recorded **whether or not** the strict test was ever met |
| trajectory | every evaluation recorded, success or not. Correlation is computed on this unfiltered population and never on the passing subset |
| seeds | PPO env `seed=123`, per-spec `reset(seed=1000+i)`, `deterministic=True`; random `default_rng(20260823)`; CMA-ES and TPE seed `20260823 + i` |

`make_specs` is imported by every runner rather than re-derived, so spec *i* is the same
spec in all four arms by construction, not by coincidence.

**No target leakage.** The target reaches a method only through the objective it is
allowed to see: PPO through its observation, CMA-ES and TPE through the dense score
`(# hard checks passed) − |boost − target| / 3` that `honest_benchmark` already used. The
strict test is applied afterwards by the harness, and no method is told the tolerance.

**Restart semantics, stated because they differ.** Random draws 20 independent points.
CMA-ES runs one uninterrupted run of 20 evaluations, population 9, no restart. TPE runs
one study of 20 trials, no restart. PPO runs a trajectory that the env *terminates* at the
first loose pass; `--fresh-restarts` then restarts it from a new sizing, which is the
closest analogue of an independent attempt. Without that flag it replays the same
trajectory and is under-credited — see section 8.

## 12. Checksums of the frozen artifacts

```
8868965260d7d169f936020588f28f6d2d3f6a66011444a1d96f525d8f508ec9  results/seq_clean40k.zip
85d3ccb80d4d07fc3c10a2da32393edb4ca1cc4a112c37d049542351e99f8013  results/seq_clean40k_train.json
5c99a693ab4d56dec7513b7170e034369a96a34b3d796e4dcb3cc82c9ebf30b2  results/target_tracking_clean40k.json
```

`results/*.zip` is gitignored, with one documented exception already in `.gitignore` for
`results/seq_agent.zip` — *"without the weights nothing in RESULTS.md can be reproduced by
anyone who clones this repo. 159 KB — small enough to track."* `seq_clean40k.zip` is
164 KB and is now the model every headline number comes from, so the same exception is
extended to it. That is the only `.gitignore` change in the freeze commit.

## 13. What "one simulation" means — measured, not assumed

`silq.experiments.simcount_audit` counts at two chokepoints: `measures.measure_all` and
`NgspiceServer._analysis`, the single function every SPICE analysis passes through.
Averaged over 3 specs × 6 steps (`results/simcount_audit.json`):

| Unit | PPO evaluation | Random / CMA-ES / TPE evaluation |
|---|---|---|
| optimizer evaluations | 1 | 1 |
| `measure_all` calls | **2.00** (`env.step` 1.00 + guarded verify 1.00) | 1.00 |
| SPICE analyses | **8.4** | 4.2 |

The factor is exactly 2, and it is not waste: `env.step`'s measurement is what produces
the observation the policy acts on. It is a real cost of the method.

Two consequences, both stated rather than absorbed:

* **The historical metric is unchanged.** "Median 4 simulations to first solve" means 4
  optimizer evaluations and always did. In `measure_all` calls that is 8; in SPICE
  analyses, about 34. Any amortization break-even against a search baseline doubles.
* **In the matched protocol PPO gets twice the simulator budget.** Budget 20 costs PPO 40
  `measure_all` calls and each baseline 20. An evaluation-matched read and a
  simulation-matched read are different comparisons; both are reported, and the
  simulation-matched read is PPO's result within its first 10 evaluations.

## 14. Train/test environment mismatch

Training ran `guarded=True, fast=True`. Every rollout since has constructed
`SequentialEqualizerEnv(fast=False)` and taken the `guarded=False` default, so the policy
acts on observations whose HD3 and noise entries are computed rather than stubbed, and on
a `_measure` that never returns `Measures(ok=False)` for a guard rejection.

The success test is unaffected — it is applied by an independent guarded `fast=False`
evaluator outside the env — so every published solve count stands as measured.

**A matched-environment rollout is possible without retraining**, and does not touch the
frozen model: construct the rollout env with `guarded=True, fast=True` while leaving the
verification evaluator at `guarded, fast=False`. `target_audit --match-train-env` does
exactly that. It is a separate audit result and is not comparable to the historical
numbers, because the trajectory the policy walks genuinely differs.

**Measured.** Artifact: `results/target_audit_matchtrain.json`, seed-0 specs, budget 20,
verification identical to the headline run.

| | eval env (headline) | train-matched env |
|---|---|---|
| loose solves | 26 / 32 | **32 / 32** |
| strict solves (±1.5 dB) | 6 / 32 | 8 / 32 |
| median \|achieved − requested\| | 2.81 dB | 2.43 dB |
| mean distinct designs per spec | 3.2 | 2.5 |

Paired exact McNemar on loose solves: 6 specs solved only in the train-matched env, 0 only in
the eval env, p = 0.031. Strict: 6 vs 4, p = 0.75.

**The mismatch penalises the policy; it does not inflate it.** The published 26/32 is the
conservative figure, and in-distribution feasibility is higher than the number we report. The
strict result does not move — 8/32 against a matched chance line of 16.9/32 — so the mismatch
is not what is hiding a retargeting signal either.

## 15. Model selection overlaps the test set

The 8-spec checkpoint curve used `--specs 8`, a prefix of the same `default_rng(0)` stream
as `--specs 32`. Specs 0–7 are shared: a checkpoint chosen on that curve was selected on a
quarter of the set it is then reported against.

This does not affect `seq_clean40k`, which is the final checkpoint of its run and was not
chosen off the curve — but it does affect any number quoted for a curve-selected
checkpoint, and those must be labelled.

**A clean untouched target set is available without retraining.** `target_audit
--spec-seed <n>` draws the specs from `default_rng(n)` with the same distribution and the
same draw order; `n = 0` is the historical set and any other `n` has never been seen by
any selection decision. Running all four arms on a fresh seed is a one-command validation
whenever the team wants it.

---

# PRE-REGISTRATION — Strategy A, week 1

**Committed before any spec-seed-2 or spec-seed-3 simulation was run.** Everything from
section 16 down was written and committed first, deliberately, so that the constants below
cannot have been chosen after seeing the result they are used to judge. The commit that
introduces this text contains no seed-2 or seed-3 artifact; check `git log` on this file
against the artifact timestamps if you want to verify that rather than trust it.

Nothing in sections 1–15 changes. The reward, the PPO hyperparameters, the design-space
bounds, the ±1.5 dB criterion and the frozen `seq_clean40k` checkpoint are all untouched by
this work. The hybrid enters the comparison as an ADDITIONAL ARM under the protocol
section 11 already froze — it does not modify any existing arm.

## 16. Surrogate model — specification and acceptance gate

### What it is

A regressor over the SPICE runs already on disk in `results/raw`:

    normalized 6-D design  ->  (dc_gain_db, boost_db, peak_freq_ghz)

The RAW ARCHIVE under `results/raw` held 414,955 recorded runs at the time of this
section, 374,455 of which carried an `acx.data`. (That is the archive, not the
surrogate corpus built from it: the rebuild reported further down this document
yields **374,588** rows, and the two numbers are different quantities measured at
different times. Both words were "corpus" in an earlier draft, which is how they
came to look like a contradiction.) All three
target-relevant metrics derive from the AC run alone (`sim/measures.py:peaking`), so the
dataset is recoverable by PARSING FILES ALREADY ON DISK. It costs zero new simulation.

Designs are normalized exactly as `ctle.decode_action` maps them — log-scaled on `w_in`,
`i_tail`, `rs`, `cs`, `r_load`; linear on `l_in` — so a distance in surrogate space is the
same distance the policy's action space uses.

### Scope, stated as a limit rather than discovered as one

  * It predicts AC metrics ONLY. It does not predict guard validity, eye height/width,
    input-referred noise, or HD3. Guard validity is the binding constraint: a uniform
    sample of the space is 23% guard-valid and 1.2% all-pass
    (`results/feasibility_band.json`, 2000 samples). The surrogate cannot see that and
    must never be asked to.
  * Its coverage is BIASED toward the regions PPO visited, because that is what generated
    the corpus. The test TARGETS are drawn fresh (`--spec-seed 2`, `3`), so this is not
    leakage of the test set; it is a stated limit on where the model is accurate.
  * IT IS A RANKER, NEVER A JUDGE. No success, solve, or pass in any reported number is
    ever decided by the surrogate. Every design that appears in a result is verified by
    the same guarded `fast=False` evaluator every other arm uses.

### Acceptance gate — PRE-REGISTERED, NOT TO BE LOOSENED

The surrogate ships into the H2 arm only if, on a CHRONOLOGICAL split of the corpus
(train on the earliest runs, test on the latest — an i.i.d. split leaks, because PPO
trajectories put a design's own one-step neighbour into the training set):

    (a) boost_db MAE  <=  0.5 dB  overall,                                   AND
    (b) >= 90% of held-out designs within 1.5 dB of truth IN THE SPARSEST
        DECILE by distance to the nearest training design.

A preliminary 40,000-record probe returned 0.295 dB MAE and 95.2% in the sparsest decile,
so this gate is a floor that is expected to clear, not a hope. **It is recorded here at
that level precisely so it cannot be relaxed after the full-corpus number is seen.** If
the full corpus misses the gate, the surrogate is CUT and the week proceeds with H1 only;
the gate does not move.

## 17. The hybrid arm — pre-registered protocol

### Motivation, and what it does not claim

PPO is measured-good at feasibility (26/32 loose, median 4 evaluations) and
measured-not-good at hitting a requested boost (section 8: on its matched chance line on
both spec sets). The hybrid tests whether SPLITTING THE PROBLEM beats asking one policy to
do both.

If it succeeds, the honest claim is *"a two-stage designer, in which the RL policy supplies
feasibility and a local refiner supplies precision, hits requested specs above matched
chance."* It is NOT a claim that the RL policy learned to retarget. That distinction is the
whole point and must survive into the writeup.

### Structure

    spec (target, channel)
      -> STAGE 1: frozen seq_clean40k, k evaluations       [2.00 measure_all each]
      -> hand off the design the policy currently holds
      -> STAGE 2: local refinement, r evaluations          [1.00 measure_all each]
      -> verification: guarded, fast=False, outside the loop, identical to every other arm

### Constants — FIXED HERE, BEFORE ANY RUN

| constant | value | why this value, decided in advance |
|---|---|---|
| `k` (PPO evaluations) | **5** | PPO's median first loose solve is 4 evaluations, a fact measured in section 8 BEFORE this experiment existed. k = 5 places the handoff just past it. |
| `r` (refinement evaluations) | **10** | forced by the budget rule below, given k = 5 |
| budget rule | **2k + r <= 20 `measure_all`** | every baseline arm spends 20. PPO costs 2.00 `measure_all` per evaluation and the search arms 1.00 (section 13, measured). Matching the SIMULATION budget rather than the evaluation count is what makes the comparison fair. |
| `sigma0` (refiner) | **0.05** | ~5% of each normalized axis, i.e. local by construction. Not searched over. |
| objective | `honest_benchmark`'s existing dense score, **unchanged**: `hard-pass count - abs(boost - target)/3` | introducing a new objective here would confound "splitting the problem helped" with "a better objective helped" |

`k` is the cherry-picking surface of this experiment and pre-registering it is the entire
defence. A `k`-sweep may be run afterwards; if it is, it is labelled EXPLORATORY and never
becomes the headline number.

### The two variants — same SPICE budget, different use of it

  * **H1** — CMA-ES seeded at the handoff design with `sigma0 = 0.05`, spending all r
    evaluations on SPICE. No surrogate involved. This is the control for H2.
  * **H2** — propose 20x r candidates, rank them by SURROGATE-PREDICTED
    `abs(boost - target)`, and spend the r SPICE evaluations only on the top-ranked.
    Identical SPICE budget to H1; more candidates considered. This is where the surrogate
    either earns its place or does not.

H1 and H2 differ in exactly one thing, so the difference between them measures the
surrogate and nothing else.

### Success criterion — SINGLE, PRE-REGISTERED

> The hybrid arm must exceed its OWN PER-METHOD MATCHED CHANCE LINE (section 11), with k
> counted as DISTINCT designs, on spec seed 2 — AND replicate that on spec seed 3.

Raw strict-solve count is explicitly NOT the criterion. Section 8 is the reason: fresh
restarts raised strict solves 6/32 -> 16/32 and 8/32 -> 20/32 while landing ON the chance
line both times, because more feasible designs produce more accidental target hits. A
hybrid that raises the strict count without clearing its line has demonstrated coverage,
not aim, and will be reported as such.

### Reporting requirements

  * **Per-target-tercile.** Measured across all 3,624 recorded guard-valid boosts, the
    fraction of distinct achieved boosts within +/-1.5 dB of a request is 0.232 on average
    but 0.086 at an 11 dB request. The top of the 5-11 dB band is genuinely harder, so a
    method that happens to draw high targets looks worse for reasons that are not the
    method. Aggregate numbers hide this; terciles do not.
  * **Both budget readings**, evaluation-matched and simulation-matched, as `final_report`
    already prints for every arm.
  * **Replication across spec seeds 2 and 3.** Neither has been seen by any selection
    decision. A result that holds on one and not the other is reported as not replicating —
    the same rule that retired the CMA-ES loose comparison and the restart arm's
    correlation statistic.

### Run order

    commit this protocol
      -> surrogate audit
      -> gate check (cut H2 here if it fails; do not move the gate)
      -> H1 + H2 implemented
      -> spec seed 2, all arms
      -> spec seed 3, all arms
      -> compare against frozen PPO and the matched chance line

Arms required on each new spec seed, since none exist yet for seeds 2 and 3:
`target_audit` (PPO replay + random), `target_audit --fresh-restarts`,
`search_audit cmaes`, `search_audit tpe`, `hybrid_audit H1`, `hybrid_audit H2`.

## 18. Amendment 1 to the pre-registration — made BEFORE any compute against it

**Status: prospective.** No spec-seed-2 or spec-seed-3 simulation had been run when this
was written, and none is contained in the commit that introduces it. The section-17 text
above is left standing exactly as first committed rather than edited in place, so the
record shows what was originally written and what was changed — an amendment that erased
its own predecessor would be worth nothing.

### 18.1 The defect being fixed

Section 17 asserted three things that cannot all hold:

  1. **H1** is "CMA-ES seeded at the handoff design with `sigma0 = 0.05`" — ADAPTIVE.
  2. **H2** is "propose 20x r candidates, rank them by surrogate-predicted
     `abs(boost - target)`, and spend the r SPICE evaluations only on the top-ranked"
     — ONE-SHOT, non-adaptive.
  3. "H1 and H2 differ in exactly one thing, so the difference between them measures the
     surrogate and nothing else."

Claim 3 is FALSE given claims 1 and 2. As written the two arms differ in TWO ways: whether
the surrogate ranks candidates, and whether the search adapts between generations. An
H2 > H1 result would therefore have been uninterpretable — it could have been the
surrogate or it could have been adaptivity, and no measurement in the experiment could
separate them. That is the same class of confound this project has spent its whole
methodology removing, and shipping it would have been worse than shipping nothing.

### 18.2 The fix

**H2 becomes the SAME adaptive CMA-ES as H1, differing only by a surrogate pre-screen.**

    H1  each generation: ask CMA-ES for `popsize` candidates
                         SPICE all of them
                         tell CMA-ES the true scores

    H2  each generation: ask CMA-ES for 20 x `popsize` candidates
                         rank them by SURROGATE-PREDICTED abs(boost - target)
                         SPICE only the `popsize` best
                         tell CMA-ES the true scores for those

Same optimizer, same seed, same `sigma0 = 0.05`, same handoff design, same initialization,
same objective, same SPICE budget. **The only difference is which candidates get simulated.**
So the comparison recovers exactly the quantity it was supposed to measure:

    H2 - H1  ~  the value of surrogate pre-screening

The `cma` API supports this directly and it was verified before this amendment was
written: `ask(number=20*popsize)` returns the oversampled population, and `tell()` accepts
the `popsize`-sized subset that was actually evaluated, after which the next generation
proceeds normally.

### 18.3 What does NOT change

Every pre-registered constant stands: `k = 5`, `r = 10`, `sigma0 = 0.05`, the
`2k + r <= 20 measure_all` budget rule, the objective left exactly as `honest_benchmark`
defines it, the matched-chance-line success criterion, per-target-tercile reporting, and
replication across spec seeds 2 and 3. This amendment changes the STRUCTURE OF H2 only, and
changes it in the direction of a stricter comparison, not a more favourable one.

### 18.4 Stage 1 termination — made explicit

`SequentialEqualizerEnv` sets `terminated = passed` on the first LOOSE pass. Stage 1
therefore has to say what it does when that fires inside the k = 5 evaluations. It is
recorded here rather than left to the code:

> **Stage 1 runs exactly k = 5 evaluations and does NOT reset on `terminated`.** The
> policy keeps stepping from the design it currently holds, and that design is the handoff.

Two reasons, both already measured rather than assumed:

  * **Resetting would import a known confound.** Section 8 measured what re-seeding after
    termination does: it is the replay artifact, and it changes the arm's distinct-design
    count by a factor of 3-6. Rolling a good design away and starting over is the behaviour
    this project already identified as a measurement problem, not a search strategy.
  * **The budget must be identical for every spec.** Ending stage 1 early on termination
    and donating the remainder to refinement would give some specs more refinement budget
    than others, and would give the EASY specs the most — a bias in favour of the arm.

Fixed k also means `2k + r` is exactly 20 `measure_all` on every spec, with no per-spec
variance to explain away.

### 18.5 The 6,000-record smoke test is NOT evidence about the gate

While implementing `surrogate_audit`, a 6,000-record slice was run as an implementation
check and returned 0.257 dB MAE with 95.8% in the sparsest decile. **That number carries no
weight in the acceptance decision and must not be cited as though it did.** The slice was
the chronologically earliest records, so it spans a single run-day and its "chronological"
split holds out almost nothing — the split it reports is not the split the gate is defined
on. Section 16's gate is judged on the FULL-CORPUS chronological audit and on nothing else.

A smoke test proves the code runs. It does not prove the model works, and the two must not
be allowed to blur, least of all in the direction of accepting something.

---

## 19. The corpus was mislabelled, the gate could not see it, and H2 is cut

This section is the record of a data-integrity failure in the surrogate subsystem, how it
was found, and what it cost. Nothing in it touches the frozen baseline or any
SPICE-measured result; the fault was confined to the new corpus parser.

### 19.1 The bug

`surrogate.metrics_from_acx` read column 1 of `results/raw/*/acx.data` as magnitude in dB.
That file is written by `NgspiceServer.ac_complex`, which calls ngspice's `wrdata` on the
complex vector `v(outp)-v(outn)`. Its layout is `[freq, real, imag]`, so column 1 is the
**real part of H**, not its magnitude. Every `boost_db` label in the corpus was therefore

    max(Re H) - Re H[0]

a quantity with no physical meaning that happens to land in a small, plausible,
positive-looking range. The correct reconstruction is `20*log10|Re + j*Im|`, then
`peaking`'s own definition on top of it: DC is the first sample, boost is
`max(mag) - mag[0]`.

`results/raw` carries no `ac.data` — the sweep `sim.measures.peaking` uses was never
persisted — so the corpus must be built from `acx.data`. Its sweep stops at 24 GHz rather
than `AC_FSTOP` = 100 GHz. On the 51 designs with independent ground truth that difference
is worth nothing (max error 0.0005 dB), but a design peaking above 24 GHz would be
understated, and that limit is real if unobserved.

### 19.2 Why the section 16 gate could not detect it

**It is structurally incapable of detecting it, and this is the transferable lesson.**

The gate grades the model on a held-out split of labels produced by *the same parser* that
produced its training labels. An error shared by training and test cancels exactly. The
gate reported **0.166 dB MAE and 98.9% in the sparsest decile** while the model was
predicting the real part of H — an honest measurement of internal consistency, and silent
on correctness.

    A model may never be validated solely against labels generated by the pipeline that
    produced those labels. That measures self-consistency. Correctness needs a source the
    pipeline did not write.

### 19.3 What exposed it

An off-distribution probe with independent ground truth. `results/feasibility_band.json`
holds 2000 uniformly drawn designs run through the real evaluator, with `dc_gain_db`,
`boost_db` and `peak_freq_ghz` recorded by `sim.measures.peaking` itself. On the 51 that
also record their design, the corpus disagreed with SPICE by up to 5.6 dB — and the
nearest-neighbour distance was **0.000**, meaning the corpus held those exact designs and
had them labelled wrong. `k=1` was no better than `k=5`, which ruled out kNN smoothing and
pointed at the labels rather than the model.

Uniform draws mattered twice over: they are independent of policy visitation, and they
land where the corpus is thinnest and where the requested targets live — which is where
the mislabelling did its worst damage.

### 19.4 The fix, and the check that is now mandatory

`metrics_from_acx` reconstructs the magnitude. Verified against all 51 ground-truth
designs: max error **0.00047 dB** (dc_gain), **0.00049 dB** (boost), **0.00005** (peak
freq).

`surrogate.assert_matches_feasibility_band` performs that comparison and **raises**.
`surrogate_audit` calls it after the corpus loads and before anything is judged. It is not
behind a flag and it is not advisory: a failure halts the audit, because every number
below it would be meaningless. The audit now runs two tests, in this order:

| | source of the answer key | what it can catch |
|---|---|---|
| **parser correctness** | independent SPICE, via `sim.measures.peaking` | mislabelled corpus |
| **model accuracy** (the section 16 gate) | corpus labels, held-out split | a poorly fitting model |

The first must pass before the second means anything.

### 19.5 Everything withdrawn

All of these were computed on the mislabelled corpus and are **void**:

- the surrogate gate PASS at 0.166 dB MAE / 98.9% sparsest decile;
- the local reach measurements (`step_reach`), including "0.54 dB median reach";
- the `sigma_derivation` reach curves and the finding that no sigma satisfied the rule;
- the `reach_ceiling` spread-compression ratios and the "plateau is physics" conclusion;
- every figure in `reach_by_region`;
- the `target_reachability` claim that 56–66% of targets sit above the corpus p99.

Those five scripts are kept in the tree **disabled**: each raises `RuntimeError` on
execution rather than merely carrying a comment, so the outputs cannot be regenerated and
mistaken for current. They exist for provenance, not for use.

**Not affected**, because the surrogate never touched them: the frozen `seq_clean40k`
baseline, all `target_audit` and `search_audit` results, the restart-arm findings, the
matched chance lines, the McNemar tests, and the handoff-gap distribution read from the
frozen artifacts.

### 19.6 The corrected gate result — H2 is cut

Corpus rebuilt: 374,588 records over 8 run-days. Parser check passed. Then:

```
  (a) boost MAE 0.456 dB  <=  0.50 dB               PASS
  (b) sparsest decile within 1.5 dB: 81.2%  >= 90%  FAIL
```

| | mislabelled | corrected |
|---|---|---|
| boost MAE (chronological) | 0.166 dB | 0.456 dB |
| sparsest decile within 1.5 dB | 98.9% | **81.2%** |
| R² | 0.953 | 0.943 |
| sd of true boost | 1.43 dB | **4.30 dB** |

That last row explains the first: the real boost distribution is three times wider than
the fictitious one, and the old model looked accurate largely because the quantity it
predicted barely varied.

Boost error rises monotonically with distance from recorded designs — 0.199 dB in the
densest decile, 0.917 dB in the sparsest. H2's screen would have operated in exactly that
sparse regime, ranking CMA-ES proposals that by construction sit away from anything
recorded. **The gate failed in precisely the place it was written to protect.**

**By the pre-registered rule of section 16, H2 is cut and week 1 proceeds with H1 only.**
The threshold does not move and this is not re-run against a looser one. No further
measurement was taken after the failure: running additional tests in search of a number
that rescues a cut arm is the behaviour pre-registration exists to prevent.

### 19.7 What H1 still owes

H1 — frozen PPO handoff into local CMA-ES, no surrogate — is unaffected by any of this,
but the concern that prompted the whole investigation is now **unmeasured rather than
resolved**: whether σ₀ = 0.05 can move the handoff meaningfully. The evidence that said it
could not was void. That question is to be settled by a real-SPICE H1 smoke test, measured
from actual candidate generation and actual SPICE outcomes — never again inferred from the
corpus. If σ₀ = 0.05 proves inadequate under real SPICE, the protocol is amended once,
prospectively, on that measurement.

### 19.8 Artifacts

Corpora are generated caches and are `.gitignore`d (`results/*.npz`); they rebuild in
~20 minutes from `results/raw` via `surrogate_audit --rebuild`. What is tracked is the
parser, its ground-truth check, and this record.

`results/surrogate_corpus_REALPART_BUG.npz` is kept on disk, untracked, as the evidence
artifact for this section. **It is invalid and is not part of any reproducibility path.**

---

## 20. The G3 line and the final comparison — architecture frozen 26 Aug 2026

Sections 1–19 end with H1 as the refinement arm. What followed asked a different question:
**can a constraint-aware numerical method close the specification more precisely than a
general-purpose optimizer, at the same budget?** It can. This section is the record of how
that was established and where the line was stopped.

### 20.1 The sequence, and what each step actually settled

| step | what it did | what it settled |
|---|---|---|
| **G3** | six-dimensional composite `d_boost` descent | precision targeting walked the peak out of band on 4 of 4 bracketed specs — one composite direction cannot serve two constraints |
| **G3.1** | one-dimensional bracketed advance | exact where it could start (0.01 and 0.07 dB in 3–4 evaluations against H1's 10) and unable to start on 5 of 8. A **starting-point** problem, not a search problem |
| **G3.2** | measured `(rs, l_in)` plane; rescue ladder for guard-invalid handoffs; peak stepped in log space by secant, not by the table | both failure branches addressed. Seed-3 slice: 6/10 strict vs H1's 5/10 — promising, not superior |
| **G3.2a** | probed a frozen low-peak calibration set on all six axes | `rs` survives as the boost coordinate below the band; `l_in` moves the peak on 4/4 **above** but 1 of 8 **below**; `r_load` takes over. A single global `(boost, peak)` pair does not generalize |
| **G3.2b** | tested `(rs, r_load)` on a fresh independent set | 2/8 entered the band. On 5 of the 6 that did not, the blocker was `T4.10_dc_gain_implausible` — **boost was never the currency, DC gain was** |
| **final** | three arms, 40 held-out specs, one 20-`measure_all` pool | below |

G3.2a and G3.2b were **preregistered with pass/fail rules written before the data**, and
both rules were reported as failing or vacuous rather than amended afterwards. G3.2a's Q1
rested on the single design of eight where `l_in` moved the peak at all; the PASS it
produced was reported as vacuous and rejected. That is why the line stopped at G3.2b
rather than continuing into G3.3.

### 20.2 The final comparison

`docs/PREREG_FINAL_COMPARISON.md` + amendment 1, written and committed before the held-out
set was simulated. 40 specs at spec-seed 23, three arms paired on the same specs, 20
`measure_all` each, routing on `reached_target` with the fallback receiving only the unused
remainder. `results/final_comparison_seed23.json`, 885 simulations.

**The preregistered primary test is the matched chance line, and no arm beats its own:**

```
arm                    strict   mean k   chance line   p(>=obs)
A  PPO -> H1            13/40     2.73      16.1/40      0.956
B  PPO -> G3.2          22/40     3.50      19.4/40      0.170
C  PPO -> G3.2 -> H1    23/40     3.67      20.7/40      0.224
```

B's raw strict count is nine specs above A's and it still does not survive: B produces more
distinct designs among its loose passes, so its line is higher too. A is **below** its own
line. **Strict solve count is not a result here and is not to be reported as one.**

**What does separate is precision:**

```
median |achieved - requested|     A 1.886   B 0.231   C 0.241 dB
stage-2 evaluations, mean         A 10.00   B  5.05   C  7.12   (B median 3.5)
reduction from the PPO handoff    A +1.902 dB (improved 14/17, worsened 2)
                                  B +3.296 dB (improved 17/17, worsened 0)
by target tercile, median error   A 0.48 -> 1.99 -> 3.00
                                  B 0.09 -> 0.20 -> 1.07
```

The coverage explanation that killed the section-8 result runs the **wrong way** here: A
makes more evaluations (10 vs 5.05) and finds more guard-valid designs per spec (7.50 vs
5.62), and still finishes further from target. The advantage also grows with target
difficulty rather than concentrating in easy specs.

### 20.3 The paired comparison, and exactly what statistical status it has

On the 27 specs where **both** arms found a `hard_pass` design — same specs, same PPO
handoff — B is closer on 24, A on 2, tied on 1, median paired difference **+1.086 dB**.

That count is descriptive and preregistered. The tests below are **POST HOC**: they were
computed after the data, were not named in the preregistration, and are recorded here as
post hoc permanently.

```
sign test              p = 1.05e-05
Wilcoxon signed-rank   W = 15.0,  p = 4.08e-06
```

They are reported because the effect is large and the direction was predicted, and they are
labelled because a test chosen after seeing the data is a weaker object than one named
before it. **The preregistered inferential test remains the chance line, and it is
negative for every arm.** No writeup may present the sign test as the preregistered result
or use it to revive a solve-count claim.

### 20.4 Arm C is dropped from the final architecture

```
G3.2 reached its 0.25 dB target, no fallback         16/40
fallback FIRED                                       15/40
G3.2 stopped short with NO budget left to hand over    9/40
```

Of 15 firings the fallback strict-recovered **1** spec, found the first `hard_pass` design
on 2, and changed nothing on the other 12 (median improvement +0.000 dB). C beats B on 1 of
28 paired specs and ties on 27, for +2.07 mean evaluations.

The reason is structural and worth stating rather than burying: **a G3.2 that is struggling
is also a G3.2 that has spent the pool.** Under a fixed budget the fallback has least to
give exactly where it is needed. This is a finding about *this budgeted implementation*,
not a claim that strategy-switching is a bad idea.

Why it fired, diagnostic only and never a control input: `peak_in_band` on 8 of 15,
`T4.10_dc_gain_implausible` on 4 — G3.2a and G3.2b reappearing on fresh specs.

### 20.5 The held-out set — disclosure

**Seed 23 is not "completely untouched", and must not be described as such.**

> Seed 23 specifications were inspected once before the final run to verify the generated
> specification distribution; no simulations, design selection, hyperparameter changes, or
> protocol changes were performed as a result.

The inspection printed the first three `(target, channel)` pairs and the min/max of each
column. Every constant in play was already frozen in committed code before seed 23 was
named. Recorded in `docs/PREREG_FINAL_COMPARISON.md` amendment 1 §3 and repeated here so
the research record carries it independently of the preregistration.

### 20.6 Other caveats that travel with these numbers

* **17 of 40 handoffs were guard-invalid**, so the handoff-reduction metric is computed
  over 23 specs, never imputed, with the exclusion count printed beside it (§5.1 of the
  preregistration). This is far higher than the seed-3 slice's 3 of 10.
* **11 / 12 / 10 specs (A / B / C) produced no `hard_pass` design at all** and drop out of
  the medians. The medians therefore run over different denominators, which is why the
  paired 27-spec comparison is the number to read.
* The fallback-start rule, the tercile cut and the arm-C recovery breakdown were all fixed
  in amendment 1 **before** the run, not chosen to fit it.

### 20.7 The architecture is frozen here

No G3.3. No further axis. No router. No reward change. No retraining.

The system as delivered is hierarchical: **PPO** learns the global feasibility landscape;
**G3.2 constrained refinement** performs target-directed precision closure where it can;
**SPICE + the guard** independently determine physical validity. **H1/CMA-ES is an external
baseline, not a component.**

The claim this record supports, in full and with nothing beyond it:

> PPO efficiently learns the feasible CTLE design space but does not, by itself, demonstrate
> target-conditioned design. Decomposing the task — PPO for global feasibility, a
> constraint-aware numerical stage for specification closure — reduced median target error
> from 1.886 dB to 0.231 dB on an untouched 40-spec test set while using roughly half the
> refinement evaluations of the CMA-ES baseline. The improvement persisted even though the
> baseline explored more distinct feasible designs, which rules out the coverage
> explanation that accounted for the earlier section 8 result. **Strict solve counts did not
> clear their matched chance line for any arm and are not claimed.**

### 20.8 Reproducing this section

```
PYTHONPATH=src python -m silq.experiments.final_comparison --gate --spec-seed 3 \
    --first 8 --specs 10 --arms ab          # must print GATE PASSED before anything else
PYTHONPATH=src python -m silq.experiments.final_comparison --spec-seed 23 \
    --first 0 --specs 40 --arms abc
PYTHONPATH=src python -m silq.experiments.final_comparison_report
```

The gate re-runs arms A and B on the burned seed-3 development slice and diffs every
outcome field against `results/g32_repair_smoke.json` and
`results/hybrid_audit_h1_seed3.json`. It reproduced both 10/10. The two solvers are
transcribed into `final_comparison.py` rather than imported — they live inside `main()` in
their own files — precisely so those two frozen artifacts stay reproducible from
**unmodified** code. `_check_constants()` refuses to run if any transcribed constant has
moved in its source module.

---

## 21. The delivered circuit and its 45-corner PVT sign-off

Section 20 froze the research architecture. This section is the *product* claim, which is
a different one: here is the circuit SILQ produced, and here is what it survives.
Preregistered in `docs/PREREG_PVT_SIGNOFF.md`, written before any corner was simulated
and before any per-spec value in the seed-23 artifact was inspected.

### 21.1 How the candidate was chosen — the rule, not the circuit

Picking "the final SILQ circuit" from 40 designs *after* measuring them is exactly the
cherry-pick this record has spent its life avoiding. Two pre-written rules removed the
freedom:

* **The sweep is not selective.** The pool is every arm-B (`PPO → G3.2`) row whose
  `strict_solved_at is not None` — all nine hard checks plus |boost − target| ≤ 1.5 dB at
  TT/1.8 V/27 °C. That is **22 candidates**, and *all 22* were swept over all 45 corners.
  None was inspected before its sweep or dropped after.
* **The flagship is a formula**: PVT-clean first, then ascending worst-corner target
  error, then TT error, then spec index. The "nothing is clean" branch was written in
  advance so that reporting a non-clean circuit could never be a post-hoc retreat.

### 21.2 The sweep is stricter than anything already in this record

`src/silq/experiments/pvt_signoff.py`, 990 guarded evaluations.

* Every corner goes through `build_evaluator(fast=False)` — the **full guard layer** and
  **real** HD3 and noise. `experiments/characterize.py` sweeps the same grid but calls
  `measure_all` directly, so no corner is ever shown to the guard; it is left untouched
  (it also writes `results/final_report.json`, which is the honest-benchmark report and
  unrelated).
* Each corner is scored against the candidate's **own** spec — `DEFAULT_SPEC`'s 9 dB
  target is never substituted.
* `boost_target_tol_db = 1.5` is **on**, adding a tenth check that is `None` everywhere
  else in this repo. Every published number in sections 1–20 was scored on nine checks;
  this sweep is scored on ten. Both verdicts are stored per corner (`pass9`, `pass10`).

Guard Tier 5 is a statement about a *search*, and this is not one. Check 19 cannot fire
(each evaluator sees at most three records). Check 20 raises `SearchHalted`; it is caught,
recorded by name, and **counted as a failing corner**. It fired zero times.

### 21.3 The result

```
PVT-clean candidates (45/45 guard-valid and all ten checks)     1 of 22

why the other corners failed                    corners
  guard: T4.10_dc_gain_implausible                  130
  guard: T2.5_mosfet_not_in_saturation               89
  peak_in_band                                       30
  boost_target                                       19
```

**One design in twenty-two survives the full PVT envelope under the full requirement
set.** That number is the honest context for the flagship and is to be reported beside it,
not behind it. The two dominant failure modes are the same two this record has been
tracking since G3.2a/G3.2b — DC gain and saturation headroom — and they concentrate at
low supply and high temperature, which is where headroom is physically scarcest.

### 21.4 The delivered circuit

Spec 2 of the held-out seed-23 set: **target 8.920 dB boost over a 14.83 dB channel.**
Provenance is one line — PPO stage-1 rollout → G3.2 constrained refinement → this design,
unmodified. Nothing was re-optimized, repaired, or hand-tuned at any corner.

```
w_in   55.784 um      rs      3287.45 ohm
l_in    0.3921 um     cs      181.62 fF
i_tail 712.27 uA      r_load  2450.30 ohm
```

TT/1.8 V/27 °C: boost 9.166 dB, error 0.246 dB. **45 of 45 corners pass**, worst-corner
target error **1.081 dB** — inside the 1.5 dB tolerance at every corner, not merely at
nominal.

Worst case across all 45 corners, against each requirement:

```
                    min            max          requirement
boost_db          7.839          9.667          target 8.920 +/- 1.5
peak_freq_ghz     1.345          1.794          1.25 - 2.5 GHz
dc_gain_db        1.041          1.553          >= 0.0
hd3_db          -61.918        -53.081          < -30.0
noise_vrms      543.5 uV       718.7 uV         < 1.5 mV
power_w         1.777 mW       2.002 mW         < 15 mW
area_mm2        0.001002       0.001002         < 0.05
eye_h_ui          0.750          0.813          >= 0.4
eye_v_mv          614.2          720.6          >= 100
```

Every constraint but one clears with more than an order of magnitude of room. **DC gain is
the binding constraint** — 1.041 dB of margin above the 0.0 dB floor at ss/1.89 V/125 °C —
which is the same finding G3.2b reached from the other direction: boost was never the
currency, DC gain was.

The worst corners are all low-supply-and-hot: `fs|1.71 V|125 °C` (1.081 dB error),
`ss|1.71 V|125 °C` (1.028), `fs|1.80 V|125 °C` (0.807). The best is `fs|1.80 V|27 °C` at
0.038 dB.

### 21.5 What this does and does not establish

It establishes that a circuit the frozen architecture produced meets the complete Astera
requirement set — boost, peak placement, DC gain, HD3, noise, power, area, and both eye
metrics — across all five process corners, VDD ±5 %, and 0–125 °C, judged by the guard
layer with real HD3 and noise at every corner.

It changes nothing in section 20. The chance line is still negative, the paired precision
result is still post-hoc-supported, and a clean sweep does not upgrade either. It is also
**not** a claim that SILQ produces PVT-robust designs in general: 1 of 22 did. The system
was never trained or scored at corners — every optimization in this repo ran at TT — so
the corner result is an out-of-distribution measurement, and it reads like one.

### 21.7 The frozen manifest, and the site

`results/delivered_circuit.json` is the single file anything downstream quotes from. It is
built by `silq.experiments.freeze_delivered`, which **measures nothing** — it reads the
artifacts that already exist, checksums them, and refuses to run if the sweep is
incomplete, if the flagship is not PVT-clean, or if the swept design is not the arm-B
design for its spec. `--verify` rebuilds and diffs, exiting non-zero on drift.

It carries: the six design values; the spec (seed 23, index 2, target and channel); the
provenance (policy sha256 `8868965260d7d169…`, PPO stage-1 result, every G3.2 step with
its boost, the budget spent); the measurement instrument including the AC grid the numbers
were taken on; all 45 corners plus the per-metric worst case; and sha256 of every input.
The delivered design is **frozen — no further optimization of this candidate.**

One trap it records rather than leaves lying: the `.ac` line inside the exported netlist
is `circuits.ctle`'s default viewing sweep, **not** the sweep that produced these numbers
(`AC_DECADE_PTS = 40`, 1 MHz–100 GHz). `peak_freq_ghz` is quantised onto that grid, so
re-running the exported deck as-is will not reproduce it exactly.

The site (`web/`, mirrored to `site/` by `scripts/build_site.py`) was updated to this
circuit: the PVT section, the metric ranges, the eye figure — regenerated at this spec's
own 14.83 dB channel, not the old 12 dB — and the readout card. Its provenance line is
now unambiguous, *PPO rollout → constrained G3.2 refinement → independently verified
across 45 PVT corners*, and the page states in the same paragraph that the optimisation
ran at TT only and that one candidate in twenty-two came back clean. The scope note
records the two things the artifact does not establish: the policy alone does not
demonstrate target-conditioned design, and strict solve counts do not clear their matched
chance null.

### 21.6 Reproducing this section

```
PYTHONPATH=src python -m silq.experiments.pvt_signoff
PYTHONPATH=src python -m silq.experiments.pvt_signoff --report-only   # no simulation
```

`results/pvt_signoff_seed23.json` holds every corner of every candidate; the run writes it
incrementally, so a killed sweep keeps its work. `results/pvt_signoff_flagship.spice` is
the delivered netlist, emitted only when the PVT-clean branch of §5 applies.
