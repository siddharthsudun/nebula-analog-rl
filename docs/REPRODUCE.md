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
NGSPICE_LIBRARY_PATH  %USERPROFILE%/eqrl-ngspice/Library/bin/ngspice{}.dll
SPICE_LIB_DIR         %USERPROFILE%/eqrl-ngspice/Library/share/ngspice
PDK_ROOT              %USERPROFILE%/pdk
PATH                  prepend  eqrl-ngspice/shim ; eqrl-ngspice/Library/bin
PYTHONPATH            src      (eqrl is NOT pip-installed into .venv)
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
./.venv/Scripts/python.exe -m eqrl.experiments.policy_rollout \
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
   `env.step()`, which simulates at `sequential_env.py:295`, and then `evaluate()`, which
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
   is +0.114. Use `eqrl.experiments.target_audit` instead, which records every spec.

## 5. Audit tooling added

Measurement only. Neither changes a reward, a bound, a hyperparameter or a criterion.

- `eqrl.experiments.target_audit` — unbiased target tracking. Records every spec's full
  trajectory whether or not it succeeds, and runs PPO and random search on the **same**
  specs, budget and validity test.
- `eqrl.experiments.chance_baseline` — empirical spec-blind expectation, computed from the
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
| `src/eqrl/guards.py` | thresholds and PDK bounds, owned by the project |
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

`eqrl.experiments.target_audit`, 32 specs, budget 20 evaluations per method, identical
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
| evaluator | `eqrl.evaluator.build_evaluator(DEFAULT_SPEC, corner="tt", fast=False, channel_loss_db=<spec's channel>)`, one guard cached per channel |
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

`eqrl.experiments.simcount_audit` counts at two chokepoints: `measures.measure_all` and
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
