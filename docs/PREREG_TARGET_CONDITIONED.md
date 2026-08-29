# PRE-REGISTRATION — Target-conditioned policy (H-TC)

**Committed before any training run, any audit run, and any simulation that this document
is used to judge.** The commit that introduces this file contains no `seq_tc*` model, no
`target_audit_*seed20260915*` artifact and no `final_report_seed20260915*` artifact. Check
`git log` on this file against the artifact timestamps rather than trusting the claim.

Nothing in `docs/REPRODUCE.md` sections 1–19 changes. The reward on the default code path,
the PPO hyperparameters, the design-space bounds, the `l_in` lower bound, the guard layer,
`specs.hard_pass`, the ±1.5 dB criterion and the frozen `results/seq_clean40k.zip`
checkpoint are all untouched. The new behaviour is opt-in behind a single flag that is off
by default; 22 tests in `tests/test_target_weight.py` pin that the default path is
bit-identical, and the 107 pre-existing reward/env tests still pass unchanged.

---

## 1. The question, and why we do not already have an answer

Every headline number in this repo is a **feasibility** number: the system finds a design
that clears all ten hard checks. The separate and harder claim — that the agent *aims*,
i.e. that asking for 9.1 dB produces something near 9.1 dB rather than merely something
legal — has never been cleanly tested.

It was tested once, with `--boost-tol`, and came back negative. **That result is
uninterpretable and we will not cite it as evidence either way.** The reason is a confound
in the code, not a judgement call:

- `equalizer_env._margins` adds a `boost_target` margin **only** when
  `spec.boost_target_tol_db` is set.
- `sequential_env._shaped` computes `passed = all(v >= 0 for v in mg.values())` over those
  margins, and `step` sets `terminated = passed`.
- So setting `--boost-tol` simultaneously (a) prices the target and (b) makes the episode
  refuse to end until the target is hit — and it also drops the old dense soft term.

One flag moved three things. A negative result from it cannot distinguish *"pricing the
target does not teach retargeting"* from *"we sparsified the terminal reward and the run
collapsed for an unrelated reason."* H-TC is the clean version of the same question.

**H-TC.** A policy whose per-step score prices the requested target densely, while
termination remains a function of feasibility alone, solves the strict ±1.5 dB retargeting
test on never-seen specs at a rate **above its own per-method matched chance line**.

---

## 2. The instrument — FROZEN HERE, BEFORE ANY RUN

`src/eqrl/envs/sequential_env.py`, opt-in third branch of `_shaped`:

```
TARGET_PRICE_REF_DB = 1.5                      # dB, frozen

err   = abs(m.boost_db - spec.target_boost_db)
t     = (TARGET_PRICE_REF_DB - err) / TARGET_PRICE_REF_DB
score = base + target_weight * clip(t, -2.0, +1.0)
```

`base` is the unchanged sum of clipped feasibility margins. `passed` is computed **before**
this term is added and is not a function of it.

Three properties, each pinned by a test:

1. **It never gates.** A feasible design stays `passed` at any target error and any
   weight; an infeasible design stays failed at a perfect target hit.
2. **It is dense.** It carries gradient across the whole 5–11 dB target band. The soft term
   it replaces is `min(|err|/3, 1)`-clipped and is identically zero beyond 3 dB — flat over
   roughly half the target range, which is a defect, not a design.
3. **It is scaled like every other check.** The same `[-2, +1]` clip the nine feasibility
   margins get, so `target_weight = 1.0` prices the target at *exactly one check's worth* —
   which is what this repo's own `_margins` docstring argued for, and what the confounded
   run could not test in isolation.

`TARGET_PRICE_REF_DB = 1.5` is deliberately the same 1.5 dB the external strict test uses,
so the priced gradient and the externally applied criterion agree about what "close" means.
**It sets the slope of a reward term. Nothing anywhere gates on it.** It is not a threshold,
and changing it would not loosen any pass/fail line — but it is frozen here regardless.

`target_weight` and `boost_tol` are **mutually exclusive**; the env raises `ValueError` if
both are given, because setting both would price the target twice and put it back inside
`passed` — the exact confound this flag exists to remove.

---

## 3. What does not change

Training is `python -m eqrl.agents.train_sequential` with **the manifest of
`results/seq_clean40k_train.json` reproduced exactly**, plus `--target-weight W` and a new
`--out`:

```
timesteps 40000   horizon 20   seed 0   n_envs 8   step_size 0.18
fast True   guarded True   shaped_invalid False   pvt False
target_range [5.0, 11.0]   channel_range [8.0, 16.0]
feasible_decode False   anchor_baseline False   anchor_noise 0.0
PPO: n_steps 128 (1024//8), batch_size 128, gamma 0.95, gae_lambda 0.95,
     ent_coef 0.005, learning_rate 3e-4, MlpPolicy
```

No PPO hyperparameter, no bound, no criterion, no guard threshold is touched. **The new
model is written to a new file. `results/seq_clean40k.zip` is not overwritten and remains
the scientific control.** The repo is committed and tagged before training starts
(REPRODUCE §2 records that the last model was trained on a dirty tree; that does not
repeat).

Evaluation is the frozen matched protocol of **REPRODUCE §11**, unmodified: 32 specs,
budget 20, guard-valid, loose = `hard_pass` with the target unscored, strict = loose **and**
|achieved − requested| ≤ 1.5 dB, `deterministic=True`, the same seeding. No method is told
the tolerance. The strict test is applied by the harness afterwards, exactly as before.

---

## 4. Spec seeds

Seeds already present in `results/` and therefore **contaminated**: **0, 1, 2, 3, 11, 23.**
(The earlier working note excluded 0, 1, 2, 3, 23 — it missed 11. 11 is excluded here.)

| role | seed | why |
|---|---|---|
| development | **3** | already spent, so re-using it costs nothing that is not already spent |
| **held-out, confirmatory** | **20260915** | never drawn. Fixed here as the Nebula submission date, so it is verifiably not a number chosen after seeing a result |

`target_audit.make_specs(n, seed)` draws from the identical distribution in the identical
order for every seed; only the draw changes. **The target distribution is not changed.**

The held-out seed is spent **exactly once**. If the confirmatory run fails, we do not draw
another seed. There is no second spec seed in this protocol.

---

## 5. Two stages, and which one may produce a headline

### Stage A — development. May not be quoted as a result.

Train at `target_weight ∈ {1.0, 3.0}`, everything else per §3. Both models evaluated on
**dev seed 3** only.

- 1.0 = the target is worth one check, matching every feasibility margin.
- 3.0 = the target spans the full clip range, i.e. clearly dominant over any single check
  but still a minority of the ~10-check budget.

**Selection rule, fixed now:** pick the model with the lower **median |achieved − requested|
dB** over the best loose-passing design per spec on seed 3; ties broken by strict solves on
seed 3. If wall-clock only allows one run, `W = 1.0` is the default and no selection occurs.

Stage A numbers are development numbers. They are reported, if at all, labelled as such,
and they are never the headline.

### Stage B — confirmatory. Run once.

The Stage-A-selected model, and the frozen `seq_clean40k` control, each evaluated on
**seed 20260915**:

```
python -m eqrl.experiments.target_audit --model results/seq_tc_<W>.zip \
    --spec-seed 20260915 --specs 32 --budget 20 --tol 1.5 --fresh-restarts \
    --out results/target_audit_freshrestart_seed20260915.json

python -m eqrl.experiments.target_audit --model results/seq_clean40k.zip \
    --spec-seed 20260915 --specs 32 --budget 20 --tol 1.5 --fresh-restarts \
    --out results/target_audit_freshrestart_ctl_seed20260915.json
```

---

## 6. Primary endpoint — ONE test, fixed now

**The `--fresh-restarts` PPO arm's strict solve count on seed 20260915, tested against its
own per-method matched chance line** (`final_report.py`, "PER-METHOD MATCHED CHANCE LINE"),
one-sided **p < 0.05**.

The line is computed exactly as that code already computes it, with no change:

- pool = the **frozen** `results/target_tracking_clean40k.json` (26 achieved boosts,
  sha256 recorded in REPRODUCE §12). **The pool is NOT augmented with the new model's
  designs.** Adding a target-conditioned model's own outputs to the null it is judged
  against would be circular, and is forbidden here.
- per spec, `p` = fraction of the pool within ±1.5 dB of *that* target;
- `k` = **distinct** feasible designs that arm produced for that spec;
- null expectation = `Σ 1 − (1 − p)^k`, p-value from 2000 draws of that Poisson-binomial.

`--fresh-restarts` is the primary arm because without it the PPO arm replays one identical
trajectory and is under-credited (REPRODUCE §8). The non-restart arm is a **secondary** and
is reported whichever way it lands.

### The bar, computed before the run

On the frozen `seq_clean40k` `--fresh-restarts` arm (seed 0), mean `k` = 3.41 and the
matched line is **16.91 / 32**; it observed **16 / 32**, p = 0.76 — indistinguishable from
not reading the target at all. Sweeping the observed count against that same k-profile
(20,000 null draws, `default_rng(20260823)`):

| observed | 16 | 18 | 20 | **21** | 22 | 23 |
|---|---|---|---|---|---|---|
| p | 0.760 | 0.388 | 0.102 | **0.037** | 0.010 | 0.002 |

**So the bar is roughly 21/32 — five more strict solves than the control.** The exact bar on
seed 20260915 will be recomputed from that run's own `k`, because that is what the matched
protocol does; this table is stated here so that nobody, including us, can pretend
afterwards that the bar was lower than it was.

**Anticipated objection, declared now:** a target-conditioned policy that converges faster
produces *fewer* distinct feasible designs, which lowers its own `k` and therefore lowers
its own chance line. That is the matched protocol working correctly — it credits aim, not
volume — but it is also a route to clearing the bar without aiming better. **Sensitivity
analysis, pre-declared:** we additionally report the same test with `k` held fixed at the
control's mean (3.41) for every spec. If the result clears the line only under its own `k`
and not under fixed `k`, we say exactly that.

---

## 7. Secondary endpoints — labelled secondary, reported whichever way they land

1. Median and mean |achieved − requested| dB over the best loose-passing design per spec.
2. Pearson corr(requested, achieved) on the **unfiltered** trajectory population — every
   evaluation, not the passing subset (REPRODUCE §11) — against the existing permutation
   null.
3. Loose (feasibility) solve rate, to show whether the flag cost feasibility.
4. Paired McNemar of TC vs `seq_clean40k` control on the same 32 specs.
5. Evaluations-to-first-strict-solve.

---

## 8. Kill switch and stopping rules

- **Training aborts** if the Stage A run's invalid rate exceeds the control's
  20,284 / 44,984 = 45.1% by more than 10 absolute points, or if `n_sims` diverges from
  ~45k by more than 25%. That would mean the flag changed the search dynamics rather than
  the objective, and the comparison would not be like-for-like. Report it; do not tune
  around it.
- **The project stops at one confirmatory run.** No second held-out seed, no re-training
  after seeing seed 20260915, no re-selection of `W` after Stage B.
- **If Stage B does not clear p < 0.05, H-TC is not supported and we write that.** The
  report says: SILQ demonstrates verified feasible design synthesis; a target-conditioned
  reward did not produce statistically demonstrable retargeting under this budget. That is
  a publishable and honest finding, and it is the finding we will publish if it is what we
  measure.

---

## 9. What may not be done under this pre-registration

No moving of any gate after data. No test-set tuning. No drawing a second spec seed. No
augmenting the chance pool. No change to the reward, PPO hyperparameters, design bounds,
guard thresholds, `hard_pass`, the ±1.5 dB tolerance, or the target distribution. No
overwriting of `results/seq_clean40k.zip`. No quoting of a Stage A number as a headline. No
report of "N/32 strict solves" without its matched chance line beside it — a raw strict
count is not a result.

Both claims stay separate in every write-up, as they always have:

- **feasibility** — verified, strong, already established;
- **retargeting** — this experiment, whose outcome is unknown at the time of writing.

---

## 10. Known weaknesses of this test, stated up front

- The chance pool is n = 26 and is drawn from this system's own achieved boosts. It is
  self-referential and, as REPRODUCE §10 notes, mildly generous to PPO. The sanity check is
  that random search lands on its own line, and it does.
- The training env runs `fast=True`, which stubs HD3 and noise; the audit env runs
  `fast=False`. That train/test mismatch is pre-existing (REPRODUCE §14), is identical for
  the control, and is not introduced here.
- 32 specs is a small confirmatory sample. A near-miss at p ≈ 0.06 will be reported as a
  near-miss, not rounded into a claim.
