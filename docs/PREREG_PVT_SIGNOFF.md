# Preregistration — final candidate selection and 45-corner PVT sign-off

Written **26 Aug 2026, before any PVT corner was simulated and before any per-spec value
in `results/final_comparison_seed23.json` was inspected.** The architecture is frozen
(`docs/REPRODUCE.md` §20, `docs/PREREG_FINAL_COMPARISON.md` amendment 2). Nothing in this
document may change a reward, a PPO hyperparameter, a design bound, a guard threshold, a
spec field, or any recorded number. This fixes only **which circuit we deliver** and
**what it has to survive**.

The point of writing it first is narrow and specific: the selection rule must not be able
to be chosen after seeing which candidate happens to survive the corners.

---

## 1. Why a preregistration is needed here at all

The seed-23 run produced one design per spec per arm. Picking "the final SILQ circuit"
from 40 of them, after measuring them, is exactly the cherry-pick this project has spent
its whole record avoiding. Two rules remove the freedom:

1. **The sweep is not selective.** Every candidate in the pool defined in §2 is swept over
   all 45 corners. No candidate is inspected before its sweep, and none is dropped after.
2. **The flagship is chosen by a formula written here**, applied to the sweep output, with
   deterministic tie-breaks.

The distribution over the whole pool is reported alongside the flagship. A single clean
circuit next to "and the other twenty-one were not" is a different claim from a single
clean circuit reported alone, and the record will carry the honest one.

## 2. The candidate pool

Source: `results/final_comparison_seed23.json`, **arm B only** (`PPO → G3.2`). Arm B is
the delivered architecture; arm A is the external baseline and arm C was dropped
(amendment 2 §2). Provenance of every candidate is therefore identical and stated in one
line: *PPO stage-1 rollout → G3.2 constrained refinement → the recorded design.*

A row enters the pool iff `row["b"]["strict_solved_at"] is not None`.

That is the project's strict criterion at TT/1.8 V/27 °C: the design passed all nine
`hard_pass` checks **and** landed within `boost_tol_db = 1.5` dB of the requested boost.
`row["b"]["best_design"]` is the design swept — `summarize()` defines it as the
minimum-`|boost − target|` design among that arm's `hard_pass` designs, so on a
strict-solved row it is necessarily itself strict.

The pool size is whatever that yields. It is **not** capped, trimmed, or sampled.

## 3. The sweep

For each candidate, all 45 corners of `silq.envs.pvt.corner_grid(spec, mode="full")`:

```
process   tt, ss, ff, sf, fs                     (5)
vdd       1.71, 1.80, 1.89 V   (nominal ±5%)     (3)
temp      0, 27, 125 °C                          (3)
```

Every corner is evaluated through `build_evaluator(..., fast=False)` — the **guard layer**
and **real** HD3 and noise, not the fast approximations. This is a stricter instrument than
`experiments/characterize.py`, which calls `measure_all` directly and never asks the guard.

The spec at every corner is the candidate's **own** spec, not `DEFAULT_SPEC`:

```python
dataclasses.replace(DEFAULT_SPEC,
                    target_boost_db=row["target_boost_db"],
                    channel_loss_db=row["channel_loss_db"],
                    boost_target_tol_db=1.5)
```

`boost_target_tol_db` is `None` everywhere else in this repo, which is why every published
number was scored on nine checks. Setting it here adds a tenth, `boost_target`, and makes
the sweep **stricter** than anything already recorded — the delivered circuit is claimed to
hit the requested boost, so it is scored on that. Both verdicts are stored per corner
(`pass9` without the tenth check, `pass10` with it) so the table can be read either way and
neither reading is retrofitted.

Recorded per corner: `guard_valid`, the rejecting check when invalid, all nine measured
quantities (`boost_db`, `peak_freq_ghz`, `dc_gain_db`, `hd3_db`, `noise_vrms`, `power_w`,
`area_mm2`, `eye_h_ui`, `eye_v_mv`), the per-check booleans, and `pass9` / `pass10`.

A corner where the guard rejects counts as **failing**. It is not retried, not re-seeded,
and not excluded.

## 4. Definitions

* **PVT-clean** — all 45 corners guard-valid **and** `pass10` true. Nothing weaker earns
  the word.
* **Worst-corner target error** — `max` over the 45 corners of `|boost_db − target_boost_db|`.
  A corner that is guard-invalid or non-convergent has no defined error and makes the
  candidate not PVT-clean by §4's first bullet, so this quantity is only ranked over
  PVT-clean candidates.
* **Corners passed** — the count of corners that are guard-valid and `pass10`, out of 45.

## 5. The flagship rule

Applied in this order to the sweep output:

1. Restrict to **PVT-clean** candidates.
2. Rank ascending by **worst-corner target error**.
3. Ties within 0.01 dB: rank ascending by the candidate's TT `best_abs_err` from the
   seed-23 artifact.
4. Still tied: lowest spec index.

The rank-1 candidate is the delivered SILQ circuit. Its netlist, its spec, its 45-corner
table and its provenance go to the site and the writeup together.

**If no candidate is PVT-clean**, that is the result and it is reported as the headline,
not buried:

> No candidate from the frozen architecture survived all 45 corners under the full
> requirement set.

and then, clearly labelled as **best available, not sign-off clean**, the candidate with
the most **corners passed** (ties by worst-corner target error over the passing corners,
then lowest spec index) is reported with its failing corners and failing checks named. This
branch is written here **in advance** so that reporting a non-clean circuit is not a
post-hoc retreat and does not require inventing a criterion after the fact.

## 6. Explicitly forbidden, whatever the sweep returns

* No re-optimization, re-tuning, or repair of any candidate at any corner. The delivered
  circuit is the one the frozen architecture produced, unmodified.
* No threshold moved — not `hd3_db_max`, `noise_vrms_max`, `power_w_max`, `area_mm2_max`,
  `eye_h_ui_min`, `eye_v_mv_min`, `dc_gain_db_min`, `boost_tol_db`, nor any guard bound.
* No corner dropped, no temperature narrowed, no `vdd_tolerance` reduced.
* No switch to `fast=True` to make HD3 or noise pass.
* No substituting `DEFAULT_SPEC`'s 9 dB target for a candidate's own target.
* No widening of the pool to arm A or arm C if arm B produces nothing clean. Arm A is the
  baseline; delivering its circuit would contradict the architecture we are claiming.
* No second spec seed drawn to look for a better candidate.

If any of these looks necessary after the data, it is reported as a finding and the
decision is the project owners'.

## 7. Artifacts

```
results/pvt_signoff_seed23.json     per-candidate, per-corner records; written
                                    incrementally so a killed run keeps its work
results/pvt_signoff_seed23.log      filtered stdout
results/pvt_signoff_flagship.spice  the rank-1 netlist, if §5 branch 1 applies
```

`experiments/characterize.py` writes `results/final_report.json`, which is **already**
the honest-benchmark final report and unrelated. It must not be run with its default
`--outdir` against this work. The sweep here is a new module and clobbers nothing.

## 8. What this does and does not establish

It establishes whether the circuit the frozen architecture produced meets the full Astera
requirement set across the required PVT envelope. It says nothing new about the research
claims in `docs/REPRODUCE.md` §20 — the chance line stays negative, the paired precision
result stays post-hoc-supported, and a clean sweep does not upgrade either.
