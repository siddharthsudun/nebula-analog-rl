# RESULT — the four inference modes: they run, and they are scaled on the wrong axis

**Verdict: the mechanism is sound, the parameterisation is not.** All four modes execute
without error and none requires separate training. But the two knobs `accurate` turns are
measurably aimed at constraints that bind on a small minority of specs, while the failure
mode that dominates the frozen record — the PPO handoff landing where G3.2 cannot repair
it — is addressed only by `thinking`, and only incidentally.

Nothing here changes a reward, a guard threshold, a PPO hyperparameter, or any recorded
number. The frozen path is untouched; §6 proposes new, separately-named arms.

Artifacts: `results/final_comparison_seed23.json` (pre-existing, n=40, 885 simulations),
plus a 4-mode × 3-spec bench and two single-spec traces described in §2–§4.

> **Applicability — read before quoting §2–§4.** Every measurement in §2, §3 and §4 was
> taken against the pipeline as it stood **before 02:33 on 06 Sep 2026**. At that time
> `src/eqrl/pipeline.py` and `src/eqrl/experiments/fastest_hedge.py` were edited to
> implement suggestions 2 and 3 of §6 *in place*, rather than as the separately-named arms
> §6 calls for. `thinking` now stops at the first rollout that reaches target, and
> `fastest` now falls back to `PREREG["r"]` when its hedge is rejected. Consequently the
> The `thinking` and `fastest` rows of §2 **have since been re-measured against HEAD**
> (06 Sep, 13:5x) and §2 now carries the HEAD numbers, with the superseded pre-edit values
> shown beside them. The step trace in §3 is still the pre-edit trace and is marked as such.
> `default` is byte-identical and unaffected; `accurate` is unaffected; no recorded
> benchmark number in `results/` is affected, because `final_comparison.py` does not route
> through `design()`'s mode dispatch. §5 is unaffected — it reads a pre-existing artifact.
>
> Validation of §7 step 1 has since been run (`results/thinking_never_feasible_validation.json`):
> of the 12 never-feasible specs, **11 became feasible and 10 solved** under `thinking`.
> That closes the gap §7 step 1 identified. See §9.

---

## 1. What the modes actually are

From `src/eqrl/pipeline.py`. Every mode runs the **same** frozen PPO stage 1 and the
**same** frozen `g32_solve`; only the inference-time search budget differs.

| mode | stage-2 budget `r` | stop tolerance | stage-1 rollouts |
|---|---|---|---|
| `default` | `PREREG["r"]` = 10 | `PREREG["stop_abs_err_db"]` = 0.25 dB | 1 |
| `accurate` | `ACCURATE_R` = 30 | `ACCURATE_STOP_ABS_ERR_DB` = 0.05 dB | 1 |
| `fastest` | `FASTEST_BUDGET` = 4 (1 hedge + 3) | inherited | 1 |
| `thinking` | `THINKING_R` = 15 per rollout | 0.05 dB | 3, offsets `(0, 4001, 9007)` |

This is **test-time compute scaling on a fixed policy**. That is why no mode needs its own
training, and the claim is well supported rather than merely convenient: §4 shows the gain
comes from sampling the *existing* policy's stochasticity, not from a better policy.

`fastest`'s only asset dependency, `results/surrogate_corpus.npz` (18 MB, 374,588 records),
is present and loads in 1.50 s.

---

## 2. Do they work? Yes — 4 modes × 3 specs, all `solved`

Re-measured against HEAD on 06 Sep 2026 13:5x, same script, same specs
(`scratchpad/mode_bench.py`; the raw JSON it wrote is not in the tree and not on disk,
so the bracketed comparisons below cannot be re-derived from this repository). Where the 02:33
edits moved a number, the superseded pre-edit value is shown in brackets.

| spec | mode | optimizer evals | abs err (dB) | wall |
|---|---|---|---|---|
| 9.0 dB / 12.0 | default | 15 | 0.493 | 18.1 s |
| | accurate | 17 | 0.471 | 14.4 s |
| | fastest | 8 | 1.146 | 11.6 s |
| | thinking | **11** *(was 16)* | **0.031** | 25.0 s |
| 7.5 dB / 10.0 | default | 7 | 0.020 | 10.1 s |
| | accurate | 7 | 0.020 | 10.8 s |
| | fastest | 7 | 0.020 | 12.1 s |
| | thinking | **7** *(was 17)* | 0.020 *(was 0.007)* | 12.2 s |
| 10.5 dB / 14.0 | default | 11 | 0.004 | 16.6 s |
| | accurate | 11 | 0.004 | 17.7 s |
| | fastest | **11** *(was 9)* | **0.004** *(was 0.497)* | 18.6 s |
| | thinking | **11** *(was 21)* | 0.004 | 19.2 s |

**Wall times are not comparable between the two runs** and never were: the 01:06 run shared
the machine with three audits, the 13:5x run with four PVT sweeps. Only `optimizer_evals`
and `abs_err_db` are deterministic, which is why the conclusions below rest on those.

The cost column is `cost.optimizer_evals` = `k + stage-2 steps` (unit 1 of
`REPRODUCE.md` §13), not wall time.

Read directly off this table:

- **`accurate` still collapses onto `default` on 2 of 3 specs** — identical evaluation
  counts, identical errors. Unchanged by the edits, and the §1 conclusion stands.
- **`fastest` now saves nothing on 2 of 3 specs**: it is *evaluation-identical to
  `default`* on specs 2 and 3. Its only divergence is spec 1, where it saves 7 evaluations
  (8 vs 15) at 2.3× the error (1.146 vs 0.493). The floor fix cost it 2 evaluations on
  spec 3 and bought a **124× error reduction** (0.497 → 0.004) — the single largest
  improvement either edit produced.
- **`thinking` is now cheaper *and* more accurate than `default` on spec 1** — 11
  evaluations against 15, for 16× less error (0.031 vs 0.493). On specs 2 and 3 it is
  evaluation-identical to `default`. Early exit is confirmed firing
  (`mode_detail.adaptive_stopped_early = True`), so this is real work skipped, not an
  accounting change.
- **The predicted `fastest`-costs-more-than-`default` case did not occur here.** Across
  these 3 specs `fastest` never exceeded `default`; it matched it twice. The 11-vs-10
  worst case in §9 remains theoretical and unobserved — on n=3, which cannot rule it out.

n = 3 is too small to conclude from. §5 uses the n=40 frozen record instead.

---

## 3. Why `accurate` collapses — the wall, not the budget

Spec 0 (9.0 dB target) in `accurate`, with the solver's own exit reason and step trace:

```
reason        : converged onto the feasibility wall
stage2 steps  : 12    evals unspent: 18 of 30
final abs err : 0.471

 0 advance   boost=6.953   err=2.047
 1 advance   boost=7.496   err=1.504
 2 advance   boost=8.479   err=0.521
 3 advance   boost=None            <- guard / hard_pass wall
 4 refine    boost=None
 5 refine    boost=None
 6 refine    boost=None
 7 refine    boost=None
 8 refine    boost=None
 9 refine    boost=8.507   err=0.493
10 refine    boost=8.521   err=0.478
11 refine    boost=8.529   err=0.471
```

The boost axis runs out of feasible circuit at **~8.53 dB**; the target is 9.0. `accurate`
stopped with **18 of its 30 evaluations unspent** because there was nothing left to buy.
`ACCURATE_R` cannot cross a wall and `ACCURATE_STOP_ABS_ERR_DB` cannot either.

Note also steps 3–8: **six of twelve evaluations returned nothing**, the bisection
re-probing a wall it had already located.

---

## 4. Why `thinking` wins — a different basin, not a harder search

Same spec, `mode="thinking"`, per-rollout breakdown from `provenance.mode_detail`:

```
winner offset: 4001        reason: "start already on target"
  offset     0   best abs err = 0.471   reached=False   <- the same basin default uses
  offset  4001   best abs err = 0.031   reached=True
  offset  9007   best abs err = 0.621   reached=False
```

Two things matter here. Offset 0 **independently reproduces** the 0.471 wall of §3. And the
winning rollout reached target with **zero stage-2 refinement steps** — PPO's stage 1 landed
0.031 dB away on its own.

So the spread across restarts on a single spec is **0.031 / 0.471 / 0.621 dB — roughly 20×**,
and the outcome is dominated by which basin the policy happens to land in, not by how hard
it refines afterwards.

---

## 5. The n=40 evidence: where the frozen pipeline actually loses

From `results/final_comparison_seed23.json` — arm B (the frozen PPO → G3.2 path),
`seq_clean40k`, spec-seed 23, 40 specs, 885 simulations. This file already existed; no new
compute was spent to produce this section.

Arm B scores **loose 28/40, strict 22/40**. Every solver exit reason:

```
 15  target reached inside the feasible set
 11  2-D repair did not reach feasibility
  7  converged onto the feasibility wall
  5  budget exhausted
  1  start already on target
  1  rescue ladder did not restore guard validity
```

The 18 strict losses decompose as:

| count | what happened | exit reason |
|---|---|---|
| 11 | **never reaches feasibility** | 2-D repair did not reach feasibility |
| 1 | **never reaches feasibility** | rescue ladder did not restore guard validity |
| 3 | feasible, off-target | budget exhausted |
| 3 | feasible, off-target | converged onto the feasibility wall |

**Two-thirds of the losses happen before G3.2 gets a feasible point to refine at all.** No
stage-2 budget and no stage-2 tolerance can reach those 12 specs.

### 5.1 Pricing `accurate`'s two knobs against this

**`r`: 10 → 30.** Reachable only by the 5 specs that exited "budget exhausted":

```
spec 11  target  8.25 dB  abs err 3.101  wall_hit=False
spec 15  target 10.08 dB  abs err 1.753  wall_hit=False
spec 30  target 10.61 dB  abs err 0.744  wall_hit=True   blocked: guard T4.10_dc_gain_implausible
spec 31  target  8.11 dB  abs err 1.659  wall_hit=False
spec 36  target  9.70 dB  abs err 0.684  wall_hit=False
```

Spec 30 is already walled, so extra evaluations cannot move it. Closing 1.66–3.10 dB with
20 more bisection steps, after 10 already failed, is optimistic. Realistic yield: about
**one spec in forty**.

**`stop_abs_err_db`: 0.25 → 0.05 dB.** This has a real target — 12 of 40 specs stopped
inside that window (specs 0, 1, 2, 7, 8, 14, 16, 17, 18, 20, 33, 38). But **the project's
strict criterion is `abs err ≤ 1.5 dB`** (`summarize()` in `final_comparison.py`). Refining
from ~0.2 dB to ~0.05 dB is 6–30× tighter than anything that gets scored. It buys nothing
any recorded metric measures.

The six feasible-but-off-target residuals are 1.66, 1.70, 1.75, 2.78, 3.10, 4.09 dB — all
well outside the range a tolerance change addresses.

### 5.2 What this says about the right axis

The dominant failure is *"the PPO handoff landed where G3.2 can't repair it."* A different
rollout is a different handoff. That is the mechanism `thinking` exercises, and §4 is one
clean instance of it. **Restart diversity is aimed at the bucket that actually matters;
refinement budget is not.**

---

## 6. Suggestions

All of these are new, separately-named arms reported on a fresh **nonzero** `--spec-seed`.
None edits `g32_solve`, the guard, the reward, or any recorded number. The precedent for a
mode owning its own stage-2 without touching frozen code already exists: `fastest` does it
via `eqrl.experiments.fastest_hedge.fastest_stage2`.

1. **Re-scale `accurate` onto restarts rather than budget.** Same rough cost envelope,
   pointed at the constraint that binds. Concretely: N restarts at `r = PREREG["r"]`
   instead of 1 restart at `r = 30`.
2. **Make `thinking` adaptive.** Run rollout 1; spend rollouts 2 and 3 only if it did not
   reach target. On the §2 data this keeps the 16× win on spec 1 and skips the pure waste
   on spec 3, and it is the honest form of the test-time-compute claim: extra compute only
   where extra compute pays.
3. **Give `fastest` a floor.** It spent `default`'s full 7 evaluations on spec 2 and saved
   nothing. Either fall back to `default` when the hedge does not beat it, or report the
   accuracy it gives up alongside the evaluations it saves.
4. **Stop the wall re-probe.** Steps 3–8 of §3 are six evaluations spent re-discovering a
   known wall. A mode that records the wall's `t` and bisects only on the feasible side
   would return those evaluations to useful search. (New arm only — `g32_solve`'s control
   flow is frozen.)
5. **Do not tighten tolerance below the scored criterion** in any new arm. Precision below
   ~1.5 dB is invisible to every recorded metric.

---

## 7. Next steps, in order

1. **Test the load-bearing claim.** Run `thinking` on exactly the 12 never-feasible specs
   of §5 and count how many become feasible. This is the claim the whole mode story rests
   on and it is **currently unproven** — §4 is one spec. ~15 min on a quiet machine.
   *Do this before quoting any mode number to a judge.*
2. **Full 32-spec sweep per mode** on a fresh nonzero `--spec-seed`, with equal evaluation
   budget accounting per mode, reporting `optimizer_evals` and `measure_all` rather than
   wall time (wall time is contention-sensitive on this machine).
3. **If step 1 confirms**, build suggestions 1 and 2 as one new arm and re-run step 2
   against it.
4. **PVT as final verification only**, per `docs/PREREG_PVT_SIGNOFF.md` and the agreed
   post-freeze sequence — never per-refinement-step.

---

## 8. Caveats

- §2 is **n = 3 specs**, one of which was easy enough that every mode tied. It establishes
  that the modes run and that `accurate` collapses onto `default`; it establishes nothing
  quantitative.
- §3 and §4 are **single-spec traces**. They are mechanism, not rate.
- §5 is n = 40 and is the only section with a defensible denominator. It is measured on
  `seq_clean40k` at spec-seed 23; it has **not** been re-measured on the seed-1/2/3
  policies.
- Every mode reports `status: solved` even at 1.146 dB error (§2, spec 1 `fastest`).
  `solved` means `verify()`'s checks passed, not that the target was met.
- Wall-clock figures in §2 were taken while three other jobs shared the machine. Treat the
  evaluation counts as the cost signal, not the seconds.

---

## 9. Post-publication changes (06 Sep 2026, 02:33)

Recorded here so the document does not silently describe code that no longer exists.

**What changed.** `thinking` gained an early exit (`break` once a rollout reaches target,
with `k_eff` corrected to `k * len(candidates)` so cost accounting stays honest).
`fastest_stage2` gained a `floor_budget` parameter; when the hedge is *not* accepted,
`g32_solve` receives `PREREG["r"]` instead of the leftover fast budget.

**Freeze assessment.** `g32_solve`'s control flow is untouched — the floor passes a budget
*number*, exactly as `accurate` and `thinking` already do. `default` is byte-identical.
`final_comparison.py` does not call `design()`, so **no recorded benchmark number moves.**

**Measured effect.**

| change | artifact | result |
|---|---|---|
| adaptive `thinking` | `results/thinking_adaptive_before_after.json` | 146 → 125 evals over 8 specs (−14%), **0 regressions** |
| `fastest` floor | `results/fastest_floor_sweep_seed99.json` | in flight; floor applied on 4 of first 9 specs |
| both, §2 re-bench | raw JSON **not in tree** | `thinking` −31%/−59%/−48% evals on the 3 specs; `fastest` error 0.497 → **0.004** on spec 3 |

**Verdict after re-measurement (06 Sep, 13:5x).** Both edits are improvements on this
bench and neither needs reverting. `thinking` does strictly less work — early exit
confirmed firing via `mode_detail.adaptive_stopped_early` — and on spec 1 is now cheaper
*and* 16× more accurate than `default`. The `fastest` floor turned that mode's worst
result on this bench (0.497 dB) into its best (0.004 dB) for 2 extra evaluations.
Consequence 1 below was **not** observed: `fastest` never exceeded `default`'s cost across
the 3 specs. Consequence 2 **was** observed exactly as predicted, on spec 2 rather than
spec 3: 0.007 → 0.020 dB. Both figures sit far inside the 1.5 dB scored criterion, so
neither moves a score.

**Two consequences that need attention.**

1. **`fastest` can now cost more than `default`.** A rejected hedge spends 1 + 10 = 11
   evaluations against `default`'s 10. A mode labelled "Fastest" that is sometimes slower
   than "Default" is a defensible engineering choice and an indefensible demo. Either
   rename the mode, or cap the floor below `PREREG["r"]`.
2. **Adaptive `thinking` can return a marginally worse design**, since stopping at the
   first on-target rollout skips a later one that might score better — observed as
   0.0088 → 0.0142 dB on spec 3. Immaterial against the 1.5 dB scored criterion, but it is
   a behaviour change, not a pure saving.

**Outstanding:** re-run the §2 bench against HEAD and replace the `thinking`/`fastest` rows.
