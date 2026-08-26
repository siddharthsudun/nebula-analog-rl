# Preregistration — final system comparison

Written before the benchmark is built or run. Everything below is fixed at commit time;
anything changed afterwards is an amendment and must be recorded as one, with the reason
and the date, not edited in place.

Nothing in this document alters the reward, the PPO hyperparameters, the design bounds,
the guard, `hard_pass`, or the frozen `seq_clean40k` model. The G3.2 controller
(`g32_repair.py`) and the H1 arm are used exactly as committed.

---

## 1. What is being compared

Three arms, on the same held-out specs, under the same total budget.

| arm | stage 1 | stage 2 |
|---|---|---|
| **A — PPO → H1** | PPO, k = 5 | H1 CMA-ES, up to 10 evaluations |
| **B — PPO → G3.2** | PPO, k = 5 | G3.2 constrained refinement, up to 10 evaluations |
| **C — PPO → G3.2 → H1** | PPO, k = 5 | G3.2 first; on stop-without-target, the **unused remainder** goes to H1 |

Arm C is the system under test. Arms A and B are its two halves run alone, so that any
difference can be attributed rather than assumed.

## 2. Budget — §13 accounting, unchanged

`measure_all` ceiling is **20 per spec**, as in every previous arm. Under §13 a PPO
evaluation costs 2.00 and a search evaluation costs 1.00, so:

```
stage 1   PPO k=5            5 x 2.00 = 10.00
stage 2   refinement         up to      10.00
                             ---------------
total                                   20.00
```

This is not a new allocation. It is what the frozen `g32_repair.py` already enforces
(`PREREG["budget_measure_all"] = 20`, `PREREG["r"] = 10`), and what H1 already used. No
controller constant changes to implement arm C.

Arm C spends from **one** pool. If G3.2 stops after 4 evaluations, H1 receives 6 — not 10.
A fallback that reached for a fresh 10 would be a 30-call system compared against 20-call
systems, which is not a comparison.

**Consequence, stated rather than hidden:** fallback-H1 is *not* the same as standalone H1.
It starts from a different design and gets fewer evaluations. Arm C is therefore a claim
about a **system under a fixed budget**, never a claim that H1 was improved. Arm A remains
the standalone H1 baseline, and the existing seed-3 H1 results
(`results/hybrid_audit_h1_seed3.json`) remain the historical baseline they already were.

## 3. Routing rule

```
run G3.2
    solver.reached_target ?
        yes -> stop, return best valid design
        no  -> budget remaining > 0 ?
                   yes -> hand best valid design + remaining budget to H1/CMA-ES
                   no  -> stop
```

That is the whole rule. There is **no threshold, no predictor, and no wall taxonomy in the
control path.**

### 3.1 Why the routing signal is `reached_target` and not `wall_hit`

The proposed design routed on an observed hard wall. Checked against the frozen seed-3 run
before adopting it, `wall_hit` does not carry that meaning:

| spec | strict solved | `wall_hit` | `reached_target` |
|---|---|---|---|
| 15 | no | **True** | False |
| 17 | **yes** | **True** | False |
| 9, 10, 16 | no | **False** | False |

`wall_hit` is set the first time *any* advance step lands outside the feasible set, whether
or not the bisection then recovers from it. So it is a *touched-the-wall* flag, not a
*stuck* flag. Routing on it would have sent a strict-solved spec (17) to the fallback and
would **not** have sent three of the four failures (9, 10, 16). It is wrong on 4 of 10.

`reached_target` is what the controller actually asserts about its own outcome, and it is
an observable state of the run, not a count. Routing on it needs no wall definition at all,
which removes the hidden-threshold problem at its root rather than managing it.

### 3.2 Wall type is a diagnostic, never a control input

`wall_hit`, `blocked_by` and `reason` are recorded per spec and reported in §5 as secondary
outcomes. They explain *why* the fallback fired. They do not decide *whether* it fires.

### 3.3 One consequence, recorded now

`reached_target` requires `|boost - target| <= 0.25 dB` (`PREREG["stop_abs_err_db"]`), which
is tighter than the 1.5 dB strict criterion. So a spec can be strict-solved and still route
to the fallback — spec 17 is exactly this case. This is accepted, not corrected: the
fallback can only add evaluations, best-of-run is kept across both stages, and tightening
or loosening `stop_abs_err_db` to tidy this up would be tuning a frozen controller constant
to suit a benchmark.

## 4. Held-out set

* **Spec seed: 23.** Seeds 0, 1, 2 and 3 appear in committed artifacts and are burned.
  Seed 2 specs 0–7 calibrated G3.2's plane; seed 2 specs 8–23 were used by
  `g32a_representative`; seed 3 specs 8–17 are G3.2's development slice. Seed 23 has been
  used for nothing.
* **n = 40** specs, `target_audit.make_specs(40, 23)`, all of them, in order. No spec is
  dropped for any reason after the fact.
* All three arms run on the **same 40 specs** (paired), so per-spec differences are
  comparable.

### 4.1 What n = 40 can and cannot resolve — stated in advance

n = 40 is chosen for the primary endpoints, which have large effects and tight
distributions (seed-3: median target error 0.07 vs 0.50 dB; stage-2 evaluations 4.1 vs
10.0). It is **not** powered to resolve a small difference in strict solve count. The
seed-3 result was 6/10 vs 5/10 — a difference of one spec — and n = 40 will not turn a
few-point solve-rate gap into a defensible claim. Solve counts are reported as secondary
and descriptive. **G3.2 is not to be described as superior to H1 on the strength of solve
count in this comparison**, at this n or any n it is likely to reach here.

## 5. Outcomes

### Primary
1. **Median absolute target error** `|achieved - requested|` over specs with at least one
   guard-valid design.
2. **`measure_all` spent** — mean and max, and stage-2 evaluations separately.
3. **Target-error reduction from the PPO handoff**, per spec: `|err_handoff| - |err_final|`.
4. **Per-method matched chance line**, `k = min(distinct rounded boosts, n_loose_pass)` —
   mandatory, as in every previous comparison. For arms B and C, distinct designs are
   pooled across stage 1 and *all* of stage 2 including any fallback portion.

### Secondary
5. Strict solves (`hard_pass` and `|err| <= 1.5 dB`).
6. Loose solves (`hard_pass`, target unscored).
7. Distinct designs *k*.
8. Failure/wall type: `reason`, `blocked_by`, whether the fallback fired, evaluations it got.

### 5.1 Outcome 3 is undefined for some specs, and is reported that way

`err_handoff` requires a guard-valid handoff. On the seed-3 slice **3 of 10 handoffs were
guard-invalid**. Rule, fixed here: outcome 3 is computed **only over specs with a
guard-valid handoff**, and the number excluded is reported beside it in every table. It is
never imputed, and a guard-invalid handoff is never scored as zero reduction.

## 6. Standing constraints that apply to this run

* No reward, PPO hyperparameter, design bound, guard or `hard_pass` change.
* No retraining, no seed or spec cherry-picking, no change to the target distribution.
* `l_in`'s lower bound is the process minimum and does not move.
* If a check fires or an ambiguity appears, it is reported, not resolved by changing a
  bound.
* Per-spec incremental writes, so a crashed run is a partial result and not a lost one.
* No arm's numbers are quoted beside another's unless both ran under this document.

## 7. Open decisions, blocking the run

Recorded here so they are answered before rather than during:

1. **n = 40 and spec seed 23** — proposed above; changeable, but only before the run.
2. Whether arm A should also be re-run on seed 23 (it must be, for a paired comparison) or
   whether the existing seed-3 H1 numbers are considered sufficient. This document assumes
   **re-run**.

---

# Amendment 1 — 26 Aug 2026

Recorded as an amendment rather than edited in place, per this document's own rule. All of
it is fixed **before** any spec of seed 23 is simulated. Items 1–2 are the reporting
additions asked for when the run was approved; item 3 is a disclosure; item 4 pins down a
rule §3 left implicit; item 5 is a gate on the code that implements the arms.

## 1. Additional reporting (requested at approval)

Added to §5, for **all three arms**, alongside the outcomes already listed:

* **Results by target tercile.** Terciles cut at the 1/3 and 2/3 quantiles of *this* set's
  40 requested targets, using `hybrid_report.terciles` unchanged.

## 2. Arm C's value proposition, reported explicitly

Added to §5, for **arm C only**:

* How often G3.2 solved the spec **before** any fallback (`reached_target` true, fallback
  never fired), and
* when it did not, **how often H1 recovered it** with the remaining budget — recovery
  counted at both the strict ±1.5 dB criterion and as target-error reduction in dB.
* Fallback frequency, the number of evaluations the fallback received, and the recorded
  `reason` / `blocked_by` for every spec on which it fired.

## 3. Disclosure — seed 23 was not entirely untouched

The approval said seed 23 must remain "completely untouched … no inspection". Before that
instruction was given, while proposing seed 23, one command was run:

```
make_specs(40, 23) -> printed first 3 (target, channel) pairs, and min/max of each column
```

That is the whole of it. **No circuit was simulated, no design was drawn, and no constant
in this document or in any controller was chosen after or because of it** — every constant
in play (`k`, `r`, `sigma0`, `step_cap`, `stop_abs_err_db`, the plane, the ladder) was
already frozen in committed code before seed 23 was named. The specs themselves are an
i.i.d. draw from the same `U(5,11) x U(8,16)` the whole project uses, so the printed range
carries no information that could bias an arm.

It is recorded here rather than left out because §4 claims seed 23 "has been used for
nothing", and that sentence is now not literally true. The set is still held out in the
sense that matters — nothing has been evaluated on it — but the choice of whether that is
good enough belongs to the reviewer, not to me. Switching to a genuinely unseen seed costs
nothing before the run and everything after it.

## 4. Arm C's fallback starting point — fixed here because §3 left it implicit

§3 says the fallback receives "best valid design". Made precise, using only rankings the
frozen controller already applies and inventing none:

1. Over **every guard-valid design the arm has evaluated** (stage 1 and G3.2 combined):
   `hard_pass` designs first, ranked by `|boost - target|`.
2. If none passes `hard_pass`: guard-valid designs ranked by `(number of failing checks,
   |boost - target|)` — the Case-2 start rule, unchanged.
3. If **no** guard-valid design exists anywhere: the PPO handoff `x`, which is what
   standalone H1 starts from in exactly this situation. The guard-invalid case is
   therefore given no special mechanism of its own.

CMA-ES `sigma0` is 0.05 and its seed is `20260823 + spec_index` — the same constant and
the same stream arm A uses, so the fallback differs from arm A only in start point and
budget, as §2 already states.

## 5. Equivalence gate on the implementation — runs before seed 23

The three arms share one 20-`measure_all` pool, so arm C cannot be assembled by running two
scripts back to back; it needs one process holding the budget. The G3.2 solver and the
CMA-ES refinement both live inside `main()` in their own files and cannot be imported.

They are therefore **transcribed** into `final_comparison.py`, for the same reason
`g32_repair.py` gives for re-implementing G3.1's advance loop rather than importing it:
`results/g32_repair_smoke.json` and `results/hybrid_audit_h1_seed3.json` must stay
reproducible from **unmodified** committed code. No frozen file is edited.

Transcription can drift, so it is not asserted, it is tested. Both controllers are fully
deterministic given the spec index, so:

> **Gate.** `final_comparison.py` runs arms A and B on **spec-seed 3, specs 8–17** — the
> already-burned development slice, never seed 23 — and each spec's outcome is compared to
> the frozen artifact. The compared fields are `best_boost_db`, `best_abs_err`,
> `loose_solved_at`, `strict_solved_at`, `n_valid`, and for arm B additionally
> `n_solver_evals`, `solver.reason`, `solver.case` and `solver.reached_target`.
>
> **Any mismatch stops the experiment and is reported.** Seed 23 is not touched until this
> gate passes, and the gate's output is committed beside the result.

---

# Amendment 2 — 26 Aug 2026, after the run

Recorded after the held-out result. Nothing here changes an analysis or a criterion; it
corrects one sentence that is no longer true, and records the decisions taken on the result.

## 1. Correction to §4

§4 says seed 23 "has been used for nothing". **That sentence is superseded.** The correct
statement, which is the one to use anywhere the held-out set is described:

> Seed 23 specifications were inspected once before the final run to verify the generated
> specification distribution; no simulations, design selection, hyperparameter changes, or
> protocol changes were performed as a result.

Seed 23 is a legitimate held-out set — nothing was evaluated on it before the run — but it
is **not** to be described as "completely untouched". Amendment 1 §3 carries the detail;
`docs/REPRODUCE.md` §20.5 carries it independently of this document.

## 2. Arm C is dropped from the final architecture

Not from the record — the arm ran, its result stands and is reported. It is dropped as a
*delivered component*, on its own numbers: 15 firings, 1 strict recovery, 2 first-feasible
finds, +0.000 dB median improvement on the remaining 12, at +2.07 mean evaluations. On 9 of
40 specs G3.2 had already spent the pool and there was nothing to hand over.

The reason is structural rather than incidental: under a fixed budget, a G3.2 that is
struggling is also a G3.2 that has spent the pool, so the fallback has least to give exactly
where it is needed. Recorded as a property of *this budgeted implementation*, not as a claim
that strategy-switching is a bad idea.

## 3. Post-hoc tests, permanently labelled

The paired comparison (B closer on 24 of 27, A on 2, tied 1, median +1.086 dB) is
descriptive and was preregistered as primary outcome 1. A sign test (p = 1.05e-05) and a
Wilcoxon signed-rank test (W = 15.0, p = 4.08e-06) were computed **after** the data and were
**not** named in this document. They are post hoc and are to be labelled as such wherever
they appear.

**The preregistered inferential test is the matched chance line, and it is negative for all
three arms.** No writeup may present a post-hoc test as the preregistered result, or use one
to revive the strict-solve-count claim §4.1 ruled out in advance.

## 4. The architecture is frozen

No G3.3, no further axis, no router, no reward change, no retraining, no new benchmark
criterion. The experimental line closes here. What follows is physical validation, the final
delivered circuit, the demo, and the writeup — none of which may alter a number above.
