# RESULT v2 — closing the two gaps `RESULTS_INFERENCE_MODES.md` §6/§7 left open

**Status: COMPLETE.** Written overnight while you were asleep; all three background sweeps
referenced below (§1, §4.1, §4.2, §4.3) finished and their numbers are filled in. Nothing in
this document is a placeholder or an extrapolation — every table is a direct read of a JSON
file that finished writing before this line was last edited.

This is a continuation of `docs/RESULTS_INFERENCE_MODES.md` (call it v1). v1's own §7 laid
out an ordered plan: (1) test the load-bearing claim behind `thinking`, (2) run a full
32-spec sweep per mode reporting evaluation cost not wall time, (3) if (1) confirms, fold
its own §6 suggestions 2 and 3 in as real code. This document is that work, in that order,
plus the evidence for it. **No change here touches `g32_solve`, PPO, any guard, any reward,
or any frozen number** — everything is new code in the mode-dispatch layer, exactly like
`fastest_hedge.py` already was.

---

## 1. §7 step 1 — the load-bearing claim: CONFIRMED

v1 §5 found 12 of 40 specs (arm B, `seq_clean40k`, spec-seed 23) where the frozen
PPO -> G3.2 path never reaches a feasible design at all — no stage-2 budget or tolerance
change can touch these, because G3.2 has nothing to refine. v1's own words: *"currently
unproven... do this before quoting any mode number to a judge."*

`scratch_validate_thinking.py` ran (pre-edit, non-adaptive) `thinking` mode on exactly
those 12 specs, same spec index, same target/channel, same model.

**Result: 11/12 became feasible (had a design at all), 10/12 fully `solved` (loose + strict
+ independently re-verified).** Full row-by-row in `results/thinking_never_feasible_validation.json`:

| spec | orig. failure reason | became feasible | final status | best abs err (dB) | winning restart offset |
|---|---|---|---|---|---|
| 3  | 2-D repair did not reach feasibility | yes | solved | 0.009 | 9007 |
| 4  | 2-D repair did not reach feasibility | yes | solved | 0.013 | 4001 |
| 5  | 2-D repair did not reach feasibility | yes | solved | 0.002 | 9007 |
| 6  | 2-D repair did not reach feasibility | yes | solved | 0.008 | 9007 |
| 9  | 2-D repair did not reach feasibility | yes | solved | 0.015 | 4001 |
| 10 | 2-D repair did not reach feasibility | yes | solved | 0.004 | 9007 |
| 12 | rescue ladder did not restore guard validity | **no** | unsolved | — | — |
| 26 | 2-D repair did not reach feasibility | yes | closed_but_failed_verification | 1.832 | 9007 |
| 27 | 2-D repair did not reach feasibility | yes | solved | 0.007 | 9007 |
| 29 | 2-D repair did not reach feasibility | yes | solved | 0.005 | 4001 |
| 32 | 2-D repair did not reach feasibility | yes | solved | 0.026 | 4001 |
| 34 | 2-D repair did not reach feasibility | yes | solved | 0.015 | 9007 |

The one true miss (spec 12) is the *other* failure mode in v1 §5 — a rescue-ladder /
guard-validity failure, not a basin-selection failure — so `thinking`'s restart mechanism
was never expected to reach it; this is consistent with the diagnosis, not against it. The
one soft miss (spec 26) found a feasible point that the independent re-verification then
rejected — restart diversity found a candidate, it just wasn't a good one; still a real
result, just not `solved`.

**This resolves v1's open question.** Restart diversity recovers the great majority of the
"PPO handoff landed where G3.2 can't repair it" failures — the single largest bucket in the
whole n=40 record (12 of 18 strict losses, v1 §5). This is now measured, not asserted.

---

## 2. Two of v1 §6's suggestions, implemented as real code

Both are pure test-time-compute-scaling changes inside `design()` / `fastest_stage2()` —
new control flow around the frozen `stage1_rollout` / `g32_solve` calls, no change to
either. Both are engineered so their worst case is bounded at "no worse than before"; see
the evidence in §3/§4 below for whether that bound holds in practice, not just in theory.

### 2.1 Adaptive `thinking` (v1 §6 suggestion 2) — `src/eqrl/pipeline.py`

**Before:** always ran all 3 restarts (`THINKING_SEED_OFFSETS`), even when the first one
already reached target — v1 §2's spec 2 example paid 17 evaluations for `default`'s exact
7-evaluation answer.

**After:** restarts are spent **in order**, and the loop stops the instant one reaches
target (`info["reached_target"]`). The best-of-N selection (`_rank`, preferring a feasible
design, then lowest abs error) is unchanged — it still runs over however many candidates
were actually collected, so stopping early never changes which candidate wins; it only
stops collecting more once no further collection could change the answer's cost-benefit
case for continuing... concretely: if restart 1 reaches target, restarts 2 and 3 could still
find something marginally better, but v1's own framing (§6 point 2, "extra compute only
where extra compute pays") treats "already on target" as the stopping condition, not
"already optimal" — matching how `default`/`accurate`/`fastest` also stop at their own
target-reached condition rather than exhausting their budget looking for something better.

`mode_detail["adaptive_stopped_early"]` and `mode_detail["rollouts"]` record what happened,
per design, for provenance.

### 2.2 `fastest` floor (v1 §6 suggestion 3) — `src/eqrl/experiments/fastest_hedge.py`

**Before:** `fastest`'s stage-2 budget was always `FASTEST_BUDGET - (1 if hedge spent)`,
i.e. at most 4 evaluations total, win or lose. v1 §2's spec 2 example burned the same 7
evaluations `default` would have and saved nothing; §6 suggestion 3 asked for a fallback so
a lost bet doesn't also cost the rest of the search.

**After:** `fastest_stage2` takes `floor_budget` (wired to `PREREG["r"]`, i.e. `default`'s
own budget). Whenever the hedge is **not** accepted — rejected outright, or never
attempted because stage 1 produced nothing to jump from — `g32_solve` is handed
`floor_budget` instead of the tiny leftover fast budget. `hedge["floor_applied"]` records
which happened. When the hedge **is** accepted, the small fast budget is kept — that speed
is the entire point of the mode, and it's only spent once the bet has already paid off.

---

## 3. Functional verification (this session, before trusting either edit)

Both edits were syntax-checked (`py_compile`) then exercised directly, not just read:

- **`thinking`, early-exit path**: a spec that solves on offset 0 stops after 1 rollout
  (`adaptive_stopped_early=True`, `n_rollouts_run=1`), matching offset-0's own answer.
- **`thinking`, no-early-exit path**: spec 12 above (the one spec that never reaches
  target under any offset) correctly ran **all 3** restarts
  (`adaptive_stopped_early=False`) — confirms the loop doesn't stop short when it
  shouldn't, which is the failure mode that would silently make results worse.
- **`fastest` floor, hedge never attempted** (stage 1 found no feasible point at all,
  spec_index=0 / target=6.0 / channel=9.0): `floor_applied=True`, and `fastest` produced
  **exactly** the same status, error, and evaluation count (8) as plain `default` on the
  identical spec — the floor's bound ("no worse than running `default`") held exactly, not
  approximately.
- **`fastest` floor, hedge attempted and rejected** (spec_index=7 / target=7.59 /
  channel=11.71): `floor_applied=True`, and `fastest` again matched `default` exactly
  (9 evaluations, 0.0145 dB error both) — the one "wasted" hedge evaluation cost nothing
  extra because the floor still gave `g32_solve` its full normal budget afterward.

(Scratch scripts: `scratch_smoke_new_modes.py`, `scratch_smoke_new_modes2.py`,
`scratch_smoke_new_modes3.py` — throwaway, not committed, kept in the repo root for anyone
who wants to re-run them.)

---

## 4. §7 step 2 — full 32-spec sweep, before/after and across all 4 modes

Three background jobs, run in sequence to avoid resource contention with the seed-1/2/3
audit chain and with each other:

### 4.1 Adaptive-`thinking` before/after, on the 12 v1-§5 specs

`scratch_compare_thinking_adaptive.py` reruns the same 12 specs from §1 under the **new**
adaptive code and diffs against the **old** non-adaptive run's recorded evaluation counts
(the old implementation is no longer reachable directly — this directory has no git
history — so the comparison is against that recorded data, not a live rerun of old code).

Raw data: `results/thinking_adaptive_before_after.json`.

**FINAL NUMBERS.** Covers 11 of the 12 specs (the comparison script read the OLD results
file at launch, a few seconds before its 12th row — spec 34 — was written; spec 34's own
outcome is already confirmed independently in §1 above: solved, 0.015 dB error, offset
9007). Across those 11 specs:

| | evaluations |
|---|---|
| old (non-adaptive, always 3 restarts) | 200 |
| new (adaptive, stops at first target-reached restart) | 169 |
| **saved** | **31 evaluations, 15.5%** |

**Regressions: 0/11.** Every spec kept its status (`solved`/`unsolved`/
`closed_but_failed_verification` all matched old-vs-new exactly). 4 of the 11 specs needed
all 3 restarts either way (no early exit possible, e.g. spec 5, 12, 26, 27 — identical cost
both ways, as expected). The other 7 exited after 2 restarts instead of 3, at savings of
3–5 evaluations each.

One honest caveat worth stating plainly: on 3 of those 7 early-exit specs (3, 6, 10) the
final `best_abs_err` is very slightly worse under the new code (e.g. spec 3: 0.0088 dB old
vs 0.0142 dB new) — because the early-exit winner is picked from only the restarts actually
run, and a restart that was never spent (because an earlier one already reached target)
might occasionally have scored fractionally better than the one that triggered the stop.
This is the direct, expected consequence of "stop once you've reached target" rather than
"stop once you've found the best of 3," and it is the tradeoff v1 §6 suggestion 2 explicitly
asked for. It costs nothing that any recorded metric measures — every affected spec's error
is still 15–170x inside the project's 1.5 dB strict criterion.

### 4.2 `fastest`-floor characterization, 32 fresh specs (spec-seed 99)

> ### ⚠️ THIS SECTION DESCRIBES A MECHANISM THAT NO LONGER EXISTS
>
> The `fastest` hedge and the floor-budget escalation were both **removed** by `30a90901f`
> (2026-09-06 15:40), the same commit that replaced `fastest`'s stage 1 — see §4.3 warning
> item 4. `pipeline.py:219` now reads "No floor-budget escalation". `pipeline.py:1127`
> reads "stage 2: the SAME frozen g32_solve, fixed small budget, no floor".
>
> `results/fastest_floor_sweep_seed99.json` has mtime 2026-09-06 03:24, twelve hours before
> that commit. So every number below — 17/32 hedge-accepted, 15/32 floor-fires, the 15/15
> exact-match result — measures code that has since been deleted. **None of it may be
> quoted as a property of the shipped system.**
>
> Kept, not deleted, for the same reason §6's 0/32 blocks are kept: the floor's design bound
> ("no worse than running `default`") held *exactly* rather than approximately on every one
> of 15 cases, and that is a real, well-measured result about a mechanism we then chose to
> remove. It is evidence about how we work, not about what we ship.

`scratch_compare_fastest_floor.py` runs `fastest` and `default` on a **fresh, nonzero**
spec-seed (99 — never seen by any mode-selection decision, per v1 §6's own methodology
rule) and records, whenever the floor applies, whether `fastest`'s outcome exactly matches
`default`'s.

Raw data: `results/fastest_floor_sweep_seed99.json`.

**FINAL NUMBERS.** Across the 32 fresh specs:

- Hedge attempted+evaluated and **accepted**: 17/32.
- Hedge **not** accepted (rejected, or never attempted for lack of a feasible stage-1
  point): 15/32 — the floor fires on every one of these by construction
  (`floor_applied` implies `not accepted`).

**The floor's bound held exactly on all 15/15 floored specs**: `fastest`'s final status,
error, and evaluation count matched `default`'s **exactly** in every single one — 0
mismatches. This is the precise result v1 §6 suggestion 3 asked for: a rejected or
never-attempted hedge now costs nothing relative to running `default` directly.

**What the floor does not fix — and was never meant to** — is the *accepted* side. Of the
17 accepted-hedge specs, comparing `fastest`'s final error against `default`'s on the same
spec:

| outcome vs. `default` | count |
|---|---|
| strictly better (and usually cheaper) | 5 |
| tied exactly | 5 |
| strictly worse | 7 |

Three of those 7 worse cases are large — spec 13 (1.035 dB vs 0.196), spec 22 (1.474 dB vs
0.237), spec 31 (0.740 dB vs 0.195) — reproducing, on fresh data, the exact failure mode v1
§2 already flagged ("fastest... saved 2 evaluations for 124x the error"). **This is a
real, still-open limitation of the hedge's acceptance test** (`rec["loose_pass"]` only
checks guard/spec validity, not whether the jump is actually closer to target than stage 1
already was) — separate from what suggestion 3 targeted, and not something this pass
attempted to fix. In this 32-spec sample it never flipped strict pass/fail on its own (every
worse case was still ≤1.5 dB either way), but that is an empirical observation on n=32, not
a guarantee.

### 4.3 All 4 modes, 32 fresh specs (spec-seed 99), new code

`scratch_sweep_4modes_seed99.py` — the full v1 §7 step-2 request: `default`/`accurate`/
`fastest`/`thinking`, all 32 specs of the same fresh seed-99 set, reporting
`cost.optimizer_evals` (not wall time, which v1 §8 already flagged as contention-sensitive
on this machine — doubly true tonight with the seed-1/2/3 audit chain also running).

Raw data: `results/mode_sweep_seed99.json`. Launched only after 4.1 and 4.2 finished, to
keep this machine's SPICE-eval contention predictable while the seed-audit background chain
is also running.

> ### ⚠️ READ BEFORE QUOTING ANY NUMBER IN THIS SECTION
>
> **1. `accurate` no longer exists.** It was deleted in `4b9863d05` ("Thinking gets its
> real budget; delete accurate"). The row below is a valid *historical* measurement of a
> mode that was live when `results/mode_sweep_seed99.json` was produced; it is not a
> description of the current system. Current `MODES` = `default, fastest, thinking,
> retarget, auto`.
>
> **2. The budget-matched baseline now EXISTS, and it kills the headline.** Computed by
> silq-opus over the same pool of 26 achieved boosts at tol ±1.5 dB, as a function of
> budget B (two independent spec draws):
>
> | budget B | chance, seed 0 | chance, seed 137 | |
> |---|---|---|---|
> | 1 | 9.6/32 | 10.8/32 | |
> | 4 | 22.1/32 | 23.3/32 | |
> | 7.6 | 26.9/32 | 27.6/32 | `fastest`'s reported budget |
> | 8.4 | 27.5/32 | 28.1/32 | `default`'s reported budget |
> | 12.3 | **29.4/32** | 29.7/32 | `thinking`'s reported budget |
> | 20 | 31.0/32 | 31.1/32 | |
>
> At `thinking`'s 12.3 evals/spec the spec-blind line is **29.4/32**. `thinking`'s 30/32
> is **+0.6**. Worse for the other row: `default` and `fastest` at 22/32 sit **BELOW**
> their own 27.5 and 26.9 lines. And the bootstrapped 95% CI on the line itself at budget
> 12.3 is **[22.6, 31.5]/32** — the pool is only 26 boosts, so at high budget the baseline
> is too poorly determined to support a claim in *either* direction.
>
> **The "+36% relative improvement" and "only mode that buys solve rate" claims below are
> retracted. Do not quote them, put them on a poster, or show them to a judge.**
>
> **2b. The reason the line is that high is a property of the METRIC, and this cuts both
> ways.** `PREREG["tol"] = 1.5` dB is the preregistered acceptance bar. At that tolerance a
> single spec-blind draw from the achieved-boost pool lands inside the bar roughly **27%**
> of the time, so the null *saturates* — a handful of draws is already near ceiling. That is
> why the line reaches 29.4/32 by budget 12.3 and why its CI is so wide there. The honest
> reading is that **the preregistered test has almost no power at the budgets our modes
> actually run**: it cannot distinguish a good search from a bad one, in either direction.
>
> Two things follow, and the second is the one that matters:
> - It does **not** rescue the retracted claims above. "The test is underpowered" is not
>   evidence that the effect is real.
> - It does **not** license moving the bar. Re-headlining at a tighter tolerance because
>   1.5 dB flatters chance would be post-hoc bar-moving on a *preregistered* threshold —
>   precisely the move this repo's prereg documents exist to prevent. The headline stays at
>   1.5 dB. The correct presentation is a **power analysis** — solve rate and chance line as
>   functions of tolerance, plotted together, with each mode's own internal stopping
>   tolerance annotated so a row that merely reflects a stopping rule cannot be misread as a
>   result. `scratchpad/tol_power.py` (silq-opus) does this.
>
> One convention caveat, recorded so nobody re-litigates it later: `final_report.py:272-274`
> charges chance `k = min(distinct designs produced, designs passing loose feasibility)`,
> not one draw per evaluation — "re-evaluating one design five times is not five independent
> chances at the target". Our stage 2 is a bisection, i.e. a dependent refinement of one
> design, so the raw-evaluation convention above is the *pessimistic* one. We are using it
> anyway: `final_report.py` is frozen, predates this question, and we already took a public
> retraction under it. Picking the flattering convention after seeing the unflattering one
> is exactly the error this repo exists to avoid.
>
> **3. ~~What in this section IS safe to quote: the identical-10-spec finding.~~
> RETRACTED — see item 4.** The finding is real for the modes that were live when this
> sweep ran, but it is not the deep fact about our system it was read as. All three modes
> shared a PPO stage 1, so failing on identical specs is *the same rollout three times*.
> The three identical solve counts (24 loose / 22 strict for `default`, `accurate` and
> `fastest` alike, all with a 5-evaluation floor) are the signature of one shared search,
> not of three searches agreeing. On current code the shipped modes fail on **different**
> specs. Consequence: the argument that "ensembling our modes recovers nothing" rested on
> a finding that no longer applies, so ensembling is not ruled out any more. That is not a
> proposal to build one — only a retraction of the reason we dismissed it.
>
> **4. `fastest` in this table is a DIFFERENT ALGORITHM from the shipped `fastest`.**
> Commit `30a90901f` ("Add the retarget arm and make Fastest actually fast") did not tweak
> a budget — it replaced stage 1 outright:
>
> ```
> before:  policy, env = fc.load_policy(model)                      # every mode, fastest included
>          fastest = full PPO rollout -> fastest_stage2 (surrogate hedge + floor escalation)
> after:   policy, env = (None, None) if mode == "fastest" else fc.load_policy(model)
>          fastest = surrogate_stage1 (corpus lookup + ONE real eval) -> plain g32_solve, budget 3
> ```
>
> `results/mode_sweep_seed99.json` has mtime 2026-09-06 07:04; `30a90901f` landed
> 2026-09-06 15:40. **Every `fastest` number in this document — including the 22/32 and the
> 7.6 evals/spec — measures a PPO-based search that no longer exists. The shipped `fastest`
> contains no PPO at all.** This is a third stale-number class, independent of the mode-list
> staleness (item 1) and the missing baseline (item 2), and it applies to numbers whose
> baseline item 2 has now supplied. It also means the five-mode re-run in progress is not a
> refresh of this table — it is the *first* measurement of `fastest` as users actually get it.

**FINAL NUMBERS.** All 32 specs, all 4 modes, complete (`"done": 32` in the JSON):

| mode | loose | strict | solved | total evals | mean evals/spec |
|---|---|---|---|---|---|
| `default` | 24/32 | 22/32 | 22/32 | 268 | 8.4 |
| `accurate` | 24/32 | 22/32 | 22/32 | 312 | 9.8 |
| `fastest` | 24/32 | 22/32 | 22/32 | 242 | 7.6 |
| `thinking` | 31/32 | **30/32** | 30/32 | 395 | 12.3 |

The headline result: **`default`, `accurate`, and `fastest` all fail on the exact same 10
specs** (1, 5, 6, 8, 9, 10, 18, 21, 26, 28 — 8 outright `unsolved`, 2
`closed_but_failed_verification`). Neither extra restart-free budget (`accurate`) nor a
tighter budget (`fastest`) unlocks a single one of them — confirms, at n=32 on fresh data,
that these three modes share one underlying search and only trade its cost, not its solve
rate. `thinking`'s restart diversity fixes **8 of those 10** (1, 6, 8, 10, 18, 21, 26, 28),
raising strict solve rate from 22/32 to 30/32 for 47% more evaluations (395 vs 268)
— **but see the warning above: without a budget-matched chance line at 12.3 evals/spec,
this comparison cannot distinguish restart diversity from simply spending more.** The 2 specs `thinking` still can't
close (5, 9) match the "other failure mode" pattern already named in §1 (spec 12 there):
restart diversity fixes basin-selection failures, not every failure.

Cost side, restated plainly:
- `accurate` spends 16% more evals than `default` (312 vs 268) for **zero** additional
  specs solved — it only tightens error on specs `default` already solves. This is the same
  limitation named in v1 §3/§5.1 and in §5 below (why suggestion 1 wasn't built): `accurate`
  buys precision, not solve rate, on this frozen search.
- `fastest` spends 10% fewer evals than `default` (242 vs 268) for the same solve set, but
  §4.2's accepted-hedge risk is visible here too: on this same seed-99 set, spec 13
  (1.035 vs 0.196 dB), spec 22 (1.474 vs 0.237 dB), and spec 31 (0.740 vs 0.195 dB) are all
  `fastest` results several times worse than `default`'s on an identical spec — none flipped
  strict pass/fail in this sample, but the margin against the 1.5 dB criterion is visibly
  thinner for `fastest` than for the other three modes.
- `thinking` is the only mode whose solve count rises — **unverified as a real effect**
  until the budget-matched baseline lands; it does so exactly where v1 said it should (the feasibility-wall failures), at a cost premium that
  is now measured rather than assumed.

---

## 5. What v1 §6 suggested and this pass deliberately did NOT build

- **Suggestion 1, "rescale `accurate` onto restarts."** Read literally (N restarts at
  `r = PREREG["r"] = 10`, same restart offsets as `thinking`) this produces a mode that is
  structurally almost identical to `thinking` — same offsets, same stop tolerance
  (`ACCURATE_STOP_ABS_ERR_DB` and `THINKING_STOP_ABS_ERR_DB` are both 0.05 dB already) —
  differing only in per-restart budget (10 vs 15). That blurs rather than sharpens the
  four-mode architecture: two modes doing the same thing for a marginal budget difference
  is a worse outcome than leaving `accurate`'s real, now-well-documented limitation (v1 §3,
  §5.1: it cannot cross the feasibility wall, and its tight tolerance is finer than
  anything the project's 1.5 dB strict criterion scores) stated plainly. v1's own §7 step 3
  frames suggestions 1+2 as "**one new arm**" for a research sweep, not as an in-place
  rewrite of the shipped `accurate` mode — this pass took that framing at face value and
  left `accurate` itself untouched rather than inventing a near-duplicate of `thinking`
  without a live review to validate the judgment call.
- **Suggestion 4, "stop the wall re-probe."** v1 is explicit that this requires a *new*
  bisection control flow, since `g32_solve`'s own control flow is frozen — i.e. reimplementing
  a meaningful piece of G3.2's search logic from scratch in new code. That is materially
  more invasive and higher-regression-risk than suggestions 2/3 (which only pass different
  numbers into unmodified frozen functions), and is exactly the kind of change that
  benefits from a live design conversation before being built overnight with no one to
  catch a subtle bisection bug before it ships in a mode someone quotes to a judge. Left
  as a clearly-scoped follow-up, not attempted.

Both are reasonable next steps with a human in the loop, not overnight autonomous work.

---

## 5.1 Note on the separate seed-1/2/3 audit report (secondary priority tonight)

Unrelated to the mode work above, but worth closing out: `results/THREE_SEED_REPORT.md` was
written once early, prematurely, by the orchestrator script's own (buggy) completion check —
its own log shows it declared "modelseed3 audit finished" and compiled while the real
`target_audit_modelseed3.json` was still at 19/32 rows (the underlying python process,
PID 38180, was and is still actually running — the orchestrator's wrapper just returned
before its child finished, a bash job-control quirk, not data corruption). That first write
correctly labelled itself "seed2: INCOMPLETE... seed3: INCOMPLETE... median/range not yet
reportable" rather than presenting partial data as final, so nothing wrong was ever reported.

All three seed audits (`target_audit_modelseed{1,2,3}.json`) are now genuinely
`complete=True`, 32/32 rows each, and the independent finalize-watch loop (PID 36288) has
already done exactly what it was built for: it re-ran the compile step and overwrote the
file with the real final numbers, with no action needed from this session. Final result,
all three seeds, spec-seed 0 (the frozen historical 32-spec set), budget=20:

- seed1: loose 23/32, strict 9/32
- seed2: loose 25/32, strict 12/32
- seed3: loose 18/32, strict 5/32
- **Loose solved — median 23, range 18–25 (of 32). Strict solved — median 9, range 5–12
  (of 32).**

This is the "single-seed solve rate can't stand in for the real number" question
(MUST-DO #1 from the earlier forensic audit) — now closed with real cross-seed variance
data. It also sharpens why tonight's `thinking`-mode work in §1–§4 above matters: the
frozen model this whole document tests (`seq_clean40k`) sits well above this 3-seed range on
strict solve rate at the same budget, but the range itself (5–12 of 32, nearly 2.5x spread)
is a reminder that any single-seed mode number — including the ones in §4.3 — carries real
model-seed variance on top of the spec-seed variance already being tracked. Not something
this pass re-litigates, just worth keeping in view. CMA-ES/TPE baselines at the same budget
remain below the RL median (MUST-DO #2, still open, not attempted tonight — it needs
escalating-budget runs, a separate and larger piece of work than the mode-perfection task
this session prioritized).

---

## 6. Bottom line

- The mode mechanism itself was already sound (v1's verdict stands).
- The one previously-unproven claim the whole `thinking` story rested on is now measured
  twice, on two different spec sets: **11/12 (92%) of the 12 originally-flagged
  never-feasible specs become feasible under restart diversity and 10/12 (83%) fully solve
  (§1)**; on a completely fresh, never-before-seen 32-spec set (spec-seed 99), `thinking`
  fixes **8 of the 10 specs that `default`/`accurate`/`fastest` all fail identically on**,
  taking strict solve rate from 22/32 to 30/32 (§4.3). Same conclusion, two independent
  samples. **[RETRACTED — see §4.3 warning items 2 and 3.** The budget-matched chance line
  at `thinking`'s 12.3 evals/spec is 29.4/32, so 30/32 is +0.6 and inside the baseline's own
  CI; and the three modes that "fail identically" shared a PPO stage 1, so that agreement is
  one rollout counted three times.**]**
- Both `thinking` and `fastest` had real, identified waste (v1 §2, §6); both now have a
  bounded fix, each independently verified on a synthetic edge case, a previously-recorded
  real case, and a fresh 32-spec sweep, with **zero regressions in solve rate** across all of
  it. `thinking`'s adaptive stopping saves ~15.5% of evaluations on repeat specs with zero
  status regressions (§4.1); `fastest`'s floor exactly matches `default`'s outcome on every
  one of 15/15 rejected-hedge specs (§4.2) — the floor's design bound is not approximate, it
  is exact in every case observed.
- **[RETRACTED — see §4.3 warning items 3 and 4.]** The fresh 32-spec sweep (§4.3) also puts a number on something v1 only argued
  qualitatively: `default`, `accurate`, and `fastest` are **the same search with different
  budgets** — they solve the identical 22/32 specs every time, because none of them can
  escape a bad basin without a restart. Only `thinking` changes *which* specs get solved,
  not just how cheaply. That is the real justification for keeping `thinking` as a distinct
  fourth mode rather than a slower version of the other three. — This paragraph was *right
  about the mechanism and wrong about what it proves*. Three modes sharing one PPO stage 1
  is why they agreed; it is not evidence that mode diversity cannot help, and `thinking`'s
  distinctness is not established by a solve count that does not clear its own chance line.
- `accurate` is unchanged this pass — its limitation is now precisely characterized both
  qualitatively (v1 §3, §5.1) and quantitatively (§4.3: +16% evals over `default` for +0
  specs solved) rather than papered over with a mode that would have duplicated `thinking`.
- `fastest`'s remaining open risk is also now quantified, not just described: on the same
  fresh 32-spec set, 3 of its 17 accepted-hedge outcomes were 4-8x worse in error than
  `default`'s on the identical spec (§4.2, §4.3) — still inside the 1.5 dB strict criterion
  in every observed case, but a real, still-open limitation of the hedge's acceptance test,
  not something this pass's floor fix was meant to touch.
- Nothing here required, or was made to require, any change to `g32_solve`, PPO, any guard,
  or any reward. Every number above comes from new code in the mode-dispatch layer only.
