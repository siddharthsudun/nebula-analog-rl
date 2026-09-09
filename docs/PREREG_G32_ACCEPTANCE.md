# Preregistration — `g32_acceptance`, constraint-aware restart seeding

Written 07 Sep 2026, **before the arm was run even once**. The arm's code existed when this
was written (it is small, and writing it was how its inertness at the defaults was
discovered); no comparison against `_reselect_for_requirements` had been executed.

---

## 1. Why this is preregistered, and why the honest answer is probably "no effect"

The arm is *expected to do nothing* on the numbers this project publishes. That is not
modesty, it is measured — see §3 — and it is exactly the situation where a preregistration
earns its keep. An arm that cannot move the headline is an arm whose author will be tempted
to find *some* cut on which it looks good. Fixing the comparison, the regime and the
success criterion in advance is what stops that.

This document therefore commits to the comparison **and** to the sentence that gets written
if the comparison comes back null.

---

## 2. What the arm actually changes

Exactly one line of behaviour, at `pipeline.py`'s `diverse_seeds` call:

| | `thinking` | `g32_acceptance` |
|---|---|---|
| stage 1 | 8 PPO restarts | identical |
| stage 2 | frozen `g32_solve` | identical |
| restart seed pool | filtered by `plausible_mask` | filtered by `acceptance_mask` |
| spec passed to seed selection | competition spec | **the user's spec** |
| verification | guarded evaluator + `hard_pass` | **identical** |

`acceptance_mask` adds exactly two checks to `plausible_mask`: `boost_db_min`/`boost_db_max`
and `area_mm2_max`.

**Only two, and this is a correction to an earlier count of five.** `plausible_mask` already
screened `dc_gain_db_min`, `peak_freq_lo_ghz` and `peak_freq_hi_ghz`. The first draft of the
reachable-set table asked which of the eleven `REQUIREMENT_FIELDS` are *computable* from the
corpus and never asked which were *already computed*.

**Power is deliberately excluded.** `sim/measures.py` computes power as
`vdd * srv.supply_current(...)` — a real SPICE call — and explicitly rejects the analytic
`vdd * dv.i_tail` shortcut. Filtering on an `i_tail` proxy would mean the search screening on
a different quantity from the one the verifier checks. The proxy figure is reported in
`scratchpad/g32_acceptance_viability.json` under a key labelled `PROXY ONLY` and is not used
as a filter anywhere.

**Composition rule, which is load-bearing:** the filter narrows the *pool*, and the existing
greedy farthest-point selection then runs *inside* that pool. It never ranks to a winner and
never diversifies before filtering. What restarts are measured to buy is **coverage** — more
distinct designs per spec, *not* better aim at the requested boost (`docs/REPRODUCE.md` §8,
"The restart arm, and what it does and does not show"). Coverage is exactly what an arm that
collapsed the pool before diversifying would destroy. So the rule is justified by the effect
that survives a chance-matched control, rather than by one that does not — see §2a, which
records what this paragraph used to say and why it was withdrawn.

### 2a. Retraction: the composition rule's first justification was a fabricated statistic

The paragraph above originally read:

> Restart diversity is the demonstrated mechanism — the same frozen policy solves 4/11
> specs at one rollout and 11/11 with restarts, at `n_surrogate = 0`.

**No such measurement exists in this repository.** There is no single-rollout arm anywhere in
the tree, and the only eleven-row artifact under `results/` is
`thinking_adaptive_before_after.json`, which compares *adaptive* restart stopping against
*fixed* three-restart stopping — not restarts against no restarts. Its actual contents:

| | old (always 3 restarts) | new (adaptive stop) |
|---|---|---|
| solved | 9 / 11 | 9 / 11 |
| unsolved | 1 / 11 | 1 / 11 |
| closed, failed verification | 1 / 11 | 1 / 11 |
| specs that spent all 3 restarts | — | 4 / 11 |

The "4" is the number of specs that consumed all three restarts. The "11" is the number of
specs in the file. Neither was ever a solve count, and the solve count they displaced is
9/11 — *identical under both arms*. The sentence was assembled from two fragments of
`docs/RESULTS_INFERENCE_MODES_V2.md` §4.1 that were not about solving at all.

This is the same failure `HANDOVER-2026-09-04.md` §1 records and retracts — a raw solve count
offered as evidence with no chance-matched control — reproduced inside the document whose
stated purpose is to prevent it. It is recorded rather than quietly deleted for that reason.

**What the restart arm actually shows**, from `docs/REPRODUCE.md` §8, which did run the
control:

| | seed 0 replay | seed 0 restarts | seed 1 replay | seed 1 restarts |
|---|---|---|---|---|
| strict solves (±1.5 dB) | 6 / 32 | 16 / 32 | 8 / 32 | 20 / 32 |
| matched chance expectation | — | 16.9 (p = 0.75) | — | 19.2 (p = 0.46) |
| mean distinct designs per spec | 3.2 | 10.0 | 1.6 | 3.7 |

The strict gain is paired-significant against replay (McNemar p = 0.0020 and p = 0.0005) and
is *fully accounted for by the extra distinct designs*. Restarts buy attempts, not aim.
That is a strictly weaker claim than the one withdrawn, and it still carries the composition
rule, because attempts are precisely what collapsing the pool would remove.

**A caveat that applies to this retraction as much as to the claim it retracts:**
`thinking_adaptive_before_after.json` is **not committed** to this repository, and neither is
`scratchpad/mode_bench5.json`. A reader cannot check either table against the tree. That is
the same defect as citing a number from an uncommitted sweep, and naming it here is not a
disclaimer — it is a debt. `results/target_tracking_clean40k.json`, which backs the chance
line in `docs/REPRODUCE.md` §11, *is* committed, so the third table above is checkable and
the first is not.

### 2b. The tolerance at which the strict criterion has any power at all

Re-scoring the same runs at stricter acceptance (`scratchpad/tol_power.py`, 28 specs)
shows the ±1.5 dB criterion is the wrong bar to argue over, in a way that cuts against this
project's own modes as often as for them:

| mode | t = 1.5 dB | matched chance | p | t = 0.5 dB | matched chance | p |
|---|---|---|---|---|---|---|
| `thinking` | 27 / 28 | 26.52 | 0.547 | 27 / 28 | 19.10 | < 0.0001 |
| `fastest` | 21 / 28 | 11.89 | 0.0001 | 21 / 28 | 4.84 | < 0.0001 |
| `default` | 16 / 28 | 25.01 | 1.000 | 16 / 28 | 16.35 | 0.672 |
| `retarget` | 16 / 28 | 25.01 | 1.000 | 16 / 28 | 16.35 | 0.672 |

At ±1.5 dB the chance line for a high-evaluation mode sits at 26.5 of 28 — the null is
saturated and the test cannot separate *any* method from *any* other. `thinking` scoring
27/28 there is not evidence of anything, and `default`/`retarget` scoring *below* their own
chance line is likewise a statement about the metric. The separating bar is t = 0.5 dB, which
is above every mode's bisection stopping tolerance and therefore disadvantages none of them.
Numbers from this table are quotable only with the tolerance attached.

---

## 3. The result that is already in hand, and which this prereg does not get to relitigate

`scratchpad/g32_acceptance_viability.py`, on the benchmark's own 32 spec draws (seed 137):

| regime | corpus kept | removed from the 512-row pool | seed sets changed |
|---|---|---|---|
| **competition defaults** | 173916 (46.4%) | mean 0.0, max 0 | **0 / 32** |
| area ceiling ÷10 | 173916 (46.4%) | mean 0.0, max 0 | 0 / 32 |
| area ceiling ÷50 | 57415 (15.3%) | mean 223.9, max 337 | 32 / 32 |
| boost range 6–9 dB | 64577 (17.2%) | mean 282.0, max 512 | 18 / 32 |
| both tightened | 19844 (5.3%) | mean 371.8, max 512 | 32 / 32 |

On the benchmark's 32 draws at the competition defaults the arm is **bit-identical to
`thinking`**, not approximately so.

- the analytic area supremum over the **whole** action space is 0.002227 mm² against a
  0.05 mm² budget — a 22× margin, and a *bound*, not a sample. `area` cannot bite at the
  default ceiling for any design the search can produce.
- the pool is the 512 rows nearest the target **in boost**, and all 32 draws lie inside
  [3, 12] dB with room to spare, so `boost_range` does not bite on them.

### 3a. Correction, twice: "inert on the benchmark" is not "inert at the defaults", and the boundary is not a boundary

The first draft of this section generalized the 0/32 to inertness at the default spec, full
stop. That is **false at the edges of the boost range**, and the error was caught by the
equivalence check in §4 rather than by the sweep — the sweep structurally could not see it.

The *second* draft then over-corrected: it bisected for the edge and reported a clean band,
**[3.0664, 11.2125] dB**. Bisection presumes the property is monotone in the target, and a
0.01 dB scan shows it is not. Both the precision and the shape were wrong. What is actually
measured:

| target | seeds vs `thinking` | sampling |
|---|---|---|
| 3.00 – 3.04 dB | **differ** (5 of 5) | 0.01 dB |
| 3.05 – 11.13 dB | **identical, no exceptions** | 0.01 dB below 3.60 and above 10.75; 0.05 dB between |
| 11.14 – 12.00 dB | **differ at 76 of 121** — ragged, not contiguous | 0.01 dB |

The identical points *above* 11.14 are 11.15–11.18, 11.20, 11.21, 11.27, 11.28, 11.30,
11.32 and 11.87. That raggedness is the finding, not noise in it.

The mechanism is the pool, not the mask: the 512 rows nearest the target *in boost* straddle
a range boundary when the target is close to one. At an 11.5 dB target, 68 of the 512 sit
above the 12 dB ceiling, and `boost_range` removes exactly those. But removing rows does not
*have* to move the selection — the greedy farthest-point step can pick the same designs from
the smaller pool. At a 3.05 dB target the mask drops 76 rows and the seeds do not change.
"Rows were filtered" and "the seeds moved" are separate events and only the second is
observable to a user.

**The 32 draws span 5.01–10.99 dB**, entirely inside the quiet interval. That is why the
sweep reported 0/32 — a property of how the benchmark samples targets, not of the arm.

The methodological point is recorded because it cost two corrections: a bisection *always*
returns a boundary, including when there is no boundary to return. It cannot report its own
inapplicability. The scan can, and did.

So at the defaults the arm is a no-op *on the benchmark*, and near the range edges it is a
small correctness improvement: it stops restarts being proposed from designs whose own
recorded boost lies outside the range the run will accept. This is a marginally *better*
result for the arm than the one first claimed, and it is recorded here because it was found
by a check written to falsify the claim, which is the only reason this document can be
trusted on the point.

**This is why the preregistered comparison below is run only under tightened requirements.**
Running it at the defaults would compare two provably identical numbers and report the tie as
evidence of something.

---

## 4. The preregistered comparison

**Null:** `_reselect_for_requirements` (`pipeline.py:594-641`) at
`REQUIREMENT_RESELECT_CAP = 4` — the mechanism already in the codebase for honouring a
tightened requirement, which re-verifies up to four alternative designs *from the same trace*
after verification fails.

**Not** a chance line. Chance is the wrong null here: the question is not "does this beat
random" but "does filtering the pool *up front* beat re-picking from the trace *afterwards*",
and only the second is the thing the arm proposes to replace.

- **Arms:** `thinking` + reselect (null) vs `g32_acceptance` + reselect (arm).
- **Specs:** the 32 draws at spec seed 137, unchanged.
- **Regimes:** the three tightened rows of §3. The defaults row is reported for completeness
  and is not part of the test.
- **Primary outcome:** number of specs whose delivered design satisfies the **user's** spec.
- **Secondary:** `optimizer_evals`, and the count of runs where reselect had to fire at all.
- **Ablation, run both ways:** constrained-pool-then-diversify (shipped) vs
  constrained-only (rank to the best-scoring admissible row, no farthest-point step). This is
  the comparison that tests whether the composition rule in §2 is doing real work rather than
  being an aesthetic preference.

---

## 5. What gets written if it comes back null

Verbatim, decided now:

> Constraint-aware restart seeding did not improve on the existing post-hoc reselection at
> cap 4. Both honour a tightened requirement; neither was measurably better at it. The arm
> ships because it fails *earlier* and more legibly — a user who tightens a constraint sees
> the search respect it rather than discovering at verification time that every restart came
> from an excluded region — not because it solves more specs.

No cut, subgroup, tolerance or spec subset will be searched for one on which it looks better.

---

## 6. Explicitly forbidden, whatever the comparison returns

1. `g32_acceptance` numbers will **not** be presented as, or alongside, the preregistered
   result of `docs/PREREG_FINAL_COMPARISON.md`. It is a different arm answering a different
   question.
2. It will **not** become `AUTO_ESCALATE`. `auto` is the mode for a user who has not read
   this file; routing them into a non-preregistered arm is the substitution this project
   refuses elsewhere.
3. It will **not** be added to `UI_MODES` as a default. It stays opt-in.
4. No existing mode's behaviour will be changed to make this one look better. `thinking`
   keeps passing the competition spec to seed selection, deliberately.
5. The verification path will not be touched. If the arm cannot win with the same verifier,
   it does not win.
6. The inertness in §3 will not be quietly dropped from the write-up if the tightened regimes
   look good. It is the arm's headline fact, not a caveat.

---

## 7. What ships with the arm regardless

The reachable-set table, in the mode's own documentation and its user-facing description:

| tier | fields | why |
|---|---|---|
| **filterable, free** (2) | `boost_db_min`/`max`, `area_mm2_max` | corpus `Y`; analytic area |
| **already filtered** (3) | `dc_gain_db_min`, `peak_freq_lo_ghz`/`hi_ghz` | `plausible_mask`, pre-existing |
| **proxy only, never silent** (1) | `power_w_max` | real SPICE; `i_tail` proxy labelled, unused |
| **unreachable** (4) | `noise_vrms_max`, `hd3_db_max`, `eye_h_ui_min`, `eye_v_mv_min` | channel/guard dependent; corpus does not carry them |

And the UI copy states the inertness **as measured in §3a**, at the width it was measured
at: at the default spec, for a target between 3.05 and 11.13 dB, this mode cannot change the
result. A user who selects a mode that provably cannot help them is told before they spend
the click — and a user asking for a target near the edge of the boost range is not told a
falsehood in the other direction.

The copy must not say the mode *does* change the result outside that interval. It sometimes
does not (11.87 dB is outside and identical), and the claim that survives measurement is the
one-sided one: inside, provably nothing; outside, possibly something.

---

## 8. What this does and does not establish

**Does:** whether filtering the restart pool up front beats re-picking from the trace
afterwards, under requirements a user actually tightened.

**Does not:** anything about the competition benchmark. The arm cannot move a preregistered
number, by §3, and no result from it should be read as if it could.
