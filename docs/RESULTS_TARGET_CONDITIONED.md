# RESULTS — Target-conditioned policy (H-TC)

Outcome of the experiment registered in `docs/PREREG_TARGET_CONDITIONED.md`, tag
`prereg-target-conditioned`. The protocol was followed as written. Nothing was changed
after data were seen.

## Verdict

**H-TC is not supported.** Pricing the requested target as a dense reward term, without
gating termination on it, did not produce statistically demonstrable retargeting.

The target-conditioned model was also **directionally worse than the untouched control on
every measure we recorded** — it solved fewer specs strictly, passed fewer specs at all,
and landed *further* from the requested boost than the model that never scored the target.

## Primary endpoint — preregistered, one test

Held-out spec seed **20260915**, spent once. Computed by `final_report.py`, the same code
path every existing chance-line number comes from.

```
method       strict   mean k   chance     p(>=observed)
PPO-restart  18/32     3.12     16.7/32    0.3070
```

The registered threshold was p < 0.05; the bar computed in advance was ~21/32. **Observed
18/32, p = 0.31.** A 20,000-draw recomputation agrees (p = 0.322); `final_report` uses
2,000 draws.

## The control did something we did not expect

The frozen `seq_clean40k` — no target term of any kind — was run on the same 32 specs as a
paired comparison.

| | loose | strict | mean k | chance | p | median &#124;err&#124; |
|---|---|---|---|---|---|---|
| TC, W = 3.0 | 24/32 | 18/32 | 3.12 | 16.7/32 | 0.3070 | 0.922 dB |
| control `seq_clean40k` | 27/32 | **24/32** | 3.66 | 18.3/32 | **0.0021** | **0.280 dB** |

Paired McNemar over the same specs: TC solved 1 the control missed, the control solved 7
the TC missed, exact two-sided **p = 0.070**.

Two things follow, and they must not be merged.

1. **Against the control, the reward change hurt.** Not significantly at p < 0.05, but the
   direction is unambiguous and consistent across all three measures.
2. **The control cleared its own matched chance line on this seed.** We are *not* claiming
   that as a retargeting result. The identical model on spec seed 0 scored 16/32 against a
   16.9/32 line, p = 0.76 — indistinguishable from ignoring the target. Same model, same
   protocol, different spec draw, 16/32 → 24/32. This was not a preregistered test of the
   control, it is one arm on one seed, and it contradicts the other seed we have.

## Pre-declared sensitivity: k held fixed at 3.41

Registered in advance because a policy that converges faster produces fewer distinct
feasible designs, lowering its own chance line.

| arm | strict | chance at k = 3.41 | p |
|---|---|---|---|
| TC, W = 3.0 | 18/32 | 22.3/32 | 0.983 |
| control | 24/32 | 22.3/32 | 0.293 |

**Under the harsher fixed-k null, neither arm clears.** The control's own-k result above
depends on the per-spec pairing of `p` and `k` — specs where an arm found no valid design
contribute k = 0 and cost it nothing, whereas fixed k charges every spec 3.41 attempts.
That is why the fixed-k line (22.3) sits above the control's own-k line (18.3) despite its
higher mean k. This is the sensitivity check doing exactly the job it was registered to do.

## Secondaries (preregistered)

| | TC W = 3.0 | control |
|---|---|---|
| loose (feasibility) | 24/32 | 27/32 |
| strict | 18/32 | 24/32 |
| median &#124;achieved − requested&#124; | 0.922 dB (n = 24) | 0.280 dB (n = 27) |
| mean &#124;err&#124; | 1.181 dB | 0.645 dB |
| median evals to first strict solve | 4.0 | 6.5 |

Unfiltered corr(requested, achieved) for the TC arm was +0.908 (n = 31) against a
permutation null of [+0.410, +0.823], p = 0.0004. **This is a secondary and it does not
rescue the primary.** This repo has a standing rule against quoting a correlation as
evidence of retargeting, for reasons recorded earlier; it is reported here for
completeness and is not offered as support for H-TC.

## What went wrong, stated plainly

**The dev selection rule was badly chosen, and it was my error.** §5 selected on median
|err| over loose-passing specs. On dev seed 3:

| W | loose | strict | median &#124;err&#124; | denominator |
|---|---|---|---|---|
| 1.0 | 29/32 | **17/32** | 1.282 dB | 29 specs |
| 3.0 | 24/32 | 15/32 | **1.116 dB** | 24 specs |

The rule selected W = 3.0. But **strict solves are the primary endpoint**, and W = 1.0 was
ahead on exactly that. Selecting on a secondary that trades feasibility for aim sent the
weaker candidate to the confirmatory run. The rule should have selected on strict solves.

A matched-subset check (post-hoc, not preregistered) confirmed W = 3.0's dev advantage was
real rather than a denominator artifact — on the 22 specs both models passed, W = 3.0 led
on median, mean and strict. So the selection was internally consistent; the metric it
optimised was simply the wrong one for the endpoint being tested.

**W = 1.0 was not run on the held-out seed and will not be.** The prereg allows one
confirmatory run; a second would be two shots at one seed. `results/seq_tc_w1.zip` and its
dev artifact exist, and any future use of them is a new experiment needing its own
registration and its own untouched seed.

## Methodological finding, arguably the more useful one

The control moved 16/32 → 24/32 between two spec draws under an identical protocol. **A
32-spec strict count has enough seed-to-seed variance that single-seed comparisons of this
kind are underpowered.** Our registered design was weaker than we assumed when we wrote it.
Any future version of this test needs either many more specs or paired-by-spec analysis as
the primary rather than a marginal count against a chance line.

## What may be said

- SILQ performs verified feasible analog design synthesis. **Unchanged, and not affected by
  this result.**
- A target-conditioned reward did **not** produce demonstrable retargeting under this
  protocol and budget, and performed worse than the unmodified baseline.
- No claim about retargeting — positive or negative — should rest on a single 32-spec seed.

## Artifacts

```
results/seq_tc_w1.zip, seq_tc_w1_train.json, seq_tc_w1_invalid.json
results/seq_tc_w3.zip, seq_tc_w3_train.json, seq_tc_w3_invalid.json
results/target_audit_freshrestart_dev3_w1.json          dev seed 3
results/target_audit_freshrestart_dev3_w3.json          dev seed 3
results/stage_a_selection.json                          selection + inputs
results/target_audit_freshrestart_seed20260915.json     held-out, TC
results/target_audit_freshrestart_ctl_seed20260915.json held-out, control
results/final_report_seed20260915.json                  primary endpoint
results/stage_b_driver.log                              execution trace
```

Kill switch (§8) passed for both training runs: invalid rate 43.8% (W = 1.0) and 43.1%
(W = 3.0) against the control's 45.1% and a 55.1% limit; `n_sims` within 1.8% of the
control. Neither run was stopped, and nothing was tuned around either.
