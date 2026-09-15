# What the multi-seed audits told us

**All 8 of 8 test runs complete.** Finished 06 Sep 2026, 07:33.

Everything here is reproducible from one read-only script:

```
PYTHONPATH=src python scripts/compile_audits.py
```

The raw results are snapshotted in `results/audit_snapshots/` (8 files) and the summary
table in `results/audit_summary.json`.

Nothing in this document changed the frozen architecture, any reward, any guard
threshold, or any recorded benchmark number. It is all measurement and analysis of
artifacts that already existed on disk.

---

## 1. What we did, in plain terms

We trained the same design-agent **four separate times** from four different random
starting points ("seeds" 0, 1, 2, 3). Same code, same reward, same everything — only the
dice rolls differ. Then we tested all four the same way.

The reason to do this is uncomfortable but simple: **if you train something once and it
works, you don't know whether you built something good or got lucky.** Training it four
times and testing all four is the cheapest defence against the most common way an ML
result turns out to be wrong.

Each of the four was tested under **two different testing rules** ("protocols"), giving
eight test runs. The difference between those two rules turned out to be the single
biggest finding here — section 4.

**Which one actually ships:** the product uses `seq_clean40k.zip`, which is **seed 0**.
Seeds 1, 2 and 3 exist only to check that seed 0 wasn't a fluke. This matters a lot for
section 6.

## 2. The one-minute version

- **The core claim holds.** The agent is roughly **four times better than random search**
  at the job it was built for, and that survives being retrained from scratch three more
  times. It was not luck.
- **We were measuring ourselves with a broken ruler.** One of our two testing rules threw
  away **68–79% of its own budget** on repeated work, under-reporting the agent by about
  **2.7×**. Fixing the ruler costs nothing and changes no architecture.
- **A real weakness exists**, in a rare kind of job, where plain random guessing beats
  our agent. It is about 3 jobs in 32, and the shipped product largely covers for it.
- **One of the four training runs came out genuinely worse** — and we predicted that from
  its training curve *before* testing it. It is not the one we ship.
- **Nothing needs rolling back.** Details in section 7.

## 3. What an "audit" measures — and what it does not

This matters, because the numbers below look worse than the frozen headline and someone
will notice.

The audit tests **the trained agent on its own**: 20 attempts at a circuit, how close does
it get. The shipped product is **the agent plus a second stage** (`G3.2`) that repairs the
agent's answer. The frozen headline — 28/40 and 22/40 — is the agent *plus* the repair
stage.

> The audit measures the engine on a test bench. The frozen report measures the car. A
> weaker bench number doesn't mean a weaker car; it means the gearbox is doing real work.

**So never compare an audit number to the frozen headline.** Different configuration,
different job set, different budget. Section 5 is a case where that distinction flips the
conclusion entirely.

Two words used throughout:

- **loose** — the design is electrically legal; it passes every safety and physics check.
- **strict** — legal *and* within 1.5 dB of the requested amplification. This is the real bar.

## 4. Finding 1 — we were measuring with a broken ruler

The largest effect in the whole exercise, and it is our own fault rather than anything
about circuits.

The agent works in episodes. When an episode ends, the harness restarts it. Under the
older rule (`default`), the restart puts it back at **the same starting point** — and
because the agent is deterministic, it then walks **exactly the same path again**. Its
remaining attempts re-do work it has already done.

Counting how many "evaluations" produced a design the run had already seen:

| test run | valid evaluations | genuinely distinct | wasted on repeats |
|---|---|---|---|
| seed 0, `default` | 392 | 102 | **74%** |
| seed 1, `default` | 391 | 98 | **75%** |
| seed 2, `default` | 378 | 81 | **79%** |
| seed 3, `default` | 351 | 114 | **68%** |
| seed 0, `fresh-restarts` | 323 | 321 | **1%** |
| seed 1, `fresh-restarts` | 328 | 322 | **2%** |
| seed 2, `fresh-restarts` | 300 | 295 | **2%** |
| seed 3, `fresh-restarts` | 239 | 233 | **3%** |

Between two-thirds and four-fifths of the budget, gone, in every single run. Restarting
from a *new* random point each time takes it to 1–3%.

What that costs in score — identical agent, identical budget, only the restart rule differs:

| seed | strict, `default` | strict, `fresh-restarts` |
|---|---|---|
| 0 (**shipped**) | 6/32 (19%) | **16/32 (50%)** |
| 1 | 9/32 (28%) | **17/32 (53%)** |
| 2 | 12/32 (38%) | **19/32 (59%)** |
| 3 | 5/32 (16%) | **9/32 (28%)** |

Roughly **2.7× the result on the shipped agent**, purely from not wasting the budget.

**Why this is good news.** It is a bug in how we *test*, not in what we *built*. The agent
was always this capable; we were under-reporting it. And it is fixable without touching
the freeze, because it lives in the test harness.

It also independently confirms the interface work: **the way to improve this system is
more genuinely different starting points, not a bigger budget or a tighter tolerance.**
Three separate measurements now say so — the mode benchmark, the never-feasible validation
(11 of 12 previously impossible jobs became solvable), and this duplicate count.

## 5. Finding 2 — a real blind spot, smaller than it first looked

Every job asks for X dB of amplification over a channel that loses Y dB. Call **margin =
Y − X**. Usually positive. Occasionally negative: you're asked for **more amplification
than the channel lost**. Rare — about 3 jobs in 32.

Pooled over all four seeds, on the raw agent:

| | jobs | our agent | random guessing |
|---|---|---|---|
| margin ≥ 0 (normal) | 116 | **77.6%** legal | 20.7% |
| margin < 0 (rare), `default` | 12 | **16.7%** | **66.7%** |
| margin < 0 (rare), `fresh` | 12 | **25.0%** | **66.7%** |

The agent is ~4× better than random in the normal case and **loses badly to random** in
the rare one. Consistent across all four independently trained copies, so it is not bad
luck in one run. New starting points barely help (17% → 25%), so it is not simple
sticking.

The likely explanation is a learned shortcut — *"required amplification tracks channel
loss"* — true for 29 of 32 jobs and false for 3. Shortcut learning of exactly this kind is
a well-known failure mode.

**The qualifier that changes the conclusion.** That is the engine on the bench. In the
shipped product the repair stage covers for it:

| | jobs | shipped product, strict |
|---|---|---|
| margin ≥ 0 | 35 | 57% |
| margin < 0 | 5 | **40%** |

40% against 57%, on five jobs — **not statistically distinguishable** at that size. So the
honest statement is: **a real and clearly identified property of the agent, and only a
possible weakness of the product.** Maximum value of fixing it is ~3 jobs in 40. It should
not jump ahead of the PVT robustness work.

*Correction against an earlier draft of this document:* with only three test runs the
negative-margin rate looked like a flat **0%**. With all eight it is 17–25%. The effect is
real but not absolute, and the earlier figure should not be quoted.

**Three jobs are unreachable by every method we have** (specs 7, 27, 29) — no agent, no
seed, no protocol, and not random search either. Their margins are +0.03, +2.19 and +2.03,
i.e. **positive**, so this is a separate phenomenon from the blind spot, not the same one.

## 6. Finding 3 — does training luck matter? Yes, once out of four

Each seed was tested against the other three pooled, using Fisher's exact test, on the
identical 32 jobs.

**Seeds 0, 1 and 2 are statistically indistinguishable from each other on both metrics
under both protocols.** Three independent training runs produced equivalent agents. That
is the result that matters, and it is the one that makes the project's core claim a
finding rather than an anecdote.

**Seed 3 is genuinely worse, and it replicates on both protocols:**

| protocol | measure | seed 3 | other three pooled | p |
|---|---|---|---|---|
| `default` | loose | 18/32 (56%) | 74/96 (77%) | **0.039** |
| `fresh` | strict | 9/32 (28%) | 52/96 (54%) | **0.014** |
| `fresh` | loose | 19/32 (59%) | 74/96 (77%) | 0.067 |

The mechanism is visible directly: seed 3 produces a median of **5 legal designs per job
out of 20 attempts, against 11 for every other seed**, and it fails to produce *any* legal
design on 5–6 jobs where the others fail on 0–3.

**Two honest caveats, both of which belong beside these numbers.** First, eight tests were
run, so a Bonferroni-corrected threshold is 0.0063 and neither p-value clears it on its
own. Second — and this is what rescues the finding — **seed 3 was flagged as suspect from
its training curve before this audit ran** (its value-estimator never converged; entropy
and value-loss were both off-pattern versus seeds 1 and 2). For a pre-specified hypothesis
confirmed independently under two protocols, this is real evidence rather than one of
eight fishing expeditions.

**What it does not mean.** Seed 3 is not shipped and never was. The product uses seed 0.
Nothing needs changing because of this result. Its actual value is as a **validation of
our training monitoring**: we predicted a bad outcome from the training curve and the
measurement confirmed it. That is a capability worth keeping.

Seed 2 is borderline *better* than the rest on strict under `default` (p = 0.097). Not
enough to claim, and we are not claiming it.

## 7. Has anything gone backwards? Rollback assessment

**Assessment: nothing needs reverting.**

### 7.1 The mid-flight source edits

`src/eqrl/pipeline.py` and `src/eqrl/experiments/fastest_hedge.py` were edited partway
through this work by another session, implementing mode-report suggestions *in place*
rather than as separately-named experiments.

Assessed and **kept**:

- No recorded benchmark number is affected — the benchmark does not route through the
  edited path.
- No audit result is affected — the audit never calls that code at all.
- `default` mode is byte-identical to before.
- The changes pass a budget *number* to an unmodified solver, exactly as two existing
  modes already do. No frozen logic changed.

**One thing to watch.** In the worst case `fastest` mode now costs **11** circuit
simulations where `default` costs 10. The author documented and bounded this deliberately
("at most one wasted evaluation relative to running `default` directly"), and the fast
path is kept whenever its shortcut works. It is a documented trade, not a silent
regression — but a mode called "fastest" that can be 10% slower than "default" must state
its worst case wherever it is described. **Recommend: measure how often the shortcut
fails, and publish that rate.**

The `thinking` change is a straight improvement: it stops as soon as an attempt succeeds
instead of always running three, and bills only for attempts actually made. More honest
accounting, not less.

### 7.2 Three duplicate-process incidents, and a false alarm I raised

On three occasions another session launched an audit writing to a file this session was
already writing (02:12 seed 3 default, 05:28 seed 2 fresh, 06:11 seed 3 fresh). Each
duplicate was terminated.

I initially reported the seed-3 data as probably corrupt. **That was wrong**, and the
correction matters more than the incident: the audit is deterministic end to end — fixed
job list, fixed environment seeds, deterministic agent, fixed random seed for the
random-search arm. The duplicates were computing **identical results**; the file only ever
held a valid prefix of one and the same sequence. **No data was lost or compromised** in
any of the three. The cost was several CPU-hours and some memory pressure.

As a safeguard, all eight completed results are copied to `results/audit_snapshots/`,
where a stray process cannot overwrite them.

A separate process (`scratch_sweep_4modes_seed99.py`, another session's four-mode sweep on
spec seed 99) was left running and untouched — independent work, not a duplicate.

## 8. What this exercise changed, on balance

**Better:**

1. The core claim — the agent substantially beats random search — now holds across
   **three independently trained copies that are statistically indistinguishable**. That
   is the difference between a result and an anecdote, and it is the single most valuable
   outcome here.
2. We found and priced a measurement bug that was under-reporting our own agent by ~2.7×.
3. We have a confirmed mechanistic reason *why* the interface's "thinking" mode works,
   rather than an observation that it does.
4. Our training monitoring demonstrably predicts a bad run before testing — seed 3 was
   called in advance and confirmed.
5. The whole analysis is reproducible from one script, and that script reproduces the
   frozen headline (28/40, 22/40) exactly, so we know it reads the frozen artifacts
   correctly.

**Worse:**

1. A real, reproducible blind spot in the agent on negative-margin jobs, largely — but not
   provably — covered by the shipped repair stage.
2. `fastest` mode's worst case is no longer strictly faster than `default`, and that needs
   saying wherever the mode is described.
3. One training run in four is a dud. Cheap to detect, and not the one we ship.

**Neither, but worth knowing:** three jobs in the benchmark appear unreachable by any
method we have, for reasons unrelated to the blind spot.

## 9. Recommended next steps, in order

1. **Change nothing on the strength of this document.** The freeze holds.
2. Proceed with the agreed sequence — PVT robustness sweep first, then the separately
   named retargeting arm. Nothing found here justifies reordering it.
3. Report policy-audit numbers under `fresh-restarts`, and say plainly why: the older rule
   wastes 68–79% of its budget and understates the agent. Report both if challenged.
4. Measure the `fastest` shortcut's failure rate and publish it beside the mode.
5. *Optional, only if time allows:* test the negative-margin hedge. Expected payoff ~3 jobs
   in 40. A curiosity, not a headline.

## 10. What these numbers are not

- They are **not** the competition headline. That remains the frozen 40-job benchmark.
- They **cannot** be compared to it directly — different configuration, job set and budget.
- Seed-to-seed differences rest on 32 jobs each: enough to catch a large effect, not enough
  to resolve a small one. Where a difference is not separable, this document says so
  rather than picking the flattering reading.
- The random-search arm scored **4/32 strict in all eight runs** — identical every time,
  because it is deterministically seeded. That is a consistency check on the harness, not
  eight independent confirmations.
