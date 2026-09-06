# The four inference modes: what they must do, and what it takes to get there

**Status:** implementation spec, handed off. Nothing here is implemented.
**Date:** 06 Sep 2026.
**Note on evidence:** every number below is measured in this session, not estimated. The
script behind each claim is named so the implementer can re-run it.

This document answers one question per mode: *is the requested behaviour reachable on the
current architecture, and if so what has to change?* The short answer: **three of the four
are reachable, one of them only after relaxing a specific guard, and all four are blocked in
the same subset of cases by a structural gap that none of the four modes addresses.**

---

## 0. The one measurement everything rests on

G3.2's stage B is a **one-dimensional line search along a single axis** — `plane["boost_axis"]`,
which is `rs`. Every other design variable is frozen at the handoff point. That loop can end
in exactly four ways, and they are **not** interchangeable:

| `info["reason"]` | what it means | more budget? | tighter tol? | more restarts? | relax guard? |
|---|---|---|---|---|---|
| `target reached inside the feasible set` | success | — | — | — | — |
| `budget exhausted` | ran out of evaluations | **yes** | no | no | no |
| `converged onto the feasibility wall` | a **guard** stopped it | no | no | maybe | **yes** |
| `no admissible step remains` | **the axis is pinned at a bound** | no | no | no | no |

Measured on 5 specs (`scratchpad/mode_diag.py`), `default` vs `accurate`:

```
  spec        mode       wall   err     s2evals  unspent  reason
  9.0/12.0    default    14.3s  0.493   10       0        budget exhausted
  9.0/12.0    accurate   11.2s  0.471   12       18       converged onto the feasibility wall
  7.5/10.0    default     6.6s  0.020    2        8       target reached
 10.5/14.0    default    10.0s  0.004    6        4       target reached
  6.2/13.5    default     8.4s  0.088    1        9       target reached  (stopped at tol 0.25)
  6.2/13.5    accurate    9.5s  0.002    2       28       target reached  (tol 0.05, +1 eval)
 11.0/12.5    default     9.0s  1.936    0       10       no admissible step remains
```

Three things follow, and they drive the whole document:

1. **`accurate` is not a no-op.** An earlier note in `RESULTS_INFERENCE_MODES.md` called it
   one, on an n=3 bench that happened to contain no case where the stop tolerance was the
   binding constraint. Spec 6.2/13.5 is that case: **0.088 → 0.002 dB for one extra
   evaluation**. That earlier claim is withdrawn.
2. **Accuracy is usually not budget-bound.** On 9.0/12.0, `accurate` stopped with **18 of 30
   evaluations unspent** because a guard walled it. More budget does nothing there.
3. **Spec 11.0/12.5 spends zero stage-2 evaluations** and returns 1.936 dB error with a full
   budget. `rs` is already at 1.0. No mode as specified fixes this — see §6.

### Which guard does the walling

`scratchpad/wall_probe.py`, 12 rejected precision steps across 4 specs:

```
T4.10_dc_gain_implausible        10
T2.5_mosfet_not_in_saturation     2
```

`T4.10` here is the **lower** bound: DC gain below `guards.DC_GAIN_DB_MIN = 0.0 dB`. That is
the CTLE trade itself — raising `rs` raises boost by degenerating the input pair, and
degeneration costs DC gain. **The accuracy ceiling is a physical trade-off, not a search
failure.** That is why mode 4's premise is correct.

### The cost floor

Same run: **~0.55 s per `measure_all`** (range 0.47–0.78), 4–5 SPICE analyses each.

| component | `measure_all` | wall |
|---|---|---|
| policy load (cold process only) | 0 | 2.2 s |
| **stage 1, k=5** | **11** | **~6.1 s** |
| stage 2, per evaluation | 1 | ~0.55 s |
| independent verification | 1–2 | ~0.8 s |
| observed totals | 12–24 | 6.6–14.3 s |

**Stage 1 alone costs ~6.1 s.** That number is what makes Fast mode impossible as structured.

### The surrogate, measured

`results/surrogate_corpus.npz` holds **374,588 designs**, builds in 0.37 s, predicts in
**7.3 µs per design**. A 100,000-candidate sweep costs under a second — about 1.3 SPICE
evaluations. Accuracy, from `REPRODUCE.md` §19.6: **0.456 dB MAE overall, 0.199 dB in the
densest distance decile, 0.917 dB in the sparsest.**

---

## 1. Default — keep it exactly as it is

**Requested:** ~20–30 s, SPICE, ±0.5 dB, prioritise feasibility.
**Measured:** 6.6–14.3 s, within ±0.5 dB on 3 of 5 specs.

**Required change: none, and none is permitted.** `default` is byte-identical to frozen arm B
and is what every recorded number in `RESULTS.md` describes. Moving `r` off 10 invalidates the
frozen record.

Two notes:

- **It is already 2–3× faster than the spec allows.** If the UI shows 20–30 s, that time is
  server/UI overhead, not the pipeline — measure `pipeline.design()` directly before
  optimising anything.
- Do not spend the headroom by editing `default`. A slower, more thorough everyday mode is
  **Thinking's** job (§3), or a new mode name — never a change to this one.

---

## 2. Fast — reachable, but **not** by trimming the current path

**Requested:** < 5 s, surrogate-first, ngspice only when required, leniency allowed, must warn
that values may not match simulation, minimal accuracy loss.

**The blocker:** a < 5 s budget is ~9 `measure_all`. **PPO stage 1 alone is 11.** The current
`fastest` mode only shrinks the *stage-2* budget (`FASTEST_BUDGET = 4`), which is why it
measured 11.6–18.6 s and saved nothing on 2 of 3 specs. **You cannot reach 5 s by tuning
stage 2. Stage 1 is the cost.**

### Required change

Replace stage 1 with a surrogate sweep and spend the whole SPICE budget on confirmation:

1. Sweep the corpus (or a 100k sample) on the surrogate for the requested target: predicted
   boost closest to target, subject to predicted feasibility. **~1 s, zero SPICE.**
2. Spend **one real guarded evaluation** on the best candidate. (~0.55 s)
3. If guard-valid, run the normal independent verification and return. (~0.8 s)
4. If guard-invalid, fall back to the next surrogate candidate — cap at 3 real evaluations
   total, then return the best guard-valid design found.

**Predicted wall time: 2.5–3.5 s warm.** Add 2.2 s cold, so the server must pre-load the policy
and corpus at startup or the first request blows the budget.

### What the warning should actually say

Worth getting right, because the honest story is **better** than "these numbers might be wrong":

- The returned boost is a **real SPICE measurement** — steps 2–3 are real evaluations.
- What is uncertain is whether a **better design exists** that the surrogate did not rank highly.

So the banner should not say "values may not be accurate":

> **⚠ Fast mode** — this circuit was *found* by a surrogate model and *measured* by SPICE. The
> reported numbers are real. A better design may exist that this mode did not look at. Typical
> target error ≈ 0.5 dB, up to ~1 dB in sparsely-sampled regions. Use Default or Accurate to
> search properly.

0.456 dB and 0.917 dB are both already measured and defensible — quote them.

---

## 3. Thinking — reachable; the ±0.1 dB goal needs one addition

**Requested:** ~60 s, more circuits, thorough search of the feasible region, ±0.1 dB and
physically feasible; and **tell the user when Default already found the best available**.

**Measured:** 7–11 evaluations, ~12–25 s. It is *under* budget, not over. 60 s is roughly
**100 `measure_all`** and the mode spends about a fifth of that.

### Required changes

1. **Spend the budget.** Raise `THINKING_ROLLOUTS` 3 → ~8 and `THINKING_R` 15 → ~25. Keep the
   adaptive early-exit — it is confirmed firing (`mode_detail.adaptive_stopped_early == True`)
   and is why the mode got *cheaper* than Default on the hard spec.
2. **Screen restarts with the surrogate instead of blind seed offsets.** The current offsets
   `(0, 4001, 9007)` are arbitrary. Rank candidate starting points on the surrogate (7.3 µs
   each) and spend real rollouts on the most promising — same cost, better starts.
3. **Set `THINKING_STOP_ABS_ERR_DB = 0.01`**, not 0.05. The mode should stop because it has
   converged, not because it hit its own tolerance; it already reaches 0.002–0.004 dB where
   the geometry allows.

### The "Default already found the best" message

**You cannot prove global optimality and must not claim it.** But `info["reason"]` gives you
the mechanism for free, and each reason licenses a *different* honest statement:

| reason | message to the user |
|---|---|
| `target reached inside the feasible set` | Hit the target. Nothing to explain. |
| `converged onto the feasibility wall` | "Closest achievable is X dB. Getting closer requires DC gain below 0 dB, which the feasibility guard rejects. **Try Accurate mode**, which allows that trade." |
| `no admissible step remains` | "The boost control is already at its limit for this channel. This target may not be reachable with this topology." (§6 — this message is currently a lie of omission) |
| `budget exhausted` | Should never appear in Thinking after change 1. If it does, it is a bug. |

Note row 2: **the wall case should route the user to Accurate, not tell them nothing better
exists.** Something better does exist; it costs DC gain.

---

## 4. Accurate — measured, and the premise is right

**Requested:** ~Default's time, SPICE, deprioritise feasibility (guard may be weakened), hit
within **0.05 dB**.

### The experiment

`scratchpad/relax_probe.py` re-runs the precision search with the DC-gain floor at 0 dB
(control) and −6 dB (relaxed), r=30, stop=0.05. **Nothing was edited** — both floors were
rebound inside one throwaway process.

```
  spec        floor   err     dcgain  evals  reached  reason
  9.0/12.0     0.0    0.471    0.01    12    False    converged onto the feasibility wall
  9.0/12.0    -6.0    0.041   -0.48     5    True     target reached
 11.0/12.5     0.0    1.936    0.25     0    False    no admissible step remains
 11.0/12.5    -6.0    1.936    0.25     0    False    no admissible step remains
 10.0/11.5     0.0     ----     ----    3    False    2-D repair did not reach feasibility
 10.0/11.5    -6.0    0.002   -4.37     4    True     target reached
 10.5/14.0     0.0    0.004    0.15     6    True     target reached
 10.5/14.0    -6.0    0.044    0.11     5    True     target reached
  7.5/10.0     0.0    0.020    1.71     2    True     target reached
  7.5/10.0    -6.0    0.020    1.71     2    True     target reached
```

**Within 0.05 dB: 2 of 5 → 4 of 5.** Spec 9.0/12.0 reached target in **fewer** evaluations
(12 → 5), because it stopped fighting a wall. Spec 10.0/11.5 went from **no design at all** to
0.002 dB.

The price is explicit and small: those designs sit at **−0.48 dB and −4.37 dB DC gain** — a real
CTLE that attenuates at DC and needs the loss made up downstream. A legitimate design choice,
not a broken circuit.

One caveat: on 10.5/14.0 the relaxed run came out *slightly worse* (0.004 → 0.044). Relaxation
changes the trajectory; it does not monotonically improve. Both are inside 0.05 dB, so the mode
goal still holds, but do not describe this as strictly better.

### Required changes

1. **Two floors must move, not one.** Both sit at 0.0 dB today:
   - `guards.DC_GAIN_DB_MIN` — the hard guard; rejects the design outright
   - `Spec.dc_gain_db_min` — the 9th `hard_pass` check; marks it "not feasible"

   Relaxing only one changes nothing.
2. **Suggested value: −6.0 dB.** It cleared every walled case measured. Name it
   `ACCURATE_DC_GAIN_DB_MIN = -6.0`.
3. Keep `ACCURATE_R = 30` and `ACCURATE_STOP_ABS_ERR_DB = 0.05` as they are.
4. **Disclose it.** The result should carry
   `mode_detail["relaxed_checks"] = {"dc_gain_db_min": -6.0}`, and the UI should state that the
   circuit attenuates at DC, and by how much.

### Two things that must not happen

**Never relax Tier 1 or Tier 2.** `T2.5_mosfet_not_in_saturation` rejects a transistor that is
not operating as an amplifier — every number measured off such a design is meaningless. Tier 1
rejects failed simulator runs. Relaxing either does not produce a less-conservative circuit, it
produces a **fake number**. Only the Tier 4 DC-gain *lower* bound is a defensible trade. Leave
`DC_GAIN_DB_MAX = 60` alone as well — that one catches parser bugs.

**Never write the relaxed floor to a process global.** `guards.DC_GAIN_DB_MIN = -6.0` is a
module-level rebind. The web server handles concurrent sessions in one process, so one user
clicking Accurate would silently relax the guard for **every other user's** Default and Thinking
run, and those results would be wrong with nothing logged anywhere. The probe script does
exactly this and is safe only because it is a single-run throwaway process.

The correct shape is a parameter threaded through the call, defaulting to the frozen value —
the pattern `stop_abs_err_db` already uses:

```python
def check_physical_plausibility(m, dv, art, *, topology_has_gain=True, vdd=1.8,
                                dc_gain_db_min: float = DC_GAIN_DB_MIN):
    ...
    if topology_has_gain and m.dc_gain_db < dc_gain_db_min:
```

`build_evaluator` takes it, `Evaluation.make_eval` takes it, `pipeline.design` passes it only
when `mode == "accurate"`. Every other caller keeps the frozen default and stays byte-identical.

---

## 5. Does any of this need retraining the PPO?

**No.** Not for any of the four modes.

Every mode already shares the same frozen `seq_clean40k` policy and the same frozen `g32_solve`.
Everything above is **test-time compute scaling and constraint selection on a fixed policy** —
how much search to spend, where to start it, and which constraints the search must respect. None
of it touches the reward, the observation space, the action space, or the weights.

That is also the stronger framing for the competition: *the modes are not four models, they are
one policy under four inference budgets.* It is a better claim than four separately trained
models, and cheap to defend, because the policy file is byte-identical in every case.

The one thing retraining *would* buy is §6 — and even there it is not the cheapest fix.

---

## 6. The gap none of the four modes closes — fix this first

Spec 11.0/12.5 returns 1.936 dB error having spent **zero** stage-2 evaluations, because `rs` is
pinned at 1.0 and the line search is one-dimensional. Budget, tolerance, restarts and guard
relaxation all leave it at 1.936 dB — confirmed in §4's table.

`scratchpad/axis_probe.py` takes that exact stuck design and steps every axis:

```
    axis   step   boost     err  dcgain  feasible
    cs    +0.08   10.230   0.770   0.25    True     <-- 60% error reduction, ONE evaluation
    cs    +0.20   11.649   0.649   0.25    False
    w_in  +0.20    9.784   1.216   0.09    True
    i_tail-0.08    9.546   1.454   0.11    True
    rs    -0.08    8.593   2.407   0.70    True     (the only axis the solver will try)
```

**`cs` cuts the error by 60% in a single evaluation, and the solver never tries it.** That is
correct physics: `rs·cs` sets the CTLE zero, so with `rs` railed, `cs` is the remaining
degeneration control.

### ✅ IMPLEMENTED — `src/eqrl/experiments/axis_retarget.py`, mode `"retarget"`

Shipped as its own arm. `g32_solve` is called **twice, completely unmodified**; the new logic
lives only between the two calls, the same way `fastest_hedge` does it. The second call
searches a different axis because it is handed a **derived plane** whose `boost_axis` is the
accepted axis and whose `d_boost_db_per_unit` is the slope measured on the probe step one
evaluation earlier — using the solver's own interface, not reaching inside it.

Budget and stop tolerance are deliberately left at Default's `fc.PREREG` values, so the A/B
below isolates the second axis and nothing else.

**Measured (`scratchpad/retarget_ab.py`, `retarget` vs `default`, 8 specs):**

```
   spec         default -> retarget           fired   axis   extra cost
   11.0/12.5      1.936 ->    0.187   BETTER  True    cs     +8 measure_all
   9.0/12.0       0.493 ->    0.493   same    False   -      +0
   9.5/13.0       2.545 ->    2.545   same    False   -      +0
   7.5/10.0       0.020 ->    0.020   same    False   -      +0
   10.5/14.0      0.004 ->    0.004   same    False   -      +0
   6.2/13.5       0.088 ->    0.088   same    False   -      +0
   10.0/11.5        inf ->      inf   same    False   -      +0
   8.2/11.0         inf ->      inf   same    False   -      +0

   better 1   worse 0   unchanged 7
```

**1.936 → 0.187 dB, and the spec now genuinely solves:** status goes
`closed_but_failed_verification` → `solved`, passing all ten checks on independent
re-measurement. On every spec whose axis is not pinned the retarget does not fire and costs
**zero** extra simulator work.

**One design mistake worth recording, because it nearly buried the result.** The first version
required the probe step itself to be feasible. The ranker's top pick, `cs +0.16`, reaches
11.226 dB (error 0.226) but *overshoots out of the feasible set* — so the axis that actually
moves this design was discarded, and the arm fell through to `w_in`, which moves it a third as
hard, for a final 1.172 dB. **An infeasible probe still identifies the right axis.** Bracketing
between a feasible point and a wall is exactly what stage B's `wall_t` branch already does. The
rule is now: accept the axis on **guard validity** plus a measured improvement, and leave the
step size to the solver. That one change is the difference between 1.172 dB and 0.187 dB.

Guard validity is still required — a guard-invalid probe measured nothing, so it says nothing
about its axis. (`r_load −0.24` was rejected on `T4.10` in exactly this run.)

### ❌ MEASURED AT n=32 ON A FRESH SEED — it fires 0/32. Read this before building on it.

`scratchpad/retarget_ab32.py`, spec-seed 137 (a draw that appears nowhere else in this repo),
32 specs, `retarget` against `default` at identical budget and tolerance:

```
retarget fired on 0/32 specs
better 0   worse 0   unchanged 32
inert on every non-fired spec  (+0 measure_all on all 32)
```

**The n=8 result above does not generalise, and the reason is a selection effect I created.**
Spec 11.0/12.5 was in that set *because* it was the known pinned case. Removing that choice
removes the entire effect: in 32 unseen specs the exit `no admissible step remains` — the only
condition this arm acts on — **never occurred once**.

Bucketing all 32 by why the arm stayed inert is the useful output, because it says where the
error actually lives:

| specs | why `retarget` could not act | owner | error behind it |
|---:|---|---|---|
| 15 | target already reached | — | all ≤ 0.246 dB |
| 6 | never reached the feasible set | repair stage | **6 unsolved** |
| 6 | **budget exhausted before a 2nd axis could be tried** | §3 Thinking (larger `r`) | 5.02, 4.11, 2.00 dB |
| 3 | **feasibility wall** (`rs` not at a bound) | §4 Accurate (DC-gain floor) | 3.92, 2.31, 2.28 dB |
| 2 | no guard-valid design at all | stage 1 | both unsolved |
| **0** | **axis pinned at a bound** | **§6, this arm** | — |

Default on this seed: 18/32 within 0.5 dB, 8 unsolved.

**Three conclusions, in order of what they change.**

1. **§6 was the wrong thing to implement first.** §3 and §4 own 9 of the 9 addressable failures
   here; §6 owns none. Build those.
2. **This arm is budget-starved, not merely useless.** Six specs could not fire because the
   first pass had already spent the budget — that is a *different* block from "the axis was
   fine". If §3 Thinking raises `r`, the pinned exit may start appearing and this arm may
   start firing. Untested, and it is the one thing that would revive §6. Do not assume it.
3. **Keep the arm; it is free.** 32/32 identical outputs and +0 `measure_all` everywhere it
   does not fire. It is safe to leave in the tree and re-measure once §3 lands. It is not a
   result to quote.

**Do not cite the 1.936 → 0.187 dB improvement as a rate.** It is one demonstration on one
hand-picked spec, and it is the only fired case that exists.

---

## 7. Freeze compliance

Everything in §2, §3, §4 and §6 is legal **only** in this shape:

- `default` stays byte-identical. No constant it reads may move.
- Every new knob is a **call parameter with the frozen value as its default**, exactly like
  `stop_abs_err_db`. No module-level rebinds, no globals, no environment variables.
- No reward, observation, action, PPO hyperparameter or design bound is touched.
- `_check_constants()` must still pass unchanged. If it fails, the change is illegal — that
  check exists precisely to catch this.
- §6 ships as a new arm under a new name, reported on a fresh `--spec-seed`. It never overwrites
  a recorded number.

---

## 8. Acceptance tests

Extend `scratchpad/mode_bench.py` to **at least 12 specs** — every conclusion above rests on n=5,
and three of the four interesting cases appeared exactly once each. Then:

| mode | criterion | current |
|---|---|---|
| default | unchanged, byte-identical to frozen arm B | holds |
| fast | p95 wall < 5 s warm; banner present; ≥1 real SPICE eval in every result | 11.6–18.6 s |
| thinking | ≥80% within 0.1 dB **excluding** wall/pinned cases; correct message on the other two | untested |
| accurate | ≥80% within 0.05 dB; `relaxed_checks` in every result; T1/T2 rejection rate **unchanged** | 40% at floor 0.0 |
| §6 arm | `no admissible step remains` rate falls; no regression on specs that already solve | not built |

The last column of the `accurate` row is the one to watch. **If relaxing the DC floor changes the
T2.5 rejection rate at all, something is wired wrong** — you relaxed more than DC gain.
