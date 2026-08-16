# Handover — night of 16–17 Aug

Branch: `guards/validation-layer`, pushed. Four commits on top of the merge (`4fe8d37`):

| commit | what |
|---|---|
| `ba0346a` | Guard layer now sees the current mirror; three silent failures fixed |
| `d67f564` | Guarded training path made runnable; measured what it trains on |
| `c783d1f` | Area model counts the whole circuit (isolated, revertible) |

Test suite: **190 passed, 3 xfailed.** Green at every commit.

No guard threshold was changed. Where a check fired, it is reported below.

---

## 1. The headline: 0.4% of the search space is a real circuit

This is the most important thing found tonight, and it needs a decision from you.

`experiments/space_validity.py` samples `ACTION_SPACE` uniformly — no agent, no
trajectory — and runs each sample through the guard layer:

```
250 uniform samples of ACTION_SPACE
VALID: 1  (0.4%)

  146  (58.4%)  T2.5_mosfet_not_in_saturation
   81  (32.4%)  T2.6_tail_current_wrong_or_zero
   22  ( 8.8%)  T1.2_solver_failure_text_in_output

DC load drop demanded against a 1.8 V rail:
  median over the whole box : 1.13 V
  fraction needing > VDD    : 41.6%
```

The cause is arithmetic, not subtlety. `i_tail` runs to 20 mA and `r_load` to 5 kΩ, so
the corner of the box asks for **50 V across the load on a 1.8 V supply**. Only 10% of
the box leaves even 0.5 V of output headroom on those two axes alone.

Training confirms it from the other direction: with guards on, **97.4% of candidates are
rejected**. The agent does pull toward the buildable region (97.4% vs 99.6% for uniform
sampling), but it is starting from a nearly empty prior.

**What this means.** It is not that the guard is too strict — the rejections spread across
three independent checks, and the load-drop arithmetic is not arguable. The declared
search space is mostly designs that cannot be built at 1.8 V.

**Your decision.** Narrowing the ranges would change what every published RL number means,
so I did not touch them. The options:

- **Narrow `i_tail` and `r_load`** so the box is mostly buildable. Best learning signal,
  but every existing sample-efficiency number would need re-running, and a judge may ask
  whether the space was narrowed until the result appeared.
- **Constrain the product** rather than the ranges — reject `(i_tail/2)·r_load > VDD` at
  decode time. Keeps the declared ranges honest, removes the impossible corner.
- **Leave it and report it.** "0.4% of the declared space is physically valid, and the
  agent still finds designs" is a legitimate and unusually honest result. It also makes
  the RL-vs-search comparison more favourable, since search suffers the same space.

My recommendation is the second, and to state the 0.4% number in the writeup either way.

---

## 2. The guard layer was blind to half the circuit

The `.op` probe still described the circuit as it was **before** the current mirror
landed. It watched two transistors and read tail current from `Itp`/`Itn` — sources that
no longer exist in the netlist. So Tier 2 was checking saturation on the two devices that
were fine and ignoring the three that were not.

Fixed: the probe now watches all five MOSFETs and the `nbias` node, and reads delivered
tail current from the mirror devices' drain current. (`print i(iref)` is rejected by
ngspice for a current source — verified against the simulator, not assumed.)

There is now a test asserting that every device and node the probe watches actually
exists in both netlists. Nothing would have caught this drift before.

### Running it against real ngspice for the first time found three more

**The mirror could not meet the 10% tail-current bound at any bias.** Mirror accuracy is
set by channel-length modulation — the reference device sits near 0.9 V while the output
device sits at ~0.3 V, and drain current drifts with that difference. Measured worst-case
delivery error across `i_tail` 0.05–20 mA and both supplies:

| mirror length | 0.5 µm | 1 µm | 2 µm | 4 µm |
|---|---|---|---|---|
| worst error | 34.0% | 8.3% | **3.7%** | 1.2% |

Tier 2.6 requires 10%. I did not touch the threshold — I changed the circuit to meet it
(2 µm, chosen over 4 µm at a quarter of the area).

**Mirror widths above 100 µm were silently unbuildable.** `sky130_fd_pr__nfet_01v8`
solves at W = 100 µm and aborts at W = 101 with `could not find a valid modelname`. The
old code clamped at 200 µm, so every design above ~10 mA was asking for a device the PDK
cannot model. Worse, on the resident-server path that abort carries **no exit code**, so
it read as a clean run that happened to produce no numbers. Width is now drawn as
parallel fingers, which is what real layout does anyway.

**The eye ceiling was a factor of two too tight, and toothless where it mattered.**
Tier 4.15 compared a differential eye height against a single-ended `I_tail × R_load`,
so it could reject a legitimately large eye. It also had no supply cap: for a 20 mA /
5 kΩ request the "theoretical maximum" evaluated to 100 V, which no measurement could
ever exceed. Now `2 × min(I_tail × R_load, VDD)`.

---

## 3. Three failures that would have ended the run silently

**`NgSpiceCommandError` derives from `NameError`.** The guard caught
`OSError/ValueError/RuntimeError`, so this escaped and terminated the process. Measured at
3–6 of 33 designs per corner, that ends a training run within minutes — and ends it with a
traceback instead of an attributable verdict. It is now a logged Invalid.

**PySpice treats every non-`Warning:` stderr line as a command failure.** ngspice
announces its DC fallback with `Note: Transient op started`. Any design whose operating
point needs that fallback was reported as failed **despite solving correctly** — the ff
corner took this path while tt and ss did not, which is a systematic bias against exactly
the corner that converges hardest. Benign notes no longer raise the flag; a real error
line in the same command still does, and the guard's own stderr scan is untouched.

**The guarded training path had never executed.** `evaluator.raw_eval` always passes
`_via_guards=True` so the seal can enforce that no measurement reaches a reward
unvalidated — but the seal is only installed in tests, so in every real run `measure_all`
was the unsealed function and raised `TypeError` before reaching the simulator. In other
words `guarded=True` could not have worked at any point before tonight.

---

## 4. Two checks fired that are yours to rule on

Per your standing rule, I did not adjust either.

**T2.8 — parameters sitting exactly on a PDK bound.** Both edges of the action space
coincide exactly with PDK limits: `l_in` min = 0.15 µm = PDK minimum, `w_in` max = 100 µm
= PDK maximum. SB3 clips Gaussian actions to the box, so clipped actions land exactly on a
bound and are marked Invalid. In practice this is minor (11 of 526 during training), but
it is structural, not incidental. Cheapest fix is to move the action-space edges strictly
inside the PDK range; that touches no guard.

**Area cannot fail, and fixing the model did not change that.** The old model counted the
two input devices plus a flat 2000 µm² constant — at default sizing the input pair is
6 µm², so the metric was 99.7% constant, and it was blind to the mirror (the largest
structure in the circuit). It now counts every device, and spans a factor of eight across
the space. But the worst design anywhere is 0.0113 mm² against a **0.05 mm² budget**, so
`area` still cannot bind. That is now a question about the budget, not the model.

---

## 5. Bias point: measured, and it went against my first answer

The input common-mode trades mirror headroom against input-pair headroom one-for-one.
Tier 2 coverage over a PVT grid favoured a high common-mode (0.84·VDD, 111/132 vs
106/132), and I initially set it there. That was wrong: the grid stops at `r_load` = 2 kΩ
and so never reaches the region where the input pair binds. The monotonicity suite does,
and splits the other way:

| VCM | 0.72 | 0.76 | 0.80 | 0.84 |
|---|---|---|---|---|
| physics-contract failures | **0/14** | 2/14 | 2/14 | 3/14 |

0.72·VDD is the only value where the whole physics contract holds. Both tables are in
`ctle.py` so the trade is visible rather than asserted.

Related: `bias_valid_rload`'s hardcoded 0.4 V headroom constant silently encoded the
**broken** bias — while the mirror sat in triode it behaved like a resistor and the output
never really pinned. I could have retuned that constant and three failures would have
disappeared; instead it is now measured from the operating point (0.504 V, `r_load` ≤
1296 Ω at 2 mA).

---

## 6. Training status

Running now: 40k steps, guarded, `fast=False`, seed 0, Tier-5 halting off, checkpoints
every 2048 into `results/checkpoints/`.

Measured throughput is **~1.3 s/step**, so the full 40k is ~14 h — it will not be finished
when you read this. Checkpoints are usable on their own (constant learning rate, so a
truncated run is just a shorter run). Expect roughly 14k steps by 08:00.

`fast` was hardcoded to `True` before tonight, which stubs HD3 and noise — two of the
eight metrics were not measurements. It now defaults to `False`.

Two sidecars are written next to the model: `_train.json` (actual sim count) and
`_invalid.json` (per-check rejection breakdown, updated every 500 steps). The second
exists because a bare invalid *rate* cannot tell a bad search space from a bad guard —
which is exactly the question that mattered tonight.

`honest_benchmark.py` had `--train-cost` defaulting to a hardcoded **14000**. That number
decides where RL breaks even against the search baselines, so it now reads the real figure
from the training record and errors out rather than falling back to a guess. Its `--model`
also defaulted to a filename training never writes.

---

## 7. What is not done

- **Phase C** (`honest_benchmark`, cumulative-cost curve) — blocked on training.
- **Phase D** (`final_report.json`, eye plot, solved design) — blocked on training.
- **Phase E** (`RESULTS.md` from measured numbers) — blocked on training.
- `web/` and `site/` untouched, as agreed.

The four decisions waiting on you: the **action space** (§1), **T2.8's action-space
edges** (§4), the **area budget** (§4), and whether to accept **VCM = 0.72** (§5).
