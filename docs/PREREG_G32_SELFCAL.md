# PRE-REGISTRATION — Are G3.2's slope constants runtime-derivable? (H-SC)

**Committed before the comparison run.** The commit that introduces this file contains no
`results/g32_selfcal_bench*.json` and no results section for it. Check `git log` on this
file against the artifact timestamp rather than trusting the sentence.

Nothing on the default code path changes. `src/silq/experiments/g32_repair.py` and the
transcription inside `src/silq/experiments/final_comparison.py` are not edited, not
imported-and-monkeypatched, and not re-frozen. The delivered circuit remains the one
produced by the frozen-slope controller regardless of how this comes out.

---

## 1. The criticism this addresses, and the one it does not

G3.2 steers with a plane whose slopes were fitted once, offline, on four specs of
spec-seed 2 — `d_boost_db_per_unit = +12.45` dB/unit on `rs`, `d_ln_peak_per_unit =
-1.15`/unit on `l_in`, and three companions. A reader cannot tell those from hand-set
numbers by looking at the controller, and "you tuned the constants until it worked" is a
fair thing to suspect of any refinement loop that ships with a table of magic slopes.

**H-SC.** Those five slopes are *measurements*, not tuning: replacing the frozen table with
slopes measured at runtime from 2–3 single-axis perturbations at the design actually in
hand yields per-spec outcomes equivalent to, or better than, the frozen table — while
paying for the perturbations out of the same refinement budget.

This says nothing about the **axis choice**. Picking `rs` for boost and `l_in` for peak came
from a six-dimensional seed-2 probe and is carried over unchanged; only the five slopes are
measured. Any claim from this run is about the slopes. It is also not a claim that the
self-calibrating variant is better and should ship — see §6.

---

## 2. The instrument — FROZEN HERE, BEFORE THE RUN

`src/silq/experiments/g32_selfcal.calibrate_plane`, already committed in `520f7fbf0` and
already gated against `results/g32_peak_probe.json` by
`src/silq/experiments/g32_selfcal_gate.py`. Its settings for this run, fixed now:

| setting | value | why |
|---|---|---|
| `x0` | `xs[-1]`, the PPO handoff | what the task specifies; see the limitation in §5 |
| `base_rec` | `s1[-1]` when the guard kept it, else `None` | reuses a measurement already paid for |
| `h` | `0.05` (`PROBE_H`) | the step the frozen plane itself was measured with |
| `two_sided` | `False` | one-sided where the base is usable; `-h` only as a retry |
| `budget` | `4` evaluations | hard cap: 2 axes × at most 2 probes |

**Cost is charged, not waived.** The calibration's evaluations come out of the same stage-2
pool the frozen arm gets. Frozen arm: `g32_solve(..., r)` with `r = 10`. Calibrated arm:
`calibrate_plane(...)` spending `c`, then `g32_solve(..., r - c)`. The calibrated arm is
therefore handicapped by exactly what it spends measuring. Any equivalence result is an
equivalence *at equal total cost*.

## 3. The comparison is paired and single-blinded by construction

Both arms are driven from the **same** stage-1 rollout, on the same spec, with the same
`evaluate`, the same rescue ladder, and the same `r`. `stage1_rollout` is seeded on
`1000 + i` and deterministic. The only thing that differs between the two arms is the
`plane` dict handed to `g32_solve`. Exact byte-match of the two arms is **not** expected
and is not the bar: different slopes put the secant on a different path.

**Internal validity gate, evaluated before any comparison number is looked at.** The
frozen arm inside this harness must reproduce `results/g32_repair_smoke.json` exactly on
the five equivalence-gate fields (`best_boost_db`, `best_abs_err`, `loose_solved_at`,
`strict_solved_at`, `n_valid`) for all ten specs. If it does not, the harness is wrong, the
comparison is void, and what gets reported is that failure — not a number.

## 4. The criterion — WRITTEN DOWN BEFORE THE RUN

Slice: **spec-seed 3, specs 8–17**, the burned development slice. No new seed is drawn.
The held-out seed 23 is not touched. Frozen-slope baseline on this slice, from the
already-committed artifact: **6/10 strict**, **7/10 loose**, **7/10 specs with any valid
design**, median `best_abs_err` **0.0676 dB**.

| # | metric | equivalence holds iff |
|---|---|---|
| P1 | strict solves (±1.5 dB), out of 10 | calibrated ≥ 5 (at most one lost) |
| P2 | loose solves, out of 10 | calibrated ≥ 6 (at most one lost) |
| S1 | specs with a valid design, out of 10 | calibrated ≥ 6 (at most one lost) |
| S2 | paired median of Δ`best_abs_err`, over specs valid in **both** arms | median Δ ≤ +0.15 dB |

Why these numbers. n = 10, so one spec is 10% and a single flipped spec is inside the noise
of a deterministic-but-chaotic secant path; a two-spec loss is not. 0.15 dB is 10% of the
±1.5 dB criterion tolerance the whole project is judged against.

**Verdict, decided by the table and not by the prose written afterwards:**

- **EQUIVALENT** iff P1 ∧ P2 ∧ S1 ∧ S2.
- **BETTER** iff EQUIVALENT and additionally (strict > 6) or (median Δ < −0.15 dB).
- **WORSE** otherwise. A WORSE outcome is reported plainly, in the same words as an
  equivalent one, and the frozen slopes stay as delivered.

Reported either way, as description and not as a test: per-spec strict/loose/`best_abs_err`
for both arms, the paired Δ per spec, the calibration cost per spec, which entries were
measured versus fell back to frozen, and the measured slopes themselves.

## 5. What this cannot establish

- **It is a burned development slice.** Specs 8–17 of seed 3 have been run before. This
  comparison is descriptive. It is not evidence about the held-out set, and no per-method
  matched chance line is involved because no solve-rate-against-chance claim is being made.
  It must never be quoted next to the seed-23 numbers as though it were one of them.
- **n = 10, paired, deterministic.** There is no sampling distribution here to put a
  p-value on, and none will be computed.
- **The slopes are measured at the handoff `xs[-1]`, but `g32_solve` may start from an
  earlier stage-1 state** when one of them is already feasible. On those specs the measured
  slopes are local to the handoff rather than to the solver's actual start point. This is a
  real limitation of the comparison, not of the idea, and is reported per spec.
- **It says nothing about the axis choice**, which stays a seed-2 development artifact.

## 6. What ships, whichever way it goes

The delivered circuit and the benchmark record stay the frozen-slope ones. Nothing is
re-frozen, no constant in `g32_repair.py` moves, and `results/final_comparison_seed23.json`
is not re-run. If H-SC holds, the claim it buys is narrow and worth exactly what it says:
*the constants can be derived at runtime, and were frozen for reproducibility, not fitted
for performance.* If H-SC fails, the honest sentence is: *the slopes can be measured at
runtime, but the measured ones perform worse on our development slice than the frozen
table, so the frozen table is what we ship, and the criticism that it is a development
artifact stands.*
