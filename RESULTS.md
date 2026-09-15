# Results

Everything below is measured through **real ngspice on the open SKY130 PDK** with
`sky130_fd_pr__nfet_01v8` transistors. Every number names the artifact in `results/` — or
the source file — it came from, so it can be checked or re-run.

One caveat applies to everything this repo published before 17 Aug. The environment
defaulted to `fast=True`, which skips the transient-HD3 and `.noise` analyses and
substitutes constants (−40 dB and 1.0 mV). So every training run, and the sample-efficiency
measurement built on it, scored two of the eight metrics as constants rather than
measurements — `experiments/generalization.py` says so in its own docstring. The PVT
characterizations were the exception; `characterize.py` has always run `fast=False`.
`fast=False` is now the default everywhere, and the measurements below were taken with it.

## The delivered system

Frozen 26 Aug 2026. The full record is `docs/REPRODUCE.md` §20–21; this is the summary.

```
spec  →  PPO global feasibility search  →  G3.2 constrained target refinement
      →  independent guarded SPICE verification  →  sized netlist + measured specs
```

- **PPO is a strong feasibility solver.** Frozen policy `results/seq_clean40k.zip` reaches a
  valid circuit in a median of **4 evaluations** vs **2,394** for the poster's parameter
  sweep. It does **not**, on its own, hit a *requested* boost above a matched-chance null;
  that is the refinement stage's job, and it is reported as the boundary of the RL claim.
- **G3.2 closes the spec.** On a 40-spec held-out set (spec-seed 23, chosen and frozen
  before it was simulated), PPO→G3.2 cut median target error from **1.886 dB** (the
  PPO→CMA-ES baseline) to **0.231 dB** at roughly half the refinement budget. The coverage
  explanation that retired the earlier retargeting result runs the wrong way here: the
  baseline made more evaluations and found more distinct designs and still finished farther
  from target (`docs/REPRODUCE.md` §20.2). **Strict solve counts do not clear their matched
  chance line for any arm and are not claimed.**
- **Delivered circuit: 45/45 PVT corners** (`results/delivered_circuit.json`). Target
  8.920 dB over a 14.83 dB channel; worst-corner error 1.081 dB; DC gain the binding
  constraint at 1.04 dB of margin. Chosen by a rule written before any corner was simulated;
  **1 of 22** held-out candidates was PVT-clean. Optimisation ran at TT only, so this is an
  out-of-distribution measurement of one generated candidate.

The rest of this file is standalone measurements, each naming its artifact.

## Headline: 86% of the designs that pass all eight specs are not circuits

CMA-ES was run over four (target boost, channel loss) specifications, 60 evaluations each,
and every design that passed all eight hard specs was then put through the guard layer.

| | count |
|---|---:|
| passed all 8 specs | 28 |
| of those, guard-valid | **4** |
| of those, guard-invalid | **24 (86%)** |
| — rejected `T4.10_dc_gain_implausible` | 14 |
| — rejected `T2.5_mosfet_not_in_saturation` | 10 |

Measured twice with identical results. Artifact: `results/pass_vs_valid.json`, script
`src/silq/experiments/pass_vs_valid.py`.

**Why the specs can be passed by a stage that amplifies nothing.** Boost is defined as peak
gain minus DC gain. It is a ratio, so a stage that *attenuates* at DC manufactures boost
for free without the peak ever rising. Nothing in the eight checks notices: `boost_range`
and `peak_in_band` are both computed from that difference, and the other six — HD3, noise,
power, area, eye height, eye width — are all *easier* to satisfy for a stage that passes
less signal. One design out of the 28, verified:

```
dc_gain   = −5.00 dB          <- attenuates
peak_gain = +1.57 dB          <- barely amplifies, anywhere
boost     =  6.56 dB          <- passes 3-12 dB, peak in band, and the other six
```

This is a hole in the published spec sheet, not only in our reward. `Spec.dc_gain_db_min`
exists to close it and is opt-in and off by default, because every earlier number in this
repo was produced without it and has to stay reproducible.

## Three failures the guard layer caught

Each is fixed, and each has a regression test that would fail if the fix were reverted.

**The tail current mirror was in triode.** At the previous 0.5·VDD input common mode the
tail node sat near 0.105 V while the mirror device needs about 0.40 V of Vdsat, so it
behaved as a resistor rather than a current source and delivered 16–46% of the requested
current: 116 µA of a requested 250 µA, 326 µA of 1000 µA, 649 µA of 4000 µA. `i_tail` was
therefore a compressed non-linear knob, and the mirror supplied none of the PVT behaviour
it was added for. Fixed by moving the common mode (measured trade, both directions
tabulated in `src/silq/circuits/ctle.py`) and by setting the mirror length from a measured
accuracy sweep: worst-case delivery error 34.0% at 0.5 µm, 8.3% at 1 µm, 3.7% at 2 µm.
The check requires 10%; the circuit was changed to meet it rather than the threshold
relaxed.

**The guard's own operating-point probe was reading a different design.**
`probe_operating_point()` runs `.op` against whatever parameters the resident server
currently holds, and the evaluator never primed the server with the candidate. So Tier 2
compared one design's operating point against another design's requested values. Found by
chasing a tenfold disagreement between two experiments over the same space:

```
same design, twice in a row, one server     verdict changed on 9 of 14
same design, unrelated design in between    verdict changed on 5 of 14
same design, fresh server each time         verdict changed on 0 of 6
```

Nothing had asserted that the same input twice gives the same answer, which is the property
that makes every other check mean anything. `tests/test_evaluator_determinism.py` now
asserts it five ways.

**Mirror widths above 100 µm were silently unbuildable.** `sky130_fd_pr__nfet_01v8` solves
at W = 100 µm and aborts at W = 101 with `could not find a valid modelname`. The old code
clamped at 200 µm, so every design above roughly 10 mA was requesting a device the PDK
cannot model — and on the resident-server path that abort carries no exit code, so it read
as a clean run that produced no numbers. Width is now drawn as parallel fingers.

## The search space is 20.4% physically valid

`experiments/space_validity.py` samples `ACTION_SPACE` uniformly — no agent, no trajectory
— and runs each sample through the guard layer.

```
250 uniform samples
VALID: 51  (20.4%)

  107   T2.5_mosfet_not_in_saturation
   54   T4.10_dc_gain_implausible
   38   T1.2_solver_failure_text_in_output
```

Artifact: `results/space_validity.json`. This is measured on the capped range (tail current
≤ 1 mA); on the original 0.05–20 mA range the density was far lower, because nothing is
valid above ~1 mA.

Validity against tail current, 30 samples per point (`results/itail_profile.json`):

| `i_tail` | valid / 30 |
|---:|---:|
| 0.05 mA | 5 |
| 0.10 mA | 7 |
| 0.20 mA | 6 |
| 0.35 mA | 9 |
| 0.50 mA | 7 |
| 0.75 mA | 2 |
| 1.0 mA | 4 |
| 1.5 mA | 0 |
| 2.0 mA | 0 |
| 4.0 mA | 0 |

There is a workable band up to about 1 mA and **nothing valid in 90 samples above it**. The
mechanism is documented in `ctle.py`: more tail current means more Vgs on the input pair,
which pulls the tail node down, which is exactly the headroom the mirror needs. Current and
mirror headroom trade against each other, and past a few hundred µA the mirror loses.

The tail-current range **was capped at 1 mA** on this measured physics (originally
0.05–20 mA). It is the one range that was narrowed, and it was narrowed because the region
above it is empty, not to flatter a result — the cap is recorded in the commit history
(`68d4760`). No other range in `ACTION_SPACE` was touched.

## PVT: the delivered circuit passes all 45 corners

The full grid is 5 process corners (TT/SS/FF/SF/FS) × 3 voltages (VDD ±5%) × 3
temperatures (0/27/125 °C) = 45, with HD3 and input-referred noise simulated at every one,
through the full guard layer and scored against the requested target. The delivered circuit
(`results/delivered_circuit.json`, `docs/REPRODUCE.md` §21):

```
all-guard-valid       : true    (`results/delivered_circuit.json → pvt.all_guard_valid`)
passed                : 45 / 45 (`results/delivered_circuit.json → pvt.corners_passed`,
                                  `results/delivered_circuit.json → pvt.corners_total`)
worst-corner tgt err  : 1.081 dB (`results/delivered_circuit.json → pvt.worst_corner_target_err_db`;
                                  fs / 1.71 V / 125 °C from `results/delivered_circuit.json → pvt.worst_corner`)
binding constraint    : DC gain, 1.04 dB over its 0 dB floor
                                  (`results/delivered_circuit.json → pvt.worst_case_by_metric.dc_gain_db.min`,
                                   `results/delivered_circuit.json → pvt.worst_case_by_metric.dc_gain_db.limit_lo`)
```

It was **not** hand-picked: all 22 held-out candidates that met the strict criterion at TT
were swept over all 45 corners under a rule written before any corner ran, and **1 of 22
came back clean** — this one. The two dominant failure modes across the other 21 were DC
gain and saturation headroom at low supply / high temperature. Optimisation ran at TT only,
so this is an out-of-distribution measurement of one generated candidate, not a claim that
the framework yields PVT-robust designs in general.

*(An earlier recheck reported that a legacy design failed 10 of 45 corners. That
measurement was withdrawn in `1b00a9659`: the failures were `-nan(ind)` returns from a
diverging differential noise analysis, not corners out of limit, and its artifact was
deliberately deleted. See the withdrawn-refutation notice in HANDOVER.md. It says nothing
about the delivered design either way.)*

The eye is computed, not assumed: the SPICE-extracted **complex** CTLE response is put in
series with a minimum-phase PCIe-Gen2 channel (skin + dielectric loss, 12 dB at Nyquist)
and a 1-tap DFE adapted to the first post-cursor, then a random NRZ pattern is run through
and folded. `src/silq/sim/eye.py`.

### The same 45 corners, measured twice by two different eye scorers

An eye number is only as good as the scorer that produced it, and this one is a
behavioural DSP model rather than a SPICE transient — so it was re-measured by a second,
independently written instrument. `compute_eye_v2` labels each column by the *transmitted*
bit rather than by the receiver's own decision sign, convolves with a finite causal FIR
instead of a circular FFT, and adapts its DFE tap by decision-directed LMS rather than
clipping to the first post-cursor. It is a different algorithm, not a re-tuning, and it
enters the frozen guard before any of its numbers can be exposed
(`src/silq/experiments/delivered_eye_audit.py`). Both versions were run over the whole
45-corner grid, with HD3 and input-referred noise simulated at every corner:

```
corners recorded      : 45 / 45 (`results/delivered_eye_audit_v2_20260909/comparison.json → corners_recorded`,
                                  `results/delivered_eye_audit_v2_20260909/comparison.json → corners_required`)
frozen scorer, 10/10  : 45      (`results/delivered_eye_audit_v2_20260909/comparison.json → corners_pass10.frozen_working_tree`)
audited v2,   10/10   : 45      (`results/delivered_eye_audit_v2_20260909/comparison.json → corners_pass10.audited_v2`)
run complete          : true    (`results/delivered_eye_audit_v2_20260909/comparison.json → complete`)
```

The two instruments agree: a median difference of 3.3 mV of eye height and a worst case of
40.8 mV, on openings of 599–721 mV against a 100 mV floor. The direction matters more than
the size — v2 reads **lower** than the frozen scorer at 44 of the 45 corners, exceeding it
by at most 0.54 mV, so the shipped number is not the optimistic one. v2 also records zero
decision errors on all 2,016 scored bits at every corner.

This is a cross-check of the measurement, not of the silicon. Both scorers share the same
SPICE-extracted complex response and the same behavioural channel, so a defect in either
of those would move both. It rules out one specific failure — that the delivered eye is an
artifact of how the frozen scorer folds and labels its samples — and nothing beyond that.

*(Note: `results/final_report.json` is a stale legacy artifact from the honest-benchmark
line, but it has no `all_pvt_pass` field. The committed artifact that carries that field is
`results/solved_design_pvt.json → all_pvt_pass`, for a different design. The schema is
constructed by `src/silq/experiments/characterize.py::characterize`; its CLI writes that
schema as `<outdir>/final_report.json`, while `solved_design_pvt.json` records no generator
provenance. The delivered result is `results/delivered_circuit.json`, whose PVT fields are
cited above.)*

## Speed: 82.2× per evaluation

Including the SKY130 corner `.lib` costs about 15 s of model parsing per ngspice launch.
Keeping the simulator resident and re-evaluating candidates with `alterparam; reset` parses
the models once:

| | median per AC evaluation |
|---|---:|
| resident libngspice server | **77.5 ms** |
| fresh subprocess per evaluation | 6370.5 ms |
| | **82.2×** |

n = 10 each, `results/speedup.json`, script `src/silq/experiments/speedup.py`. The AC sweep
here runs to 100 GHz at 40 points/decade, which is wider than the sweep earlier numbers
were taken on. An older figure of 350× compared one evaluation against a whole process
launch plus model reparse, which is not the quantity that matters during a search.

## Area is not a live constraint

The area model counts every device the netlist instantiates — input pair, all three mirror
devices, the MIM cap, the poly resistors, routing overhead — and spans a factor of eight
across the action space. It still cannot fail: the worst design anywhere is 0.0113 mm²
against a 0.05 mm² budget. Area should be reported as measured and non-binding, not as a
constraint that was met. Whether 0.05 mm² is the right budget is a spec question, not a
modelling one.

## What the RL leg does and does not show

**Does:** the frozen policy `results/seq_clean40k.zip` runs against the current code and is
a strong feasibility solver — 26/32 loose solves on the held-out set, median 4 evaluations
to the first feasible design, reproduced bit-for-bit from committed artifacts
(`docs/REPRODUCE.md` §3).

**Does not:** the policy does not, by itself, hit a *requested* boost above its matched
chance line — on both spec seeds its strict solves sit on that line, and the unfiltered
correlation between requested and achieved boost is +0.114 (`docs/REPRODUCE.md` §8). Precise
targeting is the job of the downstream G3.2 refinement stage, and the two are kept separate
on purpose. Whether the policy itself can be made to retarget (by pricing the target into
the reward without sparsifying the terminal reward) is an open, untested question, not a
settled negative.

An earlier, naive formulation of the RL problem *lost* to Bayesian search. It is kept in
the git history rather than deleted.

## Reproduce

Each script writes its own artifact into `results/` and needs no arguments:

```bash
python -m silq.experiments.speedup          # -> results/speedup.json
python -m silq.experiments.pass_vs_valid    # -> results/pass_vs_valid.json
python -m silq.experiments.space_validity   # -> results/space_validity.json
python -m silq.experiments.itail_profile    # -> results/itail_profile.json

# the delivered circuit's 45-corner PVT sign-off (read the frozen record, no simulation)
python -m silq.experiments.pvt_signoff --report-only
```

See `SETUP.md` for environment setup and `docs/ROADMAP.md` for status.
