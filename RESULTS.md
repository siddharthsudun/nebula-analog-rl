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
`src/eqrl/experiments/pass_vs_valid.py`.

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
tabulated in `src/eqrl/circuits/ctle.py`) and by setting the mirror length from a measured
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

## The search space is 10.4% physically valid

`experiments/space_validity.py` samples `ACTION_SPACE` uniformly — no agent, no trajectory
— and runs each sample through the guard layer.

```
250 uniform samples
VALID: 26  (10.4%)

  156   T2.5_mosfet_not_in_saturation
   46   T4.10_dc_gain_implausible
   22   T1.2_solver_failure_text_in_output

DC load drop demanded against a 1.8 V rail:
  median over the whole box : 1.13 V
  fraction needing > VDD    : 41.6%
```

Artifact: `results/space_validity.json`.

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
declared upper bound is 20 mA, so most of that range is dead space. The mechanism is
documented in `ctle.py`: more tail current means more Vgs on the input pair, which pulls
the tail node down, which is exactly the headroom the mirror needs. Current and mirror
headroom trade against each other, and past a few hundred µA the mirror loses.

No range in `ACTION_SPACE` has been narrowed. Narrowing it would change what every
published number means, and a judge is entitled to ask whether the space was shrunk until
the result appeared.

## PVT: the committed design fails 10 of 45 corners

The full grid is 5 process corners (TT/SS/FF/SF/FS) × 3 voltages (VDD ±5%) × 3
temperatures (0/27/125 °C) = 45, with HD3 and input-referred noise simulated at every one.
Re-checking the design in `results/solved_design.json` on the corrected circuit:

```
all_pvt_pass : false
passed       : 35 / 45
failed       : 10 / 45   (3 SS, 4 SF, 3 FS)
```

The ten failures produce no usable measurement at all (`ok: false`) — the design does not
solve at those corners, and a corner that cannot be measured cannot be counted as a pass.
Across the 35 that do solve: boost 6.42–8.05 dB, peak 1.27–1.51 GHz, HD3 −53.1 to
−51.8 dB, input noise 703–1097 µVrms, power 0.25–0.28 mW, eye 0.62 UI / 246–302 mV.

Artifact: `results/legacy_design_recheck.json`.

The eye is computed, not assumed: the SPICE-extracted **complex** CTLE response is put in
series with a minimum-phase PCIe-Gen2 channel (skin + dielectric loss, 12 dB at Nyquist)
and a 1-tap DFE adapted to the first post-cursor, then a random NRZ pattern is run through
and folded. `src/eqrl/sim/eye.py`.

**`results/final_report.json` still asserts `all_pvt_pass: true`, and it is stale.** It
covers a different legacy design (`w_in` 11.83 µm, `rs` 5.00 kΩ, `cs` 222 fF, `r_load`
1.17 kΩ) from the same generation: produced before the mirror was fixed, by a checkpoint
that expects a 14-dimensional observation where the environment now emits 18. That design
has not been rechecked on the corrected circuit. `final_report.json` is kept for the record
and is not a result; it is the most concrete falsifiable object in the repo and should not
survive to submission unqualified.

## Speed: 82.2× per evaluation

Including the SKY130 corner `.lib` costs about 15 s of model parsing per ngspice launch.
Keeping the simulator resident and re-evaluating candidates with `alterparam; reset` parses
the models once:

| | median per AC evaluation |
|---|---:|
| resident libngspice server | **77.5 ms** |
| fresh subprocess per evaluation | 6370.5 ms |
| | **82.2×** |

n = 10 each, `results/speedup.json`, script `src/eqrl/experiments/speedup.py`. The AC sweep
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

## What the RL leg does not yet show

No policy in the repo can be run against the current code. Every checkpoint in `results/`
expects a 14-dimensional observation; the environment emits 18. They were also trained on
a different circuit (ideal-ish tail, `fast=True`, the old area model), so their numbers are
not comparable to anything produced from here. The last completed guarded training run did
not produce a policy that yields a valid design.

Two things are known about why. The search space is 10.4% valid, measured. And the penalty
for a rejected design is a flat −5.0 by default, which is a fact about the code: a design
that misses saturation by 10 mV scores exactly the same as one asking for 50 V across the
load. With most candidates rejected, the reward is nearly constant everywhere, so the value
function is flat and there is no gradient to descend. A shaped penalty that ranks "nearly
valid" above "impossible" exists as an opt-in (`SequentialEqualizerEnv(invalid_shaping=True)`)
and was not in use for the completed run.

An earlier, naive formulation of the RL problem *lost* to Bayesian search. It is kept in
the git history rather than deleted.

## Reproduce

Each script writes its own artifact into `results/` and needs no arguments:

```bash
python -m eqrl.experiments.speedup          # -> results/speedup.json
python -m eqrl.experiments.pass_vs_valid    # -> results/pass_vs_valid.json
python -m eqrl.experiments.space_validity   # -> results/space_validity.json
python -m eqrl.experiments.itail_profile    # -> results/itail_profile.json

# full 45-corner PVT sign-off of a committed design
python -m eqrl.experiments.characterize --design results/solved_design.json
```

See `SETUP.md` for environment setup and `docs/ROADMAP.md` for status.
