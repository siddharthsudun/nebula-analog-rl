# Wide-swing tail implementation and bounded pilot

Implemented and measured on 10 September 2026. The experimental tail is available,
but is **not adopted**: no tested wide-swing candidate passed every nominal spec.
The original VCM sweep also has a confirmed control-path error. Its claimed
1.3 dB benefit from changing VCM is unsupported.

## What changed

- `src/eqrl/circuits/tail_variants.py`: explicit simple and wide-swing cascode
  tail decks. The differential input pair, resistive loads, source-degeneration
  Rs/Cs, and receiver DFE remain the existing single-stage CTLE architecture.
- `src/eqrl/evaluator.py`: optional netlist and operating-point probe adapters.
  Default behavior is unchanged. Experimental candidates use `build_evaluator`
  with `fast=False`, real transient HD3 and noise, and the existing guard thresholds.
- `src/eqrl/experiments/tail_mirror_pilot.py`: separate resident-simulator worker
  processes, explicit VCM parameters, measured VCM validation, complete transistor
  probes, supply-current power measurement, expanded analytical transistor area,
  exact parameter snapshots, incremental output, and evaluation/time bounds.
- `src/eqrl/experiments/vcm_physics_check.py`: runs the existing SKY130 physics
  tests at both VCM settings, setting both the batch and resident control paths.

No training, optimizer integration, PPO action-space change, checkpoint promotion,
or dashboard change was made. Frozen-file SHA-256 checks are in every run manifest.
The shipped CTLE file, guard/reward/spec files, eye implementation, delivered circuit,
policy, and final-report code remain unchanged by this work. Existing user changes
elsewhere were preserved.

## Circuit definition and limits

The source is [ECE5211 Chapter 6, part 2, Figure 12 and its following discussion](https://www.d.umn.edu/~htang/ECE5211_doc_files/ECE5211_files/Chapter6_part2.pdf).
It is a wide-swing **cascode** mirror: a cascode is added to each output branch and
the reference branch, with another diode-connected transistor setting the cascode
bias. It has nine total MOSFETs including the input pair. The original circuit has
five. This technique reduces compliance voltage relative to a conventional cascode;
it does not promise less headroom than the original simple mirror. Smaller bias-device
width is used to allow operating margin and body effect, as discussed in the source.

All added MOSFETs are SKY130 devices with grounded bulk. Every MOSFET is probed and
must meet the unchanged 50 mV saturation-headroom floor. Tail current is measured at
the CTLE source nodes through the upper output devices in the final implementation.
The first screening artifact measured lower-branch currents; that difference does
not affect its all-invalid wide-swing verdicts and those records are retained.

The added bias branch is powered from VDD and included in measured supply power.
Area includes all six full-sized mirror/cascode devices plus the smaller bias device,
with legal finger widths. Area remains an analytical estimate with the existing
fixed routing allowance, not a layout result. Like the baseline, the variant uses
ideal trimmed reference currents; autonomous bias generation, mismatch, layout,
parasitic extraction, and full PVT validation are outside this pilot.

## Corrected VCM evidence

The historical script changes `ctle.VCM`. The resident simulator instead sources
`ctle.param_deck`, whose common-mode source uses `VCM_VDD_RATIO`. The historical
script builds its evaluator before the sweep and changes neither the loaded deck
nor that ratio. It also uses different CMA-ES seeds at different labeled VCMs.
Consequently its winner differences cannot establish a VCM optimum or explain one
through an out-of-band peak constraint.

The new harness varies a real `.param vcm_ratio` at every prime/reset, including
operating-point, AC, transient, and noise analyses. It verifies `v(cm)` against the
requested voltage and saves the value with each row. Matched measurements:

| Fixed sizing | Actual VCM/VDD | Boost, dB | Peak, GHz | Worst headroom, mV |
|---|---:|---:|---:|---:|
| Historical row labeled 0.68 | 0.68 | 14.0210 | 1.2697 | 153.3 |
| Same sizing | 0.72 | 14.0827 | 1.2697 | 214.0 |
| Historical row labeled 0.72 | 0.68 | 12.6767 | 1.3450 | 113.5 |
| Same sizing | 0.72 | 12.8188 | 1.3450 | 172.7 |
| Delivered sizing | 0.68 | 9.2287 | 1.5989 | 143.0 |
| Same sizing | 0.72 | 9.1658 | 1.5989 | 203.6 |

All six rows above passed the full guard with every device saturated and tail
delivery between 96.7% and 97.9%. Guard validity is separate from spec compliance:
for example, the historical high-boost candidates exceed the 12 dB maximum.
These are three matched designs, not optimized maxima over VCM. There is no basis
here for declaring either a hump-shaped optimum or a universal monotonic relation.

Evidence: `results/tail_mirror_pilot_20260910_v2/simple.json`.

## Wide-swing outcome

The first screen used three preselected designs at VCM/VDD = 0.60, 0.64, 0.68, 0.72.
It compared the original simple mirror with 1x and 4x mirror widths, both using a
cascode-bias device width ratio of 0.18. The simple mirror passed the guard in 9/12
cases; the two cascode sizes passed in 0/24. The 4x size delivered current accurately,
but its lower devices had only about 27 mV worst headroom at the higher VCM values.

A 24-evaluation refinement at 0.68 and 0.72 tested width scales 4 and 8 with bias
width ratios 0.16 and 0.14. No case passed the full guard. Rejections include
insufficient saturation margin and solver failures; missing measurements were not
filled in or treated as zero-valued circuit performance.

A final six-evaluation check used **8x mirror width and bias width ratio 0.12**.
Five cases passed the full guard; one failed to complete characterization. All five
guard-valid cases probe nine saturated MOSFETs, with minimum headroom 61.8-62.6 mV
and tail delivery 99.86-99.97%. None passed every nominal spec.

The in-band example uses the delivered input-pair/R/C/load sizing at VCM = 0.72 VDD:

| Metric | Original simple mirror | Experimental wide-swing |
|---|---:|---:|
| Boost | 9.1658 dB | 12.0458 dB |
| Peak frequency | 1.5989 GHz | 1.3450 GHz |
| Worst saturation headroom | 203.6 mV | 62.5 mV |
| Tail-current delivery | 97.8% | 99.97% |
| DC gain | 1.3619 dB | 0.9495 dB |
| HD3 | -61.7629 dB | -49.7138 dB |
| Input-referred noise | 0.5940 mV RMS | 0.7399 mV RMS |
| Supply power | 1.8969 mW | 2.5754 mW |
| Analytical area | 0.001002 mm2 | 0.006240 mm2 |
| Scored eye width | 0.625 UI | 0.5625 UI |
| Scored eye height | 868.0 mV | 740.2 mV |
| Default nominal spec checks | Pass | **Fail: boost exceeds 12 dB** |

These measurements use the default channel, not the frozen delivered artifact's
held-out channel. They are a matched experiment, not re-sign-off of that artifact.
The other four guard-valid wide-swing cases have peaks at 1.1314 or 1.1986 GHz,
below the allowed band, and eye width 0.375 UI. They also exceed the boost range.
Do not round 12.0458 dB to 12 dB and count it as a pass. No per-target tolerance
was enabled in this screening study; nine default checks were scored explicitly.

Evidence: `results/tail_mirror_pilot_20260910_v4/wide_swing_8x_bias12.json`.
These results demonstrate a guard-valid experimental implementation, not a lower-risk
replacement or a submission-ready improvement. Additional sizing and full PVT
acceptance would be required before adoption; neither is claimed here.

## Validation and cost

- 58 unit/wiring tests passed, covering evaluator adapters, probing added devices,
  tail-current nodes, explicit VCM, area accounting, default preservation, and JSON flags.
  Supplying an experimental netlist without its matching probe (or vice versa) is
  rejected at evaluator construction. This final fail-fast check and adapter
  docstrings were added after the simulations; paired-hook execution is unchanged.
  Final source hashes are in `results/tail_mirror_implementation_validation_20260910.json`.
- The completed pilot stages took 34.625 + 35.656 + 9.219 = 79.500 seconds for
  66 full evaluation attempts. A preserved first attempt ended after two evaluations
  on a NumPy-boolean serialization error (6.672 seconds), subsequently fixed and tested.
- Total pilot cost including that failed startup: **68 evaluation attempts, 86.172 seconds**.
- Every completed pilot manifest confirms the protected file hashes are unchanged.
- Existing VCM physics-test results are recorded separately under
  `results/vcm_physics_check_20260910_v1`: **5/5 SKY130 tests passed at 0.72,
  and 5/5 passed at 0.68**, with no skips. Each setting used 24 AC evaluations,
  plus the suite's existing operating-point headroom probes. Combined wall time
  was 291.265 seconds, inside the 300-second cap; protected hashes are unchanged.
  This runs `TestPipelineSky130`, the five transistor-level direction/behavior tests,
  not the unrelated theoretical and behavioral-backend tests in the same file.
  Passing these tests supports the existing physics contract at either VCM; it
  does not establish an optimized boost maximum, full guard validity for every
  swept point, or PVT acceptance. The separate full-characterization pilot supplies
  the guard-validity evidence reported above.

Reproduce from an activated ngspice/SKY130 environment with `PYTHONPATH=src`, using
new output directories so existing evidence is never overwritten:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_tail_variants.py tests/test_evaluator.py tests/test_probe.py -q --basetemp=work/tail-variant/pytest-new
.venv/Scripts/python.exe -m eqrl.experiments.tail_mirror_pilot --phase screen --output results/tail_screen_new
.venv/Scripts/python.exe -m eqrl.experiments.tail_mirror_pilot --phase refine --output results/tail_refine_new
.venv/Scripts/python.exe -m eqrl.experiments.tail_mirror_pilot --phase confirm --output results/tail_confirm_new
.venv/Scripts/python.exe -m eqrl.experiments.vcm_physics_check --output results/vcm_physics_new
```

## Handoff corrections

The handoff's PVT statement also confuses candidates with corners: the historical
delivered artifact records 45/45 passing corners; one of 22 candidate designs
survived the full sweep. Those historical results are not freshly revalidated here.
Keep `VCM_VDD_RATIO = 0.72` and the original tail in production. The wide-swing
implementation remains an opt-in experiment with its failures and costs exposed.
