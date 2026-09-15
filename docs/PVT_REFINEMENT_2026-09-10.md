# PVT refinement implementation and validation

This work adds bounded sizing refinement for the existing single-stage SKY130 CTLE
with its simple mirror, source degeneration, and receiver-side 1-tap DFE. The shipped
VCM remains 0.72 times VDD. No frozen reward, guard, spec threshold, eye scorer,
checkpoint, delivered artifact, or historical result is changed.

## Case and acceptance contract

Before new simulation, the repair case was selected as the lowest spec index among
the historically non-clean candidates in `results/pvt_signoff_seed23.json`: index 0.
It requests 9.163598483944185 dB boost and 13.131665767025844 dB channel loss.
The historical record was 42/45 passing corners; that number is not used as the
fresh baseline. The original sizing and the frozen delivered sizing are both
remeasured at this case's target and channel. The latter is an explicitly labeled
anchor, not a new PPO result.

The ten hard checks include the unchanged +/-1.5 dB target tolerance. All candidate
evaluations use `build_evaluator(..., fast=False)`, real transient HD3 and input
noise, and complete operating-point checks. Recorded health includes per-device
saturation and headroom, tail-current delivery, node voltages, and measured VCM.
The recorded deck uses the resident simulator's VDD-scaled common-mode expression,
including at non-nominal supplies. The existing TT-versus-SS model-change gate is
checked independently in both the optimization and verification processes.

Final acceptance requires every point in the exact cross-product:

- Process: TT, SS, FF, SF, FS.
- Supply: 1.71, 1.80, 1.89 V.
- Temperature: 0, 27, 125 C.

All 45 points must pass the guard and all ten hard checks. Missing, duplicate, or
wrong corners cannot count as a full-grid pass. A fresh simulator process repeats
the selected design across all 45 points before a `verified_candidate` is exported.

## Search and budget

The standalone search objective first minimizes guard failures, then failed corners,
then quantified guard violation. Among feasible candidates it maximizes the minimum
normalized margin across specs, saturation headroom, and tail delivery, with
worst-case target error as a tiebreak. This is an optimization objective in the new
module; the frozen RL reward is unchanged. Margin normalization is explicit in
`src/eqrl/pvt_refinement.py::signed_slacks`. `hard_pass` remains authoritative for
strict and inclusive boundaries, including the strict noise limit.

Eight search corners are chosen from freshly measured baseline weaknesses, including
at least one from every process and nominal TT. These corners guide search only.
CMA-ES uses seed 20260910, population 8, five generations, and initial normalized
step size 0.035 over the same six existing analog sizing variables. The initial
center is the stronger of the two freshly measured baselines. Finalists are the two
best new candidates on the search corners. Both receive the full 45-corner check,
and the best full-grid candidate, including the baselines, receives independent
verification. Reusing the anchor is reported separately from CMA-ES improvement.

Announced bound: **550 full corner evaluations, 900 seconds wall time**, plus the
existing operating-point/model-integrity probes. Planned full-evaluation cost:

| Stage | Maximum evaluations |
|---|---:|
| Two fresh 45-corner baselines | 90 |
| 40 candidates at eight search corners | 320 |
| Two full-grid finalists | 90 |
| Independent winner verification | 45 |
| Total | 545 |

Batches run sequentially, with corners outside candidates to reduce model reloads.
The evaluator rejects batches that exceed the count limit; the parent process
enforces a common wall deadline and preserves logs and protected-file hashes.
Every measured candidate retains raw simulator artifacts.

## Implementation

- `src/eqrl/pvt_refinement.py`: explicit margins, search ranking, stress selection,
  exact-grid acceptance, correct recorded PVT decks, and guarded batched evaluation.
- `src/eqrl/experiments/pvt_optimize.py`: reproducible repair case, bounded CMA-ES,
  full-grid selection, independent verification, and conditional candidate export.
- `tests/test_pvt_refinement.py`: partial-grid/duplicate rejection, guard and target
  failures, process coverage, strict boundary semantics, budget enforcement, and
  resident-deck VCM provenance.

The initial validation run passed 55 PVT acceptance, evaluator, and probe tests.
Its first test invocation hit a missing parent directory for pytest's temporary
files; creating the workspace-local parent resolved that setup issue.

Run from the project root after activating the existing ngspice/SKY130 environment
and setting `PYTHONPATH=src`. Use a new output directory:

```powershell
.venv/Scripts/python.exe -m eqrl.experiments.pvt_optimize --output results/pvt_optimization_new
.venv/Scripts/python.exe -m pytest tests/test_pvt_refinement.py tests/test_evaluator.py tests/test_probe.py -q --basetemp=work/pvt-refinement/pytest-new
```

The original process was interrupted after two saved generations and part of the
third. Fresh baselines and those two completed generations were preserved in
`results/pvt_optimization_20260910_v1`. The continuation in
`results/pvt_optimization_20260910_v3` reconstructs CMA-ES from the same seed,
checks the regenerated candidate designs against the saved designs, and replays
their measured rankings without resimulating completed generations. Partial
generation measurements remain in the original raw archive and are not claimed
as completed generations. The continuation was bounded separately at 330 additional
full evaluations and 600 seconds. A startup directory `v2` contains no simulations;
its launch exposed a JSON tuple/list comparison issue, corrected before resuming.

Final evidence is under `results/pvt_optimization_20260910_v3`:
`config.json`, fresh `baselines.json`, `search_corners.json`, generation records,
`selection.json`, independent `verification.json`, raw measurements, and the final
manifest. `verified_candidate.json` and `verified_candidate.cir` are written only
after all 45 independent verification corners pass.

## Scope of conclusions

This is a sizing-refinement case study at one explicit target/channel pair. It
does not establish a policy-wide solve rate, robustness for other targets/channels,
PVT-aware PPO training, Monte Carlo mismatch yield, extracted-layout behavior, or
BER certification. The existing frozen eye score remains the scored eye metric.
The result remains separate from the shipped design and is not auto-promoted.

## Final measured outcome

**Completed. The selected refined design passed all 45 corners and all ten checks
again in the independent verification process.** Both optimization and verification
passed the TT/SS model-integrity check. The final manifest confirms every protected
file hash is unchanged.

| Candidate | Passing corners | Minimum normalized margin | Worst target error | Minimum device headroom |
|---|---:|---:|---:|---:|
| Original input, fresh baseline | 42/45 | -0.02390 | 1.53585 dB | 12.7991 mV |
| Delivered-sizing anchor at this spec | 45/45 | 0.07600 | 1.32437 dB | 71.9277 mV |
| Selected refinement `g3_c5`, independently verified | **45/45** | **0.17096** | **1.21814 dB** | **58.5479 mV** |
| Other full-grid finalist `g2_c7` | 45/45 | 0.13984 | 1.00436 dB | 70.4195 mV |

The search selected `g3_c5` according to the declared maximin-margin objective.
`g2_c7` has better target accuracy and headroom but a smaller minimum normalized
margin overall. It was fully measured in selection, but only `g3_c5` received the
separate-process repeat. The objective does not imply every metric improved: the
selected design's minimum headroom is lower than the anchor's, though every device
still exceeds the unchanged 50 mV requirement. Normalized margin is an optimization
score, not a probability, yield, or accuracy percentage.

The original input has two guard failures and one target-tolerance failure. Reusing
the anchor alone already repairs the 42/45 pass count to 45/45. The additional benefit
demonstrated by CMA-ES is improved minimum normalized margin and worst target error
relative to that 45/45 anchor, not an additional increase in corner pass count.
For the failing input, worst target error is computed only over guard-valid rows;
invalid circuit measurements are never passed off as usable target estimates.

The independent verification records these limits across all 45 corners:

| Metric | Verified range |
|---|---:|
| Boost | 7.9455 to 9.9057 dB |
| Peak frequency | 1.5989 to 2.1329 GHz |
| DC gain | 0.8303 to 1.6067 dB |
| HD3 | -61.8439 to -45.9117 dB |
| Input-referred noise | 0.5471 to 0.7304 mV RMS |
| Power | 2.1343 to 2.4099 mW |
| Analytical area | 0.00105012 mm2 |
| Scored eye width | 0.6250 to 0.6875 UI |
| Scored eye height | 721.9 to 875.9 mV |
| Tail-current delivery | 95.4630% to 98.6065% |

Selected sizing: input W = 51.9463 um, L = 0.276959 um, tail current = 0.858035 mA,
Rs = 3169.8803 ohm, Cs = 170.5163 fF, Rload = 2292.6266 ohm. The mirror sizing is
derived by the existing circuit generator; the 1-tap DFE remains receiver-adapted.

Deliverables:

- `results/pvt_optimization_20260910_v3/verified_candidate.json`: sizing, exact spec,
  all 45 health/measurement records, and independent acceptance result.
- `results/pvt_optimization_20260910_v3/verified_candidate.cir`: nominal TT SPICE
  netlist for the selected circuit with the actual VDD-scaled VCM expression.
- `results/pvt_optimization_20260910_v3/selection.json`: full-grid comparisons,
  including the alternate finalist and both baselines.
- `results/pvt_optimization_20260910_v3/manifest.json`: completed worker statuses,
  source/protected hashes, and measured continuation wall time.

The successful continuation used **282 optimization/selection evaluations plus 45
independent verification evaluations = 327**, in **451.453 seconds**, inside its
330-evaluation/600-second cap. The interrupted original run retains 258 raw evaluations:
90 baselines, 128 in two saved generations, and 40 partial-generation evaluations.
Thus total actual SPICE corner evaluations across the interruption and continuation
are **585**, versus 545 for an uninterrupted run; 40 partial-generation evaluations
had to be repeated. The original parent never finalized a wall-time manifest, so a
combined compute wall time is not claimed. The successful continuation replayed
the 218 completed baseline/generation evaluations without rerunning them.

The 55 PVT acceptance, evaluator, and probe tests passed before continuation.
Resumption additionally verified exact equality of the saved and regenerated CMA
candidate designs for each replayed generation. No production file was promoted,
and no claim of PVT-aware PPO training or generalization to other specs is made.


## Subsequent pipeline and corpus work

The standalone scope above describes the original case study. Production `design()` now includes a PVT repair and independent acceptance stage, and the broader 22-case repair run is underway. See [PVT_CORPUS_2026-09-10.md](PVT_CORPUS_2026-09-10.md) for live evidence and its separate scope. This does not retroactively turn the single-case result into policy-wide validation.
