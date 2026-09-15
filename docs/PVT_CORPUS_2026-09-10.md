# PVT pipeline integration and corpus evidence, 2026-09-10

**Status: complete**. 7 independently verified clean cases out of the fixed 22-case set; 22 cases completed. Historical baseline: 1/22 clean.

The historical audit contains 22 nominally successful candidates, each evaluated at 45 corners. It is not 22 corners. This experiment measures the repair rate conditional on those 22 candidates; it does not establish a fresh PPO solve rate on all requests. Pending cases are not counted as failures or successes.

## Verified corrections to the proposed plan

- Spec 0 was selected by lowest failing spec index, not greatest severity. Its 42/45 result was not the mildest failure: three cases passed 43/45.
- There are 12 cases in the 36-43 passing-corner group, six in 17-23, three in 28-32, and one clean case.
- The severe cases primarily fail DC-gain plausibility or saturation guards. Those failures do not establish a physical boost ceiling.
- The previous case already became 45/45 with the fixed delivered-sizing anchor. CMA-ES improved margins; it was not solely responsible for closing that case.

## Production behavior

`silq.pipeline.design()` now runs nominal PPO/corpus search and G3.2, nominal verification, then `apply_pvt_stage`. The stage performs fresh full-grid checks, searches if needed, and uses a second simulator process for final acceptance. A successful repair replaces the actual returned sizing, SPICE netlist and verification measurements. The previous candidate is retained under `pre_pvt`. A failed or interrupted PVT stage returns `pvt_not_verified`, never `solved`.

A fixed delivered-sizing anchor is remeasured at the requested target/channel and can be selected. Such reuse is explicitly marked `fixed_anchor_reused` and `is_ai_generated=false`. PPO itself is unchanged. `pvt=False` preserves the explicit historical nominal benchmark path; production uses PVT by default. SNR remains optional and separate. If requested, its sampled assessment cannot override failed PVT acceptance. This is not a joint SNR-by-PVT sign-off.

The dashboard shows a PVT stage and an independent 45-corner verdict. Final sizing, drawing, measurements and exports use the selected circuit. PVT measurement costs are counted in the worker and included in total measurements and SPICE analyses; nominal optimizer costs remain labeled separately.

## Fixed protocol and limits

All 22 cases use the same production repair function, ordered by historical pass count descending. First check the input on 45 corners. If clean, independently repeat it. Otherwise check the fixed anchor. If neither is clean, use five CMA-ES generations of eight designs at eight selected stress corners, fully check two finalists, then independently repeat the selected design on all 45 corners. The six sizing variables, physical guards, ten hard checks and +/-1.5 dB tolerance are retained. The process/supply/temperature grid is TT/SS/FF/SF/FS x 1.71/1.80/1.89 V x 0/27/125 C.

Per-case cap: 545 corner evaluations and 900 seconds. Full corpus cap: 11,990 corner evaluations and 19,800 seconds (5.5 hours). Workers run sequentially for the corpus, alongside the separately supervised SNR job. One additional live pipeline smoke uses target 8.92 dB and channel 12.34 dB, outside this historical set, with its own 545-corner/900-second PVT cap. Search exhaustion is reported as unresolved, not infeasible.

## Case results

| Spec | Target dB | Channel dB | Historical /45 | Fresh input /45 | Independent final /45 | Source | Status |
|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 8.920 | 14.828 | 45 | 45 | 45 | input | verified |
| 14 | 8.748 | 14.115 | 43 | 43 | 45 | anchor | verified |
| 18 | 5.960 | 11.983 | 43 | 43 | None | - | budget_exhausted |
| 24 | 5.664 | 11.643 | 43 | 43 | None | - | budget_exhausted |
| 0 | 9.164 | 13.132 | 42 | 42 | 45 | anchor | verified |
| 36 | 9.697 | 10.535 | 40 | 40 | None | - | budget_exhausted |
| 16 | 8.808 | 14.874 | 39 | 39 | 45 | anchor | verified |
| 39 | 6.752 | 11.476 | 38 | 38 | None | - | budget_exhausted |
| 28 | 8.562 | 10.774 | 37 | 37 | 45 | anchor | verified |
| 1 | 5.772 | 8.910 | 36 | 36 | None | - | budget_exhausted |
| 8 | 8.242 | 13.469 | 36 | 36 | 45 | anchor | verified |
| 20 | 9.088 | 9.003 | 36 | 36 | 45 | anchor | verified |
| 38 | 5.219 | 13.703 | 36 | 36 | None | - | budget_exhausted |
| 7 | 6.809 | 11.113 | 32 | 32 | None | - | budget_exhausted |
| 33 | 5.787 | 13.748 | 31 | 31 | None | - | budget_exhausted |
| 25 | 6.988 | 11.189 | 28 | 28 | None | - | budget_exhausted |
| 30 | 10.606 | 12.627 | 23 | 23 | None | - | budget_exhausted |
| 35 | 10.801 | 10.163 | 23 | 23 | None | - | budget_exhausted |
| 22 | 9.961 | 13.831 | 20 | None | None | - | failed |
| 13 | 9.970 | 10.027 | 17 | 17 | None | - | budget_exhausted |
| 17 | 7.986 | 9.867 | 17 | 17 | None | - | budget_exhausted |
| 19 | 9.472 | 11.595 | 17 | 17 | None | - | budget_exhausted |

Recorded completed-case corner evaluations: 900. Some work may be uncounted after interruption; inspect raw artifacts.

## Severe-case diagnosis

| Spec | Historical guard failures | Current evidence |
|---:|---|---|
| 30 | T4.10_dc_gain_implausible: 18 | unresolved within bounded sizing search; no infeasibility proof |
| 35 | T4.10_dc_gain_implausible: 20, T2.5_mosfet_not_in_saturation: 2 | unresolved within bounded sizing search; no infeasibility proof |
| 22 | T4.10_dc_gain_implausible: 20, T2.5_mosfet_not_in_saturation: 2 | unresolved within bounded sizing search; no infeasibility proof |
| 13 | T4.10_dc_gain_implausible: 18, T2.5_mosfet_not_in_saturation: 9 | unresolved within bounded sizing search; no infeasibility proof |
| 17 | T4.10_dc_gain_implausible: 25, T2.5_mosfet_not_in_saturation: 3 | unresolved within bounded sizing search; no infeasibility proof |
| 19 | T4.10_dc_gain_implausible: 23, T2.5_mosfet_not_in_saturation: 2 | unresolved within bounded sizing search; no infeasibility proof |

A passing design is a constructive feasibility witness at that specification. An unsuccessful local search is not a certified upper bound for the topology. PVT acceptance remains schematic-level: no Monte Carlo mismatch yield, extracted-layout parasitics, or low-BER certification is claimed.

## Evidence and validation

- Live machine-readable summary: `results/pvt_corpus_20260910_v1/summary.json`.
- Each case records configuration, source hashes, fresh baselines, search candidates when needed, selection, independent verification, raw SPICE artifacts, costs and the production-format final result.
- The delivered historical circuit and nominal PPO checkpoint are preserved. A repaired result is adopted for the individual live request, rather than silently replacing a global benchmark artifact.
- Validation: 75 focused Python tests passed (eight separate historical end-to-end tests excluded); nine JavaScript tests passed. The live pipeline smoke is recorded separately below.

Live pipeline smoke: {"complete": true, "accepted": true, "seconds": 416.5630000000019, "corner_evaluations": 135, "target": 8.92, "channel": 12.34}.
