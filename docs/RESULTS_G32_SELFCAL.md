# Historical result — H-SC: G3.2 self-calibration development slice

**Historical verdict: equivalent on the development slice. Current verdict: fresh
revalidation pending.** Criterion, slice and verdict rule were committed in
`docs/PREREG_G32_SELFCAL.md` (`4b4c6a3c2`) before the harness existed
(`a3e081eb5`), which ran before this document. Artifact:
`results/g32_selfcal_bench.json`, 155 simulations.

The figures below are retained for traceability, not as fresh delivery validation. A concurrent
`eye.py` self-labelling correction changes future scores. The historical artifact, the frozen
configuration, and the delivered result remain unchanged; a fresh revalidation is pending.

Reproduce:

```
PYTHONPATH=src python -m silq.experiments.g32_selfcal_bench
```

## 1. What was compared

G3.2 was run twice per spec from the **same** stage-1 handoff, on spec-seed 3 specs 8–17
(the burned development slice — the held-out seed 23 was not touched). One arm read the
frozen seed-2 slope table; the other measured its slopes at the design in hand with 2–4
single-axis perturbations and **paid for them out of the same 10-evaluation stage-2 pool**,
so it ran on `r − spent`. Everything else — evaluator, rescue ladder, budget, tolerance —
was identical. The only difference between the arms is the plane dict.

**Historical internal-harness gate: passed.** The frozen arm inside this harness reproduced
`results/g32_repair_smoke.json` exactly on all ten specs and all five outcome fields, so
the differences below are attributable to the plane and not to the harness under the historical
scoring path. This does not validate scores after the `eye.py` correction.

## 2. Archived figures — not current performance claims

| criterion | frozen | calibrated | passes |
|---|---|---|---|
| P1 strict solves (±1.5 dB) | 6/10 | 6/10 | yes |
| P2 loose solves | 7/10 | 7/10 | yes |
| S1 specs with a valid design | 7/10 | 7/10 | yes |
| S2 paired median Δ`best_abs_err` | — | **+0.0000 dB** over 7 pairs | yes |

Per spec. `Δ` is calibrated − frozen best absolute target error, so positive is worse:

| spec | calibration | frozen | calibrated | Δ (dB) | trace |
|---|---|---|---|---|---|
| 8 | 4 slopes / 2 ev | 0.0676 strict | 0.0676 strict | +0.0000 | identical |
| 9 | 0 slopes / 4 ev | no valid design | no valid design | — | identical |
| 10 | 5 slopes / 2 ev | no valid design | no valid design | — | differs |
| 11 | 5 slopes / 2 ev | 0.0054 strict | 0.0054 strict | +0.0000 | identical |
| 12 | 5 slopes / 2 ev | 0.0597 strict | 0.0597 strict | +0.0000 | identical |
| 13 | 0 slopes / 4 ev | 0.0091 strict | 0.0091 strict | +0.0000 | identical |
| 14 | 4 slopes / 2 ev | 0.1879 strict | 0.1879 strict | +0.0000 | identical |
| 15 | 0 slopes / 4 ev | 2.3320 loose | 2.3394 loose | +0.0073 | differs |
| 16 | 4 slopes / 2 ev | no valid design | no valid design | — | differs |
| 17 | 5 slopes / 2 ev | 0.4142 strict | 0.4400 strict | +0.0258 | differs |

Largest single regression: **+0.026 dB**, on a spec that stays strict-solved. No spec
changed solve status in either direction.

## 3. What the measured slopes actually were

The boost slope was genuinely measured and genuinely varies, on the same axis (`rs`) and
always with the same sign as the frozen constant:

| spec | 8 | 10 | 11 | 12 | 14 | 16 | 17 | frozen |
|---|---|---|---|---|---|---|---|---|
| `d_boost_db_per_unit` | 21.79 | 15.72 | 7.99 | 13.46 | 18.92 | 19.68 | 10.61 | **12.45** |

That is a 2.7× spread around the frozen value — and the outcomes were unchanged anyway on
six of those seven specs. **That is the real content of this result: the controller is
insensitive to its boost slope across a 2.7× range.** Which is also why the constant cannot
have been finely tuned — there is nothing to tune it against at that resolution.

The peak slope is a different story, and the honest reading is not the one the experiment
was designed to produce. `d_ln_peak_per_unit` came back as **−1.1527 on every spec where it
survived the information floor — identical to the frozen constant to four decimals.** That
is not a coincidence and it is not confirmation: the peak frequency is `freq[argmax]` on a
logarithmic AC sweep of ~40 points per decade (5.93% per point, the same `grid_step_pct` the
frozen plane records), so it is quantized. On those specs the peak moved by exactly **one
sweep grid point** for a 0.05-unit `l_in` perturbation, giving ln(1.0593)/0.05 = 1.153. The
frozen constant is that same one grid point. On specs 8, 14 and 16 the peak moved **zero**
grid points, the slope came back at 0, and the preregistered information floor correctly
rejected it and fell back.

So the peak-axis constant is not a fitted quantity at all — it is the resolution of the AC
sweep. That supports "these were not hand-tuned" by a different route than intended, and it
should be said that way rather than as "we re-measured it and got the same number."

## 4. The three specs where nothing was measured

On specs 9, 13 and 15 every perturbation was rejected by the guard or landed with the peak
on the sweep edge. The calibration fell back to the frozen plane entirely — and spent 4
evaluations finding that out. On those specs "equivalent" is true **by construction, not by
evidence**, and spec 15's +0.007 dB is purely the cost of the wasted budget. Three of ten
is not a small failure rate, and it is the main practical argument against switching the
default on.

## 5. What the historical artifact suggests, and what it does not

Before the eye-score correction, this development artifact suggested that local boost slopes
varied while the controller's archived outcomes did not change on the burned slice. The peak
axis remains limited by AC-grid quantization, and three development cases fell back to the
frozen plane. It does not establish behavior on held-out data, the delivered circuit, or the
axis choice (`rs` for boost, `l_in` for peak). No archived figure here should be quoted as a
current performance result.

**The frozen configuration remains the delivered configuration.** No constant in
`g32_repair.py` changed and the historical artifact is preserved. A fresh delivered-design
revalidation with the corrected eye scorer is pending. Until it is complete, the
self-calibrating path remains an archived development option rather than a validated default.
