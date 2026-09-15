# SNR robustness — an extension experiment

**This is not part of the Astera Labs problem statement.** That statement specifies a
CTLE retargeting task with no link SNR and no BER requirement. Nothing here changes the
objective, the reward, the guard layer, the observation, the frozen `compute_eye`
scorer, or any published benchmark number. It is a stress test run *beside* the
submission, and every number in it lives in its own artifact.

Run date: 2026-09-09. Policy under test: `results/seq_clean40k.zip` — the checkpoint
`results/delivered_circuit.json` names as the source of the delivered circuit.

---

## 1. How the project handled noise before this experiment

Three unrelated things in this repository are called "noise". Only the third is link SNR.

| name | where | what it is |
|---|---|---|
| `noise_vrms` | `sim/measures.py:input_noise`, SPICE `.noise` 10 MHz–5 GHz | integrated **input-referred circuit noise**, a scalar *constraint* (`spec.noise_vrms_max = 1.5 mV`). Scored as a reward margin and a `hard_pass` check. Never touches a waveform. |
| `anchor_noise` | `envs/sequential_env.py` | reset jitter on the **design parameters**. Nothing to do with signal noise. |
| `noise_sigma_v` | `sim/eye.py:_decision_feedback_v2` | the only **AWGN on the signal** in the project. Pre-existing; used by `experiments/snr_ber_sweep.py`. |

Two consequences follow, and they shape this entire experiment:

- **The scored eye is noiseless.** `measures.measure_all` calls `compute_eye` (v1),
  which is deliberately frozen and takes no noise argument. Only `compute_eye_v2`
  accepts noise.
- **The observation carries no noise term.** `SequentialEqualizerEnv.observation_space`
  is 18-dim: 6 design params, 8 normalised measures, target, channel, and two gap terms.

Therefore **the policy is SNR-invariant by construction**: for a given (target, channel)
it emits the identical design at every SNR. This is a property of the architecture, not
a finding about robustness, and it is why the results below separate what can move from
what cannot.

## 2. Files added

| file | why |
|---|---|
| `src/silq/experiments/policy_snr_sweep.py` | the whole experiment: rollouts, injection, verification, metrics, aggregation, plot, CSV |
| `tests/test_policy_snr_sweep.py` | 24 sanity tests on the noise mathematics and the measured-SNR recovery; no SPICE, no checkpoint |
| `docs/RESULTS_SNR_ROBUSTNESS.md` | this file |

Nothing in the measured path was modified. In particular `sim/eye.py`, `sim/measures.py`,
`envs/`, `specs.py` and `experiments/snr_ber_sweep.py` are untouched, so this study cannot
have moved a benchmark number even by accident.

The dashboard was then wired to display the result, which is presentation only — it reads
the artifact and computes nothing:

| file | change |
|---|---|
| `server.py` | one `CURATED` registry entry, and a read-only `GET /api/snr-robustness` that projects the artifact down to what the panel draws. Deliberately *not* folded into `/api/design-time`: that endpoint assembles the claim the brief asks for, and the UI keeps the two apart. |
| `dashboard/index.html` | one nav link and one `.lab-section.extension` block, placed after the benchmark section and flagged as an extension in the markup itself |
| `dashboard/static/charts.js` | `renderLineChart()` — a new renderer whose reason to exist is `shadeBelowX`, the infeasible band |
| `dashboard/static/app.js` | `renderSnrRobustness()` / `initSnrRobustness()` |
| `dashboard/static/styles.css` | classes for the extension flag, the band, and the curve; all colours are existing tokens, so both themes follow |
| `tests/test_snr_endpoint.py` | contract for the route — both anchors survive, no curve row carries an SNR-invariant metric, missing artifact degrades to the structured 404 |

Two presentation decisions are load-bearing rather than cosmetic, and both exist because
the dominant visual on this panel is a run of six zeros:

1. **The unreachable band is drawn on the chart, not written in a caption below it.** A
   reader who sees the shape before reading the prose reads four zeros as the system
   failing. The band and its on-plot label "no design can pass here" arrive at the same
   moment as the zeros.
2. **The SNR-invariant metrics are rendered once, in their own card, never as six rows.**
   Solve rate, boost error and evaluation count cannot move with SNR (§1, §6); tabulating
   them per SNR point would present an architectural constant as a robustness result.

The empirical BER column renders a measured zero as "0 errors" rather than as an upper
bound. One error in 8,192 transmitted bits is 1.2e-4, but fewer symbols are *scored* than
sent — startup exclusion — so the honest bound is looser than that ratio and is quoted as
an order of magnitude, never as a BER.

## 3. Injection point and equation

```
sigma_v = signal_rms / 10 ** (SNR_dB / 20)
noisy   = Y + rng.normal(0.0, sigma_v, size=Y.shape)
```

Injected at `sim/eye.py:_decision_feedback_v2` — **the slicer decision samples**,
post-channel, post-CTLE, before decision feedback. That is the architecturally correct
point: it is where the receiver actually decides, so the noise competes with the same
quantity the eye metric measures, and the DFE sees noisy decisions exactly as a real one
would. Injecting earlier (at the channel input) would be filtered by the CTLE and would
conflate the noise model with the frequency response under test; injecting later would
not affect the decisions at all.

`signal_rms` is the **absolute main-cursor voltage `|c0|`** of the same design's
noiseless response — the amplitude the slicer decides against. This is the same referral
`experiments/snr_ber_sweep.py` uses, so the two studies share one scale.

Reproducibility: the eye seed is explicit (`--eye-seed`, default 20260909) and offset per
spec index; inside `compute_eye_v2` the noise stream is a `SeedSequence` child
independent of the phase-selection and DFE-adaptation streams, so noise cannot tune the
sampling phase or the DFE tap.

Model limitations, stated because they bound every claim below: flat-spectrum
slicer-referred noise with no measured PSD; no channel thermal noise; no transmitter
noise; no jitter, package or clock-recovery modelling.

## 4. Verification — requested vs measured SNR

The noisy and noiseless runs share a seed, hence share bit pattern, sampling phase and
DFE tap, so their scored-sample difference *is* the injected noise — except on rows whose
preceding decision disagreed, where one-tap feedback subtracts a different value. Those
rows are excluded and the residual's standard deviation is measured directly.

| requested | measured (mean of 23) | error | across-spec σ |
|---|---|---|---|
| −5.0 dB | −5.002 dB | −0.002 dB | 0.024 dB |
| 0.0 dB | −0.002 dB | −0.002 dB | 0.021 dB |
| 5.0 dB | 4.999 dB | −0.001 dB | 0.020 dB |
| 10.0 dB | 10.000 dB | 0.000 dB | 0.019 dB |
| 12.0 dB | 12.000 dB | 0.000 dB | 0.019 dB |
| 13.0 dB | 13.000 dB | 0.000 dB | 0.019 dB |
| 14.0 dB | 14.000 dB | 0.000 dB | 0.019 dB |
| 15.0 dB | 15.000 dB | 0.000 dB | 0.019 dB |
| 16.0 dB | 16.000 dB | 0.000 dB | 0.019 dB |
| 18.0 dB | 18.000 dB | 0.000 dB | 0.019 dB |
| 20.0 dB | 20.000 dB | 0.000 dB | 0.019 dB |

The injection delivers the SNR it is asked for, at every point, to within 0.002 dB of
the request. The σ column is the spread *across the 23 designs*, not a measurement
error: each design has its own `|c0|`, so each gets its own σ, and 0.02 dB is how
closely the per-design recovery tracks the request.

## 5. How to run it

The shipped artifact was produced with an explicit SNR list, because everything
interesting happens between 10 and 20 dB and the module's default grid
(`SNR_POINTS_DB`, the six points the study was specified with) samples that span with a
single 5 dB segment:

```bash
PYTHONPATH=src python -m silq.experiments.policy_snr_sweep --specs 24 \
  --snr-db -5 0 5 10 12 13 14 15 16 18 20
```

The default six-point grid still spans the same −5…20 dB window and reproduces the
same numbers at the points it shares — the added points were verified to leave all six
original rows, both noiseless anchors and the whole SNR-invariant block bit-identical,
which is expected: the noise stream is derived from `(spec, base seed)` and not from
the SNR value or its position in the list, so adding points is purely additive.

```bash
PYTHONPATH=src python -m silq.experiments.policy_snr_sweep --specs 24
```

Full run: 24 held-out specs, ~8.5 minutes at six points, ~12 at eleven — the extra
points re-score the same designs and re-run no search. Re-aggregate and re-plot from an
existing artifact without re-simulating:

```bash
PYTHONPATH=src python -m silq.experiments.policy_snr_sweep --report-only
```

Options: `--model`, `--specs`, `--snr-db`, `--n-bits`, `--eye-seed`, `--corner`,
`--vdd`, `--temp-c`, `--boost-tol`, `--out`, `--csv`, `--plot`, `--overwrite`.

Artifacts: `results/policy_snr_sweep_v1.json` (schema `silq.policy_snr_sweep.v1`, carries
every per-spec and per-point record), `results/policy_snr_sweep_v1.csv` (one row per
spec × SNR), `results/policy_snr_sweep_v1.png`.

## 6. Results

Specs drawn with the same seed and generator as `experiments/policy_rollout.py`
(seed 0, target ~ U(5, 11) dB, channel ~ U(8, 16) dB), so all eleven SNR points see
**identical channels, identical targets, identical rollout seeds and an identical
evaluation budget**.

### SNR-invariant, reported once

Repeating these per SNR point would produce eleven identical rows and present an
architectural constant as a robustness result.

| quantity | value |
|---|---|
| specs | 24 (23 produced a valid design) |
| solve rate | 0.833 |
| strict pass, frozen v1 scorer | 0.833 |
| strict pass, v2 scorer, zero noise | **0.833** |
| mean absolute boost error | 3.18 dB |
| mean evaluations per spec | 6.8 |
| mean simulator failures per spec | 3.62 |

The v2 zero-noise anchor equalling the frozen v1 rate matters: it means the v1→v2
measurement-contract change costs nothing on this spec set, so **the entire curve below
is attributable to noise** rather than to the change of scorer.

### Per SNR point

| SNR | strict | loose | BER (empirical) | BER (semi-analytic) | eye height | eye width |
|---|---|---|---|---|---|---|
| −5 dB | 0.000 | 0.000 | 2.89e−01 | 2.87e−01 | 0.0 mV | 0.000 UI |
| 0 dB | 0.000 | 0.000 | 1.63e−01 | 1.60e−01 | 0.0 mV | 0.000 UI |
| 5 dB | 0.000 | 0.000 | 4.26e−02 | 4.08e−02 | 0.0 mV | 0.000 UI |
| 10 dB | 0.000 | 0.000 | 1.67e−03 | 1.53e−03 | 0.0 mV | 0.000 UI |
| 12 dB | 0.000 | 0.000 | 1.86e−04 | 1.63e−04 | 38.3 mV | 0.101 UI |
| 13 dB | 0.000 | 0.000 | 8.53e−05 | 4.19e−05 | 108.0 mV | 0.220 UI |
| 14 dB | 0.042 | 0.042 | 1.07e−05 | 9.13e−06 | 188.6 mV | 0.293 UI |
| 15 dB | 0.333 | 0.333 | 0 errors | 1.64e−06 | 263.6 mV | 0.378 UI |
| 16 dB | 0.625 | 0.625 | 0 errors | 2.32e−07 | 330.3 mV | 0.427 UI |
| 18 dB | 0.750 | 0.750 | 0 errors | 1.66e−09 | 440.6 mV | 0.497 UI |
| 20 dB | **0.833** | 0.833 | 0 errors | 1.45e−12 | 525.9 mV | 0.535 UI |

The five points at 12, 13, 14, 16 and 18 dB were added after the first pass, because the
specified six-point grid put the whole transition inside one 10→15 dB segment: four dead
zeros, then 0.333, and no measured shape in between. They cost no extra search — see §5.

Empirical BER reaches exactly zero from 15 dB up; above that the semi-analytic column is
the one carrying information, and the "0 errors" cells are the literal measurement, not
an upper bound (§2). Strict = every non-eye `hard_pass` check from the design's own frozen
measurement, AND-ed with the eye measured at that SNR; loose = boost-range plus that same
eye. Only the eye differs between rows.

The pass rate does not step from 0 to 0.833; it ramps over about 6 dB, and the ramp is
the sum of 23 per-design closure points rather than one threshold.

### Why the low-SNR rows are not a policy failure

`sim/eye._truth_opening` returns `positive.min() - negative.max()` — a **worst-case**
opening over every scored symbol, not an average. Under iid Gaussian noise both extremes
walk outward, so the opening loses roughly `k·σ` with `k ≈ 2·E[max of N/2 standard
normals]`. Measured against the data:

| SNR | −5 | 0 | 5 | 10 | 12 | 13 | 14 | 15 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fitted `k` | 7.06 | 6.88 | 6.54 | 5.98 | 5.78 | 5.68 | 5.53 | 5.40 | 5.27 | 4.98 | 4.66 |

`k ≈ 7` where noise dominates, exactly as the extreme-value argument predicts, drifting
down at high SNR as residual ISI takes over. The consequence is a hard ceiling:
`opening₀` cannot exceed `2·|c0|` (zero ISI, every symbol on its cursor), so
`eye_v ≥ 100 mV` is **unsatisfiable by any design in any parameter space** once
`2·|c0| − 7σ < 100 mV`.

| | SNR below which the eye_v check fails |
|---|---|
| this policy's delivered designs (mean opening₀/|c0| = 1.473) | 14.86 dB |
| the best design observed (ratio 1.699) | 13.37 dB |
| a physically perfect design (ratio 2.0) | **11.78 dB** |

So the −5, 0, 5 and 10 dB rows are zero because the metric is unreachable there, not
because the policy chose badly. The shaded band in the plot marks this region.

The denser grid turns that ceiling table into a prediction and then checks it. The best
design in the set closes at 13.37 dB, so **exactly one of the 24 specs should pass at 14
dB and none at 13** — and the measured strict rates are 0.042 (= 1/24) at 14 dB and 0.000
at 13 dB. The reachability model and the measured ramp agree at the point where they can
be made to disagree.

Do **not** read the closure SNR off the eye-height curve. The plotted polyline is
straight-line interpolation between samples, so it crosses the 100 mV spec line at about
12.9 dB — between the 38.3 mV sample at 12 dB and the 108.0 mV sample at 13 dB — which is
2 dB below the actual closure. (On the original 5 dB grid the same artefact put the
apparent crossing near 11.9 dB, a 3 dB error; a denser grid shrinks it but does not
remove it.) The measured closure is the 14.86 dB figure in the table above, computed per
design from its own noiseless opening — not a graphical crossing. Both the plot and the
dashboard panel draw that value as its own marked line for this reason.

### CTLE noise amplification

Spearman rank correlation across the 23 scored specs:

| SNR | boost ↔ BER | \|boost error\| ↔ BER | boost ↔ eye height |
|---|---|---|---|
| −5 dB | +0.33 | −0.27 | — |
| 0 dB | +0.43 | −0.34 | — |
| 5 dB | +0.51 | −0.38 | — |
| 10 dB | **+0.52** | −0.39 | — |
| 12 dB | +0.50 | −0.37 | −0.37 |
| 13 dB | +0.48 | −0.36 | −0.31 |
| 14 dB | +0.48 | −0.36 | −0.18 |
| 15 dB | +0.47 | −0.36 | −0.04 |
| 16 dB | +0.41 | −0.33 | −0.03 |
| 18 dB | +0.36 | −0.27 | +0.16 |
| 20 dB | +0.35 | −0.25 | +0.24 |

The third column is undefined below 12 dB: every design's eye height is exactly 0 mV
there, so the ranks are all tied and no correlation exists to compute.

**More CTLE boost correlates with worse BER**, most strongly around 10 dB — the classic
noise-amplification tradeoff, visible in the data. The third column shows the same
tradeoff changing hands: where noise dominates, more boost means a *smaller* eye
(−0.37 at 12 dB), and by 20 dB the sign has flipped (+0.24) because residual ISI, which
boost removes, has become the binding term. Both effects are weak and the caveat below
applies to all of them. The negative sign on the second column
is the same effect seen from the other side: this policy's boost error is mostly
undershoot, so a larger error means less boost means less amplified noise.

Caveat, and it is a real one: boost is not randomised independently of channel loss here
— higher-loss specs get more boost — so this correlation is confounded with channel loss
and is suggestive, not causal. Isolating it needs a controlled sweep holding the channel
fixed, which this experiment does not do.

## 7. Was retraining justified? No.

The eval sweep shows clear degradation, which is the stated trigger for considering
SNR-aware fine-tuning. It was not done, for three reasons that survive scrutiny:

1. **At 20 dB there is nothing to fix.** The pass rate is 0.833, identical to the
   noiseless rate. The policy is already fully robust there.
2. **At ≤10 dB no policy can fix it.** Proven above: the check is unsatisfiable by any
   design. Retraining would be optimising against a metric with no feasible point, and
   the resulting curve would look like progress while measuring nothing.
3. **All the headroom sits in one ~3 dB window**, between the 11.78 dB physical ceiling
   and this policy's 14.86 dB mean closure. The denser grid measures inside that window
   rather than bracketing it: 12 dB and 13 dB are 0.000, 14 dB is 0.042. So the window is
   real and the policy is near its top edge — but the best design it *already* produces
   closes at 13.37 dB, which means most of those 3 dB are reachable by better *targeting*
   within the existing objective, not only by SNR awareness. Above the window there is
   nothing to buy: 16 dB is already 0.625 and 18 dB 0.750, on their way to the noiseless
   0.833.

Buying that one row costs an SNR observation slot and a reward change, both of which the
26 Aug 2026 architecture freeze forbids, and a 19-dim observation is incompatible with
every existing checkpoint — there is no honest weight transfer across a changed input
dimension beyond zero-initialising the new column, which is not fine-tuning.

Nothing was retrained, so there is no catastrophic-forgetting result to report and the
Astera benchmark is untouched and unchanged.

## 8. Unexpected findings

- **The v2 scorer costs nothing on this spec set.** Its zero-noise pass rate matches the
  frozen v1 rate exactly (0.833/0.833). That was not assumed; it was measured, and it is
  what makes the curve interpretable.
- **The eye metric is an extreme-value statistic, and that dominates its noise
  behaviour.** `min − max` over 8k symbols loses ~7σ. Any future noise work that reads
  eye height as an average will misread it by a factor of about seven.
- **`solve rate = 0.833` does not mean the target boost was hit.** With `--boost-tol`
  off (the default, matching `policy_rollout`), `hard_pass` only requires boost inside
  the 3–12 dB range. The mean absolute boost error is 3.18 dB. This is pre-existing
  benchmark semantics, not an SNR result, but it is easy to misread in the table above.
- **A pre-existing SNR/BER study already exists.** `experiments/snr_ber_sweep.py`
  characterises one *fixed delivered circuit* across CTLE-off / CTLE-on / CTLE+DFE
  (`results/snr_ber_sweep_v1.json`). It shares the injection point and the referral, so
  the two are on one scale; it answers a different question (circuit-level equalisation
  gain) from this one (policy-level pass rate).

## 9. Recommended next step

If this experiment is taken further, the highest-value move is **not** SNR-aware
training. It is a **freeze-compatible closed-loop variant**: measure the environment's
eye under noise so the degraded `eye_h`/`eye_v` reach the policy through the *existing*
18-dim observation. That needs no new observation slot, no reward change, and no
checkpoint migration — every existing checkpoint stays loadable — and it answers the
sharper question of whether the policy can compensate when it can *see* the degradation.
It costs one full rollout budget *per SNR point*, since the trajectory then differs per
SNR — which is also why that variant cannot be made denser for free the way this one was.

That variant still requires a decision, because changing what the environment measures is
a measurement-path change even when the observation shape is unchanged. It has not been
run.
