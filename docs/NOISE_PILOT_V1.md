# Noise-conditioned pilot, version 1

Historical preparation notes. The active trainer now uses the [v2 contract and
watchdog](NOISE_PILOT_V2.md); v1 artifacts remain historical evidence.

This experiment trains a fresh policy with six additional observation values:
noise lower/upper bounds, signal swing and three mode indicators. The old PPO,
reward functions, guard thresholds and historical results remain unchanged.

## Input choices

| Choice | Request | Interpretation |
|---|---|---|
| Specific | `{"mode":"specific","value_vrms":0.005}` | Supplied 5 mV RMS at the CTLE input |
| Range | `{"mode":"range","low_vrms":0.001,"high_vrms":0.02}` | Supplied 1 to 20 mV RMS interval |
| Unknown | `{"mode":"unknown"}` | Explicit assumed 1 to 50 mV RMS interval |

Inputs are differential noise after the channel, before the CTLE. The pilot
assumes a flat one-sided external noise PSD from 10 MHz to 5 GHz, a 1 V
differential TX peak-to-peak swing, 5 Gb/s, nominal TT, 1.8 V and 27 C.
The assumed unknown interval is a training choice, not measured channel data.
The full supported training noise interval is 1 to 50 mV RMS. Zero external
noise is a validation control; receiver device noise still applies.

External output noise power is input noise power times the band average of
the squared CTLE transfer magnitude. Receiver output noise is directly
measured as differential `onoise_total` by SPICE over the same band. Independent
powers are added. Failed output-noise measurements invalidate the candidate;
the output measurement has no single-ended fallback.

The combined RMS is used as equivalent white Gaussian slicer noise in the v2
eye calculation. Spectrum integration is calibrated, but temporal noise
correlations are omitted. This pilot therefore tests conditioning under that
approximation; it is not a physical link sign-off model. Jitter, reflections,
crosstalk, clock recovery, and PVT robustness are outside this pilot.

The range score is the minimum score over its two endpoints, with success
requiring both endpoints to pass. This does not guarantee every interior
point passes. Noiseless phase/DFE calibration and the evaluation/noise streams
are fixed across those endpoints. Actual receiver decisions drive feedback.
Training uses 512-bit eyes; no low-BER extrapolation is used in the reward.

## Launch and acceptance

The bounded pilot requests 5120 steps, four independent simulator workers,
five PPO rollouts, and at most two hours. Source hashes, configuration,
progress, checkpoints and raw guard artifacts live in a fresh run directory.
The parent supervisor enforces the process-tree deadline, including native
simulator stalls; callback deadlines alone cannot do that.

Preflight requires the pure contract/noise/eye/environment regression tests
and real-SPICE verification that all three modes produce finite observations,
direct output noise is positive, larger input noise increases output noise,
and phase/DFE settings stay fixed across endpoint evaluations.

A completed pilot is not deployment approval. Before extending training or
wiring the new checkpoint into the production dashboard, evaluate unseen
target/channel/noise combinations against the frozen PPO and a fixed circuit
using the same noise scorer, seeds and evaluation budget. Report validity,
strict target tolerance (1.5 dB), noisy eye pass rate at sampled points,
empirical error counts, evaluation cost, and sensitivity to noise requests.
Check intermediate range points and independent patterns. Verify performance
within each mode; a pooled rate can hide failure in one mode.

The production dashboard continues using its existing policy until this
validation is complete. The noise request contract is experimental.
