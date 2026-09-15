# SNR inputs and supervised noise pilot v2

This supersedes the v1 launch contract. Historical v1 artifacts and the frozen
production PPO remain unchanged. The dashboard assesses the existing search's
candidate against noise; it explicitly reports that noise did not steer that search.
There is no automatic checkpoint promotion.

## User and API contract

`POST /api/pipeline/run` accepts optional `noise_request` with schema
`eqrl.snr.request.v2`. Omitting it preserves legacy behavior. New dashboard runs
always send their selected noise mode.

```json
{
  "mode": "measured",
  "value_vrms": 0.005,
  "signal_reference": "tx_vpp",
  "signal_value_v": 0.8,
  "bandwidth_hz": [10000000, 5000000000]
}
```

- Measured requires noise RMS, signal, and both band edges.
- Estimated replaces `value_vrms` with `budget_vrms`, or both `low_vrms` and
  `high_vrms`. A budget is evaluated from zero through its maximum.
- Unknown accepts edited assumed bounds. Missing values default visibly to
  1–50 mV RMS, 1 V TX Vpp, and 10 MHz–5 GHz. `assumed_fields` preserves the
  provenance of displayed defaults through serialization. Unknown noise is
  always an assumption, including user-edited bounds.
- Signal reference is `tx_vpp` or `ctle_input_vrms`. RMS is differential signal
  RMS in the stated band after the channel. A fixed independent 2048-bit
  channel-only calibration determines equivalent TX swing; there is no fixed
  RMS-to-Vpp conversion for a distorted waveform.

Noise is differential at the CTLE input after the channel. Supported band edges
are within 10 MHz–5 GHz and must contain resolvable calibration signal energy.
Input/output SNR uses signal and noise at matching references and in the same
band; output SNR is before DFE. TX-entered input RMS and output signal RMS are
modeled. Zero external noise has a null input SNR plus an explicit zero-noise
flag, never an invalid JSON infinity.

External noise power is integrated through the CTLE transfer function. SPICE
directly measures differential receiver output noise over the requested band.
Independent powers are added, then approximated as equivalent white slicer
noise. Five equally spaced range points use the same calibration and noise
streams. Every sampled point must pass. Results include empirical errors/counts,
worst point, per-field provenance, model assumptions, and assessment cost.
Unknown passes remain conditional. These are finite-pattern, linear behavioral
results, not continuous-range, amplitude-specific distortion, low-BER, or PVT sign-off.

## Pilot and monitoring

The fresh PPO uses 26 observations: the previous 18 plus noise bounds, equivalent
TX swing, both band edges, and three mode indicators. Modes and amplitude-entry
references are balanced by episode sampling. Training uses 0.5–1 V equivalent TX
Vpp, 1–50 mV RMS noise, and four bands: [10, 5000], [10, 2500], [100, 5000], and
[100, 2500] MHz. Estimated episodes include ranges and budgets; budget evaluation
also includes the zero external-noise control. Unknown uses the stated positive
default noise interval. Numeric coverage is an envelope, not a generalization guarantee.

The run is capped at 5120 training steps, four independent simulator workers,
and two hours including holdout checks. It uses nominal TT, 5 Gb/s, and 512-bit
training/evaluation eyes. Guard thresholds and honest improvement rewards are
preserved. Checkpoints are saved after completed optimizer updates; an interrupted
rollout is marked as such.

The separate supervisor polls every 15 seconds. It checks trainer and individual
worker progress, permits training workers to idle during holdout scoring, and
stops an owned process tree after five minutes of inactivity or the hard deadline.
Numerical non-finiteness fails the trainer; crashes are reported as failures.
Negative rewards alone do not stop training. No process-name matching or automatic
restart is used.

Initial policy, frozen PPO, and delivered fixed-circuit baselines share a frozen
12-case monitoring manifest. Cases balance modes and amplitude references, use
unseen band/target/channel/noise combinations and independent patterns, and allow
at most five candidate evaluations each. The fixed circuit is evaluated once per
case. Every complete rollout is checked on the same manifest. Two consecutive
checks stop the run if strict noisy passes fall by at least 2/12 or invalid rate
increases by at least 20 percentage points versus the initial policy. These are
operational thresholds, not statistical significance tests. Best-checkpoint
selection uses strict passes, then lower invalid rate; an exact tie retains the
earlier checkpoint. This monitoring set is not a production qualification set.

## Launch and artifacts

Use the project's `.venv/Scripts/python.exe`, `PYTHONPATH=src`, and fresh absolute
run/control directories. Start the supervisor hidden on Windows:

```text
python scripts/supervise_noise_pilot.py --run-dir <fresh results directory> --control-dir <fresh work directory> --wall-seconds 7200
```

Use `--preflight --wall-seconds 180` for the real four-worker wiring check.
The control directory owns `supervisor.json`, stdout, and stderr. The run owns
source hashes, configuration, worker progress, initial/final/best checkpoints,
`progress.json`, `status.json`, `comparison.json`, `regression.json`, raw guard
artifacts, and full per-case/per-candidate `validation_*.json` records.

Completion, regression stop, deadline stop, and failure are distinct statuses.
A budget-limited checkpoint is not a completed pilot. Deployment requires a
separate fresh qualification set and explicit promotion; this launch never
replaces `results/seq_clean40k.zip` or `results/delivered_circuit.json`.
