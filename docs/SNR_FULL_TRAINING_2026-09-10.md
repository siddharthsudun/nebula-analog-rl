# Optional SNR and full policy training, 2026-09-10

The full SNR policy run is `results/snr_full_20260910_01`. Its supervisor writes live state to `work/snr_full_20260910_01_control/supervisor.json`. This is a separate experimental policy. The deployed nominal policy remains `results/seq_clean40k.zip`.

## Scope and budget

The request to retrain the entire corpus is implemented as full online PPO training across the existing target/channel/noise distribution. PPO generates new circuit evaluations during training; historical result files are not relabeled or overwritten.

- 40,960 training transitions, four Windows-spawned simulator workers, 40 rollouts of 1,024 transitions.
- An 18-hour supervisor cap includes baseline evaluation, training and final qualification. A stalled trainer or worker is stopped after five minutes. Two consecutive material monitoring regressions stop training early.
- Target boost 5-11 dB; channel loss 8-16 dB; equivalent differential transmitter swing 0.5-1 Vpp.
- Measured, estimated and unknown noise modes; both transmitter Vpp and CTLE-input RMS references; external noise coverage 1-50 mV RMS. Estimated maximum-budget samples also include the zero-noise point.
- Four training bands: 10 MHz or 100 MHz lower edge, 2.5 GHz or 5 GHz upper edge.
- Full guarded circuit measurement, 512-bit noisy eyes, nominal TT. This run does not establish PVT robustness or low-BER compliance.

Initialization copies the frozen nominal network into the larger SNR observation network, setting the eight new input columns to zero and starting a fresh optimizer. A regression test verifies identical initial predictions. Source checkpoint SHA256: `8868965260d7d169f936020588f28f6d2d3f6a66011444a1d96f525d8f508ec9`.

The latest earlier pilot had no completion record, and its monitoring results had 0/12 strict passes. A noisy-eye pass alone is not a strict circuit pass. This run therefore starts from the frozen nominal checkpoint, rather than treating that pilot as qualified.

## Optional feature contract

The top composer contains Advanced, with an unchecked Consider SNR switch. Its fields are hidden and disabled while off. The dashboard omits `noise_request` entirely when off, and the API does not parse, score or gate on SNR when the request is absent. The engineer's manual off choice wins over subsequent parser suggestions. The seed field is removed from the UI; an internal deterministic seed remains for reproducibility.

The LLM wrapper accepts a separate `_noise_request` object, only when the text explicitly asks for SNR or external noise. The keyword reader handles measured input SNR in dB, external RMS noise, signal reference/amplitude, bandwidth and explicit opt-out. External noise is kept separate from the receiver's intrinsic noise ceiling. Missing measured inputs require completion; unsupported output-SNR targets cannot silently execute as measured input SNR.

Example: `9 dB boost, input SNR 20 dB, input signal 200 mV RMS, band 10 MHz to 5 GHz`.

A stated input SNR is converted to external RMS noise using the actual modeled channel response and the specified signal reference and bandwidth. It is not treated as millivolts or as boost.

While the new policy is training, opting in performs a clearly labeled post-search SNR assessment. The production circuit search does not yet use the experimental SNR policy. SNR results can gate the requested noise assessment; they cannot imply that SNR steered candidate selection.

## Artifacts and qualification

`config.json`, `provenance.json`, `initialization.json`, `trainer_heartbeat.json`, worker heartbeats and `progress.json` record reproducible configuration and live state. Checkpoints are saved after each rollout optimizer update. `comparison.json` reports the 12-case paired monitoring set used for checkpoint selection.

After full training reaches its step target, the best checkpoint is evaluated on a separate deterministic 24-case qualification set against the frozen policy and fixed delivered circuit. These cases are excluded from checkpoint selection. `qualification.json` records all three comparisons; no checkpoint is automatically promoted. `status.json` is the terminal completion or failure record. A running supervisor is not proof of completed training.

Validation before launch: 123 Python tests and 7 JavaScript tests passed, including SNR-off API isolation, explicit opt-out, dB conversion, LLM nested fields, policy transfer and training budgets. JavaScript syntax checks passed. Live HTTP checks verified the served Advanced controls, absent seed field, SNR parsing and opt-out. Visual browser verification was unavailable because the browser tool reported no available browser.

Launch entry point: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/start_snr_full_20260910.ps1`. It refuses duplicate runs or existing output directories.
