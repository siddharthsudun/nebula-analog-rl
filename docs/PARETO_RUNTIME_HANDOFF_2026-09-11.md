# Pareto and runtime handoff

Status: implementation and real validation in progress. This is a concise engineering handoff, not an internal reasoning transcript.

## Accepted user requirements

- Show the primary circuit immediately. Pareto work must not delay that response.
- Stream alternatives into a stable Circuit 1 / Circuit 2 / Circuit 3 bar; keep the scientist's selection stable.
- Explain the optimized quantity and measured tradeoffs for each sizing.
- Fastest: 5-second primary response, nominal checks only, explicit Check PVT button.
- Auto: 15-second primary response with automatic full PVT. Thinking: maximum 40 seconds.
- Manual PVT checks the selected exact sizing, never substitutes the delivered anchor.
- Preserve 1-stage CTLE and 1-tap DFE, physical checks, optional SNR, graphite/cyan UI.

## Current implementation

`pipeline.design(wall_seconds=...)` routes to `realtime.py`. `request_budget.py` supplies cooperative search checks and reserves time for PVT. `runtime.py` supplies a prewarmed process boundary and hard watchdog so native simulator calls cannot overrun indefinitely. The API now calls this runtime. The worker returns the primary result first, then emits background Pareto events. A new command cancels older background work between batches.

`pvt_fast.py` performs speculative full-grid measurements for the input and optional anchor in independent search and verification banks concurrently. Both exact 45-corner grids must pass. On-demand checks set `include_anchor=False`. Worker initialization is outside timed requests. Windows Job Objects own descendants.

`pareto.py` computes the measured nondominated set over power, noise, area and target error. Corpus values only propose. Local sizing perturbations around the valid primary were added after the first real smoke found no useful alternatives from 30 corpus proposals. Those perturbations still require fresh guarded measurements. IDs now hash sizing; Circuit 1 stays the primary reference and published circuits stay available if subsequently dominated, with an explicit flag.

The frontend has streaming tabs, objective labels, measured deltas, per-circuit PVT controls, selected-circuit exports, keyboard navigation and reduced-motion handling. Async schematic/PVT responses are guarded against stale selection/generation.

## Evidence and caveats so far

Earlier verified full repair: 135 measurements, 45-corner exact parity, warm 7.03-8.66 seconds. That is older evidence, not a universal budget guarantee.

First current runtime smoke (`work/pareto_runtime_smoke_v4.log`): startup 48.06 seconds, Fastest primary 0.735 seconds, Auto primary 9.859 seconds with PVT accepted. Corpus-only Pareto found no extra frontier choices on that target; local perturbations are the next real validation. A progress event missing `kind` interrupted the test harness during manual PVT; fixed by defaulting kind to note. Thinking has not yet been verified in that smoke. Do not call the new flow fully tested until the final smoke and tests pass.

A startup deadlock was found while importing NumPy with a background CRT stdin reader already blocked. Starting the reader after all native imports/prewarm resolves the observed stall. Keep that ordering.

There are other running local uvicorn instances on 8000, 8012 and 8013. Do not terminate them blindly. A previous automatic approval review blocked restarting the user's idle 8000 server. Only owned smoke-test runtime processes were terminated during debugging.

## Remaining checks

1. Run current smoke with local Pareto proposals, verify distinct measured choices, three mode timings and exact-design PVT.
2. Add tests for stable IDs/reference, stale async UI responses, objective labels and exports; hard deadline watchdog and API wiring.
3. Validate frontend rendering when a browser is available. Prior CUA had no available browser.
4. Update this file with final verified outcomes and deployment state. No blanket solve-rate or PVT generalization claims.

Backup before this change: `backups/pareto-runtime-20260911_135821/`. Use `.venv/Scripts/python.exe -X utf8`; newer frontend source is UTF-8. Do not roll back unrelated Claude changes.
