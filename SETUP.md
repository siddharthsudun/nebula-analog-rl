# Setup (verified working on macOS, Python 3.14)

These exact steps were run and confirmed on 2026-08-14.

## 1. System tools
```bash
brew install ngspice          # v47 confirmed working
ngspice --version
```

## 2. Python env
The repo was tested with **Python 3.14** using `uv` (venv + pip on 3.14 was flaky, uv worked):
```bash
cd nebula-analog-rl
uv venv --python 3.14 .venv
uv pip install --python .venv -r requirements.txt
```
(If you don't have `uv`: `brew install uv`. A normal `python3 -m venv` + `pip install`
also works on Python ≤3.13.)

## 3. Verify each layer
```bash
# Phase 0 — SPICE loop (should print boost rising with Cs)
PYTHONPATH=src .venv/bin/python -m eqrl.sim.ngspice_runner --selftest

# Env sanity
PYTHONPATH=src .venv/bin/python -c "from gymnasium.utils.env_checker import check_env; from eqrl.envs.equalizer_env import EqualizerEnv; check_env(EqualizerEnv().unwrapped, skip_render_check=True); print('OK')"

# Baseline (runs real ngspice per trial)
PYTHONPATH=src .venv/bin/python -m eqrl.baselines.sweep --method random --budget 80

# Train (PPO through the ngspice loop)
PYTHONPATH=src .venv/bin/python -m eqrl.agents.train --algo ppo --timesteps 20000
```

## Confirmed working versions
ngspice 47 · numpy 2.5.2 · gymnasium 1.3.0 · torch 2.13.0 · stable-baselines3 2.9.0 ·
optuna (for `--method bayesian`).

## LLM wrapper (bonus)
Needs `ANTHROPIC_API_KEY` set for the real Claude parser; without it, a keyword heuristic
fallback runs so nothing breaks:
```bash
PYTHONPATH=src .venv/bin/python -m eqrl.llm.spec_parser "PCIe Gen2 CTLE, ~9 dB boost, under 12 mW"
```

## Note on the current circuit
Phase 0/1 use a **behavioral** source-degenerated diff pair (VCCS transconductors) so the
loop closes fast. The physics is correct (DC gain, zero at 1/RsCs, boost = 1+gm·Rs/2), but
HD3 / noise / eye are stubbed. **Phase 1's main remaining job is swapping in real SKY130
transistor models** — that's where the analog credibility comes from for the judges.
