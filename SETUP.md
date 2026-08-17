# Setup

Two verified paths: **macOS** (below) and **Windows native** (further down). Both run
the full test suite green against real ngspice on the real SKY130 PDK.

⚠️ **Version skew is a real risk.** macOS has ngspice 47; the Windows conda-forge build
is ngspice 41. Numbers measured on one should be spot-checked on the other before they
go in RESULTS.md.

# macOS (verified 2026-08-14, Python 3.14)

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
The behavioral diff pair (VCCS transconductors) from Phase 0/1 is gone. The CTLE is now
`sky130_fd_pr__nfet_01v8` devices with a current-mirror tail, and HD3, input-referred noise
and the eye are simulated rather than stubbed.

One flag matters when reading numbers. `measure_all(fast=True)` skips the transient-HD3 and
`.noise` analyses and substitutes constants (−40 dB, 1.0 mV) — it exists so baseline sweeps
that do not need those two metrics run faster. `fast=False` measures all eight and is the
default. Any number quoted from a `fast=True` run has two placeholder metrics in it.

---

# Windows (native) — verified 2026-08-15

Full suite green: **274 passed, 3 xfailed**, real ngspice + real SKY130. No WSL needed.

Two gotchas cost the most time, both documented in `scripts/win-env.ps1`:
- conda-forge ships **two** binaries. `ngspice.exe` is the **GUI** build and blocks on
  stdin under `-b`, so batch runs appear to hang forever. `ngspice_con.exe` is the
  console build — the shim exposes it under the name the runner invokes.
- Without `SPICE_LIB_DIR`, ngspice cannot find `spinit` and silently skips the XSPICE
  code models. Plain MOSFET/R/C decks still solve, so it is easy to miss.

`sourceforge.net` was blocked from this machine (403 on every mirror), hence conda-forge
for ngspice and a direct GitHub release download for the PDK.

## 1. ngspice (conda-forge, v41)
```powershell
# micromamba is a single ~11 MB exe from GitHub releases
Invoke-WebRequest `
  -Uri "https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-win-64.exe" `
  -OutFile "$env:USERPROFILE\Downloads\micromamba.exe"

$env:MAMBA_ROOT_PREFIX = "$env:USERPROFILE\Downloads\mamba-root"
& "$env:USERPROFILE\Downloads\micromamba.exe" create -y `
  -p "$env:USERPROFILE\eqrl-ngspice" -c conda-forge ngspice

# expose the CONSOLE build under the name the runner calls
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\eqrl-ngspice\shim" | Out-Null
Copy-Item "$env:USERPROFILE\eqrl-ngspice\Library\bin\ngspice_con.exe" `
          "$env:USERPROFILE\eqrl-ngspice\shim\ngspice.exe"
```
This also installs `Library\bin\ngspice.dll`, which is what PySpice's `NgSpiceShared`
needs for the resident server. PySpice itself is not yet installed or tested here.

## 2. SKY130 PDK (~20 MB, not the full multi-GB install)
Only `sky130_fd_pr` (primitive devices) is needed — this is a transistor-level analog
project, so the standard-cell libraries (hundreds of MB each) are not required.

```powershell
.venv\Scripts\python.exe -m pip install zstandard
$tag = "sky130-e0f692f46654d6c7c99fc70a0c94a080dab53571"
foreach ($f in @("common.tar.zst","sky130_fd_pr.tar.zst")) {
  Invoke-WebRequest -Uri "https://github.com/efabless/volare/releases/download/$tag/$f" `
                    -OutFile "$env:USERPROFILE\Downloads\$f"
}
```
Extract both into `%USERPROFILE%\pdk` so that this path exists:
`%USERPROFILE%\pdk\sky130A\libs.tech\ngspice\sky130.lib.spice`
(the tarballs already contain a top-level `sky130A/`, so do not nest it again).

Confirmed present: `tt`, `ss`, `ff`, `sf`, `fs` corner sections.

## 3. Activate and verify
```powershell
. .\scripts\win-env.ps1
.venv\Scripts\python.exe -m pytest tests/ -q
```
