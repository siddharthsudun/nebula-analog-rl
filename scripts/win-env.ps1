# Activate the Windows ngspice + SKY130 toolchain for this shell.
#
#   . .\scripts\win-env.ps1      (note the leading dot — it must run in your shell)
#
# See SETUP.md "Windows (native)" for how these were installed.

$ErrorActionPreference = "Stop"

$NgPrefix = "$env:USERPROFILE\eqrl-ngspice"
$PdkRoot  = "$env:USERPROFILE\pdk"

if (-not (Test-Path "$NgPrefix\shim\ngspice.exe")) {
    Write-Error "ngspice shim not found at $NgPrefix\shim. See SETUP.md 'Windows (native)'."
}
if (-not (Test-Path "$PdkRoot\sky130A\libs.tech\ngspice\sky130.lib.spice")) {
    Write-Error "SKY130 not found at $PdkRoot. See SETUP.md 'Windows (native)'."
}

# The shim MUST come first: conda-forge's ngspice.exe is the GUI build and blocks on
# stdin under `-b`, so a batch run appears to hang. ngspice_con.exe is the console
# build; the shim exposes it under the name the runner invokes.
$env:PATH = "$NgPrefix\shim;$NgPrefix\Library\bin;$env:PATH"

# Without SPICE_LIB_DIR ngspice cannot find spinit and silently skips the XSPICE code
# models. Plain MOSFET/R/C decks still solve, so the omission is easy to miss.
$env:SPICE_LIB_DIR = "$NgPrefix\Library\share\ngspice"
$env:PDK_ROOT = $PdkRoot

Write-Host "ngspice   : $NgPrefix\shim\ngspice.exe (console build)"
Write-Host "PDK_ROOT  : $env:PDK_ROOT"
Write-Host "SPICE_LIB : $env:SPICE_LIB_DIR"
Write-Host ""
Write-Host "run tests:  .\.venv\Scripts\python.exe -m pytest tests/ -q"
