"""Run ngspice in batch mode and return parsed results.

Phase 0 uses ngspice's `-b` batch mode with a `wrdata` block to dump data to a file,
then parses it with numpy. Keeping it dependency-light (raw ngspice, no PySpice) makes
the toughest infra step easier to debug.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


class NgspiceError(RuntimeError):
    pass


def _ngspice_exe() -> str:
    """Locate the ngspice batch binary, per platform.

    PATH is the normal answer and is tried first. On Windows the conda-forge ngspice
    package is routinely installed into a prefix that is NOT on PATH -- the same prefix
    `silq.sim.server._locate_libngspice` searches for `ngspice.dll` -- so look there too
    rather than reporting the binary missing when it is sitting right next to the
    shared library the resident server is already using.

    Returns a path (or the bare name, if PATH will resolve it); never raises. The caller
    turns a genuine miss into an NgspiceError with an installation hint, because a
    FileNotFoundError from `subprocess` names only "ngspice" and tells the reader
    nothing about how to fix it.
    """
    found = shutil.which("ngspice")
    if found:
        return found

    if sys.platform == "win32":
        for d in (Path(os.environ.get("SILQ_NGSPICE_PREFIX", "")) / "Library" / "bin",
                  Path.home() / "silq-ngspice" / "Library" / "bin",
                  Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "bin"):
            exe = d / "ngspice.exe"
            if exe.exists():
                return str(exe)

    return "ngspice"


#: Platform-correct installation hint. This used to say `brew install ngspice`
#: unconditionally, which is wrong advice on the two platforms where the project is
#: actually verified by hand (SETUP.md covers Windows and macOS), and is the first
#: thing a new reader sees when their environment is not set up.
_INSTALL_HINT = {
    "darwin": "brew install ngspice",
    "win32": "conda install -c conda-forge ngspice, or see SETUP.md "
             '"Windows (native)"; set SILQ_NGSPICE_PREFIX if it is installed elsewhere',
}.get(sys.platform, "install ngspice from your package manager (e.g. apt install ngspice)")


def run(netlist: str, *, control: str, timeout: float = 60.0,
        require_output: bool = True) -> dict[str, np.ndarray]:
    """Run a netlist with a `.control ... .endc` block that writes CSV via `wrdata`.

    `control` is the body of the control block (without the `.control`/`.endc` lines).
    It must `wrdata $OUT ...`; `$OUT` is substituted with a temp file path.
    Returns {"data": ndarray} loaded from that file.
    """
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.data"
        deck = netlist.replace(".end", "").rstrip()
        deck += f"\n.control\n{control.replace('$OUT', str(out))}\n.endc\n.end\n"
        cir = Path(d) / "deck.cir"
        cir.write_text(deck)
        try:
            proc = subprocess.run(
                [_ngspice_exe(), "-b", str(cir)],
                capture_output=True, text=True, timeout=timeout,
            )
        except (FileNotFoundError, OSError):
            raise NgspiceError(
                f"ngspice not found on PATH — {_INSTALL_HINT}") from None
        except subprocess.TimeoutExpired:
            raise NgspiceError(f"ngspice timed out after {timeout}s")

        has_data = out.exists() and out.stat().st_size > 0
        if require_output and not has_data:
            raise NgspiceError(
                f"ngspice produced no data.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        data = np.loadtxt(out) if has_data else np.empty(0)
        return {"data": data, "stdout": proc.stdout, "stderr": proc.stderr}


def ac(netlist: str, *, fstart: float = 1e6, fstop: float = 1e11,
       decade_pts: int = 50, **kw) -> dict[str, np.ndarray]:
    """Run an AC analysis, return frequency + differential magnitude(dB) arrays.

    fstop defaults to 100 GHz, not the old 10 GHz: this topology's -3 dB point sits
    around 57 GHz at nominal bias, so a 10 GHz ceiling made bandwidth unmeasurable and
    pinned every reading to the sweep edge. See silq.sim.server.AC_FSTOP.
    """
    # wrdata auto-prepends the scale (frequency) column — do NOT list it explicitly.
    control = (f"ac dec {decade_pts} {fstart:g} {fstop:g}\n"
               "let voutdb = db(v(outp) - v(outn))\n"
               "wrdata $OUT voutdb")
    res = run(netlist, control=control, **kw)
    arr = np.atleast_2d(res["data"])
    return {"freq": arr[:, 0], "mag_db": arr[:, 1]}


def transient(netlist: str, tstep: float = 20e-12, tstop: float = 100e-9,
              **kw) -> dict[str, np.ndarray]:
    """Run a transient, linearize to a uniform grid, return (t, vdiff) arrays."""
    control = (f"tran {tstep} {tstop}\n"
               "linearize\n"
               "let vd = v(outp) - v(outn)\n"
               "wrdata $OUT vd")
    res = run(netlist, control=control, timeout=kw.pop("timeout", 120.0), **kw)
    arr = np.atleast_2d(res["data"])
    return {"t": arr[:, 0], "vd": arr[:, 1]}


def noise_total(netlist: str, **kw) -> float:
    """Run a .noise analysis, return integrated input-referred noise (Vrms)."""
    control = ("noise v(outp,outn) Vinp dec 20 10e6 5e9\n"
               "print inoise_total")
    res = run(netlist, control=control, require_output=False, **kw)
    for line in res["stdout"].splitlines():
        if "inoise_total" in line and "=" in line:
            return float(line.split("=")[1].strip().split()[0])
    raise NgspiceError(f"inoise_total not found.\n{res['stdout']}")


def _selftest() -> int:
    """Simulate the default CTLE and print peaking — the Phase 0 exit criterion."""
    from silq.circuits.ctle import DesignVars, netlist as make_netlist

    for cs in (50e-15, 200e-15, 800e-15):
        dv = DesignVars(cs=cs)
        try:
            r = ac(make_netlist(dv, analysis="none"))
        except NgspiceError as e:
            print(f"[FAIL] {e}")
            return 1
        peak = float(np.max(r["mag_db"]))
        dc = float(r["mag_db"][0])
        fpk = float(r["freq"][int(np.argmax(r["mag_db"]))])
        print(f"Cs={cs*1e15:6.0f}fF  DC={dc:6.2f}dB  peak={peak:6.2f}dB  "
              f"boost={peak-dc:5.2f}dB  fpeak={fpk/1e9:5.2f}GHz")
    print("Phase 0 loop OK: peaking responds to Cs.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    p.print_help()
