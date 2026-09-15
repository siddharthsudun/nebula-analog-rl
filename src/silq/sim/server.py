"""Resident ngspice server via libngspice (PySpice NgSpiceShared) — the key to fast RL.

Including the SKY130 corner `.lib` costs ~15 s of model parsing, but a simulation is
~40 ms. libngspice keeps the process resident and headless, so we parse models ONCE and
evaluate each candidate with `alterparam ...; reset; <analysis>`. Result: ~0.04 s/AC eval
instead of ~15 s. Measured: 350x.

One server is bound to one process corner (the corner is baked into the deck's `.lib`).
For PVT, use `for_corner()` per corner or `set_corner()` to reload.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def _locate_libngspice() -> None:
    """Point PySpice at the ngspice shared library, per platform.

    PySpice resolves the library from NGSPICE_LIBRARY_PATH when it is set, and falls
    back to a bundled Windows path / the system loader otherwise. An already-set value
    always wins, so an explicit environment (see scripts/win-env.ps1) is never
    overridden.

    This used to set DYLD_LIBRARY_PATH unconditionally, which is macOS-only: on Windows
    the loader ignores it, PySpice looked for a Spice64_dll directory that does not
    exist, and the resident server could not start at all.
    """
    if os.environ.get("NGSPICE_LIBRARY_PATH"):
        return

    if sys.platform == "darwin":
        # libngspice lives in Homebrew's lib dir; make sure the loader can find it.
        os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")
        return

    if sys.platform == "win32":
        # PySpice formats this with the instance id, so it needs a {} placeholder.
        candidates = [
            Path(os.environ.get("SILQ_NGSPICE_PREFIX", "")) / "Library" / "bin",
            Path.home() / "silq-ngspice" / "Library" / "bin",
            Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "bin",
        ]
        for d in candidates:
            if (d / "ngspice.dll").exists():
                os.environ["NGSPICE_LIBRARY_PATH"] = str(d / "ngspice{}.dll")
                os.environ.setdefault(
                    "SPICE_LIB_DIR", str(d.parent / "share" / "ngspice"))
                # Python 3.8+ ignores PATH for dependent-DLL resolution.
                if hasattr(os, "add_dll_directory") and d.is_dir():
                    os.add_dll_directory(str(d))
                return
        # Leave it unset: PySpice raises a clear load error, and forcing a wrong path
        # would turn that into a confusing one. See SETUP.md "Windows (native)".


_locate_libngspice()
logging.getLogger("PySpice").setLevel(logging.ERROR)

from silq.circuits.ctle import DesignVars, dv_to_params, param_deck  # noqa: E402
from silq.sim.ngspice_runner import NgspiceError  # noqa: E402


#: AC sweep for the gain/peaking measurement.
#:
#: This used to stop at 10 GHz, which is BELOW this topology's -3 dB point (~57 GHz at
#: nominal bias for the reference design). Every bandwidth reading therefore saturated
#: at the sweep edge and looked constant no matter what the design did. 100 GHz clears
#: the -3 dB point with margin while staying inside the range where the SKY130 BSIM
#: models are meaningful.
#:
#: NOTE: widening the sweep can change `peak_freq_ghz` for any design whose old argmax
#: landed on the 10 GHz boundary — those were sweep artifacts, not peaks. Results
#: measured before this change should be regenerated.
AC_FSTART = 1e6
AC_FSTOP = 1e11
AC_DECADE_PTS = 40


#: ngspice lines that are informational, not failures. PySpice whitelists only lines
#: beginning with "Warning:" and flags every other stderr line as a command failure
#: (Shared.py, send_stderr). ngspice announces its DC fallback as
#:     Note: Transient op started
#:     Note: Transient op finished successfully
#: on stderr, so any design whose operating point needs that fallback raises
#: NgSpiceCommandError even though the .op solved and returned correct node voltages.
#:
#: MEASURED: with the default deck the ff corner takes this path while tt and ss do not,
#: which made `set_corner("ff")` raise while the same deck run through the subprocess
#: path returned v(sp)=0.369 V, v(outp)=1.027 V, v(nbias)=0.828 V — a perfectly good bias.
#: Left in place this reports healthy fast-corner designs as invalid, which during
#: training is a systematic bias against exactly the corner that converges hardest.
#:
#: Demoting these is safe: the guard layer's Tier 1.2 independently scans the captured
#: stdout/stderr for real solver-failure text, so a genuine failure is still caught even
#: when PySpice does not raise.
_BENIGN_STDERR_PREFIXES = ("Note:",)


def _stderr_line_is_failure(line: str) -> bool:
    """Does this captured stderr line indicate a failed command?"""
    s = line.strip()
    return not (s.startswith(_BENIGN_STDERR_PREFIXES) or s.startswith("Warning:") or not s)


def _tolerant_shared_instance():
    """A PySpice NgSpiceShared that does not treat informational notes as errors.

    PySpice sets `_error_in_stderr` inline in a static callback before handing the line
    to the documented `send_char` hook, so the flag cannot be intercepted — it can only
    be recomputed. `clear_output()` empties `_stderr` at the start of every command, so
    recomputing across the whole captured list is exact: benign notes never raise the
    flag on their own, and a genuine error line anywhere in the same command still does.
    """
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared

    class _Tolerant(NgSpiceShared):
        def send_char(self, message, ngspice_id):
            prefix, _, content = message.partition(" ")
            if prefix == "stderr" and content.strip().startswith(_BENIGN_STDERR_PREFIXES):
                self._error_in_stderr = any(_stderr_line_is_failure(l)
                                            for l in self._stderr)
            return super().send_char(message, ngspice_id)

    ng = _Tolerant.new_instance()
    if not isinstance(ng, _Tolerant):
        # PySpice caches instances by id in a dict on the shared base class, so whoever
        # calls new_instance() first decides the class for the whole process — importing
        # a module that probes availability with NgSpiceShared.new_instance() was enough
        # to silently give everything the intolerant base back. Re-bind instead of
        # accepting that: _Tolerant overrides one method and adds no state.
        ng.__class__ = _Tolerant
    return ng


class NgspiceServer:
    """A resident libngspice instance bound to one process corner."""

    def __init__(self, corner: str = "tt"):
        from PySpice.Spice.NgSpice.Shared import NgSpiceCommandError

        self._cmd_err = NgSpiceCommandError
        self.corner = corner
        self._ng = _tolerant_shared_instance()
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)
        self._load(corner)

    def _load(self, corner: str) -> None:
        deck = self._dir / f"deck_{corner}.cir"
        deck.write_text(param_deck(corner))
        self._ng.exec_command(f"source {deck}")
        self._ng.exec_command("reset")
        self._ng.exec_command("op")   # warm the operating point / model init

    def set_corner(self, corner: str) -> None:
        if corner != self.corner:
            self.corner = corner
            self._load(corner)

    def _prime(self, dv: DesignVars, vdd: float, temp_c: float) -> None:
        # ngspice keeps every analysis result as a separate in-memory "plot" (ac1, ac2,
        # tran1, ...) and never frees one on its own -- a documented shared-library
        # behavior (ngspice-users: "ngspice shared library repeated simulations (memory
        # leaks?)"). Over a long resident process (a benchmark loop, or a dashboard
        # session that never restarts) that list grows without bound and every
        # subsequent command gets slower, independent of which mode or how much real
        # work it does -- measured locally as a ~60-75x per-eval slowdown by the ~30th
        # call, identical in shape across every pipeline mode. The prior call's data is
        # long since pulled out via wrdata + _read() by the time we get here, so nothing
        # is lost by clearing it before priming the next one.
        self._ng.exec_command("destroy all")
        p = dv_to_params(dv)
        p["vddp"] = vdd
        p["tempc"] = temp_c
        for k, v in p.items():
            self._ng.exec_command(f"alterparam {k}={v:.6g}")
        self._ng.exec_command("reset")

    def _read(self, name: str, ncol: int = 2) -> np.ndarray:
        """Load a `wrdata` file, treating unparseable or non-finite output as no data.

        A design that fails to solve does not always produce an empty file. ngspice can
        write the solver's non-finite state out verbatim, and on Windows that is spelled
        `-nan(ind)` / `1.#INF`, which numpy cannot parse — `np.loadtxt` then raises
        ValueError from inside the parser rather than returning anything. Left alone that
        crashes any caller that does not already expect a parse error, which is how a
        single degenerate candidate could end a whole benchmark sweep.

        Both cases mean the same thing — there is no trustworthy measurement here — so
        both raise NgspiceError, which callers already handle as a failed evaluation.
        """
        f = self._dir / name
        if not f.exists() or f.stat().st_size == 0:
            raise NgspiceError("ngspice produced no data (non-convergent design)")
        try:
            data = np.loadtxt(f)
        except ValueError as e:
            raise NgspiceError(
                f"ngspice wrote unparseable data to {name} ({e}); the analysis did not "
                "produce a usable solution"
            ) from e
        data = np.atleast_2d(data)
        if not np.all(np.isfinite(data)):
            raise NgspiceError(
                f"ngspice wrote non-finite values to {name}; the analysis did not "
                "converge to a usable solution"
            )
        return data

    def _analysis(self, cmd: str) -> None:
        """Run an analysis command, translating a non-convergence into NgspiceError."""
        try:
            self._ng.exec_command(cmd)
        except self._cmd_err as e:
            raise NgspiceError(f"non-convergent: {cmd}") from e

    # -- analyses ----------------------------------------------------------
    def ac(self, dv: DesignVars, vdd: float = 1.8, temp_c: float = 27.0) -> dict:
        self._prime(dv, vdd, temp_c)
        out = self._dir / "ac.data"
        out.unlink(missing_ok=True)
        self._analysis(f"ac dec {AC_DECADE_PTS} {AC_FSTART:g} {AC_FSTOP:g}")
        self._ng.exec_command("let vdb = db(v(outp)-v(outn))")
        self._ng.exec_command(f"wrdata {out} vdb")
        arr = self._read("ac.data")
        return {"freq": arr[:, 0], "mag_db": arr[:, 1]}

    def ac_complex(self, dv: DesignVars, vdd: float = 1.8, temp_c: float = 27.0,
                   fstop: float = 24e9) -> dict:
        """Complex differential transfer function H(f) = v(outp)-v(outn) for AC=1 input.

        Wider band than ac() because the eye needs the response out to several harmonics.
        wrdata writes a complex vector as [scale, real, imag].
        """
        self._prime(dv, vdd, temp_c)
        out = self._dir / "acx.data"
        out.unlink(missing_ok=True)
        self._analysis(f"ac dec 40 1e6 {fstop:g}")
        self._ng.exec_command("let vd = v(outp)-v(outn)")
        self._ng.exec_command(f"wrdata {out} vd")
        arr = self._read("acx.data")
        return {"freq": arr[:, 0], "H": arr[:, 1] + 1j * arr[:, 2]}

    def supply_current(self, dv: DesignVars, vdd: float = 1.8, temp_c: float = 27.0) -> float:
        """Actual total current drawn from VDD at the operating point (A)."""
        self._prime(dv, vdd, temp_c)
        out = self._dir / "isup.data"
        out.unlink(missing_ok=True)
        self._analysis("op")
        self._ng.exec_command("let isup = abs(i(vdd))")
        self._ng.exec_command(f"wrdata {out} isup")
        arr = self._read("isup.data")
        return float(abs(arr[-1, -1]))

    #: How many times noise_total fell back to the single-ended measurement. Counted, not
    #: hidden: the fallback is an approximation and a caller reporting noise figures should
    #: be able to say how many of them used it.
    noise_fallbacks: int = 0

    def _noise_once(self, output: str) -> float:
        """One `.noise` run, returning ngspice's integrated input-referred noise."""
        out = self._dir / "noise.data"
        out.unlink(missing_ok=True)
        self._analysis(f"noise {output} vinp dec 20 10e6 5e9")
        self._ng.exec_command("let n = inoise_total")
        self._ng.exec_command(f"wrdata {out} n")
        return float(self._read("noise.data")[-1, -1])

    def noise_total(self, dv: DesignVars, vdd: float = 1.8, temp_c: float = 27.0) -> float:
        """Integrated input-referred noise, 10 MHz-5 GHz, in Vrms.

        The differential form `noise v(outp,outn)` is the correct measurement and is tried
        first. It returns `-nan(ind)` at some corners: measured over the full 45-corner grid
        for the design in results/final_report_design.json, it solved at 42 and produced NaN
        at 3 (tt/1.89V/27C, sf/1.89V/27C, ss/1.80V/125C). AC and supply-current analyses run
        fine at those same corners, so the design is biased correctly and it is the noise
        analysis alone that diverges. Adding an `op` before it changes nothing.

        Losing those corners was not free: it cost three PVT passes on a design that is
        otherwise healthy, including one at the nominal corner, which is the most misleading
        possible place to record a failure.

        So on a non-finite result it falls back to the single-ended output and converts. For
        a balanced differential pair the two half-circuits contribute uncorrelated noise, so
        differential output noise power is 2x single-ended while differential gain is 2x,
        which puts the input-referred differential figure at 1/sqrt(2) of the single-ended
        one. That relation was validated against the 42 corners where the exact form works:

            mean ratio 1.0184, range 0.9883 - 1.0732

        i.e. the approximation is within ~2% typically and 7.3% worst case, the outliers all
        at 125 C where perfect balance is least true. It is an approximation, it is only
        used where the exact measurement is unavailable, and every use is counted in
        `noise_fallbacks`.
        """
        self._prime(dv, vdd, temp_c)
        try:
            value = self._noise_once("v(outp,outn)")
            if np.isfinite(value):
                return value
        except NgspiceError:
            pass                      # _read rejects non-finite output; fall through
        self._prime(dv, vdd, temp_c)  # the failed analysis leaves the plot dirty
        single = self._noise_once("v(outp)")
        type(self).noise_fallbacks += 1
        return single / np.sqrt(2.0)

    def transient(self, dv: DesignVars, vdd: float = 1.8, temp_c: float = 27.0,
                  tstep: float = 20e-12, tstop: float = 60e-9) -> dict:
        self._prime(dv, vdd, temp_c)
        out = self._dir / "tr.data"
        out.unlink(missing_ok=True)
        self._analysis(f"tran {tstep:g} {tstop:g}")
        self._ng.exec_command("linearize")
        self._ng.exec_command("let vd = v(outp)-v(outn)")
        self._ng.exec_command(f"wrdata {out} vd")
        arr = self._read("tr.data")
        return {"t": arr[:, 0], "vd": arr[:, 1]}

    def destroy_all_plots(self) -> None:
        """Manually clear ngspice's accumulated result-plot history.

        `_prime` already does this before every eval, so a healthy server never needs
        it. Exposed for the dashboard's "Refresh" control: a server that has been
        resident since before this fix shipped (or one some other code path primed
        without going through `_prime`) can still be slow, and restarting the whole
        process is heavier than clearing the one thing that's actually accumulating.
        """
        self._ng.exec_command("destroy all")

    def close(self) -> None:
        try:
            self._tmp.cleanup()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# libngspice is effectively a process singleton — multiple NgSpiceShared instances share
# state and corrupt each other. So we keep ONE resident server and reload the deck when a
# different PROCESS corner is needed (voltage/temperature don't reload — they're params).
_SERVER: NgspiceServer | None = None


def get_server(corner: str = "tt") -> NgspiceServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = NgspiceServer(corner)
    else:
        _SERVER.set_corner(corner)   # reloads only if the process corner changed
    return _SERVER
