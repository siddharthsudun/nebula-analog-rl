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
import tempfile
from pathlib import Path

import numpy as np

# libngspice lives in Homebrew's lib dir; make sure the loader can find it.
os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")
logging.getLogger("PySpice").setLevel(logging.ERROR)

from eqrl.circuits.ctle import DesignVars, dv_to_params, param_deck  # noqa: E402
from eqrl.sim.ngspice_runner import NgspiceError  # noqa: E402


class NgspiceServer:
    """A resident libngspice instance bound to one process corner."""

    def __init__(self, corner: str = "tt"):
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared, NgSpiceCommandError

        self._cmd_err = NgSpiceCommandError
        self.corner = corner
        self._ng = NgSpiceShared.new_instance()
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
        p = dv_to_params(dv)
        p["vddp"] = vdd
        for k, v in p.items():
            self._ng.exec_command(f"alterparam {k}={v:.6g}")
        self._ng.exec_command(f"set temp={temp_c:g}")
        self._ng.exec_command("reset")

    def _read(self, name: str, ncol: int = 2) -> np.ndarray:
        f = self._dir / name
        if not f.exists() or f.stat().st_size == 0:
            raise NgspiceError("ngspice produced no data (non-convergent design)")
        return np.atleast_2d(np.loadtxt(f))

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
        self._analysis("ac dec 40 1e6 10e9")
        self._ng.exec_command("let vdb = db(v(outp)-v(outn))")
        self._ng.exec_command(f"wrdata {out} vdb")
        arr = self._read("ac.data")
        return {"freq": arr[:, 0], "mag_db": arr[:, 1]}

    def noise_total(self, dv: DesignVars, vdd: float = 1.8, temp_c: float = 27.0) -> float:
        self._prime(dv, vdd, temp_c)
        out = self._dir / "noise.data"
        out.unlink(missing_ok=True)
        # write integrated input-referred noise via a 1-point vector
        self._analysis("noise v(outp,outn) vinp dec 20 10e6 5e9")
        self._ng.exec_command("let n = inoise_total")
        self._ng.exec_command(f"wrdata {out} n")
        arr = self._read("noise.data")
        return float(arr[-1, -1])

    def transient(self, dv: DesignVars, vdd: float = 1.8, temp_c: float = 27.0,
                  tstep: float = 20e-12, tstop: float = 100e-9) -> dict:
        self._prime(dv, vdd, temp_c)
        out = self._dir / "tr.data"
        out.unlink(missing_ok=True)
        self._analysis(f"tran {tstep:g} {tstop:g}")
        self._ng.exec_command("linearize")
        self._ng.exec_command("let vd = v(outp)-v(outn)")
        self._ng.exec_command(f"wrdata {out} vd")
        arr = self._read("tr.data")
        return {"t": arr[:, 0], "vd": arr[:, 1]}

    def close(self) -> None:
        try:
            self._tmp.cleanup()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# module-level singleton so the env reuses one resident simulator per corner
_SERVERS: dict[str, NgspiceServer] = {}


def get_server(corner: str = "tt") -> NgspiceServer:
    if corner not in _SERVERS:
        _SERVERS[corner] = NgspiceServer(corner)
    return _SERVERS[corner]
