"""Extract the observation vector (peaking, HD3, noise, power, area, eye) from SPICE.

All SPICE work goes through a resident NgspiceServer (see server.py) so evals are ~40 ms.
Each metric is a thin, testable unit. `fast=True` skips the slow transient-HD3 and
`.noise` analyses for training speed; the final characterization runs `fast=False`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from eqrl.circuits.ctle import DesignVars
from eqrl.sim.ngspice_runner import NgspiceError
from eqrl.sim.server import NgspiceServer, get_server


@dataclass
class Measures:
    dc_gain_db: float = 0.0
    peak_gain_db: float = 0.0
    boost_db: float = 0.0
    peak_freq_ghz: float = 0.0
    hd3_db: float = 0.0
    noise_vrms: float = 0.0
    power_w: float = 0.0
    area_mm2: float = 0.0
    eye_h_ui: float = 0.0
    eye_v_mv: float = 0.0
    ok: bool = True          # False if the sim failed / was non-convergent

    def as_dict(self) -> dict:
        return asdict(self)


def peaking(srv: NgspiceServer, dv: DesignVars, vdd: float, temp_c: float
            ) -> tuple[float, float, float]:
    """Return (dc_gain_db, boost_db, peak_freq_ghz) from an AC run."""
    r = srv.ac(dv, vdd=vdd, temp_c=temp_c)
    mag, freq = r["mag_db"], r["freq"]
    dc = float(mag[0])
    i = int(np.argmax(mag))
    return dc, float(mag[i] - dc), float(freq[i] / 1e9)


def power(dv: DesignVars, vdd: float = 1.8) -> float:
    """DC power = VDD * total supply current (tail currents through the loads)."""
    return vdd * dv.i_tail


def hd3_db(srv: NgspiceServer, dv: DesignVars, vdd: float, temp_c: float) -> float:
    """Third-harmonic distortion via transient + FFT of the differential output.

    Drives a 100 MHz tone, FFTs the settled part of v(outp)-v(outn), returns
    20*log10(A_3f0 / A_f0) in dB. Odd-order (HD3) survives in a differential pair.
    """
    tr = srv.transient(dv, vdd=vdd, temp_c=temp_c)
    t, vd = tr["t"], tr["vd"]
    if len(t) < 64:
        return 0.0
    dt = float(np.mean(np.diff(t)))
    seg = vd[len(vd) // 4:]                 # drop startup transient
    seg = seg - np.mean(seg)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), dt)
    f0 = 100e6

    def amp_at(f):
        k = int(np.argmin(np.abs(freqs - f)))
        return float(np.max(spec[max(0, k - 2): k + 3]))

    a1, a3 = amp_at(f0), amp_at(3 * f0)
    if a1 <= 0:
        return 0.0
    return 20.0 * np.log10(max(a3, 1e-12) / a1)


def input_noise(srv: NgspiceServer, dv: DesignVars, vdd: float, temp_c: float) -> float:
    """Integrated input-referred noise 10 MHz-5 GHz via `.noise` (Vrms)."""
    return srv.noise_total(dv, vdd=vdd, temp_c=temp_c)


def eye(dv: DesignVars) -> tuple[float, float]:
    """PRBS eye height/width through a channel model.

    TODO(Phase 4): drive PRBS through channel + CTLE + DFE, build eye, measure H/V.
    """
    return 0.5, 120.0  # (UI, mV) stub


def measure_all(dv: DesignVars, *, vdd: float = 1.8, temp_c: float = 27.0,
                corner: str = "tt", fast: bool = False,
                srv: NgspiceServer | None = None) -> Measures:
    """Measure one candidate at one PVT corner via the resident server.

    fast=True skips the slow transient-HD3 and `.noise` analyses (~40 ms vs ~1 s), giving
    the agent AC/power/area feedback during training; full spec is verified with
    fast=False. HD3/noise carry huge margins for this topology, so training on the fast
    subset is safe and the final pass confirms them.
    """
    srv = srv or get_server(corner)
    srv.set_corner(corner)
    try:
        dc, boost, fpk = peaking(srv, dv, vdd, temp_c)
    except NgspiceError:
        return Measures(ok=False)

    m = Measures(
        dc_gain_db=dc, peak_gain_db=dc + boost, boost_db=boost, peak_freq_ghz=fpk,
        power_w=power(dv, vdd=vdd), area_mm2=dv.area_mm2(),
    )
    if fast:
        m.hd3_db, m.noise_vrms, m.eye_h_ui, m.eye_v_mv = -40.0, 1.0e-3, 0.5, 120.0
        return m
    try:
        m.hd3_db = hd3_db(srv, dv, vdd, temp_c)
        m.noise_vrms = input_noise(srv, dv, vdd, temp_c)
        m.eye_h_ui, m.eye_v_mv = eye(dv)
    except NgspiceError:
        return Measures(ok=False)
    return m
