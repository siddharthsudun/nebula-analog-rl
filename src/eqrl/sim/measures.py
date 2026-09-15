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


def bandwidth_ghz(freq: np.ndarray, mag_db: np.ndarray, ref: str = "peak") -> float | None:
    """Upper -3 dB frequency, searching above the peak.

    ref="peak" : 3 dB below the maximum. The only definition that holds for an
                 ATTENUATING equalizer — this stage measures -10.6 dB at DC and peaks at
                 -0.35 dB, so DC is its floor, not a plateau, and it never falls 3 dB
                 below DC at any frequency.
    ref="dc"   : 3 dB below the DC level. Valid only when the stage has DC gain.

    Returns None when the sweep never reaches -3 dB. That case is deliberately NOT
    reported as "bandwidth = the last frequency in the sweep": doing so makes every
    design look identical at the sweep edge and hides the fact that the number is
    unmeasurable. Callers must handle None rather than receive a plausible constant.

    Pure function on arrays, so it is testable without a simulator.
    """
    i = int(np.argmax(mag_db))
    level = float(mag_db[i]) if ref == "peak" else float(mag_db[0])
    below = np.where(mag_db[i:] <= level - 3.0)[0]
    if not below.size:
        return None
    return float(freq[i + below[0]] / 1e9)


def peak_is_at_sweep_edge(freq: np.ndarray, mag_db: np.ndarray, *, tol: int = 1) -> bool:
    """True when argmax lands on (or within `tol` points of) the last sample.

    A 'peak' at the edge of the sweep is a stopped sweep, not a resonance — the response
    was still rising when we quit looking. Still relevant: `NgspiceServer.ac()` sweeps
    only to 10 GHz, and a low-Cs design peaks near 9.6 GHz, close enough to the edge that
    argmax can pin to the boundary.
    """
    return int(np.argmax(mag_db)) >= len(mag_db) - 1 - tol


def power(srv: NgspiceServer, dv: DesignVars, vdd: float, temp_c: float) -> float:
    """Real DC power = VDD * measured supply current at the operating point (W).

    Measured rather than assumed (`vdd * dv.i_tail`): with a current-mirror bias the
    delivered current is not the requested current, and the difference is exactly the
    PVT drift the mirror was added to capture.
    """
    return vdd * srv.supply_current(dv, vdd=vdd, temp_c=temp_c)


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


def eye(srv: NgspiceServer, dv: DesignVars, vdd: float, temp_c: float,
        channel_loss_db: float = 12.0) -> tuple[float, float]:
    """Eye height (UI) and vertical opening (mV) through channel + CTLE + 1-tap DFE.

    Uses the SPICE-measured complex CTLE response, a PCIe-Gen2 channel, and a real DFE.
    """
    from eqrl.sim.eye import compute_eye

    r = srv.ac_complex(dv, vdd=vdd, temp_c=temp_c)
    res = compute_eye(r["freq"], r["H"], channel_loss_db=channel_loss_db)
    return res.width_ui, res.height_v * 1e3    # (UI, mV)


def measure_all(dv: DesignVars, *, vdd: float = 1.8, temp_c: float = 27.0,
                corner: str = "tt", fast: bool = False, channel_loss_db: float = 12.0,
                srv: NgspiceServer | None = None, _via_guards: bool = False) -> Measures:
    """Measure one candidate at one PVT corner via the resident server.

    Boost, peak frequency, real supply power, area and the channel+DFE eye are ALWAYS
    simulated (no hardcoded specs). `fast=True` additionally skips only the two slowest
    analyses — the transient-HD3 and the `.noise` sweep — for baseline sweeps that don't
    need them; `fast=False` (the training/characterization default) measures all eight.

    `_via_guards` is accepted and ignored. `guards.seal_direct_access()` replaces this
    function with a wrapper that requires the flag, so that no measurement can reach a
    reward without passing Tiers 1-4; the evaluator therefore always passes it. Accepting
    it here is what lets the guarded path work whether or not the seal is installed —
    without it, every guarded evaluation raised TypeError before reaching the simulator.
    """
    from eqrl.sim.eye import compute_eye

    srv = srv or get_server(corner)
    srv.set_corner(corner)
    # one wideband complex sweep -> boost, peak frequency AND the eye
    try:
        acx = srv.ac_complex(dv, vdd=vdd, temp_c=temp_c)
    except NgspiceError:
        return Measures(ok=False)
    f, H = acx["freq"], acx["H"]
    magdb = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    band = f <= 10e9
    dc = float(magdb[0])
    i = int(np.argmax(magdb[band]))
    boost, fpk = float(magdb[i] - dc), float(f[i] / 1e9)

    m = Measures(dc_gain_db=dc, peak_gain_db=dc + boost, boost_db=boost,
                 peak_freq_ghz=fpk, area_mm2=dv.area_mm2())
    try:
        m.power_w = power(srv, dv, vdd, temp_c)                 # real supply current
    except NgspiceError:
        return Measures(ok=False)
    e = compute_eye(f, H, channel_loss_db=channel_loss_db)       # real channel+CTLE+DFE eye
    m.eye_h_ui, m.eye_v_mv = e.width_ui, e.height_v * 1e3

    if fast:
        m.hd3_db, m.noise_vrms = -40.0, 1.0e-3                   # skip only HD3 + noise sweeps
        return m
    try:
        m.hd3_db = hd3_db(srv, dv, vdd, temp_c)
        m.noise_vrms = input_noise(srv, dv, vdd, temp_c)
    except NgspiceError:
        return Measures(ok=False)
    return m
