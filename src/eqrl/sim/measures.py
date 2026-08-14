"""Extract the observation vector (peaking, HD3, noise, power, area, eye) from SPICE.

Each function is a thin, testable unit. Phase 0 implements AC/power/area; HD3, noise and
eye are stubbed with clear TODOs so the env shape is stable while they get filled in.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from eqrl.circuits.ctle import DesignVars, netlist
from eqrl.sim import ngspice_runner as ng


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


def peaking(dv: DesignVars, **corner) -> tuple[float, float, float]:
    """Return (dc_gain_db, boost_db, peak_freq_ghz) from an AC run."""
    r = ng.ac(netlist(dv, analysis="none", **corner))
    mag = r["mag_db"]
    freq = r["freq"]
    dc = float(mag[0])
    i = int(np.argmax(mag))
    return dc, float(mag[i] - dc), float(freq[i] / 1e9)


def power(dv: DesignVars, vdd: float = 1.8) -> float:
    """Static power estimate. Phase 0: VDD * tail current (behavioral)."""
    return vdd * dv.i_tail


def hd3_db(dv: DesignVars, **corner) -> float:
    """Third-harmonic distortion via transient + FFT at 100 MHz diff input.

    TODO(Phase 1): run `.tran`, FFT v(outp,outn), ratio of 3rd harmonic to fundamental.
    """
    # placeholder: return a value that trends worse with larger swing / smaller Rs
    return -35.0  # dB (stub — replace with real FFT extraction)


def input_noise(dv: DesignVars, **corner) -> float:
    """Integrated input-referred noise 10 MHz-5 GHz via `.noise`.

    TODO(Phase 1): parse ngspice `.noise` onoise/inoise integrated output.
    """
    return 1.0e-3  # Vrms (stub)


def eye(dv: DesignVars, **corner) -> tuple[float, float]:
    """PRBS eye height/width through a channel model.

    TODO(Phase 4): drive PRBS through channel + CTLE + DFE, build eye, measure H/V.
    """
    return 0.5, 120.0  # (UI, mV) stub


def measure_all(dv: DesignVars, *, vdd: float = 1.8, temp_c: float = 27.0,
                corner: str = "tt") -> Measures:
    """Full measurement of one candidate at one PVT corner."""
    kw = dict(vdd=vdd, temp_c=temp_c, corner=corner)
    try:
        dc, boost, fpk = peaking(dv, **kw)
    except ng.NgspiceError:
        return Measures(ok=False)
    return Measures(
        dc_gain_db=dc,
        peak_gain_db=dc + boost,
        boost_db=boost,
        peak_freq_ghz=fpk,
        hd3_db=hd3_db(dv, **kw),
        noise_vrms=input_noise(dv, **kw),
        power_w=power(dv, vdd=vdd),
        area_mm2=dv.area_mm2(),
        eye_h_ui=eye(dv, **kw)[0],
        eye_v_mv=eye(dv, **kw)[1],
        ok=True,
    )
