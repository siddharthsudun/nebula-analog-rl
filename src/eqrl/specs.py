"""Target specifications for the equalizer, as code.

A `Spec` is the input to the whole framework: the RL reward is computed against it,
and the LLM wrapper produces one from natural language.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    """PCIe Gen 2 receiver equalizer target (see docs/PROBLEM.md)."""

    # signaling
    data_rate_gbps: float = 5.0
    nyquist_ghz: float = 2.5

    # peaking
    boost_db_min: float = 3.0
    boost_db_max: float = 12.0
    peak_freq_lo_ghz: float = 1.25
    peak_freq_hi_ghz: float = 2.5

    # constraints (hard limits)
    hd3_db_max: float = -30.0          # linearity: HD3 must be below this
    noise_vrms_max: float = 1.5e-3     # input-referred, 10 MHz - 5 GHz
    power_w_max: float = 15e-3
    area_mm2_max: float = 0.05

    # eye
    eye_h_ui_min: float = 0.4
    eye_v_mv_min: float = 100.0

    # the specific boost we are targeting this run (within [min, max])
    target_boost_db: float = 9.0

    # PVT corners to enforce
    process_corners: tuple[str, ...] = ("tt", "ss", "ff", "sf", "fs")
    vdd_nominal: float = 1.8
    vdd_tolerance: float = 0.05
    temps_c: tuple[float, ...] = (0.0, 27.0, 125.0)

    def vdd_corners(self) -> tuple[float, ...]:
        v = self.vdd_nominal
        return (v * (1 - self.vdd_tolerance), v, v * (1 + self.vdd_tolerance))


# convenience default used across the repo
DEFAULT_SPEC = Spec()
