"""Noise calibration for the separate noise-conditioned pilot.

External noise is differential, at the CTLE input, flat in the stated band.
Receiver noise is measured directly at the differential output by SPICE.
The pilot collapses both spectra to an equivalent white slicer RMS; it does
not model temporal correlation, jitter, crosstalk or certify low BER.
"""
from __future__ import annotations

import math
import numpy as np

from eqrl.sim.ngspice_runner import NgspiceError


def flat_noise_gain(freq, response, low_hz=1e7, high_hz=5e9) -> float:
    """sqrt(integral(|H|^2 df)/bandwidth), never a peak-gain shortcut."""
    f = np.asarray(freq, dtype=float)
    h = np.asarray(response, dtype=complex)
    if (f.ndim != 1 or f.size < 2 or h.shape != f.shape
            or not np.all(np.isfinite(f)) or not np.all(np.isfinite(h))
            or not np.all(np.diff(f) > 0)
            or not np.isfinite(low_hz) or not np.isfinite(high_hz)
            or low_hz < 0 or high_hz <= low_hz
            or f[0] > low_hz or f[-1] < high_hz):
        raise ValueError("finite ordered response must cover the entire noise band")
    grid = np.concatenate(([low_hz], f[(f > low_hz) & (f < high_hz)], [high_hz]))
    power = np.interp(grid, f, np.abs(h) ** 2)
    integral = np.sum(np.diff(grid) * (power[:-1] + power[1:]) * 0.5)
    return float(np.sqrt(integral / (high_hz - low_hz)))


def combined_output_rms(input_rms: float, noise_gain: float, receiver_rms: float) -> float:
    if any(not math.isfinite(x) or x < 0 for x in (input_rms, noise_gain, receiver_rms)):
        raise ValueError("noise quantities must be finite and nonnegative")
    return math.hypot(input_rms * noise_gain, receiver_rms)


def differential_output_noise(srv, dv, *, vdd=1.8, temp_c=27.0, bandwidth_hz=(1e7, 5e9)) -> float:
    """Direct integrated output noise, 10 MHz to 5 GHz. No fallback estimate.

    Access to the resident backend is restricted to the experiment's own
    process. A failed differential measurement invalidates the candidate.
    """
    low, high = bandwidth_hz
    if not 1e7 <= low < high <= 5e9:
        raise ValueError("unsupported noise bandwidth")
    srv._prime(dv, vdd, temp_c)
    path = srv._dir / "pilot_output_noise.data"
    path.unlink(missing_ok=True)
    command = ("noise v(outp,outn) vinp dec 20 10e6 5e9" if tuple(bandwidth_hz) == (1e7, 5e9)
               else f"noise v(outp,outn) vinp dec 20 {low:g} {high:g}")
    srv._analysis(command)
    srv._ng.exec_command("let pilot_output_rms = onoise_total")
    srv._ng.exec_command(f"wrdata {path} pilot_output_rms")
    value = float(srv._read(path.name)[-1, -1])
    if not math.isfinite(value) or value <= 0:
        raise NgspiceError("pilot differential output noise is not finite and positive")
    return value
