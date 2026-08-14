"""A PCIe-Gen2-like backplane channel model.

Insertion loss combines a skin-effect term (~sqrt(f)) and a dielectric term (~f),
normalized to a chosen loss at the Nyquist frequency. The phase is reconstructed as
**minimum phase** from the magnitude (Hilbert transform of the log-magnitude), so the
channel's ISI is post-cursor-dominant — exactly the impairment a 1-tap DFE removes and a
CTLE pre-compensates. This is what gives the equalizer something real to equalize.
"""
from __future__ import annotations

import numpy as np


def insertion_loss_db(freq_hz: np.ndarray, loss_nyquist_db: float = 12.0,
                      nyquist_hz: float = 2.5e9, skin_frac: float = 0.6) -> np.ndarray:
    """Insertion loss (<= 0 dB) vs frequency, normalized to -loss_nyquist_db at Nyquist."""
    f = np.abs(freq_hz) / nyquist_hz
    shape = skin_frac * np.sqrt(f) + (1.0 - skin_frac) * f      # =1 at f=nyquist
    return -loss_nyquist_db * shape


def channel_response(freq_hz: np.ndarray, loss_nyquist_db: float = 12.0,
                     nyquist_hz: float = 2.5e9) -> np.ndarray:
    """Complex, minimum-phase channel transfer function on the given (>=0) freq grid."""
    mag = 10.0 ** (insertion_loss_db(freq_hz, loss_nyquist_db, nyquist_hz) / 20.0)
    return _minimum_phase(freq_hz, mag)


def _minimum_phase(freq_hz: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Minimum-phase reconstruction: phase = -Hilbert{ ln|H| } on a uniform grid.

    freq_hz is assumed uniform starting at 0 (an rfft grid). We mirror to a full
    spectrum, take the analytic signal of ln|H|, and read back the half-spectrum phase.
    """
    eps = 1e-9
    logm_half = np.log(np.maximum(mag, eps))
    # build full even-symmetric log-magnitude spectrum, then min-phase via cepstrum
    n_half = len(logm_half)
    full = np.concatenate([logm_half, logm_half[-2:0:-1]])       # even symmetric
    cep = np.fft.ifft(full).real                                  # real cepstrum
    m = len(full)
    w = np.zeros(m)
    w[0] = 1.0
    w[1:m // 2] = 2.0
    if m % 2 == 0:
        w[m // 2] = 1.0
    min_log = np.fft.fft(cep * w)                                 # ln H_min (analytic)
    H_full = np.exp(min_log)
    return H_full[:n_half]
