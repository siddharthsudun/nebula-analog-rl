"""Eye-diagram engine: channel + CTLE + 1-tap DFE, measured by Monte-Carlo.

The CTLE is a *linear* equalizer, so its SPICE-measured complex response H_ctle(f) fully
characterizes it in the time domain. We form the end-to-end response
H(f) = H_channel(f) · H_ctle(f), derive the single-bit **pulse response**, then run a
random NRZ pattern through it (superposition, since it's linear), apply a real **1-tap
decision-feedback equalizer** at the sampling instant, and fold the result into an eye.

Outputs: eye height (V, differential) and width (UI) — the poster's eye-opening spec.
Everything here is real DSP on the actual simulated CTLE response; nothing is stubbed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eqrl.sim.channel import channel_response


@dataclass
class EyeResult:
    height_v: float          # vertical opening at the sampling instant (differential V)
    width_ui: float          # horizontal opening (fraction of a UI)
    c0: float                # main cursor (pulse-response peak)
    c1_pre: float            # residual first post-cursor AFTER DFE (should be ~0)
    dfe_tap: float           # adapted 1-tap DFE weight (V)
    samples_per_ui: int
    sample_phase: int        # column index of the optimal sampling instant
    eye_matrix: np.ndarray   # folded UI traces for plotting (n_traces x samples_per_ui)


def _interp_H(freq_grid: np.ndarray, freq_ac: np.ndarray, H_ac: np.ndarray) -> np.ndarray:
    """Interpolate a measured complex response (log-freq) onto a linear rfft grid."""
    lg = np.log10(np.maximum(freq_grid, 1.0))
    lac = np.log10(np.maximum(freq_ac, 1.0))
    mag = np.interp(lg, lac, np.abs(H_ac), left=np.abs(H_ac[0]), right=np.abs(H_ac[-1]))
    # unwrap phase for smooth interpolation
    ph = np.unwrap(np.angle(H_ac))
    phg = np.interp(lg, lac, ph, left=ph[0], right=ph[-1])
    return mag * np.exp(1j * phg)


def compute_eye(freq_ac: np.ndarray, H_ctle: np.ndarray, *,
                bit_rate: float = 5e9, samples_per_ui: int = 16, n_bits: int = 2048,
                swing_v: float = 1.0, channel_loss_db: float = 12.0,
                nyquist_hz: float = 2.5e9, dfe_taps: int = 1,
                seed: int = 0) -> EyeResult:
    """Build the eye for an end-to-end channel+CTLE(+DFE) link.

    freq_ac/H_ctle: complex CTLE response from SPICE (differential, unit input).
    swing_v: differential input NRZ amplitude (peak-to-peak) driving the channel.
    """
    fs = bit_rate * samples_per_ui
    n = n_bits * samples_per_ui
    # end-to-end response on the rfft grid
    fgrid = np.fft.rfftfreq(n, 1.0 / fs)
    H_ch = channel_response(fgrid, channel_loss_db, nyquist_hz)
    H_eq = _interp_H(fgrid, freq_ac, H_ctle)
    H = H_ch * H_eq

    M = samples_per_ui
    amp = swing_v / 2.0                             # NRZ levels are ±amp (pp = swing_v)
    warm = 8                                        # discard startup UIs

    # --- pulse response (one UI held bit through H), folded per UI ---
    pulse = np.zeros(n)
    pulse[:M] = 1.0
    p = np.fft.irfft(np.fft.rfft(pulse) * H, n=n) * amp
    P = p[:n_bits * M].reshape(n_bits, M)
    r0 = int(np.argmax(np.max(np.abs(P), axis=1)))  # UI holding the main cursor

    # --- random NRZ pattern filtered through the end-to-end response ---
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits) * 2 - 1
    nrz = np.repeat(bits, M).astype(float)
    y = np.fft.irfft(np.fft.rfft(nrz) * H, n=n) * amp
    Y = y[:n_bits * M].reshape(n_bits, M)           # row k = waveform during bit k's UI

    # --- optimal sampling phase = column with the largest pre-DFE vertical opening ---
    def opening_at(col_vals, dec):
        o, z = col_vals[dec], col_vals[~dec]
        return (o.min() - z.max()) if (o.size and z.size) else -1e9
    best_ph, best_op = 0, -1e18
    for ph in range(M):
        col = Y[warm:, ph]
        op = opening_at(col, col >= 0)
        if op > best_op:
            best_op, best_ph = op, ph

    # cursors at the sampling phase -> adapt the 1-tap DFE to the first post-cursor
    c0 = float(P[r0, best_ph])
    c1 = float(P[r0 + 1, best_ph]) if r0 + 1 < n_bits else 0.0
    dfe_tap = float(np.clip(c1, -0.6 * abs(c0), 0.6 * abs(c0))) if dfe_taps >= 1 else 0.0

    # --- apply the DFE: subtract tap * previous decision from each UI (sequential) ---
    Yc = Y.copy()
    sign = 1.0 if c0 >= 0 else -1.0                 # account for an inverting CTLE
    prev = 0                                        # previous decided symbol (±1)
    for k in range(n_bits):
        Yc[k, :] -= dfe_tap * prev                  # cancel the post-cursor from bit k-1
        s = (Y[k, best_ph] - dfe_tap * prev) * sign
        prev = 1 if s >= 0 else -1
    # decisions at the sampling phase (sign-normalised so '1' is the positive cloud)
    ref = (Yc[warm:, best_ph] * sign) >= 0
    col = Yc[warm:, best_ph] * sign
    height = float(col[ref].min() - col[~ref].max()) if (ref.any() and (~ref).any()) else 0.0

    # horizontal opening: columns (around the sampling phase) that stay open
    open_cols = 0
    for ph in range(M):
        c = Yc[warm:, ph] * sign
        if c[ref].size and c[~ref].size and c[ref].min() > c[~ref].max():
            open_cols += 1
    width_ui = open_cols / M

    return EyeResult(height_v=max(height, 0.0), width_ui=width_ui, c0=c0,
                     c1_pre=c1 - dfe_tap, dfe_tap=dfe_tap,
                     samples_per_ui=M, sample_phase=best_ph, eye_matrix=Yc[warm:] * sign)
