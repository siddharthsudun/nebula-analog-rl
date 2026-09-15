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
import operator

import numpy as np

from silq.sim.channel import channel_response


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


def _validate_inputs(freq_ac: np.ndarray, H_ctle: np.ndarray, *, bit_rate: float,
                     samples_per_ui: int, n_bits: int, swing_v: float,
                     channel_loss_db: float, nyquist_hz: float,
                     dfe_taps: int | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate the small set of assumptions made by the FFT eye model.

    In particular, silently accepting a descending or mismatched AC sweep can make
    ``np.interp`` produce a plausible-looking but physically unrelated eye.
    """
    freq = np.asarray(freq_ac)
    response = np.asarray(H_ctle)
    if freq.ndim != 1 or response.ndim != 1 or freq.size != response.size or not freq.size:
        raise ValueError("freq_ac and H_ctle must be non-empty, one-dimensional arrays of equal length")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(response)):
        raise ValueError("freq_ac and H_ctle must contain only finite values")
    if np.any(freq < 0.0) or np.any(np.diff(freq) < 0.0):
        raise ValueError("freq_ac must be a non-decreasing, non-negative frequency grid")
    if not np.isfinite(bit_rate) or bit_rate <= 0.0:
        raise ValueError("bit_rate must be positive and finite")
    if not np.isfinite(nyquist_hz) or nyquist_hz <= 0.0:
        raise ValueError("nyquist_hz must be positive and finite")
    if not np.isfinite(swing_v) or swing_v <= 0.0:
        raise ValueError("swing_v must be positive and finite")
    if not np.isfinite(channel_loss_db):
        raise ValueError("channel_loss_db must be finite")
    try:
        M = operator.index(samples_per_ui)
        N = operator.index(n_bits)
    except TypeError as exc:
        raise ValueError("samples_per_ui and n_bits must be integers") from exc
    if M < 1 or N < 16:
        raise ValueError("samples_per_ui must be >= 1 and n_bits must be >= 16")
    if dfe_taps is not None:
        try:
            taps = operator.index(dfe_taps)
        except TypeError as exc:
            raise ValueError("dfe_taps must be an integer") from exc
        if taps < 0:
            raise ValueError("dfe_taps must be non-negative")
    return freq.astype(float, copy=False), response, M, N


def _truth_opening(values: np.ndarray, truth: np.ndarray) -> float:
    """Return the signed opening for a column and known +/-1 transmitted symbols."""
    positive = values[truth > 0]
    negative = values[truth < 0]
    if not positive.size or not negative.size:
        return -np.inf
    return float(positive.min() - negative.max())


def _contiguous_open_width(openings: np.ndarray, phase: int) -> int:
    """Length of the circular positive-opening run containing ``phase``."""
    open_mask = np.asarray(openings) > 0.0
    M = open_mask.size
    if M == 0 or not open_mask[phase]:
        return 0
    # Walk in both directions, limiting the second walk by the samples already
    # visited.  The limit is needed for an all-open eye, where modulo indexing
    # would otherwise count the same columns indefinitely.
    left = 0
    index = phase
    while left < M - 1 and open_mask[(index - 1) % M]:
        left += 1
        index = (index - 1) % M
    right = 0
    index = phase
    while right < M - 1 - left and open_mask[(index + 1) % M]:
        right += 1
        index = (index + 1) % M
    return left + 1 + right


def pulse_cursors(freq_ac: np.ndarray, H_ctle: np.ndarray, *,
                  bit_rate: float = 5e9, samples_per_ui: int = 16, n_bits: int = 2048,
                  swing_v: float = 1.0, channel_loss_db: float = 12.0,
                  nyquist_hz: float = 2.5e9) -> tuple[float, float]:
    """Main cursor c0 and first post-cursor c1 (volts) of the end-to-end channel+CTLE
    pulse response, sampled at the optimal phase. This is the ISI the 1-tap DFE must
    cancel; it feeds the behavioural DFE testbench in circuits/dfe.py. Additive helper --
    it reuses the same pulse-response math as compute_eye without changing it.
    """
    freq_ac, H_ctle, M, n_bits = _validate_inputs(
        freq_ac, H_ctle, bit_rate=bit_rate, samples_per_ui=samples_per_ui,
        n_bits=n_bits, swing_v=swing_v, channel_loss_db=channel_loss_db,
        nyquist_hz=nyquist_hz)
    fs = bit_rate * samples_per_ui
    n = n_bits * samples_per_ui
    fgrid = np.fft.rfftfreq(n, 1.0 / fs)
    H = channel_response(fgrid, channel_loss_db, nyquist_hz) * _interp_H(fgrid, freq_ac, H_ctle)
    M = samples_per_ui
    amp = swing_v / 2.0
    pulse = np.zeros(n)
    pulse[:M] = 1.0
    p = np.fft.irfft(np.fft.rfft(pulse) * H, n=n) * amp
    P = p[:n_bits * M].reshape(n_bits, M)
    r0 = int(np.argmax(np.max(np.abs(P), axis=1)))
    ph = int(np.argmax(np.abs(P[r0])))
    c0 = float(P[r0, ph])
    c1 = float(P[r0 + 1, ph]) if r0 + 1 < n_bits else 0.0
    return c0, c1


def compute_eye(freq_ac: np.ndarray, H_ctle: np.ndarray, *,
                bit_rate: float = 5e9, samples_per_ui: int = 16, n_bits: int = 2048,
                swing_v: float = 1.0, channel_loss_db: float = 12.0,
                nyquist_hz: float = 2.5e9, dfe_taps: int = 1,
                seed: int = 0) -> EyeResult:
    """Build the eye for an end-to-end channel+CTLE(+DFE) link.

    freq_ac/H_ctle: complex CTLE response from SPICE (differential, unit input).
    swing_v: differential input NRZ amplitude (peak-to-peak) driving the channel.

    The random pattern is retained as a known transmitted sequence.  ``r0`` is the
    UI delay of the main pulse cursor, so row ``k`` is scored against ``bits[k-r0]``;
    output sign is used only for polarity normalization.  This prevents a badly
    closed eye from defining its own positive and negative clouds.  The DFE still
    uses its actual sequential prior decisions, so it does not get an oracle bit
    stream during correction.
    """
    freq_ac, H_ctle, M, n_bits = _validate_inputs(
        freq_ac, H_ctle, bit_rate=bit_rate, samples_per_ui=samples_per_ui,
        n_bits=n_bits, swing_v=swing_v, channel_loss_db=channel_loss_db,
        nyquist_hz=nyquist_hz, dfe_taps=dfe_taps)
    fs = bit_rate * samples_per_ui
    n = n_bits * samples_per_ui
    # end-to-end response on the rfft grid
    fgrid = np.fft.rfftfreq(n, 1.0 / fs)
    H_ch = channel_response(fgrid, channel_loss_db, nyquist_hz)
    H_eq = _interp_H(fgrid, freq_ac, H_ctle)
    H = H_ch * H_eq

    amp = swing_v / 2.0                             # NRZ levels are ±amp (pp = swing_v)
    warm = 8                                        # discard startup UIs

    # --- pulse response (one UI held bit through H), folded per UI ---
    pulse = np.zeros(n)
    pulse[:M] = 1.0
    p = np.fft.irfft(np.fft.rfft(pulse) * H, n=n) * amp
    P = p[:n_bits * M].reshape(n_bits, M)
    r0 = int(np.argmax(np.max(np.abs(P), axis=1)))  # UI holding the main cursor
    pulse_phase = int(np.argmax(np.abs(P[r0])))
    pulse_peak = float(P[r0, pulse_phase])
    if not np.isfinite(pulse_peak) or abs(pulse_peak) <= np.finfo(float).eps:
        raise ValueError("the end-to-end response has no non-zero main pulse cursor")
    polarity = 1.0 if pulse_peak >= 0.0 else -1.0

    # --- random NRZ pattern filtered through the end-to-end response ---
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits) * 2 - 1
    nrz = np.repeat(bits, M).astype(float)
    y = np.fft.irfft(np.fft.rfft(nrz) * H, n=n) * amp
    Y = y[:n_bits * M].reshape(n_bits, M)           # row k = waveform during bit k's UI

    # Discard startup rows, including any rows before a delayed main cursor.  The
    # latter matters for a channel with an integer UI delay: the output row k then
    # carries the transmitted symbol bits[k-r0], not bits[k+r0].
    warm = max(warm, r0)
    rows = np.arange(warm, n_bits)
    truth = bits[rows - r0]

    # --- optimal sampling phase = largest truth-labelled pre-DFE opening ---
    openings_pre = np.empty(M, dtype=float)
    best_ph, best_op = pulse_phase, -np.inf
    for ph in range(M):
        op = _truth_opening(Y[rows, ph] * polarity, truth)
        openings_pre[ph] = op
        # Equal-valued columns are common for an ideal rectangular response.  Keep
        # the pulse peak phase in that case, making integer delays deterministic.
        if op > best_op + 1e-12 * max(1.0, abs(op), abs(best_op)):
            best_op, best_ph = op, ph

    # cursors at the sampling phase -> adapt the 1-tap DFE to the first post-cursor
    c0 = float(P[r0, best_ph])
    c1 = float(P[r0 + 1, best_ph]) if r0 + 1 < n_bits else 0.0
    dfe_tap = float(np.clip(c1, -0.6 * abs(c0), 0.6 * abs(c0))) if dfe_taps >= 1 else 0.0

    # --- apply the DFE: subtract tap * previous decision from each UI (sequential) ---
    Yc = Y.copy()
    prev = 0                                        # previous decided symbol (±1)
    for k in range(n_bits):
        Yc[k, :] -= dfe_tap * prev                  # cancel the post-cursor from bit k-1
        s = (Y[k, best_ph] - dfe_tap * prev) * polarity
        prev = 1 if s >= 0 else -1
    # Score against the transmitted symbols.  Keep the signed value internally so a
    # closed eye remains negative evidence; clamp only the public height output.
    openings_post = np.array([
        _truth_opening(Yc[rows, ph] * polarity, truth) for ph in range(M)
    ])
    signed_height = float(openings_post[best_ph])
    height = max(signed_height, 0.0)

    # Horizontal opening is the contiguous eye region containing the selected phase;
    # isolated open columns elsewhere are not one physical eye opening.
    width_ui = _contiguous_open_width(openings_post, best_ph) / M

    return EyeResult(height_v=max(height, 0.0), width_ui=width_ui, c0=c0,
                     c1_pre=c1 - dfe_tap, dfe_tap=dfe_tap,
                     samples_per_ui=M, sample_phase=best_ph, eye_matrix=Yc[warm:] * polarity)


# ---------------------------------------------------------------------------
# Audited v2 measurement
#
# ``compute_eye`` above is deliberately frozen: published artifacts need its exact
# historical behaviour.  The v2 path is separate because it changes the experiment
# contract, not merely an implementation detail.


@dataclass
class EyeResultV2:
    """Output of :func:`compute_eye_v2`.

    ``eye_matrix`` is polarity-normalized, DFE-corrected voltage in V and has shape
    ``(count, samples_per_ui)``.  ``margins_v`` is the signed decision margin for the
    same scored rows; it is retained for a later statistical-eye or bathtub calculation.
    ``signed_opening_v`` deliberately is not clipped.  ``height_v`` is its display-safe,
    non-negative counterpart.
    """

    height_v: float
    width_ui: float
    signed_opening_v: float
    c0: float
    c1_pre: float
    dfe_tap: float
    samples_per_ui: int
    sample_phase: int
    eye_matrix: np.ndarray
    errors: int
    count: int
    ber: float
    margins_v: np.ndarray


# The measured complex sweep terminates at a finite frequency and therefore cannot
# identify an unbounded time-domain response.  v2 makes its finite causal FIR window
# explicit rather than hiding a periodic waveform in an FFT buffer.  Thirty-two UIs is
# conservative for the PCIe-like channel in this repository, but this remains a
# behavioural approximation and must be revisited for a link with a longer impulse tail.
_V2_FIR_SPAN_UI = 32


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def _v2_causal_fir(freq_ac: np.ndarray, H_ctle: np.ndarray, *, bit_rate: float,
                   samples_per_ui: int, n_bits: int, swing_v: float,
                   channel_loss_db: float, nyquist_hz: float) -> np.ndarray:
    """Return the finite causal FIR used only by the audited v2 experiment.

    The IFFT constructs a sampled impulse response from the same end-to-end response as
    the legacy scorer.  We retain its causal first ``_V2_FIR_SPAN_UI`` UIs and subsequently
    use *linear* FFT convolution.  That is intentionally different from filtering a
    length-``n_bits`` NRZ vector with one equal-size FFT, which makes its final symbols
    wrap into the beginning of the pattern.
    """
    M = samples_per_ui
    fs = bit_rate * M
    # Keep the response sampling resolution aligned with the legacy measurement; only
    # the later waveform filtering changes to zero-padded *linear* convolution.
    n_fft = _next_power_of_two(max(n_bits * M, 2 * _V2_FIR_SPAN_UI * M))
    fgrid = np.fft.rfftfreq(n_fft, 1.0 / fs)
    H = channel_response(fgrid, channel_loss_db, nyquist_hz) * _interp_H(
        fgrid, freq_ac, H_ctle)
    impulse = np.fft.irfft(H, n=n_fft) * (swing_v / 2.0)
    n_keep = min(_V2_FIR_SPAN_UI * M, impulse.size)
    fir = impulse[:n_keep]
    if not np.all(np.isfinite(fir)) or not np.any(np.abs(fir) > np.finfo(float).eps):
        raise ValueError("the end-to-end response has no non-zero causal impulse")
    return fir


def _linear_filter_v2(x: np.ndarray, fir: np.ndarray) -> np.ndarray:
    """Finite *linear* convolution, accelerated by a zero-padded FFT."""
    n_out = x.size + fir.size - 1
    n_fft = _next_power_of_two(n_out)
    return np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(fir, n_fft), n_fft)[:n_out]


def _pulse_rows_v2(fir: np.ndarray, samples_per_ui: int) -> np.ndarray:
    """Fold a one-UI unit pulse into rows, padding its last partial row with zero."""
    pulse = _linear_filter_v2(np.ones(samples_per_ui), fir)
    padded = np.pad(pulse, (0, (-pulse.size) % samples_per_ui))
    return padded.reshape(-1, samples_per_ui)


def _waveform_rows_v2(bits: np.ndarray, fir: np.ndarray, samples_per_ui: int) -> np.ndarray:
    """Return one waveform row per transmitted UI without circular end wraparound."""
    nrz = np.repeat(bits, samples_per_ui).astype(float)
    y = _linear_filter_v2(nrz, fir)
    return y[:bits.size * samples_per_ui].reshape(bits.size, samples_per_ui)


def _decision_feedback_v2(Y: np.ndarray, *, phase: int, polarity: float, tap: float,
                          noise_sigma_v: float, rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Apply one-tap decision feedback using receiver decisions only.

    ``decisions[k]`` is derived from the noisy slicer value at row ``k``.  It never reads
    the transmitted bit, including while a prior wrong decision is fed back into later
    rows.  Noise is additive white slicer noise in V; a caller that wants spectral noise
    must supply a different waveform model rather than relabel this one.
    """
    noisy = Y if noise_sigma_v == 0.0 else Y + rng.normal(0.0, noise_sigma_v, size=Y.shape)
    corrected = noisy.copy()
    decisions = np.empty(Y.shape[0], dtype=np.int8)
    previous = 0
    for k in range(Y.shape[0]):
        corrected[k, :] -= tap * previous
        decisions[k] = 1 if corrected[k, phase] * polarity >= 0.0 else -1
        previous = int(decisions[k])
    return corrected, decisions


def _adapt_dfe_v2(Y: np.ndarray, *, phase: int, polarity: float, c0: float,
                  start_row: int, dfe_taps: int) -> float:
    """Decision-directed LMS adaptation on a training pattern separate from evaluation."""
    if dfe_taps == 0:
        return 0.0
    # The target level is a pulse-response calibration, not a transmitted training bit.
    # The update is decision-directed: both the error's sign and feedback symbol arise
    # from the receiver's own slicer history.
    target_level = abs(c0)
    limit = 0.6 * target_level
    tap = 0.0
    previous = 0
    step = 0.02
    for k in range(Y.shape[0]):
        value = Y[k, phase] - tap * previous
        decision = 1 if value * polarity >= 0.0 else -1
        if k >= start_row:
            error = value * polarity - target_level * decision
            tap = float(np.clip(tap + step * error * polarity * previous, -limit, limit))
        previous = decision
    return tap


def compute_eye_v2(freq_ac: np.ndarray, H_ctle: np.ndarray, *,
                   bit_rate: float = 5e9, samples_per_ui: int = 16,
                   n_bits: int = 2048, swing_v: float = 1.0,
                   channel_loss_db: float = 12.0, nyquist_hz: float = 2.5e9,
                   dfe_taps: int = 1, seed: int = 0,
                   noise_sigma_v: float = 0.0) -> EyeResultV2:
    """Measure an ideal linear behavioural eye with an independently scored BER.

    This is not a silicon result and it is not a low-BER certification.  The model uses
    a SPICE-measured CTLE transfer response, a deterministic channel model, an explicitly
    finite causal FIR, and optional white additive slicer noise.  It contains neither a
    measured noise PSD nor jitter, package, clock-recovery, or rare-event modelling.

    Three independent ``numpy.random.SeedSequence`` child streams are derived from
    ``seed``: one chooses the sampling phase, one adapts the DFE, and one evaluates the
    reported eye/errors.  Phase selection and DFE adaptation therefore cannot tune on the
    evaluation pattern.  With ``noise_sigma_v=0`` (the default) this is the noiseless
    behavioural measurement; a positive value is for bounded characterization experiments.

    The legacy fixed ``warm = 8`` rule is not sufficient once circular convolution is
    removed: an FIR's startup lasts for its complete retained support.  v2 excludes
    ``max(r0, ceil((len(fir)-1)/samples_per_ui))`` rows before every score.  The same
    causal FIR is then linearly convolved with every pattern, so final symbols cannot wrap
    around and contaminate the beginning of a pattern.
    """
    freq_ac, H_ctle, M, N = _validate_inputs(
        freq_ac, H_ctle, bit_rate=bit_rate, samples_per_ui=samples_per_ui,
        n_bits=n_bits, swing_v=swing_v, channel_loss_db=channel_loss_db,
        nyquist_hz=nyquist_hz, dfe_taps=dfe_taps)
    if not np.isfinite(noise_sigma_v) or noise_sigma_v < 0.0:
        raise ValueError("noise_sigma_v must be non-negative and finite")

    fir = _v2_causal_fir(
        freq_ac, H_ctle, bit_rate=bit_rate, samples_per_ui=M, n_bits=N,
        swing_v=swing_v, channel_loss_db=channel_loss_db, nyquist_hz=nyquist_hz)
    P = _pulse_rows_v2(fir, M)
    r0 = int(np.argmax(np.max(np.abs(P), axis=1)))
    pulse_phase = int(np.argmax(np.abs(P[r0])))
    pulse_peak = float(P[r0, pulse_phase])
    if not np.isfinite(pulse_peak) or abs(pulse_peak) <= np.finfo(float).eps:
        raise ValueError("the end-to-end response has no non-zero main pulse cursor")
    polarity = 1.0 if pulse_peak >= 0.0 else -1.0

    startup = max(r0, int(np.ceil((fir.size - 1) / M)))
    rows = np.arange(startup, N)
    if rows.size == 0:
        raise ValueError("n_bits is too short after the v2 FIR startup exclusion")

    # SeedSequence separates selection, adaptation and evaluation even when users reuse
    # a top-level seed.  Each path gets a complete, independently generated NRZ pattern.
    phase_rng, dfe_rng, eval_rng, noise_rng = np.random.SeedSequence(seed).spawn(4)
    bits_phase = np.random.default_rng(phase_rng).integers(0, 2, N, dtype=np.int8) * 2 - 1
    bits_dfe = np.random.default_rng(dfe_rng).integers(0, 2, N, dtype=np.int8) * 2 - 1
    bits_eval = np.random.default_rng(eval_rng).integers(0, 2, N, dtype=np.int8) * 2 - 1

    Y_phase = _waveform_rows_v2(bits_phase, fir, M)
    truth_phase = bits_phase[rows - r0]
    best_ph = pulse_phase
    best_op = _truth_opening(Y_phase[rows, best_ph] * polarity, truth_phase)
    for ph in range(M):
        op = _truth_opening(Y_phase[rows, ph] * polarity, truth_phase)
        if op > best_op + 1e-12 * max(1.0, abs(op), abs(best_op)):
            best_ph, best_op = ph, op

    c0 = float(P[r0, best_ph])
    c1 = float(P[r0 + 1, best_ph]) if r0 + 1 < P.shape[0] else 0.0
    Y_dfe = _waveform_rows_v2(bits_dfe, fir, M)
    dfe_tap = _adapt_dfe_v2(Y_dfe, phase=best_ph, polarity=polarity, c0=c0,
                             start_row=startup, dfe_taps=dfe_taps)

    Y_eval = _waveform_rows_v2(bits_eval, fir, M)
    Yc, decisions = _decision_feedback_v2(
        Y_eval, phase=best_ph, polarity=polarity, tap=dfe_tap,
        noise_sigma_v=float(noise_sigma_v), rng=np.random.default_rng(noise_rng))
    truth = bits_eval[rows - r0]
    scored = Yc[rows] * polarity
    openings = np.array([_truth_opening(scored[:, ph], truth) for ph in range(M)])
    signed_opening = float(openings[best_ph])
    errors = int(np.count_nonzero(decisions[rows] != truth))
    count = int(rows.size)
    margins = scored[:, best_ph] * truth

    return EyeResultV2(
        height_v=max(signed_opening, 0.0),
        width_ui=_contiguous_open_width(openings, best_ph) / M,
        signed_opening_v=signed_opening,
        c0=c0,
        c1_pre=c1 - dfe_tap,
        dfe_tap=dfe_tap,
        samples_per_ui=M,
        sample_phase=best_ph,
        eye_matrix=scored,
        errors=errors,
        count=count,
        ber=errors / count,
        margins_v=margins,
    )
