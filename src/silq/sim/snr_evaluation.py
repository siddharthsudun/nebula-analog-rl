"""Shared API/training scorer; all voltages are differential.

Signal RMS is band-limited and measured before DFE on an independent modeled
calibration pattern. Integrated noise is approximated as white slicer noise.
"""
from dataclasses import replace
from functools import lru_cache
import math
import numpy as np

from silq.sim.eye import compute_eye_v2, _v2_causal_fir, _waveform_rows_v2
from silq.sim.noise_budget import flat_noise_gain, combined_output_rms, differential_output_noise


def waveform_rms(freq, response, channel, band, swing=1.):
    bits = np.random.default_rng(620391).choice([-1, 1], 2048)
    fir = _v2_causal_fir(np.asarray(freq), np.asarray(response), bit_rate=5e9,
        samples_per_ui=16, n_bits=2048, swing_v=swing, channel_loss_db=channel, nyquist_hz=2.5e9)
    waveform = _waveform_rows_v2(bits, fir, 16)[32:].ravel()
    spectrum = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(waveform.size, 1/80e9)
    weights = np.full(spectrum.size, 2.)
    weights[0] = weights[-1] = 1.
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    rms = float(np.sqrt(np.sum(weights[mask]*np.abs(spectrum[mask])**2))/waveform.size)
    if not math.isfinite(rms) or rms <= 1e-12:
        raise ValueError("calibration band contains no resolvable signal energy")
    return rms


@lru_cache(maxsize=512)
def channel_unit_rms(channel, band):
    return waveform_rms([1e6, 24e9], [1.+0j, 1.+0j], channel, band)


def signal_calibration(request, channel):
    unit = channel_unit_rms(float(channel), tuple(request.bandwidth_hz))
    swing = request.signal_value_v if request.signal_reference == "tx_vpp" else request.signal_value_v/unit
    if not math.isfinite(swing) or swing <= 0 or swing > 100:
        raise ValueError("signal normalization exceeds the supported behavioral amplitude (100 Vpp)")
    return swing, swing*unit


def snr_db(signal, noise):
    # JSON cannot represent infinity. Zero external noise is an explicit control.
    return None if noise == 0 else float(20*math.log10(signal/noise))


def evaluate_snr(srv, dv, nominal, spec, request, *, seed=0, eye_bits=512, target_weight=None):
    from silq.envs.sequential_env import _shaped
    response = srv.ac_complex(dv, vdd=spec.vdd_nominal)
    band = request.bandwidth_hz
    gain = flat_noise_gain(response['freq'], response['H'], *band)
    receiver = differential_output_noise(srv, dv, vdd=spec.vdd_nominal, bandwidth_hz=band)
    swing, input_signal = signal_calibration(request, spec.channel_loss_db)
    output_signal = waveform_rms(response['freq'], response['H'], spec.channel_loss_db, band, swing)
    points, candidates, scores = [], [], []
    for input_rms in request.points():
        sigma = combined_output_rms(input_rms, gain, receiver)
        eye = compute_eye_v2(response['freq'], response['H'], n_bits=eye_bits, swing_v=swing,
                            channel_loss_db=spec.channel_loss_db, seed=seed, noise_sigma_v=sigma)
        if (not np.isfinite(eye.height_v) or not np.isfinite(eye.width_ui)
                or not 0 <= eye.width_ui <= 1 or eye.height_v < 0
                or eye.count <= 0 or not 0 <= eye.errors <= eye.count):
            raise ValueError("unusable noisy-eye measurement")
        measured = replace(nominal, eye_h_ui=eye.width_ui, eye_v_mv=eye.height_v*1e3)
        score, passed = _shaped(measured, spec, target_weight)
        if not math.isfinite(score):
            raise ValueError("non-finite noisy score")
        candidates.append(measured)
        scores.append(score)
        points.append({'input_noise_vrms': input_rms, 'output_noise_vrms': sigma,
            'input_snr_db': snr_db(input_signal, input_rms), 'output_snr_db': snr_db(output_signal, sigma),
            'input_snr_zero_noise': input_rms == 0,
            'eye_v_mv': measured.eye_v_mv, 'eye_h_ui': measured.eye_h_ui,
            'errors': eye.errors, 'count': eye.count, 'ber_empirical': eye.ber,
            'sample_phase': eye.sample_phase, 'dfe_tap': eye.dfe_tap, 'passed': bool(passed),
            'noisy_eye_passed': bool(measured.eye_v_mv >= spec.eye_v_mv_min and measured.eye_h_ui >= spec.eye_h_ui_min)})
    worst = int(np.argmin(scores))
    passed = all(p['passed'] for p in points)
    coverage = ((.001 <= request.low_vrms <= request.high_vrms <= .05
                 or (request.noise_form == 'budget' and .001 <= request.high_vrms <= .05))
                and .5 <= swing <= 1.)
    details = {'schema': 'silq.snr.result.v2', 'status': 'measured',
        'mode': request.mode, 'conditional': request.mode == 'unknown',
        'provenance': request.provenance, 'request': request.to_dict(),
        'passed': passed, 'worst_point_index': worst, 'points': points,
        'abs_target_error_db': abs(nominal.boost_db-spec.target_boost_db),
        'receiver_output_noise_vrms': receiver, 'external_noise_gain_rms': gain,
        'equivalent_tx_vpp': swing, 'modeled_input_signal_vrms': input_signal,
        'modeled_output_signal_vrms': output_signal, 'within_training_coverage': coverage,
        'bandwidth_hz': list(band), 'eye_seed': seed,
        'scope': 'sampled noise points; equivalent white slicer noise; no low-BER or continuous-range guarantee',
        'signal_reference': 'band-limited differential RMS at CTLE input and output before DFE',
        'assumptions': ['5 Gb/s NRZ, modeled minimum-phase channel, nominal TT',
            'flat external noise PSD; independent receiver noise powers',
            'independent 2048-bit channel calibration; inferred TX swing for input RMS',
            'linear behavioral signal scaling; no amplitude-specific distortion sign-off']}
    return candidates[worst], scores[worst], passed, details
