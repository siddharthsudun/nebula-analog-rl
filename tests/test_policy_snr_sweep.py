"""Sanity tests for the SNR extension study's noise model.

These test the noise mathematics and the empirical-SNR recovery, not the policy: they
run no SPICE and load no checkpoint, so they belong in the ordinary test sweep.
"""
import math

import numpy as np
import pytest

from eqrl.experiments.policy_snr_sweep import (
    SNR_POINTS_DB, measured_snr_db, semi_analytic_ber, sigma_for_snr, spearman, wilson,
)
from eqrl.sim.channel import channel_response
from eqrl.sim.eye import compute_eye_v2


def _flat_response(n: int = 400):
    """A flat unity CTLE response over the AC sweep band -- channel only, no peaking."""
    freq = np.logspace(6, 10.3, n)
    return freq, np.ones(n, dtype=complex)


# --- the equation itself ----------------------------------------------------


@pytest.mark.parametrize("snr_db", SNR_POINTS_DB)
def test_sigma_matches_the_definition(snr_db):
    amp = 0.037
    assert sigma_for_snr(amp, snr_db) == pytest.approx(amp / 10.0 ** (snr_db / 20.0))


def test_zero_db_puts_sigma_at_the_signal_amplitude():
    assert sigma_for_snr(0.05, 0.0) == pytest.approx(0.05)


def test_ten_db_steps_scale_sigma_by_sqrt_ten():
    assert sigma_for_snr(1.0, 0.0) / sigma_for_snr(1.0, 10.0) == pytest.approx(
        math.sqrt(10.0), rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_sigma_rejects_a_non_positive_or_non_finite_amplitude(bad):
    with pytest.raises(ValueError):
        sigma_for_snr(bad, 10.0)


def test_generated_noise_has_the_requested_rms():
    """The generator the injection point uses: zero-mean Gaussian at sigma_v."""
    amp, snr_db = 0.04, 6.0
    sigma = sigma_for_snr(amp, snr_db)
    draw = np.random.default_rng(7).normal(0.0, sigma, size=400_000)
    assert float(np.mean(draw)) == pytest.approx(0.0, abs=6.0 * sigma / math.sqrt(4e5))
    assert float(np.std(draw)) == pytest.approx(sigma, rel=0.01)
    assert 20.0 * math.log10(amp / float(np.std(draw))) == pytest.approx(snr_db, abs=0.1)


# --- injection: requested SNR vs what the eye engine actually delivered ------


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_measured_snr_recovers_the_requested_snr(snr_db):
    """End-to-end: inject at a requested SNR, recover it from the scored samples.

    Rows whose preceding decision differed between the clean and noisy runs are excluded
    by `measured_snr_db`, because one-tap feedback subtracts a different value there.
    """
    freq, H = _flat_response()
    common = dict(channel_loss_db=12.0, n_bits=2048, seed=4242)
    ref = compute_eye_v2(freq, H, noise_sigma_v=0.0, **common)
    amp = abs(ref.c0)
    sigma = sigma_for_snr(amp, snr_db)
    noisy = compute_eye_v2(freq, H, noise_sigma_v=sigma, **common)

    got, got_sigma, n = measured_snr_db(ref, noisy, amp)
    assert n > 1000, "too few uncontaminated samples to verify the injection"
    assert got_sigma == pytest.approx(sigma, rel=0.05)
    assert got == pytest.approx(snr_db, abs=0.5)


def test_a_noiseless_run_is_bit_identical_to_its_own_reference():
    freq, H = _flat_response()
    common = dict(channel_loss_db=12.0, n_bits=1024, seed=11)
    a = compute_eye_v2(freq, H, noise_sigma_v=0.0, **common)
    b = compute_eye_v2(freq, H, noise_sigma_v=0.0, **common)
    assert np.array_equal(a.eye_matrix, b.eye_matrix)
    assert a.ber == b.ber == 0.0 or a.ber == b.ber


def test_the_same_seed_reproduces_the_same_noise_draw():
    freq, H = _flat_response()
    common = dict(channel_loss_db=12.0, n_bits=1024, seed=99)
    sigma = sigma_for_snr(abs(compute_eye_v2(freq, H, noise_sigma_v=0.0, **common).c0), 3.0)
    a = compute_eye_v2(freq, H, noise_sigma_v=sigma, **common)
    b = compute_eye_v2(freq, H, noise_sigma_v=sigma, **common)
    assert np.array_equal(a.eye_matrix, b.eye_matrix)
    assert a.ber == b.ber


def test_ber_is_monotone_non_increasing_in_snr():
    """More noise must never mean fewer errors on a shared pattern and DFE tap."""
    freq, H = _flat_response()
    common = dict(channel_loss_db=12.0, n_bits=4096, seed=5)
    ref = compute_eye_v2(freq, H, noise_sigma_v=0.0, **common)
    amp = abs(ref.c0)
    bers = [compute_eye_v2(freq, H, noise_sigma_v=sigma_for_snr(amp, s), **common).ber
            for s in SNR_POINTS_DB]
    assert all(a >= b for a, b in zip(bers, bers[1:])), bers
    assert bers[0] > bers[-1], "BER did not respond to a 25 dB SNR swing"


def test_semi_analytic_tracks_the_empirical_count_where_both_resolve():
    freq, H = _flat_response()
    common = dict(channel_loss_db=12.0, n_bits=4096, seed=17)
    ref = compute_eye_v2(freq, H, noise_sigma_v=0.0, **common)
    amp = abs(ref.c0)
    for snr in (0.0, 5.0):
        sigma = sigma_for_snr(amp, snr)
        emp = compute_eye_v2(freq, H, noise_sigma_v=sigma, **common).ber
        est = semi_analytic_ber(ref, sigma)
        assert emp > 0.0 and est > 0.0
        # correct-feedback approximation, so it is optimistic; one decade is the bound
        assert 0.1 <= (emp + 1e-12) / (est + 1e-12) <= 10.0, (snr, emp, est)


def test_channel_response_is_deterministic():
    """The channel carries no noise of its own; the AWGN is the only stochastic term."""
    freq = np.logspace(6, 10.3, 256)
    a = channel_response(freq, loss_nyquist_db=12.0)
    b = channel_response(freq, loss_nyquist_db=12.0)
    assert np.array_equal(a, b)


# --- statistics helpers -----------------------------------------------------


def test_wilson_brackets_the_point_estimate():
    p, lo, hi = wilson(7, 16)
    assert lo < p < hi and 0.0 <= lo and hi <= 1.0


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    for k, n in ((0, 16), (16, 16)):
        p, lo, hi = wilson(k, n)
        assert 0.0 <= lo <= p <= hi <= 1.0


def test_spearman_is_one_on_a_monotone_pair_and_none_on_a_constant():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [7, 7, 7, 7]) is None
