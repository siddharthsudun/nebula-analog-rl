"""Pure-DSP regression tests for the separately versioned audited eye scorer.

These tests do not start ngspice or require a PDK.  They exercise synthetic transfer
responses on the same frequency grid the behavioural eye model consumes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from silq.sim.eye import _contiguous_open_width, compute_eye_v2


BIT_RATE = 5e9
SAMPLES_PER_UI = 16


def _grid(n_bits: int) -> np.ndarray:
    return np.fft.rfftfreq(n_bits * SAMPLES_PER_UI, 1.0 / (BIT_RATE * SAMPLES_PER_UI))


def test_v2_identity_has_an_open_eye_and_no_decision_errors():
    n_bits = 512
    freq = _grid(n_bits)
    result = compute_eye_v2(
        freq, np.ones_like(freq, dtype=complex), bit_rate=BIT_RATE,
        samples_per_ui=SAMPLES_PER_UI, n_bits=n_bits, channel_loss_db=0.0,
        dfe_taps=0, seed=17)

    assert result.height_v == pytest.approx(1.0, abs=1e-12)
    assert result.signed_opening_v == pytest.approx(1.0, abs=1e-12)
    assert result.width_ui == pytest.approx(1.0, abs=1e-12)
    assert result.errors == 0 and result.ber == 0.0
    assert result.eye_matrix.shape == (result.count, SAMPLES_PER_UI)
    assert result.margins_v.shape == (result.count,)


def test_v2_inversion_normalizes_polarity_before_scoring():
    n_bits = 512
    freq = _grid(n_bits)
    result = compute_eye_v2(
        freq, -np.ones_like(freq, dtype=complex), bit_rate=BIT_RATE,
        samples_per_ui=SAMPLES_PER_UI, n_bits=n_bits, channel_loss_db=0.0,
        dfe_taps=0, seed=19)

    assert result.c0 < 0.0
    assert result.height_v == pytest.approx(1.0, abs=1e-12)
    assert result.errors == 0 and result.ber == 0.0


def test_v2_aligns_truth_to_a_pure_integer_ui_delay():
    n_bits = 512
    delay_ui = 3
    freq = _grid(n_bits)
    H = np.exp(-2j * np.pi * freq * (delay_ui / BIT_RATE))
    result = compute_eye_v2(
        freq, H, bit_rate=BIT_RATE, samples_per_ui=SAMPLES_PER_UI, n_bits=n_bits,
        channel_loss_db=0.0, dfe_taps=0, seed=23)

    assert result.c0 > 0.0
    assert result.height_v == pytest.approx(1.0, abs=1e-10)
    assert result.errors == 0 and result.ber == 0.0


def test_v2_marks_the_two_and_three_ui_postcursor_counterexample_closed():
    """A 1-tap DFE cannot remove two later post-cursors.

    The expected BER is the 0.25 Bernoulli limit plus finite-pattern variation.  The
    reference observation from the requested reproduction was about 0.2569; with an
    independently held-out v2 evaluation pattern the appropriate assertion is a
    binomial uncertainty band, not an exact replay of that sample.
    """
    n_bits = 2048
    freq = _grid(n_bits)
    H = (1.0
         + 0.8 * np.exp(-2j * np.pi * freq * (2.0 / BIT_RATE))
         + 0.8 * np.exp(-2j * np.pi * freq * (3.0 / BIT_RATE)))
    result = compute_eye_v2(
        freq, H, bit_rate=BIT_RATE, samples_per_ui=SAMPLES_PER_UI, n_bits=n_bits,
        channel_loss_db=0.0, dfe_taps=1, seed=0)

    expected_ber = 0.2569
    # Four standard errors (and a small modelling allowance) avoids treating an
    # independent random pattern as if it had to reproduce a historic sample exactly.
    uncertainty = 4.0 * math.sqrt(expected_ber * (1.0 - expected_ber) / result.count) + 0.005
    assert result.signed_opening_v <= 0.0
    assert result.height_v == 0.0 and result.width_ui == 0.0
    assert result.ber == pytest.approx(expected_ber, abs=uncertainty)


def test_contiguous_width_does_not_add_disconnected_open_intervals():
    openings = np.array([0.2, 0.1, -0.1, -0.2, 0.3, 0.2, -0.1, -0.1])
    assert _contiguous_open_width(openings, phase=0) == 2
    assert _contiguous_open_width(openings, phase=4) == 2
