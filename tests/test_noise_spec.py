"""Contract tests for the bounded external-noise pilot input."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from eqrl.noise_spec import (
    NOISE_REFERENCE,
    NOISE_SPECTRUM,
    OBSERVATION_FEATURES,
    PILOT_BANDWIDTH_HZ,
    SWING_REFERENCE,
    UNKNOWN_NOISE_RANGE_VRMS,
    NoiseRequest,
)


def test_specific_value_is_one_deterministic_endpoint_and_frozen():
    request = NoiseRequest.from_dict({"mode": "specific", "value_vrms": 0.002})

    assert request.mode == "specific"
    assert request.low_vrms == request.high_vrms == 0.002
    assert request.endpoints() == (0.002,)
    with pytest.raises(FrozenInstanceError):
        request.swing_v = 2.0


def test_range_has_two_deterministic_endpoints():
    request = NoiseRequest.from_dict(
        {"mode": "range", "low_vrms": 0.001, "high_vrms": 0.004}
    )

    assert request.endpoints() == (0.001, 0.004)
    assert request.bandwidth_hz == PILOT_BANDWIDTH_HZ


def test_unknown_is_explicit_positive_assumed_range():
    request = NoiseRequest.from_dict({"mode": "unknown"})

    assert (request.low_vrms, request.high_vrms) == UNKNOWN_NOISE_RANGE_VRMS
    assert request.endpoints() == UNKNOWN_NOISE_RANGE_VRMS
    assert request.provenance == "assumed"
    assert request.high_vrms > 0.0


def test_zero_noise_is_valid_when_explicit_but_not_for_unknown():
    zero = NoiseRequest.from_dict({"mode": "specific", "value_vrms": 0.0})
    assert zero.endpoints() == (0.0,)

    with pytest.raises(ValueError, match="unknown noise"):
        NoiseRequest("unknown", low_vrms=0.0, high_vrms=0.0)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"mode": "range", "low_vrms": 0.004, "high_vrms": 0.001}, "greater"),
        ({"mode": "range", "low_vrms": -0.001, "high_vrms": 0.001}, "non-negative"),
        ({"mode": "specific", "low_vrms": 0.001, "high_vrms": 0.002}, "specific"),
        ({"mode": "specific", "value_vrms": math.nan}, "finite"),
        ({"mode": "range", "low_vrms": 0.001}, "both"),
    ],
)
def test_invalid_bounds_are_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        NoiseRequest.from_dict(payload)


def test_invalid_swing_and_bandwidth_are_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        NoiseRequest("specific", 0.001, 0.001, swing_v=0.0)
    with pytest.raises(ValueError, match="fixed"):
        NoiseRequest("specific", 0.001, 0.001, bandwidth_hz=(1.0e6, 5.0e9))


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "specific", "value_vrms": 0.001, "low_vrms": 0.001},
        {"mode": "range", "value_vrms": 0.001, "low_vrms": 0.001, "high_vrms": 0.002},
        {"mode": "unknown", "value_vrms": 0.001},
        {"mode": "range", "low_vrms": 0.001},
    ],
)
def test_ambiguous_or_incomplete_dict_forms_are_rejected(payload):
    with pytest.raises(ValueError):
        NoiseRequest.from_dict(payload)


def test_from_dict_can_infer_specific_or_range_without_ambiguous_fields():
    assert NoiseRequest.from_dict({"value_vrms": 0.002}).mode == "specific"
    assert NoiseRequest.from_dict({"low_vrms": 0.001, "high_vrms": 0.002}).mode == "range"
    assert NoiseRequest.from_dict({"mode": "range", "low": 0.001, "high": 0.002}).endpoints() == (
        0.001, 0.002
    )
    assert NoiseRequest.from_dict({}).mode == "unknown"


def test_unknown_fields_are_not_silently_ignored_and_unknown_serialization_roundtrips():
    with pytest.raises(ValueError, match="unknown noise request field"):
        NoiseRequest.from_dict({"noise_mvrms": 0.001})

    request = NoiseRequest.from_dict({"mode": "unknown"})
    assert NoiseRequest.from_dict(request.to_dict()) == request


def test_to_dict_records_provenance_units_and_reference():
    supplied = NoiseRequest.from_dict({"mode": "specific", "value_vrms": 0.002})
    assumed = NoiseRequest.from_dict({"mode": "unknown"})

    encoded = supplied.to_dict()
    assert encoded["provenance"] == "supplied"
    assert encoded["units"] == {"noise": "Vrms", "bandwidth": "Hz", "swing": "Vpp"}
    assert encoded["reference"] == {
        "location": NOISE_REFERENCE,
        "spectrum": NOISE_SPECTRUM,
        "swing": SWING_REFERENCE,
    }
    assert assumed.to_dict()["provenance"] == "assumed"


def test_observation_is_numeric_normalized_bounds_swing_and_one_hot():
    request = NoiseRequest.from_dict(
        {"mode": "range", "low_vrms": 0.001, "high_vrms": 0.004, "swing_v": 0.5}
    )

    observation = request.observation()
    assert OBSERVATION_FEATURES == (
        "low_vrms", "high_vrms", "swing_v", "mode_specific", "mode_range", "mode_unknown"
    )
    assert len(observation) == len(OBSERVATION_FEATURES) == 6
    assert all(isinstance(value, float) and math.isfinite(value) for value in observation)
    assert observation[:3] == pytest.approx((0.02, 0.08, 0.5))
    assert observation[3:] == (0.0, 1.0, 0.0)
