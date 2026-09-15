"""Small, explicit contract for the pilot's external input-noise condition.

The pilot evaluates a deterministic set of noise values.  This module deliberately
does not sample a hidden random value: the request carries either one value, a range,
or an explicitly assumed range when the user did not specify noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


NoiseMode = str

PILOT_BANDWIDTH_HZ: tuple[float, float] = (1.0e7, 5.0e9)
UNKNOWN_NOISE_RANGE_VRMS: tuple[float, float] = (1.0e-3, 5.0e-2)
DEFAULT_SWING_V = 1.0

NOISE_REFERENCE = "ctle_input_after_channel"
NOISE_SPECTRUM = "flat_one_sided_psd"
SWING_REFERENCE = "differential_tx_peak_to_peak"

# The observation uses the upper edge of the explicit unknown prior as its noise
# scale.  Values above that pilot scale remain visible as values greater than one.
OBSERVATION_NOISE_SCALE_VRMS = UNKNOWN_NOISE_RANGE_VRMS[1]
OBSERVATION_SWING_SCALE_V = DEFAULT_SWING_V
OBSERVATION_FEATURES = (
    "low_vrms",
    "high_vrms",
    "swing_v",
    "mode_specific",
    "mode_range",
    "mode_unknown",
)


def _number(value: Any, *, name: str) -> float:
    """Coerce a scalar to a finite float, with a useful field-specific error."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mode(value: Any) -> NoiseMode:
    if not isinstance(value, str):
        raise ValueError("mode must be one of 'specific', 'range', or 'unknown'")
    result = value.strip().lower()
    if result not in {"specific", "range", "unknown"}:
        raise ValueError("mode must be one of 'specific', 'range', or 'unknown'")
    return result


@dataclass(frozen=True)
class NoiseRequest:
    """External noise input for the bounded pilot.

    ``specific`` has one evaluation value and ``range`` has two endpoint values.
    ``unknown`` uses the explicit positive default range when no bounds are supplied.
    Noise is input-referred at the CTLE input after the channel and uses a flat,
    one-sided PSD assumption.  ``swing_v`` is differential TX peak-to-peak voltage.
    """

    mode: NoiseMode
    low_vrms: float | None = None
    high_vrms: float | None = None
    bandwidth_hz: tuple[float, float] = PILOT_BANDWIDTH_HZ
    swing_v: float = DEFAULT_SWING_V
    _provenance: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mode = _mode(self.mode)
        object.__setattr__(self, "mode", mode)

        low_input = self.low_vrms
        high_input = self.high_vrms
        assumed_unknown = mode == "unknown" and low_input is None and high_input is None

        if mode == "unknown" and (low_input is None) != (high_input is None):
            raise ValueError("unknown noise requires both low_vrms and high_vrms when supplied")

        if assumed_unknown:
            low_input, high_input = UNKNOWN_NOISE_RANGE_VRMS
        elif mode == "specific" and low_input is None and high_input is not None:
            low_input = high_input
        elif mode == "specific" and high_input is None and low_input is not None:
            high_input = low_input

        if low_input is None or high_input is None:
            raise ValueError(f"{mode} noise requires low_vrms and high_vrms")

        low = _number(low_input, name="low_vrms")
        high = _number(high_input, name="high_vrms")
        if low < 0.0 or high < 0.0:
            raise ValueError("noise bounds must be non-negative")
        if high < low:
            raise ValueError("high_vrms must be greater than or equal to low_vrms")
        if mode == "specific" and high != low:
            raise ValueError("specific noise requires low_vrms == high_vrms")
        if mode == "unknown" and high <= 0.0:
            raise ValueError("unknown noise must include a positive non-zero bound")

        try:
            bandwidth = tuple(self.bandwidth_hz)
        except TypeError as exc:
            raise ValueError("bandwidth_hz must contain exactly [1e7, 5e9] Hz") from exc
        if len(bandwidth) != 2:
            raise ValueError("bandwidth_hz must contain exactly [1e7, 5e9] Hz")
        bandwidth_values = tuple(_number(v, name="bandwidth_hz") for v in bandwidth)
        if bandwidth_values != PILOT_BANDWIDTH_HZ:
            raise ValueError("bandwidth_hz is fixed to [1e7, 5e9] Hz for the pilot")

        swing = _number(self.swing_v, name="swing_v")
        if swing <= 0.0:
            raise ValueError("swing_v must be strictly positive")

        object.__setattr__(self, "low_vrms", low)
        object.__setattr__(self, "high_vrms", high)
        object.__setattr__(self, "bandwidth_hz", bandwidth_values)
        object.__setattr__(self, "swing_v", swing)
        object.__setattr__(self, "_provenance", "assumed" if assumed_unknown else "supplied")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NoiseRequest":
        """Build a request from ``value_vrms`` or a ``low_vrms``/``high_vrms`` pair.

        If ``mode`` is omitted, a value implies ``specific``, a pair implies
        ``range``, and an empty mapping implies ``unknown``.  Combining the value form
        with bound fields is rejected because it is ambiguous even when the numbers
        happen to agree.
        """
        if not isinstance(data, Mapping):
            raise TypeError("noise request must be a mapping")

        allowed = {
            "mode", "value_vrms", "low_vrms", "high_vrms", "low", "high",
            "bandwidth_hz", "swing_v", "provenance", "units", "reference",
        }
        unknown_keys = sorted(set(data) - allowed)
        if unknown_keys:
            raise ValueError(f"unknown noise request field(s): {', '.join(unknown_keys)}")
        expected_units = {"noise": "Vrms", "bandwidth": "Hz", "swing": "Vpp"}
        expected_reference = {"location": NOISE_REFERENCE, "spectrum": NOISE_SPECTRUM,
                              "swing": SWING_REFERENCE}
        for key, expected in (("units", expected_units), ("reference", expected_reference)):
            if key in data and data[key] != expected:
                raise ValueError(f"{key} conflicts with the pilot measurement definition")

        # Accept concise low/high aliases, but do not let two spellings disagree.
        if "low" in data and "low_vrms" in data and data["low"] != data["low_vrms"]:
            raise ValueError("low and low_vrms conflict")
        if "high" in data and "high_vrms" in data and data["high"] != data["high_vrms"]:
            raise ValueError("high and high_vrms conflict")
        normalized = dict(data)
        if "low" in normalized:
            normalized["low_vrms"] = normalized["low"]
        if "high" in normalized:
            normalized["high_vrms"] = normalized["high"]
        data = normalized

        has_value = "value_vrms" in data
        has_low = "low_vrms" in data
        has_high = "high_vrms" in data
        has_bounds = has_low or has_high
        if has_value and has_bounds:
            raise ValueError("use either value_vrms or low_vrms/high_vrms, not both")

        raw_mode = data.get("mode")
        if raw_mode is None:
            if has_value:
                selected_mode = "specific"
            elif has_bounds:
                selected_mode = "range"
            else:
                selected_mode = "unknown"
        else:
            selected_mode = _mode(raw_mode)

        if selected_mode == "unknown":
            if has_value:
                raise ValueError("unknown mode cannot include value_vrms")
            if has_bounds:
                # ``to_dict`` includes the assumed bounds so that the serialized
                # record is self describing. Accept that exact marked form for
                # round trips; user supplied unknown bounds remain ambiguous.
                is_assumed_record = data.get("provenance") == "assumed"
                matches_default = (
                    has_low and has_high
                    and data["low_vrms"] == UNKNOWN_NOISE_RANGE_VRMS[0]
                    and data["high_vrms"] == UNKNOWN_NOISE_RANGE_VRMS[1]
                )
                if not (is_assumed_record and matches_default):
                    raise ValueError("unknown mode cannot include supplied noise bounds")
            low = high = None
        elif selected_mode == "specific":
            if has_value:
                low = high = data["value_vrms"]
            elif has_low and has_high:
                low, high = data["low_vrms"], data["high_vrms"]
            elif has_low:
                low = high = data["low_vrms"]
            elif has_high:
                low = high = data["high_vrms"]
            else:
                raise ValueError("specific noise requires value_vrms")
        else:  # range
            if has_value:
                raise ValueError("range mode requires low_vrms and high_vrms")
            if not (has_low and has_high):
                raise ValueError("range noise requires both low_vrms and high_vrms")
            low, high = data["low_vrms"], data["high_vrms"]

        bandwidth = data.get("bandwidth_hz", PILOT_BANDWIDTH_HZ)
        swing = data.get("swing_v", DEFAULT_SWING_V)
        return cls(selected_mode, low, high, bandwidth_hz=bandwidth, swing_v=swing)

    @property
    def provenance(self) -> str:
        """Whether the numeric bounds were supplied or explicitly assumed."""
        return self._provenance

    def endpoints(self) -> tuple[float, ...]:
        """Return the deterministic pilot values, with no hidden random draw."""
        if self.mode == "specific":
            return (self.low_vrms,)
        return (self.low_vrms, self.high_vrms)

    def observation(self) -> tuple[float, ...]:
        """Return finite numeric features for a policy observation.

        The two noise bounds are scaled by the upper edge of the explicit unknown
        pilot range, and swing is scaled by the nominal 1 V differential TX swing.
        The values are intentionally not clipped, so a supplied out-of-scale value is
        visible to the policy rather than silently collapsed at one.
        """
        return (
            float(self.low_vrms / OBSERVATION_NOISE_SCALE_VRMS),
            float(self.high_vrms / OBSERVATION_NOISE_SCALE_VRMS),
            float(self.swing_v / OBSERVATION_SWING_SCALE_V),
            float(self.mode == "specific"),
            float(self.mode == "range"),
            float(self.mode == "unknown"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready values plus the assumptions needed to interpret them."""
        return {
            "mode": self.mode,
            "low_vrms": self.low_vrms,
            "high_vrms": self.high_vrms,
            "bandwidth_hz": list(self.bandwidth_hz),
            "swing_v": self.swing_v,
            "provenance": self.provenance,
            "units": {
                "noise": "Vrms",
                "bandwidth": "Hz",
                "swing": "Vpp",
            },
            "reference": {
                "location": NOISE_REFERENCE,
                "spectrum": NOISE_SPECTRUM,
                "swing": SWING_REFERENCE,
            },
        }


__all__ = [
    "DEFAULT_SWING_V",
    "NOISE_REFERENCE",
    "NOISE_SPECTRUM",
    "NoiseRequest",
    "OBSERVATION_FEATURES",
    "PILOT_BANDWIDTH_HZ",
    "SWING_REFERENCE",
    "UNKNOWN_NOISE_RANGE_VRMS",
]
