"""Version 2 external-noise contract. Legacy noise_spec remains reproducible."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

SCHEMA = "eqrl.snr.request.v2"
MODES = ("measured", "estimated", "unknown")
REFERENCES = ("tx_vpp", "ctle_input_vrms")
DEFAULT_BAND = (1e7, 5e9)


def number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class SNRRequest:
    mode: str
    low_vrms: float
    high_vrms: float
    signal_reference: str
    signal_value_v: float
    bandwidth_hz: tuple[float, float]
    provenance: dict
    noise_form: str = "range"

    def __post_init__(self):
        if self.mode not in MODES or self.signal_reference not in REFERENCES:
            raise ValueError("unsupported noise mode or signal reference")
        for key in ("low_vrms", "high_vrms", "signal_value_v"):
            object.__setattr__(self, key, number(getattr(self, key), key))
        if not 0 <= self.low_vrms <= self.high_vrms or self.signal_value_v <= 0:
            raise ValueError("noise must be nonnegative and ordered; signal must be positive")
        if self.mode == "measured" and self.low_vrms != self.high_vrms:
            raise ValueError("measured requires one noise RMS value")
        if self.mode == "unknown" and self.high_vrms <= 0:
            raise ValueError("unknown noise must have a positive upper bound")
        if not isinstance(self.bandwidth_hz, (list, tuple)) or len(self.bandwidth_hz) != 2:
            raise ValueError("bandwidth_hz requires [low, high] edges")
        band = tuple(number(v, "bandwidth_hz") for v in self.bandwidth_hz)
        if not 1e7 <= band[0] < band[1] <= 5e9:
            raise ValueError("supported noise bands require 10 MHz <= low < high <= 5 GHz")
        object.__setattr__(self, "bandwidth_hz", band)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ValueError("noise_request must be an object")
        allowed = {"schema", "mode", "value_vrms", "budget_vrms", "low_vrms", "high_vrms",
                   "signal_reference", "signal_value_v", "bandwidth_hz", "assumed_fields"}
        if set(data) - allowed:
            raise ValueError(f"unknown noise fields: {sorted(set(data) - allowed)}")
        if data.get("schema", SCHEMA) != SCHEMA:
            raise ValueError("unsupported noise schema")
        mode = data.get("mode")
        if mode not in MODES:
            raise ValueError("mode must be measured, estimated, or unknown")
        fields = ("noise", "signal", "bandwidth")
        assumed = data.get("assumed_fields", [])
        if not isinstance(assumed, list) or any(x not in fields for x in assumed):
            raise ValueError("assumed_fields must name noise, signal, or bandwidth")
        if mode != "unknown" and assumed:
            raise ValueError("assumptions require Unknown mode")
        provenance = {k: ("assumed" if mode == "unknown" else "supplied") for k in fields}
        forms = int("value_vrms" in data) + int("budget_vrms" in data) + int("low_vrms" in data or "high_vrms" in data)
        if forms > 1:
            raise ValueError("supply one of value, budget, or range")
        if mode == "measured":
            if "value_vrms" not in data:
                raise ValueError("Measured requires value_vrms")
            low = high = data["value_vrms"]
            form = "value"
        elif "budget_vrms" in data:
            if mode != "estimated":
                raise ValueError("budget_vrms requires Estimated mode")
            low, high, form = 0., data["budget_vrms"], "budget"
        elif "low_vrms" in data and "high_vrms" in data:
            low, high, form = data["low_vrms"], data["high_vrms"], "range"
        elif mode == "unknown" and forms == 0:
            low, high, form = .001, .05, "range"
        else:
            raise ValueError("Estimated and edited Unknown require both noise range bounds")
        signal_present = "signal_reference" in data and "signal_value_v" in data
        if ("signal_reference" in data) != ("signal_value_v" in data):
            raise ValueError("supply both signal_reference and signal_value_v")
        if mode != "unknown" and (not signal_present or "bandwidth_hz" not in data):
            raise ValueError("Measured and Estimated require signal and bandwidth")
        if mode == "unknown":
            for key, present in (("noise", forms > 0), ("signal", signal_present),
                                 ("bandwidth", "bandwidth_hz" in data)):
                # Unknown numeric noise remains a user-edited assumption, not a measurement.
                provenance[key] = "assumed" if key == "noise" or not present or key in assumed else "supplied"
        return cls(mode, low, high, data.get("signal_reference", "tx_vpp"),
                   data.get("signal_value_v", 1.), data.get("bandwidth_hz", DEFAULT_BAND), provenance, form)

    def points(self):
        if self.low_vrms == self.high_vrms:
            return (self.low_vrms,)
        return tuple(self.low_vrms + (self.high_vrms-self.low_vrms)*i/4 for i in range(5))

    def endpoints(self):
        return (self.low_vrms, self.high_vrms)

    def observation(self, equivalent_tx_vpp=1.):
        return (self.low_vrms/.05, self.high_vrms/.05, equivalent_tx_vpp,
                self.bandwidth_hz[0]/5e9, self.bandwidth_hz[1]/5e9,
                *(float(self.mode == m) for m in MODES))

    def to_dict(self):
        result = {"schema": SCHEMA, "mode": self.mode,
                  "signal_reference": self.signal_reference, "signal_value_v": self.signal_value_v,
                  "bandwidth_hz": list(self.bandwidth_hz),
                  "assumed_fields": [k for k, v in self.provenance.items() if v == "assumed"]}
        if self.noise_form == "value":
            result["value_vrms"] = self.low_vrms
        elif self.noise_form == "budget":
            result["budget_vrms"] = self.high_vrms
        else:
            result.update(low_vrms=self.low_vrms, high_vrms=self.high_vrms)
        return result


def experimental_request(value):
    """Explicit compatibility adapter for old internal pilot fixtures only, not HTTP."""
    if isinstance(value, SNRRequest):
        return value
    if not isinstance(value, dict):
        value = value.to_dict()
    if value.get("mode") in ("specific", "range") or "swing_v" in value:
        mode = {"specific": "measured", "range": "estimated", "unknown": "unknown"}[value["mode"]]
        payload = {"mode": mode, "signal_reference": "tx_vpp", "signal_value_v": value.get("swing_v", 1.),
                   "bandwidth_hz": value.get("bandwidth_hz", DEFAULT_BAND)}
        if mode == "measured":
            payload["value_vrms"] = value.get("value_vrms", value.get("low_vrms"))
        elif mode == "estimated":
            payload.update(low_vrms=value["low_vrms"], high_vrms=value["high_vrms"])
        return SNRRequest.from_dict(payload)
    return SNRRequest.from_dict(value)
