"""LLM wrapper (the bonus deliverable): natural language -> Spec.

"Design me a PCIe Gen2 CTLE with about 9 dB of boost, under 12 mW" -> Spec(...).
Uses the Claude Messages API. Before extending this, load the `claude-api` skill for the
current model ids and SDK usage.
"""
from __future__ import annotations

import json

from eqrl.specs import Spec

_SYSTEM = """You convert an analog-equalizer design request into a JSON object of target
specs. Only output JSON with keys that match this schema (omit unknowns to use defaults):
target_boost_db, boost_db_min, boost_db_max, power_w_max, area_mm2_max, noise_vrms_max,
hd3_db_max, data_rate_gbps, nyquist_ghz, peak_freq_lo_ghz, peak_freq_hi_ghz,
eye_h_ui_min, eye_v_mv_min. Convert units to SI (mW->W, mV->V)."""


def parse_spec(text: str, model: str = "claude-opus-4-8") -> Spec:
    """Parse a natural-language request into a Spec via Claude.

    Requires ANTHROPIC_API_KEY. Falls back to a keyword heuristic if the SDK is absent.
    """
    try:
        import anthropic
    except ImportError:
        return _heuristic(text)

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = msg.content[0].text
    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    fields = json.loads(raw)
    return Spec(**{k: v for k, v in fields.items() if k in Spec.__annotations__})


def _heuristic(text: str) -> Spec:
    """Dependency-free fallback so the pipeline runs without an API key."""
    import re

    kw = {}
    if m := re.search(r"(\d+(?:\.\d+)?)\s*dB", text, re.I):
        kw["target_boost_db"] = float(m.group(1))
    if m := re.search(r"(\d+(?:\.\d+)?)\s*mW", text, re.I):
        kw["power_w_max"] = float(m.group(1)) * 1e-3
    return Spec(**kw)


if __name__ == "__main__":
    import sys
    s = parse_spec(" ".join(sys.argv[1:]) or "PCIe Gen2 CTLE, ~9 dB boost, under 12 mW")
    print(s)
