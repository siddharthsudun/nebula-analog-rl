"""LLM wrapper (the bonus deliverable): natural language -> Spec.

"Design me a PCIe Gen2 CTLE with about 9 dB of boost, under 12 mW" -> Spec(...).
Uses the Claude Messages API. Before extending this, load the `claude-api` skill for the
current model ids and SDK usage.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from eqrl.specs import Spec

_SYSTEM = """You convert an analog-equalizer design request into a JSON object of target
specs. Only output JSON with keys that match this schema (omit unknowns to use defaults):
target_boost_db, boost_db_min, boost_db_max, power_w_max, area_mm2_max, noise_vrms_max,
hd3_db_max, data_rate_gbps, nyquist_ghz, peak_freq_lo_ghz, peak_freq_hi_ghz,
eye_h_ui_min, eye_v_mv_min. Convert units to SI (mW->W, mV->V).

If the message contains no analog design intent at all -- it is greeting, chatter,
nonsense, or a request about something other than an equalizer -- return exactly {} and
nothing else. Do NOT invent plausible numbers to fill the schema. An empty object is the
correct answer for an unparseable request; a guess is not."""


@dataclass
class ParseResult:
    """What the parser actually extracted, separate from what the defaults would be.

    `spec` always holds a usable Spec (defaults where nothing was extracted) so callers
    that only want to run can ignore the rest. `recognised` is the honest part: the spec
    fields the text actually pinned down. When it is empty NOTHING in the message was
    understood, and `spec` is pure defaults wearing the costume of a parsed request --
    callers must not present it as an interpretation of the user's words.
    """
    spec: Spec
    recognised: dict = field(default_factory=dict)
    source: str = "heuristic"

    @property
    def understood(self) -> bool:
        return bool(self.recognised)


def parse_spec(text: str, model: str = "claude-opus-4-8") -> Spec:
    """Parse a natural-language request into a Spec via Claude.

    Convenience wrapper over `parse_spec_verbose` for callers that only need the Spec.
    Note that an unparseable request yields a DEFAULT Spec, not an error -- use
    `parse_spec_verbose` if you need to tell those two cases apart, which any interface
    that echoes the result back to a human does.
    """
    return parse_spec_verbose(text, model=model).spec


def parse_spec_verbose(text: str, model: str = "claude-opus-4-8") -> ParseResult:
    """Parse a request and report which fields were actually recognised.

    Requires an Anthropic credential for LLM parsing; otherwise uses a keyword heuristic.
    """
    # The SDK may be installed in a developer environment even when no credential is
    # configured. Avoid constructing a client in that case so offline baseline/fallback
    # runs use the documented parser instead of failing before the circuit is evaluated.
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        return _heuristic(text)
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
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        # The model answered in prose instead of JSON. That is a failed parse, not a
        # reason to silently hand back defaults.
        return ParseResult(Spec(), {}, "llm")
    fields = json.loads(raw[start:end + 1])
    kw = {k: v for k, v in fields.items() if k in Spec.__annotations__}
    return ParseResult(Spec(**kw), kw, "llm")


#: How to show each spec field to a human: (unit label, multiplier from the SI value the
#: Spec stores). A `Spec` holds power in watts and noise in volts because that is what
#: the evaluator computes in, but nobody reads a budget as "0.015 W" -- and rounded for
#: display it becomes "0.01", which is a different number. The scale lives here, next to
#: the field list, so every interface renders them the same way. Supply voltage is volts
#: and stays volts: the rule is per field, not per unit.
UNITS = {
    "data_rate_gbps": ("Gbps", 1.0), "nyquist_ghz": ("GHz", 1.0),
    "target_boost_db": ("dB", 1.0), "boost_db_min": ("dB", 1.0),
    "boost_db_max": ("dB", 1.0),
    "peak_freq_lo_ghz": ("GHz", 1.0), "peak_freq_hi_ghz": ("GHz", 1.0),
    "dc_gain_db_min": ("dB", 1.0), "hd3_db_max": ("dB", 1.0),
    "noise_vrms_max": ("mV", 1e3), "power_w_max": ("mW", 1e3),
    "area_mm2_max": ("mm²", 1.0),
    "eye_h_ui_min": ("UI", 1.0), "eye_v_mv_min": ("mV", 1.0),
    "channel_loss_db": ("dB", 1.0), "vdd_nominal": ("V", 1.0),
}

_N = r"[-+]?\d+(?:\.\d+)?"          # a number
_R = r"[-‐-―~]|to"        # a range separator: hyphen, any dash, tilde, "to"
_SI = {"m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "": 1.0}
#: A same-line gap that may not contain another "dB". Anchoring a dB field on its label
#: is not enough on its own: a sentence naming two dB quantities ("8.9 dB boost over a
#: 14 dB channel") lets a lazy gap skip past the right number to the wrong one.
_GAP = r"(?:(?!dB)[^\n])*?"
_BOOST = r"\b(?:boost|peaking|equali[sz]ation)\b"


def _heuristic(text: str) -> ParseResult:
    """Dependency-free fallback so the pipeline runs without an API key.

    Label-anchored rather than unit-anchored. The earlier version searched for the first
    `<number> dB` anywhere in the message, which is wrong the moment a real spec sheet
    mentions dB more than once -- a request quoting boost, DC gain and HD3 would have had
    its HD3 line silently read as the boost target. Every pattern here is tied to the
    name of the thing it measures and confined to one line, so an unlabelled number is
    left unrecognised instead of being attached to whichever field came first.
    """
    import re

    kw: dict = {}

    def take(key, pattern, scale=1.0, group=1):
        if m := re.search(pattern, text, re.I):
            kw[key] = float(m.group(group)) * scale

    # -- signaling --------------------------------------------------------------------
    take("data_rate_gbps", rf"({_N})\s*Gb(?:ps|/s|it/s)")
    take("nyquist_ghz", rf"nyquist[^\n]*?({_N})\s*GHz")
    # A stated data rate implies Nyquist at half of it for NRZ; only fill it in when the
    # message did not say so itself, and only for NRZ (PAM4 carries two bits per symbol).
    if "data_rate_gbps" in kw and "nyquist_ghz" not in kw and not re.search(
            r"pam-?4", text, re.I):
        kw["nyquist_ghz"] = kw["data_rate_gbps"] / 2.0

    # -- peaking ----------------------------------------------------------------------
    # A boost RANGE ("3-12 dB of boost") sets the bounds; a single value sets the target.
    if m := re.search(rf"boost[^\n]*?({_N})\s*(?:dB)?\s*(?:{_R})\s*({_N})\s*dB", text,
                      re.I):
        kw["boost_db_min"], kw["boost_db_max"] = float(m.group(1)), float(m.group(2))
    else:
        # Both word orders occur -- "8.9 dB of boost" and "boost: 8.9 dB" -- so try the
        # number-before form first, then the number-after form. Both gaps are forbidden
        # from crossing another `dB`, which is what stops "8.9 dB boost over a 14 dB
        # channel" from reporting a 14 dB boost.
        take("target_boost_db", rf"({_N})\s*dB{_GAP}{_BOOST}")
        if "target_boost_db" not in kw:
            take("target_boost_db", rf"{_BOOST}{_GAP}({_N})\s*dB")
    if m := re.search(rf"peak[^\n]*?({_N})\s*(?:GHz)?\s*(?:{_R})\s*({_N})\s*GHz", text,
                      re.I):
        kw["peak_freq_lo_ghz"], kw["peak_freq_hi_ghz"] = (float(m.group(1)),
                                                          float(m.group(2)))
    take("dc_gain_db_min", rf"dc\s*gain[^\n]*?({_N})\s*dB")

    # -- hard constraints -------------------------------------------------------------
    take("hd3_db_max", rf"(?:hd3|third[- ]harmonic|linearity)[^\n]*?({_N})\s*dB")
    if m := re.search(rf"noise[^\n]*?({_N})\s*([munµ]?)V", text, re.I):
        kw["noise_vrms_max"] = float(m.group(1)) * _SI[m.group(2).lower()]
    if m := re.search(rf"power[^\n]*?({_N})\s*([munµ]?)W", text, re.I):
        kw["power_w_max"] = float(m.group(1)) * _SI[m.group(2).lower()]
    take("area_mm2_max", rf"area[^\n]*?({_N})\s*mm")

    # -- eye --------------------------------------------------------------------------
    # "height" is the VERTICAL opening (mV) and "width" the HORIZONTAL one (UI); the Spec
    # field names invert that convention (eye_v_mv / eye_h_ui), so map by unit, not name.
    if m := re.search(rf"eye[^\n]*?({_N})\s*(m?)V", text, re.I):
        kw["eye_v_mv_min"] = float(m.group(1)) * (1.0 if m.group(2) else 1e3)
    take("eye_h_ui_min", rf"eye[^\n]*?({_N})\s*UI")

    # -- environment ------------------------------------------------------------------
    # "...over a 14 dB channel": the number must sit immediately before the word, not
    # merely somewhere on the same line. A lazy gap here reads "8.9 dB boost over a 14 dB
    # channel" as an 8.9 dB channel, silently swapping the target for the channel loss.
    take("channel_loss_db", rf"({_N})\s*dB\s+(?:insertion\s*loss\s+)?channel")
    if "channel_loss_db" not in kw:
        take("channel_loss_db",
             rf"(?:channel\s*loss|insertion\s*loss|channel)\s*[:=]?\s*(?:of\s*)?"
             rf"(?:at\s*nyquist\s*)?[:=]?\s*({_N})\s*dB")
    take("vdd_nominal", rf"(?:supply|vdd|v\s*dd)[^\n]*?({_N})\s*V\b")

    return ParseResult(Spec(**kw), kw, "heuristic")


if __name__ == "__main__":
    import sys
    r = parse_spec_verbose(
        " ".join(sys.argv[1:]) or "PCIe Gen2 CTLE, ~9 dB boost, under 12 mW")
    if not r.understood:
        raise SystemExit(
            f"[{r.source}] nothing in that request was recognised as a design spec. "
            "No spec was inferred -- state a target boost in dB (and optionally a power "
            "budget in mW).")
    print(f"[{r.source}] read {len(r.recognised)} field(s):")
    for key in sorted(r.recognised):
        unit, scale = UNITS.get(key, ("", 1.0))
        print(f"  {key:20s} {r.recognised[key] * scale:>10.4g} {unit}")
