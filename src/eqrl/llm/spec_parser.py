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
correct answer for an unparseable request; a guess is not.

Some words are ambiguous in this circuit and you must not resolve them silently. Bare
"gain" can mean the CTLE peaking (target_boost_db) or the flat-band DC gain
(dc_gain_db_min); "loss" nearly always means the channel insertion loss
(channel_loss_db). Where you had to choose a referent, set the field AND add an
"_assumptions" object mapping that field name to one short sentence saying what you
assumed and what the user should write instead if you guessed wrong. Do not add an
entry for a field the user named unambiguously."""

#: Extraction is a small, well-specified task; it does not need the largest model, and
#: the demo wants a fast answer. Override with EQRL_SPEC_MODEL. Kept in one place so a
#: model retirement is a one-line fix rather than a grep -- the previous default here
#: named a model id that no longer exists, which nobody noticed because the LLM path has
#: never run for want of a credential.
DEFAULT_MODEL = os.getenv("EQRL_SPEC_MODEL", "claude-sonnet-5")


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
    assumptions: dict = field(default_factory=dict)

    @property
    def understood(self) -> bool:
        return bool(self.recognised)


def parse_spec(text: str, model: str | None = None) -> Spec:
    """Parse a natural-language request into a Spec via Claude.

    Convenience wrapper over `parse_spec_verbose` for callers that only need the Spec.
    Note that an unparseable request yields a DEFAULT Spec, not an error -- use
    `parse_spec_verbose` if you need to tell those two cases apart, which any interface
    that echoes the result back to a human does.
    """
    return parse_spec_verbose(text, model=model).spec


def parse_spec_verbose(text: str, model: str | None = None) -> ParseResult:
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
    try:
        msg = client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        raw = msg.content[0].text
    except Exception:
        # A missing model, an expired key, a network failure. None of those are a reason
        # to refuse a request the regex can read perfectly well -- but they ARE a reason
        # to stop claiming the answer came from an LLM, so the fallback relabels itself.
        return _heuristic(text)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        # The model answered in prose instead of JSON. That is a failed parse, not a
        # reason to silently hand back defaults.
        return ParseResult(Spec(), {}, "llm")
    try:
        fields = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return ParseResult(Spec(), {}, "llm")
    assumed = fields.pop("_assumptions", None) or {}
    kw = {k: v for k, v in fields.items() if k in Spec.__annotations__}
    # Only keep notes that belong to a field we actually kept, so a stray key in the
    # model's reply cannot put an explanation on screen for a spec that was never set.
    assumed = {k: str(v) for k, v in assumed.items() if k in kw} \
        if isinstance(assumed, dict) else {}
    return ParseResult(Spec(**kw), kw, "llm", assumed)


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

#: Spelled-out quantities, rewritten to their symbol form before any pattern runs.
#: Doing it here rather than widening thirty patterns keeps one spelling change in one
#: place, and means a phrase like "nine decibels" is matched by the same rule that
#: already handles "9 dB" instead of by a parallel set that can drift out of step.
_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}
_SPELLED_UNIT = [
    (r"\bdecibels?\b", "dB"),
    (r"\bmilli[\s-]*watts?\b", "mW"),
    (r"\bmicro[\s-]*watts?\b", "uW"),
    (r"\bwatts?\b", "W"),
    (r"\bmilli[\s-]*volts?\b", "mV"),
    (r"\bgiga[\s-]*hertz\b", "GHz"),
    (r"\bohms?\b", "Ohm"),
]


def _normalise(text: str) -> str:
    """Rewrite spelled numbers and units into the symbol forms the patterns expect."""
    import re

    for word, digit in _WORD_NUM.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.I)
    for pattern, symbol in _SPELLED_UNIT:
        text = re.sub(pattern, symbol, text, flags=re.I)
    return text
#: A same-line gap that may not contain another "dB". Anchoring a dB field on its label
#: is not enough on its own: a sentence naming two dB quantities ("8.9 dB boost over a
#: 14 dB channel") lets a lazy gap skip past the right number to the wrong one.
_GAP = r"(?:(?!dB)[^\n])*?"
_BOOST = r"\b(?:boost|boosting|peaking|peak\s*gain|equali[sz]ation|eq)\b"
#: "gain" on its own is ambiguous in this circuit: it can mean the CTLE's peaking (what
#: the search actually steers on) or the DC gain (a different Spec field, and the
#: constraint that binds at PVT). Refusing the word entirely rejects ordinary requests;
#: mapping it silently rebuilds the fabrication bug this parser was fixed to remove. So
#: it maps to the peaking target AND records an assumption the caller must show.
_GAIN = r"\b(?:gain|amplification)\b"
_DCGAIN = r"(?:\bdc[\s-]*gain\b|\blow[\s-]*frequency\s*gain\b|\bflat[\s-]*band\s*gain\b)"
#: Channel insertion loss is said many ways -- "14 dB channel", "insertion loss of 14 dB",
#: "a loss margin of 13 dB", "14 dB of attenuation". Anchor on the loss noun, not on the
#: word "channel", which most requests leave implicit.
#: The loss noun AND its verb forms -- "the channel loses 14 dB" is as common in a spoken
#: request as "14 dB of insertion loss", and only the noun was ever matched.
_LOSS = (r"\b(?:channel\s*loss|insertion\s*loss|loss\s*margin|lossy|loss(?:es)?"
         r"|lose[sd]?|losing|attenuat(?:ion|es|ed)|il)\b")
#: Units that are NOT decibels. A unit-less fallback ("channel 14") must not fire on a
#: number that already belongs to another quantity ("8 Gbps", "1.8 V", "15 mW").
_NOT_DB = r"(?!\s*(?:dB|GHz|MHz|Gbps|Gb|GT|mW|uW|nW|W\b|mV|uV|V\b|UI|mm|nm|um|fF|pF))"
#: A gap that may cross neither another dB quantity NOR the other concept's keyword.
#: Without the second guard, "9.8 dB gain but a loss margin of 13 dB" lets the channel
#: pattern reach back across "gain" and claim 9.8 as the channel loss.
_GAP_NOGAIN = r"(?:(?!dB|gain|boost|peaking|amplification)[^\n])*?"
#: The connectors that may sit between a label and the number it introduces. A label-then-
#: number rule needs a TIGHT join, not a free gap: "3 dB dc gain over a 14 dB channel"
#: lets a free gap run from "dc gain" all the way to 14 and file the channel loss as the
#: DC gain. Label-first phrasings put the number immediately after the label or behind one
#: of these words; anything longer is a different clause.
_OF = r"\s*(?:of|is|:|=|>=|>|at\s*least|minimum|min|about|around|~)?\s*"


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

    text = _normalise(text)
    kw: dict = {}
    #: Readings that required a judgement call, keyed by the field they set. Every entry
    #: here MUST reach the user: an interpretation they cannot see is a guess.
    assume: dict = {}

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
    # DC gain is resolved FIRST and in both word orders. It has to be: every phrasing of
    # it contains the word "gain", so a bare-gain rule that ran earlier would swallow
    # "3 dB DC gain" and file it as the peaking target.
    take("dc_gain_db_min", rf"({_N})\s*dB\s*(?:of\s*)?{_DCGAIN}")
    if "dc_gain_db_min" not in kw:
        take("dc_gain_db_min", rf"{_DCGAIN}{_OF}({_N})\s*dB")

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
        # Only now, with every explicit reading exhausted, fall back to a bare "gain".
        # The number must not be one already claimed as the DC gain.
        if "target_boost_db" not in kw:
            for pat in (rf"({_N})\s*dB{_GAP}{_GAIN}", rf"{_GAIN}{_GAP}({_N})\s*dB"):
                if m := re.search(pat, text, re.I):
                    v = float(m.group(1))
                    if v != kw.get("dc_gain_db_min"):
                        kw["target_boost_db"] = v
                        assume["target_boost_db"] = (
                            'read "gain" as the CTLE peaking target. If you meant the '
                            "flat-band DC gain, say \"DC gain\" -- it is a different "
                            "requirement and the search does not steer on it.")
                    break
    # Same last resort as the channel: the unit left off next to the label ("boost 9").
    if "target_boost_db" not in kw and "boost_db_min" not in kw:
        take("target_boost_db", rf"{_BOOST}\s*[:=]?\s*({_N}){_NOT_DB}")
    if "target_boost_db" not in kw and "boost_db_min" not in kw:
        take("target_boost_db", rf"({_N}){_NOT_DB}\s*(?:of\s*)?{_BOOST}")

    if m := re.search(rf"peak[^\n]*?({_N})\s*(?:GHz)?\s*(?:{_R})\s*({_N})\s*GHz", text,
                      re.I):
        kw["peak_freq_lo_ghz"], kw["peak_freq_hi_ghz"] = (float(m.group(1)),
                                                          float(m.group(2)))

    # -- hard constraints -------------------------------------------------------------
    take("hd3_db_max", rf"(?:hd3|third[- ]harmonic|linearity)[^\n]*?({_N})\s*dB")
    if m := re.search(rf"noise[^\n]*?({_N})\s*([munµ]?)V", text, re.I):
        kw["noise_vrms_max"] = float(m.group(1)) * _SI[m.group(2).lower()]
    if m := re.search(rf"power[^\n]*?({_N})\s*([munµ]?)W", text, re.I):
        kw["power_w_max"] = float(m.group(1)) * _SI[m.group(2).lower()]
    # "under 12 mW" names no field, but in this circuit a milliwatt figure is only ever
    # the power budget -- and that exact phrasing is the example in solve.py's docstring,
    # which the label-anchored rule above silently ignored.
    if "power_w_max" not in kw and (m := re.search(rf"({_N})\s*([munµ])W\b", text, re.I)):
        kw["power_w_max"] = float(m.group(1)) * _SI[m.group(2).lower()]
    # "0.01 W power budget" puts the figure ahead of its noun and uses no SI prefix, so
    # neither rule above sees it. Bare watts are only safe this close to the word.
    if "power_w_max" not in kw and (
            m := re.search(rf"({_N})\s*([munµ]?)W\b[^\n]{{0,24}}?\bpower", text, re.I)):
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
             rf"(?:channel\s*loss|insertion\s*loss|channel){_OF}"
             rf"(?:at\s*nyquist\s*)?[:=]?\s*({_N})\s*dB")
    # Most requests never say "channel" -- they say "a loss margin of 13 dB", "13 dB of
    # attenuation", "13 dB insertion loss". Anchor on the loss noun in both word orders.
    # The gaps may not cross a gain/boost word, or "9.8 dB gain but a loss margin of
    # 13 dB" reaches back over "gain" and files 9.8 as the channel loss.
    # Number-before is tried FIRST: "13 dB of attenuation with 9.8 dB gain" puts the
    # channel figure ahead of its noun, and a label-first rule would run past it to 9.8.
    if "channel_loss_db" not in kw:
        take("channel_loss_db", rf"({_N})\s*dB{_GAP_NOGAIN}{_LOSS}")
    if "channel_loss_db" not in kw:
        take("channel_loss_db", rf"{_LOSS}{_OF}({_N})\s*dB")
    # Last resort: the unit was left off entirely ("9 dB gain and 13 loss", "channel 14").
    # Only ever adjacent to the label, and only for a number that has not already claimed
    # a unit of its own -- a free unit-less rule would read the 8 in "8 Gbps" as a loss.
    if "channel_loss_db" not in kw:
        take("channel_loss_db", rf"({_N}){_NOT_DB}\s*{_LOSS}")
    if "channel_loss_db" not in kw:
        take("channel_loss_db", rf"(?:{_LOSS}|\bchannel\b)\s*[:=]?\s*({_N}){_NOT_DB}")
    # Insertion loss gets quoted both ways: "14 dB of loss" and "a -14 dB channel" name
    # the same channel. The Spec stores the magnitude, so a signed figure is not a
    # different requirement -- and a negative channel_loss_db would be nonsense.
    if "channel_loss_db" in kw:
        kw["channel_loss_db"] = abs(kw["channel_loss_db"])
    take("vdd_nominal", rf"(?:supply|vdd|v\s*dd)[^\n]*?({_N})\s*V\b")

    return ParseResult(Spec(**kw), kw, "heuristic", assume)


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
        flag = "  *" if key in r.assumptions else ""
        print(f"  {key:20s} {r.recognised[key] * scale:>10.4g} {unit}{flag}")
    for key in sorted(r.assumptions):
        print(f"\n  * {key}: {r.assumptions[key]}")
