"""Type-to-circuit: natural language -> Spec, with every reading accounted for.

"Design me a PCIe Gen2 CTLE with about 9 dB of boost, under 12 mW" -> Spec(...).

Three readers, layered so the answer is never worse than the deterministic one:

  1. `_heuristic`   label-anchored regex rules. Instant, offline, deterministic. It is the
                    floor: everything it reads is kept, and it is what the test corpus in
                    tests/test_spec_parser.py pins.
  2. LLM (API)      the Anthropic Messages API, when ANTHROPIC_API_KEY / _AUTH_TOKEN is set.
  3. LLM (CLI)      the local `claude` command in non-interactive mode, when it is on PATH
                    and the account is logged in. This is what makes the LLM path actually
                    run on the demo machine, which has never had an API key configured.

The LLM is a SECOND reader, not a replacement. Its answer is merged field by field:

  * a field only the LLM found (a typo, a spelled-out standard, a phrasing no rule knows)
    is added and labelled `llm`;
  * a field both found with the same value is confirmed (`both`);
  * a field both found with DIFFERENT values is kept at the heuristic's value and reported
    in `conflicts`, because a rule that can be read is easier to trust than a model that
    cannot, and the user sees both numbers.

Every field carries its provenance in `ParseResult.sources`, and every judgement call in
`ParseResult.assumptions`. An unreadable request returns an EMPTY `recognised`, never a
default wearing the costume of an interpretation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from eqrl.specs import Spec

_SYSTEM = """You convert an analog-equalizer (CTLE) design request into a JSON object of
target specs. Output ONLY a JSON object, no prose, no code fence. Allowed keys:
target_boost_db, boost_db_min, boost_db_max, channel_loss_db, dc_gain_db_min, power_w_max,
area_mm2_max, noise_vrms_max, hd3_db_max, data_rate_gbps, nyquist_ghz, peak_freq_lo_ghz,
peak_freq_hi_ghz, eye_h_ui_min, eye_v_mv_min, vdd_nominal.
Units are SI: power in WATTS (12 mW -> 0.012), noise in VOLTS rms (1.5 mV -> 0.0015),
area in mm^2, frequencies in GHz, data rate in Gbps, eye height in mV, eye width in UI.
Channel loss is a magnitude (a "-14 dB channel" is 14).

Rules:
- Only include a key the message actually states or clearly implies. Omit everything
  else. Do NOT invent plausible numbers to fill the schema.
- If the message contains no analog design intent at all (greeting, chatter, nonsense, a
  question about something else), return exactly {}.
- Tolerate typos and spelled-out numbers ("nine db of bost" is target_boost_db 9).
- Standards imply a data rate: PCIe Gen1 2.5, Gen2 5, Gen3 8, Gen4 16, Gen5 32 Gbps;
  USB 3.0 5, USB 3.1 10; SATA 3 6; 10GbE 10.3125. Nyquist is half the NRZ data rate.
- "boost", "peaking", "equalization", "EQ" mean target_boost_db. Bare "gain" is
  ambiguous: read it as target_boost_db AND add an "_assumptions" object mapping the
  field name to one short sentence saying what you assumed. "DC gain", "flat-band gain",
  "low-frequency gain" mean dc_gain_db_min. "loss", "insertion loss", "attenuation",
  "lossy channel" mean channel_loss_db.
- A range like "8 to 10 dB of boost" sets target_boost_db to the midpoint and records an
  assumption; "3-12 dB tunable" sets boost_db_min/boost_db_max.
- Never add an _assumptions entry for a field the user named unambiguously."""

#: Extraction is a small, well-specified task; it does not need the largest model, and
#: the demo wants a fast answer. Override with EQRL_SPEC_MODEL. Kept in one place so a
#: model retirement is a one-line fix rather than a grep.
DEFAULT_MODEL = os.getenv("EQRL_SPEC_MODEL", "claude-sonnet-5")

#: Which LLM reader to use. "auto" picks the API when a credential exists, else the local
#: CLI when one is installed, else nothing. "off" is what the test-suite uses so a corpus
#: assertion is about the rules and never about a model's mood that day.
BACKENDS = ("auto", "api", "cli", "off")

#: Wall-clock cap on the CLI reader. The dashboard waits on this synchronously; a model
#: that has not answered in this long is not going to make the parse better.
CLI_TIMEOUT_S = float(os.getenv("EQRL_SPEC_CLI_TIMEOUT", "25"))

#: Fields the merge is allowed to accept from an LLM. Anything else in its reply is
#: dropped, so a hallucinated key can never reach a Spec constructor.
SPEC_FIELDS = frozenset(k for k in Spec.__annotations__
                        if k not in ("process_corners", "temps_c", "vdd_tolerance",
                                     "boost_tol_db", "boost_target_tol_db"))

#: Sanity box per field. A reader that returns a value outside it has misread a unit
#: (a "15 W" CTLE, a "9000 dB" boost) and that value is rejected with a warning rather
#: than silently planted in a slider. Generous on purpose: this catches unit slips, not
#: unusual-but-real requests.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "target_boost_db": (0.0, 40.0), "boost_db_min": (0.0, 40.0), "boost_db_max": (0.0, 40.0),
    "channel_loss_db": (0.0, 60.0), "dc_gain_db_min": (-20.0, 30.0),
    "power_w_max": (1e-5, 1.0), "area_mm2_max": (1e-4, 10.0),
    "noise_vrms_max": (1e-6, 0.1), "hd3_db_max": (-120.0, 0.0),
    "data_rate_gbps": (0.1, 224.0), "nyquist_ghz": (0.05, 112.0),
    "peak_freq_lo_ghz": (0.01, 60.0), "peak_freq_hi_ghz": (0.01, 60.0),
    "eye_h_ui_min": (0.0, 1.0), "eye_v_mv_min": (0.0, 2000.0),
    "vdd_nominal": (0.5, 5.0),
}

#: Signalling standards people name instead of a data rate. NRZ unless noted; PAM4
#: standards are left out on purpose because Nyquist is not half the bit rate there.
STANDARDS: list[tuple[str, str, float]] = [
    (r"pci\s*-?\s*e(?:xpress)?\s*(?:gen\s*)?1\b|\bgen\s*1\b", "PCIe Gen1", 2.5),
    (r"pci\s*-?\s*e(?:xpress)?\s*(?:gen\s*)?2\b|\bgen\s*2\b", "PCIe Gen2", 5.0),
    (r"pci\s*-?\s*e(?:xpress)?\s*(?:gen\s*)?3\b|\bgen\s*3\b", "PCIe Gen3", 8.0),
    (r"pci\s*-?\s*e(?:xpress)?\s*(?:gen\s*)?4\b|\bgen\s*4\b", "PCIe Gen4", 16.0),
    (r"pci\s*-?\s*e(?:xpress)?\s*(?:gen\s*)?5\b|\bgen\s*5\b", "PCIe Gen5", 32.0),
    (r"\busb\s*3\.?0\b|\busb\s*3\b(?!\.1)", "USB 3.0", 5.0),
    (r"\busb\s*3\.1\b|\busb\s*3\.2\b", "USB 3.1", 10.0),
    (r"\bsata\s*(?:3|iii|6g)\b", "SATA 3", 6.0),
    (r"\bsata\s*(?:2|ii|3g)\b", "SATA 2", 3.0),
    (r"\b10\s*g(?:b)?e\b|\b10\s*gig(?:abit)?\s*ethernet\b|\bsfp\+", "10GbE", 10.3125),
    (r"\bdisplayport\s*1\.4\b|\bhbr3\b", "DisplayPort HBR3", 8.1),
    (r"\bhdmi\s*2\.0\b", "HDMI 2.0", 6.0),
]


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
    #: per-field provenance: "heuristic", "llm" or "both"
    sources: dict = field(default_factory=dict)
    #: field -> {"heuristic": v, "llm": v} when the two readers disagreed
    conflicts: dict = field(default_factory=dict)
    #: human-readable notes that are not tied to one field (a rejected implausible value,
    #: a target outside the circuit's range, an LLM that was tried and failed)
    warnings: list = field(default_factory=list)
    #: which LLM reader ran, if any: None, "api" or "cli"
    llm_backend: str | None = None
    #: milliseconds the LLM reader took, for the UI's own honesty about latency
    llm_ms: float | None = None

    @property
    def understood(self) -> bool:
        return bool(self.recognised)


def parse_spec(text: str, model: str | None = None) -> Spec:
    """Parse a natural-language request into a Spec.

    Convenience wrapper over `parse_spec_verbose` for callers that only need the Spec.
    Note that an unparseable request yields a DEFAULT Spec, not an error -- use
    `parse_spec_verbose` if you need to tell those two cases apart, which any interface
    that echoes the result back to a human does.
    """
    return parse_spec_verbose(text, model=model).spec


# -- LLM readers -------------------------------------------------------------------------

def _extract_json(raw: str) -> dict | None:
    """The object in a model reply, or None if there is no parseable object.

    Tolerates a code fence, prose around the object, and the unquoted-key JSON some
    models emit for tiny objects. Anything that is not an object is a failed parse.
    """
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None
    blob = raw[start:end + 1]
    try:
        out = json.loads(blob)
    except json.JSONDecodeError:
        # `{ok: true}`-style output: quote bare keys and retry once.
        try:
            out = json.loads(re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', blob))
        except json.JSONDecodeError:
            return None
    return out if isinstance(out, dict) else None


def _llm_api(text: str, model: str | None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(model=model or DEFAULT_MODEL, max_tokens=512,
                                 system=_SYSTEM,
                                 messages=[{"role": "user", "content": text}])
    return msg.content[0].text


def _cli_path() -> str | None:
    """The local `claude` command, if one is installed and not disabled."""
    if os.getenv("EQRL_SPEC_LLM", "").lower() == "off":
        return None
    explicit = os.getenv("EQRL_CLAUDE_CLI")
    if explicit and os.path.exists(explicit):
        return explicit
    for name in ("claude", "claude.exe", "claude.cmd"):
        p = shutil.which(name)
        if p:
            return p
    home = os.path.expanduser("~")
    for cand in (os.path.join(home, ".local", "bin", "claude.exe"),
                 os.path.join(home, ".local", "bin", "claude")):
        if os.path.exists(cand):
            return cand
    return None


def _llm_cli(text: str, model: str | None) -> str:
    """One non-interactive turn of the local `claude` command, JSON out.

    The CLI is run with the system prompt inlined into the user turn: `-p` mode has a
    `--system-prompt` flag but its availability differs between CLI versions, and a single
    prompt that carries both is portable across all of them. The nested `CLAUDECODE`
    variable is cleared so a parser launched from inside a Claude Code session does not
    refuse to start.
    """
    exe = _cli_path()
    if not exe:
        raise RuntimeError("claude CLI not found")
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    prompt = f"{_SYSTEM}\n\nRequest:\n{text}\n\nJSON:"
    args = [exe, "-p", prompt, "--output-format", "json"]
    m = model or os.getenv("EQRL_SPEC_CLI_MODEL", "sonnet")
    if m:
        args += ["--model", m]
    out = subprocess.run(args, capture_output=True, text=True, timeout=CLI_TIMEOUT_S,
                         env=env, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"claude CLI exit {out.returncode}: {out.stderr.strip()[:200]}")
    try:
        envelope = json.loads(out.stdout)
        if isinstance(envelope, dict) and "result" in envelope:
            return str(envelope["result"])
    except json.JSONDecodeError:
        pass
    return out.stdout


def _pick_backend(backend: str) -> str | None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "off":
        return None
    have_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if backend == "api":
        return "api" if have_key else None
    if backend == "cli":
        return "cli" if _cli_path() else None
    env_pref = os.getenv("EQRL_SPEC_LLM", "auto").lower()
    if env_pref == "off":
        return None
    if env_pref == "cli":
        return "cli" if _cli_path() else None
    if have_key:
        try:
            import anthropic  # noqa: F401
            return "api"
        except ImportError:
            pass
    return "cli" if _cli_path() else None


def _llm_fields(text: str, model: str | None, which: str) -> tuple[dict, dict]:
    """(fields, assumptions) from one LLM reader, sanitised to Spec fields only."""
    raw = _llm_api(text, model) if which == "api" else _llm_cli(text, model)
    obj = _extract_json(raw)
    if obj is None:
        raise RuntimeError("model answered in prose, not JSON")
    assumed = obj.pop("_assumptions", None) or {}
    fields: dict = {}
    for k, v in obj.items():
        if k not in SPEC_FIELDS:
            continue
        try:
            fields[k] = float(v)
        except (TypeError, ValueError):
            continue
    if "channel_loss_db" in fields:
        fields["channel_loss_db"] = abs(fields["channel_loss_db"])
    assumed = ({k: str(v) for k, v in assumed.items() if k in fields}
               if isinstance(assumed, dict) else {})
    return fields, assumed


def _plausible(kw: dict, warnings: list, label: str) -> dict:
    """Drop values outside PLAUSIBLE, saying so. Mutates nothing it was handed."""
    out = {}
    for k, v in kw.items():
        lo, hi = PLAUSIBLE.get(k, (float("-inf"), float("inf")))
        if lo <= v <= hi:
            out[k] = v
        else:
            unit, scale = UNITS.get(k, ("", 1.0))
            warnings.append(f"{label} read {k} as {v * scale:g} {unit}, which is outside "
                            f"any plausible range for this circuit; ignored.")
    return out


def parse_spec_verbose(text: str, model: str | None = None, *,
                       backend: str = "auto") -> ParseResult:
    """Parse a request and report which fields were actually recognised, and by whom.

    `backend` selects the LLM reader ("auto", "api", "cli", "off"); the heuristic always
    runs. See the module docstring for the merge rule.
    """
    import time

    h = _heuristic(text)
    warnings: list = list(h.warnings)
    kw = dict(h.recognised)
    sources = {k: "heuristic" for k in kw}
    assumptions = dict(h.assumptions)
    conflicts: dict = {}

    which = _pick_backend(backend)
    llm_ms = None
    if which is not None and text.strip():
        t0 = time.perf_counter()
        try:
            lf, la = _llm_fields(text, model, which)
        except Exception as e:  # noqa: BLE001 -- the rules still stand; say what happened
            lf, la = {}, {}
            warnings.append(f"LLM reader ({which}) unavailable: "
                            f"{type(e).__name__}: {str(e)[:120]}")
            which = None
        llm_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        lf = _plausible(lf, warnings, "The LLM reader")
        for k, v in lf.items():
            if k in kw:
                if abs(kw[k] - v) <= 1e-9 * max(1.0, abs(v)):
                    sources[k] = "both"
                else:
                    conflicts[k] = {"heuristic": kw[k], "llm": v}
            else:
                kw[k] = v
                sources[k] = "llm"
                if k in la:
                    assumptions[k] = la[k]
        # A model that agrees with an ambiguous reading does not make it unambiguous.
        for k, note in la.items():
            if k in kw and k not in assumptions and sources.get(k) == "both":
                assumptions[k] = note

    # A boost range with no explicit target: the midpoint is the only defensible single
    # number, and it is declared as a choice rather than passed off as a reading.
    if "target_boost_db" not in kw and "boost_db_min" in kw and "boost_db_max" in kw:
        kw["target_boost_db"] = 0.5 * (kw["boost_db_min"] + kw["boost_db_max"])
        sources["target_boost_db"] = sources.get("boost_db_min", "heuristic")
        assumptions["target_boost_db"] = (
            "you gave a boost range, not a target; the midpoint is used as the target. "
            "Say \"9 dB boost\" to pin it.")

    kw = _range_warnings(kw, warnings)
    source = "heuristic" if which is None else f"heuristic+{which}"
    return ParseResult(Spec(**kw), kw, source, assumptions, sources, conflicts, warnings,
                       llm_backend=which, llm_ms=llm_ms)


def _range_warnings(kw: dict, warnings: list) -> dict:
    """Say when a recognised target sits outside what the circuit family can do.

    Nothing is clamped: the number the user typed stays the number they see, and the
    warning is what tells them the search will not reach it.
    """
    if "target_boost_db" in kw:
        t = kw["target_boost_db"]
        if not (Spec.boost_db_min <= t <= Spec.boost_db_max):
            warnings.append(f"A {t:g} dB boost target is outside the 3 to 12 dB range this "
                            "CTLE is specified for; the search will stop at its limit.")
    if "channel_loss_db" in kw and not (4.0 <= kw["channel_loss_db"] <= 24.0):
        warnings.append(f"A {kw['channel_loss_db']:g} dB channel is outside the 4 to 24 dB "
                        "range this equalizer was benchmarked on.")
    return kw


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

#: Plain-English names for the fields, for any surface that lists them.
LABELS = {
    "target_boost_db": "Target boost", "boost_db_min": "Boost range, low",
    "boost_db_max": "Boost range, high", "channel_loss_db": "Channel loss at Nyquist",
    "dc_gain_db_min": "DC gain floor", "hd3_db_max": "HD3 ceiling",
    "noise_vrms_max": "Input noise ceiling", "power_w_max": "Power budget",
    "area_mm2_max": "Area budget", "eye_h_ui_min": "Eye width floor",
    "eye_v_mv_min": "Eye height floor", "data_rate_gbps": "Data rate",
    "nyquist_ghz": "Nyquist frequency", "peak_freq_lo_ghz": "Peak band, low",
    "peak_freq_hi_ghz": "Peak band, high", "vdd_nominal": "Supply voltage",
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
    (r"\bdecibels?\b|\bdee\s*bee\b", "dB"),
    (r"\bmilli[\s-]*watts?\b", "mW"),
    (r"\bmicro[\s-]*watts?\b", "uW"),
    (r"\bwatts?\b", "W"),
    (r"\bmilli[\s-]*volts?\b", "mV"),
    (r"\bgiga[\s-]*hertz\b", "GHz"),
    (r"\bgigabits?\s*(?:per|/)\s*(?:second|s)\b", "Gbps"),
    (r"\bohms?\b", "Ohm"),
    (r"\bsquare\s*millimet(?:er|re)s?\b", "mm2"),
]

#: Misspellings of the six words the rules anchor on, fixed before the rules run. Each
#: entry is a real miss from the test corpus; a general fuzzy matcher was tried and
#: rejected because "less" is one edit from "loss" and "boots" is a word.
_TYPO = [
    (r"\bbo+s+t\b|\bbost\b|\bboots\b|\bbosst\b", "boost"),
    (r"\bpeeking\b|\bpeakng\b|\bpeakin\b", "peaking"),
    (r"\bchanel\b|\bchannle\b|\bchannnel\b|\bchanell\b", "channel"),
    (r"\bgian\b|\bgane\b|\bgainn\b", "gain"),
    (r"\bequalisation\b|\bequalizaton\b|\bequalistion\b", "equalization"),
    (r"\battenuaton\b|\battentuation\b", "attenuation"),
]


def _normalise(text: str) -> str:
    """Rewrite spelled numbers, units and known typos into the forms the patterns expect."""
    for word, digit in _WORD_NUM.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.I)
    for pattern, symbol in _SPELLED_UNIT:
        text = re.sub(pattern, symbol, text, flags=re.I)
    for pattern, word in _TYPO:
        text = re.sub(pattern, word, text, flags=re.I)
    # "9dB" and "9 dB" are the same token; "9 d B" is not something anyone types.
    text = re.sub(rf"({_N})\s*d\s*b\b", r"\1 dB", text, flags=re.I)
    return text


#: A same-line gap that may not contain another "dB". Anchoring a dB field on its label
#: is not enough on its own: a sentence naming two dB quantities ("8.9 dB boost over a
#: 14 dB channel") lets a lazy gap skip past the right number to the wrong one.
_GAP = r"(?:(?!dB)[^\n])*?"
_BOOST = r"\b(?:boost|boosting|peaking|peak\s*gain|equali[sz]ation|eq|emphasis)\b"
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
#: The mirror image for the boost rules: "boost over a lossy 14 dB link" must not let the
#: boost label run forward across "lossy" and claim the channel figure.
_GAP_NOLOSS = r"(?:(?!dB|loss|lose|losing|lossy|channel|attenuat|insertion)[^\n])*?"
#: The connectors that may sit between a label and the number it introduces. A label-then-
#: number rule needs a TIGHT join, not a free gap: "3 dB dc gain over a 14 dB channel"
#: lets a free gap run from "dc gain" all the way to 14 and file the channel loss as the
#: DC gain. Label-first phrasings put the number immediately after the label or behind one
#: of these words; anything longer is a different clause.
_OF = (r"\s*(?:of|is|:|=|>=|>|<=|<|at\s*least|at\s*most|minimum|maximum|min|max|about|"
       r"around|roughly|approximately|approx\.?|~|under|below|over|above|up\s*to|"
       r"should\s*be|needs?\s*to\s*be|must\s*be|target(?:ed|ing)?)?\s*")


def _heuristic(text: str) -> ParseResult:
    """Dependency-free reader so the pipeline runs without an API key or a CLI.

    Label-anchored rather than unit-anchored. The earlier version searched for the first
    `<number> dB` anywhere in the message, which is wrong the moment a real spec sheet
    mentions dB more than once -- a request quoting boost, DC gain and HD3 would have had
    its HD3 line silently read as the boost target. Every pattern here is tied to the
    name of the thing it measures and confined to one line, so an unlabelled number is
    left unrecognised instead of being attached to whichever field came first.
    """
    raw = text
    text = _normalise(text)
    kw: dict = {}
    #: Readings that required a judgement call, keyed by the field they set. Every entry
    #: here MUST reach the user: an interpretation they cannot see is a guess.
    assume: dict = {}
    warnings: list = []

    def take(key, pattern, scale=1.0, group=1):
        if m := re.search(pattern, text, re.I):
            kw[key] = float(m.group(group)) * scale

    # -- signaling --------------------------------------------------------------------
    take("data_rate_gbps", rf"({_N})\s*Gb(?:ps|/s|it/s)")
    if "data_rate_gbps" not in kw:
        take("data_rate_gbps", rf"({_N})\s*GT/?s")
    take("nyquist_ghz", rf"nyquist[^\n]*?({_N})\s*GHz")
    if "data_rate_gbps" not in kw:
        for pat, name, gbps in STANDARDS:
            if re.search(pat, text, re.I):
                kw["data_rate_gbps"] = gbps
                assume["data_rate_gbps"] = (f"read \"{name}\" as a {gbps:g} Gbps NRZ link. "
                                            "State the data rate to override.")
                break
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

    # A boost RANGE ("3-12 dB of boost", "boost between 8 and 10 dB") sets the bounds;
    # a single value sets the target.
    if m := (re.search(rf"{_BOOST}[^\n]*?({_N})\s*(?:dB)?\s*(?:{_R}|and)\s*({_N})\s*dB",
                       text, re.I)
             or re.search(rf"({_N})\s*(?:dB)?\s*(?:{_R}|and)\s*({_N})\s*dB{_GAP}{_BOOST}",
                          text, re.I)):
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        kw["boost_db_min"], kw["boost_db_max"] = lo, hi
    else:
        # Both word orders occur -- "8.9 dB of boost" and "boost: 8.9 dB" -- so try the
        # number-before form first, then the number-after form. Both gaps are forbidden
        # from crossing another `dB`, which is what stops "8.9 dB boost over a 14 dB
        # channel" from reporting a 14 dB boost.
        take("target_boost_db", rf"({_N})\s*dB{_GAP_NOLOSS}{_BOOST}")
        if "target_boost_db" not in kw:
            take("target_boost_db", rf"{_BOOST}{_GAP_NOLOSS}({_N})\s*dB")
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
        take("target_boost_db", rf"{_BOOST}\s*[:=]?\s*(?:of\s*)?({_N}){_NOT_DB}")
    if "target_boost_db" not in kw and "boost_db_min" not in kw:
        take("target_boost_db", rf"({_N}){_NOT_DB}\s*(?:of\s*)?{_BOOST}")

    if m := re.search(rf"peak[^\n]*?({_N})\s*(?:GHz)?\s*(?:{_R}|and)\s*({_N})\s*GHz", text,
                      re.I):
        kw["peak_freq_lo_ghz"], kw["peak_freq_hi_ghz"] = (float(m.group(1)),
                                                          float(m.group(2)))

    # -- hard constraints -------------------------------------------------------------
    take("hd3_db_max", rf"(?:hd3|third[- ]harmonic|linearity)[^\n]*?({_N})\s*dBc?")
    if "hd3_db_max" in kw and kw["hd3_db_max"] > 0:
        # "HD3 below 30 dB" means -30 dBc; nobody asks for positive distortion.
        kw["hd3_db_max"] = -kw["hd3_db_max"]
        assume["hd3_db_max"] = "read the HD3 figure as a magnitude below the carrier."
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
    if "area_mm2_max" not in kw:
        take("area_mm2_max", rf"({_N})\s*mm(?:2|²|\^2|\s*sq)")

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

    kw = _plausible(kw, warnings, "The keyword reader")
    assume = {k: v for k, v in assume.items() if k in kw}
    del raw
    return ParseResult(Spec(**kw), kw, "heuristic", assume,
                       {k: "heuristic" for k in kw}, {}, warnings)


if __name__ == "__main__":
    import sys
    r = parse_spec_verbose(
        " ".join(sys.argv[1:]) or "PCIe Gen2 CTLE, ~9 dB boost, under 12 mW")
    if not r.understood:
        raise SystemExit(
            f"[{r.source}] nothing in that request was recognised as a design spec. "
            "No spec was inferred -- state a target boost in dB (and optionally a power "
            "budget in mW).")
    print(f"[{r.source}] read {len(r.recognised)} field(s)"
          + (f" in {r.llm_ms:.0f} ms of LLM time" if r.llm_ms else "") + ":")
    for key in sorted(r.recognised):
        unit, scale = UNITS.get(key, ("", 1.0))
        flag = "  *" if key in r.assumptions else ""
        print(f"  {key:20s} {r.recognised[key] * scale:>10.4g} {unit:5s} "
              f"[{r.sources.get(key, '?')}]{flag}")
    for key in sorted(r.assumptions):
        print(f"\n  * {key}: {r.assumptions[key]}")
    for key, both in r.conflicts.items():
        print(f"\n  ! {key}: rules read {both['heuristic']:g}, LLM read {both['llm']:g}; "
              "kept the rules' value")
    for w in r.warnings:
        print(f"\n  - {w}")
