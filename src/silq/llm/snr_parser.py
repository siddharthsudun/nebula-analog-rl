"""Optional external-noise intent, kept separate from circuit noise limits."""
import math
import re

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
INTENT = r"\bsnr\b|signal[ -]to[ -]noise|external\s+noise"
#: The same concept the INTENT test accepts, for the rules that read a VALUE off it.
#: Those were "snr" only, so "signal-to-noise ratio of 25 dB" switched SNR on and then
#: dropped the 25 -- the request recognised, its number not -- which lands the user in
#: Advanced being asked for a figure they had already given.
SNR_WORD = r"(?:\bsnr\b|signal[ -]to[ -]noise(?:\s+ratio)?)"
OFF = r"(?:ignore|disable|without|no|skip|exclude|don't\s+(?:use|consider)|do\s+not\s+(?:use|consider))\s+(?:the\s+)?(?:snr|signal[ -]to[ -]noise|external\s+noise)\b|\bsnr\s*(?:off|disabled)\b"
KEYS = {"mode", "input_snr_db", "value_vrms", "budget_vrms", "low_vrms", "high_vrms",
        "signal_reference", "signal_value_v", "bandwidth_hz"}


def parse_noise_intent(text, llm=None):
    t = text.lower().replace("−", "-").replace("–", "-").replace("μ", "u").replace("µ", "u")
    if re.search(OFF, t):
        return {"enabled": False, "source": "explicit_opt_out", "request": None, "warnings": []}
    if not re.search(INTENT, t):
        return {"enabled": False, "source": "absent", "request": None, "warnings": []}
    found = {}
    warnings = []
    unsupported = bool(re.search(rf"\boutput\s+{SNR_WORD}|{SNR_WORD}\s*(?:>|<|at least|at most|above|below|between)|(?:minimum|maximum|target)\s+{SNR_WORD}|{SNR_WORD}\s+target", t))
    snr = re.search(rf"(?:input\s+)?{SNR_WORD}\s*(?:of|=|:|is|at)?\s*({NUMBER})\s*db\b", t)
    snr = snr or re.search(rf"({NUMBER})\s*db\s+(?:input\s+)?{SNR_WORD}", t)
    if snr:
        found.update(mode="measured", input_snr_db=float(snr[1]))
    noise = re.search(rf"external\s+noise\s*(?:of|=|:|is|at)?\s*({NUMBER})\s*(mv|uv|v)\s*(?:rms)?", t)
    if noise:
        found.update(mode="measured", value_vrms=float(noise[1])*{"v": 1, "mv": 1e-3, "uv": 1e-6}[noise[2]])
    if re.search(rf"(?:unknown\s+(?:{SNR_WORD}|external\s+noise)|(?:{SNR_WORD}|external\s+noise)\s+(?:is\s+)?unknown)", t):
        found["mode"] = "unknown"
    signal = re.search(rf"({NUMBER})\s*(mv|v)\s*(vpp|pp|peak.to.peak)", t)
    if signal:
        found.update(signal_reference="tx_vpp", signal_value_v=float(signal[1])*(.001 if signal[2] == "mv" else 1))
    rms = re.search(rf"(?:ctle[ -]input|input)\s+signal\s*(?:of|=|:|is)?\s*({NUMBER})\s*(mv|v)\s*rms", t)
    if rms:
        found.update(signal_reference="ctle_input_vrms", signal_value_v=float(rms[1])*(.001 if rms[2] == "mv" else 1))
    band = re.search(rf"({NUMBER})\s*(mhz|ghz)?\s*(?:-|to)\s*({NUMBER})\s*(mhz|ghz)\b", t)
    if band:
        found["bandwidth_hz"] = [float(band[1])*{"mhz":1e6,"ghz":1e9}[band[2] or band[4]],
                                 float(band[3])*{"mhz":1e6,"ghz":1e9}[band[4]]]
    sources = {k: "heuristic" for k in found}
    if isinstance(llm, dict):
        for key, value in llm.items():
            if key not in KEYS:
                continue
            if key not in found:
                found[key], sources[key] = value, "llm"
            elif found[key] != value:
                warnings.append(f"SNR readers disagree on {key}; the explicit keyword value is retained.")
    if unsupported:
        found = {"mode": "measured"}
        warnings.append("Output SNR targets and SNR limits are unsupported. Enter a measured input SNR, signal reference and bandwidth in Advanced before running.")
    if found.get("mode") != "unknown":
        if not any(k in found for k in ("input_snr_db", "value_vrms", "budget_vrms", "low_vrms")):
            warnings.append("Enter measured input SNR or external noise RMS in Advanced, or explicitly choose Unknown.")
        if "signal_value_v" not in found or "bandwidth_hz" not in found:
            warnings.append("SNR needs an explicit signal amplitude/reference and measurement bandwidth. Complete these in Advanced.")
    found.setdefault("mode", "measured")
    return {"enabled": True, "source": "heuristic+llm" if "llm" in sources.values() else "heuristic",
            "sources": sources, "request": found, "warnings": warnings}


def resolve_snr_request(data, channel_loss_db):
    """Convert measured input SNR to noise at the same band and reference."""
    from silq.snr_spec import SNRRequest
    if "input_snr_db" not in data:
        return SNRRequest.from_dict(data)
    raw = dict(data)
    snr = raw.pop("input_snr_db")
    if isinstance(snr, bool) or not isinstance(snr, (int, float)) or not math.isfinite(snr) or not -100 <= snr <= 200:
        raise ValueError("input_snr_db must be a finite measured value from -100 to 200 dB")
    if any(k in raw for k in ("value_vrms", "budget_vrms", "low_vrms", "high_vrms")):
        raise ValueError("supply input SNR or external noise RMS, not both")
    if raw.get("mode") != "measured":
        raise ValueError("direct input SNR currently requires Measured mode")
    probe = SNRRequest.from_dict(dict(raw, value_vrms=0.))
    signal = probe.signal_value_v
    if probe.signal_reference == "tx_vpp":
        from silq.sim.snr_evaluation import channel_unit_rms
        signal *= channel_unit_rms(channel_loss_db, probe.bandwidth_hz)
    raw["value_vrms"] = signal * 10**(-float(snr)/20)
    return SNRRequest.from_dict(raw)
