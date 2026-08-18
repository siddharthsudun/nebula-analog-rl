"""Target specifications for the equalizer, as code.

A `Spec` is the input to the whole framework: the RL reward is computed against it,
and the LLM wrapper produces one from natural language.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    """PCIe Gen 2 receiver equalizer target (see docs/PROBLEM.md)."""

    # signaling
    data_rate_gbps: float = 5.0
    nyquist_ghz: float = 2.5

    # peaking
    boost_db_min: float = 3.0
    boost_db_max: float = 12.0
    peak_freq_lo_ghz: float = 1.25
    peak_freq_hi_ghz: float = 2.5

    #: OPT-IN floor on the DC gain, in dB. `None` (the default) means `hard_pass` does
    #: not look at DC gain at all, which is the behaviour every published number in this
    #: repo was produced under.
    #:
    #: WHY IT EXISTS. `boost_db` is a RATIO — peak gain minus DC gain — so a design that
    #: ATTENUATES at DC inflates its boost for free, without the peak ever getting
    #: higher. Nothing in the eight hard checks notices: `boost_range` and `peak_in_band`
    #: are both computed from that difference, and the remaining six (HD3, noise, power,
    #: area, eye) are all satisfied more easily by a stage that passes less signal.
    #:
    #: MEASURED (results/pass_vs_valid.json, CMA-ES over 4 specs x 60 evaluations): of
    #: the 28 designs that passed all eight hard specs, 24 (86%) were rejected by the
    #: guard layer, and 14 of those 24 were rejected specifically as
    #: T4.10_dc_gain_implausible — DC gain below guards.DC_GAIN_DB_MIN. The remaining
    #: 10 were T2.5_mosfet_not_in_saturation. So the attenuate-at-DC trick is the single
    #: largest source of designs that pass the spec while not being an amplifier.
    #:
    #: The guard already catches this, but the guard is not what the competition scores
    #: against — the spec is. Setting this field closes the hole in the spec itself.
    #: Setting it to 0.0 matches guards.DC_GAIN_DB_MIN (do not read that constant from
    #: here: guard thresholds are the project owners' to set, and a spec that silently
    #: tracked them would move whenever they did).
    dc_gain_db_min: float | None = 0.0

    # constraints (hard limits)
    hd3_db_max: float = -30.0          # linearity: HD3 must be below this
    noise_vrms_max: float = 1.5e-3     # input-referred, 10 MHz - 5 GHz
    power_w_max: float = 15e-3
    area_mm2_max: float = 0.05

    # eye
    eye_h_ui_min: float = 0.4
    eye_v_mv_min: float = 100.0

    # the channel the equalizer must open (insertion loss at Nyquist, dB)
    channel_loss_db: float = 12.0

    # the specific boost we are targeting this run (within [min, max]) and its tolerance
    target_boost_db: float = 9.0
    boost_tol_db: float = 1.5

    # PVT corners to enforce
    process_corners: tuple[str, ...] = ("tt", "ss", "ff", "sf", "fs")
    vdd_nominal: float = 1.8
    vdd_tolerance: float = 0.05
    temps_c: tuple[float, ...] = (0.0, 27.0, 125.0)

    def vdd_corners(self) -> tuple[float, ...]:
        v = self.vdd_nominal
        return (v * (1 - self.vdd_tolerance), v, v * (1 + self.vdd_tolerance))


def hard_pass(m, spec: "Spec") -> tuple[bool, dict]:
    """Poster's HARD spec compliance (pass/fail), distinct from the training-reward
    tolerance. Boost is the tunable 3-12 dB *range*, not a per-corner target tolerance.

    Returns (all_pass, per-check dict).

    Eight checks, always. A ninth, `dc_gain`, appears ONLY when `spec.dc_gain_db_min` is
    set — see the field's docstring for why it is off by default and what it closes. The
    key is absent, not False, when the field is None, so callers that count the dict
    (honest_benchmark's dense score does) see exactly the same eight entries as before.
    """
    if not getattr(m, "ok", False):
        return False, {"sim_ok": False}
    checks = {
        "boost_range": spec.boost_db_min <= m.boost_db <= spec.boost_db_max,
        "peak_in_band": spec.peak_freq_lo_ghz <= m.peak_freq_ghz <= spec.peak_freq_hi_ghz,
        "hd3": m.hd3_db < spec.hd3_db_max,
        "noise": m.noise_vrms < spec.noise_vrms_max,
        "power": m.power_w < spec.power_w_max,
        "area": m.area_mm2 < spec.area_mm2_max,
        "eye_h": m.eye_h_ui >= spec.eye_h_ui_min,
        "eye_v": m.eye_v_mv >= spec.eye_v_mv_min,
    }
    if spec.dc_gain_db_min is not None:
        checks["dc_gain"] = m.dc_gain_db >= spec.dc_gain_db_min
    return all(checks.values()), checks


# convenience default used across the repo
DEFAULT_SPEC = Spec()
