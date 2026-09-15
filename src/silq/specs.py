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

    #: Floor on the DC gain, in dB. Defaults to `0.0` (the DC-gain check is ON). Set to
    #: `None` to make `hard_pass` ignore DC gain entirely — the behaviour the earliest
    #: numbers in this repo (the 86% pass-vs-valid finding) were produced under, before the
    #: floor was added to the spec. The delivered numbers are scored with the floor on.
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

    #: OPT-IN tolerance, in dB, on hitting `target_boost_db`. `None` (the default) means
    #: `hard_pass` does not look at the target at all and boost is judged only against the
    #: 3-12 dB range, which is how every result before 22 Aug 2026 was scored.
    #:
    #: WHY IT EXISTS. The submission claims RETARGETING: give the system a spec, get a
    #: design meeting THAT spec. Nothing in the eight hard checks ever referenced
    #: `target_boost_db`, so any design anywhere in a 9 dB window passed every target. The
    #: consequences were measured, not suspected: one fixed design passes 32/32 held-out
    #: specs, and re-simulating the 26 designs a trained policy solved gives a correlation
    #: between REQUESTED and ACHIEVED boost of 0.114 -- no steering at all -- with the
    #: policy scoring the same 4/32 within +/-0.5 dB as the fixed design that ignores the
    #: spec entirely (results/target_tracking_clean40k.json).
    #:
    #: Setting this makes the target a spec like any other: it appears here AND as a margin
    #: in envs.equalizer_env._margins, so the reward optimises exactly what is scored. It
    #: is opt-in for the same reason `dc_gain_db_min` is -- turning it on silently would
    #: change what every existing artifact means. The key is absent, not False, when None.
    boost_target_tol_db: float | None = None

    # PVT corners to enforce
    process_corners: tuple[str, ...] = ("tt", "ss", "ff", "sf", "fs")
    vdd_nominal: float = 1.8
    vdd_tolerance: float = 0.05
    temps_c: tuple[float, ...] = (0.0, 27.0, 125.0)

    def vdd_corners(self) -> tuple[float, ...]:
        v = self.vdd_nominal
        return (v * (1 - self.vdd_tolerance), v, v * (1 + self.vdd_tolerance))


def hard_pass(m, spec: "Spec") -> tuple[bool, dict]:
    """Poster's HARD spec compliance (pass/fail).

    Boost is the tunable 3-12 dB *range*, not a per-target tolerance -- UNLESS
    `spec.boost_target_tol_db` is set, which adds a tenth check requiring the measured
    boost to land within that tolerance of `target_boost_db`. See that field for why.

    Returns (all_pass, per-check dict).

    Eight checks, always. A ninth, `dc_gain`, appears ONLY when `spec.dc_gain_db_min` is
    set. The key is absent, not False, when the field is None, so a caller that counts the
    dict sees eight entries in that case.

    IT IS SET BY DEFAULT, and this paragraph used to say otherwise. `dc_gain_db_min`
    entered as `None` in f600d2a86 (17 Aug 2026), and 2f3ec52b3 (18 Aug 2026, "The reward
    never scored DC gain, so the policy learned to attenuate. Fix it.") changed the field's
    default to 0.0 without updating this text; the description of the default was stale
    from that day until it was corrected on 07 Sep 2026. The CODE was right throughout --
    the flip was the deliberate fix for the attenuate-at-DC hole the field's own docstring
    documents. Only this comment was wrong.

    So the count callers see at DEFAULT_SPEC is NINE, not eight. WHICH call sites that
    actually affects was audited on 07 Sep 2026, one file at a time, because an earlier
    draft of this paragraph named six and two of them were wrong:

      NINE (build their spec as `replace(DEFAULT_SPEC, target/channel)`, so they inherit
      the 0.0 floor): final_comparison.py:427, hybrid_audit.py:151, search_audit.py:117.

      EIGHT, deliberately: honest_benchmark.py -- `_DC_GAIN_DB_MIN` defaults to None
      (:70-73) and threads through at :110, with runs that set it written to a separate
      `_dcfloor` filename (:233). pass_vs_valid.py, which pins None explicitly; see the
      note there for why the pin exists and what it repairs.

      NOT IN THIS FAMILY AT ALL: diagnose_policy.py:51. Its `checks` is a
      `collections.Counter` of guard-rejection reasons and it never calls `hard_pass`.
      `sum(checks.values())` there counts STEPS, not passed checks. It was listed here in
      error; the name collision is the whole trap.

    The offset is not a constant you can subtract, which is why the audit was per-file
    rather than arithmetic. `dc_gain` passes on settled designs -- 0 failures in 3735
    PVT-audit corners -- but it is exactly what discriminates against attenuating designs
    during the SEARCH, which is what 2f3ec52b3 turned it on for. Reading that 0% failure
    rate as "inert" inverts the result and would invite deleting the check that stopped
    the policy attenuating.
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
    # DO NOT DELETE THIS AS DEAD WEIGHT. It fails 0 times in the 3735-corner PVT audit,
    # and that statistic is the check WORKING, not the check being inert: by the time a
    # design reaches a settled audit this check has already excluded the attenuating ones.
    # Where it does work is the SEARCH, where those designs are generated -- 14 of the 28
    # spec-passing designs in results/pass_vs_valid.json have DC gain below 0 dB. Removing
    # it restores the hole 2f3ec52b3 closed, and the PVT table would not show that.
    if spec.dc_gain_db_min is not None:
        checks["dc_gain"] = m.dc_gain_db >= spec.dc_gain_db_min
    if spec.boost_target_tol_db is not None:
        checks["boost_target"] = (abs(m.boost_db - spec.target_boost_db)
                                  <= spec.boost_target_tol_db)
    return all(checks.values()), checks


# convenience default used across the repo
DEFAULT_SPEC = Spec()
