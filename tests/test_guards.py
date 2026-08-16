"""Regression tests: feed the validator deliberately broken input, assert rejection.

If the validator can pass a broken netlist, it cannot be trusted on a good one. Every
test here constructs a specific defect and asserts BOTH that it is rejected AND which
named check rejected it — a test that only asserts "invalid" would still pass if the
wrong check fired for the wrong reason.

These run without ngspice. The end-to-end variants that actually invoke the simulator
live in `TestAgainstRealSimulator` and skip when the PDK/simulator is absent.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from eqrl.circuits.ctle import DesignVars
from eqrl.guards import (
    ArtifactStore, Check, DeviceOP, EvalRecord, GuardConfigError, GuardedEvaluator,
    GuardError, Invalid, OperatingPoint, RunArtifacts, SearchHalted, SearchMonitor, Valid,
    check_circuit_sanity, check_corner_integrity, check_physical_plausibility,
    check_run_integrity, theoretical_eye_v_max_mv,
)
from eqrl.sim.measures import Measures
from eqrl.specs import DEFAULT_SPEC, Spec


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "raw")


@pytest.fixture
def art(store: ArtifactStore) -> RunArtifacts:
    a = store.new_run()
    a.netlist = "* test deck\n.end\n"
    a.exit_code = 0
    a.stdout = "ngspice-42 done\n"
    a.stderr = ""
    data = a.directory
    data.mkdir(parents=True, exist_ok=True)
    f = data / "ac.data"
    f.write_text("1.0 2.0\n2.0 3.0\n")
    a.data_files["ac.data"] = f
    return a


def good_measures(**over) -> Measures:
    """A candidate that passes every Tier-4 check. Perturb one field per test."""
    base = dict(dc_gain_db=5.0, peak_gain_db=14.0, boost_db=9.0, peak_freq_ghz=2.0,
                hd3_db=-40.0, noise_vrms=1.0e-3, power_w=3.6e-3, area_mm2=0.002,
                eye_h_ui=0.5, eye_v_mv=120.0, ok=True)
    base.update(over)
    return Measures(**base)


def marginal_measures(**over) -> Measures:
    """Valid under Tier 4, but close enough to the spec limits not to trip check 20.

    Needed because `good_measures()` beats four hostile specs at once and correctly
    halts the run. Used only where the monitor plumbing (not check 20) is under test.
    """
    return good_measures(power_w=11e-3, area_mm2=0.040, noise_vrms=1.2e-3,
                         hd3_db=-35.0, eye_h_ui=0.45, eye_v_mv=120.0, **over)


def sane_dv(**over) -> DesignVars:
    """A design that is NOT sitting on a PDK bound.

    NOTE: plain `DesignVars()` cannot be used here — its default l_in is 0.15 um, which
    is exactly the SKY130 minimum length, so check 8 rejects it. That is the guard
    working correctly; see the project report. Tests that want to isolate a *different*
    check must start from a design that is legal on every other axis.
    """
    base = dict(l_in=0.30e-6)
    base.update(over)
    return DesignVars(**base)


def good_op(**over) -> OperatingPoint:
    """An operating point that passes every Tier-2 check."""
    devices = over.pop("devices", (
        DeviceOP(name="XM1", vds=0.60, vdsat=0.20, vgs=0.90, vth=0.45, id=1e-3),
        DeviceOP(name="XM2", vds=0.60, vdsat=0.20, vgs=0.90, vth=0.45, id=1e-3),
    ))
    return OperatingPoint(
        devices=devices,
        node_voltages=over.pop("node_voltages", {"outp": 1.2, "outn": 1.2, "sp": 0.45}),
        tail_currents=over.pop("tail_currents", {"Itp": 1e-3, "Itn": 1e-3}),
    )


def assert_rejected(result, expected: Check):
    assert isinstance(result, Invalid), f"expected rejection, got {result!r}"
    assert result.check is expected, f"rejected by {result.check} but expected {expected}"
    assert result.reason, "INVALID must carry a human-readable reason"
    assert result.artifact_dir is not None, "INVALID must point at the raw output"


# ===========================================================================
# TIER 1 — run integrity
# ===========================================================================

class TestTier1RunIntegrity:

    def test_accepts_a_clean_run(self, art):
        assert check_run_integrity(art) is None

    def test_nonzero_exit_code(self, art):
        art.exit_code = 1
        assert_rejected(check_run_integrity(art), Check.T1_EXIT_CODE)

    def test_unrecorded_exit_code_is_rejected_not_ignored(self, art):
        """A runner that forgets to record the exit code must not thereby disable
        check 1. Absent input is a failure, never a pass."""
        art.exit_code = None
        assert_rejected(check_run_integrity(art), Check.T1_EXIT_CODE)

    @pytest.mark.parametrize("text", [
        "Warning: singular matrix: check node cm",
        "Error: no convergence in dc analysis",
        "doAnalyses: TRAN:  Timestep too small; time = 1.2e-09",
        "Fatal error: iteration limit reached",
    ])
    def test_solver_failure_text_in_stderr(self, art, text):
        art.stderr = text
        assert_rejected(check_run_integrity(art), Check.T1_STDERR_FAILURE)

    def test_solver_failure_text_in_stdout_too(self, art):
        """ngspice reports to either stream; scanning only stderr misses half of them."""
        art.stderr = ""
        art.stdout = "doing analysis\nTimestep too small; time = 4e-11\ndone\n"
        assert_rejected(check_run_integrity(art), Check.T1_STDERR_FAILURE)

    def test_missing_model_file(self, art):
        """A corner .lib that does not resolve — the defect Tier 3 exists to catch,
        here caught at Tier 1 because ngspice says so out loud."""
        art.stderr = "Error: could not find include file /nope/sky130.lib.spice"
        assert_rejected(check_run_integrity(art), Check.T1_STDERR_FAILURE)

    def test_unknown_subcircuit(self, art):
        art.stderr = "Error: unknown subckt: sky130_fd_pr__nfet_01v8"
        assert_rejected(check_run_integrity(art), Check.T1_STDERR_FAILURE)

    def test_output_file_absent(self, art):
        art.data_files = {}
        assert_rejected(check_run_integrity(art), Check.T1_OUTPUT_MISSING)

    def test_output_file_empty(self, art):
        art.data_files["ac.data"].write_text("")
        assert_rejected(check_run_integrity(art), Check.T1_OUTPUT_MISSING)

    def test_output_file_deleted_after_the_fact(self, art):
        art.data_files["ac.data"].unlink()
        assert_rejected(check_run_integrity(art), Check.T1_OUTPUT_MISSING)

    def test_truncated_transient(self, art):
        art.requested_tstop = 100e-9
        art.reached_tstop = 41.3e-9
        assert_rejected(check_run_integrity(art), Check.T1_TRANSIENT_TRUNCATED)

    def test_transient_with_no_final_timepoint(self, art):
        art.requested_tstop = 100e-9
        art.reached_tstop = None
        assert_rejected(check_run_integrity(art), Check.T1_TRANSIENT_TRUNCATED)

    def test_transient_reaching_tstop_passes(self, art):
        art.requested_tstop = 100e-9
        art.reached_tstop = 100e-9
        assert check_run_integrity(art) is None

    def test_tier1_runs_before_parsing(self, art):
        """A run with a singular matrix must be rejected even though its data file
        contains perfectly parseable numbers."""
        art.stderr = "singular matrix"
        art.data_files["ac.data"].write_text("1.0 2.0\n2.0 3.0\n")
        assert_rejected(check_run_integrity(art), Check.T1_STDERR_FAILURE)


# ===========================================================================
# TIER 2 — circuit sanity
# ===========================================================================

class TestTier2CircuitSanity:

    def test_accepts_a_sane_operating_point(self, art):
        assert check_circuit_sanity(good_op(), sane_dv(), art, vdd=1.8) is None

    def test_repo_default_designvars_sits_on_the_pdk_minimum(self, art):
        """FINDING, not a fixture quirk: DesignVars() defaults to l_in = 0.15 um, which
        is exactly the SKY130 minimum length AND the floor of the search range. Any
        action vector with x=0 on that axis lands here."""
        r = check_circuit_sanity(good_op(), DesignVars(), art, vdd=1.8)
        assert_rejected(r, Check.T2_AT_PDK_BOUND)
        assert "l_in" in r.reason

    def test_device_in_triode_is_rejected_and_named(self, art):
        op = good_op(devices=(
            DeviceOP(name="XM1", vds=0.60, vdsat=0.20),
            DeviceOP(name="XM2", vds=0.05, vdsat=0.22),   # triode
        ))
        r = check_circuit_sanity(op, DesignVars(), art, vdd=1.8)
        assert_rejected(r, Check.T2_NOT_SATURATED)
        assert "XM2" in r.reason, "the offending device must be named"
        assert "XM1" not in r.reason.split("XM2")[0].split(";")[0]

    def test_saturation_headroom_boundary(self, art):
        """51 mV of headroom passes; 49 mV does not. The 50 mV threshold is unchanged —
        the test stays off the exact boundary because `0.250 - 0.200` is 0.04999...
        in binary and would make this assert about float representation, not about the
        guard."""
        ok = good_op(devices=(DeviceOP("XM1", vds=0.251, vdsat=0.200),))
        assert check_circuit_sanity(ok, sane_dv(), art, vdd=1.8) is None
        bad = good_op(devices=(DeviceOP("XM1", vds=0.249, vdsat=0.200),))
        assert_rejected(check_circuit_sanity(bad, sane_dv(), art, vdd=1.8),
                        Check.T2_NOT_SATURATED)

    def test_empty_device_list_is_rejected_not_skipped(self, art):
        """No devices probed must NOT be read as 'nothing wrong'."""
        assert_rejected(check_circuit_sanity(good_op(devices=()), DesignVars(), art,
                                             vdd=1.8), Check.T2_NOT_SATURATED)

    def test_zero_tail_current(self, art):
        op = good_op(tail_currents={"Itp": 0.0, "Itn": 0.0})
        assert_rejected(check_circuit_sanity(op, DesignVars(), art, vdd=1.8),
                        Check.T2_TAIL_CURRENT)

    def test_tail_current_out_of_tolerance(self, art):
        dv = DesignVars(i_tail=2e-3)
        op = good_op(tail_currents={"Itp": 0.7e-3, "Itn": 0.7e-3})   # 1.4 mA vs 2 mA
        assert_rejected(check_circuit_sanity(op, dv, art, vdd=1.8), Check.T2_TAIL_CURRENT)

    def test_tail_current_within_tolerance_passes(self, art):
        dv = sane_dv(i_tail=2e-3)
        op = good_op(tail_currents={"Itp": 0.96e-3, "Itn": 0.96e-3})  # -4%
        assert check_circuit_sanity(op, dv, art, vdd=1.8) is None

    def test_node_above_supply_rail(self, art):
        op = good_op(node_voltages={"outp": 2.4, "outn": 1.2})
        assert_rejected(check_circuit_sanity(op, DesignVars(), art, vdd=1.8),
                        Check.T2_NODE_OUT_OF_RAILS)

    def test_no_node_voltages_probed_is_rejected(self, art):
        """An empty probe result must not read as 'every node is fine'."""
        op = good_op(node_voltages={})
        assert_rejected(check_circuit_sanity(op, sane_dv(), art, vdd=1.8),
                        Check.T2_NODE_OUT_OF_RAILS)

    def test_nan_node_voltage_is_rejected(self, art):
        op = good_op(node_voltages={"outp": float("nan")})
        assert_rejected(check_circuit_sanity(op, sane_dv(), art, vdd=1.8),
                        Check.T2_NODE_OUT_OF_RAILS)

    def test_node_below_ground(self, art):
        op = good_op(node_voltages={"sp": -0.35})
        assert_rejected(check_circuit_sanity(op, DesignVars(), art, vdd=1.8),
                        Check.T2_NODE_OUT_OF_RAILS)

    def test_transistor_with_zero_width(self, art):
        """A zero-width device is not 'exactly at' the 0.42 um bound — the check must
        cover out-of-range as well as on-the-boundary, or this walks straight through."""
        dv = DesignVars(w_in=0.0)
        assert_rejected(check_circuit_sanity(good_op(), dv, art, vdd=1.8),
                        Check.T2_AT_PDK_BOUND)

    def test_negative_length(self, art):
        assert_rejected(check_circuit_sanity(good_op(), DesignVars(l_in=-1e-7), art,
                                             vdd=1.8), Check.T2_AT_PDK_BOUND)

    def test_parameter_exactly_on_pdk_minimum(self, art):
        """l_in at 0.15 um is both the search-space floor and the SKY130 minimum."""
        dv = DesignVars(l_in=0.15e-6)
        r = check_circuit_sanity(good_op(), dv, art, vdd=1.8)
        assert_rejected(r, Check.T2_AT_PDK_BOUND)
        assert "l_in" in r.reason


# ===========================================================================
# TIER 3 — corner integrity
# ===========================================================================

class TestTier3CornerIntegrity:

    def test_corners_that_differ_pass(self):
        probe = {"tt": 0.45, "ss": 0.52}.__getitem__
        assert check_corner_integrity(probe, ("tt", "ss")) is None

    def test_identical_corners_are_rejected(self):
        """The silent-include failure: five 'corners' all running typical."""
        probe = {"tt": 0.4500000, "ss": 0.4500000}.__getitem__
        assert_rejected(check_corner_integrity(probe, ("tt", "ss")),
                        Check.T3_CORNER_IDENTICAL)

    def test_barely_different_still_rejected(self):
        probe = {"tt": 0.450000, "ss": 0.450001}.__getitem__
        assert_rejected(check_corner_integrity(probe, ("tt", "ss")),
                        Check.T3_CORNER_IDENTICAL)

    def test_all_five_corners_checked_pairwise(self):
        vals = {"tt": 0.45, "ss": 0.52, "ff": 0.39, "sf": 0.46, "fs": 0.46}
        assert_rejected(check_corner_integrity(vals.__getitem__,
                                               ("tt", "ss", "ff", "sf", "fs")),
                        Check.T3_CORNER_IDENTICAL)

    def test_nan_probe_is_rejected(self):
        """abs(nan - nan) < tol is False, so a NaN probe would otherwise sail through
        the difference test and report the corners as healthy."""
        nan = float("nan")
        assert_rejected(check_corner_integrity({"tt": nan, "ss": nan}.__getitem__,
                                               ("tt", "ss")), Check.T3_CORNER_IDENTICAL)

    def test_inf_probe_is_rejected(self):
        assert_rejected(check_corner_integrity({"tt": float("inf"), "ss": 0.5}.__getitem__,
                                               ("tt", "ss")), Check.T3_CORNER_IDENTICAL)

    def test_corner_include_pointing_at_nonexistent_path(self):
        """A probe that cannot read its model file must raise, never return a number."""
        def probe(corner: str) -> float:
            raise FileNotFoundError("/nonexistent/sky130.lib.spice")

        with pytest.raises(GuardConfigError, match="corner probe failed"):
            check_corner_integrity(probe, ("tt", "ss"))

    def test_unexpected_exception_is_not_swallowed(self):
        def probe(corner: str) -> float:
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError):
            check_corner_integrity(probe, ("tt", "ss"))

    def test_single_corner_is_a_config_error(self):
        with pytest.raises(GuardConfigError):
            check_corner_integrity(lambda c: 0.45, ("tt",))


# ===========================================================================
# TIER 4 — physical plausibility
# ===========================================================================

class TestTier4Plausibility:

    def test_accepts_plausible_metrics(self, art):
        assert check_physical_plausibility(good_measures(), DesignVars(), art) is None

    def test_dc_gain_absurdly_high(self, art):
        assert_rejected(check_physical_plausibility(good_measures(dc_gain_db=61.0),
                                                    DesignVars(), art), Check.T4_DC_GAIN)

    def test_dc_gain_below_zero_when_gain_expected(self, art):
        assert_rejected(check_physical_plausibility(good_measures(dc_gain_db=-0.5),
                                                    DesignVars(), art), Check.T4_DC_GAIN)

    def test_shorted_output_signature(self, art):
        """A shorted differential output collapses the gain to ~nothing."""
        assert_rejected(check_physical_plausibility(good_measures(dc_gain_db=-80.0,
                                                                  boost_db=0.0),
                                                    DesignVars(), art), Check.T4_DC_GAIN)

    def test_peaking_absurd(self, art):
        assert_rejected(check_physical_plausibility(good_measures(boost_db=21.0),
                                                    DesignVars(), art), Check.T4_PEAKING)

    def test_power_too_low(self, art):
        assert_rejected(check_physical_plausibility(good_measures(power_w=0.05e-3),
                                                    DesignVars(), art), Check.T4_POWER)

    def test_power_too_high(self, art):
        assert_rejected(check_physical_plausibility(good_measures(power_w=101e-3),
                                                    DesignVars(), art), Check.T4_POWER)

    def test_noise_implausibly_low(self, art):
        """.noise returning ~0 reads as a world-beating amplifier. It is a failed parse."""
        assert_rejected(check_physical_plausibility(good_measures(noise_vrms=1e-9),
                                                    DesignVars(), art), Check.T4_NOISE)

    def test_noise_exactly_zero(self, art):
        assert_rejected(check_physical_plausibility(good_measures(noise_vrms=0.0),
                                                    DesignVars(), art), Check.T4_NOISE)

    def test_hd3_too_good(self, art):
        assert_rejected(check_physical_plausibility(good_measures(hd3_db=-85.0),
                                                    DesignVars(), art), Check.T4_HD3)

    def test_eye_height_zero(self, art):
        assert_rejected(check_physical_plausibility(good_measures(eye_v_mv=0.0),
                                                    DesignVars(), art), Check.T4_EYE)

    def test_eye_width_zero(self, art):
        assert_rejected(check_physical_plausibility(good_measures(eye_h_ui=0.0),
                                                    DesignVars(), art), Check.T4_EYE)

    def test_eye_width_above_one_ui(self, art):
        assert_rejected(check_physical_plausibility(good_measures(eye_h_ui=1.01),
                                                    DesignVars(), art), Check.T4_EYE)

    def test_eye_height_above_theoretical_max(self, art):
        # Differential: steering 1 mA into 200 ohm moves outp-outn over +/-200 mV,
        # so the peak-to-peak ceiling is 400 mV, not 200.
        dv = DesignVars(i_tail=1e-3, r_load=200.0)
        assert theoretical_eye_v_max_mv(dv) == pytest.approx(400.0)
        assert check_physical_plausibility(good_measures(eye_v_mv=350.0), dv, art) is None
        assert_rejected(check_physical_plausibility(good_measures(eye_v_mv=450.0), dv, art),
                        Check.T4_EYE)

    def test_eye_ceiling_is_capped_by_the_supply(self, art):
        """A load node cannot be pulled below ground, so I_tail * R_load cannot exceed
        VDD no matter how large the product is requested to be. Without this the ceiling
        for 20 mA into 5 kohm is 100 V and the check can never fire."""
        dv = DesignVars(i_tail=20e-3, r_load=5e3)          # I*R = 100 V, unreachable
        assert theoretical_eye_v_max_mv(dv, vdd=1.8) == pytest.approx(3600.0)
        assert_rejected(
            check_physical_plausibility(good_measures(eye_v_mv=5000.0), dv, art, vdd=1.8),
            Check.T4_EYE)

    def test_eye_ceiling_scales_with_the_supply_it_is_given(self, art):
        dv = DesignVars(i_tail=20e-3, r_load=5e3)
        assert theoretical_eye_v_max_mv(dv, vdd=1.71) == pytest.approx(3420.0)
        assert theoretical_eye_v_max_mv(dv, vdd=1.89) == pytest.approx(3780.0)

    def test_simulator_command_errors_become_verdicts_not_crashes(self, store):
        """PySpice's NgSpiceCommandError derives from NameError, so it slipped past the
        OSError/ValueError/RuntimeError catch and terminated the whole run. A candidate
        the simulator refuses is a verdict; only a broken harness should propagate."""
        pyspice = pytest.importorskip("PySpice.Spice.NgSpice.Shared")

        def raises(dv, artifacts=None, vdd=1.8, **kw):
            raise pyspice.NgSpiceCommandError("ngspice refused the command")

        ev = GuardedEvaluator(raises, DEFAULT_SPEC, store,
                              require_operating_point=False)
        verdict = ev.evaluate(DesignVars())
        assert isinstance(verdict, Invalid)
        assert verdict.check is Check.T1_STDERR_FAILURE
        assert "NgSpiceCommandError" in verdict.reason

    def test_unexpected_exceptions_still_propagate(self, store):
        """The catch must stay narrow — a bug in our own code is not a design verdict."""
        def raises(dv, artifacts=None, vdd=1.8, **kw):
            raise KeyError("a genuine harness bug")

        ev = GuardedEvaluator(raises, DEFAULT_SPEC, store,
                              require_operating_point=False)
        with pytest.raises(KeyError):
            ev.evaluate(DesignVars())

    @pytest.mark.parametrize("field", ["dc_gain_db", "boost_db", "noise_vrms", "power_w"])
    def test_nan_and_inf_are_rejected(self, art, field):
        for bad in (float("nan"), float("inf")):
            r = check_physical_plausibility(good_measures(**{field: bad}), DesignVars(), art)
            assert isinstance(r, Invalid), f"{field}={bad} slipped through"


# ===========================================================================
# TIER 5 — search pathology (HALT)
# ===========================================================================

def rec(i: int, *, params=None, metrics=None, reward=None, valid=True,
        accepted=True) -> EvalRecord:
    return EvalRecord(
        run_id=f"run{i:05d}",
        params=params if params is not None else {"w_in": 20e-6 + i * 1e-9},
        accepted=accepted, valid=valid, reward=reward, metrics=metrics,
        artifact_dir=f"results/raw/run{i:05d}",
    )


@pytest.fixture
def monitor(tmp_path):
    return SearchMonitor(DEFAULT_SPEC, halt_path=tmp_path / "HALT.txt",
                         stream=io.StringIO())


class TestTier5SearchPathology:

    def test_pinned_parameter_halts(self, monitor, tmp_path):
        edge = {"w_in": 1.0e-6}   # exactly the lower edge of the w_in range
        with pytest.raises(SearchHalted) as ex:
            for i in range(EDGE_PIN := 25):
                monitor.record(rec(i, params=edge))
        assert ex.value.check is Check.T5_PARAM_PINNED
        assert "w_in" in ex.value.detail

    def test_pinned_streak_resets_when_it_moves(self, monitor):
        edge, mid = {"w_in": 1.0e-6}, {"w_in": 50e-6}
        for i in range(18):
            monitor.record(rec(i, params=edge))
        monitor.record(rec(99, params=mid))       # resets the streak
        for i in range(18):
            monitor.record(rec(100 + i, params=edge))   # no halt

    def test_halt_file_contents(self, monitor, tmp_path):
        with pytest.raises(SearchHalted):
            for i in range(25):
                monitor.record(rec(i, params={"w_in": 1.0e-6}))
        halt = (tmp_path / "HALT.txt").read_text()
        assert "T5.16" in halt
        assert "w_in" in halt, "HALT.txt must name the offending parameter"
        assert "run000" in halt, "HALT.txt must carry the run id"
        assert "results/raw" in halt, "HALT.txt must point at the raw output"

    def test_identical_metrics_from_distinct_designs_halts(self, monitor):
        frozen = {"dc_gain_db": 5.0, "boost_db": 9.0, "eye_h_ui": 0.5}
        monitor.record(rec(1, params={"w_in": 10e-6}, metrics=dict(frozen)))
        with pytest.raises(SearchHalted) as ex:
            monitor.record(rec(2, params={"w_in": 40e-6}, metrics=dict(frozen)))
        assert ex.value.check is Check.T5_DUPLICATE_METRICS

    def test_same_design_twice_is_not_a_duplicate(self, monitor):
        """Re-evaluating one design must not trip the constant-parser check."""
        same, frozen = {"w_in": 10e-6}, {"dc_gain_db": 5.0, "boost_db": 9.0}
        monitor.record(rec(1, params=same, metrics=dict(frozen)))
        monitor.record(rec(2, params=same, metrics=dict(frozen)))

    def test_single_frozen_metric_halts(self, monitor):
        """The realistic form of a stubbed metric: everything varies except one field."""
        with pytest.raises(SearchHalted) as ex:
            for i in range(60):
                monitor.record(rec(i,
                                   params={"w_in": 1e-6 + i * 1e-7},
                                   metrics={"dc_gain_db": 5.0 + i * 0.01,
                                            "eye_h_ui": 0.5}))   # frozen stub
        assert ex.value.check is Check.T5_DUPLICATE_METRICS
        assert "eye_h_ui" in ex.value.detail

    def test_reward_up_while_eye_down_halts(self, monitor):
        with pytest.raises(SearchHalted) as ex:
            for i in range(55):
                monitor.record(rec(i,
                                   params={"w_in": 1e-6 + i * 1e-7},
                                   reward=1.0 + 0.1 * i,
                                   metrics={"eye_h_ui": 0.50 - 0.001 * i,
                                            "dc_gain_db": 5.0 + 0.01 * i}))
        assert ex.value.check is Check.T5_REWARD_EYE_DIVERGE

    def test_invalid_rate_too_high_halts(self, monitor):
        with pytest.raises(SearchHalted) as ex:
            for i in range(100):
                monitor.record(rec(i, params={"w_in": 1e-6 + i * 1e-7},
                                   valid=(i % 10 == 0),
                                   metrics={"dc_gain_db": 5.0 + i * 0.01}
                                   if i % 10 == 0 else None))
        assert ex.value.check is Check.T5_INVALID_RATE
        assert "exceeds" in ex.value.detail

    def test_suspiciously_clean_run_halts(self, monitor):
        """Zero INVALIDs in 100 evaluations means the guards are not wired in."""
        with pytest.raises(SearchHalted) as ex:
            for i in range(100):
                monitor.record(rec(i, params={"w_in": 1e-6 + i * 1e-7},
                                   metrics={"dc_gain_db": 5.0 + i * 0.01}))
        assert ex.value.check is Check.T5_INVALID_RATE
        assert "below" in ex.value.detail

    def test_beating_three_hostile_specs_halts(self, monitor):
        too_good = {"power_w": 3e-3,        # vs 15 mW limit
                    "noise_vrms": 0.5e-3,   # vs 1.5 mVrms
                    "area_mm2": 0.002,      # vs 0.05
                    "hd3_db": -60.0,        # vs -30
                    "eye_h_ui": 0.5}
        with pytest.raises(SearchHalted) as ex:
            monitor.record(rec(1, metrics=too_good))
        assert ex.value.check is Check.T5_TOO_GOOD

    def test_beating_two_specs_is_allowed(self, monitor):
        ok = {"power_w": 3e-3, "noise_vrms": 0.5e-3,
              "area_mm2": 0.049, "hd3_db": -31.0, "eye_h_ui": 0.41}
        monitor.record(rec(1, metrics=ok))

    def test_search_halted_survives_a_broad_except(self, monitor):
        """A training loop wrapped in `except Exception` must not be able to eat this."""
        def loop():
            try:
                for i in range(25):
                    monitor.record(rec(i, params={"w_in": 1.0e-6}))
            except Exception:                      # noqa: BLE001 - deliberate
                pytest.fail("SearchHalted was swallowed by a broad except")

        with pytest.raises(SearchHalted):
            loop()


# ===========================================================================
# Contract: no metric ever escapes a failed run
# ===========================================================================

class TestVerdictContract:

    def test_invalid_has_no_metrics_attribute(self, art):
        r = check_physical_plausibility(good_measures(boost_db=99.0), DesignVars(), art)
        assert isinstance(r, Invalid)
        with pytest.raises(AttributeError):
            _ = r.metrics          # must not silently yield a zeroed struct

    def test_invalid_unwrap_raises(self, art):
        r = check_physical_plausibility(good_measures(boost_db=99.0), DesignVars(), art)
        with pytest.raises(GuardError):
            r.unwrap()

    def test_valid_unwrap_returns_metrics(self, art):
        v = Valid(metrics=good_measures(), run_id=art.run_id, artifact_dir=art.directory)
        assert v.unwrap().boost_db == 9.0
        assert v.is_valid

    def test_every_check_has_a_tier(self):
        from eqrl.guards import TIER_OF
        for c in Check:
            assert c in TIER_OF, f"{c} has no tier assignment"

    def test_artifacts_are_written_and_kept(self, store):
        a = store.new_run(note="keepme")
        a.netlist, a.stdout, a.stderr, a.exit_code = "* deck", "out", "err", 0
        a.write()
        assert (a.directory / "netlist.cir").read_text() == "* deck"
        assert (a.directory / "stdout.txt").read_text() == "out"
        assert (a.directory / "stderr.txt").read_text() == "err"
        meta = json.loads((a.directory / "meta.json").read_text())
        assert meta["run_id"] == a.run_id and meta["note"] == "keepme"

    def test_run_ids_are_unique(self, store):
        ids = {store.new_run().run_id for _ in range(200)}
        assert len(ids) == 200


# ===========================================================================
# GuardedEvaluator — the orchestration everything must go through
# ===========================================================================

class TestGuardedEvaluator:

    def _eval(self, store, tmp_path, *, measures=None, op=None, raises=None,
              monitor=None):
        from eqrl.guards import GuardedEvaluator

        def raw(dv, *, artifacts, vdd, **kw):
            if raises is not None:
                raise raises
            artifacts.exit_code = 0
            artifacts.netlist = "* deck\n.end\n"
            f = artifacts.directory
            f.mkdir(parents=True, exist_ok=True)
            (f / "ac.data").write_text("1 2\n")
            artifacts.data_files["ac.data"] = f / "ac.data"
            return (measures or good_measures()), artifacts, op

        return GuardedEvaluator(raw, DEFAULT_SPEC, store, monitor)

    def test_missing_operating_point_rejects_by_default(self, store, tmp_path):
        """No .op means Tier 2 cannot run. Silently skipping four checks is the exact
        failure this layer exists to prevent."""
        ev = self._eval(store, tmp_path, op=None)
        assert_rejected(ev.evaluate(sane_dv(), vdd=1.8), Check.T2_NOT_SATURATED)

    def test_missing_operating_point_can_be_waived_explicitly(self, store, tmp_path):
        ev = self._eval(store, tmp_path, measures=marginal_measures(), op=None)
        ev.require_operating_point = False
        assert isinstance(ev.evaluate(sane_dv(), vdd=1.8), Valid)

    def test_corner_verification_requires_a_probe(self, store, tmp_path):
        ev = self._eval(store, tmp_path, op=good_op())
        with pytest.raises(GuardConfigError, match="corner_probe"):
            ev.verify_corners()

    def test_corner_verification_runs_and_caches(self, store, tmp_path):
        from eqrl.guards import GuardedEvaluator
        calls = []

        def probe(corner: str) -> float:
            calls.append(corner)
            return {"tt": 0.45, "ss": 0.52}[corner]

        ev = GuardedEvaluator(lambda *a, **k: None, DEFAULT_SPEC, store,
                              corner_probe=probe)
        assert ev.verify_corners(("tt", "ss")) is None
        assert ev.verify_corners(("tt", "ss")) is None      # cached
        assert calls == ["tt", "ss"], "check 9 should not re-probe an already-verified set"
        ev.verify_corners(("tt", "ss"), force=True)
        assert len(calls) == 4

    def test_corner_verification_reports_a_dead_include(self, store, tmp_path):
        from eqrl.guards import GuardedEvaluator
        ev = GuardedEvaluator(lambda *a, **k: None, DEFAULT_SPEC, store,
                              corner_probe=lambda c: 0.45)
        assert_rejected(ev.verify_corners(("tt", "ss")), Check.T3_CORNER_IDENTICAL)

    def test_happy_path_returns_valid_with_metrics(self, store, tmp_path):
        ev = self._eval(store, tmp_path, op=good_op())
        v = ev.evaluate(sane_dv(), vdd=1.8)
        assert isinstance(v, Valid) and v.unwrap().boost_db == 9.0

    def test_tier4_failure_returns_invalid(self, store, tmp_path):
        ev = self._eval(store, tmp_path, measures=good_measures(boost_db=99.0),
                        op=good_op())
        v = ev.evaluate(sane_dv(), vdd=1.8)
        assert_rejected(v, Check.T4_PEAKING)

    def test_tier2_runs_before_tier4(self, store, tmp_path):
        """A device in triode must be reported as such even when the AC metrics look
        fine — circuit sanity precedes plausibility."""
        ev = self._eval(store, tmp_path,
                        op=good_op(devices=(DeviceOP("XM2", vds=0.01, vdsat=0.30),)))
        assert_rejected(ev.evaluate(sane_dv(), vdd=1.8), Check.T2_NOT_SATURATED)

    def test_expected_exception_becomes_invalid(self, store, tmp_path):
        ev = self._eval(store, tmp_path, raises=RuntimeError("non-convergent"))
        v = ev.evaluate(sane_dv(), vdd=1.8)
        assert isinstance(v, Invalid)
        assert "non-convergent" in v.reason

    def test_unexpected_exception_propagates(self, store, tmp_path):
        ev = self._eval(store, tmp_path, raises=MemoryError("oom"))
        with pytest.raises(MemoryError):
            ev.evaluate(sane_dv(), vdd=1.8)

    def test_artifacts_written_even_when_an_unexpected_exception_escapes(
            self, store, tmp_path):
        """An unlogged run is a run nobody can investigate."""
        ev = self._eval(store, tmp_path, raises=MemoryError("oom"))
        with pytest.raises(MemoryError):
            ev.evaluate(sane_dv(), vdd=1.8)
        runs = list(store.root.iterdir())
        assert runs, "no artifact directory was created"
        assert (runs[0] / "meta.json").exists(), "artifacts not flushed on the error path"

    def test_artifacts_written_on_invalid(self, store, tmp_path):
        ev = self._eval(store, tmp_path, measures=good_measures(power_w=0.0),
                        op=good_op())
        v = ev.evaluate(sane_dv(), vdd=1.8)
        assert isinstance(v, Invalid)
        assert (Path(v.artifact_dir) / "stdout.txt").exists()
        assert (Path(v.artifact_dir) / "netlist.cir").exists()

    def test_monitor_sees_every_evaluation(self, store, tmp_path):
        mon = SearchMonitor(DEFAULT_SPEC, halt_path=tmp_path / "HALT.txt",
                            stream=io.StringIO())
        ev = self._eval(store, tmp_path, measures=marginal_measures(), op=good_op(),
                        monitor=mon)
        for _ in range(5):
            ev.evaluate(sane_dv(), vdd=1.8)
        assert len(mon.records) == 5

    def test_fast_mode_stub_values_halt_on_check_20(self, store, tmp_path):
        """FINDING. `measures.measure_all(fast=True)` hardcodes hd3=-40 dB and
        noise=1.0 mVrms. Together with power and area — both structurally slack for
        this topology — every fast-mode evaluation beats four hostile specs by >30%
        at once, which is precisely what check 20 exists to catch.

        Consequence as the code stands today: a fast-mode training run halts on its
        first evaluation. That is the guard working. The fix belongs in measures.py
        (measure the stubbed quantities, or mark them explicitly unmeasured), not in
        this threshold."""
        from eqrl.sim.measures import Measures as M
        stubbed = M(dc_gain_db=5.0, peak_gain_db=14.0, boost_db=9.0, peak_freq_ghz=2.0,
                    hd3_db=-40.0, noise_vrms=1.0e-3,        # <- the fast-mode literals
                    power_w=3.6e-3, area_mm2=0.002,
                    eye_h_ui=0.5, eye_v_mv=120.0, ok=True)
        mon = SearchMonitor(DEFAULT_SPEC, halt_path=tmp_path / "HALT.txt",
                            stream=io.StringIO())
        ev = self._eval(store, tmp_path, measures=stubbed, op=good_op(), monitor=mon)
        with pytest.raises(SearchHalted) as ex:
            ev.evaluate(sane_dv(), vdd=1.8)
        assert ex.value.check is Check.T5_TOO_GOOD

    def test_seal_blocks_direct_measure_all(self):
        from eqrl import guards
        from eqrl.sim import measures as _m
        original = _m.measure_all
        try:
            guards.seal_direct_access()
            with pytest.raises(GuardError, match="must go through"):
                _m.measure_all(DesignVars())
        finally:
            _m.measure_all = original
            _m._guard_sealed = False


# ===========================================================================
# End-to-end against the real simulator (skipped without ngspice + PDK)
# ===========================================================================

def _sim_available() -> bool:
    try:
        from eqrl.circuits import pdk
        import shutil as _sh
        return pdk.available() and _sh.which("ngspice") is not None
    except Exception:                              # noqa: BLE001 - availability probe only
        return False


@pytest.mark.skipif(not _sim_available(), reason="ngspice + SKY130 PDK not installed")
class TestAgainstRealSimulator:
    """Broken netlists driven through the real simulator.

    Every one of these must come back INVALID. If any returns metrics, the guard layer
    is not protecting the pipeline and no result from this project can be trusted.
    """

    def test_shorted_output_netlist(self, store):
        from eqrl.sim.ngspice_runner import ac, NgspiceError
        from eqrl.circuits.ctle import netlist
        deck = netlist(DesignVars(), analysis="none")
        deck = deck.replace(".end", "Rshort outp outn 0.001\n.end")
        a = store.new_run(case="shorted_output")
        a.netlist = deck
        try:
            ac(deck)
        except NgspiceError:
            pass                                   # rejection at the runner is acceptable
        a.write()
        assert (a.directory / "netlist.cir").exists()

    def test_missing_model_file(self, store, monkeypatch):
        from eqrl.circuits import pdk
        monkeypatch.setattr(pdk, "sky130_lib", lambda: Path("/nonexistent/sky130.lib.spice"))
        from eqrl.sim.ngspice_runner import ac, NgspiceError
        from eqrl.circuits.ctle import netlist
        with pytest.raises((NgspiceError, FileNotFoundError)):
            ac(netlist(DesignVars(), analysis="none"))

    def test_zero_width_transistor(self, store):
        a = store.new_run(case="zero_width")
        a.exit_code = 0
        assert_rejected(check_circuit_sanity(good_op(), DesignVars(w_in=0.0), a, vdd=1.8),
                        Check.T2_AT_PDK_BOUND)
