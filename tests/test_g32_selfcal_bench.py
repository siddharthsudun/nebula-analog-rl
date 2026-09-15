"""The H-SC verdict must come from the preregistered table, not from the run.

No SPICE. Two things are pinned here: that `verdict()` implements the table in
`docs/PREREG_G32_SELFCAL.md` section 4 on synthetic rows whose answer is known by hand,
and that the constants the module computes with are the ones that document commits to. The
second is the one that matters -- a criterion is only preregistered if it cannot be edited
afterwards without something failing.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from silq.experiments.g32_selfcal_bench import (CAL_BUDGET, ERR_ALLOWANCE_DB,
                                                GATE_FIELDS, LOSS_ALLOWANCE,
                                                validity_gate, verdict)

ROOT = Path(__file__).resolve().parent.parent
PREREG_DOC = ROOT / "docs/PREREG_G32_SELFCAL.md"
FROZEN_ARTIFACT = ROOT / "results/g32_repair_smoke.json"


def row(spec, f_strict, c_strict, f_err, c_err, f_loose=None, c_loose=None):
    """One bench row, reduced to the fields `verdict` reads."""
    def arm(strict, err, loose):
        return {"strict_solved_at": 1 if strict else None,
                "loose_solved_at": 1 if (loose if loose is not None else strict) else None,
                "best_abs_err": err}
    return {"spec": spec, "frozen": arm(f_strict, f_err, f_loose),
            "cal": arm(c_strict, c_err, c_loose)}


def ten(**kw):
    """Ten specs, all strict-solved at 0.10 dB in both arms, before any override."""
    rows = [row(i, True, True, 0.10, 0.10) for i in range(8, 18)]
    for i, patch in kw.items():
        rows[int(i[1:])] = patch
    return rows


class TestTheVerdictTable:
    def test_identical_arms_are_equivalent_not_better(self):
        m = verdict(ten())
        assert m["verdict"] == "EQUIVALENT"
        assert m["median_delta_abs_err_db"] == 0.0

    def test_losing_one_strict_spec_is_still_equivalent(self):
        """n=10, so one flipped spec is inside the noise of a chaotic secant path."""
        m = verdict(ten(s0=row(8, True, False, 0.10, 1.60)))
        assert m["strict_frozen"] - m["strict_cal"] == LOSS_ALLOWANCE
        assert m["P1_strict"] and m["verdict"] == "EQUIVALENT"

    def test_losing_two_strict_specs_is_worse(self):
        m = verdict(ten(s0=row(8, True, False, 0.10, 1.60),
                        s1=row(9, True, False, 0.10, 1.60)))
        assert not m["P1_strict"] and m["verdict"] == "WORSE"

    def test_gaining_a_strict_spec_is_better(self):
        m = verdict(ten(s0=row(8, False, True, 1.60, 0.10)))
        assert m["strict_cal"] > m["strict_frozen"] and m["verdict"] == "BETTER"

    def test_the_error_allowance_is_a_boundary_not_a_gradient(self):
        """At exactly the allowance it passes; a hair over it fails. Both directions of
        the boundary are checked because a `<` written for a `<=` is invisible otherwise."""
        at = verdict([row(i, True, True, 0.10, 0.10 + ERR_ALLOWANCE_DB)
                      for i in range(8, 18)])
        over = verdict([row(i, True, True, 0.10, 0.10 + ERR_ALLOWANCE_DB + 1e-9)
                        for i in range(8, 18)])
        assert at["S2_err"] and at["verdict"] == "EQUIVALENT"
        assert not over["S2_err"] and over["verdict"] == "WORSE"

    def test_a_big_error_improvement_alone_is_better(self):
        m = verdict([row(i, True, True, 1.0, 1.0 - 2 * ERR_ALLOWANCE_DB)
                     for i in range(8, 18)])
        assert m["verdict"] == "BETTER"

    def test_specs_invalid_in_either_arm_are_excluded_from_the_pairing(self):
        """An unpaired spec cannot contribute a difference; it is counted in S1 instead."""
        rows = ten(s0=row(8, False, False, None, None),
                   s1=row(9, False, False, 0.10, None))
        m = verdict(rows)
        assert m["paired_n"] == 8
        assert m["valid_frozen"] == 9 and m["valid_cal"] == 8

    def test_the_median_is_over_pairs_not_over_arms(self):
        """Frozen and calibrated medians can both be X while every pair moved. The paired
        median is the one that notices."""
        rows = [row(8, True, True, 0.10, 0.40), row(9, True, True, 0.40, 0.10)]
        rows += [row(i, True, True, 0.20, 0.20) for i in range(10, 18)]
        m = verdict(rows)
        frozen_med = statistics.median([r["frozen"]["best_abs_err"] for r in rows])
        cal_med = statistics.median([r["cal"]["best_abs_err"] for r in rows])
        assert frozen_med == cal_med
        assert m["median_delta_abs_err_db"] == 0.0
        assert m["paired_deltas"]["8"] == pytest.approx(0.30)


class TestTheInternalValidityGate:
    def test_it_passes_when_the_frozen_arm_is_the_frozen_arm(self):
        froz = json.loads(FROZEN_ARTIFACT.read_text())["rows"]
        rows = [{"spec": r["spec"], "frozen": r["g32"]} for r in froz]
        ok, bad = validity_gate(rows, str(FROZEN_ARTIFACT))
        assert ok and bad == []

    def test_it_fails_on_a_single_perturbed_field(self):
        froz = json.loads(FROZEN_ARTIFACT.read_text())["rows"]
        rows = [{"spec": r["spec"], "frozen": dict(r["g32"])} for r in froz]
        rows[0]["frozen"]["n_valid"] = rows[0]["frozen"]["n_valid"] + 1
        ok, bad = validity_gate(rows, str(FROZEN_ARTIFACT))
        assert not ok and len(bad) == 1 and "n_valid" in bad[0]

    def test_it_notices_a_spec_the_artifact_does_not_contain(self):
        ok, bad = validity_gate([{"spec": 999, "frozen": {}}], str(FROZEN_ARTIFACT))
        assert not ok and "absent" in bad[0]

    def test_it_covers_every_field_the_final_comparison_gate_covers(self):
        """Read as source text, not imported: `final_comparison` performs the Windows
        ngspice bootstrap at import and is unimportable on the PDK-free CI runner. This
        assertion is about code, not environment."""
        src = (ROOT / "src/silq/experiments/final_comparison.py").read_text(
            encoding="utf-8")
        block = src[src.index("def gate("):]
        listed = block[block.index("FIELDS = ["):block.index("SOLVER = [")]
        for field in GATE_FIELDS:
            assert '"%s"' % field in listed, field


class TestItCollectsWithoutASimulator:
    """CI runs `tests/` with no PDK, no SKY130 and no Windows environment. The verdict and
    the prereg-drift checks above are the ones that most need to run there -- they are what
    stops the criterion moving after the data -- so the module holding them must import
    with nothing installed.

    This failed once, for real: the first push imported `final_comparison` at module level,
    whose `os.environ["USERPROFILE"]` bootstrap does not exist on Linux, and the whole file
    died at collection with exit code 2.
    """

    def test_the_module_imports_with_no_windows_environment(self):
        """Checked in a subprocess with USERPROFILE removed, which is what the runner has.
        In-process the answer would be a lie: another test may already have imported
        `final_comparison`, leaving it cached in `sys.modules`."""
        env = {k: v for k, v in os.environ.items() if k != "USERPROFILE"}
        env["PYTHONPATH"] = str(ROOT / "src")
        r = subprocess.run(
            [sys.executable, "-c",
             "import silq.experiments.g32_selfcal_bench as m; "
             "assert m.verdict and m.validity_gate and m.CAL_BUDGET; "
             "import sys; "
             "assert 'silq.experiments.final_comparison' not in sys.modules, "
             "'imported at module level again'"],
            env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]


class TestTheCriterionIsActuallyPreregistered:
    """If any of these fails, the code and the committed criterion have diverged, and the
    run is no longer preregistered. Fix the code, not the document."""

    @pytest.fixture
    def doc(self):
        """Read with line wrapping normalized: the document is prose, and a threshold that
        happens to straddle a line break is still the committed threshold."""
        return " ".join(PREREG_DOC.read_text(encoding="utf-8").split())

    def test_the_calibration_budget_matches_the_document(self, doc):
        assert "`%d` evaluations" % CAL_BUDGET in doc

    def test_the_error_allowance_matches_the_document(self, doc):
        assert "median Δ ≤ +%.2f dB" % ERR_ALLOWANCE_DB in doc
        assert "median Δ < −%.2f dB" % ERR_ALLOWANCE_DB in doc
        assert "%.2f dB is 10%% of the ±1.5 dB" % ERR_ALLOWANCE_DB in doc

    def test_the_loss_allowance_matches_the_document(self, doc):
        base = self._baseline()
        assert "calibrated ≥ %d (at most one lost)" % (
            base["strict"] - LOSS_ALLOWANCE) in doc
        assert "calibrated ≥ %d (at most one lost)" % (
            base["loose"] - LOSS_ALLOWANCE) in doc

    def test_the_baseline_quoted_in_the_document_is_the_committed_one(self, doc):
        """The criterion is relative to the frozen arm, so a wrong baseline in the
        document would make the thresholds wrong even with the code correct."""
        base = self._baseline()
        assert "**%d/10 strict**" % base["strict"] in doc
        assert "**%d/10 loose**" % base["loose"] in doc
        assert "median `best_abs_err` **%.4f dB**" % base["median"] in doc

    @staticmethod
    def _baseline():
        rows = json.loads(FROZEN_ARTIFACT.read_text())["rows"]
        errs = [r["g32"]["best_abs_err"] for r in rows
                if r["g32"]["best_abs_err"] is not None]
        return {"strict": sum(bool(r["g32"]["strict_solved_at"]) for r in rows),
                "loose": sum(bool(r["g32"]["loose_solved_at"]) for r in rows),
                "median": round(statistics.median(errs), 4)}
