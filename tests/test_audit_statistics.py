"""Prospective unit tests; no simulator or policy is used."""
import numpy as np
import pytest

from eqrl.experiments.audit_support import write_new
from eqrl.experiments.mode_error_report import (
    conditional_chance, distribution, paired_cluster_interval)


def test_new_artifacts_cannot_replace_existing_evidence(tmp_path):
    path = tmp_path / "evidence.json"
    write_new(path, {"value": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"value": 2})
    assert path.read_bytes() == original


def test_chance_zero_budget_is_zero_even_for_a_perfect_pool():
    result = conditional_chance([8., 8.], [0, 0], [8., 8.], draws=20)
    assert result["expected_rate"] == 0
    assert result["ci95"] == [0., 0.]


def test_chance_is_independent_attempt_formula_not_a_permutation_test():
    result = conditional_chance([8.], [2], [8., 20.], draws=20)
    assert result["expected_rate"] == pytest.approx(0.75)
    assert result["not_equivalence_test"] is True


def test_paired_bootstrap_preserves_identical_modes_as_zero_difference():
    result = paired_cluster_interval(np.zeros((32, 3)), draws=20)
    assert result["estimate"] == 0
    assert result["ci95"] == [0., 0.]
    assert result["n_spec_clusters"] == 32


def test_undefined_target_errors_are_not_imputed_as_zero():
    result = distribution([None, 1., 2., 3.])
    assert result["n"] == 3
    assert result["median"] == 2.
    assert distribution([None])["median"] is None
