"""The server's background Pareto worker: streaming, cancellation and honest failure.

Alternatives are measured AFTER `/api/pipeline/run` has already answered, which is what
keeps the primary circuit off the Pareto critical path. That design has two hazards this
file pins down: a late batch from run N must never paint itself over run N+1, and a
failure to measure alternatives must never take down a run that already succeeded.

No simulator: the worker's only measurement dependency is the PVT pool, which is stubbed.
"""
from __future__ import annotations

import dataclasses
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eqrl import pareto                                         # noqa: E402
from eqrl.specs import DEFAULT_SPEC                             # noqa: E402

BASE_DESIGN = {"w_in": 5.5784369934553225e-05, "l_in": 3.9211895465850825e-07,
               "i_tail": 0.0007122746556997299, "rs": 3287.4510782957077,
               "cs": 1.8162099438699198e-13, "r_load": 2450.2967834472656, "w_dfe": 0.0}


def measures(power, noise, area, boost):
    return {"power_w": power, "noise_vrms": noise, "area_mm2": area, "boost_db": boost}


def design(offset):
    return dict(BASE_DESIGN, rs=BASE_DESIGN["rs"] + offset)


@pytest.fixture
def srv():
    import server
    with server._state_lock:
        server._run_state.update(generation=0, pareto=None, pareto_active=False,
                                 events=[], t0=0.0, stage=None, active=False)
    return server


def primary_result():
    return {
        "design": design(0),
        "verification": {"passed": True, "guard_valid": True,
                         "measures": measures(5e-3, 5e-6, 0.05, 8.0)},
        "pvt": {"accepted": True, "status": "verified"},
        "spec": {"boost_tol_db": 1.5, "target_boost_db": 8.0, "channel_loss_db": 12.0},
        "request_id": "test_request",
    }


class StubPool:
    """Stands in for the PVT worker pool. Records calls; returns canned TT rows."""

    def __init__(self, on_evaluate=None):
        self.calls = []
        self.on_evaluate = on_evaluate

    def evaluate(self, phase, candidates, grid, spec, output, until):
        self.calls.append([c["id"] for c in candidates])
        if self.on_evaluate:
            self.on_evaluate(self)
        rows = {}
        for i, c in enumerate(candidates):
            rows[c["id"]] = [{
                "passed": True, "guard_valid": True,
                "measures": measures(1e-3 * (i + 1), 9e-6 / (i + 1), 0.01 * (i + 1), 8.0),
            }]
        return rows, {"measure_all": len(candidates), "analysis": len(candidates)}


def install(monkeypatch, srv, pool, n_candidates=10):
    """Point the worker at the stub pool and at synthetic candidate sizings."""
    monkeypatch.setattr("eqrl.pvt_workers.get_pool", lambda deadline=None: pool)
    monkeypatch.setattr(pareto, "proposals", lambda result, spec, cap: [
        dict(id=f"c{i}", origin=f"stub {i}", design=design(i + 1))
        for i in range(n_candidates)])
    monkeypatch.setattr(pareto, "verification_from_row", lambda row: row)
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=8.0)
    monkeypatch.setattr("eqrl.pipeline.spec_for", lambda *a, **k: spec)


def test_alternatives_are_published_as_they_are_measured(monkeypatch, srv):
    pool = StubPool()
    install(monkeypatch, srv, pool)
    srv._pareto_worker(primary_result(), 8.0, 12.0, None, generation=0)

    with srv._state_lock:
        block = srv._run_state["pareto"]
        assert srv._run_state["pareto_active"] is False, "worker must clear its own flag"
    assert block is not None, "measured alternatives were never published"
    assert block["evaluated"] > 0
    assert pool.calls, "the worker never measured anything"
    assert all(len(batch) <= 5 for batch in pool.calls), \
        "batches stay small so a cancelled run is not stuck behind a long measurement"


def test_each_published_circuit_names_the_quantity_it_optimizes(monkeypatch, srv):
    install(monkeypatch, srv, StubPool())
    srv._pareto_worker(primary_result(), 8.0, 12.0, None, generation=0)
    with srv._state_lock:
        items = srv._run_state["pareto"]["items"]
    assert items, "no circuits offered"
    for choice in items:
        assert choice["optimized_quantity"] in pareto.OBJECTIVES
        assert choice["label"]
        assert set(choice["objectives"]) == set(pareto.OBJECTIVES)


def test_only_the_primary_circuit_keeps_its_pvt_pass(monkeypatch, srv):
    install(monkeypatch, srv, StubPool())
    result = primary_result()
    srv._pareto_worker(result, 8.0, 12.0, None, generation=0)
    with srv._state_lock:
        items = srv._run_state["pareto"]["items"]
    for choice in items:
        expected = choice["design"] == result["design"]
        assert choice["pvt"]["accepted"] is expected, \
            "PVT acceptance must follow the exact sizing that was checked"


def test_a_superseded_run_stops_and_does_not_publish(monkeypatch, srv):
    """Generation is the cancellation token: bumping it mid-flight ends the worker."""
    def bump(pool):
        if len(pool.calls) == 1:
            with srv._state_lock:
                srv._run_state["generation"] = 99

    pool = StubPool(on_evaluate=bump)
    install(monkeypatch, srv, pool)
    srv._pareto_worker(primary_result(), 8.0, 12.0, None, generation=0)

    assert len(pool.calls) == 1, "the worker kept measuring for a run that was replaced"
    with srv._state_lock:
        assert srv._run_state["pareto"] is None, \
            "a superseded run's circuits must not appear beside the current run's"


def test_a_superseded_worker_leaves_the_new_runs_flag_alone(monkeypatch, srv):
    install(monkeypatch, srv, StubPool())
    with srv._state_lock:
        srv._run_state["generation"] = 5
        srv._run_state["pareto_active"] = True
    srv._pareto_worker(primary_result(), 8.0, 12.0, None, generation=0)
    with srv._state_lock:
        assert srv._run_state["pareto_active"] is True, \
            "the old worker cleared the new run's in-progress flag"


def test_measurement_failure_does_not_raise_into_the_run(monkeypatch, srv):
    class Broken(StubPool):
        def evaluate(self, *a, **k):
            raise RuntimeError("worker stream closed")

    install(monkeypatch, srv, Broken())
    # The primary circuit has already been returned to the caller by this point, so a
    # failure here is a missing comparison, not a failed design.
    srv._pareto_worker(primary_result(), 8.0, 12.0, None, generation=0)
    with srv._state_lock:
        assert srv._run_state["pareto_active"] is False
        events = [e for e in srv._run_state["events"] if e["kind"] == "pareto_error"]
    assert events, "a failure to compare alternatives must be reported, not swallowed"
    assert "worker stream closed" in events[0]["text"]


def test_worker_is_safe_to_run_off_the_request_thread(monkeypatch, srv):
    install(monkeypatch, srv, StubPool())
    thread = threading.Thread(target=srv._pareto_worker,
                              args=(primary_result(), 8.0, 12.0, None, 0))
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "the background worker did not finish"
