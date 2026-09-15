"""Locking contracts for dashboard simulator-backed endpoints.

These use the route function directly instead of starting FastAPI's lifespan: the lifespan
intentionally starts a real background warm-up, while these tests must prove only the
endpoint's nonblocking lock behaviour with a fake evaluator.
"""
from __future__ import annotations

import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import server as dashboard_server
from silq.guards import Check, SearchHalted

ROOT = Path(__file__).resolve().parents[1]

REQUEST = {
    "fields": {
        "w_in": 10.0,
        "l_in": 0.15,
        "i_tail": 100.0,
        "rs": 1.0,
        "cs": 100.0,
        "r_load": 1_000.0,
        "w_dfe": 0.0,
    },
}


def _module_lock_assignments() -> list[ast.stmt]:
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    assignments = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "_run_lock"
               for target in targets):
            assignments.append(statement)
    return assignments


def _request() -> object:
    return dashboard_server.EvaluateRequest(**REQUEST)


class _InvalidVerdict:
    is_valid = False
    tier = 2
    check = SimpleNamespace(value="test_fake_invalid")
    reason = "fake evaluator result"
    violation = 0.0
    run_id = "test-run"
    artifact_dir = Path("results/raw/test-run")


class _BlockingEvaluator:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def evaluate(self, _dv, *, vdd: float) -> _InvalidVerdict:
        self.entered.set()
        assert self.release.wait(timeout=1.0), "the test did not release the fake evaluator"
        return _InvalidVerdict()


def test_server_has_exactly_one_module_level_simulator_lock():
    """A later assignment would silently route early and late endpoints to different locks."""
    assignments = _module_lock_assignments()
    assert len(assignments) == 1, (
        f"server.py has {len(assignments)} module-level `_run_lock` assignments at "
        f"{[a.lineno for a in assignments]}; every simulator-backed endpoint must share "
        "one lock.")


def test_concurrent_guard_evaluation_returns_structured_pipeline_busy(monkeypatch):
    """The second guard endpoint call fails immediately while the first owns the lock."""
    evaluator = _BlockingEvaluator()
    monkeypatch.setattr(dashboard_server, "_run_lock", threading.Lock())
    monkeypatch.setitem(dashboard_server._startup, "warming", False)

    def fake_get_evaluator(*, corner: str, fast: bool, channel_loss_db: float = 12.0):
        assert corner == "tt"
        assert fast is False
        assert channel_loss_db == 12.0
        return evaluator

    monkeypatch.setattr(dashboard_server, "get_evaluator", fake_get_evaluator)
    first: list[object] = []
    worker = threading.Thread(target=lambda: first.append(dashboard_server.guard_evaluate(
        _request())), daemon=True)
    worker.start()
    assert evaluator.entered.wait(timeout=1.0), "first guard request never entered the fake"

    try:
        blocked = dashboard_server.guard_evaluate(_request())
        assert blocked.status_code == 409
        assert json.loads(blocked.body) == {
            "error_code": "pipeline_busy",
            "error_number": 202,
            "label": "Error 202",
            "title": "A simulator-backed operation is already in progress",
            "detail": "This server runs one simulator-backed operation at a time because "
                      "the resident ngspice process is not reentrant.",
            "hint": "Wait for the operation already in progress to finish, then try again.",
        }
    finally:
        evaluator.release.set()
        worker.join(timeout=1.0)

    assert not worker.is_alive(), "the first guard request did not finish after release"
    assert first and first[0]["valid"] is False


def test_search_halted_releases_guard_endpoint_lock(monkeypatch, tmp_path):
    """SearchHalted is BaseException; guard_evaluate's finally must still release."""
    monkeypatch.setattr(dashboard_server, "_run_lock", threading.Lock())

    class HaltedEvaluator:
        def evaluate(self, _dv, *, vdd: float):
            raise SearchHalted(Check.T5_INVALID_RATE, "test halt", tmp_path / "HALT.txt")

    monkeypatch.setattr(dashboard_server, "get_evaluator",
                        lambda **_kwargs: HaltedEvaluator())

    halted = dashboard_server.guard_evaluate(_request())
    assert halted.status_code == 409
    assert json.loads(halted.body)["error_code"] == "search_halted"
    assert dashboard_server._run_lock.acquire(blocking=False), (
        "SearchHalted escaped the inner try but guard_evaluate did not release `_run_lock`")
    dashboard_server._run_lock.release()
