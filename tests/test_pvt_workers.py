import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from silq import pvt_workers as workers

ROOT = Path(__file__).resolve().parents[1]


def test_expired_pool_request_does_not_start_workers(monkeypatch):
    monkeypatch.setattr(workers, "_pool", None)
    monkeypatch.setattr(workers, "ResidentPool", lambda **kw: pytest.fail("started after deadline"))
    with pytest.raises(queue.Empty):
        workers.get_pool(deadline=time.monotonic() - 1)


def test_pool_lock_wait_obeys_deadline(monkeypatch):
    lock = threading.Lock()
    lock.acquire()
    monkeypatch.setattr(workers, "_pool_lock", lock)
    started = time.monotonic()
    try:
        with pytest.raises(queue.Empty):
            workers.get_pool(deadline=started + .02)
        assert time.monotonic() - started < .5
    finally:
        lock.release()


def test_startup_receives_request_deadline(monkeypatch):
    seen = []
    monkeypatch.setattr(workers, "_pool", None)
    monkeypatch.setattr(workers, "configuration_stamp", lambda: "test")
    def constructor(**kwargs):
        seen.append(kwargs["deadline"])
        raise queue.Empty()
    monkeypatch.setattr(workers, "ResidentPool", constructor)
    deadline = time.monotonic() + 1
    with pytest.raises(queue.Empty):
        workers.get_pool(deadline=deadline)
    assert seen == [deadline]


def test_busy_batch_does_not_kill_other_request():
    pool = workers.ResidentPool.__new__(workers.ResidentPool)
    pool.lock = threading.Lock()
    pool.lock.acquire()
    pool.closed = False
    try:
        with pytest.raises(queue.Empty):
            pool.evaluate("search", [], [], None, ".", time.monotonic() + .02)
        assert not pool.closed
    finally:
        pool.lock.release()


def test_corpus_cost_uses_one_measurement_per_seed():
    from silq.pipeline import _cost
    fc = SimpleNamespace(PREREG={"ppo_measure_all_per_eval": 2, "search_measure_all_per_eval": 1, "budget_measure_all": 20})
    cost = _cost(fc, 4, {"steps": []}, 3, {"measure_all": 4, "analysis": 16}, {"measure_all": 0, "analysis": 0}, corpus_seed=True)
    assert cost["optimizer_evals"] == 4
    assert cost["measure_all_charged_by_prereg"] == 4
    assert cost["measure_all_uncharged_by_prereg"] == 0


def test_subprocess_counts_propagate_to_nested_scopes():
    from silq.simcount import counting, add_worker_counts
    with counting() as outer:
        with counting() as inner:
            add_worker_counts({"measure_all": 45, "analysis": 180})
    assert outer == inner == {"measure_all": 45, "analysis": 180}


# Which corners each path certifies.
#
# This replaces `test_fastest_endpoint_keeps_full_pvt`, which asserted a contract two
# rewrites of the request path had already left behind: it monkeypatched `server.design`
# and a `pvt_wall_seconds` kwarg that `pipeline_run` stopped passing when the resident
# runtime took over, so it was pinning a claim -- "Fastest certifies all 45 corners" --
# that the code had not made for some time. Fastest in fact certified nothing at all.
#
# The agreed behaviour is now explicit: Fastest certifies tt/ss/ff so it can answer
# inside five seconds, every other mode certifies all 45, and Check PVT is always the
# full grid whatever produced the circuit. A 3-corner sweep measured 0.50 s warm against
# 3.12 s for 45 (2026-09-11, delivered sizing + anchor, both banks).


class _RecordingPool:
    """Records the grid it is handed and returns rows that match it but do not pass.

    Not passing keeps the test off `recorded_netlist` and the accepted-artifact branch --
    what is under test is the grid, not the acceptance arithmetic.
    """

    def __init__(self):
        self.grids = []

    def evaluate(self, phase, candidates, grid, spec, output, deadline):
        self.grids.append(list(grid))
        rows = {c["id"]: [dict(corner=list(corner), guard_valid=False, passed=False,
                               guard_violation=1.0, worker_pid=1 if phase == "search" else 2,
                               slacks={"x": -1.0}, target_error_db=9.9, measures={}, checks={})
                          for corner in grid]
                for c in candidates}
        return rows, {"evaluations": len(grid) * len(candidates), "measure_all": 0, "analysis": 0}


def _certify(tmp_path, **kw):
    from silq import pipeline as pl
    from silq.pvt_fast import certify
    pool = _RecordingPool()
    spec = pl.spec_for(9.0, 12.0, 1.5, None)
    design = {"w_in": 20.0, "l_in": 0.5, "r_s": 900.0, "c_s": 300e-15, "r_d": 1400.0, "i_bias": 900e-6}
    result = certify(design, spec, pool, tmp_path, time.monotonic() + 60, **kw)
    return pool, result


def test_certify_defaults_to_the_full_grid(tmp_path):
    pool, result = _certify(tmp_path)
    assert all(len(g) == 45 for g in pool.grids)
    assert result["corners_checked"] == 45 and result["grid_mode"] == "full"
    assert result["grid_label"] == "full 45-corner"


def test_certify_reduced_measures_exactly_tt_ss_ff(tmp_path):
    pool, result = _certify(tmp_path, grid_mode="reduced")
    assert pool.grids, "the pool was never asked to measure anything"
    for grid in pool.grids:
        assert [p for p, _, _ in grid] == ["tt", "ss", "ff"]
    assert result["corners_checked"] == 3
    assert result["grid_label"] == "3-corner (tt/ss/ff)"


def test_a_three_corner_result_never_says_forty_five(tmp_path):
    """The whole risk of a short grid is that it gets reported as sign-off."""
    _, result = _certify(tmp_path, grid_mode="reduced")
    assert "45" not in result["scope"], result["scope"]
    notes = []
    _certify(tmp_path / "noted", grid_mode="reduced", notify=lambda stage, text: notes.append(text))
    assert notes and "45" not in notes[0], notes


def test_fastest_is_the_only_mode_on_the_reduced_grid():
    from silq import realtime
    assert realtime.GRID_MODE.get("fastest") == "reduced"
    assert [m for m in realtime.LIMITS if realtime.GRID_MODE.get(m) == "reduced"] == ["fastest"]


def test_check_pvt_never_narrows_the_grid():
    """The button is the engineer's route back to the full sweep; it must stay full."""
    import ast
    source = (ROOT / "src" / "silq" / "runtime.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "certify"]
    assert len(calls) == 1, "the on-demand PVT path should have exactly one certify call"
    assert "grid_mode" not in {kw.arg for kw in calls[0].keywords}, (
        "the on-demand check must take certify's full-grid default")


def test_windows_job_kills_launcher_descendants(tmp_path):
    import os, subprocess, sys
    if os.name != "nt":
        pytest.skip("Windows process ownership")
    import ctypes
    from ctypes import wintypes
    marker = tmp_path / "child_pid"
    code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    job = workers.WindowsJob()
    proc = subprocess.Popen([sys.executable, "-c", code, str(marker)], creationflags=subprocess.CREATE_NO_WINDOW | 4)
    handle = None
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        job.attach_and_resume(proc)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        handle = api.OpenProcess(0x100000, False, int(marker.read_text()))
        assert handle
        job.close()
        assert api.WaitForSingleObject(handle, 2000) == 0
        proc.wait(timeout=2)
    finally:
        job.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if handle:
            api.CloseHandle(handle)
