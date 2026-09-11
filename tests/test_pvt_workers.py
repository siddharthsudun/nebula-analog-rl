import queue
import threading
import time
from types import SimpleNamespace
import pytest
from eqrl import pvt_workers as workers


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
    from eqrl.pipeline import _cost
    fc = SimpleNamespace(PREREG={"ppo_measure_all_per_eval": 2, "search_measure_all_per_eval": 1, "budget_measure_all": 20})
    cost = _cost(fc, 4, {"steps": []}, 3, {"measure_all": 4, "analysis": 16}, {"measure_all": 0, "analysis": 0}, corpus_seed=True)
    assert cost["optimizer_evals"] == 4
    assert cost["measure_all_charged_by_prereg"] == 4
    assert cost["measure_all_uncharged_by_prereg"] == 0


def test_subprocess_counts_propagate_to_nested_scopes():
    from eqrl.simcount import counting, add_worker_counts
    with counting() as outer:
        with counting() as inner:
            add_worker_counts({"measure_all": 45, "analysis": 180})
    assert outer == inner == {"measure_all": 45, "analysis": 180}


def test_fastest_endpoint_keeps_full_pvt(monkeypatch):
    import server
    calls = []
    monkeypatch.setattr(server, "design", lambda *a, **kw: calls.append(kw) or {"status": "pvt_not_verified"})
    monkeypatch.setattr(server, "describe", lambda result: "test")
    result = server.pipeline_run(server.PipelineRunRequest(target_boost_db=9, channel_loss_db=12, mode="fastest"))
    assert result["status"] == "pvt_not_verified"
    assert calls[0]["pvt"] is True and calls[0]["pvt_wall_seconds"] == 10


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
