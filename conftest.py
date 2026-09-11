"""Make `src/` importable for the test suite without requiring PYTHONPATH."""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pytest


@pytest.fixture(autouse=True)
def _no_background_warmup():
    """Keep a test's warm-up thread from reaching the test after it.

    `server._run_lock` is a module global guarding the ONE resident libngspice process.
    When a pipeline endpoint finds the runtime unready it answers 503 and, on the way out,
    `_runtime_unavailable()` schedules a warm-up; that warm-up takes the lock on a
    background thread and holds it for a real `prepare()`. In the server that is right --
    a request landing mid-warm-up gets an honest 409 instead of racing the simulator --
    but a test process does not wait for the thread, so the lock stayed held and every
    later test touching a pipeline endpoint got a 409 it never asked for. A genuine
    failure in `test_pvt_workers` therefore surfaced as an unrelated failure in
    `test_runtime_watchdog`, which passed on its own.

    Suppressing the SCHEDULER, rather than handing each test its own lock, is the fix that
    removes the thread instead of hiding it: no thread means no held lock and no test
    running concurrently with a half-built simulator. `_warm_up` itself is untouched --
    `tests/test_realtime_budgets.py` asserts on its SOURCE TEXT and never executes it, so
    nothing that checks warm-up behaviour loses coverage.
    """
    try:
        import server
    except Exception:                     # a test that does not need the dashboard
        yield
        return
    scheduled = []
    original = server._schedule_runtime_warmup
    server._schedule_runtime_warmup = lambda: scheduled.append(1)
    try:
        yield
    finally:
        server._schedule_runtime_warmup = original
