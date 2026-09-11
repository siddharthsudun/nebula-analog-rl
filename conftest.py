"""Make `src/` importable for the test suite without requiring PYTHONPATH."""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pytest


@pytest.fixture(autouse=True)
def _isolate_the_simulator_lock():
    """Give every test its own `server._run_lock`, so one test cannot wedge the next.

    `server._run_lock` is a module global guarding the single resident libngspice
    process. A request that finds the runtime unready answers 503 and, on the way out,
    `_runtime_unavailable()` schedules a warm-up; that warm-up takes this lock on a
    BACKGROUND thread and holds it for the length of a real `prepare()`. In the server
    that is the correct behaviour -- a request landing mid-warm-up gets an honest 409
    rather than racing the one simulator -- but a test process does not wait for that
    thread, so the lock stayed held and every later test that touched a pipeline endpoint
    got a 409 it never asked for.

    That is how a genuine failure in one module (`test_pvt_workers`) surfaced as an
    unrelated failure in another (`test_runtime_watchdog`), which passed on its own. It
    cost a real debugging session, so it is fixed here rather than left as folklore.

    Rebinding is the idiom `tests/test_server_locking.py` already uses. Nothing is
    weakened: each test still exercises a real acquire/release, just not one another's.
    A stranded warm-up thread keeps the old lock object and releases it whenever it
    finishes, harming nobody.
    """
    try:
        import server
    except Exception:                     # a test that does not need the dashboard
        yield
        return
    import threading
    original = server._run_lock
    server._run_lock = threading.Lock()
    try:
        yield
    finally:
        server._run_lock = original
