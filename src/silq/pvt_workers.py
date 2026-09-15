"""Resident, process-isolated PVT workers. Reuse models, never measurement results.

Search and acceptance use disjoint workers. Every batch still measures the exact
requested design/corner grid with fast=False and retains the ordinary artifacts.
Cold startup is reported separately; warm request latency is not cold latency.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


class WorkerFailure(RuntimeError):
    pass


class WindowsJob:
    """Own the launcher and every descendant; close kills the complete worker tree."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        self.c = c
        self.api = c.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.api.CloseHandle.argtypes = [w.HANDLE]
        class Basic(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                        ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                        ("active", w.DWORD), ("affinity", c.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class Extended(c.Structure):
            _fields_ = [("basic", Basic), ("io", c.c_uint64 * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(info), c.sizeof(info)):
            self.api.CloseHandle(self.handle)
            raise c.WinError(c.get_last_error())

    def attach_and_resume(self, proc):
        # The launcher starts suspended so it cannot spawn an unowned child.
        if not self.api.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise self.c.WinError(self.c.get_last_error())
        resume = self.c.WinDLL("ntdll").NtResumeProcess
        resume.argtypes = [self.c.c_void_p]
        resume.restype = self.c.c_long
        if resume(int(proc._handle)) != 0:
            raise WorkerFailure("Could not resume PVT worker")

    def close(self):
        if self.handle:
            self.api.TerminateJobObject(self.handle, 1)
            self.api.CloseHandle(self.handle)
            self.handle = None


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise queue.Empty()
    return seconds


@contextmanager
def deadline_lock(lock, deadline):
    if not lock.acquire(timeout=remaining(deadline)):
        raise queue.Empty()
    try:
        remaining(deadline)
        yield
    finally:
        lock.release()


def _reader(stream, replies):
    try:
        for line in stream:
            replies.put(json.loads(line))
    except Exception as exc:
        replies.put({"error": repr(exc)})
    finally:
        replies.put({"error": "PVT worker stream closed"})


class ResidentPool:
    def __init__(self, phases=("search", "verify"), lanes=3, deadline=None):
        if not 1 <= lanes <= 9:
            raise ValueError("PVT lanes must be between one and nine per process")
        self.lanes = lanes
        self.workers = {}
        self.lock = threading.Lock()
        self.phase_locks = {phase: threading.Lock() for phase in phases}
        self.closed = False
        self.startup_seconds = None
        self.job = WindowsJob() if os.name == "nt" else None
        self.log_dir = ROOT / "work" / "pvt_workers" / uuid.uuid4().hex
        self.log_dir.mkdir(parents=True)
        started = time.monotonic()
        deadline = min(started + 240, deadline) if deadline is not None else started + 240
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = "1"
        try:
            for phase in phases:
                for process in ("tt", "ss", "ff", "sf", "fs"):
                    for lane in range(lanes):
                        remaining(deadline)
                        key = (phase, process, lane)
                        log = (self.log_dir / f"{phase}_{process}_{lane}.log").open("w", encoding="utf-8")
                        proc = subprocess.Popen(
                            [sys.executable, "-u", "-m", "silq.pvt_workers", "--worker", process,
                             "--store", str(self.log_dir / f"gate_{phase}_{process}_{lane}")],
                            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=log, text=True, encoding="utf-8", bufsize=1,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | (4 if self.job else 0))
                        replies = queue.Queue()
                        threading.Thread(target=_reader, args=(proc.stdout, replies), daemon=True).start()
                        self.workers[key] = (proc, replies, log)
                        if self.job:
                            self.job.attach_and_resume(proc)
            for key, (_, replies, _) in self.workers.items():
                ready = replies.get(timeout=remaining(deadline))
                if ready.get("ready") is not True or ready.get("corner") != key[1]:
                    raise WorkerFailure(f"PVT startup {key}: {ready}")
            self.startup_seconds = time.monotonic() - started
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        if self.job:
            self.job.close()
        for proc, _, log in self.workers.values():
            if proc.poll() is None:
                proc.kill()
        for proc, _, log in self.workers.values():
            proc.wait()
            if proc.stdin:
                proc.stdin.close()
            if proc.stdout:
                proc.stdout.close()
            log.close()

    def evaluate(self, phase, candidates, grid, spec, store, deadline):
        with deadline_lock(getattr(self, "phase_locks", {}).get(phase, self.lock), deadline):
            if self.closed:
                raise WorkerFailure("PVT pool is closed")
            ids = [c["id"] for c in candidates]
            if len(ids) != len(set(ids)) or len(grid) != len(set(map(tuple, grid))):
                raise ValueError("duplicate PVT candidate or corner")
            out = {cid: [] for cid in ids}
            batches = {}
            for process in dict.fromkeys(c[0] for c in grid):
                corners = [c for c in grid if c[0] == process]
                if len(corners) >= self.lanes:
                    for lane in range(self.lanes):
                        batches[(phase, process, lane)] = (candidates, corners[lane::self.lanes])
                else:
                    # A nominal-only batch spreads different designs across TT lanes.
                    for corner_index, corner in enumerate(corners):
                        lanes = list(range(corner_index, self.lanes, len(corners)))
                        for offset, lane in enumerate(lanes):
                            chunk = candidates[offset::len(lanes)]
                            if chunk:
                                batches[(phase, process, lane)] = (chunk, [corner])
            pending = []
            totals = {"evaluations": 0, "measure_all": 0, "analysis": 0}
            try:
                for key, (chunk, shard) in batches.items():
                    remaining(deadline)
                    proc, replies, _ = self.workers[key]
                    request_id = uuid.uuid4().hex
                    payload = dict(request_id=request_id, candidates=chunk, grid=shard,
                                   spec=asdict(spec), store=str(Path(store).resolve()))
                    proc.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
                    proc.stdin.flush()
                    pending.append((key, proc.pid, replies, request_id, shard, [c["id"] for c in chunk]))
                for key, pid, replies, request_id, shard, chunk_ids in pending:
                    reply = replies.get(timeout=remaining(deadline))
                    if reply.get("request_id") != request_id or "error" in reply:
                        raise WorkerFailure(f"PVT batch {key}: {reply}")
                    if set(reply["out"]) != set(chunk_ids):
                        raise WorkerFailure("PVT worker returned wrong candidates")
                    for cid, rows in reply["out"].items():
                        actual = [tuple(r["corner"]) for r in rows]
                        if len(actual) != len(shard) or set(actual) != set(map(tuple, shard)):
                            raise WorkerFailure("PVT worker returned incomplete/duplicate grid")
                        for row in rows:
                            row["worker_pid"] = pid
                            row["worker_phase"] = phase
                        out[cid].extend(rows)
                    for name in totals:
                        totals[name] += reply[name]
                if totals["evaluations"] != len(candidates) * len(grid):
                    raise WorkerFailure("PVT worker evaluation count mismatch")
            except BaseException:
                # Never leave timed-out workers computing into the next request.
                self.close()
                raise
            for rows in out.values():
                if len(rows) != len(grid) or {tuple(r["corner"]) for r in rows} != set(map(tuple, grid)):
                    raise WorkerFailure("Incomplete or duplicated merged PVT grid")
                rows.sort(key=lambda r: (r["corner"][0], r["corner"][2], r["corner"][1]))
            return out, totals


class ResidentEvaluator:
    def __init__(self, pool, phase, spec, store, limit, deadline):
        self.pool, self.phase, self.spec, self.store = pool, phase, spec, Path(store)
        self.limit, self.deadline = limit, deadline
        self.evaluations = 0
        self.measure_all = self.analysis = 0
        self.corner_integrity_passed = True  # checked by every worker at startup

    def evaluate_batch_parallel(self, candidates, grid):
        from silq.pvt_refinement import EvaluationBudgetExceeded
        if self.evaluations + len(candidates) * len(grid) > self.limit:
            raise EvaluationBudgetExceeded("batch would exceed the full-evaluation budget")
        rows, counts = self.pool.evaluate(self.phase, candidates, grid, self.spec, self.store, self.deadline)
        self.evaluations += counts["evaluations"]
        self.measure_all += counts["measure_all"]
        self.analysis += counts["analysis"]
        return rows


_pool = None
_pool_lock = threading.Lock()


def configuration_stamp():
    """Loaded Python code and model-location identity, checked between requests."""
    paths = ['pvt_workers.py', 'pvt_refinement.py', 'evaluator.py', 'guards.py',
             'specs.py', 'circuits/ctle.py', 'sim/server.py', 'sim/measures.py',
             'sim/probe.py', 'sim/eye.py', 'sim/channel.py']
    digest = hashlib.sha256()
    for path in paths:
        digest.update((ROOT/'src/silq'/path).read_bytes())
    for name in ('PDK_ROOT', 'NGSPICE_LIBRARY_PATH', 'SPICE_LIB_DIR', 'SILQ_PVT_LANES'):
        digest.update(f'{name}={os.environ.get(name, "")}'.encode())
    return digest.hexdigest()


def get_pool(deadline=None):
    global _pool
    deadline = time.monotonic() + 240 if deadline is None else deadline
    with deadline_lock(_pool_lock, deadline):
        stamp = configuration_stamp()
        if _pool is not None and getattr(_pool, 'configuration_stamp', None) != stamp:
            _pool.close()
        if _pool is None or _pool.closed:
            _pool = ResidentPool(lanes=int(os.environ.get("SILQ_PVT_LANES", "3")), deadline=deadline)
            _pool.configuration_stamp = stamp
        return _pool


def close_pool():
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


atexit.register(close_pool)


def worker_main(process, store):
    from contextlib import redirect_stdout
    from silq.agents.train_noise_pilot import _configure_ngspice
    _configure_ngspice()
    wire = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            from silq.evaluator import build_evaluator
            from silq.guards import ArtifactStore
            from silq.specs import DEFAULT_SPEC, Spec
            from silq.sim.server import get_server, NgspiceServer
            from silq.pvt_refinement import PVTEvaluator
            gate = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False, store=ArtifactStore(store))
            failure = gate.verify_corners(("tt", "ss"))
            if failure is not None:
                raise WorkerFailure(str(failure))
            get_server(process)
        wire.write(json.dumps(dict(ready=True, corner=process)) + "\n")
        wire.flush()
        for line in sys.stdin:
            payload = json.loads(line)
            try:
                with redirect_stdout(sys.stderr):
                    from silq.sim import measures
                    counts = dict(measure_all=0, analysis=0)
                    real_measure, real_analysis = measures.measure_all, NgspiceServer._analysis
                    def measure(*a, **kw):
                        counts["measure_all"] += 1
                        return real_measure(*a, **kw)
                    def analysis(*a, **kw):
                        counts["analysis"] += 1
                        return real_analysis(*a, **kw)
                    measures.measure_all, NgspiceServer._analysis = measure, analysis
                    try:
                        grid = [tuple(c) for c in payload["grid"]]
                        if any(c[0] != process for c in grid):
                            raise WorkerFailure("worker received wrong process corner")
                        ev = PVTEvaluator(Spec(**payload["spec"]), payload["store"], limit=10**9)
                        ev.corner_integrity_passed = True
                        result = ev.evaluate_batch(payload["candidates"], grid)
                    finally:
                        measures.measure_all, NgspiceServer._analysis = real_measure, real_analysis
                reply = dict(request_id=payload["request_id"], out=result,
                             evaluations=ev.evaluations, **counts)
            except Exception as exc:
                reply = dict(request_id=payload["request_id"], error=repr(exc))
            wire.write(json.dumps(reply, allow_nan=False) + "\n")
            wire.flush()
    except Exception as exc:
        wire.write(json.dumps(dict(error=repr(exc))) + "\n")
        wire.flush()
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--store", required=True)
    args = parser.parse_args()
    worker_main(args.worker, args.store)
