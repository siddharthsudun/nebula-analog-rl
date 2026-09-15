"""Guarded, bounded PVT refinement utilities for an already sized CTLE.

The search objective is separate from the frozen RL reward. Reduced-corner
scores guide search only; acceptance requires the exact full corner grid.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from eqrl.circuits import ctle
from eqrl.specs import Spec, hard_pass

ROOT = Path(__file__).resolve().parents[2]


class EvaluationBudgetExceeded(RuntimeError):
    pass


def signed_slacks(m, spec: Spec, health: dict) -> dict[str, float]:
    """Dimensionless distance to each existing acceptance boundary.

Scales are declared here, not learned from results. No spec/guard threshold is
changed. hard_pass remains authoritative at strict versus inclusive boundaries.
"""
    from eqrl.guards import SATURATION_HEADROOM_V, TAIL_CURRENT_TOLERANCE
    boost_span = spec.boost_db_max - spec.boost_db_min
    freq_span = spec.peak_freq_hi_ghz - spec.peak_freq_lo_ghz
    out = {
        "boost_low": (m.boost_db - spec.boost_db_min) / boost_span,
        "boost_high": (spec.boost_db_max - m.boost_db) / boost_span,
        "peak_low": (m.peak_freq_ghz - spec.peak_freq_lo_ghz) / freq_span,
        "peak_high": (spec.peak_freq_hi_ghz - m.peak_freq_ghz) / freq_span,
        "hd3": (spec.hd3_db_max - m.hd3_db) / abs(spec.hd3_db_max),
        "noise": (spec.noise_vrms_max - m.noise_vrms) / spec.noise_vrms_max,
        "power": (spec.power_w_max - m.power_w) / spec.power_w_max,
        "area": (spec.area_mm2_max - m.area_mm2) / spec.area_mm2_max,
        "eye_h": (m.eye_h_ui - spec.eye_h_ui_min) / spec.eye_h_ui_min,
        "eye_v": (m.eye_v_mv - spec.eye_v_mv_min) / spec.eye_v_mv_min,
        "headroom": (health["worst_headroom_v"] - SATURATION_HEADROOM_V) / SATURATION_HEADROOM_V,
        "tail_delivery": (TAIL_CURRENT_TOLERANCE - abs(health["tail_delivery_ratio"] - 1)) / TAIL_CURRENT_TOLERANCE,
    }
    if spec.dc_gain_db_min is not None:
        out["dc_gain"] = (m.dc_gain_db - spec.dc_gain_db_min) / 3.0
    if spec.boost_target_tol_db is None or spec.boost_target_tol_db <= 0:
        raise ValueError("PVT refinement requires an explicit positive target tolerance")
    out["boost_target"] = 1 - abs(m.boost_db - spec.target_boost_db) / spec.boost_target_tol_db
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("non-finite PVT slack")
    return out


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot score an empty corner set")
    valid = [r for r in rows if r["guard_valid"]]
    return {
        "corners": len(rows),
        "guard_valid": len(valid),
        "passed": sum(bool(r.get("passed", False)) for r in rows),
        "worst_slack": min((min(r["slacks"].values()) for r in valid), default=-1e6),
        "worst_target_error_db": max((r["target_error_db"] for r in valid), default=None),
        "min_headroom_mv": min((r["health"]["worst_headroom_v"] * 1e3
                                for r in rows if "health" in r), default=None),
        "tail_delivery_pct_range": [
            min((r["health"]["tail_delivery_ratio"] * 100 for r in rows if "health" in r), default=None),
            max((r["health"]["tail_delivery_ratio"] * 100 for r in rows if "health" in r), default=None),
        ],
        "guard_violation_sum": sum(float(r.get("guard_violation") or 0) for r in rows
                                   if not r["guard_valid"]),
    }


def rank_key(rows: list[dict]) -> tuple:
    s = summarize(rows)
    return (s["corners"] - s["guard_valid"], s["corners"] - s["passed"],
            s["guard_violation_sum"], -s["worst_slack"],
            s["worst_target_error_db"] if s["worst_target_error_db"] is not None else 1e6)


def full_grid_pass(rows: list[dict], grid: list[tuple]) -> bool:
    """Duplicates, omissions, wrong corners, and failed checks all prevent acceptance."""
    actual = [tuple(r["corner"]) for r in rows]
    expected = list(map(tuple, grid))
    return (len(actual) == len(expected) == len(set(actual)) and
            set(actual) == set(expected) and
            all(r["guard_valid"] and r.get("passed", False) for r in rows))


def stress_corners(rows: list[dict], grid: list[tuple], limit: int = 8) -> list[tuple]:
    """Include every process and the weakest baseline corners, including TT nominal."""
    if limit < len({p for p, _, _ in grid}):
        raise ValueError("stress budget cannot cover every process")
    ordered = sorted(rows, key=lambda r: (
        r["guard_valid"], r.get("passed", False),
        min(r.get("slacks", {"missing": -1e6}).values())))
    selected = []

    def add(c):
        c = tuple(c)
        if c not in selected and len(selected) < limit:
            selected.append(c)

    for process in dict.fromkeys(c[0] for c in grid):
        add(next(r["corner"] for r in ordered if r["corner"][0] == process))
    nominal = next((c for c in grid if c[0] == "tt" and c[2] == 27 and abs(c[1] - 1.8) < 1e-6), None)
    if nominal:
        add(nominal)
    for row in ordered:
        add(row["corner"])
    return sorted(selected, key=lambda c: (c[0], c[2], c[1]))


def recorded_netlist(dv, *, vdd, temp_c, corner, analysis="op", **_kw):
    """Record the actual resident-deck VCM scaling, including non-nominal supplies."""
    params = dict(ctle.dv_to_params(dv), vddp=vdd, tempc=temp_c)
    overrides = "\n".join(f".param {k}={v:.6g}" for k, v in params.items())
    return ctle.param_deck(corner).replace(".end\n", overrides + "\n" + ctle._ANALYSIS[analysis] + "\n.end\n")


class PVTEvaluator:
    """One resident simulator, sequential corner-major batches, complete health records."""
    def __init__(self, spec, store, limit: int):
        from eqrl.guards import ArtifactStore
        self.spec, self.store = spec, ArtifactStore(store)
        self.limit, self.evaluations = limit, 0
        self.last_op = None
        self.corner_integrity_passed = False

    def evaluate_batch(self, candidates: list[dict], grid: list[tuple]) -> dict[str, list[dict]]:
        from eqrl.evaluator import build_evaluator
        from eqrl.guards import SearchHalted
        from eqrl.sim.probe import CTLE_NODES, ProbeError, probe_operating_point
        from eqrl.sim.server import get_server

        if self.evaluations + len(candidates) * len(grid) > self.limit:
            raise EvaluationBudgetExceeded("batch would exceed the full-evaluation budget")
        out = {c["id"]: [] for c in candidates}

        def read_op(srv):
            op = probe_operating_point(srv, nodes=CTLE_NODES + ("cm",))
            self.last_op = op
            if abs(op.node_voltages["cm"] - self.expected_cm) > 1e-5:
                raise ProbeError("measured common-mode does not match actual VDD times shipped ratio")
            return op

        for proc, vdd, temp in sorted(grid, key=lambda c: (c[0], c[2], c[1])):
            # The factory switches the singleton to the requested process before
            # priming. Keeping corners outside candidates avoids repeated reloads.
            ev = build_evaluator(self.spec, corner=proc, temp_c=temp, fast=False,
                                 store=self.store, server_factory=get_server,
                                 netlist_builder=recorded_netlist, operating_point_probe=read_op)
            if not self.corner_integrity_passed:
                # TT versus SS exercises the existing model-change gate. SF/SS
                # can share NMOS characteristics, so do not demand pairwise
                # distinct thresholds from all five process labels.
                failure = ev.verify_corners(("tt", "ss"))
                if failure is not None:
                    raise RuntimeError(f"corner-integrity gate failed: {failure}")
                self.corner_integrity_passed = True
            for candidate in candidates:
                dv = ctle.DesignVars(**candidate["design"])
                self.last_op, self.expected_cm = None, vdd * ctle.VCM_VDD_RATIO
                self.evaluations += 1
                row = {"corner": [proc, vdd, temp], "guard_valid": False,
                       "passed": False, "evaluation": self.evaluations}
                try:
                    v = ev.evaluate(dv, vdd=vdd)
                except SearchHalted as exc:
                    row["guard_rejection"] = f"SearchHalted: {exc}"
                else:
                    row.update(guard_valid=v.is_valid, artifact_dir=str(v.artifact_dir))
                    if v.is_valid:
                        m = v.unwrap()
                        passed, checks = hard_pass(m, self.spec)
                        row.update(passed=bool(passed), checks={k: bool(x) for k, x in checks.items()},
                                   measures=asdict(m), target_error_db=abs(m.boost_db - self.spec.target_boost_db))
                    else:
                        row.update(guard_rejection=v.check.value, reason=v.reason,
                                   guard_violation=v.violation)
                if self.last_op is not None:
                    op = self.last_op
                    row["health"] = {
                        "nodes_v": dict(op.node_voltages),
                        "all_saturated": all(d.saturated for d in op.devices),
                        "worst_headroom_v": min(d.headroom for d in op.devices),
                        "tail_delivery_ratio": sum(abs(v) for v in op.tail_currents.values()) / dv.i_tail,
                        "devices": [dict(asdict(d), saturated=d.saturated, headroom_v=d.headroom)
                                    for d in op.devices],
                    }
                if row["guard_valid"]:
                    row["slacks"] = signed_slacks(m, self.spec, row["health"])
                out[candidate["id"]].append(row)
            print(f"corner {proc}/{vdd:.2f}/{temp:g}: total {self.evaluations}/{self.limit} evaluations", flush=True)
        return out

    def evaluate_batch_parallel(self, candidates: list[dict], grid: list[tuple]) -> dict[str, list[dict]]:
        """Same contract and result as evaluate_batch, sharded one subprocess per
        distinct process corner (tt/ss/ff/sf/fs) present in `grid`.

        Model reload (~15s) only happens when the SKY130 process model changes,
        so evaluate_batch's corner-outer loop already reloads at most 5 times for
        the full 45-corner grid -- but does so serially in one resident ngspice
        process. Running one worker per process model lets those (at most 5)
        reloads, and the corner evals behind them, happen concurrently instead.
        Falls back to the plain sequential path when only one process is present
        (nothing to shard) or the batch is too small to be worth a subprocess.
        """
        import os
        import subprocess
        import sys
        import tempfile
        from eqrl.evaluator import build_evaluator
        from eqrl.sim.server import get_server

        if self.evaluations + len(candidates) * len(grid) > self.limit:
            raise EvaluationBudgetExceeded("batch would exceed the full-evaluation budget")
        shards: dict[str, list] = {}
        for c in grid:
            shards.setdefault(c[0], []).append(c)
        if len(shards) <= 1 or len(candidates) * len(grid) < 8:
            return self.evaluate_batch(candidates, grid)
        if not self.corner_integrity_passed:
            # Same gate evaluate_batch runs before its first corner -- it doesn't
            # depend on which corner triggers it and doesn't consume an evaluation,
            # so it's cheaper and simpler to run once here than to thread a
            # "skip it" flag through every shard worker.
            proc0, vdd0, temp0 = grid[0]
            probe = build_evaluator(self.spec, corner=proc0, temp_c=temp0, fast=False,
                                    store=self.store, server_factory=get_server)
            failure = probe.verify_corners(("tt", "ss"))
            if failure is not None:
                raise RuntimeError(f"corner-integrity gate failed: {failure}")
            self.corner_integrity_passed = True
        out = {c["id"]: [] for c in candidates}
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        with tempfile.TemporaryDirectory(prefix="pvt_shard_") as tmp:
            tmp = Path(tmp)
            running = []
            from eqrl.pvt_workers import WindowsJob
            job = WindowsJob() if os.name == "nt" else None
            try:
                for i, (proc, shard_grid) in enumerate(shards.items()):
                    payload = tmp / f"shard_{i}.json"
                    payload.write_text(json.dumps({
                        "spec": asdict(self.spec), "store": str(self.store.root),
                        "candidates": candidates, "grid": shard_grid,
                    }), encoding="utf-8")
                    log = (tmp / f"shard_{i}.log").open("w", encoding="utf-8")
                    p = subprocess.Popen(
                        [sys.executable, "-m", "eqrl.pvt_refinement", str(payload)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | (4 if job else 0))
                    running.append((proc, p, log, payload))
                    if job:
                        job.attach_and_resume(p)
                for proc, p, log, payload in running:
                    code = p.wait()
                    log.close()
                    if code != 0:
                        raise RuntimeError(
                            f"PVT corner-shard worker for process {proc!r} failed (exit {code}); " +
                            payload.with_suffix('.log').read_text(encoding="utf-8")[-2000:])
                    result = json.loads(payload.with_suffix(".out.json").read_text())
                    for cid, rows in result["out"].items():
                        out[cid].extend(rows)
                    self.evaluations += result["evaluations"]
                    from eqrl.simcount import add_worker_counts
                    add_worker_counts(result["counts"])
            finally:
                if job:
                    job.close()
                for _, process, _, _ in running:
                    if process.poll() is None:
                        process.kill()
                for _, process, log, _ in running:
                    process.wait()
                    log.close()
        return out


def _shard_worker_main(payload_path: Path) -> None:
    """Subprocess entry point for evaluate_batch_parallel: evaluate one process
    corner's candidates x (vdd, temp) slice in its own resident ngspice, write
    the same {candidate_id: rows} shape evaluate_batch returns."""
    from eqrl.agents.train_noise_pilot import _configure_ngspice, _set_single_threaded
    _set_single_threaded()
    _configure_ngspice()
    payload = json.loads(payload_path.read_text())
    spec = Spec(**payload["spec"])
    ev = PVTEvaluator(spec, payload["store"], limit=10**9)
    ev.corner_integrity_passed = True  # already verified once by the coordinator
    grid = [tuple(c) for c in payload["grid"]]
    from eqrl.simcount import counting
    with counting() as counts:
        out = ev.evaluate_batch(payload["candidates"], grid)
    payload_path.with_suffix(".out.json").write_text(
        json.dumps({"out": out, "evaluations": ev.evaluations, "counts": counts}), encoding="utf-8")


if __name__ == "__main__":
    import sys
    _shard_worker_main(Path(sys.argv[1]))
