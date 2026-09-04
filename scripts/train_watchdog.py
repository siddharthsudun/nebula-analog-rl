"""Watch a training run's LEARNING and stop it when it goes wrong.

scripts/train_supervised.py already handles the run's *liveness*: native crashes inside
libngspice, restarts, checkpoint resume. It cannot tell a healthy run from one that is
burning seven hours learning nothing, because a diverging PPO run exits 0 and saves a
model like any other.

This is the other half. It tails the supervisor's log, parses stable-baselines3's own
iteration tables, and compares them against the trajectory the delivered policy actually
followed -- results/seq_clean40k_supervised.log, 39 logged iterations, seed 0, the run
that produced results/seq_clean40k.zip. Every threshold below is derived from that curve
rather than chosen from intuition:

    steps      std   entropy  expl_var  approx_kl  value_loss
     2048    0.996     -8.51     0.003     0.0076       745.0
    20480    0.940     -8.14     0.009     0.0112       127.0
    40960    0.838     -7.46     0.164     0.0186        24.6

The shape that matters: `std` decays smoothly and only to 0.84 -- the policy stays wide.
`value_loss` falls monotonically by 30x. `approx_kl` stays under 0.02. A new seed that
departs from that envelope is not "a different seed", it is a broken run, and the cheapest
moment to learn that is at 5,000 steps rather than at 40,000.

    python scripts/train_watchdog.py --log results/seed1.log --heartbeat results/seed1_heartbeat.json
    python scripts/train_watchdog.py --log results/seed1.log --stop --pid 1234

Report-only by default. `--stop` terminates the supervisor AND the trainer tree beneath it
on a breach (see `_terminate_tree`); the verdict is always written to <log>.watchdog.json
so the decision is auditable afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path

#: Reference envelope, read off seed 0 (see module docstring). Each entry is
#: (name, predicate(value, steps) -> bool is_breach, human explanation).
#:
#: These are deliberately LOOSE. The purpose is to catch a run that has genuinely come
#: apart, not to enforce that every seed retraces seed 0 -- seed variation is the entire
#: point of the experiment, and a watchdog that stops a merely-different run destroys the
#: result it was meant to protect.
CHECKS: list[tuple[str, str]] = [
    ("std_collapse",
     "policy went deterministic: the action std is far below seed 0's, so the run has "
     "stopped exploring and every later evaluation will retrace one trajectory"),
    ("kl_blowup",
     "policy updates are too large: approx_kl is several times seed 0's, which is how a "
     "PPO run destroys a working policy in a handful of updates"),
    ("value_divergence",
     "the value function is getting worse, not better: seed 0's value_loss fell "
     "monotonically by 30x and explained_variance rose"),
    ("invalid_rate",
     "the policy is CURRENTLY spending its time outside the physically valid region. "
     "Judged on the marginal rate between checks, never the cumulative one -- see "
     "_marginal_invalid"),
    ("stalled",
     "no progress in the heartbeat: the trainer is alive but not stepping"),
]


def parse_iterations(text: str) -> list[dict]:
    """Pull stable-baselines3's iteration tables out of a log full of simulator chatter.

    The log is ~40k lines, the overwhelming majority of them `Note: Transient op ...`
    emitted by ngspice, so a table is found by its `| key | value |` rows rather than by
    position.
    """
    rows, cur = [], {}
    for line in text.splitlines():
        m = re.match(r"\|\s+(\w+)\s+\|\s+([-\d.e+]+)\s*\|", line)
        if m:
            try:
                cur[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif line.startswith("---") and cur.get("total_timesteps"):
            rows.append(cur)
            cur = {}
    if cur.get("total_timesteps"):
        rows.append(cur)
    return [r for r in rows if "std" in r]


#: Minimum simulations between two readings before their marginal rate means anything.
#: Below this the ratio is dominated by which episode happened to straddle the boundary.
_MARGINAL_MIN_SIMS = 300


def _marginal_invalid(prev: tuple[int, int] | None,
                      cur: tuple[int, int] | None) -> float | None:
    """Invalid fraction of the simulations run BETWEEN two readings.

    `*_invalid.json` reports cumulative counters: n_invalid / n_sims over the whole run.
    That number cannot answer "is the policy stuck outside the valid region now", because
    every invalid design from the first thousand steps stays in the denominator forever.
    Early in a run it is high by construction -- seed 1 read 0.86 at step 0 and 0.81 at
    4,096 while its PPO diagnostics tracked seed 0 almost exactly.

    Thresholding the cumulative rate against seed 0's FINAL 0.419 therefore compares two
    different quantities. Whether the old check would have fired on seed 0 itself is NOT
    known: seed 0's log records its invalid count exactly once, in the closing summary, so
    there is no intermediate value to check against. What is established is the mechanism
    (a cumulative ratio cannot fall faster than its history allows) and seed 1's readings
    above. Differencing two readings gives the rate over just the interval between them,
    which is the quantity the check was always meant to be about.

    Returns None when there is no previous reading or too few simulations to be meaningful.
    """
    if prev is None or cur is None:
        return None
    d_sims, d_invalid = cur[0] - prev[0], cur[1] - prev[1]
    if d_sims < _MARGINAL_MIN_SIMS:
        return None
    return max(0.0, min(1.0, d_invalid / d_sims))


def assess(rows: list[dict], marginals: list[float],
           hb_age_s: float | None) -> list[tuple[str, str]]:
    """Return the list of (check, detail) breaches. Empty means healthy."""
    out: list[tuple[str, str]] = []
    if not rows:
        return out
    last = rows[-1]
    steps = last["total_timesteps"]

    # std: seed 0 held 0.996 -> 0.838. Anything under 0.55 has stopped exploring; the
    # allowance widens with steps because some decay is the point of learning.
    floor = 0.55 if steps < 30_000 else 0.45
    if last["std"] < floor:
        out.append(("std_collapse",
                    f"std={last['std']:.3f} at {int(steps)} steps (floor {floor}; "
                    f"seed 0 was 0.90 at 30k, 0.84 at 41k)"))

    # approx_kl: seed 0 stayed in 0.008-0.028. Sustained means two consecutive tables,
    # so one noisy update does not stop a good run.
    kls = [r.get("approx_kl", 0.0) for r in rows[-2:]]
    if len(kls) == 2 and all(k > 0.05 for k in kls):
        out.append(("kl_blowup",
                    f"approx_kl={kls[-1]:.4f} for 2 consecutive updates "
                    f"(seed 0 peaked at 0.028)"))

    # value function: only judged after it has had a chance to learn. Seed 0 was already
    # at value_loss 213 by 10k steps and never rose across a 3-table window.
    if steps > 12_000 and len(rows) >= 4:
        window = [r.get("value_loss", 0.0) for r in rows[-4:]]
        ev = last.get("explained_variance", 0.0)
        # 1.5x, not 2x: value-function failure in practice is a steady climb rather than
        # an explosion, and requiring a doubling only catches the runs that were going to
        # be obvious anyway. Paired with explained_variance < -0.2 -- a value head doing
        # worse than predicting the mean -- so ordinary noise cannot trip it.
        if window[-1] > window[0] * 1.5 and ev < -0.2:
            out.append(("value_divergence",
                        f"value_loss {window[0]:.0f} -> {window[-1]:.0f} over 4 updates "
                        f"with explained_variance={ev:.3f}"))

    # Sustained across two intervals, for the same reason approx_kl is: one interval that
    # happens to land on a bad patch of the search space is not a broken run. The gate is
    # 12k rather than 8k steps because the marginal rate is still legitimately high while
    # the policy is doing its initial exploration.
    if steps > 12_000 and len(marginals) >= 2 and all(m > 0.60 for m in marginals[-2:]):
        out.append(("invalid_rate",
                    f"marginal invalid rate {marginals[-2]:.3f} then {marginals[-1]:.3f} "
                    f"at {int(steps)} steps -- the policy is still producing mostly "
                    f"unbuildable designs (seed 0's cumulative finished at 0.419)"))

    if hb_age_s is not None and hb_age_s > 1800:
        out.append(("stalled", f"heartbeat last written {hb_age_s / 60:.0f} min ago"))

    return out


def _terminate_tree(pid: int) -> str:
    """Stop the supervisor AND everything it spawned. Returns a human summary.

    `os.kill(pid, SIGTERM)` is TerminateProcess on Windows and reaches exactly one
    process. The tree here is three deep -- this watchdog signals scripts/train_supervised,
    whose child is `python -m eqrl.agents.train_sequential`, which itself holds --n-envs
    SubprocVecEnv workers, each with a resident libngspice. Signalling only the top of that
    leaves ten processes training on while this script prints "stopping" and exits.

    A watchdog that reports a run stopped when nothing stopped is worse than no watchdog:
    it converts a detected failure into a silent one, and whoever reads the verdict file
    believes a breach was acted on.

    Children are terminated before the parent so the supervisor cannot observe its child
    dying and helpfully restart it -- that is exactly what it is built to do on a nonzero
    exit, and it would resurrect the run this function exists to end.
    """
    try:
        import psutil                      # not in requirements.txt; optional by design
    except ImportError:
        # Best effort without psutil. On Windows taskkill /T walks the tree; elsewhere the
        # process group does. Either way, say what was actually attempted.
        if os.name == "nt":
            import subprocess
            rc = subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"taskkill /T /F on {pid} returned {rc} (psutil not installed)"
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return f"SIGTERM to process group of {pid} (psutil not installed)"

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return f"pid {pid} was already gone"

    victims = parent.children(recursive=True) + [parent]
    for v in victims:
        try:
            v.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(victims, timeout=15)
    for v in alive:                        # libngspice can ignore a polite terminate
        try:
            v.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)
    return f"terminated {len(gone)} of {len(victims)} processes, force-killed {len(alive)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="supervisor log to tail")
    p.add_argument("--heartbeat", default=None)
    p.add_argument("--invalid", default=None, help="the trainer's *_invalid.json")
    p.add_argument("--interval", type=float, default=300.0)
    p.add_argument("--pid", type=int, default=None, help="supervisor pid, for --stop")
    p.add_argument("--stop", action="store_true",
                   help="terminate the run on a breach instead of only reporting it")
    p.add_argument("--once", action="store_true", help="assess once and exit")
    args = p.parse_args()

    log = Path(args.log)
    verdict_path = log.with_suffix(log.suffix + ".watchdog.json")

    prev_counts: tuple[int, int] | None = None
    marginals: list[float] = []

    while True:
        rows = parse_iterations(log.read_text(errors="ignore")) if log.exists() else []
        invalid = None
        counts: tuple[int, int] | None = None
        if args.invalid and Path(args.invalid).exists():
            try:
                blob = json.loads(Path(args.invalid).read_text())
                invalid = blob.get("invalid_rate")          # cumulative; reported only
                if blob.get("n_sims") is not None:
                    counts = (int(blob["n_sims"]), int(blob["n_invalid"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass          # the trainer rewrites this file; a torn read is not a fault
        m = _marginal_invalid(prev_counts, counts)
        if m is not None:
            marginals.append(m)
            prev_counts = counts
        elif prev_counts is None:
            prev_counts = counts
        hb_age = None
        if args.heartbeat and Path(args.heartbeat).exists():
            hb_age = time.time() - Path(args.heartbeat).stat().st_mtime

        breaches = assess(rows, marginals, hb_age)
        last = rows[-1] if rows else {}
        verdict = {
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations_seen": len(rows),
            "steps": int(last.get("total_timesteps", 0)),
            "std": last.get("std"),
            "entropy_loss": last.get("entropy_loss"),
            "explained_variance": last.get("explained_variance"),
            "approx_kl": last.get("approx_kl"),
            "value_loss": last.get("value_loss"),
            # Both, explicitly named. The cumulative one is context for a human reading
            # this file; only the marginal one is thresholded.
            "invalid_rate_cumulative": invalid,
            "invalid_rate_marginal": marginals[-1] if marginals else None,
            "invalid_marginal_history": marginals[-8:],
            "healthy": not breaches,
            "breaches": [{"check": c, "detail": d} for c, d in breaches],
        }
        verdict_path.write_text(json.dumps(verdict, indent=2))

        state = "HEALTHY" if not breaches else "BREACH"
        mstr = f"{marginals[-1]:.3f}" if marginals else "n/a"
        print(f"[watchdog] {state} steps={verdict['steps']} std={verdict['std']} "
              f"kl={verdict['approx_kl']} vloss={verdict['value_loss']} "
              f"invalid_marg={mstr} (cum={invalid})", flush=True)
        for c, d in breaches:
            print(f"[watchdog]   {c}: {d}", flush=True)

        if breaches and args.stop and args.pid:
            print(f"[watchdog] stopping pid {args.pid} and its trainer tree", flush=True)
            try:
                outcome = _terminate_tree(args.pid)
            except Exception as exc:        # noqa: BLE001 - report, never mask, a failed stop
                outcome = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[watchdog] {outcome}", flush=True)
            # Rewrite the verdict with what the stop actually did. Without this the file
            # records that a breach was detected but not whether anything was killed, and
            # the two are not the same claim.
            verdict["stop_attempted"] = True
            verdict["stop_outcome"] = outcome
            verdict_path.write_text(json.dumps(verdict, indent=2))
            return
        if args.once or breaches:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
