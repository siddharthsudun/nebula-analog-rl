"""Run a training job to completion across native crashes.

Two 40k runs have now died silently mid-simulation with no Python traceback -- the log
ends inside a `Note: Transient op` pair. No traceback means the death is in native code
(libngspice inside a SubprocVecEnv worker, or the OS reclaiming one of nine resident
simulators), so there is nothing to catch in Python and no amount of exception handling
inside the trainer would help. The first death cost ~4,600 steps; the second cost 21,000
steps and went unnoticed for fifteen hours.

This supervises the trainer from outside instead: on a nonzero exit it finds the newest
checkpoint, works out how many steps remain, and resumes. Restarting is also the only
mitigation available for the leak, since a fresh process gets fresh simulators.

Cost of a crash drops from "the whole run" to "at most save_freq steps". A heartbeat file
records progress so the run's state can be read in one cheap call rather than by tailing
a log full of simulator chatter.

    python scripts/train_supervised.py --target 40000 --out results/seq_dcfix40k.zip \
        -- --guarded --fast --n-envs 8 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "results" / "checkpoints"


def toolchain_env() -> dict:
    """Same variables scripts/win-env.ps1 exports, so the child needs no shell setup."""
    env = dict(os.environ)
    home = Path(env["USERPROFILE"])
    ng, pdk = home / "eqrl-ngspice", home / "pdk"
    env.setdefault("NGSPICE_LIBRARY_PATH", str(ng / "Library" / "bin" / "ngspice{}.dll"))
    env.setdefault("SPICE_LIB_DIR", str(ng / "Library" / "share" / "ngspice"))
    env.setdefault("PDK_ROOT", str(pdk))
    env["PATH"] = f"{ng/'shim'};{ng/'Library'/'bin'};{env['PATH']}"
    # eqrl is not pip-installed into .venv, so `python -m eqrl.agents...` only resolves
    # when src/ is on the path. The previous runs got that from whichever shell launched
    # them, which is why they could not be restarted from anywhere else.
    src = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src};{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    env["PYTHONUNBUFFERED"] = "1"
    return env


def newest_checkpoint(prefix: str) -> tuple[Path | None, int]:
    """Highest-step checkpoint for this prefix, chosen by step count and not mtime.

    mtime ordering is wrong here: a resumed run rewrites earlier-numbered files if the
    prefix is reused, and picking one of those would silently retrain ground already
    covered while reporting progress.
    """
    best, best_steps = None, 0
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    if not CKPT_DIR.is_dir():
        return None, 0
    for f in CKPT_DIR.iterdir():
        m = pat.match(f.name)
        if m and int(m.group(1)) > best_steps:
            best, best_steps = f, int(m.group(1))
    return best, best_steps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, required=True,
                   help="total timesteps the model should reach, counting resumed steps")
    p.add_argument("--out", required=True)
    p.add_argument("--max-restarts", type=int, default=40)
    p.add_argument("--log", default=None)
    p.add_argument("passthrough", nargs=argparse.REMAINDER,
                   help="everything after `--` is handed to train_sequential unchanged")
    args = p.parse_args()

    extra = [a for a in args.passthrough if a != "--"]
    out = Path(args.out)
    prefix = out.stem
    log = Path(args.log) if args.log else ROOT / "results" / f"{prefix}_supervised.log"
    beat = ROOT / "results" / f"{prefix}_heartbeat.json"
    env = toolchain_env()
    py = sys.executable

    attempts: list[dict] = []
    t0 = time.time()
    stalls = 0

    def heartbeat(state: str, steps: int) -> None:
        beat.write_text(json.dumps({
            "state": state, "prefix": prefix, "target": args.target,
            "steps_reached": steps,
            "pct": round(100.0 * steps / args.target, 1),
            "attempts": len(attempts), "stalls": stalls,
            "elapsed_min": round((time.time() - t0) / 60.0, 1),
            "history": attempts[-12:],
            "log": str(log),
        }, indent=2))

    while len(attempts) < args.max_restarts:
        ckpt, steps = newest_checkpoint(prefix)
        if steps >= args.target:
            heartbeat("done", steps)
            print(f"[supervisor] target reached at {steps} steps")
            return

        remaining = args.target - steps
        cmd = [py, "-m", "eqrl.agents.train_sequential",
               "--timesteps", str(remaining), "--out", str(out), *extra]
        if ckpt is not None:
            # train_sequential passes reset_num_timesteps=False when resuming, and SB3
            # then treats total_timesteps as ADDITIONAL steps -- hence `remaining` rather
            # than `target`, or every restart would extend the run instead of finishing it.
            cmd += ["--resume", str(ckpt)]

        heartbeat("running", steps)
        # Poll the checkpoint directory while the child runs. Without this the heartbeat
        # is written once per attempt and then goes stale for the whole run, so a job an
        # hour into training still reports the step count it started from -- which reads
        # exactly like a hung process. Checking costs one directory listing a minute and
        # makes the file answer "how far along is it" instead of "where did it begin".
        stop = threading.Event()

        def _tick() -> None:
            while not stop.wait(60):
                heartbeat("running", newest_checkpoint(prefix)[1])

        ticker = threading.Thread(target=_tick, daemon=True)
        ticker.start()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        banner = (f"\n{'='*72}\n[supervisor] attempt {len(attempts)+1} at {stamp}: "
                  f"{steps} -> {args.target} ({remaining} to go)"
                  f"{' resuming ' + ckpt.name if ckpt else ' cold start'}\n{'='*72}\n")
        print(banner.strip(), flush=True)
        with log.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(banner)
            fh.flush()
            try:
                rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh,
                                     stderr=subprocess.STDOUT)
            finally:
                stop.set()

        _, after = newest_checkpoint(prefix)
        attempts.append({"attempt": len(attempts) + 1, "started": stamp,
                         "resumed_from_steps": steps, "exit_code": rc,
                         "steps_after": after})
        print(f"[supervisor] exit {rc}, checkpoint now at {after} steps", flush=True)

        if rc == 0:
            # A clean exit means model.learn() returned and the model was saved. Trust it.
            heartbeat("done", max(after, args.target))
            print(f"[supervisor] training completed cleanly -> {out}")
            return

        # A crash that produced no new checkpoint means resuming lands in the same place
        # and crashes again. Restarting on that is a spin, not a recovery.
        if after <= steps:
            stalls += 1
            if stalls >= 3:
                heartbeat("stalled", after)
                print(f"[supervisor] three restarts with no progress past {after} steps; "
                      f"stopping. See {log}", file=sys.stderr)
                return
        else:
            stalls = 0
        heartbeat("restarting", after)
        time.sleep(10)          # let the OS reclaim nine simulator processes

    heartbeat("restart_limit", newest_checkpoint(prefix)[1])
    print("[supervisor] hit --max-restarts", file=sys.stderr)


if __name__ == "__main__":
    main()
