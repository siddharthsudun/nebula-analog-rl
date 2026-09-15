"""Do the four modes work, and do they actually trade accuracy against cost?

Nothing in results/ measures this. target_audit.py does its own model.predict rollout and
has no --mode, so every audit number on file (23/32 loose, 9/32 strict, ...) says nothing
about modes at all. This is the first measurement of them.

Reports, per mode: wall time, the two counted cost units, delivered boost, |error| vs the
requested target, and whether verify() passed. Three specs, so one lucky spec cannot make
a mode look good.

Reads only. Writes one scratchpad JSON, nothing in results/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(r"C:\Users\talk2.000\Desktop\Claude\nebula-analog-rl")
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

_NG = Path(os.environ["USERPROFILE"]) / "eqrl-ngspice"
os.environ.setdefault("PDK_ROOT", str(Path(os.environ["USERPROFILE"]) / "pdk"))
os.environ["PATH"] = f"{_NG / 'shim'};{_NG / 'Library' / 'bin'};{os.environ['PATH']}"

from eqrl import pipeline as pl  # noqa: E402

SPECS = [(9.0, 12.0, 0), (7.5, 10.0, 1), (10.5, 14.0, 2)]
MODES = ("default", "accurate", "fastest", "thinking")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mode_bench.json")


def cost_of(r: dict) -> tuple:
    """Unit 1 (optimizer evals) and the measure_all count -- the two units a mode moves.

    `design()` returns these under "cost"; the key names come from _cost() in pipeline.py,
    not from guessing, because the first pass of this script guessed and printed None.
    """
    c = r.get("cost") or {}
    return c.get("optimizer_evals"), c.get("measure_all") or c.get("search_measure_all")


def main() -> None:
    from eqrl.experiments.final_comparison import load_policy
    t = time.perf_counter()
    load_policy(str(REPO / "results" / "seq_clean40k.zip"))
    print(f"warm-up {time.perf_counter() - t:.1f}s\n", flush=True)

    print(f"{'spec':>14}  {'mode':9} {'wall':>7} {'meas':>5} {'anal':>5} "
          f"{'boost':>7} {'|err|':>7}  status", flush=True)
    rows = []
    for tgt, chan, si in SPECS:
        for mode in MODES:
            t = time.perf_counter()
            try:
                # pvt=False: this is a mode-vs-mode nominal comparison; PVT repair (now
                # pipeline.design()'s default) is a shared post-verification stage, not
                # something any mode controls, and would swamp every other cost/wall
                # column here with its own ~80-150s if left on.
                r = pl.design(target_boost_db=tgt, channel_loss_db=chan,
                              spec_index=si, mode=mode, pvt=False)
                dt = time.perf_counter() - t
                v = r.get("verification") or {}
                m = v.get("measures") or {}
                boost = m.get("boost_db")
                err = v.get("abs_err_db")  # computed by verify() against the real spec
                meas, anal = cost_of(r)
                row = {"target": tgt, "channel": chan, "spec_index": si, "mode": mode,
                       "wall_s": round(dt, 1), "measure_all": meas, "analysis": anal,
                       "boost_db": boost, "abs_err_db": err,
                       "status": r.get("status"), "error": None}
                print(f"{tgt:5.1f}dB/{chan:4.1f}  {mode:9} {dt:6.1f}s "
                      f"{str(meas):>5} {str(anal):>5} "
                      f"{'--' if boost is None else f'{boost:6.2f}'} "
                      f"{'--' if err is None else f'{err:6.3f}'}  {r.get('status')}",
                      flush=True)
            except BaseException as e:  # SearchHalted is a BaseException
                dt = time.perf_counter() - t
                row = {"target": tgt, "channel": chan, "spec_index": si, "mode": mode,
                       "wall_s": round(dt, 1), "error": f"{type(e).__name__}: {e}"}
                print(f"{tgt:5.1f}dB/{chan:4.1f}  {mode:9} {dt:6.1f}s  "
                      f"*** {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            rows.append(row)
            OUT.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
