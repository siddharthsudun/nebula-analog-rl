"""G3.2a step 1: choose the low-peak calibration designs, and freeze them before probing.

WHY A SEPARATE STEP. G3.2's plane was fitted on designs whose peaks span 1.90 to 3.19 GHz,
at or above the 1.25-2.5 GHz band, and applied to handoffs at 0.567 and 0.952 GHz, below it.
The controller was extrapolating a sign and a gain into a region the calibration never
visited, and that -- not the step sizing -- is why Stage A repaired 0 of 3. The question
G3.2a exists to answer is whether the same two-coordinate plane survives below the band or
whether the physics changes qualitatively there.

The selection is a separate script, run and committed BEFORE the probe, so that which
designs were calibrated on cannot have been influenced by what the probe found. The
alternative -- probing the three seed-3 specs that failed -- would fit the calibration on
the test set and is exactly what this project's discipline forbids.

WHY RANDOM DESIGNS AND NOT PPO HANDOFFS. The question is about circuit physics in a region
of the design space, not about the policy. Uniform sampling covers that region without the
policy's bias, and it is independent of every spec seed by construction. results/
feasibility_band.json already sampled 2000 such points at seed 7 and found 214 guard-valid
designs below the band, so the region is known to be populated -- but that artifact stores
the physical design only for the 51 IN-BAND records, so the low-peak vectors it found
cannot be recovered from it and have to be re-drawn. A fresh seed is used rather than 7 so
that the draw is not silently the same one under a different name.

Whether these random designs represent the sub-region PPO's low-peak handoffs occupy is a
separate validity question, answered separately by g32a_representative.py rather than
assumed here.

PRE-REGISTERED SELECTION RULE, fixed before the first simulation:
    draw x ~ U[0,1]^6 at SELECT_SEED, in order;
    accept if guard-valid AND SWEEP_FLOOR < peak_freq_ghz < 1.25 GHz;
    take the first N_SELECT accepted, in draw order.
No other property is consulted -- not boost, not which hard checks fail, not distance to
anything. Taking them in draw order rather than, say, the eight lowest peaks keeps the set
from being concentrated at one extreme of the regime.

Changes no threshold, bound, reward, model, or benchmark criterion.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

P = Path(os.environ.get("USERPROFILE") or Path.home()) / "eqrl-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import argparse
import numpy as np

from eqrl.circuits.ctle import ACTION_SPACE, decode_action
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC, hard_pass

DIMS = list(ACTION_SPACE.keys())

SELECT_SEED = 11          # not 7 (feasibility_band), not 2 or 3 (the spec seeds)
N_SELECT = 8              # matches the high-peak calibration's 8 specs
MAX_DRAWS = 200           # ceiling; the pool statistics predict ~75 draws for 8 accepts
SWEEP_FLOOR_GHZ = 0.0011  # a peak at AC_FSTART is a design with no peak, not a low peak

#: The calibration is a property of the circuit, not of any one spec, but the guard and
#: hard_pass need a spec to evaluate against. This is the default channel and a mid-range
#: target, fixed here so the selection cannot be steered by a spec choice. The target does
#: not enter the selection rule at all -- only guard validity and the peak do.
SELECT_CHANNEL_DB = 12.0
SELECT_TARGET_DB = 8.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/g32a_lowpeak_set.json")
    # Additive, defaults unchanged: G3.2b needs a SECOND independent low-peak set drawn by
    # this exact rule, and a second copy of the rule would be a second thing to keep in
    # sync. The default invocation still reproduces results/g32a_lowpeak_set.json.
    p.add_argument("--seed", type=int, default=SELECT_SEED)
    args = p.parse_args()
    seed = args.seed

    lo = DEFAULT_SPEC.peak_freq_lo_ghz
    guard = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                            channel_loss_db=SELECT_CHANNEL_DB)
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=SELECT_TARGET_DB,
                               channel_loss_db=SELECT_CHANNEL_DB)

    print("G3.2a SELECT | seed %d | accept guard-valid with %.4f < peak < %.2f GHz | "
          "take first %d in draw order" % (seed, SWEEP_FLOOR_GHZ, lo, N_SELECT),
          flush=True)

    rng = np.random.default_rng(seed)
    chosen, n_draw, n_valid = [], 0, 0
    while len(chosen) < N_SELECT and n_draw < MAX_DRAWS:
        x = rng.random(len(DIMS))
        n_draw += 1
        try:
            v = guard.evaluate(decode_action(x), vdd=spec.vdd_nominal)
        except Exception:
            continue
        if not v.is_valid:
            continue
        n_valid += 1
        m = v.unwrap()
        pk = float(m.peak_freq_ghz)
        if not (SWEEP_FLOOR_GHZ < pk < lo):
            continue
        ok, checks = hard_pass(m, spec)
        chosen.append({"draw": n_draw, "x": [float(t) for t in x],
                       "peak_freq_ghz": pk, "boost_db": float(m.boost_db),
                       "dc_gain_db": float(m.dc_gain_db), "loose_pass": bool(ok),
                       "failing": [c for c, g in checks.items() if not g],
                       "design": dataclasses.asdict(decode_action(x))})
        print("  accepted #%d at draw %3d: peak %.4f GHz  boost %6.2f  dc %6.2f"
              % (len(chosen), n_draw, pk, m.boost_db, m.dc_gain_db), flush=True)

    Path(args.out).write_text(json.dumps(
        {"select_seed": seed, "n_select": N_SELECT, "max_draws": MAX_DRAWS,
         "sweep_floor_ghz": SWEEP_FLOOR_GHZ, "band_lo_ghz": lo,
         "select_channel_db": SELECT_CHANNEL_DB, "select_target_db": SELECT_TARGET_DB,
         "n_draws_used": n_draw, "n_guard_valid_seen": n_valid,
         "complete": len(chosen) == N_SELECT, "designs": chosen}, indent=1))
    print("\n%d drawn, %d guard-valid, %d accepted below the band"
          % (n_draw, n_valid, len(chosen)), flush=True)
    if len(chosen) < N_SELECT:
        print("STOPPED SHORT of %d -- reporting what was found rather than widening the "
              "rule" % N_SELECT, flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
