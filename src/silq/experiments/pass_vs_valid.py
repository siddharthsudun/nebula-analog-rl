"""Do the designs that "pass all 8 specs" survive the guard?

honest_benchmark.evaluate() scores a candidate with hard_pass() on a direct measure_all()
call. It never consults the guard layer. So the search baselines are solving

    "meet the 8 specs"

while the RL agent was trained against

    "meet the 8 specs AND be a physically valid circuit"

Those are different problems, and the easier one is the baselines'. A mirror sitting in
triode still produces an AC response with gain and peaking, so a design can satisfy every
spec while not being the circuit it claims to be. CMA-ES finding a full pass in 2
evaluations, on a space measured at 0.4-0.7% guard-valid, is what prompted this.

Two questions, one experiment:

  1. FAIRNESS  -- of the designs the baselines declare solved, how many are guard-valid?
                  If few, the published comparison is measuring two different tasks.
  2. FEASIBILITY -- does ANY guard-valid design pass all 8 specs? If not, this is a
                  topology or spec problem, and no amount of RL training addresses it.

Nothing here changes behaviour; it only measures.
"""
import collections
import dataclasses
import json
import os
from pathlib import Path

P = Path(os.environ.get("USERPROFILE") or Path.home()) / "silq-ngspice"
os.environ.setdefault("NGSPICE_LIBRARY_PATH", str(P / "Library" / "bin" / "ngspice{}.dll"))
os.environ.setdefault("SPICE_LIB_DIR", str(P / "Library" / "share" / "ngspice"))
os.environ.setdefault("PDK_ROOT", str(Path(os.environ.get("USERPROFILE") or Path.home()) / "pdk"))
os.environ["PATH"] = os.pathsep.join([str(P / "shim"), str(P / "Library" / "bin"), os.environ["PATH"]])
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(P / "Library" / "bin"))

import numpy as np

from silq.circuits.ctle import ACTION_SPACE, decode_action
from silq.evaluator import build_evaluator
from silq.sim.measures import measure_all
from silq.specs import DEFAULT_SPEC, hard_pass

N = len(ACTION_SPACE)
VDD = 1.8
ev = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False)


def spec_pass(dv, target, channel):
    """Exactly what honest_benchmark scores on: unguarded measure + hard_pass.

    `dc_gain_db_min=None` IS THE POINT OF THIS EXPERIMENT, and it is pinned rather than
    inherited. The question here is whether designs meeting the PUBLISHED eight-check spec
    are real circuits, so the eight-check spec is the correct one to score against --
    `honest_benchmark._DC_GAIN_DB_MIN` defaults to None for the same reason, and this
    function's first line claims to match it.

    IT STOPPED MATCHING, SILENTLY, AND THIS PIN IS THE REPAIR. Both this file and
    `results/pass_vs_valid.json` were committed in 6d465c93a (17 Aug 2026), when
    `Spec.dc_gain_db_min` was still None. 2f3ec52b3 (18 Aug 2026) flipped that default to
    0.0 -- correctly, it closes the attenuate-at-DC hole -- and this file was never touched
    again. So `replace(DEFAULT_SPEC, ...)` began yielding NINE checks while the shipped
    artifact, and `docs/PASS_VS_VALID.md` which cites it, describe eight.

    The artifact is not wrong; it is self-proving at eight. 14 of its 28 spec-passing
    designs are rejected as T4.10_dc_gain_implausible, i.e. DC gain below 0 dB, which no
    design that had also cleared a `dc_gain >= 0 dB` hard check could be. But re-running
    this script at HEAD would have scored nine and produced a different split, so anyone
    checking the document against the code would have found them in contradiction and had
    no way to tell which was right. Pinning makes the script reproduce its own artifact.
    """
    spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                               channel_loss_db=channel, dc_gain_db_min=None)
    m = measure_all(dv, corner="tt", vdd=spec.vdd_nominal, fast=False,
                    channel_loss_db=channel)
    if not m.ok:
        return False, None, {}
    ok, checks = hard_pass(m, spec)
    return ok, m, checks


def guard_valid(dv):
    try:
        v = ev.evaluate(dv, vdd=VDD)
    except Exception as e:
        return False, f"EXC:{type(e).__name__}"
    return (True, None) if v.is_valid else (False, v.check.value)


def main():
    rng = np.random.default_rng(0)
    SPECS = [(float(rng.uniform(5, 11)), float(rng.uniform(8, 16))) for _ in range(4)]
    BUDGET = 60

    cells = collections.Counter()
    why_invalid = collections.Counter()
    passing_designs = []

    print("Searching for spec-passing designs the way the benchmark does, then asking "
          "the guard about each one.\n")

    for si, (target, channel) in enumerate(SPECS):
        import cma
        es = cma.CMAEvolutionStrategy(N * [0.5], 0.25,
                                      {"bounds": [0, 1], "maxfevals": BUDGET,
                                       "seed": si + 1, "verbose": -9})
        used = 0
        found = 0
        while not es.stop() and used < BUDGET:
            xs = es.ask()
            costs = []
            for x in xs:
                dv = decode_action(np.asarray(x))
                ok, m, checks = spec_pass(dv, target, channel)
                used += 1
                if ok:
                    found += 1
                    valid, reason = guard_valid(dv)
                    cells["pass_valid" if valid else "pass_invalid"] += 1
                    if not valid:
                        why_invalid[reason] += 1
                    passing_designs.append({
                        "target_boost_db": target, "channel_loss_db": channel,
                        "design": dv.__dict__, "guard_valid": valid,
                        "guard_reason": reason,
                        "boost_db": getattr(m, "boost_db", None),
                        "eye_v_mv": getattr(m, "eye_v_mv", None),
                    })
                    costs.append(-10.0)
                else:
                    score = (sum(1.0 for v in checks.values() if v) if checks else -10.0)
                    costs.append(-score)
            es.tell(xs, costs)
        print(f"  spec {si} (boost {target:.1f} dB, channel {channel:.1f} dB): "
              f"{found} spec-passing designs in {used} evaluations", flush=True)

    total_pass = cells["pass_valid"] + cells["pass_invalid"]
    print(f"\nDesigns that pass all 8 specs: {total_pass}")
    if total_pass:
        print(f"   guard-VALID   : {cells['pass_valid']:4d}  "
              f"({cells['pass_valid']/total_pass*100:.1f}%)")
        print(f"   guard-INVALID : {cells['pass_invalid']:4d}  "
              f"({cells['pass_invalid']/total_pass*100:.1f}%)")
        for k, n in why_invalid.most_common():
            print(f"        {n:4d}  {k}")

    print("\nANSWERS")
    if total_pass == 0:
        print("  fairness    : no spec-passing design found in this budget; inconclusive.")
        print("  feasibility : none found. Widen the budget before concluding anything.")
    else:
        frac = cells["pass_valid"] / total_pass
        print(f"  fairness    : {(1-frac)*100:.0f}% of 'solved' designs are NOT valid "
              f"circuits. The baselines are solving an easier problem than the agent."
              if frac < 0.9 else
              "  fairness    : spec-passing designs are essentially all guard-valid; "
              "the comparison is sound on this axis.")
        print(f"  feasibility : {cells['pass_valid']} design(s) both pass all 8 specs and "
              f"survive the guard." if cells["pass_valid"] else
              "  feasibility : NO design both passes the specs and is a valid circuit. "
              "That is a topology/spec problem, not an RL problem.")

    Path("results").mkdir(exist_ok=True)
    Path("results/pass_vs_valid.json").write_text(json.dumps({
        "specs": SPECS, "budget_per_spec": BUDGET,
        "pass_valid": cells["pass_valid"], "pass_invalid": cells["pass_invalid"],
        "why_invalid": dict(why_invalid.most_common()),
        "designs": passing_designs[:40],
    }, indent=2))
    print("\nwrote results/pass_vs_valid.json")


if __name__ == "__main__":
    main()
