"""G3 PRIMITIVE TEST: can a finite-difference SPICE probe predict the DIRECTION of a step?

THIS IS NOT THE G3 CONTROLLER. It builds nothing on top of the probe. The surrogate was
built into an architecture and gated against labels its own pipeline produced, and it took
an off-distribution probe with independent ground truth to notice the labels were wrong
(section 19). The lesson recorded there was: measure the primitive first, against something
the mechanism did not itself produce. So before any controller exists, this asks the one
question the whole of Generation 3 rests on --

    if a small probe says "increasing rs raises boost", does a refinement-sized step in
    that direction actually raise boost when SPICE is asked?

-- and it answers it with two independent SPICE measurements: the probe at step h, and the
verification at a larger step s. The probe never grades itself.

THE MEASUREMENT, per spec and per action dimension i:

    probe        b(x0 + h e_i), b(x0 - h e_i)     ->  central difference g_i
    verify       b(x0 + s e_i), b(x0 - s e_i)     ->  actual delta at a real step size

    AGREEMENT    sign(g_i) == sign(b(x0 + s e_i) - b(x0))

s is deliberately LARGER than h. A probe that only predicts its own step size is useless to
a controller, which has to commit a move it did not measure. The verify points are shared
across every h, so adding a second probe size costs 12 evaluations rather than 24.

WHAT WOULD KILL GENERATION 3. Not "agreement below 100%" -- the boost surface is nonlinear
and some dimensions barely move it, so some disagreement is expected and is not the
question. What would kill it is agreement near 50% on the dimensions the probe itself calls
INFLUENTIAL. A probe that cannot tell which way is uphill on the axes it is most confident
about cannot steer, and no amount of controller design fixes that. Agreement is therefore
reported stratified by |g_i|, not as a single pooled number: a pooled figure is dominated
by the near-flat axes where the sign is meaningless by construction.

Also recorded, because both decide whether the idea is affordable rather than merely
correct: how often a probe point is guard-INVALID (a wasted simulation), and the local
sensitivity |g_i| per dimension (which is what a later adaptive version would use to stop
probing axes that do not matter).

Stage 1 is regenerated with the code hybrid_audit uses, unchanged and deterministic, so
these are the SAME handoff designs the committed H1 runs used.

Changes no threshold, no bound, no reward, no model. Runs SPICE; loads the frozen policy.
"""
from __future__ import annotations

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

import argparse
import numpy as np

from silq.circuits.ctle import ACTION_SPACE, decode_action
from silq.evaluator import build_evaluator
from silq.specs import DEFAULT_SPEC, hard_pass
from silq.experiments.target_audit import make_specs

DIMS = list(ACTION_SPACE.keys())

# The step sizes are declared here, before the run, and are not tuned afterwards.
#   h: probe sizes. 0.02 and 0.05 bracket the CMA-ES sigma0 the refiner already uses.
#   s: the verification step -- larger than every h on purpose (see module docstring).
PROBE_H = (0.02, 0.05)
VERIFY_S = 0.10
# A dimension counts as INFLUENTIAL when the probe's own slope estimate implies at least
# this much boost change over the verification step. It is a readout threshold for
# stratifying the report, not a criterion anything passes or fails.
INFLUENTIAL_DB = 0.25


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/seq_clean40k.zip")
    p.add_argument("--specs", type=int, default=8)
    p.add_argument("--spec-seed", type=int, default=2)
    p.add_argument("--k", type=int, default=5, help="stage-1 evals, as H1 (section 18.4)")
    p.add_argument("--out", default="results/g3_probe_check.json")
    args = p.parse_args()

    specs = make_specs(32, args.spec_seed)[:args.specs]
    guards: dict[float, object] = {}
    n_sim = 0

    def guard_for(channel: float):
        if channel not in guards:
            guards[channel] = build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=channel)
        return guards[channel]

    def evaluate(x, target, channel):
        """One verified evaluation, identical to hybrid_audit.evaluate. None = rejected."""
        nonlocal n_sim
        n_sim += 1
        dv = decode_action(np.asarray(x))
        spec = dataclasses.replace(DEFAULT_SPEC, target_boost_db=target,
                                   channel_loss_db=channel)
        try:
            v = guard_for(channel).evaluate(dv, vdd=spec.vdd_nominal)
        except Exception:
            return None
        if not v.is_valid:
            return None
        m = v.unwrap()
        ok, _ = hard_pass(m, spec)
        return {"boost_db": float(m.boost_db), "loose_pass": bool(ok)}

    from stable_baselines3 import PPO
    from silq.envs.sequential_env import SequentialEqualizerEnv
    model = PPO.load(args.model)
    env = SequentialEqualizerEnv(fast=False, seed=123, guarded=False)

    def stage1(i, target, channel):
        """Byte-identical to hybrid_audit.stage1: k evals, NO reset on terminated."""
        env.reset(seed=1000 + i)
        env._target, env._channel = target, channel
        obs = env._obs(env._measure(env._x))
        n = 1
        evaluate(env._x, target, channel)
        while n < args.k:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, _term, _trunc, _ = env.step(a)
            evaluate(env._x, target, channel)
            n += 1
        return np.array(env._x, dtype=np.float64)

    print("G3 PRIMITIVE PROBE CHECK | %d specs, spec-seed %d | probe h=%s, verify s=%.2f"
          % (len(specs), args.spec_seed, PROBE_H, VERIFY_S), flush=True)
    print("  asks ONLY whether a probe predicts the sign of a larger real SPICE step.\n",
          flush=True)

    rows = []
    for i, (target, channel) in enumerate(specs):
        x0 = stage1(i, target, channel)
        base = evaluate(x0, target, channel)
        rec = {"spec": i, "target_boost_db": target, "channel_loss_db": channel,
               "handoff_valid": base is not None,
               "handoff_boost_db": None if base is None else base["boost_db"],
               "dims": {}}
        if base is None:
            # Probing from an invalid design is a different question (there is no baseline
            # boost to difference against), and is recorded rather than silently skipped.
            print("  spec %2d  handoff INVALID -- no baseline to difference against" % i,
                  flush=True)
            rows.append(rec)
            Path(args.out).write_text(json.dumps(
                {"probe_h": PROBE_H, "verify_s": VERIFY_S,
                 "influential_db": INFLUENTIAL_DB, "spec_seed": args.spec_seed,
                 "k": args.k, "model": args.model, "n_measure_all": n_sim,
                 "complete": False, "rows": rows}, indent=1))
            continue

        b0 = base["boost_db"]
        for d, name in enumerate(DIMS):
            e = np.zeros(len(DIMS))
            e[d] = 1.0
            entry = {"probes": {}, "verify": {}}

            # -- verification points, shared by every probe size --------------------
            for sgn, tag in ((+1.0, "plus"), (-1.0, "minus")):
                xv = np.clip(x0 + sgn * VERIFY_S * e, 0.0, 1.0)
                step = float(xv[d] - x0[d])           # the step actually taken after clip
                r = evaluate(xv, target, channel)
                entry["verify"][tag] = {
                    "step": step, "valid": r is not None,
                    "boost_db": None if r is None else r["boost_db"],
                    "delta_db": None if r is None else r["boost_db"] - b0}

            # -- probes -------------------------------------------------------------
            for h in PROBE_H:
                xp = np.clip(x0 + h * e, 0.0, 1.0)
                xm = np.clip(x0 - h * e, 0.0, 1.0)
                span = float(xp[d] - xm[d])
                rp = evaluate(xp, target, channel)
                rm = evaluate(xm, target, channel)
                g = None
                if rp is not None and rm is not None and span > 0:
                    g = (rp["boost_db"] - rm["boost_db"]) / span
                entry["probes"][str(h)] = {
                    "slope_db_per_unit": g, "span": span,
                    "plus_valid": rp is not None, "minus_valid": rm is not None,
                    # what the probe PREDICTS the verification step will do
                    "pred_delta_db": None if g is None else g * VERIFY_S}
            rec["dims"][name] = entry

        # A compact per-spec line: agreement at the larger probe size, influential dims only
        agree = tot = 0
        for name in DIMS:
            pr = rec["dims"][name]["probes"][str(PROBE_H[-1])]
            vf = rec["dims"][name]["verify"]["plus"]
            if (pr["slope_db_per_unit"] is None or vf["delta_db"] is None
                    or abs(pr["pred_delta_db"]) < INFLUENTIAL_DB):
                continue
            tot += 1
            agree += int(np.sign(pr["pred_delta_db"]) == np.sign(vf["delta_db"]))
        print("  spec %2d  target %5.2f  handoff boost %6.2f | sign agreement %d/%d "
              "on influential dims | %d sims" % (i, target, b0, agree, tot, n_sim),
              flush=True)

        rows.append(rec)
        Path(args.out).write_text(json.dumps(
            {"probe_h": PROBE_H, "verify_s": VERIFY_S, "influential_db": INFLUENTIAL_DB,
             "spec_seed": args.spec_seed, "k": args.k, "model": args.model,
             "n_measure_all": n_sim, "complete": False, "rows": rows}, indent=1))

    Path(args.out).write_text(json.dumps(
        {"probe_h": PROBE_H, "verify_s": VERIFY_S, "influential_db": INFLUENTIAL_DB,
         "spec_seed": args.spec_seed, "k": args.k, "model": args.model,
         "n_measure_all": n_sim, "complete": True, "rows": rows}, indent=1))
    print("\n%d measure_all total. wrote %s" % (n_sim, args.out), flush=True)


if __name__ == "__main__":
    main()
