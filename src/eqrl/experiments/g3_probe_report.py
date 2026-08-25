"""Reads g3_probe_check and answers one question: can the probe steer?

THE GATE IS AGREEMENT ON THE DIMENSIONS THE PROBE ITSELF CALLS INFLUENTIAL. A pooled
agreement number over all six axes is not the quantity of interest and is actively
misleading: most axes barely move boost at the handoff, their sign is decided by
rounding, and they drag any pooled figure toward 50% no matter how well the probe works
where it matters. A controller only ever commits moves along axes it believes are
influential, so that subset is what has to be right.

    influential   |predicted delta over the verify step| >= 0.25 dB, the probe's own call
    agreement     sign(predicted delta) == sign(measured delta), two independent SPICE
                  measurements, the probe at h and the verification at the larger s

Reported against a 50% coin, one-sided, because the null is "the probe knows nothing about
direction". Also reported for the non-influential axes, where agreement near chance is the
EXPECTED result and not a failure -- printing only the flattering stratum would be the same
error as reporting only the kinder budget read.

SECONDARY, and they decide affordability rather than correctness:
  * magnitude calibration -- predicted vs measured delta in dB. Direction alone gets a
    controller moving; magnitude is what lets it size a step instead of guessing.
  * per-dimension influence rate -- how often each axis is worth probing at all, which is
    the input to any later version that stops probing flat axes.
  * probe waste -- fraction of probe simulations that come back guard-INVALID.

Reads one artifact. Runs no simulation, loads no policy, changes no threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default="results/g3_probe_check.json")
    args = p.parse_args()

    d = json.loads(Path(args.artifact).read_text())
    rows, thr, s = d["rows"], d["influential_db"], d["verify_s"]
    hs = [str(h) for h in d["probe_h"]]

    print("=" * 94)
    print("G3 PRIMITIVE: does a finite-difference SPICE probe predict the direction of a "
          "real step?")
    print("  %s | spec-seed %s | %d specs | verify step %.2f | influential >= %.2f dB"
          % (args.artifact, d["spec_seed"], len(rows), s, thr))
    print("  complete=%s | %d measure_all" % (d.get("complete"), d["n_measure_all"]))
    print("=" * 94)

    valid = [r for r in rows if r["handoff_valid"]]
    print("\n  %d of %d handoffs were guard-valid and could be probed at all."
          % (len(valid), len(rows)))
    if not valid:
        print("  nothing to analyse.")
        return

    dims = list(valid[0]["dims"])
    n_probe = n_bad = 0

    for h in hs:
        obs = []          # (predicted_delta_db, measured_delta_db, dim)
        for r in valid:
            for name in dims:
                pr = r["dims"][name]["probes"][h]
                n_probe += 2
                n_bad += int(not pr["plus_valid"]) + int(not pr["minus_valid"])
                if pr["slope_db_per_unit"] is None:
                    continue
                # Both verification directions are independent measurements of the same
                # slope estimate, so both are used; the prediction flips sign with the step.
                for tag, sgn in (("plus", +1.0), ("minus", -1.0)):
                    vf = r["dims"][name]["verify"][tag]
                    if vf["delta_db"] is None:
                        continue
                    obs.append((sgn * pr["pred_delta_db"], vf["delta_db"], name))

        infl = [(p_, m, n) for p_, m, n in obs if abs(p_) >= thr]
        flat = [(p_, m, n) for p_, m, n in obs if abs(p_) < thr]
        print("\n--- probe h = %s " % h + "-" * 72)
        print("  %d usable predictions: %d on influential axes, %d on near-flat axes"
              % (len(obs), len(infl), len(flat)))

        for label, group in (("INFLUENTIAL (the gate)", infl), ("near-flat", flat)):
            if not group:
                print("  %-22s  no observations" % label)
                continue
            ok = sum(1 for p_, m, _ in group if np.sign(p_) == np.sign(m))
            pv = stats.binomtest(ok, len(group), 0.5, alternative="greater").pvalue
            print("  %-22s  sign agreement %3d/%-3d = %5.1f%%   p(vs coin) %.2e%s"
                  % (label, ok, len(group), 100 * ok / len(group), pv,
                     "   <- steers" if (pv < 0.05 and label.startswith("INFL")) else ""))

        if infl:
            err = [abs(p_ - m) for p_, m, _ in infl]
            rel = [abs(p_ - m) / max(abs(m), 1e-9) for p_, m, _ in infl]
            print("  magnitude calibration   median |pred - measured| %.2f dB   "
                  "median relative error %.0f%%"
                  % (float(np.median(err)), 100 * float(np.median(rel))))

    # Which axes are worth probing -- the input to an adaptive version.
    h = hs[-1]
    print("\n--- per-dimension influence at h = %s " % h + "-" * 55)
    print("  dimension   influential   median |slope|      agreement when influential")
    for name in dims:
        sl, hit, tot = [], 0, 0
        for r in valid:
            pr = r["dims"][name]["probes"][h]
            if pr["slope_db_per_unit"] is None:
                continue
            sl.append(abs(pr["pred_delta_db"]))
            for tag, sgn in (("plus", +1.0), ("minus", -1.0)):
                vf = r["dims"][name]["verify"][tag]
                if vf["delta_db"] is None or abs(pr["pred_delta_db"]) < thr:
                    continue
                tot += 1
                hit += int(np.sign(sgn * pr["pred_delta_db"]) == np.sign(vf["delta_db"]))
        n_infl = sum(1 for v in sl if v >= thr)
        print("  %-11s %2d/%-2d  %5.0f%%   %7.2f dB          %s"
              % (name, n_infl, len(sl), 100 * n_infl / max(len(sl), 1),
                 float(np.median(sl)) if sl else float("nan"),
                 "-" if not tot else "%d/%d = %.0f%%" % (hit, tot, 100 * hit / tot)))

    print("\n--- cost " + "-" * 84)
    print("  %d probe simulations, %d guard-INVALID (%.0f%% wasted)"
          % (n_probe, n_bad, 100 * n_bad / max(n_probe, 1)))
    print("  %d measure_all for %d specs = %.0f per spec, against H1's 20 for a whole spec"
          % (d["n_measure_all"], len(rows), d["n_measure_all"] / max(len(rows), 1)))


if __name__ == "__main__":
    main()
