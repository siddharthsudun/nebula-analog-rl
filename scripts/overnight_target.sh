#!/usr/bin/env bash
# Validation for seq_target40k -- the first run where hitting the requested target boost is
# both scored and rewarded (--boost-tol 1.5).
#
# THE CONTROL MATTERS MORE THAN THE HEADLINE. A better number under the new test could come
# from the new training OR from nothing at all, so the old model is re-tested under the same
# strict rule. If seq_clean40k scores the same, the training did nothing and the change is
# cosmetic -- the same trap the anchored 32/32 fell into.
set -u
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe
HB=results/seq_target40k_heartbeat.json
TOL=1.5

echo "[1/6] waiting for seq_target40k"
until grep -qE '"state": "(done|stalled|restart_limit|failed)"' "$HB" 2>/dev/null; do sleep 120; done
$PY -c "import json;d=json.load(open('$HB'));print('  state=%s steps=%s attempts=%s stalls=%s'%(d['state'],d['steps_reached'],d['attempts'],d['stalls']))"

NEW=results/seq_target40k.zip
OLD=results/seq_clean40k.zip
[ -f "$NEW" ] || NEW=$(ls -1 results/checkpoints/seq_target40k_*_steps.zip | sed 's/.*_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)

echo "[2/6] NEW model, STRICT test (+/- ${TOL} dB on target) -- the honest retargeting number"
$PY -m eqrl.experiments.policy_rollout --model "$NEW" --specs 32 --boost-tol $TOL \
    --out results/rollout_target40k_strict.json 2>&1 | grep -E "solved [0-9]+/|median"

echo "[3/6] CONTROL: OLD model, SAME strict test -- did the training do the work?"
$PY -m eqrl.experiments.policy_rollout --model "$OLD" --specs 32 --boost-tol $TOL \
    --out results/rollout_clean40k_strict.json 2>&1 | grep -E "solved [0-9]+/|median"

echo "[4/6] NEW model, OLD loose test -- did we lose general solve ability? (was 26/32)"
$PY -m eqrl.experiments.policy_rollout --model "$NEW" --specs 32 \
    --out results/rollout_target40k_loose.json 2>&1 | grep -E "solved [0-9]+/|median"

echo "[5/6] reward audit -- the baseline fix must still hold under the new margin"
$PY -m eqrl.experiments.reward_audit --model "$NEW" --episodes 12 \
    --out results/reward_audit_target40k.json 2>&1 | grep -viE "^Note:|^Warning" | tail -12

echo "[6/6] target tracking: correlation between requested and achieved boost"
$PY - <<'PYX' 2>&1 | grep -viE "^Note:|^Warning"
import json, dataclasses, statistics as st
import numpy as np
from eqrl.circuits.ctle import DesignVars
from eqrl.evaluator import build_evaluator
from eqrl.specs import DEFAULT_SPEC

fields = {f.name for f in dataclasses.fields(DesignVars)}
guards, summary = {}, {}
for tag, path in [("NEW/strict", "results/rollout_target40k_strict.json"),
                  ("NEW/loose",  "results/rollout_target40k_loose.json"),
                  ("OLD/strict", "results/rollout_clean40k_strict.json")]:
    try:
        rows = json.load(open(path))["rows"]
    except Exception as e:
        print("  %-11s unreadable: %s" % (tag, e)); continue
    pts = []
    for r in rows:
        if r.get("solved_at") is None:
            continue
        ch = r["channel_loss_db"]
        guards.setdefault(ch, build_evaluator(DEFAULT_SPEC, corner="tt", fast=False,
                                              channel_loss_db=ch))
        dv = DesignVars(**{k: v for k, v in r["design"].items() if k in fields})
        v = guards[ch].evaluate(dv, vdd=DEFAULT_SPEC.vdd_nominal)
        if v.is_valid:
            pts.append((r["target_boost_db"], v.unwrap().boost_db))
    if len(pts) < 3:
        print("  %-11s only %d solved designs; correlation not meaningful" % (tag, len(pts)))
        summary[tag] = {"n": len(pts)}
        continue
    t = [a for a, _ in pts]; b = [c for _, c in pts]
    corr = float(np.corrcoef(t, b)[0, 1])
    errs = sorted(abs(c - a) for a, c in pts)
    print("  %-11s n=%2d  corr=%+.3f  median|err|=%.2f dB  within 0.5/1.0/1.5: %d/%d/%d"
          % (tag, len(pts), corr, st.median(errs),
             sum(e <= 0.5 for e in errs), sum(e <= 1.0 for e in errs),
             sum(e <= 1.5 for e in errs)))
    summary[tag] = {"n": len(pts), "corr": corr, "median_abs_err": st.median(errs)}
json.dump(summary, open("results/target_tracking_compare.json", "w"), indent=1)
print("\n  baseline for comparison: OLD model / OLD loose test was corr=+0.114, n=26")
PYX
echo "TARGET40K VALIDATION COMPLETE"
