#!/usr/bin/env bash
# Extend the curve over the checkpoints written after the resume, roll out the final
# model, then re-measure the best checkpoint on a much larger held-out set.
#
# Eight specs cannot separate 1/8 from 2/8 -- at n=8 that difference is one spec and well
# inside sampling noise, so the peak in the curve is suggestive and not yet a measurement.
# The larger set is what makes the headline number defensible.
set -u
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe
bash scripts/rollout_curve.sh 2>&1 | grep -E "steps|solved|/8|wrote"

echo "=== final model ==="
$PY -m silq.experiments.policy_rollout --model results/seq_dcfix40k.zip --specs 8 \
    --out results/rollouts/rollout_final.json 2>&1 | grep -E "spec [0-9]|solved [0-9]+/"

echo "=== best checkpoint, 32 held-out specs ==="
BEST=$($PY - <<'PYX'
import json, re
from pathlib import Path
best, bs = None, -1
for f in Path("results/rollouts").glob("rollout_*.json"):
    m = re.search(r"rollout_(\d+)\.json", f.name)
    if not m:
        continue
    d = json.loads(f.read_text())
    # Ties go to the EARLIER checkpoint: fewer steps for the same solve rate is the
    # better model to report, and it avoids crediting a late lucky update.
    if d["solved"] > bs:
        best, bs = int(m.group(1)), d["solved"]
print(best)
PYX
)
echo "best checkpoint: $BEST steps"
$PY -m silq.experiments.policy_rollout \
    --model "results/checkpoints/seq_dcfix40k_${BEST}_steps.zip" --specs 32 \
    --out "results/rollout_best_32spec.json" 2>&1 | grep -E "spec [0-9]|solved [0-9]+/|median|ratio|sweep"
