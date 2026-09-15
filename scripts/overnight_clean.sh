#!/usr/bin/env bash
# Post-training validation for seq_clean40k -- the first run with BOTH the reward-baseline
# fix and the corrected 8-16 dB channel range. Waits for the supervisor to finish, then
# answers three questions in order of importance:
#   1. Did the reward fix hold in training? (returns must sit under the ~25 honest ceiling)
#   2. What is the honest, UNANCHORED 32-spec number, comparable to the old 2/32?
#   3. Is a mid-training checkpoint better than the final model?
# Nothing here changes a threshold or a bound. It only measures.
set -u
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe
HB=results/seq_clean40k_heartbeat.json

echo "[1/5] waiting for seq_clean40k to finish"
until grep -qE '"state": "(done|stalled|restart_limit|failed)"' "$HB" 2>/dev/null; do sleep 120; done
$PY -c "import json;d=json.load(open('$HB'));print('  state=%s steps=%s attempts=%s stalls=%s'%(d['state'],d['steps_reached'],d['attempts'],d['stalls']))"

MODEL=results/seq_clean40k.zip
if [ ! -f "$MODEL" ]; then
  MODEL=$(ls -1 results/checkpoints/seq_clean40k_*_steps.zip | sed 's/.*_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  echo "  no final zip; falling back to highest checkpoint: $MODEL"
fi

echo "[2/5] reward audit -- does the fix hold? (honest ceiling ~25)"
$PY -m silq.experiments.reward_audit --model "$MODEL" --episodes 12 \
    --out results/reward_audit_clean40k.json 2>&1 | grep -viE "^Note:|^Warning" | tail -25

echo "[3/5] UNANCHORED 32-spec rollout of the final model (compare against the old 2/32)"
$PY -m silq.experiments.policy_rollout --model "$MODEL" --specs 32 \
    --out results/rollout_clean40k_32.json 2>&1 | grep -E "spec [0-9]|solved [0-9]+/|median|invalid"

echo "[4/5] checkpoint curve, 8 specs, every other checkpoint"
for CK in $(ls -1 results/checkpoints/seq_clean40k_*_steps.zip | sed 's/.*_\([0-9]*\)_steps.zip/\1 &/' | sort -n | awk 'NR%2==1' | cut -d' ' -f2); do
  N=$(echo "$CK" | sed 's/.*_\([0-9]*\)_steps.zip/\1/')
  R=$($PY -m silq.experiments.policy_rollout --model "$CK" --specs 8 \
        --out "results/rollouts/clean40k_${N}.json" 2>&1 | grep -oE "solved [0-9]+/[0-9]+" | tail -1)
  echo "  ${N} steps: ${R:-no result}"
done

echo "[5/5] best checkpoint from the curve, at 32 specs"
BEST=$($PY - <<'PYX'
import json, re
from pathlib import Path
best, bs = None, -1
for f in Path("results/rollouts").glob("clean40k_*.json"):
    m = re.search(r"clean40k_(\d+)\.json", f.name)
    if not m: continue
    try: d = json.loads(f.read_text())
    except Exception: continue
    # Ties go to the EARLIER checkpoint: same solve rate for fewer steps is the better
    # model to report, and it avoids crediting one lucky late update.
    if d.get("solved", -1) > bs:
        best, bs = int(m.group(1)), d["solved"]
print(best if best is not None else "")
PYX
)
if [ -n "$BEST" ]; then
  echo "  best on the 8-spec curve: $BEST steps"
  $PY -m silq.experiments.policy_rollout \
      --model "results/checkpoints/seq_clean40k_${BEST}_steps.zip" --specs 32 \
      --out results/rollout_clean40k_best32.json 2>&1 | grep -E "solved [0-9]+/|median|invalid"
else
  echo "  curve produced no readable results; skipping"
fi
echo "CLEAN40K VALIDATION COMPLETE"
