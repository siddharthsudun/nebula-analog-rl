#!/usr/bin/env bash
# Finish the step_size experiment, then run the submission's central measurement.
set -u
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe

echo "[1/4] waiting for the step_size=0.09 run to finish"
until grep -qE '"state": "(done|stalled|restart_limit)"' results/seq_step009_heartbeat.json 2>/dev/null; do sleep 60; done
$PY -c "import json;d=json.load(open('results/seq_step009_heartbeat.json'));print('  ',d['state'],d['steps_reached'],'steps, attempts',d['attempts'])"

echo "[2/4] rolling out step_size=0.09 on the SAME 32 held-out specs"
$PY -m silq.experiments.policy_rollout --model results/seq_step009.zip --specs 32 \
    --out results/rollout_step009_32.json 2>&1 | grep -E "solved [0-9]+/|median"

echo "[3/4] choosing the model to benchmark"
BEST=$($PY - <<'PYX'
import json
from pathlib import Path
cands = {"results/seq_dcfix40k.zip": "results/rollout_best_32spec.json",
         "results/seq_step009.zip": "results/rollout_step009_32.json"}
best, bn = "results/seq_dcfix40k.zip", -1
for model, res in cands.items():
    f = Path(res)
    if not f.exists():
        continue
    n = json.loads(f.read_text())["solved"]
    print(f"  {model}: {n}/32", flush=True)
    if n > bn:
        best, bn = model, n
print(f"  -> {best}", flush=True)
Path("results/best_model.txt").write_text(best)
PYX
)
echo "$BEST"
MODEL=$(cat results/best_model.txt)

echo "[4/4] amortization benchmark: RL vs random vs CMA-ES vs TPE"
# --require-valid and --dc-gain-db-min 0 make every method answer the same question the
# guard layer asks, so a design that passes on paper but is not a buildable circuit does
# not count as a success for anyone.
$PY -m silq.experiments.honest_benchmark --model "$MODEL" \
    --rl-specs 32 --search-specs 6 --seeds 5 --budget 150 \
    --require-valid --dc-gain-db-min 0 --outdir results 2>&1 | grep -vE "^Note:"
echo "OVERNIGHT COMPLETE"
