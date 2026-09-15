#!/usr/bin/env bash
# REPRODUCTION AUDIT -- re-runs the three seq_clean40k measurements that the submission
# depends on, writing to NEW files so the originals stay intact for comparison.
# Changes nothing. Reads only.
set -u
export PYTHONPATH=src
PY=./.venv/Scripts/python.exe
M=results/seq_clean40k.zip

echo "[1/3] LOOSE 32-spec rollout  (original reported 26/32, median 4)"
$PY -m silq.experiments.policy_rollout --model "$M" --specs 32 \
    --out results/repro_clean40k_loose.json 2>&1 | grep -E "solved [0-9]+/|median"

echo "[2/3] STRICT 32-spec rollout, +/-1.5 dB  (original reported 10/32, median 4.5)"
$PY -m silq.experiments.policy_rollout --model "$M" --specs 32 --boost-tol 1.5 \
    --out results/repro_clean40k_strict.json 2>&1 | grep -E "solved [0-9]+/|median"

echo "[3/3] reward audit  (original: valid-after-valid +4.355, after-invalid -0.047)"
$PY -m silq.experiments.reward_audit --model "$M" --episodes 12 \
    --out results/repro_reward_audit.json 2>&1 | grep -viE "^Note:|^Warning" | tail -12
echo "REPRO COMPLETE"
