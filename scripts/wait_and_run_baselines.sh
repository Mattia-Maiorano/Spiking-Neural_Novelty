#!/usr/bin/env bash
set -euo pipefail

TARGET_PID=35467
echo "Waiting for external training process (PID ${TARGET_PID}) to finish..."
while kill -0 "${TARGET_PID}" 2>/dev/null; do
    sleep 5
done

echo "External training process finished. CPU/GPU is now free."
echo "Launching comparative baseline benchmark training on MPS..."

.venv/bin/python experiments/train_baselines.py \
    --epochs 20 \
    --patience 6 \
    --total-trajectories 1000 \
    --batch-size 32 \
    --device mps \
    --output-dir results/baselines

echo "Comparative baseline benchmark completed successfully."

