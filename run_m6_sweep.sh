#!/usr/bin/env bash
# M6 parameter-count scaling sweep: 4 configs x 3 seeds, equal 3000-step budget.
# Runs sequentially on CPU (parallel CPU runs would contend and skew timing).
# Does NOT touch the frozen checkpoints/pico/ v0.1 release.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate

SEEDS="1337 42 7"
CONFIGS="m6_tiny m6_small m6_pico m6_big"
mkdir -p checkpoints/m6

for cfg in $CONFIGS; do
  for s in $SEEDS; do
    out="checkpoints/m6/${cfg}_s${s}"
    echo "=================== RUN ${cfg} seed ${s} -> ${out} ==================="
    python train.py --config "configs/${cfg}.yaml" --seed "$s" --out-dir "$out" \
      --run-name "${cfg}_s${s}" --device cpu 2>&1 \
      | grep -vE "UserWarning|functional_tensor"
    echo "--- exit code: ${PIPESTATUS[0]} ---"
  done
done
echo "SWEEP_COMPLETE"
