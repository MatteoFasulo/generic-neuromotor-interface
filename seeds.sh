#!/bin/bash
set -e

SEEDS=(1379 8642 5601 1234 42)
EXPERIMENT="discrete_gestures"

for seed in "${SEEDS[@]}"; do
  echo "Running experiment=$EXPERIMENT with seed=$seed"
  HYDRA_FULL_ERROR=1 python -m generic_neuromotor_interface.train --config-name=${EXPERIMENT} seed=${seed} eval=False
done
