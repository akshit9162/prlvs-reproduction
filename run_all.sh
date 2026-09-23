#!/bin/bash
# Full PRLVS reproduction pipeline. Waits for features, sanity-checks the
# training loop cheaply, then trains and evaluates both readings of the paper.
set -u
PY="${PY:-python3}"
ROOT="${TVSUM_ROOT:?set TVSUM_ROOT to the ydata-tvsum50-v1_1 directory}"
cd "$(dirname "$0")"

echo "[$(date +%H:%M:%S)] waiting for features..."
until [ -f features_tvsum.npz ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] features ready: $(ls -lh features_tvsum.npz | awk '{print $5}')"

# cheap sanity pass first: 2 epochs each, catches shape/NaN bugs in ~2 min
for M in faithful repaired; do
  echo "[$(date +%H:%M:%S)] sanity $M (2 epochs)"
  $PY -u train_prlvs.py features_tvsum.npz 2 $M 2>&1 | grep -viE "warn" | tail -3 || exit 1
done

for M in faithful repaired; do
  echo "[$(date +%H:%M:%S)] TRAINING $M (300 epochs)"
  $PY -u train_prlvs.py features_tvsum.npz 300 $M 2>&1 | grep -viE "warn"
  echo "[$(date +%H:%M:%S)] EVAL $M"
  $PY -u eval_prlvs.py features_tvsum.npz prlvs_$M.pt "$ROOT" 2>&1 | grep -viE "warn"
  mv prlvs_eval.json prlvs_eval_$M.json 2>/dev/null
done
echo "[$(date +%H:%M:%S)] ALL DONE"
