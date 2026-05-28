#!/usr/bin/env bash
# Symmetric counterpart to the DiT-200k convergence test: train L-MLP-main at 200k
# steps for 3 seeds so the budget-artifact comparison is apples-to-apples. Then
# finish the phases the previous overnight script crashed before completing:
# CLAP-FAD on the new evals, bootstrap CIs over the full Track B set, re-aggregate,
# re-render figures.
set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH=.

LOG_DIR="logs/lmlp200k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "log dir: $LOG_DIR"

LMLP_200K=(
  configs/track_b_lmlp_200k_s0.yaml
  configs/track_b_lmlp_200k_s1.yaml
  configs/track_b_lmlp_200k_s2.yaml
)

for cfg in "${LMLP_200K[@]}"; do
  name=$(basename "$cfg" .yaml)
  exp_id=$(grep -m1 '^experiment_id:' "$cfg" | awk '{print $2}')
  final_ckpt="experiments/${exp_id}/checkpoints/step_0200000.pt"
  if [[ -f "$final_ckpt" ]]; then
    echo "[have final] $name"
  else
    echo "[train ] $name (200k)"
    python -m src.train --config "$cfg" > "$LOG_DIR/${name}.train.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "  FAIL rc=$rc"; continue
    fi
  fi
  for step in step_0050000 step_0100000 step_0150000 step_0200000; do
    ck="experiments/${exp_id}/checkpoints/${step}.pt"
    mf="experiments/${exp_id}/eval/${step}/metrics.json"
    [[ -f "$ck" ]] || continue
    [[ -f "$mf" ]] && { echo "[have eval] $name @ $step"; continue; }
    echo "[eval  ] $name @ $step"
    python scripts/eval_checkpoint.py --ckpt "$ck" --n_samples 500 --n_test 200 \
        > "$LOG_DIR/${name}.eval.${step}.log" 2>&1
  done
done

echo "[clap-fad] scoring new evals"
python scripts/run_clap_fad_batch.py --track b > "$LOG_DIR/clap_fad.log" 2>&1

echo "[bootstrap] computing FAD CIs (1000 resamples)"
python scripts/bootstrap_fad_cis.py --track b --boot_n 1000 \
    > "$LOG_DIR/bootstrap.log" 2>&1

echo "[aggregate] regenerating tables"
python scripts/aggregate_results.py > "$LOG_DIR/aggregate.log" 2>&1

echo "[figures] regenerating figures"
python scripts/make_figures.py > "$LOG_DIR/figures.log" 2>&1

echo "=== lmlp 200k overnight pass complete $(date -Iseconds) ==="
