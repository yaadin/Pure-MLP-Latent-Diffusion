#!/usr/bin/env bash
# Overnight pass to address the paper critique:
#   1. Train 8 new-seed Track B runs (seeds 3, 4 for all 4 archs, 100k steps)
#   2. Train 3 DiT-small Track B runs to 200k steps (convergence test)
#   3. Evaluate everything at every checkpoint that exists
set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH=.

LOG_DIR="logs/critique_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "log dir: $LOG_DIR"

# === Phase 1: extra-seed runs (8 configs at 100k) ===
EXTRA_SEEDS=(
  configs/track_b_lmlp_s3.yaml
  configs/track_b_lmlp_s4.yaml
  configs/track_b_lmlp_no_fnn_z_s3.yaml
  configs/track_b_lmlp_no_fnn_z_s4.yaml
  configs/track_b_dit_s3.yaml
  configs/track_b_dit_s4.yaml
  configs/track_b_unet_s3.yaml
  configs/track_b_unet_s4.yaml
)

for cfg in "${EXTRA_SEEDS[@]}"; do
  name=$(basename "$cfg" .yaml)
  exp_id=$(grep -m1 '^experiment_id:' "$cfg" | awk '{print $2}')
  final_ckpt="experiments/${exp_id}/checkpoints/step_0100000.pt"
  if [[ -f "$final_ckpt" ]]; then
    echo "[have final] $name"
  else
    echo "[train ] $name"
    python -m src.train --config "$cfg" > "$LOG_DIR/${name}.train.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "  FAIL rc=$rc"; continue
    fi
  fi
  for step in step_0050000 step_0100000; do
    ck="experiments/${exp_id}/checkpoints/${step}.pt"
    mf="experiments/${exp_id}/eval/${step}/metrics.json"
    [[ -f "$ck" ]] || continue
    [[ -f "$mf" ]] && { echo "[have eval] $name @ $step"; continue; }
    echo "[eval  ] $name @ $step"
    python scripts/eval_checkpoint.py --ckpt "$ck" --n_samples 500 --n_test 200 \
        > "$LOG_DIR/${name}.eval.${step}.log" 2>&1
  done
done

# === Phase 2: DiT 200k convergence test ===
DIT_200K=(
  configs/track_b_dit_200k_s0.yaml
  configs/track_b_dit_200k_s1.yaml
  configs/track_b_dit_200k_s2.yaml
)

for cfg in "${DIT_200K[@]}"; do
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

# === Phase 3: CLAP-FAD on every new eval ===
echo "[clap-fad] scoring new evals"
python scripts/run_clap_fad_batch.py --track b > "$LOG_DIR/clap_fad.log" 2>&1

# === Phase 4: bootstrap 95% CIs on CLAP-FAD for all Track B ===
echo "[bootstrap] computing FAD CIs"
python scripts/bootstrap_fad_cis.py --track b --boot_n 1000 \
    > "$LOG_DIR/bootstrap.log" 2>&1

# === Phase 5: re-aggregate and re-make figures ===
echo "[aggregate] regenerating tables and figures"
python scripts/aggregate_results.py > "$LOG_DIR/aggregate.log" 2>&1
python scripts/make_figures.py > "$LOG_DIR/figures.log" 2>&1

echo "=== critique overnight pass complete $(date -Iseconds) ==="
