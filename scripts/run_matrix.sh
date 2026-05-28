#!/usr/bin/env bash
# Train + evaluate the full Phase 2 matrix sequentially.
#
# Args (optional):
#   $1 = track filter (a | b | both)         default: a
#   $2 = arch filter  (regex over arch slug) default: '.*'
#   $3 = seed filter  (regex over seed)      default: '.*'
#
# Examples:
#   ./scripts/run_matrix.sh                          # all Track A configs
#   ./scripts/run_matrix.sh both                     # all 24 configs
#   ./scripts/run_matrix.sh a 'lmlp|dit' '0|1'       # specific subset
#
# Skips configs whose final checkpoint already exists.
set -u
cd "$(dirname "$0")/.."

TRACK="${1:-a}"
ARCH_RE="${2:-.*}"
SEED_RE="${3:-.*}"

source .venv/bin/activate
export PYTHONPATH=.

if [[ "$TRACK" == "both" ]]; then TRACK_GLOB='[ab]'; else TRACK_GLOB="$TRACK"; fi

mapfile -t CONFIGS < <(
  ls configs/track_${TRACK_GLOB}_*_s[0-9].yaml 2>/dev/null | sort
)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "no matching configs"; exit 1
fi

echo "matching configs: ${#CONFIGS[@]}"
for c in "${CONFIGS[@]}"; do echo "  $c"; done | head -30

mkdir -p logs
LOG_DIR="logs/matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "logs -> $LOG_DIR"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  # apply filters
  echo "$name" | grep -qE "_(lmlp(_no_fnn_z)?|dit|unet)_" || continue
  arch_part=$(echo "$name" | sed -E 's/^track_[ab]_(.+)_s[0-9]+$/\1/')
  seed_part=$(echo "$name" | grep -oE '_s[0-9]+$' | tr -d '_s')
  echo "$arch_part" | grep -qE "^($ARCH_RE)$" || { echo "[skip arch] $name"; continue; }
  echo "$seed_part" | grep -qE "^($SEED_RE)$" || { echo "[skip seed] $name"; continue; }

  exp_id=$(grep -m1 '^experiment_id:' "$cfg" | awk '{print $2}')
  final_ckpt="experiments/${exp_id}/checkpoints/step_0100000.pt"

  if [[ -f "$final_ckpt" ]]; then
    echo "[have final] $name -> $final_ckpt"
  else
    echo "[train]  $name  cfg=$cfg"
    log="$LOG_DIR/${name}.train.log"
    python -m src.train --config "$cfg" > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "  FAIL rc=$rc ; see $log"
      continue
    fi
  fi

  # Evaluate the final ckpt (and one mid-train ckpt for trajectory).
  for step_ckpt in step_0050000 step_0100000; do
    ckpt="experiments/${exp_id}/checkpoints/${step_ckpt}.pt"
    [[ -f "$ckpt" ]] || continue
    metrics_file="experiments/${exp_id}/eval/${step_ckpt}/metrics.json"
    if [[ -f "$metrics_file" ]]; then
      echo "[have eval] $name @ $step_ckpt"
      continue
    fi
    log="$LOG_DIR/${name}.eval.${step_ckpt}.log"
    echo "[eval]   $name @ $step_ckpt"
    python scripts/eval_checkpoint.py --ckpt "$ckpt" --n_samples 500 --n_test 200 > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "  eval FAIL rc=$rc ; see $log"
    fi
  done
done

echo "matrix complete"
