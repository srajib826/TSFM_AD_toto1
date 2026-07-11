#!/usr/bin/env bash
# LoRA anomaly fine-tuning of Toto (maskloss v2 + hierarchical sampler).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/rajib/miniconda3/etc/profile.d/conda.sh
conda activate toto_ft

PREPARED_DIR="${PREPARED_DIR:-$SCRIPT_DIR/prepared_total}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/toto-single-stage_mtsbench_HS}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Datadog/Toto-Open-Base-1.0}"
DEVICE="${DEVICE:-cuda}"

# Geometry (must match prepare_total)
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"

# LoRA
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Objective (maskloss v2 defaults)
MARGIN_M="${MARGIN_M:-5}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-1.0}"
HINGE_MODE="${HINGE_MODE:-per_step}"
MARGIN_MODE="${MARGIN_MODE:-relative}"
AGG_MODE="${AGG_MODE:-batch_global}"
P_ANOM="${P_ANOM:-0.3333333333333333}"

# Optim / schedule / loop
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
STABLE_STEPS="${STABLE_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-2200}"
TRAIN_BATCH_WINDOWS="${TRAIN_BATCH_WINDOWS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EVAL_EVERY="${EVAL_EVERY:-200}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-42}"

ARGS=(
  --prepared_dir         "$PREPARED_DIR"
  --output_dir           "$OUTPUT_DIR"
  --pretrained_model     "$PRETRAINED_MODEL"
  --device               "$DEVICE"
  --normal_signal_length "$NORMAL_SIGNAL_LENGTH"
  --context_length       "$CONTEXT_LENGTH"
  --prediction_length    "$PREDICTION_LENGTH"
  --lora_r               "$LORA_R"
  --lora_alpha           "$LORA_ALPHA"
  --lora_dropout         "$LORA_DROPOUT"
  --margin_m             "$MARGIN_M"
  --margin_lambda        "$MARGIN_LAMBDA"
  --hinge_mode           "$HINGE_MODE"
  --margin_mode          "$MARGIN_MODE"
  --agg_mode             "$AGG_MODE"
  --p_anom               "$P_ANOM"
  --lr                   "$LR"
  --min_lr               "$MIN_LR"
  --warmup_steps         "$WARMUP_STEPS"
  --stable_steps         "$STABLE_STEPS"
  --decay_steps          "$DECAY_STEPS"
  --max_steps            "$MAX_STEPS"
  --train_batch_windows  "$TRAIN_BATCH_WINDOWS"
  --grad_accum           "$GRAD_ACCUM"
  --eval_every           "$EVAL_EVERY"
  --log_every            "$LOG_EVERY"
  --seed                 "$SEED"
)
if [ -n "${DATASETS:-}" ]; then
  ARGS+=(--datasets $DATASETS)
fi

echo "finetune_toto_anomaly.py ${ARGS[*]}"
python -u finetune_toto_anomaly.py "${ARGS[@]}"
