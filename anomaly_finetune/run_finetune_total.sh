#!/usr/bin/env bash
# LoRA anomaly fine-tuning of Toto (maskloss v2 + hierarchical sampler).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# source /home/rajib/miniconda3/etc/profile.d/conda.sh
source /userdata1/rajib/miniconda3/etc/profile.d/conda.sh
conda activate toto_ft

# Import `toto` from the repo checkout (../toto), not from site-packages, so local
# edits to the model take effect without reinstalling.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
# WarmupStableDecayLR derives its total from these three phases, NOT from max_steps:
# they must sum to MAX_STEPS, or the tail of the run is stranded at min_lr.
WARMUP_STEPS="${WARMUP_STEPS:-300}"
STABLE_STEPS="${STABLE_STEPS:-3200}"
DECAY_STEPS="${DECAY_STEPS:-2500}"
MAX_STEPS="${MAX_STEPS:-6000}"
# agg_mode=batch_global computes L_good / the hinge as ratios of sums over ONE
# micro-batch, so train_batch_windows -- not the accumulated total -- is the
# contrastive pool. Keep it as large as the 8GB card allows (6 peaks at ~7.1GB);
# grad_accum only denoises the gradient, so spend the time on more steps instead.
TRAIN_BATCH_WINDOWS="${TRAIN_BATCH_WINDOWS:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EVAL_EVERY="${EVAL_EVERY:-250}"
# Keep eval <= train batch; at 8 it can OOM against a train peak of 6.
EVAL_BATCH_WINDOWS="${EVAL_BATCH_WINDOWS:-6}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-42}"

SCHED_TOTAL=$((WARMUP_STEPS + STABLE_STEPS + DECAY_STEPS))
if [ "$SCHED_TOTAL" -ne "$MAX_STEPS" ]; then
  echo "ERROR: warmup+stable+decay ($SCHED_TOTAL) != max_steps ($MAX_STEPS)." >&2
  echo "       The LR schedule would not line up with the run length." >&2
  exit 1
fi

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
  --eval_batch_windows   "$EVAL_BATCH_WINDOWS"
  --log_every            "$LOG_EVERY"
  --seed                 "$SEED"
)
if [ -n "${DATASETS:-}" ]; then
  ARGS+=(--datasets $DATASETS)
fi

echo "finetune_toto_anomaly.py ${ARGS[*]}"
python -u finetune_toto_anomaly.py "${ARGS[@]}"
