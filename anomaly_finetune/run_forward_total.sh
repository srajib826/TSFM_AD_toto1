#!/usr/bin/env bash
# Evaluate Toto (zero-shot base and/or fine-tuned LoRA checkpoint) on the mTSBench
# test split, computing VUS-PR/-ROC etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/rajib/miniconda3/etc/profile.d/conda.sh
# source /userdata1/rajib/miniconda3/etc/profile.d/conda.sh

conda activate toto_ft

# VUS_ROC_VUS_PR lives in the Chronos workspace; add it to PYTHONPATH (read-only use).
VUS_ROOT="${VUS_ROOT:-/home/rajib/Sir_git_TSAD/TSFM-anomaly/Chronos_Finetuning/rajib_work_space}"
# VUS_ROOT="${VUS_ROOT:-/userdata1/rajib/TSFM-anomaly/Chronos_Finetuning/rajib_work_space}"
# Import `toto` from the repo checkout (../toto), not from site-packages, so local
# edits to the model take effect without reinstalling.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}:${VUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PREPARED_DIR="${PREPARED_DIR:-$SCRIPT_DIR/prepared_total}"
CHECKPOINT="${CHECKPOINT:-/home/rajib/TSFM_AD_toto1/anomaly_finetune/toto-single-stage_mtsbench_HS/best-ckpt}"          # empty => zero-shot base
DEVICE="${DEVICE:-cuda}"
SCORE_METHOD="${SCORE_METHOD:-interval}"
AGG_METHOD="${AGG_METHOD:-topk_mean}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
# Peak memory here is the posterior draw, not the model: MixtureSameFamily.sample()
# transiently holds an (S,B,V,P,K) tensor. BATCH_WINDOWS caps B, SAMPLE_CHUNK caps S
# per draw. Neither changes the result -- all NUM_SAMPLES samples are still used.
# Measured on the widest dataset (cicids, 72 variates): 16/32 peaks at 1.7GB of 8GB.
# The upstream default of 16 windows with an unchunked 256-sample draw OOMs here.
BATCH_WINDOWS="${BATCH_WINDOWS:-16}"
SAMPLE_CHUNK="${SAMPLE_CHUNK:-32}"
# fp16 backbone: ~1.4x faster (6.5h -> 4.5h over the full test split) and the q10/q90
# drift vs fp32 is smaller than the sampling noise already present at NUM_SAMPLES=256.
# Set AMP=0 to force fp32.
AMP="${AMP:-1}"

ARGS=(
  --prepared_dir  "$PREPARED_DIR"
  --device        "$DEVICE"
  --score_method  "$SCORE_METHOD"
  --agg_method    "$AGG_METHOD"
  --num_samples   "$NUM_SAMPLES"
  --batch_windows "$BATCH_WINDOWS"
  --sample_chunk  "$SAMPLE_CHUNK"
)
if [ "$AMP" != "0" ]; then
  ARGS+=(--amp)
fi
if [ -n "$CHECKPOINT" ]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "${DATASETS:-}" ]; then
  ARGS+=(--datasets $DATASETS)
fi

echo "forward.py ${ARGS[*]}"
python -u forward.py "${ARGS[@]}"
