#!/usr/bin/env bash
# Evaluate Toto (zero-shot base and/or fine-tuned LoRA checkpoint) on the mTSBench
# test split, computing VUS-PR/-ROC etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/rajib/miniconda3/etc/profile.d/conda.sh
conda activate toto_ft

# VUS_ROC_VUS_PR lives in the Chronos workspace; add it to PYTHONPATH (read-only use).
VUS_ROOT="${VUS_ROOT:-/home/rajib/Sir_git_TSAD/TSFM-anomaly/Chronos_Finetuning/rajib_work_space}"
export PYTHONPATH="${SCRIPT_DIR}:${VUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PREPARED_DIR="${PREPARED_DIR:-$SCRIPT_DIR/prepared_total}"
CHECKPOINT="${CHECKPOINT:-}"          # empty => zero-shot base
DEVICE="${DEVICE:-cuda}"
SCORE_METHOD="${SCORE_METHOD:-interval}"
AGG_METHOD="${AGG_METHOD:-topk_mean}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
OUT_CSV="${OUT_CSV:-$SCRIPT_DIR/eval_results.csv}"

ARGS=(
  --prepared_dir  "$PREPARED_DIR"
  --device        "$DEVICE"
  --score_method  "$SCORE_METHOD"
  --agg_method    "$AGG_METHOD"
  --num_samples   "$NUM_SAMPLES"
  --out_csv       "$OUT_CSV"
)
if [ -n "$CHECKPOINT" ]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "${DATASETS:-}" ]; then
  ARGS+=(--datasets $DATASETS)
fi

echo "forward.py ${ARGS[*]}"
python -u forward.py "${ARGS[@]}"
