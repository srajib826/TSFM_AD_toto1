#!/usr/bin/env bash
# Carve mTSBench into Toto anomaly windows (train / val / test) per dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/rajib/miniconda3/etc/profile.d/conda.sh
conda activate toto_ft

DATA_ROOT="${DATA_ROOT:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/prepared_total}"

NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-64}"
TEST_STRIDE="${TEST_STRIDE:-64}"
TEST_FRACTION="${TEST_FRACTION:-0.5}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-42}"

ARGS=(
  --data_root            "$DATA_ROOT"
  --output_dir           "$OUTPUT_DIR"
  --normal_signal_length "$NORMAL_SIGNAL_LENGTH"
  --context_length       "$CONTEXT_LENGTH"
  --prediction_length    "$PREDICTION_LENGTH"
  --stride               "$STRIDE"
  --test_stride          "$TEST_STRIDE"
  --test_fraction        "$TEST_FRACTION"
  --val_fraction         "$VAL_FRACTION"
  --seed                 "$SEED"
)
# Optional subset:  DATASETS="SMD MSL SMAP" bash run_prepare_total.sh
if [ -n "${DATASETS:-}" ]; then
  ARGS+=(--datasets $DATASETS)
fi

echo "prepare_total.py ${ARGS[*]}"
python -u prepare_total.py "${ARGS[@]}"
