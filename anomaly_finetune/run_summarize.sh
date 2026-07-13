#!/usr/bin/env bash
# Aggregate the per-series CSVs from forward.py into per-dataset means.
#
#   ./run_summarize.sh                          # summarize the default CSV
#   ./run_summarize.sh eval_ft.csv              # summarize one run
#   ./run_summarize.sh eval_base.csv eval_ft.csv    # compare runs, per dataset
#   METRIC=VUS-ROC ./run_summarize.sh a.csv b.csv   # compare on another metric
#
# No PYTHONPATH/toto/GPU needed here -- this only reads CSVs with pandas.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


METRIC="${METRIC:-VUS-PR}"          # only used when >1 CSV is given

# Positional args are the CSVs; fall back to the zero-shot run.
if [ "$#" -gt 0 ]; then
  CSVS=("$@")
else
  CSVS=("${IN_CSV:-$SCRIPT_DIR/eval_results_FT_bestckpt.csv}")
fi

for c in "${CSVS[@]}"; do
  if [ ! -f "$c" ]; then
    echo "missing: $c" >&2
    exit 1
  fi
done

# Single CSV => write the per-dataset table beside it as <name>_by_dataset.csv.
# Multi CSV  => write the side-by-side comparison instead.
if [ "${#CSVS[@]}" -eq 1 ]; then
  DEFAULT_OUT="${CSVS[0]%.csv}_by_dataset.csv"
else
  DEFAULT_OUT="$SCRIPT_DIR/eval_compare_${METRIC}.csv"
fi
OUT_CSV="${OUT_CSV:-$DEFAULT_OUT}"

ARGS=("${CSVS[@]}" --metric "$METRIC" --out_csv "$OUT_CSV")

echo "summarize_results.py ${ARGS[*]}"
python -u summarize_results.py "${ARGS[@]}"
