#!/usr/bin/env python
"""Aggregate the per-series eval CSV from forward.py into per-dataset means.

Two overall rows are reported because they answer different questions. The test
split is dominated by a few datasets (SVDB/SMAP/MITDB are half the series, while
8 datasets have a single series), so:

  MACRO = mean of the per-dataset means -- one vote per dataset. This is the
          headline number; it is what mTSBench-style comparisons report.
  MICRO = mean over all series -- one vote per series, hence ~50% SVDB+SMAP+MITDB.

Usage:
    python summarize_results.py eval_results_ZS.csv
    python summarize_results.py eval_base.csv eval_ft.csv --metric VUS-PR   # compare runs
"""
import argparse
import os
import sys

import pandas as pd

ID_COLS = ["series_id", "dataset"]


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing column(s) {missing}")
    metrics = [c for c in df.columns if c not in ID_COLS]
    df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
    return df, metrics


def summarize(df, metrics):
    """Per-dataset means + macro/micro overalls, as one frame."""
    per_ds = df.groupby("dataset")[metrics].mean()
    per_ds.insert(0, "n_series", df.groupby("dataset").size())
    per_ds = per_ds.sort_values(metrics[0], ascending=False)

    macro = per_ds[metrics].mean()          # unweighted over datasets
    micro = df[metrics].mean()              # unweighted over series
    overall = pd.DataFrame([macro, micro], index=["MACRO (mean of datasets)",
                                                  "MICRO (mean of series)"])
    overall.insert(0, "n_series", [len(per_ds), len(df)])
    return per_ds, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="one or more eval_results CSVs")
    ap.add_argument("--metric", default="VUS-PR",
                    help="metric used for the side-by-side when >1 CSV is given")
    ap.add_argument("--out_csv", default=None,
                    help="write the per-dataset table (single-CSV mode) here")
    args = ap.parse_args()

    if len(args.csv) == 1:
        df, metrics = load(args.csv[0])
        per_ds, overall = summarize(df, metrics)
        table = pd.concat([per_ds, overall])
        print(f"\n{args.csv[0]}  --  {len(df)} series / {df.dataset.nunique()} datasets\n")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        if args.out_csv:
            table.to_csv(args.out_csv, index_label="dataset")
            print(f"\nwrote {args.out_csv}")
        return

    # Multi-CSV: one column per run, so checkpoints line up row-wise.
    cols, overalls = {}, {}
    for path in args.csv:
        df, metrics = load(path)
        if args.metric not in metrics:
            sys.exit(f"{path}: no metric {args.metric!r} (have: {metrics})")
        per_ds, overall = summarize(df, metrics)
        name = os.path.splitext(os.path.basename(path))[0]
        cols[name] = per_ds[args.metric]
        overalls[name] = overall[args.metric]

    table = pd.concat([pd.DataFrame(cols), pd.DataFrame(overalls)])
    runs = list(cols)
    if len(runs) == 2:
        table["delta"] = table[runs[1]] - table[runs[0]]
    print(f"\n{args.metric} by dataset\n")
    print(table.to_string(float_format=lambda v: f"{v:+.4f}" if v < 0 else f"{v:.4f}"))
    if args.out_csv:
        table.to_csv(args.out_csv, index_label="dataset")
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
