"""
Whole-mTSBench sliding-window data prep for **Toto 1.0** anomaly fine-tuning
(maskloss v2 + hierarchical sampler), with per-dataset TRAIN / VAL / TEST splits.

This is the Toto counterpart of the Chronos
`TOTAL_RUN_maskloss_v2_HS/prepare_total.py`. The per-window layout is
byte-compatible with the Chronos pipeline (see `toto_prep_lib.py`):

    target        : (F, NORMAL_SIGNAL_LENGTH + context_length + prediction_length)
                    = [ normal(256) | context(512) | future(64) ]   -> (F, 832)
    future_labels : (prediction_length,) int 0/1

Toto reads `target` directly as `series (variates=F, time=832)`; patch_size=64 so
768 = 12 context patches and the 64-step horizon is one patch (read from the [REG]
token during fine-tuning).

Splits (file-based, seeded — no series leaks across splits):
  * TEST : `--test_fraction` (0.5) of each dataset's *test.csv files. ORDERED, with
           positional metadata -> `test_model_inputs.pkl` + `test_series_meta.pkl`.
           Used by forward.py (VUS-PR) and for eval_test_* loss curves.
  * VAL  : `--val_fraction` (0.1) of the remaining train-pool files -> SHUFFLED
           `val_model_inputs.pkl` (+ `val_n_anom.npy`). Used for eval_val_* curves.
  * TRAIN: the rest -> SHUFFLED `train_model_inputs.pkl` (+ `train_n_anom.npy`).
           UNCAPPED / thresholdless — both dataset & class imbalance are handled at
           train time by the hierarchical sampler (see finetune_toto_anomaly.py).

Datasets with exactly one *test.csv are TEST-ONLY (no train/val windows).

Usage
-----
    python prepare_total.py                          # carve everything
    python prepare_total.py --datasets SMD MSL SMAP  # subset
"""

import argparse
import glob
import json
import logging
import os
import pickle

import numpy as np

from toto_prep_lib import (
    NORMAL_SIGNAL_LENGTH,
    anomaly_step_counts,
    build_pairs_for_files,
    hs_expected_anom_step_frac,
    list_datasets,
    logger,
    pairs_to_model_inputs,
    split_files,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _carve(files, args, min_req, tag, normal_signal_length):
    """Build model-input dicts for a set of files (no meta)."""
    pairs, _, _, _ = build_pairs_for_files(
        files, args.context_length, args.prediction_length, args.stride, min_req, tag, normal_signal_length
    )
    return pairs_to_model_inputs(pairs, normal_signal_length=normal_signal_length, include_meta=False)


def _dump(inputs, path, shuffle_rng=None):
    if shuffle_rng is not None:
        shuffle_rng.shuffle(inputs)
    with open(path, "wb") as f:
        pickle.dump(inputs, f)


def prepare(args) -> None:
    normal_signal_length = args.normal_signal_length
    os.makedirs(args.output_dir, exist_ok=True)
    per_ds_root = os.path.join(args.output_dir, "per_dataset")
    os.makedirs(per_ds_root, exist_ok=True)

    datasets = list_datasets(args.data_root, args.datasets)
    if not datasets:
        raise SystemExit(f"No datasets with *test.csv found under {args.data_root}")
    logger.info(f"Datasets ({len(datasets)}): {datasets}")
    logger.info(
        "TRAIN is UNCAPPED / thresholdless — every train window is kept. Both imbalances "
        "are handled at train time by the hierarchical sampler."
    )

    # A series only needs to be long enough for one full [context|future] window; the
    # normal prefix is drawn separately from the whole series (the Chronos invariant).
    min_req = max(args.min_length, args.context_length + args.prediction_length)

    manifest: dict = {"datasets": {}, "config": vars(args).copy()}
    train_datasets: list = []
    grand_anom = grand_norm = 0
    grand_anom_steps = grand_total_steps = 0

    for ds in datasets:
        ddir = os.path.join(args.data_root, ds)
        test_csvs = sorted(glob.glob(os.path.join(ddir, "**", "*test.csv"), recursive=True))
        train_files, val_files, test_files, mode = split_files(
            test_csvs, args.test_fraction, args.val_fraction, args.seed
        )

        logger.info("=" * 78)
        logger.info(
            f"{ds}: {len(test_csvs)} *test.csv  [{mode}]  "
            f"train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}"
        )

        ds_out = os.path.join(per_ds_root, ds)
        os.makedirs(ds_out, exist_ok=True)

        entry = {
            "mode": mode,
            "n_test_csv": len(test_csvs),
            "train_files": [os.path.basename(f) for f in train_files],
            "val_files": [os.path.basename(f) for f in val_files],
            "test_files": [os.path.basename(f) for f in test_files],
            "train_windows": 0,
            "val_windows": 0,
            "test_windows": 0,
            "test_series": 0,
            "test_anomalous": 0,
            "in_train_pool": False,
        }

        # ── TEST half (UNCAPPED, ordered + metadata) ─────────────────────────
        test_pairs, _, _, test_meta = build_pairs_for_files(
            test_files, args.context_length, args.prediction_length, args.test_stride, min_req,
            f"{ds}/test", normal_signal_length,
        )
        test_inputs = pairs_to_model_inputs(test_pairs, normal_signal_length=normal_signal_length, include_meta=True)
        with open(os.path.join(ds_out, "test_model_inputs.pkl"), "wb") as f:
            pickle.dump(test_inputs, f)
        with open(os.path.join(ds_out, "test_series_meta.pkl"), "wb") as f:
            pickle.dump(test_meta, f)
        t_anom = int((anomaly_step_counts(test_inputs) >= 1).sum()) if test_inputs else 0
        entry.update({"test_windows": len(test_inputs), "test_series": len(test_meta), "test_anomalous": t_anom})
        logger.info(f"  TEST  -> {len(test_inputs)} windows ({len(test_meta)} series), {t_anom} anomalous")

        # ── VAL half (shuffled) ──────────────────────────────────────────────
        if val_files:
            val_inputs = _carve(val_files, args, min_req, f"{ds}/val", normal_signal_length)
            _dump(val_inputs, os.path.join(ds_out, "val_model_inputs.pkl"),
                  shuffle_rng=np.random.default_rng(args.seed))
            val_n_anom = anomaly_step_counts(val_inputs)
            np.save(os.path.join(ds_out, "val_n_anom.npy"), val_n_anom)
            entry["val_windows"] = len(val_inputs)
            logger.info(f"  VAL   -> {len(val_inputs)} windows")

        # ── TRAIN half — carve and keep EVERYTHING ───────────────────────────
        if train_files:
            train_inputs = _carve(train_files, args, min_req, f"{ds}/train", normal_signal_length)
            n_anom = anomaly_step_counts(train_inputs)
            n_anom_win = int((n_anom >= 1).sum())
            n_norm_win = len(train_inputs) - n_anom_win

            if n_anom_win == 0:
                raise SystemExit(
                    f"{ds}: train half has {len(train_inputs)} windows but ZERO anomaly windows. "
                    f"The hierarchical sampler's anomalous branch would have an all-zero weight "
                    f"vector for this dataset. Exclude it with --datasets, or re-split."
                )
            if n_anom_win < args.min_anom_windows:
                logger.warning(
                    f"  [{ds}] only {n_anom_win} anomaly windows — level 1 revisits this dataset "
                    f"uniformly, so these few windows are revisited heavily. Watch for overfitting."
                )

            _dump(train_inputs, os.path.join(ds_out, "train_model_inputs.pkl"),
                  shuffle_rng=np.random.default_rng(args.seed))
            np.save(os.path.join(ds_out, "train_n_anom.npy"), n_anom)

            exp_frac, nat_frac = hs_expected_anom_step_frac(n_anom, args.prediction_length, args.p_anom)
            entry.update(
                {
                    "train_windows": len(train_inputs),
                    "train_anom_windows": n_anom_win,
                    "train_norm_windows": n_norm_win,
                    "train_anom_steps": int(n_anom.sum()),
                    "train_total_steps": len(train_inputs) * args.prediction_length,
                    "natural_anom_step_frac": round(nat_frac, 4),
                    "hs_expected_anom_step_frac": round(exp_frac, 4),
                    "in_train_pool": True,
                    "channels": int(train_inputs[0]["target"].shape[0]),
                }
            )
            train_datasets.append(ds)
            grand_anom += n_anom_win
            grand_norm += n_norm_win
            grand_anom_steps += int(n_anom.sum())
            grand_total_steps += len(train_inputs) * args.prediction_length
            logger.info(
                f"  TRAIN -> kept ALL {len(train_inputs)} windows (anom={n_anom_win}, normal={n_norm_win}); "
                f"F={entry['channels']}, anomaly steps {100.0 * nat_frac:.1f}% natural -> "
                f"{100.0 * exp_frac:.1f}% under HS (p_anom={args.p_anom:.3f})"
            )
        else:
            logger.info("  TRAIN -> none (test-only dataset; invisible to level 1)")

        manifest["datasets"][ds] = entry

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 78)
    tot = grand_anom + grand_norm
    logger.info("TRAIN POOL (uncapped; both imbalances are the sampler's job)")
    logger.info(f"  datasets in pool : {len(train_datasets)}/{len(datasets)}  {train_datasets}")
    logger.info(f"  total windows    : {tot}  ({100.0 * grand_anom / max(1, tot):.1f}% anomaly-bearing)")
    logger.info(
        f"  anomaly steps    : {grand_anom_steps}/{grand_total_steps} "
        f"({100.0 * grand_anom_steps / max(1, grand_total_steps):.1f}% natural)"
    )
    logger.info(
        f"  per-window target: [{normal_signal_length} normal | {args.context_length} context | "
        f"{args.prediction_length} future]; F varies per dataset"
    )

    manifest["train_pool"] = {
        "datasets": train_datasets,
        "n_datasets": len(train_datasets),
        "total_windows": tot,
        "anomaly_windows": grand_anom,
        "normal_windows": grand_norm,
        "anomaly_steps": grand_anom_steps,
        "total_steps": grand_total_steps,
        "natural_anom_step_frac": round(grand_anom_steps / max(1, grand_total_steps), 4),
        "per_dataset_draw_prob": round(1.0 / max(1, len(train_datasets)), 4),
        "capped": False,
        "thresholded": False,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest -> {os.path.join(args.output_dir, 'manifest.json')}")


def main():
    p = argparse.ArgumentParser(description="mTSBench data prep for Toto anomaly fine-tuning (HS).")
    p.add_argument("--data_root", default="/home/rajib/mTSBench/Datasets/mTSBench")
    p.add_argument("--output_dir", default=os.path.join(_HERE, "prepared_total"))
    p.add_argument("--datasets", nargs="*", default=None)

    # Window geometry — must match run_finetune / run_forward
    p.add_argument("--normal_signal_length", type=int, default=NORMAL_SIGNAL_LENGTH)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--stride", type=int, default=64, help="Train/val sliding-window stride")
    p.add_argument("--test_stride", type=int, default=64,
                   help="Test stride (MUST equal prediction_length so test windows tile contiguously)")
    p.add_argument("--min_length", type=int, default=50)

    # Split
    p.add_argument("--test_fraction", type=float, default=0.5)
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of the train-pool files carved for validation")
    p.add_argument("--seed", type=int, default=42)

    # Reporting only
    p.add_argument("--p_anom", type=float, default=1.0 / 3.0)
    p.add_argument("--min_anom_windows", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "prepare_total.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logger.info(f"Config: {vars(args)}")
    prepare(args)
    logger.info("Done.")


if __name__ == "__main__":
    main()
