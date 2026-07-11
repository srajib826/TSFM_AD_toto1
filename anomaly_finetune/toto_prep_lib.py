"""
Window-carving primitives for Toto 1.0 anomaly fine-tuning.

Ported VERBATIM (behaviour-identical) from the Chronos pipeline
`SMD_run/prepare_smd_split.py`, so the Toto windows are byte-compatible with the
Chronos ones: same NORMAL_SIGNAL_LENGTH, same sliding-window pairing, same
per-timestep future labels, same `[normal_signal | context | future]` layout.

Per-window layout (model-agnostic — this is just `(variates, time)`):

    target        : (F, NORMAL_SIGNAL_LENGTH + context_length + prediction_length)
                    = [ normal(256) | context(512) | future(64) ]   -> (F, 832)
    future_labels : (prediction_length,) int 0/1   (0=normal, 1=anomaly)

Toto consumes `target` directly as `series (variates=F, time=832)`; the `normal`
prefix may be NaN-left-padded, which the Toto padding_mask handles. The horizon
(last `prediction_length` steps) is what the model is trained to forecast, and
`future_labels` tags each horizon step normal/anomaly for the masked-margin loss.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd

# Same instruction-prefix length as the Chronos pipeline.
NORMAL_SIGNAL_LENGTH = 256

logger = logging.getLogger("toto_prep")


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_as_multivariate(csv_path: str):
    """
    Load one *test.csv file.

    Returns
    -------
    features : float32 array (n_variates, time_steps) — timestamp/is_anomaly excluded.
    labels   : int32 array (time_steps,), 1=anomaly 0=normal (all-zero if column absent).
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    if not feature_cols:
        return None, None
    try:
        features = df[feature_cols].values.T.astype(np.float32)
        labels = (
            df["is_anomaly"].values.astype(np.int32)
            if "is_anomaly" in df.columns
            else np.zeros(df.shape[0], dtype=np.int32)
        )
        return features, labels
    except Exception as e:
        logger.warning(f"Error processing {csv_path}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly boundary / normal-zone helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_anomaly_boundaries(labels: np.ndarray):
    """Contiguous anomaly regions as (start, end) with end EXCLUSIVE."""
    boundaries, in_anom, start = [], False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_anom:
            in_anom, start = True, i
        elif v == 0 and in_anom:
            in_anom = False
            boundaries.append((start, i))
    if in_anom:
        boundaries.append((start, len(labels)))
    return boundaries


def get_normal_zones(boundaries, total: int):
    """Normal (non-anomaly) zones as (start, end) pairs."""
    zones, prev = [], 0
    for s, e in boundaries:
        if s > prev:
            zones.append((prev, s))
        prev = e
    if prev < total:
        zones.append((prev, total))
    return zones


def extract_normal_signal(data: np.ndarray, normal_zones, length: int):
    """
    Return a (F, length) reference normal signal sampled from the series' normal zones.

      1. If a single normal zone is long enough, take its last `length` timesteps.
      2. Otherwise concatenate normal zones (longest first) until enough.
      3. If still short, left-pad with NaN.

    Returns None if there are no normal zones at all.
    """
    if not normal_zones:
        return None

    sorted_zones = sorted(normal_zones, key=lambda z: z[1] - z[0], reverse=True)
    s, e = sorted_zones[0]
    if e - s >= length:
        return data[:, e - length : e].astype(np.float32, copy=False)

    chunks, collected = [], 0
    for s, e in sorted_zones:
        chunks.append(data[:, s:e])
        collected += e - s
        if collected >= length:
            break

    combined = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if combined.shape[1] >= length:
        return combined[:, -length:]

    F = combined.shape[0]
    pad = np.full((F, length - combined.shape[1]), np.nan, dtype=np.float32)
    return np.concatenate([pad, combined], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Pair construction
# ─────────────────────────────────────────────────────────────────────────────
def create_pairs(data, labels, context_length, prediction_length, stride):
    """
    Slide a window over the series. For each start t (from context_length onward):

      context        = data[:, t - context_length : t]      (always full, real steps)
      future         = data[:, t : t + prediction_length]   (full window only)
      future_labels  = labels[t : t + prediction_length]    (one label per future step)

    Windows with fewer than `prediction_length` future steps remaining are skipped.
    """
    pairs = []
    total = data.shape[1]
    for t in range(context_length, total, stride):
        fut_end = t + prediction_length
        if fut_end > total:
            break
        ctx = data[:, t - context_length : t].astype(np.float32, copy=False)
        fut = data[:, t:fut_end].astype(np.float32, copy=False)
        fut_labels = labels[t:fut_end].astype(np.int32, copy=False)
        pairs.append(
            {
                "context": {"target": ctx},
                "future": {"target": fut},
                "future_labels": fut_labels,
                "future_start": int(t),
                "future_end": int(fut_end),
            }
        )
    return pairs


def _attach_normal_signal(pairs, normal_sig):
    """In-place: attach the same per-series normal_signal reference to every pair."""
    for p in pairs:
        p["normal_signal"] = normal_sig


def pairs_to_model_inputs(pairs, normal_signal_length: int = NORMAL_SIGNAL_LENGTH, include_meta: bool = False):
    """
    Convert pairs to fixed-length model inputs:

        target = [ normal_signal (N) | context (C) | future (P) ]

    Each output dict carries `future_labels` (P,). When `include_meta=True` (TEST
    split) each entry also carries series_id / future_start / future_end /
    series_length so the per-step predictions can be scattered back onto the series
    timeline for series-based metrics (VUS-PR etc.).
    """
    out = []
    for p in pairs:
        ctx, fut = p["context"]["target"], p["future"]["target"]
        normal = p.get("normal_signal")
        if normal is None:
            normal = np.full((ctx.shape[0], normal_signal_length), np.nan, dtype=np.float32)
        target = np.concatenate([normal, ctx, fut], axis=1)
        entry = {"target": target, "future_labels": p["future_labels"]}
        if include_meta:
            entry["series_id"] = p.get("series_id")
            entry["future_start"] = p.get("future_start")
            entry["future_end"] = p.get("future_end")
            entry["series_length"] = p.get("series_length")
        out.append(entry)
    return out


def build_pairs_for_files(files, context_length, prediction_length, stride, min_req, tag, normal_signal_length):
    """
    Load each CSV (in order), build pairs with the per-series normal prefix attached.
    Returns (all_pairs, used_series_ids, n_skipped, series_meta).

    series_meta: series_id -> {length, labels (full per-timestamp ground truth),
    n_features, context_length}.
    """
    all_pairs, used, skipped, series_meta = [], [], 0, {}
    for path in files:
        feat, lbl = load_csv_as_multivariate(path)
        if feat is None or feat.shape[1] < min_req:
            length = feat.shape[1] if feat is not None else "None"
            logger.info(f"  skip {os.path.basename(path)} (length={length} < {min_req})")
            skipped += 1
            continue
        sid = os.path.basename(path)
        pairs = create_pairs(feat, lbl, context_length, prediction_length, stride)
        zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
        normal_sig = extract_normal_signal(feat, zones, normal_signal_length)
        _attach_normal_signal(pairs, normal_sig)
        for p in pairs:
            p["series_id"] = sid
            p["series_length"] = int(feat.shape[1])
        all_pairs.extend(pairs)
        used.append(sid)
        series_meta[sid] = {
            "length": int(feat.shape[1]),
            "labels": lbl.astype(np.int32, copy=False),
            "n_features": int(feat.shape[0]),
            "context_length": int(context_length),
        }
    return all_pairs, used, skipped, series_meta


# ─────────────────────────────────────────────────────────────────────────────
#  File discovery + three-way (train / val / test) file split
# ─────────────────────────────────────────────────────────────────────────────
def list_datasets(data_root: str, only):
    """Dataset = any sub-directory of data_root containing >=1 *test.csv file."""
    names = []
    for entry in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, entry)
        if not os.path.isdir(d):
            continue
        if only and entry not in only:
            continue
        if glob.glob(os.path.join(d, "**", "*test.csv"), recursive=True):
            names.append(entry)
    return names


def split_files(test_csvs, test_fraction: float, val_fraction: float, seed: int):
    """
    File-based three-way split. Returns (train_files, val_files, test_files, mode).

    Same seed / permutation convention as the Chronos pipeline, so the TEST assignment
    matches Chronos exactly (val is carved additionally from the train pool):
      - exactly 1 file  -> TEST-ONLY (train & val empty).
      - >=2 files        -> hold out `test_fraction` for test; from the remaining train
                            pool, carve `val_fraction` for val (>=1 file kept for train).
    """
    if len(test_csvs) == 1:
        return [], [], list(test_csvs), "test_only"

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(test_csvs))
    n_test = max(1, min(len(test_csvs) - 1, int(round(len(test_csvs) * test_fraction))))
    test_idx = set(perm[:n_test].tolist())
    train_pool = [test_csvs[i] for i in range(len(test_csvs)) if i not in test_idx]
    test_files = [test_csvs[i] for i in sorted(test_idx)]

    val_files = []
    train_files = train_pool
    if val_fraction > 0 and len(train_pool) > 1:
        perm_tp = rng.permutation(len(train_pool))
        n_val = max(1, int(round(len(train_pool) * val_fraction)))
        n_val = min(n_val, len(train_pool) - 1)  # keep >=1 training file
        val_set = set(perm_tp[:n_val].tolist())
        val_files = [train_pool[i] for i in sorted(val_set)]
        train_files = [train_pool[i] for i in range(len(train_pool)) if i not in val_set]

    return train_files, val_files, test_files, "file_split"


def anomaly_step_counts(inputs) -> np.ndarray:
    """Per-window anomaly-step count n_anom in [0, prediction_length]."""
    return np.asarray([int(np.asarray(d["future_labels"]).sum()) for d in inputs], dtype=np.int16)


def hs_expected_anom_step_frac(n_anom: np.ndarray, H: int, p_anom: float):
    """Expected anomaly-STEP fraction the HS sampler delivers for ONE dataset."""
    n = n_anom.astype(np.float64)
    n_norm = H - n
    natural = float(n.mean() / H) if len(n) else float("nan")
    s_a, s_n = n.sum(), n_norm.sum()
    if s_a <= 0 or s_n <= 0:
        return float("nan"), natural
    e_anom = float((n * n).sum() / s_a)
    e_norm = float((n_norm * n).sum() / s_n)
    return float((p_anom * e_anom + (1.0 - p_anom) * e_norm) / H), natural
