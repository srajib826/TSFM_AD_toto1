"""
Data loading, hierarchical sampler (HS), and collation for Toto anomaly fine-tuning.

Reads the per-dataset windows produced by `prepare_total.py`
(`per_dataset/<DS>/{train,val}_model_inputs.pkl`, each entry
`{"target": (F, 832), "future_labels": (64,)}`) and turns batches of windows into
padded Toto tensors: `series (B, maxF, 832)`, `padding_mask`, `id_mask`, and
`future_labels (B, 64)`.

Batching: each window is an INDEPENDENT sample (Toto's space attention never crosses
the batch dimension). Windows with different variate counts F are padded up to the
batch max; the padded variates get a separate `id_mask` group so they never leak into
the real window's space attention, and `padding_mask` zeroes them out of scaling/loss.

HS sampler (port of `finetune_anomaly_simple.py:190-343`), batch counts WINDOWS:
  level 1  dataset ~ Uniform(K)
  level 2  kind    ~ Bernoulli(p_anom)             (anomalous vs normal)
  level 3  window  ~ count-weighted within dataset (n_anom | H - n_anom)
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from typing import NamedTuple

import numpy as np
import torch

logger = logging.getLogger("toto_anomaly_data")


class TotoAnomalyBatch(NamedTuple):
    series: torch.Tensor          # (B, maxF, T)
    padding_mask: torch.Tensor    # (B, maxF, T) bool
    id_mask: torch.Tensor         # (B, maxF, T) long
    future_labels: torch.Tensor   # (B, P) long (0/1), shared across variates
    var_counts: torch.Tensor      # (B,) long, real variate count F_b per window

    def to(self, device):
        return TotoAnomalyBatch(
            self.series.to(device),
            self.padding_mask.to(device),
            self.id_mask.to(device),
            self.future_labels.to(device),
            self.var_counts.to(device),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Pools
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyPool:
    """A flat pool of windows plus per-window dataset id and anomaly-step count."""

    def __init__(self, windows, ds_of_window, ds_names, n_anom, prediction_length):
        self.windows = windows
        self.ds_of_window = np.asarray(ds_of_window, dtype=np.int64)
        self.ds_names = ds_names
        self.n_anom = np.asarray(n_anom, dtype=np.int64)
        self.H = int(prediction_length)

    def __len__(self):
        return len(self.windows)


def _load_split(prepared_dir, split, datasets=None, prediction_length=64):
    """Load `<split>_model_inputs.pkl` across per-dataset folders into an AnomalyPool."""
    per_ds = os.path.join(prepared_dir, "per_dataset")
    all_windows, ds_of_window, n_anom, ds_names = [], [], [], []
    found = sorted(glob.glob(os.path.join(per_ds, "*")))
    for ds_path in found:
        ds = os.path.basename(ds_path)
        if datasets and ds not in datasets:
            continue
        pkl = os.path.join(ds_path, f"{split}_model_inputs.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            windows = pickle.load(f)
        if not windows:
            continue
        k = len(ds_names)
        ds_names.append(ds)
        na_path = os.path.join(ds_path, f"{split}_n_anom.npy")
        if os.path.exists(na_path):
            na = np.load(na_path)
        else:
            na = np.asarray([int(np.asarray(w["future_labels"]).sum()) for w in windows], dtype=np.int64)
        for w, a in zip(windows, na):
            all_windows.append(w)
            ds_of_window.append(k)
            n_anom.append(int(a))
    logger.info(
        f"[{split}] loaded {len(all_windows)} windows across {len(ds_names)} datasets: {ds_names}"
    )
    return AnomalyPool(all_windows, ds_of_window, ds_names, n_anom, prediction_length)


def load_train_pool(prepared_dir, datasets=None, prediction_length=64):
    return _load_split(prepared_dir, "train", datasets, prediction_length)


def load_eval_pool(prepared_dir, split, datasets=None, prediction_length=64):
    """split in {'val', 'test'}."""
    return _load_split(prepared_dir, split, datasets, prediction_length)


# ─────────────────────────────────────────────────────────────────────────────
#  Collation
# ─────────────────────────────────────────────────────────────────────────────
def collate_windows(windows) -> TotoAnomalyBatch:
    """
    Pad a list of window dicts (`target (F,T)`, `future_labels (P,)`) into a batch.
    Padded variates get id_mask group 1 (real variates = 0) and padding_mask=False.
    """
    b = len(windows)
    max_f = max(int(w["target"].shape[0]) for w in windows)
    T = int(windows[0]["target"].shape[1])
    P = int(np.asarray(windows[0]["future_labels"]).shape[0])

    series = torch.zeros(b, max_f, T, dtype=torch.float32)
    padding_mask = torch.zeros(b, max_f, T, dtype=torch.bool)
    id_mask = torch.ones(b, max_f, T, dtype=torch.long)   # default group 1 (padded)
    future_labels = torch.zeros(b, P, dtype=torch.long)
    var_counts = torch.zeros(b, dtype=torch.long)

    for i, w in enumerate(windows):
        tgt = np.asarray(w["target"], dtype=np.float32)   # (F, T)
        f = tgt.shape[0]
        valid = ~np.isnan(tgt)                            # NaN = missing (e.g. normal-prefix pad)
        tgt = np.nan_to_num(tgt, nan=0.0)
        series[i, :f] = torch.from_numpy(tgt)
        padding_mask[i, :f] = torch.from_numpy(valid)
        id_mask[i, :f] = 0                                # real variates -> group 0
        future_labels[i] = torch.as_tensor(np.asarray(w["future_labels"], dtype=np.int64))
        var_counts[i] = f

    return TotoAnomalyBatch(series, padding_mask, id_mask, future_labels, var_counts)


# ─────────────────────────────────────────────────────────────────────────────
#  Hierarchical sampler
# ─────────────────────────────────────────────────────────────────────────────
def _cum_from_weights(w: np.ndarray):
    """Normalized cumulative distribution from non-negative weights, or None if all zero."""
    s = float(w.sum())
    if s <= 0:
        return None
    return np.cumsum(w / s)


class HSSampler:
    """Hierarchical dataset→kind→window sampler. Yields batches of window indices."""

    def __init__(self, pool: AnomalyPool, batch_windows: int, p_anom: float = 1.0 / 3.0, seed: int = 42):
        self.pool = pool
        self.batch_windows = int(batch_windows)
        self.p_anom = float(p_anom)
        if not 0.0 <= self.p_anom <= 1.0:
            raise ValueError(f"p_anom must be in [0,1], got {self.p_anom}")
        self.rng = np.random.default_rng(seed)
        self.n_ds = len(pool.ds_names)
        if self.n_ds == 0:
            raise ValueError("empty training pool")

        H = pool.H
        self._groups = []  # per dataset: (member_indices, cum_normal, cum_anomalous)
        for k, name in enumerate(pool.ds_names):
            members = np.flatnonzero(pool.ds_of_window == k)
            n_a = pool.n_anom[members].astype(np.float64)
            n_n = float(H) - n_a
            cum_a = _cum_from_weights(n_a)
            cum_n = _cum_from_weights(n_n)
            if cum_a is None:
                logger.warning(f"[hs] dataset '{name}' has zero anomaly steps; anomalous branch -> normal branch.")
                cum_a = cum_n
            if cum_n is None:
                logger.warning(f"[hs] dataset '{name}' has zero normal steps; normal branch -> anomalous branch.")
                cum_n = cum_a
            if cum_a is None:
                raise RuntimeError(f"[hs] dataset '{name}' has no drawable windows.")
            self._groups.append((members, cum_n, cum_a))

        # Monitoring
        self._seen_batches = 0
        self._seen_anom_steps = 0
        self._seen_total_steps = 0
        self._seen_ds_draws = np.zeros(self.n_ds, dtype=np.int64)

    def _draw_window(self) -> int:
        k = self.rng.integers(self.n_ds)                                  # level 1
        members, cum_n, cum_a = self._groups[k]
        cum = cum_a if self.rng.random() < self.p_anom else cum_n         # level 2
        j = int(np.searchsorted(cum, self.rng.random(), side="right"))    # level 3
        if j >= len(members):
            j = len(members) - 1
        self._seen_ds_draws[k] += 1
        return int(members[j])

    def draw_batch(self):
        """Return a list of `batch_windows` global window indices."""
        idx = [self._draw_window() for _ in range(self.batch_windows)]
        self._seen_batches += 1
        self._seen_anom_steps += int(self.pool.n_anom[idx].sum())
        self._seen_total_steps += len(idx) * self.pool.H
        return idx

    def realized_report(self) -> str:
        if self._seen_total_steps == 0:
            return "[hs] no draws yet"
        tot = max(int(self._seen_ds_draws.sum()), 1)
        mix = "  ".join(f"{n}={100.0 * c / tot:.1f}%" for n, c in zip(self.pool.ds_names, self._seen_ds_draws))
        return (
            f"[hs] realized over {self._seen_batches} batches: anomaly-step fraction "
            f"{100.0 * self._seen_anom_steps / self._seen_total_steps:.1f}% "
            f"(uniform dataset target {100.0 / self.n_ds:.1f}%): {mix}"
        )

    def batches(self):
        """Infinite generator of collated training batches."""
        while True:
            idx = self.draw_batch()
            yield collate_windows([self.pool.windows[i] for i in idx])


def iter_eval_batches(pool: AnomalyPool, batch_windows: int):
    """Deterministic in-order batches over an eval pool (val/test)."""
    for i in range(0, len(pool), batch_windows):
        chunk = pool.windows[i : i + batch_windows]
        if chunk:
            yield collate_windows(chunk)
