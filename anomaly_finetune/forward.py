"""
Anomaly evaluation for Toto 1.0 on the held-out mTSBench TEST split.

Toto counterpart of the Chronos `TOTAL_RUN_maskloss_v2_HS/forward.py`. Consumes the
ordered per-dataset test pkls from `prepare_total.py`
(`per_dataset/<DS>/{test_model_inputs.pkl, test_series_meta.pkl}`), runs the
`[SEP]/[REG]` Toto model, turns the [REG]-token horizon forecast error into a
per-timestamp anomaly score, reassembles it onto each series' timeline, and computes
VUS-PR / VUS-ROC / AUC-PR / AUC-ROC (+ F1 variants) per series.

Layout per test window (from data prep):
    target = [ normal(N) | context(C) | future(P) ]     (F, N+C+P)
We feed [ normal | context ] (length N+C, with SEP/REG inserted by the model) and
predict the P-step future, comparing to target[:, N+C:].

The SAME script evaluates either model:
    # zero-shot base (untrained SEP/REG)
    python forward.py --datasets SMD
    # fine-tuned LoRA checkpoint
    python forward.py --datasets SMD --checkpoint <out>/finetuned-ckpt

Requires the VUS metrics package on PYTHONPATH (see run_forward_total.sh).
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
#  Per-feature scoring / aggregation / normalization  (verbatim from Chronos)
# ─────────────────────────────────────────────────────────────────────────────
def compute_feature_score(actual, q10, q50, q90, method):
    """All inputs (n_features, P). Returns per-feature per-step score (n_features, P)."""
    if method == "mse":
        return (actual - q50) ** 2
    if method == "smape":
        eps = 1e-8
        return np.abs(actual - q50) / (np.abs(actual) + np.abs(q50) + eps)
    if method == "interval":
        upper = np.maximum(0.0, actual - q90)
        lower = np.maximum(0.0, q10 - actual)
        return upper + lower
    band = (q90 - q10) + 1e-8
    return np.abs(actual - q50) / band


def robust_normalize_rows(mat):
    out = np.empty_like(mat, dtype=float)
    for f in range(mat.shape[0]):
        row = mat[f]
        p1, p99 = np.percentile(row, 1), np.percentile(row, 99)
        denom = p99 - p1
        out[f] = 0.0 if denom < 1e-8 else (np.clip(row, p1, p99) - p1) / denom
    return out


def aggregate_features(mat, method, k):
    if method == "l2":
        return np.sqrt((mat ** 2).sum(axis=0))
    if method == "max":
        return mat.max(axis=0)
    if method == "mean":
        return mat.mean(axis=0)
    k = min(k, mat.shape[0])
    topk = np.sort(mat, axis=0)[-k:, :]
    return topk.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(args, device):
    from toto_anomaly_model import TotoAnomalyModel

    meta = {}
    base_id = args.model_id
    if args.checkpoint:
        meta_path = os.path.join(args.checkpoint, "toto_ft_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        base_id = meta.get("pretrained_model", base_id)

    wrapper = TotoAnomalyModel.from_pretrained(
        base_id,
        normal_signal_length=meta.get("normal_signal_length", args.normal_signal_length),
        context_length=meta.get("context_length", args.context_length),
        prediction_length=meta.get("prediction_length", args.prediction_length),
    )
    if args.checkpoint:
        from peft import PeftModel

        model = PeftModel.from_pretrained(wrapper, args.checkpoint)
        mode = "FINE-TUNED (LoRA + trained SEP/REG)"
    else:
        model = wrapper
        mode = "ZERO-SHOT (base weights, untrained SEP/REG)"
    model.to(device).eval()
    print(f"Loaded [{mode}] from {base_id}"
          + (f" + adapter {args.checkpoint}" if args.checkpoint else ""))
    return model, wrapper


@torch.no_grad()
def predict_quantiles(model, wrapper, chunk_windows, device, num_samples, qlevels):
    """Run the model on a chunk of test windows; return list of (q10,q50,q90) per window."""
    from toto_anomaly_data import collate_windows

    batch = collate_windows(chunk_windows).to(device)
    inp_len = wrapper.input_length
    inp = batch.series[..., :inp_len]
    pmask = batch.padding_mask[..., :inp_len]
    imask = batch.id_mask[..., :inp_len]

    dist, loc_b, scale_b = model(inp, pmask, imask)              # dist over (B,V,P); (B,V,1)
    samples = dist.sample((num_samples,))                        # (S,B,V,P) scaled
    samples = samples * scale_b + loc_b                          # unscale
    ql = torch.tensor(qlevels, device=samples.device, dtype=samples.dtype)

    # Quantile per window (torch.quantile has a max-element limit that a full batch
    # of samples can exceed).
    out = []
    for i, w in enumerate(chunk_windows):
        f = int(w["target"].shape[0])
        q = torch.quantile(samples[:, i, :f], ql, dim=0).float().cpu().numpy()  # (3,f,P)
        out.append((q[0], q[1], q[2]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def eval_dataset(ds, args, model, wrapper, device, get_metrics):
    ds_dir = os.path.join(args.prepared_dir, "per_dataset", ds)
    test_pkl = os.path.join(ds_dir, "test_model_inputs.pkl")
    meta_pkl = os.path.join(ds_dir, "test_series_meta.pkl")
    if not (os.path.exists(test_pkl) and os.path.exists(meta_pkl)):
        print(f"[{ds}] no test pkl, skipping")
        return None
    with open(test_pkl, "rb") as f:
        windows = pickle.load(f)
    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)
    if not windows:
        print(f"[{ds}] empty test set, skipping")
        return None

    N = wrapper.normal_signal_length
    C = wrapper.context_length
    P = wrapper.prediction_length
    fut_lo, fut_hi = N + C, N + C + P
    print(f"[{ds}] {len(windows)} windows / {len(meta)} series; predict {P}, compare target[:,{fut_lo}:{fut_hi}]")

    feat_scores = {sid: np.full((m["n_features"], m["length"]), np.nan, dtype=np.float32) for sid, m in meta.items()}
    covered = {sid: [None, None] for sid in meta}

    qlevels = [0.1, 0.5, 0.9]
    for i in range(0, len(windows), args.batch_windows):
        chunk = windows[i : i + args.batch_windows]
        preds = predict_quantiles(model, wrapper, chunk, device, args.num_samples, qlevels)
        for w, (q10, q50, q90) in zip(chunk, preds):
            actual = np.asarray(w["target"][:, fut_lo:fut_hi], dtype=np.float32)  # (F,P)
            fscore = compute_feature_score(actual, q10, q50, q90, args.score_method)
            sid, fs, fe = w["series_id"], w["future_start"], w["future_end"]
            feat_scores[sid][:, fs:fe] = fscore
            cl, ch = covered[sid]
            covered[sid][0] = fs if cl is None else min(cl, fs)
            covered[sid][1] = fe if ch is None else max(ch, fe)

    results = defaultdict(list)
    for sid, m in meta.items():
        lo, hi = covered[sid]
        if lo is None:
            continue
        fmat = feat_scores[sid][:, lo:hi]
        if args.score_method != "smape":
            fmat = robust_normalize_rows(fmat)
        fmat = np.nan_to_num(fmat, nan=0.0)
        y_score = aggregate_features(fmat, args.agg_method, args.topk)
        if args.smooth_window > 1:
            y_score = uniform_filter1d(y_score, size=args.smooth_window)
        y_true = m["labels"][lo:hi].astype(int)
        if y_true.sum() == 0:
            continue
        res = get_metrics(y_score, y_true, slidingWindow=args.sliding_window_VUS,
                          version=args.vus_version, thre=args.vus_thre)
        print(f"  {sid:<26} VUS-PR={res['VUS-PR']:.4f}  VUS-ROC={res['VUS-ROC']:.4f}  "
              f"AUC-PR={res['AUC-PR']:.4f}  AUC-ROC={res['AUC-ROC']:.4f}")
        results["series_id"].append(sid)
        results["dataset"].append(ds)
        for k in ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
                  "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F"):
            results[k].append(res[k])
    return results


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    from VUS_ROC_VUS_PR.metrics import get_metrics

    model, wrapper = load_model(args, device)

    datasets = args.datasets
    if not datasets:
        per_ds = os.path.join(args.prepared_dir, "per_dataset")
        datasets = sorted(d for d in os.listdir(per_ds) if os.path.isdir(os.path.join(per_ds, d)))

    all_results = defaultdict(list)
    for ds in datasets:
        r = eval_dataset(ds, args, model, wrapper, device, get_metrics)
        if r:
            for k, v in r.items():
                all_results[k].extend(v)

    if not all_results["series_id"]:
        print("No series scored.")
        return

    print(f"\n================ MEAN OVER {len(all_results['series_id'])} SERIES ================")
    for k in ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
              "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F"):
        print(f"  {k:<16}: {np.mean(all_results[k]):.4f}")

    try:
        import pandas as pd

        pd.DataFrame(all_results).to_csv(args.out_csv, index=False)
        print(f"\nPer-series results -> {args.out_csv}")
    except Exception as e:
        print(f"(could not write csv: {e})")


def parse_args():
    p = argparse.ArgumentParser(description="Toto anomaly eval on mTSBench test pkls")
    p.add_argument("--prepared_dir", default=os.path.join(_HERE, "prepared_total"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--model_id", default="Datadog/Toto-Open-Base-1.0")
    p.add_argument("--checkpoint", default=None, help="Path to finetuned-ckpt (LoRA adapter dir)")
    p.add_argument("--device", default="cuda")

    p.add_argument("--normal_signal_length", type=int, default=256)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)

    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--batch_windows", type=int, default=16)

    p.add_argument("--score_method", default="interval",
                   choices=["mse", "interval", "normalized_deviation", "smape"])
    p.add_argument("--agg_method", default="topk_mean", choices=["l2", "max", "mean", "topk_mean"])
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--smooth_window", type=int, default=5)

    p.add_argument("--sliding_window_VUS", type=int, default=100)
    p.add_argument("--vus_version", default="opt", choices=["opt", "opt_mem"])
    p.add_argument("--vus_thre", type=int, default=250)
    p.add_argument("--out_csv", default=os.path.join(_HERE, "eval_results.csv"))
    return p.parse_args()


if __name__ == "__main__":
    main()
