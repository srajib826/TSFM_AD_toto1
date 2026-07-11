"""
Render Toto anomaly fine-tuning loss curves from a run's `trainer_state.json`
(headless CLI version of `loss_curves.ipynb`).

Plots, over training steps, with eval_val_* / eval_test_* overlays auto-detected:
  - Total loss (`loss`)
  - Pinball loss split normal vs anomaly (`normal_loss` / `anomaly_loss`)
  - MSE split normal vs anomaly (`mse_normal_step` / `mse_anomaly_step`)
  - Anomaly hinge-active fraction (`anomaly_active_frac`)
  - Learning rate (`lr`)

The training loop emits exactly the schema the reference Chronos
`TOTAL_RUN_maskloss_v2/loss_curves.ipynb` reads.

Usage:
    python plot_loss_curves.py --run_dir toto-single-stage_mtsbench_HS
    python plot_loss_curves.py --state path/to/trainer_state.json --out curves
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_state(run_dir, state):
    if state:
        return state
    if run_dir:
        return os.path.join(run_dir, "trainer_state.json")
    found = glob.glob("**/trainer_state.json", recursive=True)
    assert found, "no trainer_state.json found"
    return max(found, key=os.path.getmtime)


def plot_split(df, normal_col, anomaly_col, title, ylabel, out_png):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, color in [(normal_col, "#2ca02c"), (anomaly_col, "#d62728")]:
        if col in df:
            s = df[df[col].notna()]
            if len(s):
                ax.plot(s["step"], s[col], color=color, label=f"train {col}")
        for c in [c for c in df.columns if c.startswith("eval") and c.endswith(col)]:
            s = df[df[c].notna()]
            if len(s):
                ax.plot(s["step"], s[c], marker="o", ms=3, ls="--", label=c)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_png}")


def plot_single(df, col, title, ylabel, out_png):
    if col not in df:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    s = df[df[col].notna()]
    ax.plot(s["step"], s[col], label=f"train {col}")
    for c in [c for c in df.columns if c.startswith("eval") and c.endswith(col)]:
        s2 = df[df[c].notna()]
        if len(s2):
            ax.plot(s2["step"], s2[c], marker="o", ms=3, ls="--", label=c)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=None)
    p.add_argument("--state", default=None)
    p.add_argument("--out", default=None, help="output dir for PNGs (default: run dir / cwd)")
    args = p.parse_args()

    state_path = find_state(args.run_dir, args.state)
    print("reading:", state_path)
    with open(state_path) as f:
        hist = json.load(f)["log_history"]
    df = pd.DataFrame(hist)
    print(f"{len(df)} log entries; columns: {list(df.columns)}")

    out_dir = args.out or (args.run_dir or os.path.dirname(state_path) or ".")
    os.makedirs(out_dir, exist_ok=True)

    plot_single(df, "loss", "Total loss (L_good + lambda * L_bad)", "loss",
                os.path.join(out_dir, "curve_total_loss.png"))
    plot_split(df, "normal_loss", "anomaly_loss", "Pinball loss (per-step)", "loss",
               os.path.join(out_dir, "curve_pinball.png"))
    plot_split(df, "mse_normal_step", "mse_anomaly_step", "MSE (per-step squared error)", "MSE",
               os.path.join(out_dir, "curve_mse.png"))
    plot_single(df, "anomaly_active_frac", "Anomaly hinge-active fraction", "fraction",
                os.path.join(out_dir, "curve_active_frac.png"))
    plot_single(df, "lr", "Learning rate", "lr", os.path.join(out_dir, "curve_lr.png"))


if __name__ == "__main__":
    main()
