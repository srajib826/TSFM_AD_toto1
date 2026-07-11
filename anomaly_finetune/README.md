# Toto 1.0 anomaly-detection fine-tuning (maskloss v2 + HS)

Ports the Chronos-2 anomaly fine-tuning method onto **Toto 1.0**: LoRA fine-tuning
with a per-step masked-margin loss, a hierarchical sampler, learnable `[SEP]`/`[REG]`
tokens, and VUS-PR/-ROC evaluation on mTSBench.

Everything lives here in `/home/rajib/toto/anomaly_finetune/` (nothing is written into
the Chronos workspace).

---

## 1. One-time setup

```bash
conda create -y -n toto_ft python=3.11
conda activate toto_ft
pip install -r requirements.txt
pip install -e /home/rajib/toto            # the Toto 1.0 package itself
```

> **Note:** `requirements.txt` pins `setuptools<81` on purpose — setuptools 81+ removed
> `pkg_resources`, which `lightning.fabric` (a transitive Toto dep) imports, so a newer
> setuptools makes `import toto` fail.

External resources used (read-only, referenced by absolute path):
- Raw data: `/home/rajib/mTSBench/Datasets/mTSBench`
- VUS metrics package: `VUS_ROC_VUS_PR` under
  `/home/rajib/Sir_git_TSAD/TSFM-anomaly/Chronos_Finetuning/rajib_work_space`
  (added to `PYTHONPATH` by `run_forward_total.sh`).

---

## 2. What to run (in order)

### Step 1 — Prepare data (raw CSVs → windows, train/val/test)
```bash
bash run_prepare_total.sh
# subset:   DATASETS="SMD MSL SMAP" bash run_prepare_total.sh
```
Writes `prepared_total/per_dataset/<DS>/{train,val,test}_model_inputs.pkl` (+ n_anom
sidecars, `test_series_meta.pkl`) and `manifest.json`. Each window is
`target (F, 832)` = `[normal(256) | context(512) | future(64)]` plus
`future_labels (64,)`.

### Step 2 — LoRA fine-tune
```bash
bash run_finetune_total.sh
```
Trains LoRA adapters + the learnable `[SEP]`/`[REG]` tokens with the mask-loss margin
objective and the hierarchical sampler. Outputs to
`toto-single-stage_mtsbench_HS/`:
- `finetuned-ckpt/` — LoRA adapter + trained `[SEP]`/`[REG]` (`special_tokens`) + `toto_ft_meta.json`
- `trainer_state.json` — loss history (train + `eval_val_*` + `eval_test_*`)

Common overrides (env vars): `MARGIN_M`, `MARGIN_LAMBDA`, `P_ANOM`, `LR`, `MAX_STEPS`,
`TRAIN_BATCH_WINDOWS`, `GRAD_ACCUM`, `LORA_R`, `DATASETS`, `DEVICE`.

### Step 3 — Evaluate (VUS-PR / VUS-ROC / AUC / F1)
```bash
# fine-tuned model
CHECKPOINT=toto-single-stage_mtsbench_HS/finetuned-ckpt bash run_forward_total.sh
# zero-shot base (no --checkpoint) for comparison
bash run_forward_total.sh
```
Writes per-series metrics to `eval_results.csv` and prints the mean.

### Step 4 — Plot the loss / forecasting-error curves
```bash
python plot_loss_curves.py --run_dir toto-single-stage_mtsbench_HS   # PNGs
# or open loss_curves.ipynb and set RUN_DIR
```

---

## 3. Files

| File | Role |
|---|---|
| `toto_prep_lib.py`, `prepare_total.py` | data prep: mTSBench → windows, train/val/test splits |
| `toto_anomaly_model.py` | `TotoAnomalyModel` — `[NORMAL][SEP][CONTEXT][REG][HORIZON]`, horizon read from `[REG]` |
| `toto_anomaly_data.py` | window pool loader + hierarchical sampler (HS) + collation |
| `finetune_toto_anomaly.py` | LoRA + per-step mask-loss margin objective + eval + `trainer_state.json` |
| `forward.py` | `[REG]`-readout forecast → interval scoring → VUS metrics |
| `run_prepare/finetune/forward_total.sh` | runners (activate `toto_ft`) |
| `loss_curves.ipynb`, `plot_loss_curves.py` | forecasting-error / loss curves |
| `requirements.txt` | pinned deps for the `toto_ft` env |

---

## 4. Key design notes

- **Architecture change:** `[SEP]` is inserted after the normal-signal prefix and
  `[REG]` after the context; `[REG]`'s output patch is the 64-step horizon forecast
  (Toto's analogue of Chronos's REG-driven forecast). Implemented as an
  `nn.Embedding(2, D)` (`special_tokens`) on a wrapper around the pretrained backbone —
  **no edits to the Toto library**. LoRA keeps it trainable via
  `modules_to_save=["special_tokens"]`.
- **Objective:** `L = L_good + margin_lambda · L_bad`. `future_labels` split the
  horizon into normal/anomaly steps; `L_good` minimizes normal-step loss, `L_bad`
  hinges anomaly-step loss **up** toward a margin. HS balances datasets (level 1
  uniform) and class (level 2 `p_anom=1/3`, level 3 count-weighted) — no threshold,
  nothing discarded.
- **Deviation from Chronos (important):** Toto's per-step loss is an NLL that goes
  **negative** for well-predicted steps, so Chronos's *multiplicative* relative margin
  (`margin_m × L_good`) goes inert. This pipeline uses an **additive** relative margin:
  anomaly-step loss must exceed the window's own normal loss by `margin_m` nats.

---

## 5. Repo cleanup (optional, unrelated to this pipeline)

These monorepo folders are **not** used by this pipeline or by Toto 1.0 and can be
removed if you want a leaner tree: `toto2/`, `boom/`, `dd_unit_scaling/`,
`toto_ts.egg-info/`, `pytest.ini` (Toto-1.0 test-suite config only). After deleting,
re-run `pip install -e /home/rajib/toto` to refresh the editable-install metadata.
