"""
Single-stage anomaly-aware LoRA fine-tuning for **Toto 1.0** (maskloss v2 + HS).

Port of the Chronos `TOTAL_RUN_maskloss_v2_HS/finetune_anomaly_simple.py` objective
onto Toto, using the `[SEP]/[REG]` wrapper (`toto_anomaly_model.py`) and the
hierarchical sampler (`toto_anomaly_data.py`).

Objective (per-step masked margin):

    L = L_good + margin_lambda * L_bad

  * `future_labels` (0/1 per horizon step) split the 64-step forecast (read from the
    [REG] token) into NORMAL and ANOMALY steps.
  * L_good     = mean per-step loss over NORMAL steps                (minimised)
  * L_bad_term = hinge pushing ANOMALY-step loss UP toward a margin  (self-saturating)
      - hinge_mode=per_step  : hinge each anomaly step then average
      - margin_mode=relative : margin = margin_m * (this window's own normal error)
      - agg_mode=batch_global: pool all steps in the batch equally

The per-step loss is Toto's native `CombinedLoss(distr, scaled_target)` (already
elementwise), so no Toto model change is needed for the loss itself.

Only the LoRA adapters and the `[SEP]/[REG]` `special_tokens` embedding are trained.
Metrics are logged to `trainer_state.json` (schema-compatible with the reference
`loss_curves.ipynb`), split into train / eval_val_* / eval_test_*.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

logger = logging.getLogger("finetune_toto_anomaly")


class TqdmLoggingHandler(logging.StreamHandler):
    """Route log records through tqdm.write so they don't corrupt the progress bar."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)

_HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
#  Margin objective + monitoring
# ─────────────────────────────────────────────────────────────────────────────
def _fresh_acc():
    return {"n_sum": 0.0, "n_mse_sum": 0.0, "n_cnt": 0.0,
            "a_sum": 0.0, "a_mse_sum": 0.0, "a_cnt": 0.0, "a_active_sum": 0.0}


def compute_margin_loss(model, wrapper, combined_loss, batch, args, acc):
    """
    Returns the scalar total loss and updates the monitoring accumulator `acc`
    (running sums split by step label). Mirrors finetune_anomaly_simple.py:463-550.
    """
    input_len = wrapper.input_length
    P = wrapper.prediction_length

    series = batch.series
    pmask = batch.padding_mask
    imask = batch.id_mask
    eps = torch.finfo(series.dtype).eps

    inp = series[..., :input_len]
    inp_pmask = pmask[..., :input_len]
    inp_imask = imask[..., :input_len]

    dist, loc_b, scale_b = model(inp, inp_pmask, inp_imask)  # dist over (B,V,P); loc_b/scale_b (B,V,1)

    horizon = series[..., input_len:input_len + P]           # (B,V,P)
    horizon_valid = pmask[..., input_len:input_len + P]      # (B,V,P) bool
    scaled_target = (horizon - loc_b) / (scale_b + eps)

    per_step = combined_loss(dist, scaled_target)            # (B,V,P) per-step NLL+robust
    with torch.no_grad():
        mean_pred = dist.mean                                # (B,V,P) scaled-space mean
        per_step_mse = (scaled_target - mean_pred) ** 2      # (B,V,P) scaled-space MSE

    # future_labels are per horizon STEP (shared across variates); AND with validity.
    labels = batch.future_labels.unsqueeze(1).expand(-1, series.shape[1], -1)  # (B,V,P)
    valid = horizon_valid.float()
    normal_step = (labels == 0).float() * valid
    anomaly_step = (labels == 1).float() * valid

    eps_c = 1.0
    # Per-window (= per batch element) normal reference, pooling over (V, step).
    cnt_norm_w = normal_step.sum(dim=(1, 2))                 # (B,)
    cnt_anom_w = anomaly_step.sum(dim=(1, 2))
    has_norm = (cnt_norm_w > 0).float()
    has_anom = (cnt_anom_w > 0).float()
    n_norm_steps = normal_step.sum()
    n_anom_steps = anomaly_step.sum()

    L_good_w = (per_step * normal_step).sum(dim=(1, 2)) / cnt_norm_w.clamp(min=eps_c)  # (B,)

    if args.margin_mode == "relative":
        # NOTE (Toto vs Chronos): Chronos uses a MULTIPLICATIVE relative margin
        # (margin_m * L_good) because its per-step loss is pinball (>= 0). Toto's
        # per-step loss is an NLL that goes NEGATIVE for well-predicted steps, so a
        # multiplicative margin points the wrong way (5 * -2 = -10 < -2) and the hinge
        # goes inert. The sign-safe generalization is an ADDITIVE gap: the anomaly loss
        # must exceed this window's own normal loss by `margin_m` nats. This keeps the
        # self-scaling-per-window property the relative margin was designed for.
        margin_eff = (L_good_w.detach() + args.margin_m).view(-1, 1, 1)  # (B,1,1)
    else:
        margin_eff = per_step.new_full((), float(args.margin_tau))

    # ── monitoring accumulation ────────────────────────────────────────────────
    acc["n_sum"] += (per_step * normal_step).sum().item()
    acc["n_mse_sum"] += (per_step_mse * normal_step).sum().item()
    acc["n_cnt"] += n_norm_steps.item()
    acc["a_sum"] += (per_step * anomaly_step).sum().item()
    acc["a_mse_sum"] += (per_step_mse * anomaly_step).sum().item()
    acc["a_cnt"] += n_anom_steps.item()
    with torch.no_grad():
        step_active = ((margin_eff - per_step) > 0).float() * anomaly_step
        acc["a_active_sum"] += step_active.sum().item()

    # ── L_good ─────────────────────────────────────────────────────────────────
    if args.agg_mode == "batch_global":
        L_good = (per_step * normal_step).sum() / n_norm_steps.clamp(min=eps_c)
    else:  # per_window
        L_good = (L_good_w * has_norm).sum() / has_norm.sum().clamp(min=eps_c)

    # ── L_bad_term ─────────────────────────────────────────────────────────────
    if args.hinge_mode == "per_step":
        step_hinge = torch.clamp(margin_eff - per_step, min=0.0) * anomaly_step
        if args.agg_mode == "batch_global":
            L_bad_term = step_hinge.sum() / n_anom_steps.clamp(min=eps_c)
        else:
            hinge_w = step_hinge.sum(dim=(1, 2)) / cnt_anom_w.clamp(min=eps_c)
            L_bad_term = (hinge_w * has_anom).sum() / has_anom.sum().clamp(min=eps_c)
    else:  # pooled
        if args.margin_mode == "relative":
            L_bad_w = (per_step * anomaly_step).sum(dim=(1, 2)) / cnt_anom_w.clamp(min=eps_c)
            margin_w = L_good_w.detach() + args.margin_m  # additive gap (NLL-safe; see note above)
            hinge_w = torch.clamp(margin_w - L_bad_w, min=0.0) * has_anom
            if args.agg_mode == "batch_global":
                L_bad_term = (hinge_w * cnt_anom_w).sum() / n_anom_steps.clamp(min=eps_c)
            else:
                L_bad_term = hinge_w.sum() / has_anom.sum().clamp(min=eps_c)
        else:
            if args.agg_mode == "batch_global":
                L_bad = (per_step * anomaly_step).sum() / n_anom_steps.clamp(min=eps_c)
                L_bad_term = torch.clamp(args.margin_tau - L_bad, min=0.0)
            else:
                L_bad_w = (per_step * anomaly_step).sum(dim=(1, 2)) / cnt_anom_w.clamp(min=eps_c)
                hinge_w = torch.clamp(args.margin_tau - L_bad_w, min=0.0) * has_anom
                L_bad_term = hinge_w.sum() / has_anom.sum().clamp(min=eps_c)

    return L_good + args.margin_lambda * L_bad_term


def acc_to_metrics(acc, prefix=""):
    out = {}
    if acc["n_cnt"] > 0:
        out[f"{prefix}normal_loss"] = round(acc["n_sum"] / acc["n_cnt"], 4)
        out[f"{prefix}mse_normal_step"] = round(acc["n_mse_sum"] / acc["n_cnt"], 4)
    if acc["a_cnt"] > 0:
        out[f"{prefix}anomaly_loss"] = round(acc["a_sum"] / acc["a_cnt"], 4)
        out[f"{prefix}mse_anomaly_step"] = round(acc["a_mse_sum"] / acc["a_cnt"], 4)
        out[f"{prefix}anomaly_active_frac"] = round(acc["a_active_sum"] / acc["a_cnt"], 4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA setup
# ─────────────────────────────────────────────────────────────────────────────
def build_lora_model(wrapper, args):
    from peft import LoraConfig, get_peft_model

    target_modules = sorted(
        name
        for name, mod in wrapper.named_modules()
        if isinstance(mod, nn.Linear) and (name.startswith("backbone.transformer") or name == "backbone.unembed")
    )
    if not target_modules:
        raise RuntimeError("No LoRA target Linear modules found under backbone.transformer")
    logger.info(f"LoRA targets {len(target_modules)} Linear modules (e.g. {target_modules[:3]} ...)")

    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=["special_tokens"],  # keep [SEP]/[REG] trainable + checkpointed
        bias="none",
    )
    model = get_peft_model(wrapper, cfg)
    return model


def log_trainable(model):
    trainable, total = 0, 0
    names = []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if "special_tokens" in n or ("lora" in n.lower() and len(names) < 4):
                names.append(n)
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.3f}%)")
    logger.info(f"  sample trainable: {names}")
    # Sanity: special_tokens must be trainable
    assert any("special_tokens" in n and p.requires_grad for n, p in model.named_parameters()), \
        "special_tokens is not trainable — check modules_to_save"


# ─────────────────────────────────────────────────────────────────────────────
#  Eval
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, wrapper, combined_loss, pool, subset, args, device):
    """Evaluate over `subset` (built once by build_eval_subset), which gives every
    dataset in the pool an equal window budget -- so a plain mean over it is already
    balanced across datasets. Per-dataset losses are returned alongside the mean so the
    spread across datasets is visible in the curves.
    """
    from toto_anomaly_data import iter_eval_batches

    model.eval()
    acc = _fresh_acc()
    ds_sum, ds_cnt = defaultdict(float), defaultdict(int)
    loss_sum, n_batches = 0.0, 0
    for ds, batch in iter_eval_batches(pool, args.eval_batch_windows, subset):
        batch = batch.to(device)
        loss = compute_margin_loss(model, wrapper, combined_loss, batch, args, acc)
        l = float(loss.item())
        loss_sum += l
        n_batches += 1
        ds_sum[ds] += l
        ds_cnt[ds] += 1
    model.train()

    if not n_batches:
        return {}
    metrics = acc_to_metrics(acc)
    metrics["loss"] = round(loss_sum / n_batches, 4)
    for ds in ds_cnt:
        metrics[f"ds_{ds}_loss"] = round(ds_sum[ds] / ds_cnt[ds], 4)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[TqdmLoggingHandler(),
                  logging.FileHandler(os.path.join(args.output_dir, "finetune.log"))],
    )
    logger.info(f"Config: {vars(args)}")
    torch.manual_seed(args.seed)

    from toto.model.losses import CombinedLoss
    from toto.model.scheduler import WarmupStableDecayLR
    from toto_anomaly_data import HSSampler, build_eval_subset, load_eval_pool, load_train_pool
    from toto_anomaly_model import TotoAnomalyModel

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Device: {device}")

    # Model
    wrapper = TotoAnomalyModel.from_pretrained(
        args.pretrained_model,
        normal_signal_length=args.normal_signal_length,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )
    model = build_lora_model(wrapper, args)
    model.to(device)
    log_trainable(model)
    combined_loss = CombinedLoss()

    # Data
    datasets = args.datasets if args.datasets else None
    train_pool = load_train_pool(args.prepared_dir, datasets, args.prediction_length)
    sampler = HSSampler(train_pool, args.train_batch_windows, p_anom=args.p_anom, seed=args.seed)
    train_gen = sampler.batches()

    val_pool = load_eval_pool(args.prepared_dir, "val", datasets, args.prediction_length)
    test_pool = load_eval_pool(args.prepared_dir, "test", datasets, args.prediction_length)
    # Fixed, dataset-stratified eval subsets, built once and reused at every eval so the
    # curves are comparable step to step.
    val_subset = build_eval_subset(val_pool, args.eval_windows, p_anom=args.p_anom, seed=args.seed)
    test_subset = build_eval_subset(test_pool, args.eval_windows, p_anom=args.p_anom, seed=args.seed)

    # Optim
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=args.weight_decay)
    scheduler = WarmupStableDecayLR(
        optimizer=optimizer, warmup_steps=args.warmup_steps, stable_steps=args.stable_steps,
        decay_steps=args.decay_steps, min_lr=args.min_lr, base_lr=args.lr,
    )

    log_history = []

    def flush_state():
        with open(os.path.join(args.output_dir, "trainer_state.json"), "w") as f:
            json.dump({"log_history": log_history}, f, indent=1)

    model.train()
    train_acc = _fresh_acc()
    running_loss, running_n = 0.0, 0
    best_val = {"loss": float("inf"), "step": -1}

    pbar = tqdm(
        range(1, args.max_steps + 1),
        desc="train",
        unit="step",
        dynamic_ncols=True,
    )
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            batch = next(train_gen).to(device)
            loss = compute_margin_loss(model, wrapper, combined_loss, batch, args, train_acc)
            (loss / args.grad_accum).backward()
            step_loss += float(loss.item()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        running_loss += step_loss
        running_n += 1
        pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if step % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            entry = {"step": step, "loss": round(running_loss / running_n, 4), "lr": lr}
            entry.update(acc_to_metrics(train_acc))
            log_history.append(entry)
            logger.info(
                f"step {step}/{args.max_steps}  loss={entry['loss']}  "
                f"normal={entry.get('normal_loss')}  anomaly={entry.get('anomaly_loss')}  "
                f"active={entry.get('anomaly_active_frac')}  lr={lr:.2e}"
            )
            logger.info("  " + sampler.realized_report())
            train_acc = _fresh_acc()
            running_loss, running_n = 0.0, 0
            flush_state()

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.max_steps):
            eval_entry = {"step": step}
            if val_subset:
                vm = evaluate(model, wrapper, combined_loss, val_pool, val_subset, args, device)
                eval_entry.update({f"eval_val_{k}": v for k, v in vm.items()})
            if test_subset:
                tm = evaluate(model, wrapper, combined_loss, test_pool, test_subset, args, device)
                eval_entry.update({f"eval_test_{k}": v for k, v in tm.items()})
            log_history.append(eval_entry)
            logger.info(f"[eval] step {step}: {eval_entry}")
            flush_state()

            # Keep the best-validation adapter so a long run isn't all-or-nothing.
            v_loss = eval_entry.get("eval_val_loss")
            if v_loss is not None and v_loss < best_val["loss"]:
                best_val.update(loss=v_loss, step=step)
                model.save_pretrained(os.path.join(args.output_dir, "best-ckpt"))
                logger.info(f"  new best val loss {v_loss:.4f} @ step {step} -> best-ckpt")

    pbar.close()
    logger.info(f"Best val loss {best_val['loss']:.4f} @ step {best_val['step']}")

    # Save adapter (includes special_tokens via modules_to_save) + meta
    ckpt_dir = os.path.join(args.output_dir, "finetuned-ckpt")
    model.save_pretrained(ckpt_dir)
    meta = {
        "pretrained_model": args.pretrained_model,
        "normal_signal_length": args.normal_signal_length,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "patch_size": wrapper.patch_size,
    }
    with open(os.path.join(ckpt_dir, "toto_ft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    flush_state()
    logger.info(f"Saved adapter + meta -> {ckpt_dir}")
    logger.info("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Toto anomaly LoRA fine-tuning (maskloss v2 + HS)")
    p.add_argument("--prepared_dir", default=os.path.join(_HERE, "prepared_total"))
    p.add_argument("--output_dir", default=os.path.join(_HERE, "toto-single-stage_mtsbench_HS"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--pretrained_model", default="Datadog/Toto-Open-Base-1.0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # geometry (must match prepare_total)
    p.add_argument("--normal_signal_length", type=int, default=256)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # objective
    p.add_argument("--margin_m", type=float, default=5.0,
                   help="relative-margin gap (NLL nats): anomaly-step loss is pushed to exceed "
                        "the window's own normal loss by this much (additive, sign-safe for NLL).")
    p.add_argument("--margin_tau", type=float, default=12.0)
    p.add_argument("--margin_lambda", type=float, default=1.0)
    p.add_argument("--hinge_mode", choices=["per_step", "pooled"], default="per_step")
    p.add_argument("--margin_mode", choices=["relative", "absolute"], default="relative")
    p.add_argument("--agg_mode", choices=["batch_global", "per_window"], default="batch_global")
    p.add_argument("--p_anom", type=float, default=1.0 / 3.0)

    # optim / schedule
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--stable_steps", type=int, default=1000)
    p.add_argument("--decay_steps", type=int, default=1000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # loop
    p.add_argument("--max_steps", type=int, default=2200)
    p.add_argument("--train_batch_windows", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batch_windows", type=int, default=8)
    p.add_argument("--eval_windows", type=int, default=1320,
                   help="Total windows in the fixed eval subset, split equally across datasets "
                        "(11 val datasets -> 120 each). Replaces the old --eval_max_batches, which "
                        "walked the pool in order and so only ever reached the first dataset.")
    return p.parse_args()


if __name__ == "__main__":
    main()
