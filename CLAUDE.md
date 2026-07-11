# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This monorepo contains four independent sub-packages:

| Directory | Package | Purpose |
|-----------|---------|---------|
| `toto/` | `toto-ts` (PyPI) | Toto 1.0 — legacy, supports fine-tuning and exogenous variables |
| `toto2/` | `toto-2` (PyPI) | Toto 2.0 — current, u-μP-scaled, inference-only |
| `dd_unit_scaling/` | `dd-unit-scaling` (PyPI) | Compile-friendly u-μP extension used internally by Toto 2.0 |
| `boom/` | — | BOOM benchmark evaluation notebooks and utilities |

The root-level `pyproject.toml`, `requirements.txt`, `pytest.ini`, and `mypy.ini` belong to **Toto 1.0** (the `toto/` sub-package). Toto 2.0 and dd-unit-scaling have their own `pyproject.toml` inside their directories.

## Commands

### Toto 1.0 (`toto/`)

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run all non-GPU tests
pytest toto/test -m "not cuda"

# Run a single test file
pytest toto/test/model/scaler_test.py

# Format
black toto

# Type check
mypy toto --config-file mypy.ini --check-untyped-defs
```

### Toto 2.0 (`toto2/`)

```bash
# Install
pip install "toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"
# or for development:
cd toto2 && pip install -e ".[dev]"

# Run tests
pytest toto2/tests/

# Lint / format
ruff check toto2
ruff format toto2
```

### dd-unit-scaling (`dd_unit_scaling/`)

```bash
cd dd_unit_scaling && pip install -e ".[optim]"
ruff check dd_unit_scaling
```

## Code Style

- **Toto 1.0**: `black` (line-length 120) + `isort` (black profile); checked in CI.
- **Toto 2.0 / dd-unit-scaling**: `ruff` (line-length 120, rules E/F/I).
- Pre-commit hooks run `black` and `isort` on `toto/`.

## Architecture

### Toto 2.0 (`toto2/toto2/model.py`)

Entire inference-only implementation lives in a single file. Key components:

- **`Toto2Model`** — top-level `nn.Module`, loads from HuggingFace via `PyTorchModelHubMixin`. Has a `forecast()` method that returns 9 quantile levels `[0.1, …, 0.9]` with shape `(9, batch, n_variates, horizon)`.
- **`VariateTimeTransformerDecoder`** — decoder-only transformer with alternating *time* attention layers (causal, with RoPE) and *variate* attention layers (full attention across variates). The alternation pattern is controlled by `layer_group_size` and `num_variate_layers_per_group` in `Toto2ModelConfig`.
- **`PatchedCausalStdScaler`** — causal standard-deviation scaler; statistics are computed per-patch in a rolling fashion so the model never sees future statistics.
- **`KVCache` / `StaticKVCacheLayer`** — pre-allocated KV cache enabling *block decoding*: the context is encoded once, then the prediction horizon is decoded in blocks, feeding each block's median back as context for the next. Enabled by setting `decode_block_size` in `forecast()`.
- **`QuantileKnotsOutputHead`** — projects transformer outputs to 9 fixed quantile knots via a residual MLP.
- **`Toto2GluonTSModel`** — wraps `Toto2Model` in a GluonTS-compatible interface.

`Toto2ModelConfig` (in `toto2/toto2/configuration.py`) controls architecture. `residual_attn_ratio` must be set explicitly (use `Toto2ModelConfig.compute_residual_attn_ratio(context_length, patch_size)`).

### Toto 1.0 (`toto/`)

- **`TotoBackbone`** (`model/backbone.py`) — transformer with *Proportional Factorized Space-Time Attention* (alternating timewise + spacewise layers). Outputs a parametric distribution (Student-T mixture by default).
- **`Toto`** (`model/toto.py`) — thin wrapper around `TotoBackbone` with HuggingFace hub loading and state-dict key remapping for fused/unfused SwiGLU.
- **`TotoForecaster`** (`inference/forecaster.py`) — sampling-based probabilistic forecaster; returns `forecast.median`, `forecast.samples`, `forecast.quantile(q)`.
- **`MaskedTimeseries` / `CausalMaskedTimeseries`** (`data/util/dataset.py`) — input data containers. Fine-tuning uses `CausalMaskedTimeseries`.
- **`FinetuneDataModule`** (`data/datamodule/finetune_datamodule.py`) — Lightning data module for fine-tuning; requires `collate_causal` as `collate_fn`.

Fine-tuning is driven by `toto/scripts/finetune_toto.py` and configured via `toto/scripts/configs/finetune_config.yaml`.

### dd-unit-scaling (`dd_unit_scaling/`)

Provides `dd_unit_scaling` (imported as `uu`) with u-μP-aware replacements for `nn.Linear`, `nn.RMSNorm`, `nn.Dropout`, and `DepthModuleList`. Also exports `functional` (imported as `U`) with `residual_split`, `residual_add`, and `silu`. These are used pervasively in Toto 2.0.

## Test Markers (Toto 1.0)

- `cuda` — requires a CUDA GPU; excluded from CPU CI
- `real_aws` — requires real AWS credentials
- `stress` — stress tests

Use `-m "not cuda"` to run only CPU-compatible tests.
