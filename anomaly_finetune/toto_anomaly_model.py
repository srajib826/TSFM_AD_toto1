"""
`TotoAnomalyModel` — a thin wrapper around a pretrained Toto 1.0 backbone that adds
learnable `[SEP]` and `[REG]` tokens so the patch sequence reads

        [ NORMAL ][ SEP ][ CONTEXT ][ REG ][ HORIZON ]

exactly like the Chronos-2 anomaly model (`chronos/chronos2/model.py`), but adapted
to Toto's decoder-only, one-patch-ahead architecture.

Toto has no future placeholder patches: the horizon is the causal next-patch
prediction from the last position. So `[REG]` is an extra learnable patch token
appended after the context; **its output is the 64-step horizon forecast**. `[SEP]`
is a learnable patch token inserted at the normal|context boundary.

Design notes
------------
* The two tokens live in a single `nn.Embedding(2, embed_dim)` named
  `special_tokens` (index 0 = SEP, 1 = REG). Being a *module*, it can be passed to
  peft `LoraConfig(modules_to_save=["special_tokens"])` so LoRA keeps it trainable
  and checkpointed alongside the adapter.
* No edit to the Toto library — we compose the backbone's public sub-modules
  (`scaler`, `patch_embed`, `transformer`, `unembed`, `output_distribution`).
* Time attention is causal and RoPE uses sequence-order positions, so inserting
  interior tokens is positionally safe (matches Toto's own fusion-token precedent).
* The distribution head always outputs **scaled space**; `forward` returns
  `(scaled_dist, loc_b, scale_b)` where `loc_b/scale_b` are the causal scale at the
  forecast boundary (last input patch). Training normalizes the horizon target by
  these; inference samples then un-scales with them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat


class TotoAnomalyModel(nn.Module):
    def __init__(
        self,
        backbone,
        normal_signal_length: int = 256,
        context_length: int = 512,
        prediction_length: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.patch_size = backbone.patch_embed.patch_size
        self.stride = backbone.patch_embed.stride

        self.normal_signal_length = normal_signal_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_length = normal_signal_length + context_length  # e.g. 768

        assert self.stride == self.patch_size, (
            f"SEP/REG insertion assumes non-overlapping patches, but stride ({self.stride}) "
            f"!= patch_size ({self.patch_size})."
        )
        assert self.input_length % self.patch_size == 0, (
            f"input_length ({self.input_length}) must be divisible by patch_size ({self.patch_size})"
        )
        assert normal_signal_length % self.patch_size == 0, (
            f"normal_signal_length ({normal_signal_length}) must be divisible by patch_size ({self.patch_size})"
        )
        assert prediction_length == self.patch_size, (
            f"This model reads the horizon from a single [REG] patch, so prediction_length "
            f"({prediction_length}) must equal patch_size ({self.patch_size})."
        )

        self.num_input_patches = self.input_length // self.patch_size  # 12
        self.sep_patch_index = normal_signal_length // self.patch_size  # 4 (after normal patches)

        # [SEP]=0, [REG]=1
        self.special_tokens = nn.Embedding(2, self.embed_dim)
        nn.init.normal_(self.special_tokens.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_pretrained(cls, model_id: str = "Datadog/Toto-Open-Base-1.0", **kwargs):
        from toto.model.toto import Toto

        backbone = Toto.from_pretrained(model_id).model
        return cls(backbone, **kwargs)

    def _insert_special_tokens(self, emb: torch.Tensor, rid: torch.Tensor):
        """
        emb : (B, V, P, D) patch embeddings for the input region (P = num_input_patches)
        rid : (B, V, P)    reduced id mask (constant per variate)

        Returns emb2 (B, V, P+2, D), rid2 (B, V, P+2) with layout
            [ normal(sep_idx) | SEP | context(P-sep_idx) | REG ].
        """
        b, v, p, d = emb.shape
        s = self.sep_patch_index

        sep = repeat(self.special_tokens.weight[0].to(emb.dtype), "d -> b v 1 d", b=b, v=v)
        reg = repeat(self.special_tokens.weight[1].to(emb.dtype), "d -> b v 1 d", b=b, v=v)
        emb2 = torch.cat([emb[:, :, :s], sep, emb[:, :, s:], reg], dim=2)

        # Inserted tokens take their variate's own group id (all patches of a variate
        # share one id), so space-attention grouping is unchanged.
        col = rid[:, :, :1]
        rid2 = torch.cat([rid[:, :, :s], col, rid[:, :, s:], col], dim=2)
        return emb2, rid2

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        inputs: torch.Tensor,          # (B, V, input_length)  = [normal|context]
        input_padding_mask: torch.Tensor,  # (B, V, input_length) bool
        id_mask: torch.Tensor,         # (B, V, input_length) int group ids
    ):
        bb = self.backbone
        assert inputs.shape[-1] == self.input_length, (
            f"expected input length {self.input_length}, got {inputs.shape[-1]}"
        )

        # Causal patch scaling (statistics up to each patch; no future leakage).
        scaled, loc, scale = bb.scaler(
            inputs,
            padding_mask=input_padding_mask,
            weights=torch.ones_like(inputs),
        )
        scaled = scaled.to(next(bb.parameters()).dtype)

        # Patch-embed, then splice in [SEP]/[REG].
        emb, rid = bb.patch_embed(scaled, id_mask)          # (B,V,P,D), (B,V,P)
        emb2, rid2 = self._insert_special_tokens(emb, rid)  # (B,V,P+2,D), (B,V,P+2)

        # Transformer (time attn causal ⇒ REG sees all of NORMAL+SEP+CONTEXT).
        transformed = bb.transformer(emb2, rid2)            # (B,V,P+2,D)

        # Horizon readout from the [REG] token (last position).
        reg_out = transformed[:, :, -1:, :]                 # (B,V,1,D)
        flattened = rearrange(
            bb.unembed(reg_out),
            "b v s (p d) -> b v (s p) d",
            d=self.embed_dim,
        )                                                    # (B,V,prediction_length,D)
        scaled_dist = bb.output_distribution(flattened)      # dist over (B,V,pred_len), scaled space

        # Causal scale at the forecast boundary (last input patch), broadcast over horizon.
        loc_b = loc[..., -1:]                                # (B,V,1)
        scale_b = scale[..., -1:]                            # (B,V,1)
        return scaled_dist, loc_b, scale_b
