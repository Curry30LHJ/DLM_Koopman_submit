"""True HaKAN feature backbone with switchable instance normalization."""
from __future__ import annotations

import torch
from torch import nn

from .hahn_kan_block import HahnKANBlock
from .patch_embedding import PatchEmbedding
from .revin import RevIN


VALID_NORMALIZATION_MODES = {"current_revin", "no_revin", "revin_stats"}


class HaKANBackbone(nn.Module):
    def __init__(
        self,
        channels=3,
        context_length=39,
        patch_len=5,
        stride=2,
        d_model=32,
        depth=5,
        use_revin=True,
        affine=True,
        subtract_last=False,
        normalization_mode="current_revin",
    ):
        super().__init__()
        if normalization_mode not in VALID_NORMALIZATION_MODES:
            raise ValueError(f"normalization_mode must be one of {sorted(VALID_NORMALIZATION_MODES)}")
        if normalization_mode == "current_revin" and not use_revin:
            normalization_mode = "no_revin"
        self.normalization_mode = normalization_mode
        self.use_revin = normalization_mode != "no_revin"
        self.revin = RevIN(channels, affine=affine, subtract_last=subtract_last) if self.use_revin else None
        self.patch_embedding = PatchEmbedding(channels, context_length, patch_len, stride, d_model)
        self.blocks = nn.ModuleList([HahnKANBlock(d_model, self.patch_embedding.patch_num) for _ in range(depth)])
        self.feature_dim = channels * d_model * self.patch_embedding.patch_num
        self.scale_feature_dim = 2 * channels if normalization_mode == "revin_stats" else 0
        self.last_revin_stats = None
        self.last_scale_features = None

    def _normalize(self, history):
        self.last_revin_stats = None
        self.last_scale_features = None
        if self.revin is None:
            return history
        normalized = self.revin.norm(history)
        if self.normalization_mode == "revin_stats":
            location = self.revin.last if self.revin.subtract_last else self.revin.mean
            mean = location.squeeze(1)
            log_stdev = torch.log(self.revin.stdev.squeeze(1) + self.revin.eps)
            self.last_revin_stats = {"mean": mean, "log_stdev": log_stdev}
            self.last_scale_features = torch.cat([mean, log_stdev], dim=-1)
        return normalized

    def forward(self, history):
        x = self._normalize(history)
        x = self.patch_embedding(x.transpose(1, 2))
        b, c, p, d = x.shape
        x = x.reshape(b * c, p, d)
        for block in self.blocks:
            x = x + block(x)
        return x.reshape(b, c, p, d).flatten(1)
