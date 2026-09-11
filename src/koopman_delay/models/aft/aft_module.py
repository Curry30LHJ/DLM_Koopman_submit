"""Independent causal attention-free latent memory; no reference code is copied."""

from __future__ import annotations

import math
import torch
from torch import nn


class AttentionFreeLatentMemory(nn.Module):
    def __init__(self, latent_dim: int, context_length: int) -> None:
        super().__init__()
        self.latent_dim, self.context_length = latent_dim, context_length
        self.Wq, self.Wk, self.Wv = (nn.Linear(latent_dim, latent_dim) for _ in range(3))
        self.position_bias = nn.Parameter(torch.zeros(context_length, context_length))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer("future_mask", torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.latent_dim or history.shape[1] > self.context_length:
            raise ValueError("history must be [B,T,latent_dim] with T <= context_length")
        length = history.shape[1]
        q, k, v = torch.sigmoid(self.Wq(history)), self.Wk(history) / math.sqrt(self.latent_dim), self.Wv(history) / math.sqrt(self.latent_dim)
        logits = k.unsqueeze(1) + self.position_bias[:length, :length].unsqueeze(0).unsqueeze(-1)
        mask = self.future_mask[:length, :length]
        if bool(mask.all()):
            return torch.zeros_like(history[:, -1])
        masked_logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
        weights = torch.softmax(masked_logits, dim=2)
        attended = q * (weights * v.unsqueeze(1)).sum(dim=2)
        return self.alpha * attended[:, -1]
