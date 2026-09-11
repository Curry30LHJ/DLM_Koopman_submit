"""Pure history-flattened MLP Koopman model for Delayed Lorenz Benchmark III."""
from __future__ import annotations

import torch
from torch import nn

from .common import ControlledKoopmanModel, validate_batch


class MLPKoopman(ControlledKoopmanModel):
    """Flatten a fixed state history, lift it with a small MLP, and propagate z."""

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 0,
        history_horizon: int = 20,
        lift_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history_horizon = int(history_horizon)
        self.lift_dim = int(lift_dim)
        self.hidden_dim = int(hidden_dim)
        self.z_dim = self.state_dim + self.lift_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.history_horizon * self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.lift_dim),
        )
        self.A = nn.Parameter(torch.empty(self.z_dim, self.z_dim))
        self.B = nn.Parameter(torch.empty(self.action_dim, self.z_dim))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        validate_batch(batch, self.history_horizon, self.state_dim, self.action_dim)
        states = batch["states_history"]
        flat = states.reshape(states.shape[0], self.history_horizon * self.state_dim)
        lift = self.encoder(flat)
        return torch.cat([states[:, -1], lift], dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, : self.state_dim]

    def advance(self, z: torch.Tensor, u: torch.Tensor, _history: list[torch.Tensor]) -> torch.Tensor:
        return z @ self.A + u @ self.B

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["MLPKoopman"]
