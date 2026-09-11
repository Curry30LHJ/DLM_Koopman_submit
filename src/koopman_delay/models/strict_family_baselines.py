"""Contract-exact H20 strict-family baseline encoders for Benchmark I."""
from __future__ import annotations

import torch
from torch import nn

from .common import ControlledKoopmanModel, validate_batch
from .hakan.backbone import HaKANBackbone


class _StrictFamilyKoopman(ControlledKoopmanModel):
    """Common controlled Koopman state-prefix and linear-dynamics contract."""

    def __init__(self, state_dim: int = 9, action_dim: int = 3, history_horizon: int = 20, lift_dim: int = 32) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_horizon = history_horizon
        self.lift_dim = lift_dim
        self.z_dim = state_dim + lift_dim
        self.A = nn.Parameter(torch.empty(self.z_dim, self.z_dim))
        self.B = nn.Parameter(torch.empty(action_dim, self.z_dim))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def _joint_tokens(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        validate_batch(batch, self.history_horizon, self.state_dim, self.action_dim)
        states = batch["states_history"]
        return torch.cat([states[:, :-1], batch["actions_history"]], dim=-1), states[:, -1]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, : self.state_dim]

    def advance(self, z: torch.Tensor, u: torch.Tensor, _history: list[torch.Tensor]) -> torch.Tensor:
        return z @ self.A + u @ self.B


class StrictMLPKoopman(_StrictFamilyKoopman):
    """M1: causal raw-history MLP lift without a temporal encoder."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        token_dim = self.state_dim + self.action_dim
        history_tokens = self.history_horizon - 1
        self.mlp_lift = nn.Sequential(
            nn.Linear(history_tokens * token_dim, 210),
            nn.GELU(),
            nn.Linear(210, self.lift_dim),
        )
        self.last_mlp_input: torch.Tensor | None = None

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, current_state = self._joint_tokens(batch)
        self.last_mlp_input = tokens.flatten(1)
        psi = self.mlp_lift(self.last_mlp_input)
        return torch.cat([current_state, psi], dim=-1)


class StrictLSTMKoopman(_StrictFamilyKoopman):
    """M2: all LSTM history outputs are flattened before the lift."""

    def __init__(self, hidden_dim: int = 16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.state_dim + self.action_dim, hidden_dim, num_layers=1, batch_first=True)
        self.history_lift = nn.Linear((self.history_horizon - 1) * hidden_dim, self.lift_dim)
        self.last_lstm_sequence: torch.Tensor | None = None
        self.last_history_lift_input: torch.Tensor | None = None

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, current_state = self._joint_tokens(batch)
        self.last_lstm_sequence, _ = self.lstm(tokens)
        self.last_history_lift_input = self.last_lstm_sequence.flatten(1)
        psi = self.history_lift(self.last_history_lift_input)
        return torch.cat([current_state, psi], dim=-1)


class StrictHaKANKoopman(_StrictFamilyKoopman):
    """M3: shared token adapter plus strict HaKAN lifting only."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.token_adapter = nn.Linear(self.state_dim + self.action_dim, 16)
        self.hakan_backbone = HaKANBackbone(
            channels=16,
            context_length=self.history_horizon - 1,
            patch_len=5,
            stride=2,
            d_model=32,
            depth=5,
            normalization_mode="no_revin",
        )
        self.lift = nn.Linear(self.hakan_backbone.feature_dim, self.lift_dim)
        self.last_joint_tokens: torch.Tensor | None = None
        self.last_hakan_input: torch.Tensor | None = None

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_joint_tokens, current_state = self._joint_tokens(batch)
        self.last_hakan_input = self.token_adapter(self.last_joint_tokens)
        psi = self.lift(self.hakan_backbone(self.last_hakan_input))
        return torch.cat([current_state, psi], dim=-1)


__all__ = ["StrictMLPKoopman", "StrictLSTMKoopman", "StrictHaKANKoopman"]
