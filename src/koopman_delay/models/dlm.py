"""LSTM-fronted Koopman candidates for HaKAN comparison."""
from __future__ import annotations

import torch
from torch import nn

from .common import ControlledKoopmanModel, validate_batch
from .aft.aft_module import AttentionFreeLatentMemory
from .hakan.backbone import HaKANBackbone


class _BaseLSTMKoopman(ControlledKoopmanModel):
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        history_horizon: int = 40,
        lift_dim: int = 32,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_horizon = history_horizon
        self.lift_dim = lift_dim
        self.hidden_dim = hidden_dim
        self.z_dim = state_dim + lift_dim
        self.A = nn.Parameter(torch.empty(self.z_dim, self.z_dim))
        self.B = nn.Parameter(torch.empty(action_dim, self.z_dim))
        self.last_lstm_input = None
        self.last_lstm_sequence = None
        self.last_hakan_input = None
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, : self.state_dim]

    def advance(self, z: torch.Tensor, u: torch.Tensor, _history: list[torch.Tensor]) -> torch.Tensor:
        return z @ self.A + u @ self.B

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMHaKANKoopman(_BaseLSTMKoopman):
    """Joint [x_k,u_k] LSTM encoder followed by a no-RevIN HaKAN backbone."""

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        history_horizon: int = 40,
        lift_dim: int = 32,
        hidden_dim: int = 16,
        projection_dim: int | None = None,
        patch_len: int = 5,
        stride: int = 2,
        d_model: int = 32,
        depth: int = 5,
        current_state_conditioning: bool = False,
    ) -> None:
        super().__init__(state_dim, action_dim, history_horizon, lift_dim, hidden_dim)
        self.projection_dim = projection_dim
        self.lstm = nn.LSTM(
            state_dim + action_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        hakan_channels = projection_dim if projection_dim is not None else hidden_dim
        self.projection = nn.Linear(hidden_dim, hakan_channels) if projection_dim is not None else nn.Identity()
        self.hakan_backbone = HaKANBackbone(
            channels=hakan_channels,
            context_length=history_horizon - 1,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            depth=depth,
            normalization_mode="no_revin",
        )
        self.lift = nn.Linear(self.hakan_backbone.feature_dim, lift_dim)
        self.current_state_conditioning = current_state_conditioning
        if current_state_conditioning:
            self.current_state_projection = nn.Linear(state_dim, hidden_dim, bias=False)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        validate_batch(batch, self.history_horizon, self.state_dim, self.action_dim)
        states, actions = batch["states_history"], batch["actions_history"]
        lstm_input = torch.cat([states[:, :-1], actions], dim=-1)
        lstm_sequence, _ = self.lstm(lstm_input)
        conditioned_sequence = lstm_sequence
        if self.current_state_conditioning:
            conditioned_sequence = lstm_sequence + self.current_state_projection(states[:, -1]).unsqueeze(1)
        hakan_input = self.projection(conditioned_sequence)
        features = self.hakan_backbone(hakan_input)
        psi = self.lift(features)
        self.last_lstm_input = lstm_input
        self.last_lstm_sequence = lstm_sequence
        self.last_hakan_input = hakan_input
        return torch.cat([states[:, -1], psi], dim=-1)


class LSTMOnlyKoopman(_BaseLSTMKoopman):
    """Joint [x_k,u_k] LSTM encoder without HaKAN lifting."""

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        history_horizon: int = 40,
        lift_dim: int = 32,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__(state_dim, action_dim, history_horizon, lift_dim, hidden_dim)
        self.lstm = nn.LSTM(
            state_dim + action_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.lift = nn.Linear(hidden_dim, lift_dim)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        validate_batch(batch, self.history_horizon, self.state_dim, self.action_dim)
        states, actions = batch["states_history"], batch["actions_history"]
        lstm_input = torch.cat([states[:, :-1], actions], dim=-1)
        lstm_sequence, _ = self.lstm(lstm_input)
        psi = self.lift(lstm_sequence[:, -1])
        self.last_lstm_input = lstm_input
        self.last_lstm_sequence = lstm_sequence
        self.last_hakan_input = None
        return torch.cat([states[:, -1], psi], dim=-1)


class LSTMMLPKoopman(_BaseLSTMKoopman):
    """Joint [x_k,u_k] LSTM encoder with an MLP lift head."""

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        history_horizon: int = 40,
        lift_dim: int = 32,
        hidden_dim: int = 16,
        mlp_hidden_dim: int = 256,
    ) -> None:
        super().__init__(state_dim, action_dim, history_horizon, lift_dim, hidden_dim)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.lstm = nn.LSTM(
            state_dim + action_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.mlp_lift = nn.Sequential(
            nn.Linear(hidden_dim * (history_horizon - 1), mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, lift_dim),
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        validate_batch(batch, self.history_horizon, self.state_dim, self.action_dim)
        states, actions = batch["states_history"], batch["actions_history"]
        lstm_input = torch.cat([states[:, :-1], actions], dim=-1)
        lstm_sequence, _ = self.lstm(lstm_input)
        psi = self.mlp_lift(lstm_sequence.reshape(lstm_sequence.shape[0], -1))
        self.last_lstm_input = lstm_input
        self.last_lstm_sequence = lstm_sequence
        self.last_hakan_input = None
        return torch.cat([states[:, -1], psi], dim=-1)


class DLMKoopman(LSTMHaKANKoopman):
    """LSTM-HaKAN encoder with causal AFT residual before Koopman propagation."""

    def __init__(self, aft_context_length: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self.aft_context_length = aft_context_length
        self.aft = AttentionFreeLatentMemory(self.z_dim, aft_context_length)
        self.aft_call_count = 0
        self.aft_context_lengths: list[int] = []
        self.aft_correction_norms: list[float] = []
        self.aft_context_examples: list[torch.Tensor] = []
        self.last_aft_context = None
        self.last_aft_correction = None

    def _reset_aft_diagnostics(self) -> None:
        self.aft_call_count = 0
        self.aft_context_lengths = []
        self.aft_correction_norms = []
        self.aft_context_examples = []
        self.last_aft_context = None
        self.last_aft_correction = None

    def predict_one_step(self, batch: dict[str, torch.Tensor], current_action: torch.Tensor) -> torch.Tensor:
        self._reset_aft_diagnostics()
        return super().predict_one_step(batch, current_action)

    def latent_open_loop_rollout(
        self,
        batch: dict[str, torch.Tensor],
        future_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._reset_aft_diagnostics()
        return super().latent_open_loop_rollout(batch, future_actions)

    def advance(self, z: torch.Tensor, u: torch.Tensor, latent_history: list[torch.Tensor]) -> torch.Tensor:
        if len(latent_history) < 2:
            return super().advance(z, u, latent_history)
        context = torch.stack(latent_history[-self.aft_context_length :], dim=1)
        correction = self.aft(context)
        self.aft_call_count += 1
        self.aft_context_lengths.append(context.shape[1])
        self.aft_correction_norms.append(float(correction.detach().norm()))
        self.last_aft_context = context.detach().clone()
        self.last_aft_correction = correction.detach().clone()
        if len(self.aft_context_examples) < 3:
            self.aft_context_examples.append(context.detach().clone())
        return (z + correction) @ self.A + u @ self.B


class StateLSTMHaKANKoopman(_BaseLSTMKoopman):
    """State-only LSTM encoder followed by a no-RevIN HaKAN backbone."""

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        history_horizon: int = 40,
        lift_dim: int = 32,
        hidden_dim: int = 16,
        projection_dim: int | None = None,
        patch_len: int = 5,
        stride: int = 2,
        d_model: int = 32,
        depth: int = 5,
    ) -> None:
        super().__init__(state_dim, action_dim, history_horizon, lift_dim, hidden_dim)
        self.projection_dim = projection_dim
        self.lstm = nn.LSTM(
            state_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        hakan_channels = projection_dim if projection_dim is not None else hidden_dim
        self.projection = nn.Linear(hidden_dim, hakan_channels) if projection_dim is not None else nn.Identity()
        self.hakan_backbone = HaKANBackbone(
            channels=hakan_channels,
            context_length=history_horizon,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            depth=depth,
            normalization_mode="no_revin",
        )
        self.lift = nn.Linear(self.hakan_backbone.feature_dim, lift_dim)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        validate_batch(batch, self.history_horizon, self.state_dim, self.action_dim)
        states = batch["states_history"]
        lstm_sequence, _ = self.lstm(states)
        hakan_input = self.projection(lstm_sequence)
        features = self.hakan_backbone(hakan_input)
        psi = self.lift(features)
        self.last_lstm_input = states
        self.last_lstm_sequence = lstm_sequence
        self.last_hakan_input = hakan_input
        return torch.cat([states[:, -1], psi], dim=-1)


__all__ = ["LSTMHaKANKoopman", "DLMKoopman", "LSTMMLPKoopman", "LSTMOnlyKoopman", "StateLSTMHaKANKoopman"]
