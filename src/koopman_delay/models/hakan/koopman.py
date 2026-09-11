"""HaKAN controlled Koopman model used by the frozen benchmarks."""
from __future__ import annotations

import torch
from torch import nn

from ..common import ControlledKoopmanModel, validate_batch
from .backbone import HaKANBackbone


class HaKANKoopman(ControlledKoopmanModel):
    """Direct HaKAN history encoder with linear latent Koopman dynamics.

    Attribute names and parameter names are preserved from the accepted model
    implementation so frozen checkpoints remain tied to the same state-dict
    contract. Historical standalone AFT/MLP-residual subclasses were removed;
    the active AFT model is ``DLMKoopman`` in ``models/dlm.py``.
    """

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 1,
        history_horizon: int = 40,
        lift_dim: int = 32,
        normalization_mode: str = "current_revin",
        **backbone,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_horizon = history_horizon
        self.normalization_mode = normalization_mode
        self.backbone = HaKANBackbone(
            channels=state_dim + action_dim,
            context_length=history_horizon - 1,
            normalization_mode=normalization_mode,
            **backbone,
        )
        self.lift = nn.Linear(
            self.backbone.feature_dim + self.backbone.scale_feature_dim,
            lift_dim,
        )
        self.z_dim = state_dim + lift_dim
        self.A = nn.Parameter(torch.empty(self.z_dim, self.z_dim))
        self.B = nn.Parameter(torch.empty(action_dim, self.z_dim))
        self.enable_runtime_diagnostics = False
        self.one_step_aft_context_length = None
        self.rollout_aft_context_lengths: list[int] = []
        self.aft_correction_norms: list[float] = []
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        validate_batch(
            batch,
            self.history_horizon,
            self.state_dim,
            self.action_dim,
        )
        states = batch["states_history"]
        actions = batch["actions_history"]
        tokens = torch.cat([states[:, :-1], actions], dim=-1)
        features = self.backbone(tokens)
        if self.normalization_mode == "revin_stats":
            features = torch.cat(
                [features, self.backbone.last_scale_features],
                dim=-1,
            )
        return torch.cat([states[:, -1], self.lift(features)], dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, : self.state_dim]

    def advance(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        _latent_history: list[torch.Tensor],
    ) -> torch.Tensor:
        return z @ self.A + u @ self.B

    def _reset_runtime_diagnostics(self) -> None:
        self.one_step_aft_context_length = None
        self.rollout_aft_context_lengths = []
        self.aft_correction_norms = []

    def predict_one_step(
        self,
        batch: dict[str, torch.Tensor],
        current_action: torch.Tensor,
    ) -> torch.Tensor:
        self._reset_runtime_diagnostics()
        result = super().predict_one_step(batch, current_action)
        if self.rollout_aft_context_lengths:
            self.one_step_aft_context_length = self.rollout_aft_context_lengths[0]
        return result

    def latent_open_loop_rollout(
        self,
        batch: dict[str, torch.Tensor],
        future_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._reset_runtime_diagnostics()
        return super().latent_open_loop_rollout(batch, future_actions)


__all__ = ["HaKANKoopman"]
