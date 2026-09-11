"""Shared batch validation and autoregressive controlled-Koopman utilities."""

from __future__ import annotations

import torch
from torch import nn


def validate_batch(batch: dict[str, torch.Tensor], history_horizon: int, state_dim: int, action_dim: int) -> None:
    states, actions = batch["states_history"], batch["actions_history"]
    if states.ndim != 3 or states.shape[1:] != (history_horizon, state_dim):
        raise ValueError(f"states_history must be [B,{history_horizon},{state_dim}]")
    if actions.ndim != 3 or actions.shape[1:] != (history_horizon - 1, action_dim):
        raise ValueError(f"actions_history must be [B,{history_horizon - 1},{action_dim}]")
    if states.shape[0] != actions.shape[0]:
        raise ValueError("state/action batch sizes differ")


class ControlledKoopmanModel(nn.Module):
    """Base class for row-vector dynamics: z_next = z @ A + u @ B."""

    def _current_action(self, batch: dict[str, torch.Tensor], current_action: torch.Tensor | None = None) -> torch.Tensor:
        action = batch.get("current_action") if current_action is None else current_action
        if action is None:
            raise ValueError("current_action is required; actions_history ends at u[t-1]")
        if action.ndim != 2 or action.shape != (batch["states_history"].shape[0], self.action_dim):
            raise ValueError(f"current_action must be [B,{self.action_dim}]")
        return action

    def predict_one_step(self, batch: dict[str, torch.Tensor], current_action: torch.Tensor) -> torch.Tensor:
        z = self.encode(batch)
        return self.decode(self.advance(z, self._current_action(batch, current_action), [z]))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.predict_one_step(batch, self._current_action(batch))

    def latent_open_loop_rollout(self, batch: dict[str, torch.Tensor], future_actions: torch.Tensor | None = None) -> torch.Tensor:
        actions = batch.get("future_actions") if future_actions is None else future_actions
        if actions is None:
            raise ValueError("future_actions is required for rollout")
        if actions.ndim != 3 or actions.shape[0] != batch["states_history"].shape[0] or actions.shape[2] != self.action_dim:
            raise ValueError(f"future_actions must be [B,R,{self.action_dim}]")
        current = self._current_action(batch)
        if actions.shape[1] and not torch.allclose(actions[:, 0], current):
            raise ValueError("future_actions[:,0] must equal current_action")
        z = self.encode(batch)
        states, latent_history = [], [z]
        for step in range(actions.shape[1]):
            # Formal rollout keeps only the initial latent and free predicted latents in context.
            z = self.advance(z, actions[:, step], latent_history)
            latent_history.append(z)
            states.append(self.decode(z))
        return torch.stack(states, dim=1)

    def rollout(self, batch: dict[str, torch.Tensor], future_actions: torch.Tensor | None = None) -> torch.Tensor:
        return self.latent_open_loop_rollout(batch, future_actions)
