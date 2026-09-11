"""Explicit current-state variants; frozen strict-family classes stay unchanged."""
import torch
from torch import nn

from .strict_family_baselines import StrictMLPKoopman, StrictHaKANKoopman, StrictLSTMKoopman


class CurrentStateLSTMKoopman(StrictLSTMKoopman):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_state_projection = nn.Linear(self.state_dim, self.hidden_dim, bias=False)

    def encode(self, batch):
        tokens, current = self._joint_tokens(batch)
        h0 = self.current_state_projection(current).unsqueeze(0)
        self.last_lstm_sequence, _ = self.lstm(tokens, (h0, torch.zeros_like(h0)))
        self.last_history_lift_input = self.last_lstm_sequence.flatten(1)
        return torch.cat([current, self.history_lift(self.last_history_lift_input)], dim=-1)


class CurrentStateMLPKoopman(StrictMLPKoopman):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_state_projection = nn.Linear(self.state_dim, 210, bias=False)

    def encode(self, batch):
        tokens, current = self._joint_tokens(batch)
        self.last_mlp_input = tokens.flatten(1)
        hidden = self.mlp_lift[0](self.last_mlp_input) + self.current_state_projection(current)
        psi = self.mlp_lift[2](self.mlp_lift[1](hidden))
        return torch.cat([current, psi], dim=-1)


class CurrentStateHaKANKoopman(StrictHaKANKoopman):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_state_projection = nn.Linear(self.state_dim, 16, bias=False)

    def encode(self, batch):
        self.last_joint_tokens, current = self._joint_tokens(batch)
        self.last_hakan_input = self.token_adapter(self.last_joint_tokens) + self.current_state_projection(current).unsqueeze(1)
        psi = self.lift(self.hakan_backbone(self.last_hakan_input))
        return torch.cat([current, psi], dim=-1)
