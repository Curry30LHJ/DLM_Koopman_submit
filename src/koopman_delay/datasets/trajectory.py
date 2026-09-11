"""Trajectory-level split and shared history-window construction."""

from __future__ import annotations

import numpy as np


class StandardScaler:
    def fit(self, arrays: list[np.ndarray]) -> "StandardScaler":
        stacked = np.concatenate(arrays, axis=0); self.mean_ = stacked.mean(axis=0); self.std_ = np.maximum(stacked.std(axis=0), 1e-8); return self
    def transform(self, values: np.ndarray) -> np.ndarray: return (values - self.mean_) / self.std_
    def inverse_transform(self, values: np.ndarray) -> np.ndarray: return values * self.std_ + self.mean_


def split_trajectories(trajectories: list[dict], seed: int = 0) -> dict[str, list[dict]]:
    ordered = sorted(trajectories, key=lambda row: str(row["id"]))
    groups = {"train": [], "val": [], "test_benchmark": [], "test_initial_shift": [], "test_input_shift": []}
    for index, item in enumerate(ordered):
        groups[list(groups)[index % len(groups)]].append(item)
    return groups


def make_windows(states: np.ndarray, actions: np.ndarray, history_horizon: int = 40,
                 rollout_horizon: int = 1) -> dict[str, np.ndarray]:
    if states.shape[0] != actions.shape[0] + 1: raise ValueError("states must have exactly one more row than actions")
    if rollout_horizon < 1: raise ValueError("rollout_horizon must be positive")
    if actions.shape[0] < history_horizon + rollout_horizon - 1: raise ValueError("trajectory is shorter than history plus rollout horizon")
    rows = actions.shape[0] - history_horizon - rollout_horizon + 2
    target_state = np.stack([states[i + history_horizon] for i in range(rows)])
    return {
        "states_history": np.stack([states[i:i + history_horizon] for i in range(rows)]),
        "actions_history": np.stack([actions[i:i + history_horizon - 1] for i in range(rows)]),
        "current_action": np.stack([actions[i + history_horizon - 1] for i in range(rows)]),
        "target_state": target_state,
        "future_actions": np.stack([actions[i + history_horizon - 1:i + history_horizon - 1 + rollout_horizon] for i in range(rows)]),
        "future_states": np.stack([states[i + history_horizon:i + history_horizon + rollout_horizon] for i in range(rows)]),
        "targets": target_state,
    }
