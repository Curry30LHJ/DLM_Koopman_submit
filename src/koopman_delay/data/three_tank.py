from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from koopman_delay.datasets import make_windows
from koopman_delay.training import UnifiedTrainer

REQUIRED_KEYS = (
    "states_history",
    "actions_history",
    "current_action",
    "target_state",
    "future_actions",
    "future_states",
)


class DictDataset(Dataset):
    def __init__(self, rows: dict[str, torch.Tensor]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return int(self.rows["states_history"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: self.rows[key][index] for key in REQUIRED_KEYS}


def load_manifest_without_confirmation(data_dir: Path) -> dict:
    manifest = json.loads(
        (Path(data_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    locked = {"independent_confirmation_locked", "formal_public_evaluation_locked"}
    accessible = [row for row in manifest["trajectories"] if row["split"] not in locked]
    return {
        **manifest,
        "accessible_trajectories": accessible,
        "independent_confirmation_access": "locked_not_loaded",
        "formal_public_access": "locked_not_loaded",
    }


def _normalize(rows: dict[str, np.ndarray], manifest: dict) -> dict[str, np.ndarray]:
    state_mean = np.asarray(manifest["state_mean"])
    state_std = np.asarray(manifest["state_std"])
    action_mean = np.asarray(manifest["action_mean"])
    action_std = np.asarray(manifest["action_std"])
    out = {}
    for key, value in rows.items():
        if key in ("states_history", "target_state", "future_states"):
            out[key] = (value - state_mean) / state_std
        elif key in ("actions_history", "current_action", "future_actions"):
            out[key] = (value - action_mean) / action_std
        else:
            out[key] = value
    return out


def _load_split_windows(
    data_dir: Path,
    manifest: dict,
    split: str,
    history_horizon: int,
    rollout_horizon: int,
    max_trajectories: int | None = None,
) -> dict[str, np.ndarray]:
    pieces = []
    rows = [row for row in manifest["trajectories"] if row["split"] == split]
    if max_trajectories is not None:
        rows = rows[:max_trajectories]
    for row in rows:
        payload = np.load(Path(data_dir) / split / row["file"])
        pieces.append(
            make_windows(
                payload["states"], payload["actions"], history_horizon, rollout_horizon
            )
        )
    return {
        key: np.concatenate([piece[key] for piece in pieces], axis=0)
        for key in pieces[0]
        if key in REQUIRED_KEYS
    }


def _torch_rows(rows: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor(value, dtype=torch.float32) for key, value in rows.items()
    }


def _move(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _loss_metrics(model, rows: dict[str, torch.Tensor], device) -> dict:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        components = UnifiedTrainer(model).loss_components(_move(rows, device))
    model.train(was_training)
    return {
        f"val_{key}_loss": float(value.detach().cpu())
        for key, value in components.items()
    }
