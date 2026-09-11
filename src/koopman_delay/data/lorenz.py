from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

STATE_DIM = 3
ACTION_DIM = 0
HISTORY_HORIZON = 20
H_TRAIN = 30
DATA_ROOT = Path(__file__).resolve().parents[3] / "artifacts/datasets/lorenz"


def assert_training_data_access(data_dir: Path) -> None:
    """Reject any evaluation split that is explicitly locked from training.

    Training callers may point at a data root or directly at a split directory.
    Walk the supplied path's ancestors so a LOCK.json remains authoritative in
    both cases, before any manifest or trajectory is opened.
    """
    root = Path(data_dir).resolve()
    for candidate in (root, *root.parents):
        lock_path = candidate / "LOCK.json"
        if not lock_path.is_file():
            continue
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid training access lock: {lock_path}") from exc
        status = str(lock.get("status", "")).upper()
        training_access = lock.get("training_access")
        if status == "LOCKED_NOT_LOADED" or training_access is False:
            raise PermissionError(
                f"locked evaluation split cannot be used for training: {candidate}"
            )


def _load_manifest_and_scaler(data_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    root = Path(data_dir)
    assert_training_data_access(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    access = manifest.get("access", {})
    if access.get("formal_public_access") != "locked_not_loaded":
        raise RuntimeError("formal-public access lock is not intact")
    if access.get("independent_confirmation_access") != "locked_not_loaded":
        raise RuntimeError("confirmation access lock is not intact")
    scaler = json.loads((root / "scaler.json").read_text(encoding="utf-8"))
    if scaler.get("fit_split") != "train":
        raise RuntimeError("scaler is not train-fitted")
    mean = np.asarray(scaler["state_mean"], dtype=np.float64)
    std = np.asarray(scaler["state_std"], dtype=np.float64)
    if mean.shape != (STATE_DIM,) or std.shape != (STATE_DIM,):
        raise ValueError("Delayed Lorenz scaler must have three state channels")
    return manifest, mean, std


def _trajectory_count(split: str) -> int:
    if split == "train":
        return 50
    if split == "val":
        return 12
    raise ValueError(f"only train/val are allowed, got {split!r}")


def _window_rows(
    split: str,
    rollout_horizon: int,
    data_dir: Path,
) -> dict[str, np.ndarray]:
    if rollout_horizon < 1:
        raise ValueError("rollout_horizon must be positive")
    _manifest, mean, std = _load_manifest_and_scaler(data_dir)
    arrays = []
    for index in range(_trajectory_count(split)):
        path = Path(data_dir) / split / f"{split}_{index:03d}.npy"
        raw = np.load(path).astype(np.float64)
        if raw.shape != (1000, STATE_DIM):
            raise ValueError(f"unexpected trajectory shape in {path}: {raw.shape}")
        arrays.append((raw - mean) / std)
    histories, targets, future_states = [], [], []
    for arr in arrays:
        last_anchor = arr.shape[0] - rollout_horizon - 1
        for anchor in range(HISTORY_HORIZON, last_anchor + 1):
            histories.append(arr[anchor - HISTORY_HORIZON + 1 : anchor + 1])
            targets.append(arr[anchor + 1])
            future_states.append(arr[anchor + 1 : anchor + 1 + rollout_horizon])
    n_rows = len(histories)
    return {
        "states_history": np.asarray(histories, dtype=np.float32),
        "actions_history": np.zeros(
            (n_rows, HISTORY_HORIZON - 1, ACTION_DIM), dtype=np.float32
        ),
        "current_action": np.zeros((n_rows, ACTION_DIM), dtype=np.float32),
        "target_state": np.asarray(targets, dtype=np.float32),
        "future_actions": np.zeros(
            (n_rows, rollout_horizon, ACTION_DIM), dtype=np.float32
        ),
        "future_states": np.asarray(future_states, dtype=np.float32),
    }


def numpy_windows(
    split: str,
    rollout_horizon: int = H_TRAIN,
    data_dir: Path = DATA_ROOT,
) -> dict[str, np.ndarray]:
    return _window_rows(split, rollout_horizon, Path(data_dir))


def load_windows(data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load only train and val after the lock guard, never evaluation splits."""
    assert_training_data_access(data_dir)
    train = numpy_windows("train", H_TRAIN, data_dir)
    val = numpy_windows("val", H_TRAIN, data_dir)
    if (
        train["states_history"].shape[0] != 47_500
        or val["states_history"].shape[0] != 11_400
    ):
        raise RuntimeError("Benchmark III requires 50*950 train and 12*950 val windows")
    return train, val


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device
