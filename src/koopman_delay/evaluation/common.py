from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from koopman_delay.models.dlm import DLMKoopman
from koopman_delay.training import baselines
from .lorenz_metrics import inverse_scale


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected(row, device):
    """Load one hash-identified neural state dict with strict architecture matching."""
    path = Path(row["checkpoint"])
    if sha(path) != row["checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint changed")
    nx, nu = (3, 0) if row["system"] == "lorenz" else (9, 3)
    if row["model"] == "dlm_koopman":
        model = DLMKoopman(
            state_dim=nx,
            action_dim=nu,
            history_horizon=20,
            aft_context_length=20,
            current_state_conditioning=True,
        )
    else:
        model = baselines.MODELS[row["model"]](
            state_dim=nx, action_dim=nu, history_horizon=20
        )
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False), strict=True
    )
    return model.to(device).eval()


def causal_windows(states, actions, origins, mean, std, action_mean, action_std):
    """Build 20-state histories and H100 futures; truth stays outside model inputs.

    Shapes are (windows, time, channels); Lorenz actions have zero channels.
    Origin k includes x[k] but never x[k+1] in the encoding history.
    """
    origins = np.asarray(origins, dtype=int)
    if np.any(origins < 19) or np.any(origins + 100 >= len(states)):
        raise ValueError("Illegal H100 forecast origins")
    history = np.stack([states[k - 19 : k + 1] for k in origins])
    past_u = np.stack([actions[k - 19 : k] for k in origins])
    future_u = np.stack([actions[k : k + 100] for k in origins])
    truth = np.stack([states[k + 1 : k + 101] for k in origins])
    inputs = {
        "states_history": (history - mean) / std,
        "actions_history": (past_u - action_mean) / action_std,
        "future_actions": (future_u - action_mean) / action_std,
    }
    inputs["current_action"] = inputs["future_actions"][:, 0]
    return inputs, truth


def predict(model, inputs, device, batch_size=128):
    """Frozen float32 neural inference path, returning normalized H100 predictions."""
    # Future truth is deliberately absent from all model inputs.
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs["states_history"]), batch_size):
            batch = {
                key: torch.as_tensor(
                    value[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                for key, value in inputs.items()
            }
            predictions.append(model.rollout(batch).cpu().numpy())
    return np.concatenate(predictions)


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectory_curves(
    prediction_physical: np.ndarray,
    truth_physical: np.ndarray,
    state_std_physical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint and cumulative physical-space NRMSE for one seed/trajectory."""
    prediction, truth = np.asarray(prediction_physical, dtype=np.float64), np.asarray(
        truth_physical, dtype=np.float64
    )
    scale = np.asarray(state_std_physical, dtype=np.float64).reshape(1, 1, -1)
    if (
        prediction.shape != truth.shape
        or prediction.ndim != 3
        or scale.shape[-1] != prediction.shape[-1]
    ):
        raise ValueError(
            "prediction/truth/std violate the G1.1 trajectory metric contract"
        )
    scaled_square = ((prediction - truth) / scale) ** 2
    endpoint = np.sqrt(scaled_square.mean(axis=(0, 2)))
    cumulative = np.sqrt(
        (
            np.cumsum(scaled_square, axis=1)
            / np.arange(1, prediction.shape[1] + 1)[None, :, None]
        ).mean(axis=(0, 2))
    )
    return endpoint, cumulative
