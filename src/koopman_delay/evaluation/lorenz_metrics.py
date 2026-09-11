"""Frozen metric pipeline for the delayed-Lorenz benchmark.

Model predictions are produced in train-normalized state coordinates.  This
module makes the single permitted conversion explicit: inverse-scale the
prediction once, then normalize the physical residual by the frozen training
state standard deviation.  Truth arrays supplied to this module are always
physical-space arrays.
"""

from __future__ import annotations

import numpy as np


def inverse_scale(
    prediction_normalized: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> np.ndarray:
    """Convert a normalized prediction to physical state units exactly once."""

    prediction = np.asarray(prediction_normalized, dtype=np.float64)
    mean = np.asarray(state_mean, dtype=np.float64)
    std = np.asarray(state_std, dtype=np.float64)
    return prediction * std + mean


def compute_cum_nrmse(
    prediction_physical: np.ndarray,
    truth_physical: np.ndarray,
    state_std: np.ndarray,
    horizon: int,
) -> float:
    """Compute cumulative physical-space NRMSE through ``horizon``.

    The denominator is the frozen train-split state standard deviation.  The
    mean is deliberately not subtracted from either physical array: centering
    belongs only to the model input/output coordinate conversion.
    """

    prediction = np.asarray(prediction_physical, dtype=np.float64)
    truth = np.asarray(truth_physical, dtype=np.float64)
    scale = np.maximum(np.asarray(state_std, dtype=np.float64), 1e-12)
    if horizon < 1 or prediction.shape[0] < horizon or truth.shape[0] < horizon:
        raise ValueError(
            "prediction and truth must cover the requested positive horizon"
        )
    if prediction.shape[1:] != truth.shape[1:]:
        raise ValueError("prediction and truth shapes must agree after the time axis")
    residual = (prediction[:horizon] - truth[:horizon]) / scale
    if not np.isfinite(residual).all():
        raise FloatingPointError("non-finite physical-space residual")
    return float(np.sqrt(np.mean(residual**2)))


def compute_cum_nrmse_from_normalized(
    prediction_normalized: np.ndarray,
    truth_physical: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    horizon: int,
) -> float:
    """Apply the frozen pipeline to normalized predictions and physical truth."""

    prediction_physical = inverse_scale(prediction_normalized, state_mean, state_std)
    return compute_cum_nrmse(prediction_physical, truth_physical, state_std, horizon)
