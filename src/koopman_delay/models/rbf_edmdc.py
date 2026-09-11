"""RBF-EDMDc baselines for controlled open-loop prediction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RBFEDMDcConfig:
    mode: str
    n_centers: int
    kernel_width: float
    ridge_alpha: float
    random_seed: int = 6
    physical_delay_steps: int | None = None
    physical_delay_variant: str = "separator_state"


def _recycle_composition(x3: np.ndarray) -> np.ndarray:
    x_a3 = x3[:, 0]
    x_b3 = x3[:, 1]
    t3 = x3[:, 2]
    x_c3 = 1.0 - x_a3 - x_b3
    denom = np.maximum(3.5 * x_a3 + 1.0 * x_b3 + 0.5 * x_c3, 1e-8)
    return np.stack([3.5 * x_a3 / denom, x_b3 / denom, t3], axis=1)


def build_embedding(
    states_history: np.ndarray,
    mode: str,
    physical_delay_steps: int | None = None,
    physical_delay_variant: str = "separator_state",
) -> np.ndarray:
    states = np.asarray(states_history, dtype=np.float64)
    if states.ndim == 2:
        states = states[None, :, :]
    if states.ndim != 3 or states.shape[-1] < 1:
        raise ValueError("states_history must have shape [N,H,D] or [H,D] with D >= 1")
    if mode == "markov":
        return states[:, -1]
    if mode == "delay":
        return states.reshape(states.shape[0], -1)
    if mode == "physical_delay_d3":
        delay = int(physical_delay_steps or 0)
        if states.shape[-1] != 3:
            raise ValueError("physical_delay_d3 requires Delayed Lorenz state_dim == 3")
        if delay != 5:
            raise ValueError("physical_delay_d3 requires the frozen Delayed Lorenz lag of 5 observations")
        if states.shape[1] < delay + 1:
            raise ValueError("states_history is shorter than physical_delay_steps + 1")
        return np.concatenate([states[:, -1], states[:, -delay - 1, [1, 2]]], axis=1)
    if mode == "physical_delay":
        delay = int(physical_delay_steps or 0)
        if delay < 0:
            raise ValueError("physical_delay_steps must be non-negative")
        if states.shape[1] < delay + 1:
            raise ValueError("states_history is shorter than physical_delay_steps + 1")
        delayed_separator = states[:, -delay - 1, [6, 7, 8]]
        if physical_delay_variant == "separator_state":
            delayed_features = delayed_separator
        elif physical_delay_variant == "recycle_composition":
            delayed_features = _recycle_composition(delayed_separator)
        else:
            raise ValueError(f"unsupported physical_delay_variant: {physical_delay_variant}")
        return np.concatenate([states[:, -1], delayed_features], axis=1)
    if mode == "physical_delay_recycle_composition":
        return build_embedding(states, "physical_delay", physical_delay_steps, "recycle_composition")
    if mode == "full_state_delay_ablation":
        delay = int(physical_delay_steps or 0)
        if delay < 0:
            raise ValueError("physical_delay_steps must be non-negative")
        if states.shape[1] < delay + 1:
            raise ValueError("states_history is shorter than physical_delay_steps + 1")
        return np.concatenate([states[:, -1], states[:, -delay - 1]], axis=1)
    raise ValueError(f"unsupported RBF-EDMDc mode: {mode}")


def thinplate_rbf(points: np.ndarray, centers: np.ndarray, kernel_width: float) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    if kernel_width <= 0:
        raise ValueError("kernel_width must be positive")
    scaled = (x[:, None, :] - c[None, :, :]) / float(kernel_width)
    radius = np.linalg.norm(scaled, axis=2)
    out = np.zeros_like(radius)
    mask = radius > 0
    out[mask] = radius[mask] ** 2 * np.log(radius[mask])
    return out


def _ridge_solve(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    lhs = design.T @ design
    if alpha > 0:
        lhs = lhs + float(alpha) * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = design.T @ target
    return np.linalg.solve(lhs, rhs)


def _next_history(states_history: np.ndarray, next_states: np.ndarray) -> np.ndarray:
    return np.concatenate([states_history[:, 1:, :], next_states[:, None, :]], axis=1)


class RBFEDMDc:
    def __init__(self, config: RBFEDMDcConfig) -> None:
        self.config = config

    def fit(self, states_history: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> "RBFEDMDc":
        states = np.asarray(states_history, dtype=np.float64)
        acts = np.asarray(actions, dtype=np.float64)
        targets = np.asarray(next_states, dtype=np.float64)
        if acts.ndim == 1:
            acts = acts[:, None]
        if states.ndim != 3 or states.shape[-1] < 1 or targets.shape != (states.shape[0], states.shape[-1]):
            raise ValueError("fit expects states_history [N,H,D] and next_states [N,D]")
        if acts.ndim != 2 or acts.shape[0] != states.shape[0]:
            raise ValueError("fit expects actions [N,U]")
        current_embedding = build_embedding(
            states,
            self.config.mode,
            self.config.physical_delay_steps,
            self.config.physical_delay_variant,
        )
        next_embedding = build_embedding(
            _next_history(states, targets),
            self.config.mode,
            self.config.physical_delay_steps,
            self.config.physical_delay_variant,
        )
        self.centers_ = self._select_centers(current_embedding)
        phi = self.transform_embedding(current_embedding)
        phi_next = self.transform_embedding(next_embedding)
        design = np.concatenate([phi, acts], axis=1)
        transition = _ridge_solve(design, phi_next, self.config.ridge_alpha)
        self.A_ = transition[: phi.shape[1]]
        self.B_ = transition[phi.shape[1] :]
        self.C_ = _ridge_solve(phi_next, targets, self.config.ridge_alpha)
        self.feature_dim_ = phi.shape[1]
        self.history_horizon_ = states.shape[1]
        self.state_dim_ = states.shape[2]
        self.action_dim_ = acts.shape[1]
        return self

    def _select_centers(self, embeddings: np.ndarray) -> np.ndarray:
        n_centers = min(int(self.config.n_centers), embeddings.shape[0])
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_centers, random_state=self.config.random_seed, n_init=3)
            return km.fit(embeddings).cluster_centers_.astype(np.float64)
        except Exception:
            rng = np.random.default_rng(self.config.random_seed)
            idx = rng.choice(embeddings.shape[0], size=n_centers, replace=False)
            return embeddings[idx].copy()

    def transform_embedding(self, embedding: np.ndarray) -> np.ndarray:
        emb = np.asarray(embedding, dtype=np.float64)
        if emb.ndim == 1:
            emb = emb[None, :]
        rbf = thinplate_rbf(emb, self.centers_, self.config.kernel_width)
        bias = np.ones((emb.shape[0], 1), dtype=np.float64)
        return np.concatenate([bias, emb, rbf], axis=1)

    def predict_one_step(self, states_history: np.ndarray, action: np.ndarray) -> np.ndarray:
        history = np.asarray(states_history, dtype=np.float64)
        if history.ndim == 2:
            history = history[None, :, :]
        act = np.asarray(action, dtype=np.float64)
        if act.ndim == 1:
            act = act.reshape(1, -1) if history.shape[0] == 1 else act.reshape(history.shape[0], -1)
        if act.ndim != 2 or act.shape != (history.shape[0], self.action_dim_):
            raise ValueError(f"predict_one_step expects action [N,{self.action_dim_}]")
        phi = self.transform_embedding(
            build_embedding(history, self.config.mode, self.config.physical_delay_steps, self.config.physical_delay_variant)
        )
        phi_next = phi @ self.A_ + act @ self.B_
        return phi_next @ self.C_

    def rollout(
        self,
        states_history: np.ndarray,
        future_actions: np.ndarray,
        future_truth_forbidden: np.ndarray | None = None,
    ) -> np.ndarray:
        del future_truth_forbidden
        history = np.asarray(states_history, dtype=np.float64).copy()
        if history.ndim != 2 or history.shape[1] < 1:
            raise ValueError("rollout expects states_history [H,D] with D >= 1")
        actions = np.asarray(future_actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[:, None]
        preds = []
        for action in actions:
            next_state = self.predict_one_step(history, action)[0]
            preds.append(next_state)
            history = np.concatenate([history[1:], next_state[None, :]], axis=0)
        return np.asarray(preds, dtype=np.float64)

    def rollout_batch(self, states_history: np.ndarray, future_actions: np.ndarray) -> np.ndarray:
        history = np.asarray(states_history, dtype=np.float64).copy()
        actions = np.asarray(future_actions, dtype=np.float64)
        if history.ndim != 3 or history.shape[-1] < 1:
            raise ValueError("rollout_batch expects states_history [N,H,D] with D >= 1")
        if actions.ndim != 3 or actions.shape[0] != history.shape[0] or actions.shape[2] != self.action_dim_:
            raise ValueError(f"rollout_batch expects future_actions [N,T,{self.action_dim_}]")
        preds = []
        for step in range(actions.shape[1]):
            next_state = self.predict_one_step(history, actions[:, step])
            preds.append(next_state)
            history = np.concatenate([history[:, 1:, :], next_state[:, None, :]], axis=1)
        return np.stack(preds, axis=1)
