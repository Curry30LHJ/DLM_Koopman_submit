"""Single-stage current-state baselines with exact checkpoint continuation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from koopman_delay.models.current_state_baselines import CurrentStateMLPKoopman, CurrentStateHaKANKoopman, CurrentStateLSTMKoopman
from koopman_delay.training import staged as shared
from koopman_delay.training import lorenz as lorenz


@dataclass(frozen=True)
class Config:
    system: str
    model: str
    seed: int
    max_epochs: int = 800
    patience: int = 50
    rollout_horizon: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-4


MODELS = {"mlp_koopman": CurrentStateMLPKoopman, "lstm_koopman": CurrentStateLSTMKoopman, "hakan_koopman": CurrentStateHaKANKoopman}
FUSION = {"mlp_koopman": "first Linear + Wc*x before GELU", "lstm_koopman": "h0=Wc*x; c0=0; unchanged 19 history tokens", "hakan_koopman": "token adapter + broadcast Wc*x before HaKAN backbone"}


def build_model(config, device):
    dims = {"three_tank": (9, 3), "lorenz": (3, 0)}
    nx, nu = dims[config.system]
    return MODELS[config.model](state_dim=nx, action_dim=nu, history_horizon=20, lift_dim=32).to(device)


def load_data(data_dir, config, device):
    data_config = shared.RunConfig(smoke=False, seed=config.seed, rollout_horizon=config.rollout_horizon, batch_size=config.batch_size)
    factory = lorenz.load_data if config.system == "lorenz" else shared._load_data
    return factory(data_dir, data_config, device)


def run(data_dir, output_dir, config, device="cuda", resume=None, *, data_factory=load_data, stop_after=None):
    device = lorenz.native._device(device)
    output_dir = Path(output_dir)
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader, val, _, _ = data_factory(Path(data_dir), config, device)
    state = {"epoch": 0, "best_val": float("inf"), "best_epoch": None, "stale": 0, "history": [], "best_state": None}
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint.get("format") != "current-state-baseline-v1" or checkpoint["config"] != asdict(config):
            raise ValueError("Resume configuration/architecture mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        state = checkpoint["state"]
        torch.set_rng_state(shared.normalized_rng_state(checkpoint["rng_state"]))
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if checkpoint["cuda_rng_state"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(shared.normalized_cuda_rng_states(checkpoint["cuda_rng_state"]))
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    while state["epoch"] < config.max_epochs and state["stale"] < config.patience:
        started = time.perf_counter()
        metrics = shared._epoch(model, optimizer, loader, val, device)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Non-finite training/validation metric")
        state["epoch"] += 1
        if metrics["val_rollout_loss"] < state["best_val"]:
            state.update(best_val=metrics["val_rollout_loss"], best_epoch=state["epoch"], stale=0,
                         best_state=shared._clone_state_dict(model.state_dict()))
        else:
            state["stale"] += 1
        row = {"epoch": state["epoch"], **metrics, "epoch_compute_seconds": time.perf_counter() - started}
        state["history"].append(row)
        shared._atomic_torch_save({"format": "current-state-baseline-v1", "config": asdict(config), "state": state,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, output_dir / "training_state.pt")
        print(json.dumps({**row, "epoch_including_save_seconds": time.perf_counter() - started}), flush=True)
        if stop_after is not None and state["epoch"] >= stop_after:
            return state
    shared._atomic_torch_save(state["best_state"], output_dir / "best_model.pt")
    evidence = {"config": asdict(config), "epochs": state["epoch"], "best_epoch": state["best_epoch"],
                "best_val_rollout_30_loss": state["best_val"], "early_stopped": state["stale"] >= config.patience,
                "formal_public_access": "locked_not_loaded", "independent_confirmation_access": "locked_not_loaded"}
    evidence.update(fusion=FUSION[config.model], parameter_count=sum(p.numel() for p in model.parameters()),
                    added_parameters=model.current_state_projection.weight.numel(), current_state_projection_bias=False)
    (output_dir / "training_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("three_tank", "lorenz"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--seed", type=int, choices=(2, 4, 6), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, Config(args.system, args.model, args.seed), args.device, args.resume)


if __name__ == "__main__":
    main()
