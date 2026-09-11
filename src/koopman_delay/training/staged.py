"""Isolated train/validation-only staged H20 LSTM-HaKAN-AFT runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


import numpy as np
import torch
from torch.utils.data import DataLoader

from koopman_delay.models.dlm import DLMKoopman, LSTMHaKANKoopman
from koopman_delay.training import StagedResidualProtocol, UnifiedTrainer
from koopman_delay.data.three_tank import (
    DictDataset,
    _load_split_windows,
    _loss_metrics,
    _move,
    _normalize,
    _torch_rows,
    load_manifest_without_confirmation,
)


@dataclass(frozen=True)
class RunConfig:
    smoke: bool
    rollout_horizon: int = 20
    seed: int = 6
    stage_a_epochs: int = 200
    stage_c_epochs: int = 10
    stage_d_epochs: int = 20
    patience_limit: int = 0
    stage_a_patience: int = 0
    stage_c_patience: int = 0
    stage_d_patience: int = 0
    batch_size: int = 64
    current_state_conditioning: bool = False
    spectral_radius_weight: float = 0.0
    spectral_radius_max: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.spectral_radius_weight) or self.spectral_radius_weight < 0:
            raise ValueError("spectral_radius_weight must be finite and nonnegative")
        if not np.isfinite(self.spectral_radius_max) or self.spectral_radius_max <= 0:
            raise ValueError("spectral_radius_max must be finite and positive")

    @classmethod
    def create(cls, smoke: bool) -> "RunConfig":
        return cls(smoke=smoke, stage_a_epochs=1 if smoke else 200, stage_c_epochs=1 if smoke else 10, stage_d_epochs=2 if smoke else 20)

    def patience_for(self, stage: str) -> int:
        return {"A": self.stage_a_patience, "C": self.stage_c_patience, "D": self.stage_d_patience}[stage]


@dataclass
class StageResult:
    best_epoch: int = -1
    best_val: float = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    last_epoch: int = -1
    last_val: float = float("inf")
    last_state: dict[str, torch.Tensor] | None = None


@dataclass
class StageSelection:
    stage: str
    epoch: int
    val: float
    model_state: dict[str, torch.Tensor]


@dataclass
class RunState:
    config: RunConfig
    stage: str = "A"
    stage_epoch: int = 0
    epoch: int = 0
    best_epoch: int = -1
    best_val: float = float("inf")
    patience: int = 0
    stages: dict[str, StageResult] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, config: RunConfig) -> "RunState":
        return cls(config=config)


@dataclass
class RestoredCheckpoint:
    state: RunState
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None


def _clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _hash(params) -> str:
    return hashlib.sha256(b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in params)).hexdigest()


def spectral_radius_penalty(matrix, radius_max=1.0):
    radius = torch.linalg.eigvals(matrix).abs().max()
    return torch.relu(radius - radius_max).square(), radius


def _epoch(model, optimizer, loader, val, device, spectral_radius_weight=0.0, spectral_radius_max=1.0) -> dict[str, float]:
    model.train()
    trainer = UnifiedTrainer(model)
    train_losses = []
    spectral_logs = []
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad()
        components = trainer.loss_components(batch)
        loss = components["total"]
        if spectral_radius_weight:
            if (model.state_dim, model.action_dim) != (3, 0):
                raise ValueError("Spectral regularization is restricted to Lorenz D3/A0")
            penalty, radius = spectral_radius_penalty(model.A, spectral_radius_max)
            loss = loss + spectral_radius_weight * penalty
            spectral_logs.append([float(components["one_step"].detach()), float(components["rollout"].detach()),
                                  float(penalty.detach()), float(radius.detach())])
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))
    validation = _loss_metrics(model, val, device)
    metrics = {
        "train_total_loss": float(np.mean(train_losses)),
        "val_one_step_loss": validation["val_one_step_loss"],
        "val_rollout_loss": validation["val_rollout_loss"],
    }
    if spectral_logs:
        one, rollout, penalty, radius = np.mean(spectral_logs, axis=0)
        with torch.no_grad():
            _, final_radius = spectral_radius_penalty(model.A, spectral_radius_max)
        metrics.update(train_one_step_loss=float(one), train_rollout_loss=float(rollout),
                       train_spectral_penalty=float(penalty), train_spectral_weighted_loss=float(spectral_radius_weight * penalty),
                       train_spectral_radius_mean=float(radius), spectral_radius_last=float(final_radius),
                       spectral_A_trainable=bool(model.A.requires_grad))
    return metrics


def build_models(device: torch.device, current_state_conditioning: bool = False) -> tuple[LSTMHaKANKoopman, DLMKoopman]:
    return (
        LSTMHaKANKoopman(state_dim=9, action_dim=3, history_horizon=20, current_state_conditioning=current_state_conditioning).to(device),
        DLMKoopman(state_dim=9, action_dim=3, history_horizon=20, aft_context_length=20, current_state_conditioning=current_state_conditioning).to(device),
    )


def make_protocol(base, aft) -> StagedResidualProtocol:
    return StagedResidualProtocol(base, aft, "aft", 1e-3)


def initialize_aft_from_baseline(base, aft) -> None:
    expected = {key for key in aft.state_dict() if not key.startswith("aft.")}
    if set(base.state_dict()) != expected:
        raise ValueError("baseline/AFT encoder architecture mismatch")
    make_protocol(base, aft).stage_b_load_zero_residual()


def make_stage_optimizer(protocol: StagedResidualProtocol, stage: str) -> torch.optim.Optimizer:
    if stage == "A":
        return torch.optim.Adam(protocol.baseline.parameters(), lr=1e-4)
    if stage == "C":
        return protocol.stage_c_freeze_backbone()["optimizer"]
    if stage == "D":
        return protocol.stage_d_joint_finetune()["optimizer"]
    raise ValueError(f"unsupported training stage: {stage}")


def parameter_groups(base, aft) -> dict[str, set[int]]:
    protocol = make_protocol(base, aft)
    backbone = {id(parameter) for parameter in protocol.get_backbone_parameters()}
    residual = {id(parameter) for parameter in protocol.get_residual_parameters()}
    return {"backbone": backbone, "aft": residual, "overlap": backbone & residual}


def make_probe_batch(device: torch.device, rollout_horizon: int) -> dict[str, torch.Tensor]:
    current_action = torch.randn(1, 3, device=device)
    future_actions = torch.randn(1, rollout_horizon, 3, device=device)
    future_actions[:, 0] = current_action
    return {
        "states_history": torch.randn(1, 20, 9, device=device),
        "actions_history": torch.randn(1, 19, 3, device=device),
        "current_action": current_action,
        "target_state": torch.randn(1, 9, device=device),
        "future_actions": future_actions,
        "future_states": torch.randn(1, rollout_horizon, 9, device=device),
    }


def alpha_zero_prediction_check(base, aft, probe) -> dict[str, float | bool]:
    with torch.no_grad():
        baseline = base.latent_open_loop_rollout(probe, probe["future_actions"])
        residual = aft.latent_open_loop_rollout(probe, probe["future_actions"])
        error = (baseline - residual).abs()
    return {"equal": bool(float(error.max()) == 0.0), "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean())}


def record_stage_result(
    state: RunState,
    stage: str,
    epoch: int,
    score: float | dict[str, float],
    model_state: dict[str, torch.Tensor],
    learning_rates: list[float] | None = None,
) -> None:
    metrics = {"val_rollout_loss": float(score)} if isinstance(score, float) else score
    rollout_score = float(metrics["val_rollout_loss"])
    result = state.stages.setdefault(stage, StageResult())
    snapshot = _clone_state_dict(model_state)
    result.last_epoch, result.last_val, result.last_state = epoch, rollout_score, snapshot
    if rollout_score < result.best_val:
        result.best_epoch, result.best_val, result.best_state = epoch, rollout_score, _clone_state_dict(model_state)
        state.best_epoch, state.best_val, state.patience = epoch, rollout_score, 0
    else:
        state.patience += 1
    state.history.append({"stage": stage, "epoch": epoch, **metrics, "learning_rates": list(learning_rates or [])})


def stage_patience_exhausted(state: RunState, stage: str) -> bool:
    limit = state.config.patience_for(stage)
    return limit > 0 and state.patience >= limit


def resume_config_is_compatible(saved: RunConfig, requested: RunConfig, resume_stage: str) -> bool:
    if (saved.spectral_radius_weight, saved.spectral_radius_max) != (requested.spectral_radius_weight, requested.spectral_radius_max):
        return False
    if (saved.smoke, saved.rollout_horizon, saved.seed) != (requested.smoke, requested.rollout_horizon, requested.seed):
        return False
    if (saved.batch_size, saved.current_state_conditioning) != (requested.batch_size, requested.current_state_conditioning):
        return False
    completed = {"A": (), "C": ("A",), "D": ("A", "C"), "E": ("A", "C", "D")}[resume_stage]
    for stage in completed:
        if (
            getattr(saved, f"stage_{stage.lower()}_epochs"),
            saved.patience_for(stage),
        ) != (
            getattr(requested, f"stage_{stage.lower()}_epochs"),
            requested.patience_for(stage),
        ):
            return False
    return True


def select_stage_e(state: RunState) -> StageSelection:
    candidates = [
        StageSelection(stage, result.best_epoch, result.best_val, result.best_state)
        for stage, result in state.stages.items()
        if result.best_state is not None
    ]
    if not candidates:
        raise ValueError("stage E requires at least one completed candidate stage")
    return min(candidates, key=lambda candidate: candidate.val)


def _serialize_state(state: RunState) -> dict[str, Any]:
    stages = {
        stage: {
            "best_epoch": result.best_epoch,
            "best_val": result.best_val,
            "best_state": result.best_state,
            "last_epoch": result.last_epoch,
            "last_val": result.last_val,
            "last_state": result.last_state,
        }
        for stage, result in state.stages.items()
    }
    return {"config": asdict(state.config), "stage": state.stage, "stage_epoch": state.stage_epoch, "epoch": state.epoch, "best_epoch": state.best_epoch, "best_val": state.best_val, "patience": state.patience, "stages": stages, "history": state.history, "diagnostics": state.diagnostics}


def _deserialize_state(payload: dict[str, Any]) -> RunState:
    state = RunState.new(RunConfig(**payload["config"]))
    state.stage, state.stage_epoch, state.epoch = payload["stage"], payload["stage_epoch"], payload["epoch"]
    state.best_epoch, state.best_val, state.patience = payload["best_epoch"], payload["best_val"], payload["patience"]
    state.history = payload["history"]
    state.diagnostics = payload.get("diagnostics", {})
    state.stages = {stage: StageResult(**result) for stage, result in payload["stages"].items()}
    return state


def _optimizer_stage(optimizer: torch.optim.Optimizer, fallback: str) -> str:
    return "D" if len(optimizer.param_groups) == 2 else fallback


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def normalized_rng_state(value: torch.Tensor) -> torch.Tensor:
    """torch.set_rng_state requires a CPU ByteTensor after CUDA checkpoint loading."""
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def normalized_cuda_rng_states(values: list[torch.Tensor]) -> list[torch.Tensor]:
    return [normalized_rng_state(value) for value in values]


def write_stage_artifacts(output_dir: Path, state: RunState, stage: str) -> None:
    result = state.stages[stage]
    if result.best_state is None or result.last_state is None:
        raise ValueError(f"stage {stage} has no best and last checkpoints")
    stem = f"stage_{stage.lower()}"
    _atomic_torch_save(result.best_state, output_dir / f"{stem}_best.pt")
    _atomic_torch_save(result.last_state, output_dir / f"{stem}_last.pt")


def restore_stage_best(state: RunState, stage: str, model) -> None:
    result = state.stages.get(stage)
    if result is None or result.best_state is None:
        raise ValueError(f"stage {stage} has no best checkpoint to restore")
    model.load_state_dict(result.best_state, strict=True)


def save_checkpoint(path: Path, state: RunState, base, aft, optimizer, scheduler=None) -> None:
    payload = {
        "format_version": 1,
        "run_state": _serialize_state(state),
        "model_state": _clone_state_dict(aft.state_dict()),
        "baseline_model_state": _clone_state_dict(base.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_stage": _optimizer_stage(optimizer, state.stage),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scheduler_class": None if scheduler is None else scheduler.__class__.__name__,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "parameter_groups": {key: sorted(values) for key, values in parameter_groups(base, aft).items()},
    }
    _atomic_torch_save(payload, path)


def load_checkpoint(path: Path, base, aft, device: torch.device) -> RestoredCheckpoint:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {path}")
    state = _deserialize_state(payload["run_state"])
    if any(model.current_state_conditioning != state.config.current_state_conditioning for model in (base, aft)):
        raise ValueError("checkpoint encoder architecture mismatch")
    base.load_state_dict(payload["baseline_model_state"], strict=True)
    aft.load_state_dict(payload["model_state"], strict=True)
    protocol = make_protocol(base, aft)
    optimizer = make_stage_optimizer(protocol, payload["optimizer_stage"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler = None
    if payload["scheduler_state"] is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scheduler.load_state_dict(payload["scheduler_state"])
    torch.set_rng_state(normalized_rng_state(payload["rng_state"]))
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    if payload["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(normalized_cuda_rng_states(payload["cuda_rng_state"]))
    return RestoredCheckpoint(state, optimizer, scheduler)


def validate_output_directory(output_dir: Path) -> None:
    if any(part.lower() == "joint-h5" for part in Path(output_dir).parts):
        raise ValueError("refusing to write staged H20 artifacts into protected Joint-H5 output")


def load_training_manifest(data_dir: Path) -> dict:
    manifest = load_manifest_without_confirmation(Path(data_dir))
    if manifest.get("independent_confirmation_access") not in (None, "locked_not_loaded"):
        raise ValueError("independent confirmation must remain locked")
    if manifest.get("formal_public_access") not in (None, "locked_not_loaded"):
        raise ValueError("formal public evaluation must remain locked")
    return manifest


def _load_data(data_dir: Path, config: RunConfig, device: torch.device):
    manifest = load_training_manifest(data_dir)
    limit = 1 if config.smoke else None
    train = _torch_rows(_normalize(_load_split_windows(data_dir, manifest, "train", 20, config.rollout_horizon, max_trajectories=limit), manifest))
    val = _torch_rows(_normalize(_load_split_windows(data_dir, manifest, "val", 20, config.rollout_horizon, max_trajectories=limit), manifest))
    loader = DataLoader(DictDataset(train), batch_size=4 if config.smoke else config.batch_size, shuffle=True)
    return loader, val, device, manifest


def _write_evidence(output_dir: Path, state: RunState, base, aft) -> dict:
    selected = select_stage_e(state)
    groups = parameter_groups(base, aft)
    evidence = {
        "rollout_horizon": state.config.rollout_horizon,
        "smoke": state.config.smoke,
        "batch_size": state.config.batch_size,
        "current_state_conditioning": state.config.current_state_conditioning,
        "encoder_conditioning": "H_bar = H + W_c x_k (broadcast before projection); W_c has no bias" if state.config.current_state_conditioning else "none",
        "selection_metric": f"val_rollout_{state.config.rollout_horizon}_loss",
        "stage_best_val_rollout_loss": {stage: result.best_val for stage, result in state.stages.items()},
        "stage_last_val_rollout_loss": {stage: result.last_val for stage, result in state.stages.items()},
        f"stage_best_val_rollout_{state.config.rollout_horizon}_loss": {stage: result.best_val for stage, result in state.stages.items()},
        f"stage_last_val_rollout_{state.config.rollout_horizon}_loss": {stage: result.last_val for stage, result in state.stages.items()},
        "stage_b_source": state.diagnostics.get("stage_b_source"),
        "stage_d_source": state.diagnostics.get("stage_d_source"),
        "selected_stage": selected.stage,
        "selected_best_epoch": selected.epoch,
        "best_val_rollout_loss": selected.val,
        f"best_val_rollout_{state.config.rollout_horizon}_loss": selected.val,
        "alpha_zero_prediction_equal": state.diagnostics.get("alpha_zero_prediction_equal", False),
        "alpha_zero_max_abs_error": state.diagnostics.get("alpha_zero_max_abs_error", float("nan")),
        "alpha_zero_mean_abs_error": state.diagnostics.get("alpha_zero_mean_abs_error", float("nan")),
        "frozen_backbone_sha256_before": state.diagnostics.get("frozen_backbone_sha256_before"),
        "frozen_backbone_sha256_after": state.diagnostics.get("frozen_backbone_sha256_after"),
        "frozen_backbone_unchanged": state.diagnostics.get("frozen_backbone_unchanged"),
        "parameter_groups": {"backbone": len(groups["backbone"]), "aft": len(groups["aft"]), "overlap": len(groups["overlap"])},
        "observed_context_lengths": aft.aft_context_lengths,
        "independent_confirmation_unlock": False,
        "formal_training_unlock": True,
        "independent_confirmation_access": "locked_not_loaded",
        "formal_public_access": "locked_not_loaded",
    }
    if state.config.spectral_radius_weight:
        with torch.no_grad():
            selected_penalty, selected_radius = spectral_radius_penalty(selected.model_state["A"], state.config.spectral_radius_max)
        evidence.update(spectral_radius_weight=state.config.spectral_radius_weight,
                        spectral_radius_max=state.config.spectral_radius_max,
                        selected_spectral_radius=float(selected_radius), selected_spectral_penalty=float(selected_penalty))
    (output_dir / "parameter_manifest.json").write_text(json.dumps([{"name": name, "shape": list(parameter.shape), "sha256": hashlib.sha256(parameter.detach().cpu().numpy().tobytes()).hexdigest(), "group": "aft" if name.startswith("aft.") else "backbone"} for name, parameter in aft.named_parameters()], indent=2) + "\n")
    (output_dir / "staged_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def run(
    data_dir: Path,
    output_dir: Path,
    smoke: bool = False,
    device: str = "cuda",
    resume: Path | None = None,
    config: RunConfig | None = None,
    stop_after_epoch: int | None = None,
    stop_after_stage: str | None = None,
    model_factory=None,
    data_factory=None,
):
    validate_output_directory(output_dir)
    config = config or RunConfig.create(smoke)
    model_factory = model_factory or build_models
    data_factory = data_factory or _load_data
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, val, _, _ = data_factory(data_dir, config, dev)
    if resume is None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        base, aft = model_factory(dev, config.current_state_conditioning)
        state = RunState.new(config)
        optimizer = make_stage_optimizer(make_protocol(base, aft), "A")
        scheduler = None
    else:
        base, aft = model_factory(dev, config.current_state_conditioning)
        restored = load_checkpoint(resume, base, aft, dev)
        state, optimizer, scheduler = restored.state, restored.optimizer, restored.scheduler
        if not resume_config_is_compatible(state.config, config, state.stage):
            raise ValueError("resume configuration mismatch")
        state.config = config
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    checkpoint = output_dir / "training_state.pt"
    spectral_log = output_dir / "training_log.jsonl"
    last_logged_epoch = 0
    if config.spectral_radius_weight and spectral_log.exists():
        for line in spectral_log.read_text(encoding="utf-8").splitlines():
            try:
                last_logged_epoch = max(last_logged_epoch, json.loads(line)["epoch"])
            except json.JSONDecodeError:
                raise ValueError("Incomplete spectral training log; preserve it before resuming")
    probe = _move({key: value[:1] for key, value in val.items()}, dev)
    if resume is not None and state.stage == "E" and (output_dir / "staged_evidence.json").exists():
        return json.loads((output_dir / "staged_evidence.json").read_text())
    limits = {"A": state.config.stage_a_epochs, "C": state.config.stage_c_epochs, "D": state.config.stage_d_epochs}
    while state.stage in limits:
        stage = state.stage
        if stage == "C" and state.stage_epoch == 0:
            if "stage_b_source" not in state.diagnostics:
                initialize_aft_from_baseline(base, aft)
                state.diagnostics["stage_b_source"] = "A_best"
                alpha = alpha_zero_prediction_check(base, aft, probe)
                state.diagnostics.update({"alpha_zero_prediction_equal": alpha["equal"], "alpha_zero_max_abs_error": alpha["max_abs_error"], "alpha_zero_mean_abs_error": alpha["mean_abs_error"]})
                protocol = make_protocol(base, aft)
                optimizer = make_stage_optimizer(protocol, "C")
                state.diagnostics["frozen_backbone_sha256_before"] = _hash(protocol.get_backbone_parameters())
        model = base if stage == "A" else aft
        while state.stage_epoch < limits[stage]:
            if config.spectral_radius_weight:
                metrics = _epoch(model, optimizer, loader, val, dev, config.spectral_radius_weight, config.spectral_radius_max)
            else:
                metrics = _epoch(model, optimizer, loader, val, dev)
            record_stage_result(
                state,
                stage,
                state.stage_epoch,
                metrics,
                aft.state_dict() if stage != "A" else base.state_dict(),
                [float(group["lr"]) for group in optimizer.param_groups],
            )
            state.stage_epoch += 1
            state.epoch += 1
            save_checkpoint(checkpoint, state, base, aft, optimizer, scheduler)
            if config.spectral_radius_weight and state.epoch > last_logged_epoch:
                row = {"stage": stage, "epoch": state.epoch, "stage_epoch": state.stage_epoch, **metrics,
                       "learning_rates": [float(group["lr"]) for group in optimizer.param_groups]}
                with spectral_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                print(json.dumps(row), flush=True)
                last_logged_epoch = state.epoch
            if stop_after_epoch is not None and state.epoch >= stop_after_epoch:
                return {"interrupted": True, "stage": state.stage, "epoch": state.epoch, "training_state": str(checkpoint)}
            if stage_patience_exhausted(state, stage):
                break
        if stage == "A":
            write_stage_artifacts(output_dir, state, "A")
            restore_stage_best(state, "A", base)
            _atomic_torch_save(state.stages["A"].best_state, output_dir / "baseline_best.pt")
            initialize_aft_from_baseline(base, aft)
            state.diagnostics["stage_b_source"] = "A_best"
            alpha = alpha_zero_prediction_check(base, aft, probe)
            state.diagnostics.update({"alpha_zero_prediction_equal": alpha["equal"], "alpha_zero_max_abs_error": alpha["max_abs_error"], "alpha_zero_mean_abs_error": alpha["mean_abs_error"]})
            protocol = make_protocol(base, aft)
            optimizer = make_stage_optimizer(protocol, "C")
            state.diagnostics["frozen_backbone_sha256_before"] = _hash(protocol.get_backbone_parameters())
            state.stage, state.stage_epoch, state.patience = "C", 0, 0
        elif stage == "C":
            write_stage_artifacts(output_dir, state, "C")
            restore_stage_best(state, "C", aft)
            state.diagnostics["frozen_backbone_sha256_after"] = _hash(make_protocol(base, aft).get_backbone_parameters())
            state.diagnostics["frozen_backbone_unchanged"] = state.diagnostics["frozen_backbone_sha256_before"] == state.diagnostics["frozen_backbone_sha256_after"]
            state.diagnostics["stage_d_source"] = "C_best"
            optimizer = make_stage_optimizer(make_protocol(base, aft), "D")
            state.stage, state.stage_epoch, state.patience = "D", 0, 0
        else:
            write_stage_artifacts(output_dir, state, "D")
            state.stage, state.stage_epoch = "E", 0
        save_checkpoint(checkpoint, state, base, aft, optimizer, scheduler)
        if stop_after_stage == stage:
            return {"interrupted": True, "stage": state.stage, "stage_epoch": state.stage_epoch, "epoch": state.epoch, "training_state": str(checkpoint)}
    selected = select_stage_e(state)
    for stage in ("A", "C", "D"):
        write_stage_artifacts(output_dir, state, stage)
    if selected.stage != "A":
        aft.load_state_dict(selected.model_state, strict=True)
    _atomic_torch_save(selected.model_state, output_dir / "best_model.pt")
    return _write_evidence(output_dir, state, base, aft)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path, help="path to a training_state.pt checkpoint")
    parser.add_argument("--stage-a-epochs", type=int)
    parser.add_argument("--stage-c-epochs", type=int)
    parser.add_argument("--stage-d-epochs", type=int)
    parser.add_argument("--stage-a-patience", type=int)
    parser.add_argument("--stage-c-patience", type=int)
    parser.add_argument("--stage-d-patience", type=int)
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--stop-after-stage", choices=("A", "C", "D"))
    return parser.parse_args(argv)


def config_from_args(args) -> RunConfig:
    defaults = RunConfig.create(args.smoke)
    return RunConfig(
        smoke=args.smoke,
        rollout_horizon=defaults.rollout_horizon if args.rollout_horizon is None else args.rollout_horizon,
        seed=defaults.seed,
        stage_a_epochs=defaults.stage_a_epochs if args.stage_a_epochs is None else args.stage_a_epochs,
        stage_c_epochs=defaults.stage_c_epochs if args.stage_c_epochs is None else args.stage_c_epochs,
        stage_d_epochs=defaults.stage_d_epochs if args.stage_d_epochs is None else args.stage_d_epochs,
        patience_limit=defaults.patience_limit,
        stage_a_patience=0 if args.stage_a_patience is None else args.stage_a_patience,
        stage_c_patience=0 if args.stage_c_patience is None else args.stage_c_patience,
        stage_d_patience=0 if args.stage_d_patience is None else args.stage_d_patience,
    )


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir, args.smoke, args.device, args.resume, config=config_from_args(args), stop_after_stage=args.stop_after_stage), indent=2))


if __name__ == "__main__":
    main()
