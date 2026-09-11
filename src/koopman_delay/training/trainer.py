"""Unified metrics and staged AFT/MLP training protocol; no training on import."""
from __future__ import annotations
import torch

class UnifiedTrainer:
    """Teacher-free one-step plus latent open-loop rollout loss."""
    def __init__(self, model, learning_rate: float = 1e-3, lambda_one: float = 1.0, lambda_rollout: float = 1.0):
        self.model, self.optimizer = model, torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.lambda_one, self.lambda_rollout = lambda_one, lambda_rollout
    def loss_components(self, batch):
        target = batch.get("target_state", batch.get("targets"))
        if target is None: raise ValueError("target_state is required")
        future_actions, future_states = batch.get("future_actions"), batch.get("future_states")
        if future_actions is None or future_states is None: raise ValueError("future_actions and future_states are required")
        one = torch.nn.functional.mse_loss(self.model.predict_one_step(batch, batch["current_action"]), target)
        rollout = torch.nn.functional.mse_loss(self.model.latent_open_loop_rollout(batch, future_actions), future_states)
        return {"one_step": one, "rollout": rollout, "total": self.lambda_one * one + self.lambda_rollout * rollout}
    def loss(self, batch): return self.loss_components(batch)["total"]
    def train_step(self, batch): self.optimizer.zero_grad(); value=self.loss(batch); value.backward(); self.optimizer.step(); return float(value.detach())


class StagedResidualProtocol:
    """Construct the required A-D optimizer states without executing epochs."""
    warmup_epochs = 10
    def __init__(self, baseline, residual_model, residual_name: str, lr_aft: float = 1e-3):
        self.baseline, self.residual_model, self.residual_name, self.lr_aft = baseline, residual_model, residual_name, lr_aft
    def trainable_components(self):
        model = self.residual_model
        return {name: hasattr(model, name) for name in ("backbone", "lift", "A", "B", "aft", "mlp_residual")}
    def _trunk(self, model): return model.hakan if hasattr(model, "hakan") else model
    def get_residual_parameters(self, model=None, residual_name=None):
        module = getattr(model or self.residual_model, residual_name or self.residual_name)
        return list(module.parameters())
    def get_backbone_parameters(self, model=None):
        current = model or self.residual_model
        if hasattr(current, "hakan"):
            return list(current.hakan.parameters())
        residual_ids = {id(parameter) for parameter in self.get_residual_parameters(current)}
        return [parameter for parameter in current.parameters() if id(parameter) not in residual_ids]
    def stage_a_baseline_checkpoint(self):
        baseline_state = self.baseline.state_dict()
        target_state = self._trunk(self.residual_model).state_dict()
        if not (set(baseline_state) & set(target_state)):
            raise ValueError("Stage A baseline shares no checkpoint keys with residual backbone")
        return {"stage": "A", "model": self.baseline.__class__.__name__, "state_dict": {key: value.detach().cpu().clone() for key, value in baseline_state.items()}}
    def stage_b_load_zero_residual(self):
        self._trunk(self.residual_model).load_state_dict(self.baseline.state_dict(), strict=False)
        module = getattr(self.residual_model, self.residual_name)
        if not hasattr(module, "alpha"): raise ValueError("residual module must expose zero gate alpha")
        module.alpha.data.zero_()
        return {"stage": "B", "residual_zero_gate_initialized": True}
    def stage_c_freeze_backbone(self):
        for parameter in self.get_backbone_parameters(): parameter.requires_grad = False
        residual = self.get_residual_parameters()
        for parameter in residual: parameter.requires_grad = True
        optimizer = torch.optim.Adam(residual, lr=self.lr_aft)
        return {"stage": "C", "warmup_epochs": self.warmup_epochs, "optimizer": optimizer}
    def stage_d_joint_finetune(self):
        backbone, residual = self.get_backbone_parameters(), self.get_residual_parameters()
        overlap = {id(parameter) for parameter in backbone} & {id(parameter) for parameter in residual}
        if overlap: raise RuntimeError("backbone and residual parameter groups overlap")
        for parameter in backbone + residual: parameter.requires_grad = True
        return {"stage": "D", "optimizer": torch.optim.Adam([{"params": backbone, "lr": self.lr_aft * 0.1}, {"params": residual, "lr": self.lr_aft}])}
