"""Current-state-conditioned Lorenz H30/B256; shared resumable A/C/D loop."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader


from koopman_delay.models.dlm import DLMKoopman, LSTMHaKANKoopman
from koopman_delay.data import lorenz as native
from koopman_delay.training import staged as staged

MANIFEST_SHA = "d43b77ca6cfe53fc3861352f6f2f3c9b5099496cee2bbfb439f0a75f9d64f2bd"
SCALER_SHA = "8f89ce7b6061c2d16360772f3bc871f66a48e55e0f4acdb68c79e8e91c500f2b"


def experiment_config(seed: int) -> staged.RunConfig:
    return staged.RunConfig(smoke=False, seed=seed, rollout_horizon=30, batch_size=256, current_state_conditioning=True, stage_a_epochs=800, stage_c_epochs=40, stage_d_epochs=80, stage_a_patience=50, stage_c_patience=10, stage_d_patience=10)


def build_models(device, current_state_conditioning=True):
    kwargs = dict(state_dim=3, action_dim=0, history_horizon=20, current_state_conditioning=current_state_conditioning)
    return LSTMHaKANKoopman(**kwargs).to(device), DLMKoopman(aft_context_length=20, **kwargs).to(device)


def load_data(data_dir: Path, config: staged.RunConfig, device):
    if config.rollout_horizon != 30 or config.smoke:
        raise ValueError("Lorenz data factory requires the full canonical H30 train/val protocol")
    native.assert_training_data_access(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = dict(manifest)
    declared = payload.pop("manifest_sha256")
    actual = hashlib.sha256((json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    if declared != MANIFEST_SHA or actual != MANIFEST_SHA:
        raise ValueError("Lorenz frozen manifest mismatch")
    if hashlib.sha256((data_dir / "scaler.json").read_bytes()).hexdigest() != SCALER_SHA:
        raise ValueError("Lorenz frozen scaler mismatch")
    train, val = native.load_windows(data_dir)
    train, val = staged._torch_rows(train), staged._torch_rows(val)
    return DataLoader(staged.DictDataset(train), batch_size=config.batch_size, shuffle=True), val, device, manifest


def run(data_dir, output_dir, seed, device="cuda", resume=None):
    # Fail explicitly on an unavailable requested GPU, rather than silently
    # turning the scheduled GPU experiment into a CPU run.
    native._device(device)
    return staged.run(Path(data_dir), Path(output_dir), device=device, resume=resume, config=experiment_config(seed), model_factory=build_models, data_factory=load_data)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(2, 4, 6, 8), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output_dir or root / "outputs/current_state_lorenz_h30_b256" / f"seed_{args.seed}"
    print(json.dumps(run(args.data_dir, output, args.seed, args.device, args.resume), indent=2))


if __name__ == "__main__":
    main()
