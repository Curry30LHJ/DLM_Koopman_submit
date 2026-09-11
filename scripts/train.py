"""Train the published neural configurations; --smoke uses synthetic data only."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.utils.data import DataLoader
from koopman_delay.training import (
    staged,
    baselines,
    lorenz,
    three_tank,
    spectral,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_factory(system):
    def load(data_dir, config, device):
        nx, nu = (3, 0) if system == "lorenz" else (9, 3)
        generator = torch.Generator().manual_seed(2)
        shapes = {
            "states_history": (4, 20, nx),
            "actions_history": (4, 19, nu),
            "current_action": (4, nu),
            "target_state": (4, nx),
            "future_actions": (4, 30, nu),
            "future_states": (4, 30, nx),
        }
        rows = {
            key: torch.randn(shape, generator=generator) * 0.1
            for key, shape in shapes.items()
        }
        rows["current_action"] = rows["future_actions"][:, 0].clone()
        rows["target_state"] = rows["future_states"][:, 0].clone()
        return (
            DataLoader(staged.DictDataset(rows), batch_size=2, shuffle=True),
            rows,
            torch.device(device),
            {},
        )

    return load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("lorenz", "three_tank"), required=True)
    parser.add_argument(
        "--model",
        choices=(
            "mlp_koopman",
            "lstm_koopman",
            "hakan_koopman",
            "dlm_koopman",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic two-batch test; no trajectory data",
    )
    args = parser.parse_args(argv)
    if not args.smoke and args.data_dir is None:
        parser.error("--data-dir is required unless --smoke is used")
    data = args.data_dir or args.output_dir / "synthetic_input"
    extra = {"data_factory": synthetic_factory(args.system)} if args.smoke else {}
    if args.model == "dlm_koopman":
        config = (
            spectral.experiment_config(args.seed)
            if args.system == "lorenz"
            else three_tank.experiment_config(args.seed)
        )
        if args.smoke:
            config = replace(
                config,
                smoke=True,
                stage_a_epochs=1,
                stage_c_epochs=1,
                stage_d_epochs=1,
                batch_size=2,
            )
        factory = (
            lorenz.build_models if args.system == "lorenz" else staged.build_models
        )
        extra.setdefault(
            "data_factory",
            lorenz.load_data if args.system == "lorenz" else staged._load_data,
        )
        result = staged.run(
            data,
            args.output_dir,
            device=args.device,
            resume=args.resume,
            config=config,
            model_factory=factory,
            **extra,
        )
    else:
        config = baselines.Config(args.system, args.model, args.seed)
        if args.smoke:
            config = replace(config, max_epochs=1, batch_size=2)
        result = baselines.run(
            data,
            args.output_dir,
            config,
            device=args.device,
            resume=args.resume,
            **extra,
        )
    evidence_path = args.output_dir / (
        "staged_evidence.json"
        if args.model == "dlm_koopman"
        else "training_evidence.json"
    )
    (args.output_dir / "run_context.json").write_text(
        json.dumps(
            {
                "synthetic_smoke": args.smoke,
                "data_source": "synthetic" if args.smoke else "train_validation_only",
                "system": args.system,
                "model": args.model,
            },
            indent=2,
        )
    )
    print(evidence_path.read_text())


if __name__ == "__main__":
    main()
