"""Train one seed of the current-state-conditioned M5 H30/B256 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


from koopman_delay.training import staged as staged


def experiment_config(seed: int) -> staged.RunConfig:
    return staged.RunConfig(smoke=False, seed=seed, rollout_horizon=30, batch_size=256, current_state_conditioning=True, stage_a_epochs=800, stage_c_epochs=40, stage_d_epochs=80, stage_a_patience=50, stage_c_patience=10, stage_d_patience=10)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(2, 4, 6, 8), help="one manuscript seed; this command runs only this seed")
    parser.add_argument("--output-dir", type=Path, help="new empty output directory; defaults to outputs/current_state_h30_b256/seed_<seed>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, help="matching current-state H30/B256 training_state.pt only")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir or root / "outputs" / "current_state_h30_b256" / f"seed_{args.seed}"
    result = staged.run(args.data_dir, output_dir, device=args.device, resume=args.resume, config=experiment_config(args.seed))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
