"""Opt-in Lorenz spectral-radius regularization; never overwrites prior runs."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

from koopman_delay.training import lorenz as lorenz


def experiment_config(seed=2, weight=1.0, radius_max=1.0):
    return replace(lorenz.experiment_config(seed), spectral_radius_weight=weight, spectral_radius_max=radius_max)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(2,4,6,8), default=2)
    parser.add_argument("--spectral-weight", type=float, default=1.0)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = experiment_config(args.seed, args.spectral_weight, args.rho_max)
    lorenz.native._device(args.device)
    output = args.output_dir or Path(__file__).resolve().parents[3]/"outputs/current_state_lorenz_spectral_h30_b256"/f"seed_{args.seed}"
    evidence = lorenz.staged.run(args.data_dir, output, device=args.device, resume=args.resume, config=config,
                                model_factory=lorenz.build_models, data_factory=lorenz.load_data)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
