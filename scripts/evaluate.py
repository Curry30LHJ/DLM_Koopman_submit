"""Recompute frozen metrics or replay selected models, without training."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from koopman_delay.evaluation import reproduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("report", "infer", "controlled"), default="report"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="External artifact package following INPUTS.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/evaluation",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail on any prediction byte difference; intended for the frozen CUDA environment",
    )
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.artifacts.resolve()):
        parser.error("--output-dir must be outside --artifacts to preserve input files")
    if args.mode == "controlled":
        reproduce.controlled(
            args.artifacts,
            args.device,
            output=args.output_dir,
            require_exact=args.require_exact,
        )
    else:
        reproduce.natural(
            args.artifacts,
            args.device,
            infer=args.mode == "infer",
            output=args.output_dir,
            require_exact=args.require_exact,
        )


if __name__ == "__main__":
    main()
