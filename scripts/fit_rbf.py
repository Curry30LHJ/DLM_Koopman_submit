"""Single closed-form fit with frozen hyperparameters; no search or neural training."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from koopman_delay.data import lorenz, three_tank
from koopman_delay.models.rbf_edmdc import RBFEDMDc, RBFEDMDcConfig

ROOT = Path(__file__).resolve().parents[1]


def fit(system, name, data_dir, output_dir, roster_path, smoke=False):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output_dir}")
    rows = json.loads(Path(roster_path).read_text(encoding="utf-8"))
    row = next(r for r in rows if r["system"] == system and r["model"] == name)
    config = RBFEDMDcConfig(**row["config"])
    horizon = 30 if system == "lorenz" else 20
    if smoke:
        rng = np.random.default_rng(6)
        nx, nu = (3, 0) if system == "lorenz" else (9, 3)

        def synthetic(n):
            states = rng.normal(0.2, 0.01, (n, 1, nx))
            return dict(
                states_history=np.repeat(states, 20, axis=1),
                current_action=np.zeros((n, nu)),
                target_state=states[:, 0],
                future_actions=np.zeros((n, horizon, nu)),
                future_states=np.repeat(states, horizon, axis=1),
            )

        train, val = synthetic(max(160, config.n_centers + 1)), synthetic(4)
    elif system == "lorenz":
        train, val = lorenz.load_windows(data_dir)
    else:
        manifest = three_tank.load_manifest_without_confirmation(data_dir)
        train, val = [
            three_tank._normalize(
                three_tank._load_split_windows(data_dir, manifest, split, 20, 20),
                manifest,
            )
            for split in ("train", "val")
        ]
    model = RBFEDMDc(config).fit(
        train["states_history"], train["current_action"], train["target_state"]
    )
    prediction = model.rollout_batch(val["states_history"], val["future_actions"])
    error = prediction - val["future_states"]
    if not np.isfinite(error).all():
        raise RuntimeError("Nonfinite validation prediction; not clipped or accepted")
    # Frozen fit evidence selects validation MSE (I H20, III H30); no search here.
    selector = float(np.mean(np.square(error)))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "best_model.npz",
        config=json.dumps(row["config"], sort_keys=True),
        A=model.A_,
        B=model.B_,
        C=model.C_,
        centers=model.centers_,
    )
    result = dict(
        system=system,
        model=name,
        seed=None,
        fit_random_seed=config.random_seed,
        config=row["config"],
        history_horizon=20,
        validation_horizon=horizon,
        train_windows=len(train["states_history"]),
        val_windows=len(val["states_history"]),
        selector=selector,
        selector_metric="MSE",
        validation_rmse=float(np.sqrt(selector)),
        synthetic_smoke=smoke,
        frozen_weight_equality_not_asserted=True,
    )
    (output_dir / "fit_evidence.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("lorenz", "three_tank"), required=True)
    parser.add_argument(
        "--model", choices=("rbf_markov", "rbf_physical_delay"), required=True
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--roster", type=Path, required=True, help="External RBF configuration JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and args.data_dir is None:
        parser.error("--data-dir is required unless --smoke is used")
    data = args.data_dir or args.output_dir / "synthetic_input"
    print(
        json.dumps(
            fit(args.system, args.model, data, args.output_dir, args.roster, args.smoke), indent=2
        )
    )


if __name__ == "__main__":
    main()
