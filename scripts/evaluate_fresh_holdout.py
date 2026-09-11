"""Generate and evaluate the pre-frozen fresh holdout without training."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from koopman_delay.data.holdout_generation import generate
from koopman_delay.evaluation import common, reproduce


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def state_curves(prediction, truth, std):
    """RMS over origins per state, then cumulative over forecast time."""
    with np.errstate(over="ignore", invalid="ignore"):
        mse = np.mean(np.square((prediction - truth) / std), axis=0)
        return np.sqrt(mse), np.sqrt(np.cumsum(mse, axis=0) / np.arange(1, 101)[:, None])


def run(protocol_path, device, output_dir=None, data_dir=None):
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text())
    out = Path(output_dir).resolve() if output_dir is not None else ROOT / protocol["result_root"]
    data_root = Path(data_dir).resolve() if data_dir is not None else ROOT / protocol["data_root"]
    if (output_dir is None) != (data_dir is None):
        raise ValueError("Provide both --output-dir and --data-dir for a separate rerun")
    if output_dir is not None and out.exists():
        raise FileExistsError("Rerun output directory must not exist")
    if out.resolve().is_relative_to(data_root.resolve()) or data_root.resolve().is_relative_to(out.resolve()):
        raise ValueError("Output and generated data directories must be separate")
    if (out / "COMPLETE.json").exists() or (out / "status.json").exists():
        raise FileExistsError("This frozen execution already has a status; inspect it, do not overwrite")
    if data_root.exists():
        raise FileExistsError("Holdout data already exist; refusing regeneration")
    for path, expected in protocol["input_sha256"].items():
        if common.sha(ROOT / path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = dict(state="running", pid=os.getpid(), started_at=stamp(), generated=0, evaluated=0, protocol_sha256=common.sha(protocol_path))
    save_json(out / "status.json", status)
    save_json(out / "environment.json", dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device=device, cuda=torch.version.cuda, command=sys.argv))
    try:
        data_root.mkdir(parents=True)
        save_json(data_root / "LOCK.json", dict(training_access=False, status="EVALUATION_ONLY", purpose="fresh_holdout_evaluation_only"))
        manifest = []
        for system, seeds in protocol["seeds"].items():
            folder = data_root / system
            folder.mkdir()
            for index, seed in enumerate(seeds):
                states, actions = generate(system, index, seed)
                path = folder / f"fresh_{index:03d}.npz"
                np.savez_compressed(path, states=states, actions=actions, seed=np.asarray(seed, dtype=np.int64))
                row = dict(system=system, trajectory=path.stem, seed=seed, file=(path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()), sha256=common.sha(path), state_shape=list(states.shape), action_shape=list(actions.shape), all_finite=bool(np.isfinite(states).all() and np.isfinite(actions).all()))
                manifest.append(row)
                status.update(generated=len(manifest), updated_at=stamp())
                save_json(out / "status.json", status)
                print(json.dumps(dict(event="generated", **row)), flush=True)
        save_json(data_root / "MANIFEST.json", manifest)
        if not all(row["all_finite"] for row in manifest):
            raise RuntimeError("Nonfinite generated trajectory retained; no seed replacement permitted")
        all_curves, per_trajectory, per_state, audit = [], [], [], []
        for row in protocol["models"]:
            system, name = row["system"], row["model"]
            mean, std, am, ast = reproduce.scaler(ROOT, system)
            model = reproduce.load(ROOT, row, device)
            curves = []
            for entry in [x for x in manifest if x["system"] == system]:
                with np.load(ROOT / entry["file"], allow_pickle=False) as data:
                    states, actions = data["states"], data["actions"]
                origins = np.arange(20, 900) if system == "lorenz" else np.arange(19, 901)
                inputs, truth = common.causal_windows(states, actions, origins, mean, std, am, ast)
                prediction = common.inverse_scale(reproduce.predict(model, inputs, device), mean, std)
                folder = out / "raw" / system / name
                folder.mkdir(parents=True, exist_ok=True)
                rawpath = folder / (entry["trajectory"] + ".npz")
                np.savez_compressed(rawpath, prediction_physical=prediction, truth_physical=truth, origins=origins)
                ep, cu = common.trajectory_curves(prediction, truth, std)
                curves.append((ep, cu))
                se, sc = state_curves(prediction, truth, std)
                for h in range(100):
                    per_trajectory.append(dict(system=system, model=name, trajectory=entry["trajectory"], seed=row["seed"], horizon=h+1, endpoint=ep[h], cumulative=cu[h]))
                    for c in range(len(std)):
                        per_state.append(dict(system=system, model=name, trajectory=entry["trajectory"], component=c, horizon=h+1, endpoint=se[h,c], cumulative=sc[h,c]))
                check = dict(system=system, model=name, trajectory=entry["trajectory"], windows=len(origins), nonfinite_prediction_values=int((~np.isfinite(prediction)).sum()), nonfinite_prediction_windows=int((~np.isfinite(prediction).all(axis=(1,2))).sum()), raw_sha256=common.sha(rawpath))
                audit.append(check)
                status.update(evaluated=len(audit), updated_at=stamp(), elapsed_seconds=time.monotonic()-started)
                save_json(out / "status.json", status)
                print(json.dumps(dict(event="evaluated", **check)), flush=True)
            average = np.mean(curves, axis=0)
            all_curves.extend(dict(system=system, model=name, seed=row["seed"], split="fresh_holdout", trajectories=len(curves), horizon=h+1, endpoint=average[0,h], cumulative=average[1,h], aggregation="trajectory_RMS_then_equal_trajectory_mean") for h in range(100))
            del model
        tables = out / "tables"
        tables.mkdir()
        common.write_table(tables / "H1_H100.csv", all_curves)
        common.write_table(tables / "H1_H60.csv", [r for r in all_curves if r["horizon"] <= 60])
        common.write_table(tables / "SUMMARY_H20_H30_H50_H60_H100.csv", [r for r in all_curves if r["horizon"] in (20,30,50,60,100)])
        common.write_table(tables / "PER_TRAJECTORY_H1_H100.csv", per_trajectory)
        common.write_table(tables / "PER_STATE_H1_H100.csv", per_state)
        common.write_table(out / "RAW_AUDIT.csv", audit)
        status.update(state="completed", finished_at=stamp(), elapsed_seconds=time.monotonic()-started, nonfinite_prediction_windows=sum(r["nonfinite_prediction_windows"] for r in audit))
        save_json(out / "status.json", status)
        save_json(out / "COMPLETE.json", status)
    except Exception as exc:
        status.update(state="failed", error=repr(exc), updated_at=stamp())
        save_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, help="New output directory for a separate rerun")
    parser.add_argument("--data-dir", type=Path, help="New generated-data directory for a separate rerun")
    args = parser.parse_args()
    run(args.protocol, args.device, args.output_dir, args.data_dir)
