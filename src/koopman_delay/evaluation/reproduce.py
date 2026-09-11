"""Frozen six-model data verification, metrics and inference."""

import argparse, csv, json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
from . import common as ev
from . import controlled_protocol as protocol
from koopman_delay.model_names import MODEL_NAMES, SLUG_BY_PREDICTION_KEY


def scaler(package, system):
    file = "scaler.json" if system == "lorenz" else "manifest.json"
    s = json.loads((package / "datasets" / system / file).read_text())
    return tuple(
        np.asarray(s.get(k, []), dtype=np.float64)
        for k in ("state_mean", "state_std", "action_mean", "action_std")
    )


def load(package, row, device):
    if row["model"] in ("rbf_markov", "rbf_physical_delay"):
        from koopman_delay.models.rbf_edmdc import (
            RBFEDMDc,
            RBFEDMDcConfig,
        )

        path = package / row["checkpoint"]
        assert ev.sha(path) == row["checkpoint_sha256"]
        model = RBFEDMDc(RBFEDMDcConfig(**row["config"]))
        with np.load(path, allow_pickle=False) as arrays:
            model.A_, model.B_, model.C_, model.centers_ = (
                arrays[k] for k in ("A", "B", "C", "centers")
            )
        model.feature_dim_ = model.A_.shape[0]
        model.history_horizon_ = 20
        model.state_dim_, model.action_dim_ = (
            (3, 0) if row["system"] == "lorenz" else (9, 3)
        )
        return model
    return ev.load_selected(
        {**row, "checkpoint": str(package / row["checkpoint"])}, device
    )


def roster(package):
    return json.loads((package / "MODEL_ROSTER.json").read_text()) + json.loads(
        (package / "RBF_ROSTER.json").read_text()
    )


def predict(model, inputs, device):
    if hasattr(model, "rollout_batch"):
        return model.rollout_batch(inputs["states_history"], inputs["future_actions"])
    return ev.predict(model, inputs, device)


def natural(package, device, infer=False, output=None, require_exact=False):
    output = Path(output) if output is not None else ROOT / "outputs/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    manuscript = []
    manuscript_per = []
    for row in roster(package):
        system, name = row["system"], row["model"]
        mean, std, am, ast = scaler(package, system)
        paths = sorted((package / "raw" / system / name).glob("*.npz"))
        assert len(paths) == 11
        local_rms = []
        model = load(package, row, device) if infer else None
        for rawpath in paths:
            with np.load(rawpath, allow_pickle=False) as raw:
                physical, truth, origins = (
                    raw["prediction_physical"],
                    raw["truth_physical"],
                    raw["origins"],
                )
            assert np.array_equal(
                origins,
                np.arange(20, 900) if system == "lorenz" else np.arange(19, 901),
            )
            assert physical.shape == truth.shape == (len(origins), 100, len(std))
            delta = 0.0
            if infer:
                data = (
                    package
                    / "datasets"
                    / system
                    / "formal_public"
                    / (rawpath.stem + (".npy" if system == "lorenz" else ".npz"))
                )
                if system == "lorenz":
                    states = np.load(data, allow_pickle=False)
                    actions = np.empty((len(states) - 1, 0))
                else:
                    with np.load(data, allow_pickle=False) as z:
                        states, actions = z["states"], z["actions"]
                inputs, target = ev.causal_windows(
                    states, actions, origins, mean, std, am, ast
                )
                assert np.array_equal(target, truth)
                predicted = ev.inverse_scale(predict(model, inputs, device), mean, std)
                delta = float(np.max(np.abs(predicted - physical)))
            rms_ep, rms_cu = ev.trajectory_curves(physical, truth, std)
            local_rms.append((rms_ep, rms_cu))
            manuscript_per.extend(
                dict(
                    system=system,
                    model=name,
                    seed=row["seed"],
                    split="formal_public",
                    trajectory=rawpath.stem,
                    horizon=h + 1,
                    endpoint=rms_ep[h],
                    cumulative=rms_cu[h],
                    aggregation="trajectory_RMS_then_equal_trajectory_mean",
                )
                for h in range(100)
            )
            checks.append(
                dict(
                    system=system,
                    model=name,
                    trajectory=rawpath.stem,
                    windows=len(origins),
                    nonfinite_windows=int(
                        (~np.isfinite(physical).all(axis=(1, 2))).sum()
                    ),
                    replay_max_abs_delta=delta if infer else "NOT_RUN",
                )
            )
            print(
                json.dumps(
                    dict(
                        system=system,
                        model=name,
                        trajectory=rawpath.stem,
                        inference=infer,
                        max_abs_delta=delta,
                    )
                ),
                flush=True,
            )
        rms = np.mean(local_rms, axis=0)
        manuscript.extend(
            dict(
                system=system,
                model=name,
                seed=row["seed"],
                split="formal_public",
                horizon=h + 1,
                endpoint=rms[0, h],
                cumulative=rms[1, h],
                aggregation="trajectory_RMS_then_equal_trajectory_mean",
            )
            for h in range(100)
        )
        del model
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    ev.write_table(tables / "MANUSCRIPT_FORMAL_H1_H100.csv", manuscript)
    ev.write_table(
        tables / "MANUSCRIPT_FORMAL_H1_H60.csv",
        [r for r in manuscript if r["horizon"] <= 60],
    )
    ev.write_table(
        tables / "MANUSCRIPT_FORMAL_H20_H30_H50_H60_H100.csv",
        [r for r in manuscript if r["horizon"] in (20, 30, 50, 60, 100)],
    )
    ev.write_table(
        tables / "MANUSCRIPT_FORMAL_PER_TRAJECTORY_H1_H100.csv", manuscript_per
    )
    ev.write_table(
        output / ("INFERENCE_REPLAY.csv" if infer else "RAW_AUDIT.csv"), checks
    )
    for system in ("three_tank", "lorenz"):
        example = []
        for name in (
            "mlp_koopman",
            "lstm_koopman",
            "hakan_koopman",
            "dlm_koopman",
            "rbf_markov",
            "rbf_physical_delay",
        ):
            with np.load(
                sorted((package / "raw" / system / name).glob("*.npz"))[0],
                allow_pickle=False,
            ) as raw:
                example.extend(
                    dict(
                        model=name,
                        horizon=h + 1,
                        component=c,
                        truth=raw["truth_physical"][0, h, c],
                        prediction=raw["prediction_physical"][0, h, c],
                    )
                    for c in range(raw["truth_physical"].shape[2])
                    for h in range(60)
                )
        ev.write_table(tables / f"{system}_fixed_origin_H60.csv", example)
    if infer and require_exact:
        assert all(
            r["replay_max_abs_delta"] == 0 for r in checks
        ), "Inference differs; inspect INFERENCE_REPLAY.csv; frozen raw retained"


def controlled(package, device, truth_only=False, output=None, require_exact=False):
    output = Path(output) if output is not None else ROOT / "outputs/controlled"
    output.mkdir(parents=True, exist_ok=True)
    out = package / "controlled"
    mean, std, am, ast = scaler(package, "three_tank")
    with np.load(out / "FROZEN_INPUTS.npz", allow_pickle=False) as z:
        histories, past, profiles, fixed = (
            z["histories"],
            z["past_actions"],
            z["profiles"],
            z["fixed_actions"],
        )
    rows = [
        r
        for r in roster(package)
        if r["system"] == "three_tank" and r["model"] != "rbf_markov"
    ]
    models = {} if truth_only else {r["model"]: load(package, r, device) for r in rows}
    coefficients = (
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (0.5, 0.5),
    )
    all_metrics = []
    audit = []
    replay_checks = []
    for panel in ("A", "B"):
        frozen_truth_path = out / f"PANEL_{panel}_FROZEN_TRUTH.npz"
        frozen_truth = None
        if not truth_only and frozen_truth_path.exists():
            with np.load(frozen_truth_path, allow_pickle=False) as z:
                frozen_truth = z["truth_physical"]
        conditions = list(
            csv.DictReader((out / f"ORIGINAL_{panel}_CONDITIONS.csv").open())
        )
        assert len(conditions) == 100
        shs = []
        ahs = []
        fas = []
        truths = []
        requested = []
        for ci, row in enumerate(conditions):
            hi = int(row["history_id"][1:])
            sh, ah = histories[hi], past[hi]
            if panel == "A":
                pi = int(row["profile_index"])
                base = profiles[hi, 0]
                if pi < 7:
                    req = profiles[hi, pi]
                else:
                    signs = {7: (1, 1, 1), 8: (-1, -1, -1), 9: (1, -1, 1)}[pi]
                    req = base + sum(
                        sign * (profiles[hi, index] - base)
                        for sign, index in zip(signs, (1, 3, 5))
                    )
                noise = protocol.gen_noise(protocol.NOISE_BASE_A + hi)
            else:
                fi = int(row["slice"][1:])
                c0, c1 = coefficients[fi]
                req = (
                    fixed[fi]
                    if fi < 3
                    else fixed[0]
                    + c0 * (fixed[1] - fixed[0])
                    + c1 * (fixed[2] - fixed[0])
                )
                noise = protocol.gen_noise(protocol.NOISE_BASE_B + fi)
            app, clip_count = protocol.clip_actions(req)
            truth = (
                protocol.plant_frozen_noise(sh, app, noise)
                if frozen_truth is None
                else frozen_truth[ci]
            )
            assert protocol.sha_of(sh) == row["states_history_sha256"]
            assert protocol.sha_of(ah) == row["actions_history_sha256"]
            assert protocol.sha_of(app) == row["applied_future_actions_sha256"]
            assert protocol.sha_of(noise) == row["future_noise_sha256"]
            truth_delta_hash = (
                protocol.sha_of((truth - mean) / std) == row["truth_sha256"]
            )
            assert truth_delta_hash, (panel, row["condition_id"], "truth mismatch")
            shs.append(sh)
            ahs.append(ah)
            fas.append(app)
            truths.append(truth)
            requested.append(req)
            audit.append(
                dict(
                    panel=panel,
                    condition_id=row["condition_id"],
                    original_history_action_noise_truth_hash_match=True,
                    clip_count=clip_count,
                )
            )
        inputs = dict(
            states_history=(np.array(shs) - mean) / std,
            actions_history=(np.array(ahs) - am) / ast,
            future_actions=(np.array(fas) - am) / ast,
        )
        inputs["current_action"] = inputs["future_actions"][:, 0]
        truth = np.array(truths)
        if truth_only:
            np.savez_compressed(frozen_truth_path, truth_physical=truth)
            print(
                json.dumps(
                    dict(
                        panel=panel, truth_conditions=100, original_truth_sha_exact=True
                    )
                ),
                flush=True,
            )
            continue
        raw = dict(
            condition_ids=np.array([r["condition_id"] for r in conditions]),
            states_history_physical=np.array(shs),
            actions_history_physical=np.array(ahs),
            future_actions_physical=np.array(fas),
            requested_actions_physical=np.array(requested),
            truth_physical=truth,
        )
        for name, model in models.items():
            physical = ev.inverse_scale(predict(model, inputs, device), mean, std)
            raw[MODEL_NAMES[name]["prediction_key"]] = physical
            error = np.sqrt(np.mean(((physical - truth) / std) ** 2, axis=2))
            assert np.isfinite(error).all()
            all_metrics.extend(
                dict(
                    panel=panel,
                    model=name,
                    condition_id=conditions[c]["condition_id"],
                    horizon=h + 1,
                    endpoint=error[c, h],
                )
                for c in range(100)
                for h in range(100)
            )
        rbf_raw = dict(
            condition_ids=raw["condition_ids"], prediction_m8=raw.pop("prediction_m8")
        )
        for path, values in (
            (out / f"PANEL_{panel}_RBF_RAW.npz", rbf_raw),
            (out / f"PANEL_{panel}_RAW.npz", raw),
        ):
            with np.load(path, allow_pickle=False) as previous:
                assert set(previous.files) == set(values)
                for key, value in values.items():
                    if key.startswith("prediction_"):
                        delta = float(np.max(np.abs(previous[key] - value)))
                        replay_checks.append(
                            dict(
                                panel=panel,
                                model=SLUG_BY_PREDICTION_KEY[key],
                                max_abs_delta=delta,
                            )
                        )
                    else:
                        assert np.array_equal(
                            previous[key], value
                        ), f"Frozen input/truth mismatch: {key}"
        print(
            json.dumps(
                dict(panel=panel, conditions=100, models=5, truth_hash_exact=True)
            ),
            flush=True,
        )
    if truth_only:
        ev.write_table(output / "FROZEN_TRUTH_AUDIT.csv", audit)
        return
    ev.write_table(output / "CONTROLLED_REPLAY.csv", replay_checks)
    if require_exact:
        assert all(
            r["max_abs_delta"] == 0 for r in replay_checks
        ), "Controlled inference differs; frozen raw retained"
    ev.write_table(output / "PER_CONDITION_H1_H100.csv", all_metrics)
    ev.write_table(output / "CONDITION_AUDIT.csv", audit)
    envelope = []
    for panel in ("A", "B"):
        for name in models:
            values = np.array(
                [
                    r["endpoint"]
                    for r in all_metrics
                    if r["panel"] == panel and r["model"] == name
                ]
            ).reshape(100, 100)
            avg, low, high = values.mean(0), values.min(0), values.max(0)
            envelope.extend(
                dict(
                    panel=panel,
                    model=name,
                    horizon=h,
                    forecast_time=h * 0.005,
                    row_role="PLOTTING_ORIGIN" if h == 0 else "DATA",
                    mean=0.0 if h == 0 else avg[h - 1],
                    minimum=0.0 if h == 0 else low[h - 1],
                    maximum=0.0 if h == 0 else high[h - 1],
                )
                for h in range(101)
            )
    ev.write_table(output / "ENVELOPE_H0_H100.csv", envelope)
    ev.write_table(
        output / "ENVELOPE_H0_H60.csv", [r for r in envelope if r["horizon"] <= 60]
    )
