"""Verify manuscript Figure 5 data against the archived physical predictions."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = {
    "mlp_koopman": "prediction_mlp",
    "lstm_koopman": "prediction_lstm",
    "hakan_koopman": "prediction_hakan",
    "dlm_koopman": "prediction_m5",
    "rbf_physical_delay": "prediction_m8",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify():
    for row in read_csv(ROOT / "SHA256.csv"):
        data = (ROOT / row["path"]).read_bytes()
        assert len(data) == int(row["bytes"]), row["path"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]

    normalization = json.loads((ROOT / "STATE_NORMALIZATION.json").read_text(encoding="utf-8"))
    std = np.asarray(normalization["state_std"], dtype=np.float64)
    assert normalization["normalization_fit_split"] == "train"
    assert std.shape == (9,) and np.isfinite(std).all() and np.all(std > 0)

    rows = read_csv(DATA / "PER_CONDITION_H1_H100.csv")
    metrics = {}
    for row in rows:
        key = (row["panel"], row["model"], row["condition_id"], int(row["horizon"]))
        assert key not in metrics, key
        metrics[key] = float(row["endpoint"])
    assert len(metrics) == 2 * 5 * 100 * 100

    curves = {
        (r["panel"], r["model"], int(r["horizon"])): r
        for r in read_csv(DATA / "ENVELOPE_H0_H100.csv")
    }
    max_metric_delta = 0.0
    max_curve_delta = 0.0
    results = {}
    for panel in ("A", "B"):
        with np.load(DATA / f"PANEL_{panel}_RAW.npz", allow_pickle=False) as neural, \
             np.load(DATA / f"PANEL_{panel}_RBF_RAW.npz", allow_pickle=False) as rbf, \
             np.load(DATA / f"PANEL_{panel}_FROZEN_TRUTH.npz", allow_pickle=False) as frozen:
            ids = neural["condition_ids"].tolist()
            assert len(ids) == len(set(ids)) == 100
            assert ids == rbf["condition_ids"].tolist()
            condition_rows = read_csv(DATA / f"ORIGINAL_{panel}_CONDITIONS.csv")
            assert ids == [row["condition_id"] for row in condition_rows]
            truth = neural["truth_physical"]
            assert truth.shape == (100, 100, 9)
            assert np.array_equal(truth, frozen["truth_physical"])
            for name, key in MODELS.items():
                prediction = (rbf if name == "rbf_physical_delay" else neural)[key]
                assert prediction.shape == truth.shape and np.isfinite(prediction).all()
                error = np.sqrt(np.mean(((prediction - truth) / std) ** 2, axis=2))
                saved = np.array([
                    [metrics[(panel, name, condition, h)] for h in range(1, 101)]
                    for condition in ids
                ])
                max_metric_delta = max(max_metric_delta, float(np.max(np.abs(error - saved))))
                for h in range(1, 101):
                    row = curves[(panel, name, h)]
                    actual = error[:, h - 1]
                    for field, expected in (("mean", actual.mean()),
                                            ("minimum", actual.min()),
                                            ("maximum", actual.max())):
                        max_curve_delta = max(max_curve_delta, abs(float(row[field]) - expected))
                if name == "dlm_koopman":
                    j = error[:, :60].mean(axis=1)
                    results[panel] = {"mean_J": float(j.mean()), "median_J": float(np.median(j))}
    assert max_metric_delta < 1e-12, max_metric_delta
    assert max_curve_delta < 1e-12, max_curve_delta
    assert abs(results["A"]["mean_J"] - 0.05152) < 0.000005
    assert abs(results["B"]["mean_J"] - 0.01141) < 0.000005
    print(json.dumps({"status": "PASS", "conditions_per_panel": 100,
                      "models": len(MODELS), "metric_rows": len(metrics),
                      "max_metric_delta": max_metric_delta,
                      "max_curve_delta": max_curve_delta,
                      "DLM_K": results}, indent=2))


if __name__ == "__main__":
    verify()
