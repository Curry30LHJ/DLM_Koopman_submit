"""Verify frozen fresh-holdout identities and independently recompute summary metrics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/fresh_holdout"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    complete = json.loads((OUT / "COMPLETE.json").read_text())
    assert digest(OUT / "PROTOCOL.json") == complete["protocol_sha256"]
    for file, sha in protocol["input_sha256"].items():
        assert digest(ROOT / file) == sha, file
    data = json.loads((ROOT / protocol["data_root"] / "MANIFEST.json").read_text())
    assert len(data) == 24
    for row in data:
        assert digest(ROOT / row["file"]) == row["sha256"]
    lookup = {(r["system"], r["trajectory"]): r for r in data}
    summary = list(csv.DictReader((OUT / "tables/SUMMARY_H20_H30_H50_H60_H100.csv").open()))
    audit = list(csv.DictReader((OUT / "RAW_AUDIT.csv").open()))
    assert len(audit) == 144 and len(summary) == 60
    for row in audit:
        path = OUT / "raw" / row["system"] / row["model"] / (row["trajectory"] + ".npz")
        assert digest(path) == row["raw_sha256"]
    max_delta = 0.0
    for row in summary:
        system, model, h = row["system"], row["model"], int(row["horizon"])
        scalerfile = ROOT / "datasets" / system / ("scaler.json" if system == "lorenz" else "manifest.json")
        std = np.array(json.loads(scalerfile.read_text())["state_std"])
        endpoint, cumulative = [], []
        paths = sorted((OUT / "raw" / system / model).glob("*.npz"))
        assert len(paths) == 12
        for path in paths:
            with np.load(path, allow_pickle=False) as raw:
                truth, prediction, origins = raw["truth_physical"], raw["prediction_physical"], raw["origins"]
            with np.load(ROOT / lookup[system,path.stem]["file"], allow_pickle=False) as original:
                assert np.array_equal(truth, np.stack([original["states"][k+1:k+101] for k in origins]))
            error = (prediction-truth)/std
            # Direct L2 norm, independent of the evaluator's square/mean/cumsum path.
            endpoint.append(np.linalg.norm(error[:,h-1].ravel())/np.sqrt(error[:,h-1].size))
            cumulative.append(np.linalg.norm(error[:,:h].ravel())/np.sqrt(error[:,:h].size))
        for name, expected in [("endpoint", np.mean(endpoint)), ("cumulative", np.mean(cumulative))]:
            actual = float(row[name])
            assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True), row
            if np.isfinite(actual):
                max_delta = max(max_delta, float(abs(actual-expected)))
    result = dict(status="PASS", checkpoints=len(protocol["models"]), trajectories=len(data), raw_archives=len(audit), summary_rows=len(summary), independent_summary_max_abs_delta=max_delta, nonfinite_prediction_windows=sum(int(r["nonfinite_prediction_windows"]) for r in audit))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
