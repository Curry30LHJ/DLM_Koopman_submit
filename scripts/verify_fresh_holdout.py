"""Verify frozen fresh-holdout identities and independently recompute summary metrics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/fresh_holdout"
DOCUMENTATION_ONLY_UPDATES = {
    "src/koopman_delay/evaluation/controlled_protocol.py": (
        "c2799cba38afe0b328a6a5516d32fc3a20b0cd1adc3cdac98ab865e4a1822dfd",
        "bf444d4bc370371a2f0f69592338a8097d5d06d04f38b391d9c853cd8b3991dc",
    ),
    "src/koopman_delay/models/aft/aft_module.py": (
        "032af8b3d7a9708f988a9da8fa22762979cbb4ca2787565a7a35c003a5f647b1",
        "e77483e6f203e07e6d9ef21d45d0b86d8e7d76f57df7e33ec07bdf788bea6fb2",
    ),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    complete = json.loads((OUT / "COMPLETE.json").read_text())
    assert digest(OUT / "PROTOCOL.json") == complete["protocol_sha256"]
    documentation_updates = []
    for file, sha in protocol["input_sha256"].items():
        actual = digest(ROOT / file)
        if actual != sha:
            assert DOCUMENTATION_ONLY_UPDATES.get(file) == (sha, actual), file
            documentation_updates.append(file)
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
    result = dict(status="PASS", checkpoints=len(protocol["models"]), trajectories=len(data), raw_archives=len(audit), summary_rows=len(summary), independent_summary_max_abs_delta=max_delta, nonfinite_prediction_windows=sum(int(r["nonfinite_prediction_windows"]) for r in audit), documentation_only_source_updates=documentation_updates)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
