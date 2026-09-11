"""Synthetic-only checks; no experiment assets are read."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koopman_delay.evaluation import common, reproduce


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("system", ["lorenz", "three_tank"])
@pytest.mark.parametrize("model", ["mlp_koopman", "lstm_koopman", "hakan_koopman", "dlm_koopman"])
def test_neural_smoke(system, model, tmp_path):
    torch.set_num_threads(1)
    out = tmp_path / "run"
    script("train").main(["--system", system, "--model", model, "--smoke", "--device", "cpu", "--output-dir", str(out)])
    assert json.loads((out / "run_context.json").read_text())["synthetic_smoke"]
    assert (out / "best_model.pt").exists()


@pytest.mark.parametrize("system", ["lorenz", "three_tank"])
@pytest.mark.parametrize("model", ["rbf_markov", "rbf_physical_delay"])
def test_rbf_external_roster(system, model, tmp_path):
    mode = "markov" if model == "rbf_markov" else ("physical_delay_d3" if system == "lorenz" else "physical_delay")
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps([dict(system=system, model=model, config=dict(mode=mode, n_centers=4, kernel_width=.5, ridge_alpha=1e-4, random_seed=7, physical_delay_steps=5))]))
    result = script("fit_rbf").fit(system, model, tmp_path, tmp_path / "fit", roster, smoke=True)
    assert result["fit_random_seed"] == 7 and np.isfinite(result["selector"])


@pytest.mark.parametrize("name,args", [("train", ["--system", "lorenz", "--model", "dlm_koopman"]), ("fit_rbf", ["--system", "lorenz", "--model", "rbf_markov"]), ("evaluate", [])])
def test_missing_external_input_rejected(name, args, tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / (name + ".py")), *args, "--output-dir", str(tmp_path / "out")], capture_output=True, text=True)
    assert proc.returncode == 2 and "required" in proc.stderr


def test_external_report_and_causal_windows(tmp_path):
    package = tmp_path / "external"
    neural, rbf = [], []
    for system, nx, origins in [("lorenz", 3, np.arange(20, 900)), ("three_tank", 9, np.arange(19, 901))]:
        folder = package / "datasets" / system
        folder.mkdir(parents=True)
        (folder / ("scaler.json" if nx == 3 else "manifest.json")).write_text(json.dumps(dict(state_mean=[0]*nx, state_std=[2]*nx)))
        for model in ("mlp_koopman", "lstm_koopman", "hakan_koopman", "dlm_koopman", "rbf_markov", "rbf_physical_delay"):
            row = dict(system=system, model=model, seed=None if model.startswith("rbf") else 2)
            (rbf if model.startswith("rbf") else neural).append(row)
            folder = package / "raw" / system / model
            folder.mkdir(parents=True)
            for i in range(11):
                truth = np.zeros((len(origins), 100, nx))
                np.savez_compressed(folder / f"trajectory_{i:03}.npz", truth_physical=truth, prediction_physical=truth+1, origins=origins)
    (package / "MODEL_ROSTER.json").write_text(json.dumps(neural))
    (package / "RBF_ROSTER.json").write_text(json.dumps(rbf))
    out = tmp_path / "report"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--artifacts", str(package), "--output-dir", str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    import csv
    rows = list(csv.DictReader((out / "tables/MANUSCRIPT_FORMAL_H1_H100.csv").open()))
    assert len(rows) == 1200
    assert all(float(r["endpoint"]) == .5 and float(r["cumulative"]) == .5 for r in rows)
    states = np.arange(120*3).reshape(120, 3)
    inputs, truth = common.causal_windows(states, np.empty((119, 0)), [19], np.zeros(3), np.ones(3), np.empty(0), np.empty(0))
    assert np.array_equal(inputs["states_history"][0], states[:20])
    assert np.array_equal(truth[0], states[20:120])
    assert "future_states" not in inputs


@pytest.mark.parametrize("system,nx,nu", [("lorenz", 3, 0), ("three_tank", 9, 3)])
def test_external_checkpoint_strict_loading(system, nx, nu, tmp_path):
    from koopman_delay.models import DLMKoopman
    original = DLMKoopman(state_dim=nx, action_dim=nu, history_horizon=20, aft_context_length=20, current_state_conditioning=True)
    path = tmp_path / "synthetic.pt"
    torch.save(original.state_dict(), path)
    row = dict(system=system, model="dlm_koopman", checkpoint="synthetic.pt", checkpoint_sha256=common.sha(path))
    loaded = reproduce.load(tmp_path, row, "cpu")
    assert all(torch.equal(v, loaded.state_dict()[k]) for k,v in original.state_dict().items())
    row["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        reproduce.load(tmp_path, row, "cpu")


@pytest.mark.parametrize("relative", [".", "controlled"])
def test_output_inside_artifacts_rejected_before_access(relative, tmp_path):
    artifacts = tmp_path / "absent-input"
    output = artifacts / relative
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--mode", "controlled", "--artifacts", str(artifacts), "--output-dir", str(output)], capture_output=True, text=True)
    assert proc.returncode == 2 and "must be outside" in proc.stderr
    assert not artifacts.exists()
