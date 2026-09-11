# DLM-Koopman model source

This repository contains model definitions, system equations, neural training, RBF fitting and evaluation code. Training defaults, both systems' frozen training/validation trajectories and their normalization statistics are included. Trained weights, RBF experiment rosters, formal-public/confirmation data and result archives are not included. It is **not a self-contained reproduction package for manuscript results**.

## Installation and interface

Use Python 3.10. Install PyTorch 2.1.2 for your CPU or CUDA platform, then run `python -m pip install -e .` from this directory. The installer includes NumPy 1.26.4, scikit-learn 1.3.0 (RBF KMeans centers) and Gymnasium 1.2.1 (three-tank environment interface).

`from koopman_delay.models import DLMKoopman` exposes the proposed model. Explicitly pass `current_state_conditioning=True` to enable its current-state fusion; the constructor default is `False`. Set state/action dimensions and history length for the intended system.

The neural interface accepts normalized tensors: `states_history` has shape `[batch, history, state_dim]`, `actions_history` has `[batch, history-1, action_dim]`, and `current_action` has `[batch, action_dim]`. `rollout` additionally takes `future_actions` shaped `[batch, horizon, action_dim]`, whose first action equals `current_action`. Autonomous systems use zero-width action tensors. Normalization is supplied by the caller.

## Source layout

- `src/koopman_delay/models/dlm.py`: DLM-Koopman and shared lifting classes.
- `src/koopman_delay/models/current_state_baselines.py`: MLP, LSTM and HaKAN current-state variants.
- `src/koopman_delay/models/rbf_edmdc.py`: Markov and physical-delay RBF models selected by configuration.
- `src/koopman_delay/models/hakan/` and `aft/`: model components.
- `src/koopman_delay/systems/three_tank/dynamics.py` and `systems/lorenz/dynamics.py`: system equations and integration code.

## Training and evaluation

Training data are in `datasets/lorenz` and `datasets/three_tank`, each with 50 training and 12 validation trajectories. Validation supports the existing early stopping and checkpoint selection. Dataset identities are recorded in `datasets/SHA256.csv` and `datasets/SOURCES.json`. For other inputs use [INPUTS.md](INPUTS.md). Output directories are separate from inputs.

```sh
python scripts/train.py --system lorenz --model dlm_koopman --data-dir datasets/lorenz --output-dir /path/to/new-run --device cpu
python scripts/fit_rbf.py --system lorenz --model rbf_markov --roster /path/to/rbf-roster.json --data-dir datasets/lorenz --output-dir /path/to/new-fit
python scripts/evaluate.py --artifacts /path/to/artifacts --mode report --output-dir /path/to/tables
```

Neural defaults use H30/B256. DLM uses A/C/D stages, and Lorenz DLM includes its spectral loss term. `train.py --smoke` runs small synthetic stages without trajectory data; `fit_rbf.py --smoke` still requires an explicit RBF roster. `evaluate.py` supports table regeneration, model inference and the controlled protocol, with no training. These are protocol-specific loaders, not generic dataset adapters.

No experiment results or trained models are bundled. License selection remains pending; see [LICENSE_STATUS.md](LICENSE_STATUS.md).
