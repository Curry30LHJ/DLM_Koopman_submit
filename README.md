# DLM-Koopman model source

This repository contains **model definitions and system equations only**. The training/evaluation pipeline, trained weights, datasets and complete experiment configuration are not included. It is **not a self-contained reproduction package for manuscript results**.

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

No experiment results or trained models are claimed by this source-only release. License selection remains pending; see [LICENSE_STATUS.md](LICENSE_STATUS.md).
