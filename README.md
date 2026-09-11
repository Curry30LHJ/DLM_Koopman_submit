# DLM-Koopman model source

This repository contains model definitions, system equations, neural training, RBF fitting and evaluation code. Training defaults, both systems' frozen training/validation trajectories and their normalization statistics are included. Frozen selected weights and a new simulated holdout evaluation are also included. Historical formal-public/confirmation arrays and previous result archives are not included.

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

## Frozen models and fresh holdout

`checkpoints/` contains the 8 neural seed-2 and 4 frozen RBF weights. Their identities and configurations are fixed in [PROTOCOL.json](evaluation/fresh_holdout/PROTOCOL.json). `datasets/fresh_holdout/` contains 12 newly generated trajectories per system, isolated from training. [SUMMARY_H20_H30_H50_H60_H100.csv](evaluation/fresh_holdout/tables/SUMMARY_H20_H30_H50_H60_H100.csv) reports the new evaluation; `evaluation/fresh_holdout/raw/` retains every H100 prediction, truth and origin. Per-trajectory and per-state curves, logs and data hashes accompany the summary. [REPORT.md](evaluation/fresh_holdout/REPORT.md) records execution and checks; `python scripts/verify_fresh_holdout.py` verifies the archive. No plots are generated.

Previous formal-public results were used during development diagnostics. Historical confirmation sets had also been accessed; neither is presented here as a fresh test. The present holdout was generated after freezing model weights, training scalers, generation seeds and metrics, without retraining or selecting samples using its results. It is a new simulated sample from the original system distributions, not external physical validation. All 12 trajectories per system contribute equally; neural weights still represent one training seed, not a multi-seed study.

To regenerate the fixed seeds and evaluate into new directories (CPU supported; archived execution used CUDA):

```sh
python scripts/evaluate_fresh_holdout.py --protocol evaluation/fresh_holdout/PROTOCOL.json --data-dir outputs/replay-data --output-dir outputs/replay-results --device cpu
```

Both directories must be new; the archived inputs/results are never overwritten. This repeats the published fixed-seed test, not a new independent test. See [INPUTS.md](INPUTS.md) for other evaluation inputs.
## Figure 5: controlled A/B conditions

[Figure 5 summary](evaluation/figure5/SUMMARY_H20_H30_H50_H60_H100.csv) and [condition manifest](evaluation/figure5/CONDITIONS.csv) archive both panels:

- A: fix each of 12 causal histories; vary its future inputs across seven profiles (base and each Q channel plus/minus one training standard deviation).
- B: fix each of the seven future-input profiles from `fresh_000`; apply it to all 12 histories with the same future noise. Histories include different terminal current states; this is not an isolated memory-only intervention.

Each panel has 84 conditions (12 x 7), rather than the historical 77 (11 x 7). All six frozen models are included: 1008 model-condition predictions. Full past replay rebuilds the 26-state internal delay buffer; base continuation is checked at absolute tolerance 1e-10. Applied inputs are clipped to plant bounds and shared with every model. `evaluation/figure5/inputs/` saves complete branch buffers, past actions, original process noise, future noise, histories, requested/applied inputs and new plant truth. Model NPZ files retain all H100 predictions and condition IDs. H60/H100 curves, per-state/per-condition metrics and grouped mean/min/max are supplied; min/max are condition ranges, not confidence intervals. No images are generated.

These are pre-frozen operating-condition extensions on the same fresh holdout already reported in the natural evaluation, not a second previously unseen confirmation sample. Models were not retrained or selected using these tests. See [REPORT.md](evaluation/figure5/REPORT.md) and [PROTOCOL.json](evaluation/figure5/PROTOCOL.json).

```sh
python scripts/verify_figure5.py
python scripts/evaluate_figure5.py --protocol evaluation/figure5/PROTOCOL.json --output-dir outputs/figure5-replay --device cpu
```

The replay output must be new. Historical non-independent simulation arrays were never included in this submission repository; the natural fresh-holdout archive and training/validation datasets remain intact.

License selection remains pending; see [LICENSE_STATUS.md](LICENSE_STATUS.md).
