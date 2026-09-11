# Fresh simulated holdout evaluation

The protocol and model/scaler/code hashes were frozen before generation. All 24 pre-listed seeds were generated once and all 144 model/trajectory predictions were retained. No training, parameter selection, checkpoint change or sample replacement took place. This is a fresh simulated sample from the original generating distributions, not external physical validation. Historical formal-public and previously consumed confirmation sets are not reused here.

Execution: 2026-09-11 12:38:46 to 12:39:55 UTC, 68.407 seconds, CUDA, PID 27720. All 24 generated trajectories and all predictions are finite. The archive contains H100 truth/prediction/origin arrays, H60 curves, endpoint/cumulative summaries at H20/30/50/60/100, per-trajectory and per-state metrics. The default aggregation is per-trajectory RMS followed by equal weighting of 12 trajectories. The 8 neural checkpoints use training seed 2; the 4 RBF checkpoints are fixed closed-form fits, not additional neural seeds.

Generator precheck used already-public train_000 for each system: Lorenz seed1000 states were bitwise identical. Three-tank seed6 actions were bitwise identical and state maximum absolute difference was 2.7755575615628914e-17; the canonical simulation function AST was identical. The latter is recorded as platform rounding, not claimed bitwise reproduction. No implementation was altered to improve new holdout results.

Verification covers frozen input and output hashes, all raw truths against generated trajectories, all retained samples, and independent L2-norm recomputation of the summary metrics. `scripts/verify_fresh_holdout.py` checks the committed archive. Original state/action arrays and checkpoints are not replaced or fitted. CPU replay can differ numerically from the archived CUDA execution.

Results describe these frozen checkpoints on this simulated holdout; no cross-training-seed or statistical-significance claim.
