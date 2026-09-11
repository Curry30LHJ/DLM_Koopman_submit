# Figure 5 operating-condition extension

A and B were frozen before condition scoring on the 12 three-tank fresh-holdout trajectories already used for natural evaluation. Models/scalers were unchanged. This tests fixed-history/varied-input and fixed-input/varied-history behavior on that same holdout; it is not a second unseen confirmation sample. No historical formal-public or confirmation arrays were added.

Window440 (origin459) is identical for all histories. A has12 histories x7 own-input profiles; B has7 fixed fresh_000 profiles x12 histories. Full seed/noise/past-action replay reconstructs each delay buffer, without interpolation. A changes future inputs with within-history common noise; B changes natural histories (including current state), holding future input/noise fixed. Applied profiles are bounded by plant clipping.

Execution completed on2026-09-11 at13:36:59 UTC, PID53544, in10.234seconds. All168 conditions and1008 model-condition predictions were retained, with no nonfinite outputs. Public train_000 precheck: past reconstruction difference2.7755575615628914e-17, base future difference0, below pre-frozen absolute tolerance1e-10. A base continuation passed for all12 histories; B base-natural check applies only to fresh_000.

All seven fresh_000 A/B inputs, histories, buffers, noise, truths and model predictions are exactly equal. B applied actions/noise are identical across histories within each profile. Independent summary metric recomputation has maximum absolute difference4.440892098500626e-16. Per-condition and per-state data plus grouped curves are supplied. Mean/min/max show condition variation, not uncertainty over training seeds or confidence intervals.
