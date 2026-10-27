Reviewer supplementary test data package
Generated: 2026-10-25T15:40:53.204133+00:00

This package collects the test data and audit outputs used for the responses to Reviewer 1 and Reviewer 2. It intentionally excludes neural checkpoints, optimizer caches, manuscript backups, rendered PDF caches, and duplicate raw archives.

Included experiments
--------------------
01_cstr_delay: CSTR recycle-delay sensitivity at tau=0.010, 0.025, and 0.050 h. Six predictors were fitted separately at each delay; the compact archive retains all model predictions and truth values at the prespecified reporting steps 1, 5, 10, 20, 30, and 60 for all 12 test trajectories per condition.
02_cstr_history: CSTR observation-history sensitivity at L_h=6, 20, and 40 with M=20. The compact archive retains all model predictions and truth values at the same six reporting steps for all 12 common test trajectories per history length.
03_measurement_noise: Frozen-checkpoint measurement-noise audit for Lorenz and CSTR/three-tank data. All archived model files and diagnostic scopes (all/current/past where available) are retained at the six reporting steps; DLM-K initial-lift arrays are retained separately.
04_rigid_dynamics: Raw resolution and convergence audit arrays, together with the numerical summaries.
05_multirate_delay: Raw multi-rate interface data and audit outputs.
06_model_audits: Small parameter, storage, fixed-parameter, scalability, observables, MPC, chemical-literature, and benchmark-scope audit outputs.

Precision and size note
-----------------------
The source delay/history prediction archives contain 100 rollout steps and duplicate the same truth array for every model. To keep this single package below 1 GB, the compact archives deduplicate truth across models, retain every value used in the manuscript tables/figures (steps 1/5/10/20/30/60), and store prediction/truth arrays as float32. Origins, state standard deviations, noise levels, scopes, manifests, protocols, summaries, and audit files are retained. The complete 100-step source archives remain in the workspace locations recorded in 00_metadata/source_experiment_index.csv.

Loading compact arrays
----------------------
Use numpy.load(..., allow_pickle=False). For 01 and 02, prediction has shape (condition, model, test, origin, reporting_step, state); truth has shape (condition, test, origin, reporting_step, state). For 03, each system archive has prediction_normalized with shape (file, origin, reporting_step, state); use the companion index CSV to map each file to system, model, trajectory, noise level, and diagnostic scope.

The numerical summaries are copied without conversion. Non-finite values are retained in the compact arrays and are counted in the corresponding metadata JSON files.
