# Reviewer supplementary experiments

This directory archives the compact outputs used to address the reviewer comments. The package is grouped by experiment rather than by manuscript section:

- `01_cstr_delay`: CSTR delay sensitivity and delay-dependent fitted models.
- `02_cstr_history`: CSTR observation-history length study.
- `03_measurement_noise`: process-noise and measurement-noise evaluation.
- `04_rigid_dynamics`: integration-step and rigid-dynamics checks.
- `05_multirate_delay`: multirate and delay-grid evaluation.
- `06_model_audits`: parameter counts, storage accounting, fixed-parameter checks, observables, benchmark scope, chemical-literature context and MPC scope.

The archive contains the report-step predictions, source trajectories, metrics, manifests and audit tables used in the revision. Report-step arrays use steps 1, 5, 10, 20, 30 and 60; non-finite values are retained. Neural checkpoints, optimizer state, manuscript backups and rendered PDF caches are excluded. The package is about 225 MB and its largest individual file is below GitHub's standard 100 MB file limit.

`reviewer_test_data_20261025/checksums.sha256` records the SHA-256 digest of every archived experiment file. Verify the package from the repository root with:

```sh
python scripts/verify_reviewer_experiments.py
```

The package-level bilingual description is in `reviewer_test_data_20261025/README_CN_EN.txt`. The reviewer-to-artifact mapping is in `reviewer_test_data_20261025/00_metadata/reviewer_issue_map.csv`.
