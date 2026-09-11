# Tables 2/3 and Figures 2/4: independent-test manuscript assets

This directory archives the data and accepted graphics used in the current manuscript, including the median-error CSTR example in Figure 4. The scientific source is commit **c12e955b8fe80030ac1e6c7dc131e8f4ec91cc44** of this repository. The independent-test raw predictions already exist under `evaluation/fresh_holdout/raw/`; they are not duplicated here. The original training, models, scalers and evaluation protocol are unchanged.

| Manuscript item | Archived input | Figure/table output |
|---|---|---|
| Table 2, Lorenz | `data/independent_test_H1_H60.csv`, Lorenz rows | `tables/lorenz_table.tex` |
| Table 3, CSTR | Same CSV, three-tank rows | `tables/three_tank_table.tex` |
| Figure 2, Lorenz | `data/lorenz_fixed_origin_H60.csv` for states; shared summary CSV for error | `figures/fig_2_lorenz_evaluation_h60.*` |
| Figure 4, CSTR | `data/figure4_median_window_H60.csv` | `figures/fig_4_cstr_representative_h60.*` |

Each figure is archived as vector PDF, editable-text SVG and 600 dpi PNG. Their bytes match the accepted local manuscript figures. Full manuscript source/PDF, local paths, backups and unrelated audit logs are not part of this upload. Figure 5 is outside this archive; see its separate repository records.

## Data definitions and model identities

The summary CSV contains 720 rows: two systems, six predictors, sixty forecast steps. `independent_test_per_trajectory_H1_H100.csv` contains 14,400 trajectory-level records. The main tables report steps 1, 5, 10, 20, 30 and 60. For each step, the reported error is the root mean squared error over legal windows and state coordinates on one trajectory, standardized by the frozen training state scales, followed by an equal-weight mean over the twelve test trajectories. The archived column name `endpoint` denotes this single-step error, not a cumulative average. `cumulative` is a distinct archived metric and is not used by these main tables.

| CSV short key | Raw archive model | Display label |
|---|---|---|
| mlp | mlp_koopman | MLP–K |
| lstm | lstm_koopman | LSTM–K |
| hakan | hakan_koopman | HaKAN–K |
| m6 | rbf_markov | Markov RBF |
| m8 | rbf_physical_delay | Delay–RBF |
| m5 | dlm_koopman | DLM–K |

The displayed order is the order above. These are single trained neural instances and frozen RBF fits, not multi-seed mean predictions. The LaTeX tables retain manuscript citation keys; they are fragments rather than standalone documents.

## Fixed illustrations

Figure 2 retains `fresh_000`, zero-based origin 20. All six models share its observed history and truth. The state panels use physical coordinates; time is `h/30` in system units. Its aggregate error panel uses all twelve trajectories, not just this example.

Figure 4 uses `fresh_003`, zero-based origin 43 (window index 24 among origins 19–900). Selection is post-hoc: using only DLM–K's absolute step-60 error, select the trajectory nearest the median across twelve trajectory-level errors, then the window nearest the median across that trajectory's 882 window errors. Medians use linear interpolation. Equal-distance candidates within `rtol=1e-12, atol=1e-15` are resolved by the lowest trajectory index, then the earliest origin. No baseline advantage or visual separation enters the rule. All six models use the same chosen trajectory, origin, truth and future inputs. This is an actual rollout, not a pointwise median curve or a median example for every model/state/step.

`selection/` retains the rule, exact result and all candidate scores. `data/selected_prescribed_inputs.csv` contains the sixty applied future inputs in kJ/h; the first row is the input at origin 43. Figure 4's 3,240 rows equal six models × nine state coordinates × sixty steps. State order is xA1, xB1, T1, xA2, xB2, T2, xA3, xB3, T3. Compositions are dimensionless, temperatures are K, and forecast time is `h*0.005` hours. Neither figure invents a step-zero prediction.

## Reproduction without training or inference

The plotting tools need only NumPy, pandas, Matplotlib and Pillow, not PyTorch or this repository's model installation. A separate plotting environment is recommended; do not change the experiment environment to reproduce graphics. `requirements-figures.txt` records the versions used for exact PNG-pixel checks.

From the repository root:

```sh
python manuscript_assets/h60_independent_test/scripts/verify_data.py
python manuscript_assets/h60_independent_test/scripts/render_figures.py --check-only
python manuscript_assets/h60_independent_test/scripts/render_figures.py --output-dir outputs/manuscript-figures
```

The output directory must be new. No archived file is overwritten. The check-only command compares PNG pixels and LaTeX table text with the accepted files; pixel-identical rendering requires the recorded Matplotlib/font versions. STIX fonts are supplied by Matplotlib. The PDF/SVG/PNG data, dimensions, styles and labels reproduce the accepted graphics; PDF metadata timestamps and SVG object identifiers may differ on re-export.

To additionally compare against the source raw predictions, use `verify_data.py --source-root .` with a checkout containing the source files. `SOURCE_SNAPSHOT_MAP.json` pins their Git-blob/SHA256 identities to the scientific source commit, so later repository changes cannot silently alter the source being checked. This verifies existing arrays only; it never trains a model, generates a trajectory or calls model inference.

`ARCHIVE_MANIFEST.json` records hashes of every published asset. `VERIFICATION.json` records source regression and rendering acceptance. The publication preserves later repository updates, including the separate Figure 5 archive, while keeping these items tied to their stated source commit. License selection follows the repository's existing `LICENSE_STATUS.md`; this archive grants no additional license.
