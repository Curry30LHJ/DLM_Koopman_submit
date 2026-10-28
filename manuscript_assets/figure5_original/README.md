# Manuscript Figure 5: original 100-condition diagnostics

This archive contains the data and the figure used in the submitted and revised manuscripts.

The figure contains two post-hoc CSTR diagnostics, each with 100 conditions and five frozen predictors: MLP-K, LSTM-K, HaKAN-K, Delay-RBF and DLM-K. The input-profile panel varies ten future-input profiles for each of ten histories. The history panel compares ten histories under each of ten fixed future-input/noise pairs. Conditions sharing histories or input/noise are not independent replicates. The historical test windows in this diagnostic are separate from the independent simulated test set used in the primary comparison. The delay buffer was reconstructed for the controlled interventions; these are not exact natural-trajectory restarts.

The supplied PDF is the Figure 5 asset in both manuscript versions. `data/` contains the archived physical predictions, frozen truth, state and action histories, applied and requested future inputs, condition manifests and condition-level metrics. The paths recorded inside the condition CSVs describe the original archive; the arrays needed for the figure are supplied in `PANEL_A_RAW.npz` and `PANEL_B_RAW.npz`. Delay-RBF predictions are in the corresponding `PANEL_*_RBF_RAW.npz` files, in the same condition order. `MODEL_NAME_MAP.json` maps the archived prediction keys to the model names. `STATE_NORMALIZATION.json` records the training-only normalization used for these results.

For condition $c$, prediction step $h$ and state $j$, the plotted endpoint error is

```
error[c,h] = sqrt(mean_j(((prediction[c,h,j] - truth[c,h,j]) / state_std[j])**2))
```

The upper panels average this error over the 100 conditions at each step. The lower panels summarize `J[c] = mean(error[c,1:60])`. The step-zero curve points are plotting origins, not measurements. For DLM-K, the mean `J` is 0.05152 for input-profile variation and 0.01141 for history variation.

Run the following command from the repository root to check the archived file hashes and recompute all condition errors and mean curves from the physical predictions:

```sh
python manuscript_assets/figure5_original/verify.py
```

This verification does not train models or regenerate trajectories. `SHA256.csv` records the checksums of the supplied data, normalization, model-name map and figure. The controlled archive is derived from the earlier `current_state_seed2_spectral_h60` diagnostic; model names in the metric CSV were standardized without changing the numerical values.
