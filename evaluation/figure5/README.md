# Additional 84-condition holdout experiment

This directory contains an additional controlled test using 12 existing fresh-holdout histories and seven input profiles per panel. It is not the data source of the manuscript's Figure 5. The original figure and its 100-condition diagnostics are archived at [manuscript_assets/figure5_original](../../manuscript_assets/figure5_original/).

This extension uses all six predictors, including Markov RBF, and tests input and history variation on the same holdout used for the primary evaluation. It is not a second independent confirmation sample. `REPORT.md`, `PROTOCOL.json`, the predictions and the existing verification script describe the frozen 84-condition experiment.

Two source files were edited only in comments or docstrings after the protocol was frozen. The verification script reports and accepts these two exact documented hash changes; it still checks every other frozen input against the protocol. Replaying the original experiment with its strict source hashes requires the code at commit `0e02e3170d23b820490554ad965ea2341255fe3c`. The stored protocol and predictions have not been changed.

```sh
python scripts/verify_figure5.py
```
