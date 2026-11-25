# DLM-Koopman

Model and results for *A delay-aware latent-memory Koopman-structured model for nonlinear system identification and prediction*.

We tested DLM-Koopman on delayed Lorenz–63 and a reactor–separator with a delayed recycle stream. The model encodes an observed history once, then uses recent latent states to correct each forecast step. [Model source](src/koopman_delay/models/dlm.py)

## Results

At step 60, DLM-Koopman’s mean prediction error was **0.7133** on Lorenz–63 and **0.006058** on the reactor–separator. Each system had twelve independent simulated test trajectories. For each trajectory, the error is the root mean square across forecast origins and state coordinates after scaling each coordinate by its training standard deviation. The tables give the mean across trajectories.

| Experiment | Reported comparison | Data and figures |
| --- | --- | --- |
| Independent forecasts | Six predictors at steps 1, 5, 10, 20, 30 and 60 | [Errors](results/forecast_errors.csv) · [Figure 2, Lorenz](figures/figure_2_lorenz.pdf) · [Figure 4, reactor](figures/figure_4_reactor.pdf) |
| Input and history variation | Five predictors under 100 reactor conditions per panel | [Figure 5](figures/figure_5_input_and_history.pdf) |
| Recycle delay and history length | Three recycle delays (0.010, 0.025 and 0.050 h) and three observation histories (6, 20 and 40 samples) | [Errors at steps 30 and 60](results/reactor_sensitivity.csv) · [Figure 6](figures/figure_6_recycle_delay.pdf) · [Figure 7](figures/figure_7_history_length.pdf) |
| Measurement noise | Step-60 error with noise added to the observed history at 0%, 1%, 5% and 10% of each state’s training standard deviation | [Errors](results/measurement_noise.csv) · [Noise figure](figures/measurement_noise.pdf) |

The two sample forecasts follow Lorenz state `x1` in Figure 2 and reactor temperature `T1` (K) in Figure 4. Each pairs 60 simulated reference values with the corresponding DLM-Koopman predictions. [Sample forecasts](samples/forecast_examples.csv)

[License status](LICENSE_STATUS.md)
