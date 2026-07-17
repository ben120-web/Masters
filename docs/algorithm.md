# Algorithm design

Let the observed window be `x = s + n`, where `s` is the clean ECG and `n` is
electrode-motion artefact. The network learns the residual noise estimate
`n_hat = f_theta(x)` and returns:

`s_hat = x - f_theta(x)`

Residual prediction gives the model an identity path and makes the desired
behaviour explicit: preserve morphology that is not supported as artefact.

## Reference network

The maintained model uses a 1-D convolutional encoder, configurable residual
blocks with batch normalisation and GELU activation, and a convolutional noise
head. Odd kernels and symmetric padding preserve temporal length. AdamW,
gradient clipping and validation-loss early stopping provide a compact,
repeatable baseline.

Training minimises sample-weighted mean squared error:

`L(theta) = (1 / N) * sum((s_hat_i - s_i)^2)`

Loss aggregation is weighted by the number of signal elements, so a short final
batch cannot bias the reported epoch or validation loss. The best checkpoint is
deep-copied to CPU storage before subsequent optimisation steps.

## Evaluation

Subjects are assigned wholly to train, validation or test before windows are
generated. The pipeline reports RMSE, normalised correlation and SNR
improvement. These metrics quantify signal fidelity but do not establish
diagnostic preservation; external validation and morphology-specific endpoints
are required before any clinical claim.

The default synthetic experiment exists to exercise the system. It should be
compared with identity and signal-processing baselines before model promotion,
particularly at high input SNR where unnecessary denoising can be harmful.

## Finite-horizon model-predictive control

The maintained [linear MPC implementation](../coursework/quadcopter_control/model_predictive_controller.py)
builds the lifted prediction model

`Y = O x_k + M U`

and solves the unconstrained quadratic tracking objective

`min_U (R - O x_k - M U)^T Q (R - O x_k - M U) + U^T R_u U`.

The constant gain is formed with a linear solve rather than an explicit matrix
inverse. If `n_u` is the input dimension and `N_u` the control horizon, offline
factorisation is cubic in `N_u n_u`; each online update is a matrix-vector
product in the lifted output and control dimensions. Only the first optimal
input is applied before the problem is updated, giving the receding-horizon
feedback law. Tests exercise closed-loop directionality and invalid dimensions.
