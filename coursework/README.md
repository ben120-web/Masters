# MSc engineering modules

The modules in this directory showcase the breadth of the MSc alongside the
maintained ECG denoising dissertation project. They preserve the original
analysis while adding clear entry points and verification where the work is
actively executable.

| Module | Work demonstrated | Evidence |
| --- | --- | --- |
| [Quadcopter control and estimation](quadcopter_control/) | Kalman state estimation, PID comparison, lifted linear MPC, bounded optimisation and closed-loop analysis | [Technical report](quadcopter_control/REPORT.md) · [tested controller](quadcopter_control/model_predictive_controller.py) |
| [Aircraft pitch MPC](aircraft_mpc/) | Constrained quadratic control, flight-envelope inequalities, lifted dynamics and receding-horizon simulation | [Technical report](aircraft_mpc/REPORT.md) · [controller](aircraft_mpc/controller.py) |
| [Control and estimation](control_and_estimation/) | Bayesian estimation, finite-horizon LQ optimal control, reachability, Lyapunov analysis and ellipsoidal methods | LaTeX derivations and Python numerical work |
| [Digital signal processing](digital_signal_processing/) | Spectral inspection, mains-frequency cancellation, adaptive notch filtering and moving-median high-pass filtering | MATLAB implementations, test signals and result figures |
| [Intelligent systems and control](intelligent_systems_and_control/) | AR/ARX modelling, forward feature selection, linear and multilayer-perceptron forecasting | MATLAB experiments, data and result figures |
| [Kalman filtering](kalman_filter/) | Recursive state prediction and measurement correction experiments | Python implementations |
| [Wireless sensors](wireless_sensors/) | Environmental sensor distributions, correlation and pollutant regression analysis | R analysis scripts |

## Reproduce the maintained control demonstrations

Install the repository development environment, then run:

```bash
python -m coursework.quadcopter_control.demo
python -m coursework.aircraft_mpc.MPC_Controller
pytest -q tests/test_mpc.py tests/test_aircraft_mpc.py
```

The reports distinguish deterministic model results from physical-system
claims and record the assumptions needed to interpret each result.
