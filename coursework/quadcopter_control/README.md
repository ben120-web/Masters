# Quadcopter control and estimation

This coursework explores altitude control for a discrete-time quadcopter model:

1. estimate the state with a Kalman filter;
2. form lifted prediction matrices for a finite-horizon controller;
3. solve a quadratic tracking objective and apply only the first control input;
4. repeat after the next state estimate arrives.

[`model_predictive_controller.py`](model_predictive_controller.py) contains a
tested, reusable implementation of the unconstrained linear MPC formulation.
The other scripts are retained as historical simulations and should be read as
coursework rather than maintained production modules.
