# Quadcopter altitude MPC — technical report

## Aim

The exercise combines a constant-acceleration vertical plant, state estimation
and receding-horizon control. The maintained experiment isolates the controller
so its prediction model, optimiser and closed-loop behaviour can be tested
deterministically. The original scripts retain the Kalman-filter and PID work
used during development.

## Discrete model

With altitude `z`, vertical velocity `v`, commanded vertical acceleration `u`
and sample time `T = 0.1 s`, the plant is

```text
[z(k+1)]   [1  T] [z(k)]   [T²/2] u(k)
[v(k+1)] = [0  1] [v(k)] + [  T ]

y(k) = [1  0] x(k).
```

The demonstration constrains `u` to `[-3, 3] m/s²`. This is a deliberately
compact vertical-axis model: attitude dynamics, motor lag, thrust-to-mass
uncertainty and gravity-feedforward error are outside its scope.

## Finite-horizon controller

For prediction horizon `N = 50` and control horizon `Nᵤ = 15`, repeated
substitution gives

```text
Y = O x(k) + M U.
```

The controller minimises

```text
J(U) = (R - O x(k) - M U)ᵀ Q (R - O x(k) - M U) + Uᵀ Rᵤ U,
```

with `Q = I₅₀` and `Rᵤ = 0.1 I₁₅`. Without bounds, the optimum is obtained by
solving the symmetric linear system

```text
(Mᵀ Q M + Rᵤ) U = Mᵀ Q (R - O x(k)).
```

No explicit matrix inverse is formed. With actuator bounds, the same convex
quadratic problem is solved by projected gradient descent. The step size is the
reciprocal of the Hessian's largest eigenvalue, and convergence is measured by
the infinity norm of the projected update. Only the first input is applied;
the state is then measured or estimated and the horizon is solved again.

## Reproducible result

Command:

```bash
python -m coursework.quadcopter_control.demo
```

Scenario: zero initial altitude and velocity, 40 m step, 20 s simulation.

| Measure | Result |
| --- | ---: |
| Final altitude | 40.000000 m |
| 2% settling time | 7.0 s |
| Peak overshoot | 0.567% |
| Peak commanded acceleration | 3.0 m/s² |
| Objective reduction | >99.999% |

These are deterministic simulation results for the stated linear model, not
flight-test claims.

## Verification

Automated tests check that:

- the controller moves an integrator in the correct tracking direction;
- scalar references are expanded consistently across the horizon;
- every bounded input remains inside the specified actuator interval;
- the 40 m closed loop reaches its target with the ±3 m/s² bound active; and
- invalid horizons, dimensions, weights and runtime vectors fail explicitly.

Run them with `pytest -q tests/test_mpc.py`.

## Engineering assessment

The lifted matrices are built once. For `nᵤ` inputs and control horizon `Nᵤ`,
the unconstrained factorisation is cubic in `Nᵤ nᵤ`; each feedback update is a
matrix-vector product. The bounded solver additionally iterates over the same
small dense Hessian. Production flight control would require a real-time QP
solver with deadline monitoring, disturbance modelling, state-estimator
integration, explicit feasibility handling, actuator rate constraints and
hardware-in-the-loop validation.
