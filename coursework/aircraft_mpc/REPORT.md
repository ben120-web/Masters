# Aircraft pitch MPC — technical report

## Objective

The controller regulates the longitudinal state

```text
x = [alpha, q, theta]ᵀ
```

where `alpha` is angle of attack, `q` is pitch rate and `theta` is pitch angle.
Elevator deflection `delta` is the sole control input. All optimisation values
are represented in radians; degrees are used below only for readability.

## Plant model

The supplied discrete linear model is

```text
    [ 0.983500   2.782  0 ]        [0.012930]
A = [-0.0006821  0.978  0 ],   B = [0.001000],
    [-0.0009730  2.804  1 ]        [0.001425]

x(k+1) = A x(k) + B delta(k).
```

Repeated substitution over horizon `N` produces

```text
X = Pₓ x(k) + Pᵤ U,
```

where `X` stacks the next `N` states and `U` stacks the `N` elevator moves.
The implementation verifies this lifted model against direct propagation in an
automated test.

## Cost and constraints

For the regulation experiment, `N = 20` and

```text
Q = diag(50, 10, 1000),   R = 0.05.
```

The pitch-angle weight is deliberately dominant because `theta` is an
integrating state in this model. The convex quadratic objective is

```text
J(U) = sum(xᵢᵀ Q xᵢ + deltaᵢᵀ R deltaᵢ).
```

Every predicted state is subject to

```text
|alpha|         <= 11.5°
|q|             <= 14° per sample
|theta|         <= 35°
|theta - alpha| <= 23°
```

and every input satisfies

```text
-24° <= delta <= 27°.
```

The absolute-value conditions are expanded into pairs of linear inequalities.
After substituting the lifted dynamics, the online constraint has the standard
QP form

```text
(I_N kron Gₓ) Pᵤ U <= 1_N kron hₓ - (I_N kron Gₓ) Pₓ x(k).
```

SciPy SLSQP solves the small dense convex problem with an analytical gradient.
The clipped unconstrained optimum starts the first solve; subsequent solves
shift the preceding optimal sequence to provide a receding-horizon warm start.
The implementation rejects an infeasible current state and raises an explicit
error if the optimiser fails or returns a constraint violation above `1e-7`.

## Reproducible result

Command:

```bash
python -m coursework.aircraft_mpc.MPC_Controller
```

Scenario: regulation from `[8°, 0°, 12°]` to the origin for 150 samples.

| Measure | Result |
| --- | ---: |
| Final angle of attack | -1.862382° |
| Final pitch rate | -0.000233° per sample |
| Final pitch angle | 0.089771° |
| State-norm reduction | 87.072% |
| Peak absolute elevator demand | 27.000000° |
| Maximum constraint violation | 0 rad |
| Mean SLSQP iterations per sample | 15.887 |

The elevator reaches its positive limit, demonstrating that actuator bounds are
part of the optimisation rather than clipped after the solve. The predicted and
realised state histories remain within the specified envelope.

## Verification

`pytest -q tests/test_aircraft_mpc.py` checks:

- lifted predictions against direct recursive dynamics;
- the full state and elevator constraint set on an optimal horizon;
- closed-loop state-norm reduction without violations; and
- explicit rejection of an initial state outside the flight envelope.

## Limitations and next steps

This is an educational controller around the supplied linear model. The source
material does not document the sample interval or aerodynamic operating point,
so pitch-rate units are described per sample and no continuous-time claim is
made. The simulation excludes disturbances, sensor noise, estimator dynamics,
model uncertainty, elevator slew rate and actuator lag. SLSQP is suitable for a
reproducible numerical study, not a flight-certified real-time implementation.

A stronger continuation would identify the operating-point metadata, add a
Kalman estimator and disturbance model, use a deterministic QP solver, compute
a terminal invariant set, analyse recursive feasibility, and validate timing
and robustness in hardware-in-the-loop tests.
