# Constrained MPC for aircraft pitch

This module implements the aircraft-pitch controller developed from the MSc
control assignment. It regulates angle of attack, pitch rate and pitch angle
with elevator deflection while enforcing the full flight-envelope constraints
over every predicted state.

## What is implemented

- lifted discrete dynamics for the three-state, single-input aircraft model;
- a correctly expanded quadratic state-tracking and elevator-effort cost;
- asymmetric elevator limits;
- angle-of-attack, pitch-rate, pitch-angle and `|theta - alpha|` constraints;
- warm-started receding-horizon simulation with solver diagnostics; and
- tests that compare lifted predictions with direct state propagation and
  verify all inequalities.

Run the deterministic regulation experiment:

```bash
python -m coursework.aircraft_mpc.MPC_Controller
```

From `[alpha, q, theta] = [8°, 0°, 12°]`, the 150-step scenario reduces the
state norm by 87.072%, respects every state constraint, and keeps elevator
deflection within `[-24°, 27°]`.

See the [technical report](REPORT.md) for the derivation, constraint matrices,
result table and limitations. New code should import `AircraftPitchMPC` from
`coursework.aircraft_mpc`; `MPC_Controller.py` is retained as the original-name
command-line entry point.
