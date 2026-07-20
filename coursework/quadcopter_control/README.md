# Quadcopter altitude control and state estimation

This module demonstrates the control path developed for a discrete-time
quadcopter altitude model: estimate altitude and vertical velocity, predict the
plant response over a finite horizon, optimise the acceleration command, and
apply the first move as receding-horizon feedback.

## Maintained implementation

[`model_predictive_controller.py`](model_predictive_controller.py) provides a
reusable NumPy implementation with:

- validated MIMO system and weight dimensions;
- lifted prediction matrices for distinct prediction and control horizons;
- exact unconstrained control and bounded convex-QP control;
- actuator bounds enforced over the complete control sequence;
- predicted-output, objective and convergence diagnostics; and
- deterministic closed-loop simulation.

Run the documented 40 m altitude step:

```bash
python -m coursework.quadcopter_control.demo
```

The reference scenario settles within the 2% band in 7.0 s, limits commanded
vertical acceleration to ±3.0 m/s², and has 0.567% overshoot. See the
[technical report](REPORT.md) for the formulation, verification evidence and
limitations.

The original Kalman-filter and PID comparison scripts remain alongside the
maintained MPC implementation to show the progression of the coursework. They
are exploratory scripts; the tested API and reproducible result above are the
recommended entry points.
