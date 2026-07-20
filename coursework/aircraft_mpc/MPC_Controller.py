"""Run the reproducible aircraft-pitch MPC demonstration.

The capitalised filename is retained for compatibility with the original
coursework repository. New code should import :mod:`coursework.aircraft_mpc`.
"""

from __future__ import annotations

import json

import numpy as np

from coursework.aircraft_mpc import default_controller


def run_simulation(*, steps: int = 150) -> dict[str, object]:
    """Regulate a feasible pitch state and return report-ready metrics."""
    controller = default_controller()
    initial_state = np.deg2rad(np.array([8.0, 0.0, 12.0]))
    result = controller.simulate(initial_state, steps=steps)
    final_degrees = np.rad2deg(result.states[-1])
    return {
        "steps": steps,
        "initial_state_degrees": np.rad2deg(initial_state).round(6).tolist(),
        "final_state_degrees": final_degrees.round(6).tolist(),
        "state_norm_reduction_percent": round(
            100.0
            * (1.0 - np.linalg.norm(final_degrees) / np.linalg.norm(np.rad2deg(initial_state))),
            3,
        ),
        "maximum_absolute_elevator_degrees": round(
            float(np.max(np.abs(np.rad2deg(result.inputs)))), 6
        ),
        "maximum_constraint_violation_radians": result.maximum_constraint_violation,
        "mean_solver_iterations": round(float(np.mean(result.solver_iterations)), 3),
    }


if __name__ == "__main__":
    print(json.dumps(run_simulation(), indent=2))
