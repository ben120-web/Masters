"""Reproducible bounded-altitude MPC demonstration."""

from __future__ import annotations

import json

import numpy as np

from .model_predictive_controller import ModelPredictiveControl


def altitude_controller(*, sample_time: float = 0.1) -> ModelPredictiveControl:
    """Build a double-integrator altitude controller with acceleration limits."""
    state_matrix = np.array([[1.0, sample_time], [0.0, 1.0]])
    input_matrix = np.array([[0.5 * sample_time**2], [sample_time]])
    output_matrix = np.array([[1.0, 0.0]])
    prediction_horizon = 50
    control_horizon = 15
    return ModelPredictiveControl(
        state_matrix,
        input_matrix,
        output_matrix,
        prediction_horizon,
        control_horizon,
        control_weight=np.eye(control_horizon) * 0.1,
        output_weight=np.eye(prediction_horizon),
        input_bounds=(-3.0, 3.0),
    )


def run_simulation(*, steps: int = 200, target_altitude: float = 40.0) -> dict[str, float]:
    """Run the report scenario and return deterministic performance metrics."""
    sample_time = 0.1
    controller = altitude_controller(sample_time=sample_time)
    result = controller.simulate_closed_loop([0.0, 0.0], [target_altitude], steps=steps)
    altitude = result.states[:, 0]
    error = np.abs(altitude - target_altitude)
    tolerance = 0.02 * abs(target_altitude)
    settling_candidates = [
        index for index in range(error.size) if np.all(error[index:] <= tolerance)
    ]
    settling_time = settling_candidates[0] * sample_time if settling_candidates else float("nan")
    overshoot = max(0.0, float(np.max(altitude) - target_altitude))
    return {
        "target_altitude_metres": target_altitude,
        "final_altitude_metres": round(float(altitude[-1]), 6),
        "settling_time_seconds_2_percent": round(settling_time, 3),
        "overshoot_percent": round(100.0 * overshoot / abs(target_altitude), 3),
        "maximum_absolute_acceleration_metres_per_second_squared": round(
            float(np.max(np.abs(result.inputs))), 6
        ),
        "objective_reduction_percent": round(
            100.0 * (1.0 - result.objectives[-1] / result.objectives[0]), 6
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_simulation(), indent=2))
