import numpy as np
import pytest

from coursework.aircraft_mpc import AIRCRAFT_A, AIRCRAFT_B, default_controller


def test_aircraft_lifted_model_matches_direct_propagation() -> None:
    controller = default_controller(prediction_horizon=5)
    state = np.deg2rad(np.array([2.0, 0.25, 4.0]))
    inputs = np.deg2rad(np.array([-2.0, 1.0, 0.5, -0.25, 0.0]))

    lifted = (controller.state_map @ state + controller.input_map @ inputs).reshape(5, 3)
    direct = []
    current = state
    for control in inputs:
        current = AIRCRAFT_A @ current + AIRCRAFT_B[:, 0] * control
        direct.append(current)

    np.testing.assert_allclose(lifted, direct, atol=1e-12)


def test_aircraft_mpc_enforces_full_envelope() -> None:
    controller = default_controller(prediction_horizon=12)
    state = np.deg2rad(np.array([8.0, 0.0, 12.0]))

    solution = controller.solve(state)

    assert solution.inputs.shape == (12, 1)
    assert solution.states.shape == (12, 3)
    assert solution.maximum_constraint_violation <= 1e-9
    assert np.all(solution.inputs >= np.deg2rad(-24.0) - 1e-9)
    assert np.all(solution.inputs <= np.deg2rad(27.0) + 1e-9)
    assert np.max(controller.constraints.state_violation(solution.states)) <= 1e-9


def test_aircraft_closed_loop_reduces_state_norm_without_violations() -> None:
    controller = default_controller(prediction_horizon=12)
    initial_state = np.deg2rad(np.array([8.0, 0.0, 12.0]))

    result = controller.simulate(initial_state, steps=60)

    assert result.maximum_constraint_violation <= 1e-8
    assert np.linalg.norm(result.states[-1]) < 0.4 * np.linalg.norm(initial_state)
    assert result.objectives[-1] < result.objectives[0]


def test_aircraft_mpc_rejects_state_outside_flight_envelope() -> None:
    controller = default_controller(prediction_horizon=5)

    with pytest.raises(ValueError, match="violates the aircraft envelope"):
        controller.solve(np.deg2rad(np.array([12.0, 0.0, 0.0])))
