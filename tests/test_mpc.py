import numpy as np
import pytest

from coursework.quadcopter_control.demo import altitude_controller
from coursework.quadcopter_control.model_predictive_controller import (
    ModelPredictiveControl,
)


def integrator_controller(*, bounded: bool = False) -> ModelPredictiveControl:
    return ModelPredictiveControl(
        state_matrix=np.array([[1.0]]),
        input_matrix=np.array([[1.0]]),
        output_matrix=np.array([[1.0]]),
        prediction_horizon=4,
        control_horizon=2,
        control_weight=np.eye(2) * 0.05,
        output_weight=np.eye(4),
        input_bounds=(-0.25, 0.25) if bounded else None,
    )


def test_mpc_drives_an_integrator_towards_reference() -> None:
    controller = integrator_controller()
    state = np.array([[0.0]])

    plan = controller.plan(state, np.array([1.0]))
    next_state = controller.propagate(state, plan.inputs[0])

    assert plan.converged
    assert plan.iterations == 1
    assert plan.inputs.shape == (2, 1)
    assert plan.outputs.shape == (4, 1)
    assert plan.inputs[0, 0] > 0
    assert 0 < next_state.item() < 1.1


def test_bounded_mpc_respects_actuator_limits() -> None:
    controller = integrator_controller(bounded=True)

    plan = controller.plan([0.0], [10.0])

    assert plan.converged
    assert np.all(plan.inputs <= 0.25 + 1e-12)
    assert np.all(plan.inputs >= -0.25 - 1e-12)
    assert plan.inputs[0, 0] == pytest.approx(0.25)


def test_altitude_demo_reaches_target_with_bounded_acceleration() -> None:
    controller = altitude_controller()

    result = controller.simulate_closed_loop([0.0, 0.0], [40.0], steps=120)

    assert result.states.shape == (121, 2)
    assert result.inputs.shape == (120, 1)
    assert result.states[-1, 0] == pytest.approx(40.0, abs=0.1)
    assert np.max(np.abs(result.inputs)) <= 3.0 + 1e-12
    assert result.objectives[-1] < result.objectives[0] * 1e-5


def test_mpc_rejects_invalid_horizon_and_weight_dimensions() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ModelPredictiveControl(
            np.eye(1),
            np.eye(1),
            np.eye(1),
            prediction_horizon=2,
            control_horizon=3,
            control_weight=np.eye(3),
            output_weight=np.eye(2),
        )

    with pytest.raises(ValueError, match="positive semidefinite"):
        ModelPredictiveControl(
            np.eye(1),
            np.eye(1),
            np.eye(1),
            prediction_horizon=2,
            control_horizon=1,
            control_weight=np.eye(1),
            output_weight=-np.eye(2),
        )


def test_mpc_rejects_bad_runtime_shapes() -> None:
    controller = integrator_controller()

    with pytest.raises(ValueError, match="reference must contain"):
        controller.plan([0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="state must contain"):
        controller.control([0.0, 1.0], [1.0])
