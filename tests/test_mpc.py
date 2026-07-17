import numpy as np
import pytest

from coursework.quadcopter_control.model_predictive_controller import (
    ModelPredictiveControl,
)


def test_mpc_drives_an_integrator_towards_reference() -> None:
    controller = ModelPredictiveControl(
        state_matrix=np.array([[1.0]]),
        input_matrix=np.array([[1.0]]),
        output_matrix=np.array([[1.0]]),
        prediction_horizon=4,
        control_horizon=2,
        control_weight=np.eye(2) * 0.05,
        output_weight=np.eye(4),
    )
    state = np.array([[0.0]])
    reference = np.ones((4, 1))

    first_control = controller.control(state, reference)
    next_state = controller.propagate(state, first_control)

    assert first_control.item() > 0
    assert 0 < next_state.item() < 1.1
    assert controller.control_sequence(state, reference).shape == (2, 1)


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
