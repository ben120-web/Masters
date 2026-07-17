"""Finite-horizon linear model-predictive control.

The controller solves the unconstrained quadratic tracking problem

    min_U (R - O x - M U)^T Q (R - O x - M U) + U^T R_u U

for a discrete-time linear system. It is retained as a compact, executable
example of the lifted-matrix MPC formulation used in the MSc control work.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ModelPredictiveControl:
    """Unconstrained linear MPC with a receding control horizon."""

    def __init__(
        self,
        state_matrix: FloatArray,
        input_matrix: FloatArray,
        output_matrix: FloatArray,
        prediction_horizon: int,
        control_horizon: int,
        control_weight: FloatArray,
        output_weight: FloatArray,
    ) -> None:
        self.a = np.asarray(state_matrix, dtype=np.float64)
        self.b = np.asarray(input_matrix, dtype=np.float64)
        self.c = np.asarray(output_matrix, dtype=np.float64)
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self._validate_dimensions(control_weight, output_weight)
        self.control_weight = np.asarray(control_weight, dtype=np.float64)
        self.output_weight = np.asarray(output_weight, dtype=np.float64)
        self.observability, self.control_map = self._lifted_matrices()
        hessian = self.control_map.T @ self.output_weight @ self.control_map
        hessian += self.control_weight
        right_hand_side = self.control_map.T @ self.output_weight
        # solve() is more stable than explicitly forming the matrix inverse.
        self.gain = np.linalg.solve(hessian, right_hand_side)

    @property
    def state_size(self) -> int:
        return self.a.shape[0]

    @property
    def input_size(self) -> int:
        return self.b.shape[1]

    @property
    def output_size(self) -> int:
        return self.c.shape[0]

    def _validate_dimensions(
        self, control_weight: FloatArray, output_weight: FloatArray
    ) -> None:
        if self.prediction_horizon < 1 or self.control_horizon < 1:
            raise ValueError("MPC horizons must be positive")
        if self.control_horizon > self.prediction_horizon:
            raise ValueError("control_horizon cannot exceed prediction_horizon")
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            raise ValueError("state_matrix must be square")
        if self.b.shape[0] != self.a.shape[0]:
            raise ValueError("input_matrix row count must match the state size")
        if self.c.shape[1] != self.a.shape[0]:
            raise ValueError("output_matrix column count must match the state size")
        expected_control = self.control_horizon * self.b.shape[1]
        expected_output = self.prediction_horizon * self.c.shape[0]
        if np.shape(control_weight) != (expected_control, expected_control):
            raise ValueError("control_weight has the wrong shape")
        if np.shape(output_weight) != (expected_output, expected_output):
            raise ValueError("output_weight has the wrong shape")

    def _lifted_matrices(self) -> tuple[FloatArray, FloatArray]:
        observability = np.zeros(
            (self.prediction_horizon * self.output_size, self.state_size)
        )
        control_map = np.zeros(
            (
                self.prediction_horizon * self.output_size,
                self.control_horizon * self.input_size,
            )
        )
        for prediction_step in range(self.prediction_horizon):
            row = slice(
                prediction_step * self.output_size,
                (prediction_step + 1) * self.output_size,
            )
            observability[row] = self.c @ np.linalg.matrix_power(
                self.a, prediction_step + 1
            )
            for control_step in range(min(prediction_step + 1, self.control_horizon)):
                column = slice(
                    control_step * self.input_size,
                    (control_step + 1) * self.input_size,
                )
                power = prediction_step - control_step
                control_map[row, column] = (
                    self.c @ np.linalg.matrix_power(self.a, power) @ self.b
                )
        return observability, control_map

    def control_sequence(self, state: FloatArray, reference: FloatArray) -> FloatArray:
        """Return the optimal open-loop sequence for the current state."""
        state_vector = np.asarray(state, dtype=np.float64).reshape(self.state_size, 1)
        reference_vector = np.asarray(reference, dtype=np.float64).reshape(
            self.prediction_horizon * self.output_size, 1
        )
        tracking_error = reference_vector - self.observability @ state_vector
        return self.gain @ tracking_error

    def control(self, state: FloatArray, reference: FloatArray) -> FloatArray:
        """Return the first receding-horizon input to apply to the plant."""
        return self.control_sequence(state, reference)[: self.input_size]

    def propagate(self, state: FloatArray, control: FloatArray) -> FloatArray:
        state_vector = np.asarray(state, dtype=np.float64).reshape(self.state_size, 1)
        control_vector = np.asarray(control, dtype=np.float64).reshape(self.input_size, 1)
        return self.a @ state_vector + self.b @ control_vector
