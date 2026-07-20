"""Finite-horizon linear model-predictive control.

The controller builds the lifted prediction model ``Y = O x + M U`` and solves
the quadratic tracking problem

``min_U (R - O x - M U)^T Q (R - O x - M U) + U^T R_u U``.

The unconstrained problem is solved once as a linear system. Optional actuator
bounds are handled with a dependency-free projected-gradient QP solver, keeping
this coursework example executable anywhere NumPy is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControlPlan:
    """One finite-horizon optimisation result."""

    inputs: FloatArray
    outputs: FloatArray
    objective: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ClosedLoopResult:
    """State, output and actuator histories from receding-horizon control."""

    states: FloatArray
    outputs: FloatArray
    inputs: FloatArray
    objectives: FloatArray


class ModelPredictiveControl:
    """Linear MPC with a receding control horizon and optional input bounds.

    Parameters use the discrete-time system convention ``x+ = A x + B u`` and
    ``y = C x``. ``control_weight`` and ``output_weight`` are the already-lifted
    quadratic weights, which allows individual horizon steps to be tuned.
    """

    def __init__(
        self,
        state_matrix: ArrayLike,
        input_matrix: ArrayLike,
        output_matrix: ArrayLike,
        prediction_horizon: int,
        control_horizon: int,
        control_weight: ArrayLike,
        output_weight: ArrayLike,
        *,
        input_bounds: tuple[ArrayLike, ArrayLike] | None = None,
        solver_tolerance: float = 1e-9,
        solver_max_iterations: int = 10_000,
    ) -> None:
        self.a = np.asarray(state_matrix, dtype=np.float64)
        self.b = np.asarray(input_matrix, dtype=np.float64)
        self.c = np.asarray(output_matrix, dtype=np.float64)
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.control_weight = np.asarray(control_weight, dtype=np.float64)
        self.output_weight = np.asarray(output_weight, dtype=np.float64)
        self.solver_tolerance = float(solver_tolerance)
        self.solver_max_iterations = solver_max_iterations

        self._validate_dimensions()
        self._validate_weights()
        self.lower_input, self.upper_input = self._prepare_bounds(input_bounds)
        self.observability, self.control_map = self._lifted_matrices()
        self.hessian = self.control_map.T @ self.output_weight @ self.control_map
        self.hessian += self.control_weight
        self._validate_hessian()
        self._right_hand_map = self.control_map.T @ self.output_weight
        # This exact gain is used by the fast unconstrained path and as the
        # initial point for the bounded QP.
        self.gain = np.linalg.solve(self.hessian, self._right_hand_map)

    @property
    def state_size(self) -> int:
        return self.a.shape[0]

    @property
    def input_size(self) -> int:
        return self.b.shape[1]

    @property
    def output_size(self) -> int:
        return self.c.shape[0]

    @property
    def is_bounded(self) -> bool:
        return self.lower_input is not None

    def _validate_dimensions(self) -> None:
        if isinstance(self.prediction_horizon, bool) or self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be a positive integer")
        if isinstance(self.control_horizon, bool) or self.control_horizon < 1:
            raise ValueError("control_horizon must be a positive integer")
        if not isinstance(self.prediction_horizon, int) or not isinstance(
            self.control_horizon, int
        ):
            raise ValueError("MPC horizons must be integers")
        if self.control_horizon > self.prediction_horizon:
            raise ValueError("control_horizon cannot exceed prediction_horizon")
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            raise ValueError("state_matrix must be square")
        if self.b.ndim != 2 or self.b.shape[0] != self.a.shape[0]:
            raise ValueError("input_matrix row count must match the state size")
        if self.c.ndim != 2 or self.c.shape[1] != self.a.shape[0]:
            raise ValueError("output_matrix column count must match the state size")
        if not all(np.all(np.isfinite(matrix)) for matrix in (self.a, self.b, self.c)):
            raise ValueError("system matrices must contain only finite values")
        if self.solver_tolerance <= 0 or not np.isfinite(self.solver_tolerance):
            raise ValueError("solver_tolerance must be finite and positive")
        if (
            isinstance(self.solver_max_iterations, bool)
            or not isinstance(self.solver_max_iterations, int)
            or self.solver_max_iterations < 1
        ):
            raise ValueError("solver_max_iterations must be a positive integer")

    def _validate_weights(self) -> None:
        expected_control = self.control_horizon * self.input_size
        expected_output = self.prediction_horizon * self.output_size
        if self.control_weight.shape != (expected_control, expected_control):
            raise ValueError("control_weight has the wrong shape")
        if self.output_weight.shape != (expected_output, expected_output):
            raise ValueError("output_weight has the wrong shape")
        for name, weight in (
            ("control_weight", self.control_weight),
            ("output_weight", self.output_weight),
        ):
            if not np.all(np.isfinite(weight)):
                raise ValueError(f"{name} must contain only finite values")
            if not np.allclose(weight, weight.T, atol=1e-12):
                raise ValueError(f"{name} must be symmetric")
            if np.min(np.linalg.eigvalsh(weight)) < -1e-12:
                raise ValueError(f"{name} must be positive semidefinite")

    def _validate_hessian(self) -> None:
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(self.hessian)))
        if minimum_eigenvalue <= 0:
            raise ValueError(
                "the combined MPC cost must be positive definite; increase control_weight"
            )

    def _prepare_bounds(
        self, bounds: tuple[ArrayLike, ArrayLike] | None
    ) -> tuple[FloatArray | None, FloatArray | None]:
        if bounds is None:
            return None, None
        lower = self._expand_input_bound(bounds[0], "lower input bound")
        upper = self._expand_input_bound(bounds[1], "upper input bound")
        if np.any(lower > upper):
            raise ValueError("lower input bounds cannot exceed upper input bounds")
        return lower, upper

    def _expand_input_bound(self, value: ArrayLike, name: str) -> FloatArray:
        bound = np.asarray(value, dtype=np.float64).reshape(-1)
        sequence_size = self.control_horizon * self.input_size
        if bound.size == 1:
            bound = np.full(sequence_size, bound.item())
        elif bound.size == self.input_size:
            bound = np.tile(bound, self.control_horizon)
        elif bound.size != sequence_size:
            raise ValueError(f"{name} must be scalar, one value per input, or one per control move")
        if np.any(np.isnan(bound)):
            raise ValueError(f"{name} cannot contain NaN")
        return bound

    def _lifted_matrices(self) -> tuple[FloatArray, FloatArray]:
        observability = np.zeros((self.prediction_horizon * self.output_size, self.state_size))
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
            observability[row] = self.c @ np.linalg.matrix_power(self.a, prediction_step + 1)
            for control_step in range(min(prediction_step + 1, self.control_horizon)):
                column = slice(
                    control_step * self.input_size,
                    (control_step + 1) * self.input_size,
                )
                power = prediction_step - control_step
                control_map[row, column] = self.c @ np.linalg.matrix_power(self.a, power) @ self.b
        return observability, control_map

    def _state_vector(self, state: ArrayLike) -> FloatArray:
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector.size != self.state_size:
            raise ValueError(f"state must contain {self.state_size} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("state must contain only finite values")
        return vector.reshape(-1, 1)

    def _reference_vector(self, reference: ArrayLike) -> FloatArray:
        vector = np.asarray(reference, dtype=np.float64).reshape(-1)
        expected = self.prediction_horizon * self.output_size
        if vector.size == self.output_size:
            vector = np.tile(vector, self.prediction_horizon)
        elif vector.size != expected:
            raise ValueError(f"reference must contain {self.output_size} or {expected} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("reference must contain only finite values")
        return vector.reshape(-1, 1)

    def plan(self, state: ArrayLike, reference: ArrayLike) -> ControlPlan:
        """Optimise a control sequence and return predictions and diagnostics."""
        state_vector = self._state_vector(state)
        reference_vector = self._reference_vector(reference)
        tracking_error = reference_vector - self.observability @ state_vector
        right_hand_side = self._right_hand_map @ tracking_error

        inputs: FloatArray
        iterations: int
        converged: bool
        if not self.is_bounded:
            inputs = np.asarray(self.gain @ tracking_error, dtype=np.float64)
            iterations = 1
            converged = True
        else:
            inputs, iterations, converged = self._solve_box_qp(right_hand_side)

        outputs = self.observability @ state_vector + self.control_map @ inputs
        residual = reference_vector - outputs
        objective = float(
            (residual.T @ self.output_weight @ residual).item()
            + (inputs.T @ self.control_weight @ inputs).item()
        )
        return ControlPlan(
            inputs=inputs,
            outputs=outputs,
            objective=objective,
            iterations=iterations,
            converged=converged,
        )

    def _solve_box_qp(self, right_hand_side: FloatArray) -> tuple[FloatArray, int, bool]:
        assert self.lower_input is not None and self.upper_input is not None
        lower = self.lower_input.reshape(-1, 1)
        upper = self.upper_input.reshape(-1, 1)
        inputs = np.clip(np.linalg.solve(self.hessian, right_hand_side), lower, upper)
        step_size = 1.0 / float(np.max(np.linalg.eigvalsh(self.hessian)))

        for iteration in range(1, self.solver_max_iterations + 1):
            gradient = self.hessian @ inputs - right_hand_side
            updated = np.clip(inputs - step_size * gradient, lower, upper)
            change = float(np.linalg.norm(updated - inputs, ord=np.inf))
            inputs = updated
            if change <= self.solver_tolerance * (1.0 + float(np.linalg.norm(inputs, ord=np.inf))):
                return inputs, iteration, True
        return inputs, self.solver_max_iterations, False

    def control_sequence(self, state: ArrayLike, reference: ArrayLike) -> FloatArray:
        """Return the optimal open-loop sequence for the current state."""
        return self.plan(state, reference).inputs

    def control(self, state: ArrayLike, reference: ArrayLike) -> FloatArray:
        """Return the first receding-horizon input to apply to the plant."""
        return self.control_sequence(state, reference)[: self.input_size]

    def propagate(self, state: ArrayLike, control: ArrayLike) -> FloatArray:
        state_vector = self._state_vector(state)
        control_vector = np.asarray(control, dtype=np.float64).reshape(-1)
        if control_vector.size != self.input_size:
            raise ValueError(f"control must contain {self.input_size} values")
        if not np.all(np.isfinite(control_vector)):
            raise ValueError("control must contain only finite values")
        return self.a @ state_vector + self.b @ control_vector.reshape(-1, 1)

    def simulate_closed_loop(
        self, initial_state: ArrayLike, reference: ArrayLike, *, steps: int
    ) -> ClosedLoopResult:
        """Run a deterministic receding-horizon simulation."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = self._state_vector(initial_state)
        states = [state.ravel()]
        outputs: list[FloatArray] = []
        inputs: list[FloatArray] = []
        objectives: list[float] = []

        for _ in range(steps):
            plan = self.plan(state, reference)
            if not plan.converged:
                raise RuntimeError("bounded MPC solver did not converge")
            control = plan.inputs[: self.input_size]
            outputs.append((self.c @ state).ravel())
            inputs.append(control.ravel())
            objectives.append(plan.objective)
            state = self.propagate(state, control)
            states.append(state.ravel())

        return ClosedLoopResult(
            states=np.asarray(states),
            outputs=np.asarray(outputs),
            inputs=np.asarray(inputs),
            objectives=np.asarray(objectives),
        )
