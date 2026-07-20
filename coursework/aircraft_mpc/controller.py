"""Constrained linear MPC for the aircraft pitch coursework model.

The state is ``[angle of attack, pitch rate, pitch angle]`` and the input is
elevator deflection, all expressed in radians (or radians per sample for rate).
The finite-horizon quadratic programme enforces every constraint from the
assignment model at every predicted step.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize

FloatArray = NDArray[np.float64]

AIRCRAFT_A = np.array(
    [
        [0.9835, 2.782, 0.0],
        [-0.0006821, 0.978, 0.0],
        [-0.0009730, 2.804, 1.0],
    ],
    dtype=np.float64,
)
AIRCRAFT_B = np.array([[0.01293], [0.00100], [0.001425]], dtype=np.float64)


@dataclass(frozen=True)
class AircraftConstraints:
    """Linear state and actuator limits for the pitch model."""

    state_matrix: FloatArray
    state_limit: FloatArray
    input_lower: FloatArray
    input_upper: FloatArray

    @classmethod
    def coursework_limits(cls) -> AircraftConstraints:
        """Return the limits stated in the original coursework brief."""
        state_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        state_limit = np.deg2rad(np.array([11.5, 11.5, 14.0, 14.0, 35.0, 35.0, 23.0, 23.0]))
        return cls(
            state_matrix=state_matrix,
            state_limit=state_limit,
            input_lower=np.deg2rad(np.array([-24.0])),
            input_upper=np.deg2rad(np.array([27.0])),
        )

    def validate(self, *, state_size: int, input_size: int) -> None:
        if self.state_matrix.ndim != 2 or self.state_matrix.shape[1] != state_size:
            raise ValueError("constraint state_matrix has the wrong shape")
        if self.state_limit.shape != (self.state_matrix.shape[0],):
            raise ValueError("constraint state_limit has the wrong shape")
        if self.input_lower.shape != (input_size,) or self.input_upper.shape != (input_size,):
            raise ValueError("input constraints must contain one bound per actuator")
        arrays = (
            self.state_matrix,
            self.state_limit,
            self.input_lower,
            self.input_upper,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("constraints must contain only finite values")
        if np.any(self.input_lower > self.input_upper):
            raise ValueError("lower input bounds cannot exceed upper input bounds")

    def state_violation(self, states: ArrayLike) -> FloatArray:
        """Return positive inequality violations for one state or a state batch."""
        values = np.asarray(states, dtype=np.float64)
        values = np.atleast_2d(values)
        if values.shape[1] != self.state_matrix.shape[1]:
            raise ValueError("states have the wrong shape")
        return values @ self.state_matrix.T - self.state_limit

    def contains(self, state: ArrayLike, *, tolerance: float = 1e-9) -> bool:
        return bool(np.max(self.state_violation(state)) <= tolerance)


@dataclass(frozen=True)
class MPCSolution:
    """Optimal sequence, predicted states and solver diagnostics."""

    inputs: FloatArray
    states: FloatArray
    objective: float
    iterations: int
    maximum_constraint_violation: float


@dataclass(frozen=True)
class SimulationResult:
    """Closed-loop histories for a receding-horizon experiment."""

    states: FloatArray
    inputs: FloatArray
    objectives: FloatArray
    solver_iterations: NDArray[np.int64]
    maximum_constraint_violation: float


class AircraftPitchMPC:
    """State- and actuator-constrained finite-horizon aircraft pitch MPC."""

    def __init__(
        self,
        state_matrix: ArrayLike,
        input_matrix: ArrayLike,
        *,
        prediction_horizon: int,
        state_weight: ArrayLike,
        input_weight: ArrayLike,
        constraints: AircraftConstraints,
        terminal_weight: ArrayLike | None = None,
        solver_tolerance: float = 1e-9,
        solver_max_iterations: int = 300,
    ) -> None:
        self.a = np.asarray(state_matrix, dtype=np.float64)
        self.b = np.asarray(input_matrix, dtype=np.float64)
        self.prediction_horizon = prediction_horizon
        self.state_weight = np.asarray(state_weight, dtype=np.float64)
        self.input_weight = np.asarray(input_weight, dtype=np.float64)
        self.terminal_weight = np.asarray(
            state_weight if terminal_weight is None else terminal_weight,
            dtype=np.float64,
        )
        self.constraints = constraints
        self.solver_tolerance = float(solver_tolerance)
        self.solver_max_iterations = solver_max_iterations
        self._validate()

        self.state_map, self.input_map = self._lifted_prediction_matrices()
        self.state_cost = np.kron(np.eye(self.prediction_horizon), self.state_weight)
        terminal = slice(-self.state_size, None)
        self.state_cost[terminal, terminal] = self.terminal_weight
        self.input_cost = np.kron(np.eye(self.prediction_horizon), self.input_weight)
        self.hessian = 2.0 * (self.input_map.T @ self.state_cost @ self.input_map + self.input_cost)
        self._constraint_state_map = np.kron(
            np.eye(self.prediction_horizon), self.constraints.state_matrix
        )
        self._constraint_input_map = self._constraint_state_map @ self.input_map
        self._lifted_state_limit = np.tile(self.constraints.state_limit, self.prediction_horizon)
        self._lower_input = np.tile(self.constraints.input_lower, self.prediction_horizon)
        self._upper_input = np.tile(self.constraints.input_upper, self.prediction_horizon)

    @property
    def state_size(self) -> int:
        return self.a.shape[0]

    @property
    def input_size(self) -> int:
        return self.b.shape[1]

    def _validate(self) -> None:
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            raise ValueError("state_matrix must be square")
        if self.b.ndim != 2 or self.b.shape[0] != self.a.shape[0]:
            raise ValueError("input_matrix row count must match the state size")
        if not all(np.all(np.isfinite(matrix)) for matrix in (self.a, self.b)):
            raise ValueError("system matrices must contain only finite values")
        if (
            isinstance(self.prediction_horizon, bool)
            or not isinstance(self.prediction_horizon, int)
            or self.prediction_horizon < 1
        ):
            raise ValueError("prediction_horizon must be a positive integer")
        if self.state_weight.shape != (self.state_size, self.state_size):
            raise ValueError("state_weight has the wrong shape")
        if self.terminal_weight.shape != (self.state_size, self.state_size):
            raise ValueError("terminal_weight has the wrong shape")
        if self.input_weight.shape != (self.input_size, self.input_size):
            raise ValueError("input_weight has the wrong shape")
        for name, weight, positive_definite in (
            ("state_weight", self.state_weight, False),
            ("terminal_weight", self.terminal_weight, False),
            ("input_weight", self.input_weight, True),
        ):
            if not np.all(np.isfinite(weight)) or not np.allclose(weight, weight.T, atol=1e-12):
                raise ValueError(f"{name} must be finite and symmetric")
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(weight)))
            threshold = 0.0 if positive_definite else -1e-12
            if minimum_eigenvalue <= threshold:
                qualifier = "positive definite" if positive_definite else "positive semidefinite"
                raise ValueError(f"{name} must be {qualifier}")
        if self.solver_tolerance <= 0 or not np.isfinite(self.solver_tolerance):
            raise ValueError("solver_tolerance must be finite and positive")
        if (
            isinstance(self.solver_max_iterations, bool)
            or not isinstance(self.solver_max_iterations, int)
            or self.solver_max_iterations < 1
        ):
            raise ValueError("solver_max_iterations must be a positive integer")
        self.constraints.validate(state_size=self.state_size, input_size=self.input_size)

    def _lifted_prediction_matrices(self) -> tuple[FloatArray, FloatArray]:
        states = np.zeros((self.prediction_horizon * self.state_size, self.state_size))
        inputs = np.zeros(
            (
                self.prediction_horizon * self.state_size,
                self.prediction_horizon * self.input_size,
            )
        )
        for prediction_step in range(self.prediction_horizon):
            row = slice(
                prediction_step * self.state_size,
                (prediction_step + 1) * self.state_size,
            )
            states[row] = np.linalg.matrix_power(self.a, prediction_step + 1)
            for input_step in range(prediction_step + 1):
                column = slice(
                    input_step * self.input_size,
                    (input_step + 1) * self.input_size,
                )
                inputs[row, column] = (
                    np.linalg.matrix_power(self.a, prediction_step - input_step) @ self.b
                )
        return states, inputs

    def _state_vector(self, state: ArrayLike) -> FloatArray:
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector.size != self.state_size:
            raise ValueError(f"state must contain {self.state_size} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("state must contain only finite values")
        return vector

    def _reference_vector(self, reference: ArrayLike | None) -> FloatArray:
        if reference is None:
            return np.zeros(self.prediction_horizon * self.state_size)
        vector = np.asarray(reference, dtype=np.float64).reshape(-1)
        expected = self.prediction_horizon * self.state_size
        if vector.size == self.state_size:
            vector = np.tile(vector, self.prediction_horizon)
        elif vector.size != expected:
            raise ValueError(f"reference must contain {self.state_size} or {expected} values")
        if not np.all(np.isfinite(vector)):
            raise ValueError("reference must contain only finite values")
        return vector

    def solve(
        self,
        state: ArrayLike,
        reference: ArrayLike | None = None,
        *,
        initial_guess: ArrayLike | None = None,
    ) -> MPCSolution:
        """Solve one constrained finite-horizon quadratic programme."""
        current_state = self._state_vector(state)
        if not self.constraints.contains(current_state):
            violation = float(np.max(self.constraints.state_violation(current_state)))
            raise ValueError(f"current state violates the aircraft envelope by {violation:.3e} rad")
        target = self._reference_vector(reference)
        free_response = self.state_map @ current_state
        error = free_response - target
        linear_cost = 2.0 * self.input_map.T @ self.state_cost @ error
        inequality_upper = self._lifted_state_limit - (self._constraint_state_map @ free_response)

        sequence_size = self.prediction_horizon * self.input_size
        if initial_guess is None:
            # The clipped unconstrained optimum is a stronger starting point
            # than zero for this lightly actuated aircraft model.
            guess = np.clip(
                np.linalg.solve(self.hessian, -linear_cost),
                self._lower_input,
                self._upper_input,
            )
        else:
            guess = np.asarray(initial_guess, dtype=np.float64).reshape(-1)
            if guess.size != sequence_size or not np.all(np.isfinite(guess)):
                raise ValueError(f"initial_guess must contain {sequence_size} finite values")
            guess = np.clip(guess, self._lower_input, self._upper_input)

        def objective(inputs: FloatArray) -> float:
            return float(0.5 * inputs @ self.hessian @ inputs + linear_cost @ inputs)

        def gradient(inputs: FloatArray) -> FloatArray:
            return self.hessian @ inputs + linear_cost

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
            )
            result = minimize(
                objective,
                guess,
                jac=gradient,
                method="SLSQP",
                bounds=Bounds(self._lower_input, self._upper_input),
                constraints=(
                    LinearConstraint(
                        self._constraint_input_map,
                        -np.inf,
                        inequality_upper,
                    ),
                ),
                options={
                    "ftol": self.solver_tolerance,
                    "maxiter": self.solver_max_iterations,
                    "disp": False,
                },
            )

        optimal_inputs = np.asarray(result.x, dtype=np.float64)
        predicted_states = (free_response + self.input_map @ optimal_inputs).reshape(
            self.prediction_horizon, self.state_size
        )
        state_violation = max(
            0.0, float(np.max(self.constraints.state_violation(predicted_states)))
        )
        input_violation = float(
            max(
                0.0,
                np.max(self._lower_input - optimal_inputs),
                np.max(optimal_inputs - self._upper_input),
            )
        )
        maximum_violation = max(state_violation, input_violation)
        if not result.success or maximum_violation > 1e-7:
            raise RuntimeError(
                "aircraft MPC optimisation failed: "
                f"{result.message}; max violation={maximum_violation:.3e}"
            )

        residual = predicted_states.reshape(-1) - target
        actual_objective = float(
            residual @ self.state_cost @ residual
            + optimal_inputs @ self.input_cost @ optimal_inputs
        )
        return MPCSolution(
            inputs=optimal_inputs.reshape(self.prediction_horizon, self.input_size),
            states=predicted_states,
            objective=actual_objective,
            iterations=int(result.nit),
            maximum_constraint_violation=maximum_violation,
        )

    def simulate(
        self,
        initial_state: ArrayLike,
        *,
        steps: int,
        reference: ArrayLike | None = None,
    ) -> SimulationResult:
        """Run deterministic receding-horizon regulation or tracking."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = self._state_vector(initial_state)
        states = [state.copy()]
        inputs: list[FloatArray] = []
        objectives: list[float] = []
        iterations: list[int] = []
        maximum_violation = 0.0
        guess: FloatArray | None = None

        for _ in range(steps):
            solution = self.solve(state, reference, initial_guess=guess)
            control = solution.inputs[0]
            state = self.a @ state + self.b @ control
            states.append(state.copy())
            inputs.append(control.copy())
            objectives.append(solution.objective)
            iterations.append(solution.iterations)
            maximum_violation = max(maximum_violation, solution.maximum_constraint_violation)
            flattened = solution.inputs.reshape(-1)
            guess = np.concatenate((flattened[self.input_size :], flattened[-self.input_size :]))

        state_history = np.asarray(states)
        realised_violation = max(
            0.0, float(np.max(self.constraints.state_violation(state_history)))
        )
        maximum_violation = max(maximum_violation, realised_violation)
        return SimulationResult(
            states=state_history,
            inputs=np.asarray(inputs),
            objectives=np.asarray(objectives),
            solver_iterations=np.asarray(iterations, dtype=np.int64),
            maximum_constraint_violation=maximum_violation,
        )


def default_controller(*, prediction_horizon: int = 20) -> AircraftPitchMPC:
    """Build the documented aircraft regulator used by the report and tests."""
    return AircraftPitchMPC(
        AIRCRAFT_A,
        AIRCRAFT_B,
        prediction_horizon=prediction_horizon,
        state_weight=np.diag([50.0, 10.0, 1000.0]),
        input_weight=np.array([[0.05]]),
        constraints=AircraftConstraints.coursework_limits(),
    )
