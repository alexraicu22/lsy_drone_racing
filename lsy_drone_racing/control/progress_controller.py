"""Jointly choose physical accelerations and progress along the observed racing line."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from lsy_drone_racing.control.calibrated_controller import CalibratedMPC
from lsy_drone_racing.control.predictive_spline_controller import HORIZON, PREDICTION_DT
from lsy_drone_racing.control.tracking_controller import TrackingController

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
    from lsy_drone_racing.control.spline_controller import Scene


class ProgressMPC(CalibratedMPC):
    """A bounded contouring MPC, linearized locally around the last progress proposal."""

    def __init__(self, settings: ConfigDict):
        """Reuse identified dynamics; optionally group future controls to reduce solve cost."""
        super().__init__(acceleration_limit=20.0, vertical_limit=5.0)
        self.settings = settings
        self.route = None
        self.scene = None
        self.start = 0.0
        self.warm_progress = None
        self.warm_controls = None
        self.previous_start = 0.0
        self.attempts = self.accepts = 0
        self.last_progress = None
        size = 3 * HORIZON
        self.position_matrix = np.zeros((size, size))
        self.velocity_matrix = np.zeros((size, size))
        for axis in range(3):
            indices = np.arange(axis, size, 3)
            model = self.vertical if axis == 2 else self
            self.position_matrix[np.ix_(indices, indices)] = model.position_control
            self.velocity_matrix[np.ix_(indices, indices)] = model.velocity_control
        blocks = int(settings.get("progress_control_blocks", HORIZON))
        if not 1 <= blocks <= HORIZON:
            raise ValueError("Progress control blocks must be between 1 and the horizon length")
        boundaries = (
            np.arange(HORIZON + 1)
            if blocks == HORIZON
            else np.unique(np.rint(np.linspace(0, 1, blocks + 1) ** 1.5 * HORIZON).astype(int))
        )
        self.block_ids = np.searchsorted(boundaries[1:], np.arange(HORIZON), side="right")
        blocks = len(boundaries) - 1
        self.expand = np.kron(np.eye(blocks)[self.block_ids], np.eye(3))
        self.control_size = self.expand.shape[1]
        self.full_position_matrix = self.position_matrix
        self.position_matrix = self.position_matrix @ self.expand
        self.velocity_matrix = self.velocity_matrix @ self.expand
        self.difference = np.eye(HORIZON) - np.eye(HORIZON, k=-1)
        self.control_difference = np.kron(self.difference, np.eye(3)) @ self.expand
        self.progress_constraint = np.c_[np.zeros((HORIZON, self.control_size)), self.difference]

    def set_route(self, route: SpeedProfile | None, start: float, scene: Scene) -> None:
        """Reset warm starts after replanning; arc distances never refer to an old route."""
        if route is not self.route:
            self.warm_progress = self.warm_controls = None
            self.last_progress = None
        self.route, self.start, self.scene = route, start, scene

    def _optimize(self, state: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Minimize cross-track error and actuator effort while rewarding forward progress."""
        route, start = self.route, self.start
        end = float(route.distance[-1])
        times = PREDICTION_DT * np.arange(1, HORIZON + 1)
        tangent = route.path(start, nu=1)
        tangent /= max(np.linalg.norm(tangent), 1e-8)
        speed = max(0.0, float(state[1] @ tangent))
        maximum_speed = float(self.settings.get("progress_max_speed", 6.0))
        if self.warm_progress is None:
            nominal = np.minimum(end, start + speed * times + 1.0 * times**2)
            controls = np.broadcast_to(previous, (HORIZON, 3)).copy()
        else:
            # The control loop is 20 ms; the prediction grid is 40 ms.
            old_times = np.r_[0, times]
            old_distances = np.r_[self.previous_start, self.warm_progress]
            anchor = np.interp(0.02, old_times, old_distances)
            nominal = start + np.interp(times + 0.02, old_times, old_distances) - anchor
            nominal = np.clip(nominal, start, end)
            controls = 0.5 * (
                self.warm_controls + np.vstack([self.warm_controls[1:], self.warm_controls[-1:]])
            )
        nominal = start + np.cumsum(
            np.clip(np.diff(np.r_[start, nominal]), 0, maximum_speed * PREDICTION_DT)
        )
        nominal = np.minimum(nominal, end)
        positions = route.path(nominal)
        tangents = route.path(nominal, nu=1)
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-8)
        tangent_matrix = np.zeros((3 * HORIZON, HORIZON))
        for step in range(HORIZON):
            tangent_matrix[3 * step : 3 * step + 3, step] = tangents[step]
        free_position, free_velocity = np.empty_like(positions), np.empty_like(positions)
        for axis in range(3):
            model = self.vertical if axis == 2 else self
            free_position[:, axis] = model.position_state @ state[:, axis]
            free_velocity[:, axis] = model.velocity_state @ state[:, axis]
        # Locally r(s) = r(nominal) + tangent * (s - nominal).
        position_matrix = np.c_[self.position_matrix, -tangent_matrix]
        position_target = positions.ravel() - tangent_matrix @ nominal - free_position.ravel()
        lag_matrix = tangent_matrix.T @ position_matrix
        lag_target = tangent_matrix.T @ position_target
        contour_matrix = position_matrix - tangent_matrix @ lag_matrix
        contour_target = position_target - tangent_matrix @ lag_target
        progress_velocity = tangent_matrix @ self.difference / PREDICTION_DT
        velocity_matrix = np.c_[self.velocity_matrix, -progress_velocity]
        velocity_target = -free_velocity.ravel()
        velocity_target[:3] -= tangents[0] * start / PREDICTION_DT
        contour_weight = float(self.settings.get("progress_contour_weight", 60.0))
        lag_weight = float(self.settings.get("progress_lag_weight", 20.0))
        matrix = np.vstack(
            [
                contour_weight * contour_matrix,
                lag_weight * lag_matrix,
                2.0 * velocity_matrix,
                np.c_[0.25 * self.expand, np.zeros((3 * HORIZON, HORIZON))],
                np.c_[1.5 * self.control_difference, np.zeros((3 * HORIZON, HORIZON))],
            ]
        )
        target = np.r_[
            contour_weight * contour_target,
            lag_weight * lag_target,
            2.0 * velocity_target,
            np.zeros(3 * HORIZON),
            1.5 * np.r_[previous, np.zeros(3 * (HORIZON - 1))],
        ]
        quadratic = matrix.T @ matrix
        linear = -matrix.T @ target
        # Reward progress throughout the horizon, so postponing acceleration
        # until its last steps is worse than making the same progress earlier.
        linear[-HORIZON:] -= float(self.settings.get("progress_reward", 40.0))
        control_bound = np.tile(
            [self.acceleration_limit, self.acceleration_limit, self.vertical_limit],
            self.control_size // 3,
        )
        trust = float(self.settings.get("progress_trust", 0.25))
        bounds = Bounds(
            np.r_[-control_bound, np.maximum(start, nominal - trust)],
            np.r_[control_bound, np.minimum(end, nominal + trust)],
        )
        first = np.r_[start, np.zeros(HORIZON - 1)]
        constraint = LinearConstraint(
            self.progress_constraint, first, first + maximum_speed * PREDICTION_DT
        )
        grouped = np.array(
            [
                controls[self.block_ids == block].mean(axis=0)
                for block in range(self.control_size // 3)
            ]
        )
        initial = np.r_[np.clip(grouped.ravel(), -control_bound, control_bound), nominal]

        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            product = quadratic @ value
            return (float(0.5 * value @ product + linear @ value) / 100, (product + linear) / 100)

        result = minimize(
            objective,
            initial,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": int(self.settings.get("progress_iterations", 25)), "ftol": 1e-5},
        )
        value = result.x
        increments = self.difference @ value[self.control_size :] - first
        if (
            not np.isfinite(value).all()
            or increments.min() < -1e-5
            or increments.max() > maximum_speed * PREDICTION_DT + 1e-5
            or objective(value)[0] > objective(initial)[0] + 1e-6
        ):
            raise ValueError("Progress solve did not produce a feasible improvement")
        controls = (self.expand @ value[: self.control_size]).reshape(HORIZON, 3)
        # Project the box-constrained solve onto the same total-force bound as
        # the existing tracker, then recompute and check the resulting motion.
        controls *= np.minimum(
            1, self.acceleration_limit / np.maximum(np.linalg.norm(controls, axis=1), 1e-8)
        )[:, None]
        predicted = free_position + (self.full_position_matrix @ controls.ravel()).reshape(
            HORIZON, 3
        )
        segment_starts = np.vstack([state[0], predicted[:-1]])
        probes = (
            segment_starts[:, None]
            + (predicted - segment_starts)[:, None] * np.linspace(0, 1, 5)[None, :, None]
        )
        if not self.scene.free(probes.reshape(-1, 3), start=state[0]).all():
            raise ValueError("Progress prediction loses observed clearance")
        self.warm_controls, self.warm_progress = controls, value[self.control_size :].copy()
        self.previous_start = start
        self.last_progress = self.warm_progress.copy()
        return controls[0], predicted

    def solve(
        self,
        state: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        bias: np.ndarray,
        previous: np.ndarray,
        close_tracking: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use progress MPC for racing; retain checked fixed-reference tracking as fallback."""
        if self.route is not None and not close_tracking:
            self.attempts += 1
            try:
                acceleration, prediction = self._optimize(state, self.gain * previous + bias)
            except (ValueError, np.linalg.LinAlgError):
                self.warm_controls = self.warm_progress = None
            else:
                self.accepts += 1
                return (acceleration - bias) / self.gain, prediction
        return super().solve(state, positions, velocities, bias, previous, close_tracking)


class ProgressController(TrackingController):
    """Use the public-map planner with online acceleration/progress optimization."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Leave all previous controllers independent and available for comparison."""
        super().__init__(obs, info, config)
        self.mpc = ProgressMPC(config.controller)

    def _reference(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """The timed reference is only a fallback; the racing solve uses arc-length geometry."""
        reference = super()._reference(obs)
        racing = self.phase == "race" and not self.tight_corridor
        route = self.curve if racing else None
        start = float(route.distance[self.progress]) if racing else 0.0
        self.mpc.set_route(route, start, self.scene)
        return reference

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Expose progress-solver use in the existing evaluation diagnostics."""
        command = super().compute_control(obs, info)
        self.trajectory_attempts = self.mpc.attempts
        self.trajectory_accepts = self.mpc.accepts
        return command
