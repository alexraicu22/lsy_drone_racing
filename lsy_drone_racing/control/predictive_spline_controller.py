"""Predictive acceleration tracking for the existing gate-aware route.

pixi run python scripts/sim.py --config=level2.toml --controller=predictive_spline_controller.py
Uses a small bounded least-squares MPC; no training or generated solver is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm
from scipy.optimize import lsq_linear

from lsy_drone_racing.control.continuous_spline_controller import ContinuousSplineController
from lsy_drone_racing.control.spline_controller import APPROACH_DISTANCE, gate_points

if TYPE_CHECKING:
    from ml_collections import ConfigDict

HORIZON = 25
PREDICTION_DT = 0.04
ACCELERATION_LAG = 0.06
ACCELERATION_LIMIT = 8.0
CRUISE_SCALE = 1.5


class AccelerationMPC:
    """Predict position, velocity and acceleration with a first-order actuator lag."""

    def __init__(self, acceleration_lag: float = ACCELERATION_LAG):
        """Precompute dynamics and weighted prediction matrices for a one-second horizon."""
        continuous = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1 / acceleration_lag, 1 / acceleration_lag],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
        discrete = expm(PREDICTION_DT * continuous)
        self.transition, self.input = discrete[:3, :3], discrete[:3, 3]
        state = np.zeros((HORIZON, 3, 3))
        control = np.zeros((HORIZON, 3, HORIZON))
        for row in range(HORIZON):
            state[row] = np.linalg.matrix_power(self.transition, row + 1)
            for column in range(row + 1):
                control[row, :, column] = (
                    np.linalg.matrix_power(self.transition, row - column) @ self.input
                )
        self.position_state, self.velocity_state = state[:, 0], state[:, 1]
        self.position_control, self.velocity_control = control[:, 0], control[:, 1]
        difference = np.eye(HORIZON) - np.eye(HORIZON, k=-1)
        self.matrices = {}
        for weight in [30.0, 60.0, 90.0, 180.0]:
            matrix = np.vstack(
                [
                    weight * self.position_control,
                    4 * self.velocity_control,
                    0.25 * np.eye(HORIZON),
                    1.5 * difference,
                ]
            )
            self.matrices[weight] = matrix, np.linalg.pinv(matrix)

    def solve(
        self,
        state: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        bias: np.ndarray,
        previous: np.ndarray,
        close_tracking: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose bounded accelerations and return the resulting predicted positions."""
        close_tracking = close_tracking or getattr(self, "prefer_close_tracking", False)
        repeated_bias = np.broadcast_to(bias, (HORIZON, 3))
        free_position = self.position_state @ state + self.position_control @ repeated_bias
        free_velocity = self.velocity_state @ state + self.velocity_control @ repeated_bias
        controls = np.zeros((HORIZON, 3))
        for axis in range(3):
            weight = (90.0 if close_tracking else 30.0) * (2 if axis == 2 else 1)
            matrix, inverse = self.matrices[weight]
            target = np.concatenate(
                [
                    weight * (positions[:, axis] - free_position[:, axis]),
                    4 * (velocities[:, axis] - free_velocity[:, axis]),
                    np.zeros(HORIZON),
                    1.5 * np.r_[previous[axis], np.zeros(HORIZON - 1)],
                ]
            )
            solution = inverse @ target
            bound = 5.0 if axis == 2 else ACCELERATION_LIMIT
            if np.any(np.abs(solution) > bound):
                solution = lsq_linear(
                    matrix, target, bounds=(-bound, bound), method="bvls", tol=1e-4, max_iter=30
                ).x
            controls[:, axis] = solution
        norm = np.linalg.norm(controls, axis=-1)
        controls *= np.minimum(1.0, ACCELERATION_LIMIT / np.maximum(norm, 1e-8))[:, None]
        predicted = free_position + self.position_control @ controls
        return controls[0], predicted


class PredictiveSplineController(ContinuousSplineController):
    """Track the route with predictive acceleration and an observed disturbance bias."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep the public-map planner and initialize the predictive tracker."""
        super().__init__(obs, info, config)
        self.speed_scale = float(config.controller.get("speed_scale", CRUISE_SCALE))
        self.mpc = AccelerationMPC()
        self.previous_velocity = np.asarray(obs["vel"]).copy()
        self.measured_acceleration = np.zeros(3)
        self.bias = np.zeros(3)
        self.previous_input = np.zeros(3)
        self.was_predictive = False
        self.mpc_steps = 0
        self.mpc_fallbacks = 0

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Look ahead on the current route; accept only predictions clear of obstacles."""
        velocity = np.asarray(obs["vel"])
        acceleration = np.clip((velocity - self.previous_velocity) / self.dt, -30, 30)
        self.previous_velocity = velocity.copy()
        self.measured_acceleration += (
            self.dt
            / (getattr(self, "acceleration_filter", 0.06) + self.dt)
            * (acceleration - self.measured_acceleration)
        )
        if self.was_predictive:
            expected = self.previous_input
            if hasattr(self, "command_gain"):
                self.filtered_command += (
                    self.dt
                    / (self.command_lag + self.dt)
                    * (self.command_gain * self.previous_input - self.filtered_command)
                )
                expected = self.filtered_command
            self.bias += (
                self.dt / (0.5 + self.dt) * (self.measured_acceleration - expected - self.bias)
            )
            bound = getattr(self, "bias_limit", 4.0)
            self.bias = np.clip(self.bias, -bound, bound)
        fallback = super().compute_control(obs, info)
        self.was_predictive = False
        if self.phase == "takeoff" or self.holding:
            return fallback
        positions, velocities = self._reference(obs)
        state = np.stack([obs["pos"], velocity, self.measured_acceleration])
        for close_tracking in [False, True]:
            control, predicted = self.mpc.solve(
                state, positions, velocities, self.bias, self.previous_input, close_tracking
            )
            if self.scene.free(predicted, start=np.asarray(obs["pos"])).all():
                self.previous_input = control
                self.was_predictive = True
                self.mpc_steps += 1
                # Zero position/velocity errors in the onboard state controller:
                # the outer MPC supplies the required acceleration directly.
                return np.concatenate(
                    [obs["pos"], velocity, control, [0, 0, 0, 1], np.zeros(3)]
                ).astype(np.float32)
        self.previous_input = np.zeros(3)
        self.mpc_fallbacks += 1
        return fallback

    def _reference(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Sample the installed trajectory; experimental pacing can override this seam."""
        pace = float(getattr(self, "reference_pace", 1.0))
        future = pace * PREDICTION_DT * np.arange(1, HORIZON + 1)
        if self.phase == "race":
            distance = self.curve.distance[self.progress]
            start_time = np.interp(distance, self.curve.distance, self.curve.time)
        else:
            start_time = self.elapsed
        times = np.minimum(start_time + future, self.duration)
        positions = self.curve(times)
        velocities = pace * self.curve(times, nu=1)
        if self.phase == "cross":
            approach, centre, _ = gate_points(obs, self.entry)
            normal = (centre - approach) / APPROACH_DISTANCE
            positions = centre + np.outer((positions - centre) @ normal, normal)
            velocities = np.outer(velocities @ normal, normal)
        return positions, velocities
