"""Advance the MPC racing reference by 25 ms while preserving the proven route planner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear

from lsy_drone_racing.control.calibrated_controller import CalibratedMPC
from lsy_drone_racing.control.predictive_spline_controller import HORIZON, PREDICTION_DT
from lsy_drone_racing.control.submission_controller import SubmissionController

if TYPE_CHECKING:
    from ml_collections import ConfigDict


class TrackingMPC(CalibratedMPC):
    """Make velocity and command-change weights explicit for controlled comparisons."""

    def __init__(self, velocity_weight: float, smoothing_weight: float):
        """Keep the calibrated dynamics, prediction horizon and physical bounds."""
        super().__init__(acceleration_limit=20.0, vertical_limit=5.0)
        self.velocity_weight = velocity_weight
        self.smoothing_weight = smoothing_weight
        difference = np.eye(HORIZON) - np.eye(HORIZON, k=-1)
        for model in [self, self.vertical]:
            for weight in model.matrices:
                matrix = np.vstack(
                    [
                        weight * model.position_control,
                        velocity_weight * model.velocity_control,
                        0.25 * np.eye(HORIZON),
                        smoothing_weight * difference,
                    ]
                )
                model.matrices[weight] = matrix, np.linalg.pinv(matrix)

    def solve(
        self,
        state: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        bias: np.ndarray,
        previous: np.ndarray,
        close_tracking: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the reference with bounded physical accelerations, then decode the command."""
        controls = np.zeros((HORIZON, 3))
        free_positions = np.zeros_like(positions)
        for axis in range(3):
            model = self.vertical if axis == 2 else self
            free_position = model.position_state @ state[:, axis]
            free_velocity = model.velocity_state @ state[:, axis]
            free_positions[:, axis] = free_position
            weight = (90.0 if close_tracking else 30.0) * (2 if axis == 2 else 1)
            matrix, inverse = model.matrices[weight]
            target = np.r_[
                weight * (positions[:, axis] - free_position),
                self.velocity_weight * (velocities[:, axis] - free_velocity),
                np.zeros(HORIZON),
                self.smoothing_weight
                * np.r_[self.gain[axis] * previous[axis] + bias[axis], np.zeros(HORIZON - 1)],
            ]
            control = inverse @ target
            bound = self.vertical_limit if axis == 2 else self.acceleration_limit
            if np.any(np.abs(control) > bound):
                control = lsq_linear(
                    matrix, target, bounds=(-bound, bound), method="bvls", tol=1e-4, max_iter=30
                ).x
            controls[:, axis] = control
        controls *= np.minimum(
            1, self.acceleration_limit / np.maximum(np.linalg.norm(controls, axis=-1), 1e-8)
        )[:, None]
        prediction = free_positions.copy()
        for axis in range(3):
            model = self.vertical if axis == 2 else self
            prediction[:, axis] += model.position_control @ controls[:, axis]
        return (controls[0] - bias) / self.gain, prediction


class TrackingController(SubmissionController):
    """Keep the route and all sensing/recovery checks while changing MPC tracking."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep the baseline solver weights; the faster reference is selected independently."""
        super().__init__(obs, info, config)
        self.mpc = TrackingMPC(
            float(config.controller.get("tracking_velocity_weight", 4.0)),
            float(config.controller.get("tracking_smoothing_weight", 1.5)),
        )

    def _reference(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Advance the racing reference slightly without changing the route or fallback phases."""
        advance = float(self.config.controller.get("tracking_advance", 0.025))
        if self.phase != "race" or advance == 0:
            return super()._reference(obs)
        gate_advance = self.config.controller.get("tracking_gate_advance")
        if gate_advance is not None:
            for centre, rotation in zip(self.scene.gates, self.scene.rotations):
                local = (obs["pos"] - centre) @ rotation
                if abs(local[0]) < 0.40 and np.max(np.abs(local[1:])) < 0.45:
                    advance = min(advance, float(gate_advance))
        distance = self.curve.distance[self.progress]
        now = np.interp(distance, self.curve.distance, self.curve.time)
        times = np.minimum(now + advance + PREDICTION_DT * np.arange(1, HORIZON + 1), self.duration)
        return self.curve(times), self.curve(times, nu=1)
