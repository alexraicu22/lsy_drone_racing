"""Gate-aware racing with acceleration response identified from public flight telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.pace_controller import PaceController
from lsy_drone_racing.control.predictive_spline_controller import HORIZON, AccelerationMPC

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from scipy.interpolate import BPoly

COMMAND_GAIN = np.array([1.25, 1.27, 1.72])


class CalibratedMPC(AccelerationMPC):
    """Optimize actual steady acceleration, then invert the measured command response."""

    def __init__(self, acceleration_limit: float = 8.0, vertical_limit: float = 5.0):
        """Use measured horizontal actuator lag rather than the old 0.06-second guess."""
        super().__init__(acceleration_lag=0.145)
        self.vertical = AccelerationMPC(acceleration_lag=0.094)
        self.gain = COMMAND_GAIN.copy()
        self.acceleration_limit = acceleration_limit
        self.vertical_limit = vertical_limit

    def solve(
        self,
        state: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        bias: np.ndarray,
        previous: np.ndarray,
        close_tracking: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict each axis with its measured lag and return a physical state command."""
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
                4 * (velocities[:, axis] - free_velocity),
                np.zeros(HORIZON),
                1.5 * np.r_[self.gain[axis] * previous[axis] + bias[axis], np.zeros(HORIZON - 1)],
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


class CalibratedController(PaceController):
    """Preserve route planning and adapt residual acceleration bias during each race."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Initialize identified dynamics without reading randomized simulator properties."""
        super().__init__(obs, info, config)
        self.adaptive_pace = bool(config.controller.get("pace_adaptive", True))
        self.online_force = bool(config.controller.get("calibrated_online_thrust", False))
        self.mpc = CalibratedMPC()
        self.command_gain = COMMAND_GAIN
        self.command_lag = np.array([0.17, 0.16, 0.11])
        self.acceleration_filter = 0.02
        self.bias_limit = 8.0
        self.bias = np.array([0.0, 0.0, 4.9])
        self.filtered_command = -self.bias.copy()
        self.force_gain = 1.5
        self.previous_position = np.asarray(obs["pos"]).copy()
        self.previous_axis = (
            Rotation.from_quat(obs["quat"]).as_matrix()[:, 2]
            if "quat" in obs
            else np.array([0.0, 0, 1])
        )
        if config.controller.get("calibrated_merge_takeoff", True):
            old = self.__dict__.copy()
            try:
                self._approach({key: np.asarray(value) for key, value in obs.items()})
            except ValueError:
                self.__dict__.update(old)

    def _geometry(self, obs: dict) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray]:
        radius = float(self.config.controller.get("calibrated_pole_radius", 0.22))
        self.scene.pole_radius[self.scene.poles_known] = radius
        unknown_radius = self.config.controller.get("calibrated_unknown_radius")
        if unknown_radius is not None:
            self.scene.pole_radius[~self.scene.poles_known] = float(unknown_radius)
        return super()._geometry(obs)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Estimate thrust response from measured motion along the drone's body axis."""
        if not self.online_force:
            return super().compute_control(obs, info)
        position, velocity = np.asarray(obs["pos"]), np.asarray(obs["vel"])
        axis = (
            Rotation.from_quat(obs["quat"]).as_matrix()[:, 2]
            if "quat" in obs
            else np.array([0.0, 0, 1])
        )
        if self.was_predictive:
            gravity = np.array([0.0, 0, 9.81])
            acceleration = (velocity - self.previous_velocity) / self.dt
            mean_axis = axis + self.previous_axis
            mean_axis /= max(np.linalg.norm(mean_axis), 1e-8)
            # The onboard state loop still damps motion within each 20 ms action.
            correction = (
                np.array([0.4, 0.4, 1.25]) / 0.0393 * (position - self.previous_position)
                + np.array([0.2, 0.2, 0.5]) / 0.0393 * (velocity - self.previous_velocity)
            ) / 2
            denominator = np.dot(self.previous_input + gravity - correction, mean_axis)
            if denominator > 3 and np.linalg.norm(self.previous_input) < 8:
                observed = np.dot(acceleration + gravity, mean_axis) / denominator
                if 1.0 < observed < 2.5:
                    self.force_gain += self.dt / (0.15 + self.dt) * (observed - self.force_gain)
        self.previous_position, self.previous_axis = position.copy(), axis
        self.mpc.gain[:] = self.force_gain
        self.bias = np.array([0.0, 0, 9.81 * (self.force_gain - 1)])
        # The physical thrust estimate replaces the old additive bias observer.
        self.was_predictive = False
        return super().compute_control(obs, info)
