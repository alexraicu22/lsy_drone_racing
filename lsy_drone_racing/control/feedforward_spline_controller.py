"""Test acceleration feedforward while following the existing gate-aware route.

pixi run python scripts/sim.py --config=level2.toml --controller=feedforward_spline_controller.py
The classical and RL controllers remain available separately.
Experimental: this variant reached only 50% completion and was not selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.continuous_spline_controller import (
    LOOKAHEAD,
    VELOCITY_FEEDFORWARD,
    ContinuousSplineController,
)

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.racing_spline_controller import SpeedProfile

MAX_FEEDFORWARD_ACCELERATION = 3.0
FEEDFORWARD_GAIN = 0.4


def path_acceleration(profile: SpeedProfile, distance: float, velocity: np.ndarray) -> np.ndarray:
    """Anticipate turns; let the velocity controller handle forward acceleration."""
    samples = np.clip(distance + np.array([-0.05, 0.05]), 0, profile.distance[-1])
    spacing = samples[1] - samples[0]
    if spacing < 1e-6:
        return np.zeros(3)
    tangents = profile.path(samples, nu=1)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=-1, keepdims=True), 1e-8)
    speeds = VELOCITY_FEEDFORWARD * np.interp(samples, profile.distance, profile.speed)
    field = tangents * speeds[:, None]
    tangent = profile.path(distance, nu=1)
    tangent /= max(np.linalg.norm(tangent), 1e-8)
    forward_speed = max(0.0, float(np.dot(velocity, tangent)))
    acceleration = (field[1] - field[0]) / spacing * forward_speed
    acceleration -= tangent * np.dot(acceleration, tangent)
    norm = np.linalg.norm(acceleration)
    return acceleration * min(1.0, MAX_FEEDFORWARD_ACCELERATION / max(norm, 1e-8))


class FeedforwardSplineController(ContinuousSplineController):
    """Anticipate acceleration and turns instead of relying only on tracking error."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep the validated route planner and initialize a smooth acceleration command."""
        super().__init__(obs, info, config)
        self.acceleration_command = np.zeros(3)
        self.previous_curve = None

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Add bounded, filtered acceleration feedforward on continuous sections."""
        command = super().compute_control(obs, info)
        if self.phase != "race" or self.holding:
            self.acceleration_command[:] = 0
            return command
        if self.curve is not self.previous_curve:
            self.acceleration_command[:] = 0
            self.previous_curve = self.curve
        lookahead = 0.05 if self.tight_corridor else LOOKAHEAD
        distance = min(self.curve.distance[self.progress] + lookahead, self.curve.distance[-1])
        desired = path_acceleration(self.curve, distance, np.asarray(obs["vel"]))
        self.acceleration_command += (
            self.dt / (0.08 + self.dt) * (desired - self.acceleration_command)
        )
        command[6:9] = FEEDFORWARD_GAIN * self.acceleration_command
        return command
