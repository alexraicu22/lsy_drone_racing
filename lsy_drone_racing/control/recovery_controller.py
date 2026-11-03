"""Recover stalled gate approaches by moving to a checked sensing position."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.outlier_controller import OutlierController
from lsy_drone_racing.control.spline_controller import Scene, gate_points, make_route

if TYPE_CHECKING:
    from ml_collections import ConfigDict


class RecoveryController(OutlierController):
    """Keep normal racing unchanged; try a new viewpoint after a stationary stall."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Track stationary time independently of planned trajectory clocks."""
        self.stall_position = np.asarray(obs["pos"]).copy()
        self.stall_gate = int(obs["n_gates_passed"])
        self.stall_seconds = 0.0
        self.recovery_attempts = 0
        super().__init__(obs, info, config)

    def _recover(self, obs: dict) -> bool:
        """Choose a clear nearby approach that brings the gate within sensor range."""
        entry = int(obs["n_gates_passed"])
        gate = int(obs["gate_sequence"][entry])
        scene = Scene(obs, self.config)
        scene.pole_radius = np.where(scene.poles_known, 0.20, 0.37)
        _, centre, _ = gate_points(obs, entry)
        normal = scene.rotations[gate][:, 0] * int(obs["gate_sequence_direction"][entry])
        sideways = scene.rotations[gate][:, 1]
        start = np.asarray(obs["pos"])
        candidates = []
        for stand_off in [0.32, 0.40]:
            for lateral in [-0.25, 0.25]:
                target = centre - stand_off * normal + lateral * sideways
                if np.linalg.norm(target - start) < 0.18 or not scene.free(target)[0]:
                    continue
                try:
                    route = scene.route(start, target)
                    curve, duration = make_route(route, np.asarray(obs["vel"]), scene)
                except ValueError:
                    continue
                candidates.append((duration, curve, target))
        if not candidates:
            return False
        duration, curve, target = min(candidates, key=lambda candidate: candidate[0])
        self.entry = entry
        self.scene = scene
        self.curve, self.duration, self.goal = curve, duration, target
        self.elapsed = 0.0
        self.phase = "approach"
        self.staging = True
        self.holding = False
        self.hold_position = None
        self.signature = self._signature(obs)
        self.profile_retry = duration
        self.replans += 1
        self.recovery_attempts += 1
        return True

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Change viewpoint only after two seconds without meaningful movement."""
        obs = {key: np.asarray(value) for key, value in obs.items()}
        passed = int(obs["n_gates_passed"])
        position = obs["pos"]
        if passed != self.stall_gate or np.linalg.norm(position - self.stall_position) > 0.15:
            self.stall_gate, self.stall_position = passed, position.copy()
            self.stall_seconds = 0.0
        else:
            self.stall_seconds += self.dt
        if (
            self.stall_seconds > 2.0
            and np.linalg.norm(obs["vel"]) < 0.45
            and passed < len(obs["gate_sequence"])
        ):
            self._recover(obs)
            self.stall_seconds = 0.0
            self.stall_position = position.copy()
        return super().compute_control(obs, info)
