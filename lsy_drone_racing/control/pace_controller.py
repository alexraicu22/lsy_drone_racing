"""Increase trajectory pace while retaining the baseline route and MPC safety checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.flow_racing_controller import FlowRacingController

if TYPE_CHECKING:
    from ml_collections import ConfigDict


class PaceController(FlowRacingController):
    """Warp reference time consistently, reducing pace when approaching an unseen gate."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep the validated path construction and original MPC acceleration limits."""
        self.pace = float(config.controller.get("pace", 1.5))
        self.reference_pace = 1.0
        self.adaptive_pace = bool(config.controller.get("pace_adaptive", False))
        super().__init__(obs, info, config)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Carry speed where tracking allows it; keep sensing and tight routes cautious."""
        pace = self.pace
        passed = int(obs["n_gates_passed"])
        if self.phase == "takeoff" or self.holding or self.tight_corridor:
            pace = 1.0
        elif passed < len(obs["gate_sequence"]):
            gate = int(obs["gate_sequence"][passed])
            separation = np.linalg.norm(np.asarray(obs["pos"])[:2] - obs["gates_pos"][gate][:2])
            if not obs["gates_visited"][gate] and separation < 0.95:
                pace = min(pace, 1.0)
        if self.adaptive_pace:
            position, velocity = np.asarray(obs["pos"]), np.asarray(obs["vel"])
            for gate in np.flatnonzero(obs["gates_visited"]):
                rotation = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
                local = (position - obs["gates_pos"][gate]) @ rotation
                if abs(local[0]) < 0.20 and np.linalg.norm(local[1:]) < 0.30:
                    pace = min(pace, 1.0)  # Keep the whole body clear before turning hard.
                if passed < len(obs["gate_sequence"]) and gate == int(obs["gate_sequence"][passed]):
                    direction = int(obs["gate_sequence_direction"][passed])
                    forward = local[0] * direction
                    if -0.65 < forward < -0.20:
                        local_velocity = velocity @ rotation
                        crossing_time = -forward / max(0.5, local_velocity[0] * direction)
                        miss = np.linalg.norm(local[1:] + crossing_time * local_velocity[1:])
                        if miss > 0.06:
                            pace = min(pace, 1.0)
            if position[2] < 0.30:
                pace = min(pace, 1.0)
        self.reference_pace = self.speed_factor = pace
        return super().compute_control(obs, info)
