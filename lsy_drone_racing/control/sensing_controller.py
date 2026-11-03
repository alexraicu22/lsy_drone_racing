"""Join a short moving prefix to replanned routes after public sensing updates."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BPoly

from lsy_drone_racing.control.recovery_controller import RecoveryController

if TYPE_CHECKING:
    from ml_collections import ConfigDict


class SensingController(RecoveryController):
    """Keep the fast tracker and connect new routes to its existing forward motion."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Leave the original controller available as an independent fallback."""
        self.sensing_splices = 0
        super().__init__(obs, info, config)

    def _geometry(self, obs: dict) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray]:
        seconds = float(self.config.controller.get("sensing_splice_seconds", 0.10))
        if seconds > 0 and self.phase == "race" and not self.tight_corridor:
            joined = self._moving_geometry(obs, seconds)
            if joined is not None:
                self.sensing_splices += 1
                return joined
        return super()._geometry(obs)

    def _moving_geometry(
        self, obs: dict, seconds: float
    ) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray] | None:
        """Connect measured motion to the new route a short time ahead, if clear."""
        profile = self.curve
        if not hasattr(profile, "distance") or np.linalg.norm(obs["vel"]) < 0.5:
            return None
        now = np.interp(profile.distance[self.progress], profile.distance, profile.time)
        future = min(now + seconds, profile.time[-1])
        position, velocity = profile(future), profile(future, nu=1)
        length = np.linalg.norm(position - obs["pos"])
        if not 0.04 < length < 0.35 or np.dot(position - obs["pos"], obs["vel"]) <= 0:
            return None
        gate = int(obs["gate_sequence"][self.entry])
        normal = self.scene.rotations[gate][:, 0] * obs["gate_sequence_direction"][self.entry]
        if np.dot(position - self.scene.gates[gate], normal) > -0.08:
            return None  # Keep every required crossing in the newly planned section.
        future_obs = dict(obs, pos=position, vel=velocity)
        old_scene = copy.deepcopy(self.scene)
        try:
            curve, centres, knots = super()._geometry(future_obs)
            prefix = BPoly.from_derivatives(
                [0, length],
                [
                    [obs["pos"], obs["vel"] / max(np.linalg.norm(obs["vel"]), 1), np.zeros(3)],
                    [curve(0), curve(0, nu=1), curve(0, nu=2)],
                ],
            )
            joined = BPoly(
                np.concatenate([prefix.c, curve.c], axis=1), np.r_[prefix.x, curve.x[1:] + length]
            )
            samples = np.linspace(0, joined.x[-1], max(120, int(joined.x[-1] / 0.008)))
            if self.scene.free(joined(samples), start=obs["pos"]).all():
                return joined, centres, knots + 1
        except ValueError:
            pass
        self.scene = old_scene
        return None
