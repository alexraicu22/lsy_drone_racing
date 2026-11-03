"""Run the learned speed policy through the standard simulator.

pixi run python scripts/sim.py --config=level2.toml --controller=speed_racing_controller.py
Add --render=True to watch.
Set SPEED_RACING_MODEL to select another checkpoint.
"""

import os

import numpy as np
from ml_collections import ConfigDict

from lsy_drone_racing.control.continuous_spline_controller import ContinuousSplineController
from lsy_drone_racing.control.speed_racing_rl import (
    DEFAULT_MODEL,
    SpeedSetting,
    load_policy,
    speed_features,
)


class SpeedRacingController(ContinuousSplineController):
    """Keep route planning and flight control; let PPO select speed."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Load the policy once and share training's observation/action transforms."""
        super().__init__(obs, info, config)
        self.model = load_policy(os.environ.get("SPEED_RACING_MODEL", str(DEFAULT_MODEL)))
        if self.model.speed_policy_frequency != config.env.freq:
            raise ValueError("Speed policy and simulator control frequencies differ")
        self.setting = SpeedSetting()
        self.seconds = 0.0

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Apply deterministic speed choices after classical takeoff."""
        if self.phase != "takeoff":
            features = speed_features(obs, self, self.setting.factor, self.seconds)
            action, _ = self.model.predict(features, deterministic=True)
            self.speed_factor = self.setting.update(action, self.dt)
        command = super().compute_control(obs, info)
        self.seconds += self.dt
        return command
