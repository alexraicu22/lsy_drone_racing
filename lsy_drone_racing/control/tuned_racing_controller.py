"""A tunable version of the 25 ms tracker; existing submission controllers stay intact."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.predictive_spline_controller import HORIZON, PREDICTION_DT
from lsy_drone_racing.control.recovery_controller import RecoveryController
from lsy_drone_racing.control.tracking_controller import TrackingMPC

if TYPE_CHECKING:
    from ml_collections import ConfigDict

DEFAULTS = {
    "profile_acceleration": 16.0,
    "profile_max_speed": 6.0,
    "profile_mpc_acceleration": 20.0,
    "profile_gate_speed": 2.8,
    "profile_sensor_speed": 2.0,
    "profile_gate_offset": 0.05,
    "tracking_advance": 0.025,
    "tracking_velocity_weight": 4.0,
    "tracking_smoothing_weight": 1.5,
    "calibrated_unknown_radius": 0.37,
    "calibrated_online_thrust": True,
}


class TunedRacingController(RecoveryController):
    """Let explicit trial settings reach both the planner and the tracking solver."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Copy the config; apply defaults only to missing controller options."""
        config = copy.deepcopy(config)
        for key, value in DEFAULTS.items():
            if key not in config.controller:
                config.controller[key] = value
        super().__init__(obs, info, config)
        self.mpc = TrackingMPC(
            float(config.controller.tracking_velocity_weight),
            float(config.controller.tracking_smoothing_weight),
        )
        self.mpc.acceleration_limit = float(config.controller.profile_mpc_acceleration)
        self.mpc.vertical_limit = float(config.controller.get("profile_mpc_vertical", 5.0))

    def _reference(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Keep the baseline phase behavior; tune racing reference advance independently."""
        advance = float(self.config.controller.tracking_advance)
        if self.phase != "race" or advance == 0:
            return super()._reference(obs)
        distance = self.curve.distance[self.progress]
        now = np.interp(distance, self.curve.distance, self.curve.time)
        times = np.minimum(now + advance + PREDICTION_DT * np.arange(1, HORIZON + 1), self.duration)
        return self.curve(times), self.curve(times, nu=1)
