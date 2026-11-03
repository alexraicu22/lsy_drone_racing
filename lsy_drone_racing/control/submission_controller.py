"""Selected fast controller with tuning in code for the standard Level 2 evaluator."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from lsy_drone_racing.control.recovery_controller import RecoveryController

if TYPE_CHECKING:
    from ml_collections import ConfigDict


class SubmissionController(RecoveryController):
    """Use the selected fast settings without adding options to level2.toml."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep the supplied environment configuration and tune only this controller."""
        config = copy.deepcopy(config)
        config.controller.update(
            profile_sensor_speed=2.0,
            calibrated_unknown_radius=0.37,
            calibrated_online_thrust=True,
            profile_acceleration=16.0,
            profile_max_speed=6.0,
            profile_gate_speed=2.8,
            profile_mpc_acceleration=20.0,
        )
        super().__init__(obs, info, config)
