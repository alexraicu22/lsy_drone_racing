"""Candidate 3 selected from the simulator search, with its tuning frozen in code."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from lsy_drone_racing.control.tuned_racing_controller import TunedRacingController

if TYPE_CHECKING:
    from ml_collections import ConfigDict

PARAMETERS = {
    "profile_acceleration": 21.407867597640234,
    "profile_mpc_acceleration": 17.99261808968027,
    "profile_gate_speed": 2.844462281477182,
    "profile_sensor_speed": 2.1048238434860127,
    "tracking_advance": 0.012475951490443402,
    "tracking_velocity_weight": 2.647005144615795,
    "tracking_smoothing_weight": 1.2491277557027947,
    "profile_gate_offset": 0.0032064856143038424,
}


class Candidate3Controller(TunedRacingController):
    """Apply the selected parameters without changing the standard Level 2 environment."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep caller configuration intact and freeze the search result for submission."""
        config = copy.deepcopy(config)
        config.controller.update(PARAMETERS)
        super().__init__(obs, info, config)
