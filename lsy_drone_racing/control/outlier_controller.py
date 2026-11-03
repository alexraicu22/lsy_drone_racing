"""Experiment: raise only the slow tight-corridor fallback's speed cap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lsy_drone_racing.control.profile_controller import ProfileController, retime

if TYPE_CHECKING:
    import numpy as np

    from lsy_drone_racing.control.racing_spline_controller import SpeedProfile


class OutlierController(ProfileController):
    """Keep the frozen racing line and regular pace; speed up narrow-route recovery."""

    def _follow(self, profile: SpeedProfile, centre_knots: np.ndarray, tight: bool) -> None:
        if tight:
            # Tight routes can be installed by crossing recovery, after the last
            # approach observation. Use the installed route's actual start speed.
            obs = {"vel": profile(0, nu=1), "gates_visited": self.scene.gate_margin <= 0.1001}
            profile = retime(
                profile,
                centre_knots,
                obs,
                self.scene,
                maximum_speed=0.85,
                acceleration_limit=4.0,
                gate_speed=0.85,
                sensor_speed=0.85,
                vertical_limit=3.0,
            )
        super()._follow(profile, centre_knots, tight)
