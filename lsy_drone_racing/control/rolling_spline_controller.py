"""Keep moving through sensing waypoints when a checked braking tail is available.

pixi run python scripts/sim.py --config=level2.toml --controller=rolling_spline_controller.py
Experimental: the initial trial did not improve lap times; not the selected controller.
"""

from ml_collections import ConfigDict

from lsy_drone_racing.control.continuous_spline_controller import ContinuousSplineController


class RollingSplineController(ContinuousSplineController):
    """Use a moving planning horizon instead of stopping at every sensing endpoint."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Preserve the baseline; request up to 0.6 m/s at safe sensing endpoints."""
        self.sensing_terminal_speed = 0.6
        super().__init__(obs, info, config)
