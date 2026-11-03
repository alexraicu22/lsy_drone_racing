"""Retry checked moving exits during inspection; keep the racing-line baseline intact.

pixi run python scripts/sim.py --config=level2.toml --controller=flow_racing_controller.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.racing_line_controller import RacingLineController

if TYPE_CHECKING:
    from ml_collections import ConfigDict

RETRY_INTERVAL = 0.25  # Simulation seconds between attempts, rather than every control tick.


class FlowRacingController(RacingLineController):
    """Leave inspection as soon as a checked route fits the measured moving state."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Preserve the existing speed profile, sensing margins and MPC settings."""
        self.flow_retry = 0.0
        self.moving_resumes = 0
        super().__init__(obs, info, config)

    def _resume(self, obs: dict) -> None:
        """Prefer a continuous route, then a short crossing; keep the fallback on failure."""
        old = self.__dict__.copy()
        for plan in [self._approach, self._cross]:
            try:
                plan(obs)
            except ValueError:
                pass
            else:
                if self.phase in {"race", "cross"} and not self.holding:
                    self.moving_resumes += 1
                    return
            # Planning replaces route fields. Restore them before trying another
            # option so a rejected proposal cannot alter the active trajectory.
            self.__dict__.update(old)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Do not wait for near-zero velocity before retrying an inspected gate exit."""
        obs = {key: np.asarray(value) for key, value in obs.items()}
        self.flow_retry = max(0.0, self.flow_retry - self.dt)
        passed = int(obs["n_gates_passed"])
        if (
            self.phase == "inspect"
            and not self.holding
            and self.flow_retry == 0
            and passed < len(obs["gate_sequence"])
            and passed == self.entry
            and obs["gates_visited"][int(obs["gate_sequence"][passed])]
        ):
            # The inherited controller also processes sensing updates and
            # revalidates the route against newly revealed objects.
            self._resume(obs)
            self.flow_retry = RETRY_INTERVAL
        return super().compute_control(obs, info)
