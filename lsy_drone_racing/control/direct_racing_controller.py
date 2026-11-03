"""Run direct PPO racing through scripts/sim.py.

Uses the default [env] control_mode = "state":
pixi run python scripts/sim.py --config=level0.toml --controller=direct_racing_controller.py

Default checkpoint: saves/velocity_simple_level0_2m/final.zip.
Set DIRECT_RACING_MODEL to use another checkpoint path.
The standard sim.py ends at 30 seconds; the RL script allows 120 for this experiment.
"""

import os

import numpy as np
from ml_collections import ConfigDict

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.direct_racing_rl import (
    DEFAULT_MODEL,
    GateProgress,
    Observations,
    VelocityCommands,
    load_policy,
)


class DirectRacingController(Controller):
    """Use the trained racing policy without learning during a race."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Load the checkpoint and initialize shared policy preprocessing."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "state":
            raise ValueError('Velocity racing needs [env] control_mode = "state"')
        self.model = load_policy(os.environ.get("DIRECT_RACING_MODEL", str(DEFAULT_MODEL)))
        if self.model.racing_frequency != config.env.freq:
            raise ValueError("Checkpoint control frequency differs from environment")
        self.encoder = Observations()
        self.commands = VelocityCommands(obs, config.env.freq)
        self.takeoff_remaining = self.commands.takeoff_steps
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.first_step = True
        self.progress = None
        self.dt = 1 / config.env.freq

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Convert the deterministic policy output into a flight command."""
        if self.takeoff_remaining > 0:
            self.takeoff_remaining -= 1
            return self.commands.command()
        if self.first_step:
            self.progress = GateProgress(obs)
        features = self.encoder.encode(
            obs,
            self.previous_action,
            self.commands.state(obs),
            self.progress.state(),
            reset=self.first_step,
        )
        self.first_step = False
        action, _ = self.model.predict(features, deterministic=True)
        self.previous_action = np.clip(action, -1, 1).astype(np.float32)
        return self.commands.step(self.previous_action, obs)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Use the same progress timeout as training, after automatic takeoff."""
        if self.first_step or terminated or truncated:
            return False
        return self.progress.update(obs, self.dt)
