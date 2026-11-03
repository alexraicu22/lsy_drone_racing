"""Use progress MPC only on observed, roomy sections with small tracking error."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.predictive_spline_controller import PREDICTION_DT
from lsy_drone_racing.control.progress_controller import ProgressMPC
from lsy_drone_racing.control.tracking_controller import TrackingController

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
    from lsy_drone_racing.control.spline_controller import Scene


def clear_section(controller: TrackingController, obs: dict) -> bool:
    """Check observed geometry and current error without selecting particular gate numbers."""
    if controller.phase != "race" or controller.tight_corridor or controller.holding:
        return False
    entry = int(obs["n_gates_passed"])
    if entry >= len(obs["gate_sequence"]):
        return False
    gate = int(obs["gate_sequence"][entry])
    settings = controller.config.controller
    if not obs["gates_visited"][gate]:
        if not settings.get("hybrid_unknown_braking", False):
            return False
        # A distant unknown target can still have observed free space before
        # it. Require a clear stopping tail outside its inflated uncertainty.
        velocity = np.asarray(obs["vel"])
        speed = np.linalg.norm(velocity)
        braking = speed * 0.15 + speed**2 / (2 * 4.0) + 0.10
        stop = obs["pos"] + velocity / max(speed, 1e-8) * braking
        if not controller.scene.clear_segment(obs["pos"], stop):
            return False
    route, scene = controller.curve, controller.scene
    distance = float(route.distance[controller.progress])
    tangent = route.path(distance, nu=1)
    tangent /= max(np.linalg.norm(tangent), 1e-8)
    error = obs["pos"] - route.path(distance)
    error -= tangent * float(error @ tangent)
    limit = float(controller.config.controller.get("hybrid_tracking_error", 0.055))
    if np.linalg.norm(error) > limit:
        return False
    length = max(
        float(settings.get("hybrid_lookahead_distance", 0.6)),
        float(settings.get("hybrid_lookahead_seconds", 0.5)) * np.linalg.norm(obs["vel"]),
    )
    samples = route.path(np.linspace(distance, min(route.distance[-1], distance + length), 16))
    unknown_gates = ~np.asarray(obs["gates_visited"], dtype=bool)
    if unknown_gates.any():
        separation = np.linalg.norm(samples[:, None] - scene.gates[None, unknown_gates], axis=-1)
        if separation.min() < float(settings.get("hybrid_unknown_gate_distance", 0.7)):
            return False
    if len(scene.poles):
        separation = np.linalg.norm(samples[:, None, :2] - scene.poles[None, :, :2], axis=-1)
        separation -= scene.pole_radius[None, :]
        separation[samples[:, None, 2] > scene.poles[None, :, 2] + 0.15] = np.inf
        margin = float(controller.config.controller.get("hybrid_pole_margin", 0.08))
        if separation.min() < margin:
            return False
        if (~scene.poles_known).any() and separation[:, ~scene.poles_known].min() < 0.35:
            return False
    return True


class HybridMPC(ProgressMPC):
    """Enter progress mode after a stable interval; immediately leave it on a failed guard."""

    def __init__(self, settings: ConfigDict):
        """Keep both solvers' existing calibrated dynamics and force bounds."""
        super().__init__(settings)
        self.enabled = False
        self.clear_seconds = 0.0
        self.cooldown = 0.0
        self.observed_route = None
        self.guarded_scene = None
        self.guarded_original = None
        self.switches = 0

    def _disable(self) -> None:
        if self.enabled:
            self.switches += 1
        self.enabled = False
        self.clear_seconds = 0.0
        self.cooldown = float(self.settings.get("hybrid_cooldown", 0.20))

    def update_mode(
        self, route: SpeedProfile | None, start: float, scene: Scene, eligible: bool, dt: float
    ) -> None:
        """New routes and unsafe observations override the hold period immediately."""
        self.cooldown = max(0.0, self.cooldown - dt)
        if route is not self.observed_route:
            self.observed_route = route
            self._disable()
        if not eligible:
            self._disable()
        elif self.cooldown == 0 and not self.enabled:
            self.clear_seconds += dt
            if self.clear_seconds >= float(self.settings.get("hybrid_enter_delay", 0.10)):
                self.enabled = True
                self.switches += 1
        self.set_route(route if self.enabled else None, start, scene)
        if self.guarded_original is not scene:
            self.guarded_original = scene
            self.guarded_scene = copy.deepcopy(scene)
            extra = float(self.settings.get("hybrid_prediction_margin", 0.02))
            self.guarded_scene.gate_margin += extra
            self.guarded_scene.pole_radius += extra

    def _optimize(self, state: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Require extra predicted clearance and bounded cross-track error before using progress."""
        try:
            control, prediction = super()._optimize(state, previous)
            count = max(
                1,
                min(
                    len(prediction),
                    int(
                        np.ceil(
                            float(self.settings.get("hybrid_prediction_seconds", 1.0))
                            / PREDICTION_DT
                        )
                    ),
                ),
            )
            target = self.route.path(self.last_progress[:count])
            tangent = self.route.path(self.last_progress[:count], nu=1)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
            error = prediction[:count] - target
            error -= tangent * np.sum(error * tangent, axis=1, keepdims=True)
            maximum = float(self.settings.get("hybrid_prediction_error", 0.08))
            if np.linalg.norm(error, axis=1).max() > maximum:
                raise ValueError("Progress prediction exceeds the hybrid tracking margin")
            # The progress solver already checks dense segments with normal margins.
            # Add a cushion here before handing control away from the proven tracker.
            starts = np.vstack([state[0], prediction[: count - 1]])
            probes = (
                starts[:, None]
                + (prediction[:count] - starts)[:, None] * np.linspace(0, 1, 5)[None, :, None]
            )
            if not self.guarded_scene.free(probes.reshape(-1, 3), start=state[0]).all():
                raise ValueError("Progress prediction lacks the hybrid clearance cushion")
            return control, prediction
        except (ValueError, np.linalg.LinAlgError):
            self._disable()
            raise


class HybridController(TrackingController):
    """Switch by public geometry and tracking quality, preserving the existing fallback."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep all other controllers available independently."""
        super().__init__(obs, info, config)
        self.mpc = HybridMPC(config.controller)

    def _reference(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        reference = super()._reference(obs)
        racing = self.phase == "race" and not self.tight_corridor
        route = self.curve if racing else None
        start = float(route.distance[self.progress]) if racing else 0.0
        self.mpc.update_mode(route, start, self.scene, clear_section(self, obs), self.dt)
        return reference

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Expose use of the progress branch in the standard local evaluation CSV."""
        command = super().compute_control(obs, info)
        self.trajectory_attempts = self.mpc.attempts
        self.trajectory_accepts = self.mpc.accepts
        return command
