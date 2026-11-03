"""Plan acceleration-limited pace along the whole route instead of changing clock speed."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.calibrated_controller import CalibratedController, CalibratedMPC
from lsy_drone_racing.control.continuous_spline_controller import continuous_route
from lsy_drone_racing.control.racing_line_controller import optimize_line

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from scipy.interpolate import BPoly

    from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
    from lsy_drone_racing.control.spline_controller import Scene


def retime(
    profile: SpeedProfile,
    knots: np.ndarray,
    obs: dict,
    scene: Scene,
    maximum_speed: float,
    acceleration_limit: float,
    gate_speed: float,
    sensor_speed: float = 1.5,
    vertical_limit: float | None = None,
) -> SpeedProfile:
    """Anticipate gate-speed limits and share acceleration between turning and braking."""
    distance = profile.distance
    position = profile.path(distance)
    probes = distance[:-1, None] + np.diff(distance)[:, None] * np.linspace(0, 1, 9)
    second = np.linalg.norm(profile.path(probes, nu=2), axis=-1).max(axis=-1)
    tangent = np.linalg.norm(profile.path(probes, nu=1), axis=-1).max(axis=-1)
    vertical_tangent = profile.path(probes, nu=1)[..., 2]
    vertical_second = profile.path(probes, nu=2)[..., 2]
    second = np.maximum(np.r_[second[0], second], np.r_[second, second[-1]])
    tangent = np.maximum(np.r_[tangent[0], tangent], np.r_[tangent, tangent[-1]])
    # Keep a reserve for simultaneous forward and sideways acceleration.
    speed = np.minimum(
        maximum_speed / np.maximum(tangent, 1),
        np.sqrt(0.75 * acceleration_limit / np.maximum(second, 1e-6)),
    )
    if vertical_limit is not None:
        bound = np.abs(vertical_second).max(axis=-1)
        bound = np.maximum(np.r_[bound[0], bound], np.r_[bound, bound[-1]])
        speed = np.minimum(speed, np.sqrt(0.75 * vertical_limit / np.maximum(bound, 1e-6)))
    for knot in knots:
        near = np.abs(distance - profile.knot_distance[knot]) < 0.24
        speed[near] = np.minimum(speed[near], gate_speed)
    for gate in np.flatnonzero(~np.asarray(obs["gates_visited"], bool)):
        near = np.linalg.norm(position[:, :2] - scene.gates[gate, :2], axis=-1) < 1.0
        speed[near] = np.minimum(speed[near], sensor_speed)
    narrow = scene.poles_known & (scene.pole_radius < 0.20)
    if narrow.any():
        clearance = np.linalg.norm(position[:, None, :2] - scene.poles[None, narrow, :2], axis=-1)
        clearance -= scene.pole_radius[None, narrow]
        clearance[position[:, None, 2] > scene.poles[None, narrow, 2] + 0.15] = np.inf
        speed = np.minimum(speed, np.clip(0.8 + 3 * clearance.min(axis=-1), 0.6, maximum_speed))
    unknown = ~scene.poles_known
    if unknown.any():
        clearance = np.linalg.norm(position[:, None, :2] - scene.poles[None, unknown, :2], axis=-1)
        clearance -= scene.pole_radius[None, unknown]
        clearance[position[:, None, 2] > scene.poles[None, unknown, 2] + 0.15] = np.inf
        speed = np.minimum(speed, np.clip(1 + 3 * clearance.min(axis=-1), 0.8, maximum_speed))
    speed[0] = min(speed[0], np.linalg.norm(obs["vel"]))
    speed[-1] = 0.0
    spacing = np.diff(distance)
    for _ in range(4):
        for indices in [range(len(speed) - 1), range(len(speed) - 2, -1, -1)]:
            forward = indices.step > 0
            for index in indices:
                source, target = (index, index + 1) if forward else (index + 1, index)
                lateral = second[source] * speed[source] ** 2
                available = np.sqrt(max(0, acceleration_limit**2 - lateral**2))
                available = min(0.8 * acceleration_limit, available / max(1, tangent[source]))
                if vertical_limit is not None:
                    direction = vertical_tangent[index] * (1 if forward else -1)
                    first = vertical_second[index] * speed[source] ** 2
                    last = vertical_second[index] * speed[target] ** 2
                    valid = np.abs(direction) > 1e-6
                    numerator = np.where(
                        direction > 0,
                        vertical_limit - np.maximum(first, last),
                        -vertical_limit - np.minimum(first, last),
                    )
                    if valid.any():
                        available = min(
                            available, max(0, np.min(numerator[valid] / direction[valid]))
                        )
                speed[target] = min(
                    speed[target], np.sqrt(speed[source] ** 2 + 2 * available * spacing[index])
                )
    intervals = 2 * spacing / np.maximum(speed[:-1] + speed[1:], 1e-8)
    result = copy.copy(profile)
    result.speed = speed
    result.time = np.r_[0, np.cumsum(intervals)]
    result.acceleration = np.diff(speed) / intervals
    result.x = np.array([0.0, result.time[-1]])
    # Acceleration is quadratic within each arc-length interval. Bound its
    # entire curve using subdivided Bezier control points, including short peaks
    # a uniform time grid can miss.
    fraction = np.array([0.0, 0.5, 1.0 - 1e-9])
    samples = distance[:-1, None] + spacing[:, None] * fraction
    squared_speed = speed[:-1, None] ** 2 + np.diff(speed**2)[:, None] * fraction
    acceleration = (
        profile.path(samples, nu=2) * squared_speed[..., None]
        + profile.path(samples, nu=1) * result.acceleration[:, None, None]
    )
    controls = acceleration.copy()
    controls[:, 1] = 2 * acceleration[:, 1] - 0.5 * (acceleration[:, 0] + acceleration[:, 2])
    for _ in range(3):
        first = 0.5 * (controls[:, 0] + controls[:, 1])
        second = 0.5 * (controls[:, 1] + controls[:, 2])
        middle = 0.5 * (first + second)
        controls = np.concatenate(
            [
                np.stack([controls[:, 0], first, middle], axis=1),
                np.stack([middle, second, controls[:, 2]], axis=1),
            ]
        )
    ratio = np.linalg.norm(controls, axis=-1).max() / acceleration_limit
    if vertical_limit is not None:
        ratio = max(ratio, np.abs(controls[..., 2]).max() / vertical_limit)
    if ratio > 1:
        factor = np.sqrt(1 / ratio) * 0.995
        result.speed = speed * factor
        result.time /= factor
        result.acceleration *= factor**2
        result.x[1] /= factor
    return result


class ProfileController(CalibratedController):
    """Use a single consistent timed reference with braking planned before tight gates."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Keep online gate updates and calibrated acceleration tracking."""
        self.planning_obs = obs
        self.profile_retry = 0.0
        super().__init__(obs, info, config)
        self.pace = 1.0
        self.adaptive_pace = False
        self.reference_pace = self.speed_factor = 1.0
        self.mpc = CalibratedMPC(
            acceleration_limit=float(config.controller.get("profile_mpc_acceleration", 10.0)),
            vertical_limit=float(config.controller.get("profile_mpc_vertical", 5.0)),
        )

    def _approach(self, obs: dict) -> None:
        self.planning_obs = obs
        super()._approach(obs)

    def _follow(self, profile: SpeedProfile, centre_knots: np.ndarray, tight: bool) -> None:
        if not tight:
            profile = retime(
                profile,
                centre_knots,
                self.planning_obs,
                self.scene,
                float(self.config.controller.get("profile_max_speed", 4.0)),
                float(self.config.controller.get("profile_acceleration", 8.5)),
                float(self.config.controller.get("profile_gate_speed", 1.8)),
                float(self.config.controller.get("profile_sensor_speed", 1.5)),
                self.config.controller.get("profile_vertical_acceleration"),
            )
        super()._follow(profile, centre_knots, tight)

    def _geometry(self, obs: dict) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray]:
        margin = float(self.config.controller.get("profile_gate_margin", 0.10))
        self.scene.gate_margin[np.asarray(obs["gates_visited"], bool)] = margin
        self.scene.pole_radius[self.scene.poles_known] = float(
            self.config.controller.get("calibrated_pole_radius", 0.22)
        )
        unknown = self.config.controller.get("calibrated_unknown_radius")
        if unknown is not None:
            self.scene.pole_radius[~self.scene.poles_known] = float(unknown)
        for entry in range(self.entry, len(obs["gate_sequence"])):
            gate = int(obs["gate_sequence"][entry])
            if not obs["gates_visited"][gate]:
                break
            distances = np.linalg.norm(
                self.scene.poles[:, :2] - self.scene.gates[gate, :2], axis=-1
            )
            allowed = np.maximum(0.13, distances - 0.04)
            known = self.scene.poles_known
            self.scene.pole_radius[known] = np.minimum(
                self.scene.pole_radius[known], allowed[known]
            )
        geometry, centres, knots = continuous_route(
            obs, self.scene, self.entry, close_entrances=True
        )
        optimized, gain = optimize_line(
            geometry,
            knots,
            obs,
            self.scene,
            self.speed_scale,
            self.acceleration_scale,
            max_gate_offset=float(self.config.controller.get("profile_gate_offset", 0.05)),
        )
        if gain > 0:
            self.line_optimizations += 1
            return optimized, tuple(optimized(optimized.x[knots])), knots
        return geometry, centres, knots

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Resume a timed route when an approach fallback no longer needs to stop."""
        self.profile_retry = max(0.0, self.profile_retry - self.dt)
        passed = int(obs["n_gates_passed"])
        if (
            self.phase == "approach"
            and not self.holding
            and self.profile_retry == 0
            and passed < len(obs["gate_sequence"])
            and obs["gates_visited"][int(obs["gate_sequence"][passed])]
            and np.linalg.norm(obs["vel"]) < 1.6
        ):
            old = self.__dict__.copy()
            try:
                self._approach({key: np.asarray(value) for key, value in obs.items()})
            except ValueError:
                self.__dict__.update(old)
            else:
                if self.phase != "race":
                    self.__dict__.update(old)
            self.profile_retry = 0.3
        return super().compute_control(obs, info)
