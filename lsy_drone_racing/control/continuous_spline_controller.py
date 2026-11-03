"""Continuous gate lookahead with the existing full-state flight controller.

pixi run python scripts/sim.py --config=level2.toml --controller=continuous_spline_controller.py
Unknown gates are approached for sensing, never crossed at their nominal pose.
The older spline controllers and RL experiments remain available separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BPoly

from lsy_drone_racing.control.racing_spline_controller import (
    GATE_SPEED,
    TRACKING_ACCELERATION,
    RacingSplineController,
    SpeedProfile,
)
from lsy_drone_racing.control.spline_controller import APPROACH_DISTANCE, Scene, gate_points

if TYPE_CHECKING:
    from ml_collections import ConfigDict

CRUISE_SPEED = 2.2
DEFAULT_ACCELERATION_SCALE = 2.25
SENSING_SPEED = 1.2
LOOKAHEAD = 0.10
EXTRA_POLE_CLEARANCE = 0.08
VELOCITY_FEEDFORWARD = 0.9
TIGHT_POLE_RADIUS = 0.14
TIGHT_SPEED = 0.5


def safe_sensing_speed(curve: BPoly, scene: Scene, requested: float) -> float:
    """Allow a moving horizon end only if its straight braking tail remains clear."""
    if requested <= 0:
        return 0.0
    end = curve(curve.x[-1])
    direction = end - curve(max(0.0, curve.x[-1] - 0.05))
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return 0.0
    direction /= norm
    for speed in [requested, 0.75 * requested, 0.5 * requested]:
        braking_distance = speed**2 / (2 * TRACKING_ACCELERATION) + 0.05
        if scene.clear_segment(end, end + direction * braking_distance):
            return speed
    return 0.0


def continuous_route(
    obs: dict,
    scene: Scene,
    entry: int,
    gate_headings: dict[int, np.ndarray] | None = None,
    close_entrances: bool = False,
) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray]:
    """Connect ordered gate corridors and detours into one collision-checked spline."""
    points = [np.asarray(obs["pos"], dtype=float)]
    directions = [np.asarray(obs["vel"], dtype=float)]
    centres = []
    centre_knots = []

    def append(point: np.ndarray, direction: np.ndarray | None = None) -> None:
        if np.linalg.norm(point - points[-1]) > 0.01:
            points.append(point)
            directions.append(direction)

    for target in range(entry, len(obs["gate_sequence"])):
        gate = int(obs["gate_sequence"][target])
        approach, centre, _ = gate_points(obs, target)
        normal = (centre - approach) / APPROACH_DISTANCE
        if gate_headings is not None and target in gate_headings:
            normal = np.asarray(gate_headings[target])
            normal = normal / np.linalg.norm(normal)
        known = bool(obs["gates_visited"][gate])
        # Unknown frames have no guaranteed opening. Stop before them if sensing
        # does not reveal a usable pose while we approach.
        distances = [0.40, 0.30, 0.22, 0.60, 0.15] if known else [0.48, 0.60, 0.36, 0.70, 0.30]
        if known and close_entrances:
            distances.extend([0.10, 0.06])
        entrance = centre - distances[0] * normal
        for distance in distances:
            candidate = centre - distance * normal
            if scene.free(candidate)[0] and (not known or scene.clear_segment(candidate, centre)):
                entrance = candidate
                break
        start = points[-1]
        checked_count = len(points)
        aligned = (
            target == entry
            and known
            and -0.40 < np.dot(start - centre, normal) < -0.02
            and np.linalg.norm((start - centre) - normal * np.dot(start - centre, normal)) < 0.20
        )
        try:
            if not aligned:
                for point in scene.route(start, entrance)[1:]:
                    append(point)
                directions[-1] = normal
            if not known:
                break
            if not scene.clear_segment(points[-1], centre):
                raise ValueError("Gate entrance is blocked")
            append(centre, normal)
            centre_knot = len(points) - 1
            # Keep moving beyond the plane; prefer enough room to start the next
            # turn, but accept a shorter exit if a pole is close behind the gate.
            for distance in [0.35, 0.24, 0.18, 0.12, 0.06]:
                exit_point = centre + distance * normal
                if scene.clear_segment(centre, exit_point):
                    append(exit_point, normal)
                    break
            else:
                raise ValueError("Gate exit is blocked")
            centres.append(centre)
            centre_knots.append(centre_knot)
        except ValueError:
            if not centres:
                raise
            points = points[:checked_count]
            directions = directions[:checked_count]
            break  # End at the last checked exit; retry as new objects are revealed.

    if len(points) < 2:
        raise ValueError("No moving route")
    points = np.asarray(points)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    times = np.concatenate([[0], np.cumsum(np.maximum(lengths, 0.05))])
    # Gate tangents follow their required crossing direction. Other tangents
    # blend detours. Reduce corner blending if the smooth curve cuts a margin.
    for blend in [0.8, 0.4, 0.15, 0.0]:
        velocities = np.zeros_like(points)
        initial_velocity = np.asarray(obs["vel"])
        velocities[0] = initial_velocity / max(np.linalg.norm(initial_velocity), 1.0)
        for index in range(1, len(points) - 1):
            direction = directions[index]
            if direction is None:
                direction = points[index + 1] - points[index - 1]
                magnitude = blend
            else:
                magnitude = max(blend, 0.15)
            velocities[index] = magnitude * direction / max(np.linalg.norm(direction), 1e-8)
        curve = BPoly.from_derivatives(
            times, [[point, velocity, np.zeros(3)] for point, velocity in zip(points, velocities)]
        )
        samples = np.linspace(0, times[-1], max(100, int(times[-1] / 0.008)))
        if scene.free(curve(samples), start=points[0]).all():
            return curve, tuple(centres), np.asarray(centre_knots, dtype=int)
    raise ValueError("Continuous curve cannot maintain clearance")


class ContinuousSplineController(RacingSplineController):
    """Carry velocity through known gates and replan when public sensing changes."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Read independent speed and acceleration settings for reproducible sweeps."""
        self.speed_scale = float(config.controller.get("speed_scale", 1.0))
        self.acceleration_scale = float(
            config.controller.get("acceleration_scale", DEFAULT_ACCELERATION_SCALE)
        )
        if not all(
            np.isfinite(value) and value > 0
            for value in [self.speed_scale, self.acceleration_scale]
        ):
            raise ValueError("Speed and acceleration scales must be positive and finite")
        self.fallback_counts = {}
        self.transition_retry = 0.0
        self.tight_corridor = False
        self.speed_factor = 1.0  # Optional speed policy; 1 preserves classical behavior.
        super().__init__(obs, info, config)

    def _approach(self, obs: dict) -> None:
        self.entry = int(obs["n_gates_passed"])
        self.scene = Scene(obs, self.config)
        # Continuous turns need more tracking room than stopped moves.
        self.scene.pole_radius += EXTRA_POLE_CLEARANCE
        try:
            geometry, centres, centre_knots = self._geometry(obs)
            terminal_speed = 0.0
            next_entry = self.entry + len(centres)
            if next_entry < len(obs["gate_sequence"]):
                gate = int(obs["gate_sequence"][next_entry])
                if not obs["gates_visited"][gate]:
                    terminal_speed = safe_sensing_speed(
                        geometry, self.scene, getattr(self, "sensing_terminal_speed", 0.0)
                    )
            profile = SpeedProfile(
                geometry,
                self.scene,
                np.linalg.norm(obs["vel"]),
                CRUISE_SPEED * self.speed_scale,
                centres,
                sensor_speed=SENSING_SPEED * self.speed_scale,
                gate_speed=GATE_SPEED * self.speed_scale,
                acceleration_scale=self.acceleration_scale,
                terminal_speed=terminal_speed,
            )
        except ValueError as error:
            reason = str(error)
            self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1
            if self._tight_route(obs):
                return
            super()._approach(obs)
            return
        self._follow(profile, centre_knots, tight=False)

    def _geometry(self, obs: dict) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray]:
        """Build the gate corridors; specialized planners can improve the racing line."""
        return continuous_route(obs, self.scene, self.entry)

    def _follow(self, profile: SpeedProfile, centre_knots: np.ndarray, tight: bool) -> None:
        """Install a checked route and its ordered crossing locations."""
        self.tight_corridor = tight
        self.curve, self.duration = profile, float(profile.x[-1])
        self.goal = profile(self.duration)
        self.elapsed = 0.0
        self.progress = 0
        self.route_positions = profile.path(profile.distance)
        self.planned_entry = self.entry
        # Track the actual ordered knots, including gates used twice in one lap.
        self.crossing_indices = np.searchsorted(
            profile.distance, profile.knot_distance[centre_knots]
        )
        self.phase = "race"
        self.holding = False
        self.hold_position = None
        self.replans += 1

    def _tight_route(self, obs: dict) -> bool:
        """Use a smaller tracking cushion at low speed around already revealed poles."""
        gate = int(obs["gate_sequence"][self.entry])
        if not obs["gates_visited"][gate] or np.linalg.norm(obs["vel"]) > 0.7:
            return False
        scene = Scene(obs, self.config)
        scene.pole_radius[scene.poles_known] = TIGHT_POLE_RADIUS
        try:
            geometry, _, centre_knots = continuous_route(obs, scene, self.entry)
            if not len(centre_knots):
                return False
            profile = SpeedProfile(
                geometry,
                scene,
                np.linalg.norm(obs["vel"]),
                TIGHT_SPEED,
                sensor_speed=TIGHT_SPEED,
                gate_speed=TIGHT_SPEED,
            )
        except ValueError:
            return False
        self.scene = scene
        self._follow(profile, centre_knots, tight=True)
        return True

    def _cross(self, obs: dict) -> None:
        """Avoid cycling inspection points when only the straight exit is blocked."""
        try:
            super()._cross(obs)
        except ValueError:
            if not self._tight_route(obs):
                raise
        else:
            if self.phase != "cross":
                self._tight_route(obs)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Follow actual route progress and the planned speed at that distance."""
        passed = int(obs["n_gates_passed"])
        clear_of_poles = not len(self.scene.poles) or np.all(
            np.linalg.norm(np.asarray(obs["pos"])[:2] - self.scene.poles[:, :2], axis=-1)
            > self.scene.pole_radius + 0.18
        )
        leave_tight_route = (
            self.phase == "race"
            and self.tight_corridor
            and passed > self.planned_entry
            and clear_of_poles
        )
        if passed < len(obs["gate_sequence"]) and (
            self.phase != "race" and self.entry < passed or leave_tight_route
        ):
            # A confirmed crossing is enough to start the next section. The
            # stopped fallback used to finish braking at its exit point first.
            try:
                self._approach(obs)
            except ValueError:
                self.phase = "approach"
                self.holding = True
                self.retry_delay = 0.0
            self.signature = self._signature(obs)
        self.transition_retry = max(0.0, self.transition_retry - self.dt)
        if (
            self.phase == "approach"
            and not self.holding
            and passed < len(obs["gate_sequence"])
            and self.transition_retry == 0
        ):
            approach, centre, _ = gate_points(obs, passed)
            normal = (centre - approach) / APPROACH_DISTANCE
            offset = np.asarray(obs["pos"]) - centre
            lateral = offset - normal * np.dot(offset, normal)
            lateral_velocity = np.asarray(obs["vel"]) - normal * np.dot(obs["vel"], normal)
            gate = int(obs["gate_sequence"][passed])
            near_goal = np.linalg.norm(np.asarray(obs["pos"]) - self.goal) < 0.08
            aligned = (
                bool(obs["gates_visited"][gate])
                and not self.staging
                and -0.55 < np.dot(offset, normal) < -0.03
                and np.linalg.norm(lateral) < 0.04
                and np.linalg.norm(lateral_velocity) < 0.12
                and np.linalg.norm(obs["vel"]) < 0.65
            )
            if near_goal and aligned:
                self.transition_retry = 0.2
                try:
                    self._cross(obs)
                except ValueError:
                    pass  # Keep the checked approach while sensing or braking continues.
        if self.phase != "race":
            return super().compute_control(obs, info)
        signature = self._signature(obs)
        changed = signature != self.signature
        self.signature = signature
        if passed >= len(obs["gate_sequence"]):
            return np.concatenate([obs["pos"], np.zeros(6), [0, 0, 0, 1], np.zeros(3)])
        if changed or self.elapsed >= self.duration:
            try:
                if self.elapsed >= self.duration:
                    # A sensing horizon can end without revealing a usable exit.
                    # Use alignment/recovery instead of rebuilding a tiny route every tick.
                    super()._approach(obs)
                else:
                    self._approach(obs)
            except ValueError:
                self.phase = "approach"
                self.holding = True
                self.retry_delay = 0.0
            if self.phase != "race":
                return super().compute_control(obs, info)
        self.entry = passed
        # Project onto a local part of the route. Restrict progress at the next
        # required gate until the environment confirms its actual crossing.
        profile = self.curve
        end = min(len(profile.distance), self.progress + 35)
        gate_index = passed - self.planned_entry
        if gate_index < len(self.crossing_indices):
            end = min(end, self.crossing_indices[gate_index] + 4)
        end = max(end, self.progress + 1)
        candidates = self.route_positions[self.progress : end]
        nearest = int(np.argmin(np.linalg.norm(candidates - obs["pos"], axis=-1)))
        self.progress += nearest
        distance = float(profile.distance[self.progress])
        remaining = profile.distance[-1] - distance
        lookahead = min(
            distance + (0.05 if self.tight_corridor else LOOKAHEAD), profile.distance[-1]
        )
        position = profile.path(lookahead)
        tangent = profile.path(lookahead, nu=1)
        tangent /= max(np.linalg.norm(tangent), 1e-8)
        # The position carrot helps tracking; leave some speed headroom for that
        # correction. The existing flight controller supplies acceleration.
        speed = VELOCITY_FEEDFORWARD * float(np.interp(lookahead, profile.distance, profile.speed))
        factor = min(self.speed_factor, 1.0) if self.tight_corridor else self.speed_factor
        speed *= factor
        speed = min(
            speed,
            np.sqrt(
                profile.speed[-1] ** 2
                + 2 * TRACKING_ACCELERATION * self.acceleration_scale * remaining
            ),
        )
        velocity = tangent * speed
        self.elapsed = float(np.interp(distance, profile.distance, profile.time))
        if remaining < 0.05:
            self.elapsed = self.duration
        return np.concatenate(
            [position - self.tracking_offset, velocity, np.zeros(3), [0, 0, 0, 1], np.zeros(3)]
        ).astype(np.float32)
