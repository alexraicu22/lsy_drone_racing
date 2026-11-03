"""Spline racing with a curvature, acceleration, braking, and clearance speed profile.

pixi run python scripts/sim.py --config=level2.toml --controller=racing_spline_controller.py
The original spline_controller.py remains the conservative baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control.spline_controller import (
    APPROACH_DISTANCE,
    Scene,
    SplineController,
    gate_points,
    make_route,
)

if TYPE_CHECKING:
    from scipy.interpolate import BPoly

MAX_SPEED = 2.6
GATE_SPEED = 1.2
LONGITUDINAL_ACCELERATION = 3.0
LATERAL_ACCELERATION = 2.5
TRACKING_ACCELERATION = 1.8
TRACKING_LATERAL_ACCELERATION = 1.5


class SpeedProfile:
    """Traverse fixed geometry quickly, slowing for curvature and braking in advance."""

    def __init__(
        self,
        curve: BPoly,
        scene: Scene,
        initial_speed: float,
        speed_limit: float,
        gate_centres: tuple[np.ndarray, ...] = (),
        sensor_speed: float | None = None,
        gate_speed: float = GATE_SPEED,
        acceleration_scale: float = 1.0,
        terminal_speed: float = 0.0,
        max_duration: float | None = None,
    ):
        """Convert the geometric spline into arc length, then perform forward/backward passes."""
        parameter = np.linspace(0, curve.x[-1], max(120, int(curve.x[-1] / 0.01)))
        positions = curve(parameter)
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=-1))])
        keep = np.concatenate([[True], np.diff(arc) > 1e-7])
        sampled_positions = positions[keep]
        arc, parameter = arc[keep], parameter[keep]
        if arc[-1] < 1e-4:
            raise ValueError("Cannot time a stationary path")
        self.knot_distance = np.interp(curve.x, parameter, arc)
        self.distance = np.linspace(0, arc[-1], max(16, int(arc[-1] / 0.025) + 1))
        parameter = np.interp(self.distance, arc, parameter)
        positions = np.column_stack(
            [np.interp(self.distance, arc, sampled_positions[:, axis]) for axis in range(3)]
        )
        tangent = curve(parameter, nu=1)
        norm = np.linalg.norm(tangent, axis=-1)
        tangent = tangent / np.maximum(norm[:, None], 1e-8)
        tangent[0] = (positions[1] - positions[0]) / np.linalg.norm(positions[1] - positions[0])
        tangent[-1] = (positions[-1] - positions[-2]) / np.linalg.norm(
            positions[-1] - positions[-2]
        )
        self.path = CubicHermiteSpline(self.distance, positions, tangent)
        derivative = self.path(self.distance, nu=1)
        second = self.path(self.distance, nu=2)
        curvature = np.linalg.norm(np.cross(derivative, second), axis=-1) / np.maximum(
            np.linalg.norm(derivative, axis=-1) ** 3, 1e-8
        )
        speed = np.minimum(
            speed_limit,
            np.sqrt(LATERAL_ACCELERATION * acceleration_scale / np.maximum(curvature, 1e-6)),
        )
        acceleration_limit = LONGITUDINAL_ACCELERATION * acceleration_scale
        if sensor_speed is not None:
            # Tight gate corridors can put curvature peaks between arc samples.
            # Bound derivatives throughout each interval, not only at its ends.
            probes = self.distance[:-1, None] + np.diff(self.distance)[:, None] * np.linspace(
                0, 1, 9
            )
            second_bound = np.linalg.norm(self.path(probes, nu=2), axis=-1).max(axis=-1)
            tangent_bound = np.linalg.norm(self.path(probes, nu=1), axis=-1).max(axis=-1)
            second_bound = np.maximum(
                np.r_[second_bound[0], second_bound], np.r_[second_bound, second_bound[-1]]
            )
            tangent_bound = np.maximum(
                np.r_[tangent_bound[0], tangent_bound], np.r_[tangent_bound, tangent_bound[-1]]
            )
            speed = np.minimum(
                speed,
                np.sqrt(
                    TRACKING_LATERAL_ACCELERATION
                    * acceleration_scale
                    / np.maximum(second_bound, 1e-6)
                ),
            )
            speed = np.minimum(speed, speed_limit / np.maximum(tangent_bound, 1.0))
            acceleration_limit = (
                TRACKING_ACCELERATION * acceleration_scale / max(float(tangent_bound.max()), 1.0)
            )
        for centre in gate_centres:
            near_gate = np.linalg.norm(positions - centre, axis=-1) < 0.25
            speed[near_gate] = np.minimum(speed[near_gate], gate_speed)
        if sensor_speed is not None:
            for centre, margin in zip(scene.gates, scene.gate_margin):
                if margin > 0.1001:
                    sensing_zone = np.linalg.norm(positions[:, :2] - centre[:2], axis=-1) < 0.95
                    speed[sensing_zone] = np.minimum(speed[sensing_zone], sensor_speed)
        if len(scene.poles):
            clearance = np.linalg.norm(positions[:, None, :2] - scene.poles[None, :, :2], axis=-1)
            clearance -= scene.pole_radius[None, :]
            clearance[positions[:, None, 2] > scene.poles[None, :, 2] + 0.15] = np.inf
            # Curvature limits handle known turns. Keep extra caution around
            # unrevealed poles, whose position bounds are already inflated.
            unknown = ~scene.poles_known
            if unknown.any():
                speed = np.minimum(
                    speed, np.clip(1.0 + 3 * clearance[:, unknown].min(axis=-1), 0.8, speed_limit)
                )
        speed[0] = min(float(initial_speed), speed[0])
        speed[-1] = min(terminal_speed, speed[-1])
        spacing = np.diff(self.distance)
        for index in range(1, len(speed)):
            speed[index] = min(
                speed[index],
                np.sqrt(speed[index - 1] ** 2 + 2 * acceleration_limit * spacing[index - 1]),
            )
        for index in range(len(speed) - 2, -1, -1):
            speed[index] = min(
                speed[index],
                np.sqrt(speed[index + 1] ** 2 + 2 * acceleration_limit * spacing[index]),
            )
        self.speed = speed
        intervals = 2 * spacing / np.maximum(speed[:-1] + speed[1:], 1e-8)
        self.time = np.concatenate([[0], np.cumsum(intervals)])
        self.acceleration = np.diff(speed) / intervals
        self.x = np.array([0.0, self.time[-1]])
        if max_duration is not None and self.x[-1] > max_duration:
            raise ValueError("Route timing exceeds the candidate's time bound")
        samples = self(np.linspace(0, self.x[-1], max(60, int(self.x[-1] / 0.01))))
        if not scene.free(samples, start=positions[0]).all():
            raise ValueError("Time-profile interpolation lost obstacle clearance")

    def __call__(self, time: float | np.ndarray, nu: int = 0) -> np.ndarray:
        """Evaluate position, velocity, or acceleration with the chain rule."""
        time = np.asarray(time)
        clipped = np.clip(time, 0, self.time[-1])
        index = np.clip(
            np.searchsorted(self.time, clipped, side="right") - 1, 0, len(self.speed) - 2
        )
        elapsed = clipped - self.time[index]
        acceleration = self.acceleration[index]
        speed = self.speed[index] + acceleration * elapsed
        distance = (
            self.distance[index] + self.speed[index] * elapsed + 0.5 * acceleration * elapsed**2
        )
        distance = np.clip(distance, self.distance[index], self.distance[index + 1])
        if nu == 0:
            return self.path(distance)
        speed = np.where(time > self.time[-1], 0.0, speed)
        if nu == 1:
            return self.path(distance, nu=1) * speed[..., None]
        if nu == 2:
            distance = np.minimum(
                distance, np.nextafter(self.distance[index + 1], self.distance[index])
            )
            acceleration = np.where(time >= self.time[-1], 0.0, acceleration)
            return (
                self.path(distance, nu=2) * speed[..., None] ** 2
                + self.path(distance, nu=1) * acceleration[..., None]
            )
        raise ValueError("Only derivatives 0, 1, and 2 are supported")


def refine_route(points: np.ndarray, scene: Scene) -> list[np.ndarray]:
    """Shorten detours only when both adjacent segments retain extra clearance."""
    route = np.array(points, copy=True)
    # Keep a tracking cushion rather than moving nodes directly onto obstacle margins.
    original_radius = scene.pole_radius.copy()
    scene.pole_radius = original_radius + 0.04
    try:
        for _ in range(2):
            for index in range(1, len(route) - 1):
                candidate = 0.5 * route[index] + 0.25 * (route[index - 1] + route[index + 1])
                if scene.clear_segment(route[index - 1], candidate) and scene.clear_segment(
                    candidate, route[index + 1]
                ):
                    route[index] = candidate
    finally:
        scene.pole_radius = original_radius
    return list(route)


class RacingSplineController(SplineController):
    """Keep the baseline's sensing and collision logic, improve its racing line and timing."""

    def _approach(self, obs: dict) -> None:
        super()._approach(obs)
        if np.linalg.norm(obs["pos"] - self.goal) < 0.15:
            return  # A short alignment move needs no new speed optimization.
        try:
            route = refine_route(self.curve(self.curve.x), self.scene)
            geometry, _ = make_route(route, np.asarray(obs["vel"]), self.scene)
            profile = SpeedProfile(geometry, self.scene, np.linalg.norm(obs["vel"]), MAX_SPEED)
        except ValueError:
            return  # Keep the already checked baseline curve if refinement cannot fit.
        self.curve, self.duration = profile, float(profile.x[-1])

    def _cross(self, obs: dict) -> None:
        try:
            super()._cross(obs)
        except ValueError:
            approach, centre, _ = gate_points(obs, self.entry)
            normal = (centre - approach) / APPROACH_DISTANCE
            for distance in [0.08, 0.02]:
                goal = centre - distance * normal
                if np.linalg.norm(obs["pos"] - goal) > 0.04 and self.scene.clear_segment(
                    np.asarray(obs["pos"]), goal
                ):
                    self.phase = "inspect"
                    self._segment(obs, goal)
                    self.holding = False
                    return
            raise
        self.holding = False
        if self.phase == "cross":
            try:
                profile = SpeedProfile(
                    self.curve, self.scene, np.linalg.norm(obs["vel"]), GATE_SPEED
                )
            except ValueError:
                return
            self.curve, self.duration = profile, float(profile.x[-1])

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Add moderate velocity feedforward while tracking the timed approach."""
        if self.phase == "inspect" and self.holding and self.retry_delay <= self.dt:
            self.phase = "approach"  # Retry planning instead of holding indefinitely.
        command = super().compute_control(obs, info)
        if self.phase == "approach" and not self.holding:
            command[3:6] = (
                0.5
                * self.curve(min(self.elapsed, self.duration), nu=1)
                * getattr(self, "speed_factor", 1.0)
            )
        return command
