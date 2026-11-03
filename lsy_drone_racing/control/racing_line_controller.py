"""Optimize the multi-gate racing line, then track it with the existing MPC.

pixi run python scripts/sim.py --config=level2.toml --controller=racing_line_controller.py
Only revealed gate openings are crossed; unrevealed poses remain nominal lookahead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.continuous_spline_controller import CRUISE_SPEED, SENSING_SPEED
from lsy_drone_racing.control.predictive_spline_controller import PredictiveSplineController
from lsy_drone_racing.control.racing_spline_controller import GATE_SPEED, SpeedProfile

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.spline_controller import Scene

MAX_GATE_OFFSET = 0.05
MAX_GATE_SLOPE = 0.35
MIN_TIME_GAIN = 0.005
LINE_ACCELERATION_SCALE = 2.5


def crossing_offset(
    centre: np.ndarray, plane: np.ndarray, incoming: np.ndarray, outgoing: np.ndarray, limit: float
) -> np.ndarray:
    """Minimize distance through a bounded part of the opening, with a centre preference."""

    def objective(offset: np.ndarray) -> tuple[float, np.ndarray]:
        point = centre + plane @ offset
        first, second = point - incoming, point - outgoing
        first_length, second_length = (
            max(np.linalg.norm(first), 1e-8),
            max(np.linalg.norm(second), 1e-8),
        )
        cost = first_length + second_length + 0.5 * np.dot(offset, offset)
        gradient = plane.T @ (first / first_length + second / second_length) + offset
        return float(cost), gradient

    result = minimize(
        objective,
        np.zeros(2),
        jac=True,
        bounds=[(-limit, limit)] * 2,
        method="L-BFGS-B",
        options={"maxiter": 12, "ftol": 1e-7},
    )
    return np.clip(result.x, -limit, limit)


def fit_line(points: np.ndarray, velocities: np.ndarray, corridor_nodes: set[int]) -> BPoly:
    """Keep gate headings and blend detours into a smooth curve from the measured state."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    if np.any(lengths < 0.025):
        raise ValueError("Optimized knots are too close")
    times = np.r_[0, np.cumsum(np.maximum(lengths, 0.05))]
    velocities = velocities.copy()
    for index in range(1, len(points) - 1):
        if index not in corridor_nodes:
            direction = points[index + 1] - points[index - 1]
            velocities[index] = (
                np.linalg.norm(velocities[index]) * direction / max(np.linalg.norm(direction), 1e-8)
            )
    return BPoly.from_derivatives(
        times, [[point, velocity, np.zeros(3)] for point, velocity in zip(points, velocities)]
    )


def optimize_line(
    geometry: BPoly,
    centre_knots: np.ndarray,
    obs: dict,
    scene: Scene,
    speed_scale: float,
    acceleration_scale: float,
    max_gate_offset: float = MAX_GATE_OFFSET,
) -> tuple[BPoly, float]:
    """Greedily improve modelled traversal time across gate corridors and detours.

    The routing topology and endpoints stay fixed. Every accepted candidate must
    improve the full acceleration/braking profile and keep the whole spline clear.
    """
    points = geometry(geometry.x)
    velocities = geometry(geometry.x, nu=1)
    corridor_nodes = {0, len(points) - 1}
    for index in centre_knots:
        corridor_nodes.update(range(max(0, index - 1), min(len(points), index + 2)))

    def timed(curve: BPoly, time_bound: float) -> float:
        parameter = np.linspace(0, curve.x[-1], max(100, int(curve.x[-1] / 0.01)))
        if not scene.free(curve(parameter), start=points[0]).all():
            raise ValueError("Optimized curve lost clearance")
        profile = SpeedProfile(
            curve,
            scene,
            np.linalg.norm(obs["vel"]),
            CRUISE_SPEED * speed_scale,
            tuple(curve(curve.x[centre_knots])),
            sensor_speed=SENSING_SPEED * speed_scale,
            gate_speed=GATE_SPEED * speed_scale,
            acceleration_scale=acceleration_scale,
            max_duration=time_bound,
        )
        return float(profile.x[-1])

    try:
        original_time = best_time = timed(geometry, 30.0)
    except ValueError:
        return geometry, 0.0
    best = geometry
    entry = int(obs["n_gates_passed"])
    sequence = obs["gate_sequence"]
    for offset, index in enumerate(centre_knots):
        if index == 0:
            continue
        gate = int(sequence[entry + offset])
        centre = np.asarray(obs["gates_pos"][gate])
        rotation = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
        normal = rotation[:, 0] * int(obs["gate_sequence_direction"][entry + offset])
        plane = rotation[:, 1:]
        incoming = points[0] if offset == 0 else points[centre_knots[offset - 1]]
        next_entry = entry + offset + 1
        outgoing = (
            np.asarray(obs["gates_pos"][int(sequence[next_entry])])
            if next_entry < len(sequence)
            else centre + normal
        )
        limit = max(0.0, min(max_gate_offset, 0.2 - scene.gate_margin[gate] - 0.025))
        shift = crossing_offset(centre, plane, incoming, outgoing, limit)
        # Try shortening first, then oblique crossings, then a bounded offset.
        for entry_distance, exit_distance, offset_fraction, heading_fraction in [
            (0.24, 0.20, 0.0, 0.0),
            (0.24, 0.20, 0.0, 1.0),
            (0.24, 0.20, 1.0, 1.0),
            (0.30, 0.24, 0.5, 0.7),
        ]:
            candidate = points.copy()
            candidate_velocity = velocities.copy()
            displacement = offset_fraction * shift
            crossing = centre + plane @ displacement
            first, second = crossing - incoming, outgoing - crossing
            direction = first / max(np.linalg.norm(first), 1e-8) + second / max(
                np.linalg.norm(second), 1e-8
            )
            slope = (
                heading_fraction
                * (plane.T @ direction)
                / max(float(np.dot(direction, normal)), 0.25)
            )
            # At the inflated frame's front/back planes, stay inside the opening.
            slope_limit = np.minimum(MAX_GATE_SLOPE, (0.095 - np.abs(displacement)) / 0.115)
            slope = np.clip(slope, -slope_limit, slope_limit)
            heading = normal + plane @ slope
            unit_heading = heading / np.linalg.norm(heading)
            candidate[index] = crossing
            for node, signed_distance in [(index - 1, -entry_distance), (index + 1, exit_distance)]:
                if 0 < node < len(points) - 1:
                    candidate[node] = crossing + signed_distance * heading
                    candidate_velocity[node] = unit_heading * np.linalg.norm(velocities[node])
            candidate_velocity[index] = unit_heading * np.linalg.norm(velocities[index])
            try:
                curve = fit_line(candidate, candidate_velocity, corridor_nodes)
                duration = timed(curve, best_time - MIN_TIME_GAIN)
            except ValueError:
                continue
            points, velocities, best, best_time = candidate, candidate_velocity, curve, duration

    # Cut unnecessary detours, while preserving all gate corridors.
    for index in [i for i in range(1, len(points) - 1) if i not in corridor_nodes][:6]:
        candidate = points.copy()
        candidate[index] = 0.5 * points[index] + 0.25 * (points[index - 1] + points[index + 1])
        try:
            curve = fit_line(candidate, velocities, corridor_nodes)
            duration = timed(curve, best_time - MIN_TIME_GAIN)
        except ValueError:
            continue
        points, best, best_time = candidate, curve, duration
    return best, original_time - best_time


class RacingLineController(PredictiveSplineController):
    """Keep MPC tracking and improve the public-map line whenever new poses are revealed."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Retain the existing controller as fallback and count accepted line improvements."""
        super().__init__(obs, info, config)
        self.acceleration_scale = float(
            config.controller.get("acceleration_scale", LINE_ACCELERATION_SCALE)
        )
        self.line_optimizations = 0

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Use NumPy consistently instead of compiling tiny JAX expressions while planning."""
        public_obs = {key: np.asarray(value) for key, value in obs.items()}
        return super().compute_control(public_obs, info)

    def _geometry(self, obs: dict) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray]:
        geometry, centres, centre_knots = super()._geometry(obs)
        optimized, improvement = optimize_line(
            geometry, centre_knots, obs, self.scene, self.speed_scale, self.acceleration_scale
        )
        if improvement > 0:
            self.line_optimizations += 1
            return optimized, tuple(optimized(optimized.x[centre_knots])), centre_knots
        return geometry, centres, centre_knots
