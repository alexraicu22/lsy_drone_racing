"""Gate-aware replanning, separate from the RL experiments.

pixi run python scripts/sim.py --config=level2.toml --controller=spline_controller.py

Uses only public observations. Gate and obstacle poses stay nominal until the
environment reveals them. No training or saved model is needed.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from ml_collections import ConfigDict

SPEED = 1.6  # metres/second
ACCELERATION = 4.0  # metres/second squared
CLEARANCE = 0.10  # Drone radius plus tracking margin.
POLE_CLEARANCE = 0.18  # Extra room for tracking error while turning around thin poles.
APPROACH_DISTANCE = 0.40
EXIT_DISTANCE = 0.20


def gate_points(obs: dict, entry: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ordered approach, centre, and exit for either crossing direction."""
    gate = int(obs["gate_sequence"][entry])
    centre = np.asarray(obs["gates_pos"][gate])
    normal = Rotation.from_quat(np.asarray(obs["gates_quat"][gate])).apply([1, 0, 0])
    normal *= int(obs["gate_sequence_direction"][entry])
    return centre - APPROACH_DISTANCE * normal, centre, centre + EXIT_DISTANCE * normal


class Scene:
    """Conservative collision checks using observed poles and oriented gate frames."""

    def __init__(self, obs: dict, config: ConfigDict):
        """Inflate unknown objects by their configured position uncertainty."""
        self.gates = np.asarray(obs["gates_pos"])
        self.rotations = Rotation.from_quat(np.asarray(obs["gates_quat"])).as_matrix()
        self.poles = np.asarray(obs["obstacles_pos"])
        self.poles_known = np.asarray(obs["obstacles_visited"], dtype=bool)
        randomizations = config.env.get("randomizations", {})
        gate_spec = randomizations.get("gate_pos", {}).get("kwargs", {})
        pole_spec = randomizations.get("obstacle_pos", {}).get("kwargs", {})
        gate_error = np.max(np.abs(gate_spec.get("maxval", [0, 0, 0])))
        pole_error = np.linalg.norm(pole_spec.get("maxval", [0, 0, 0])[:2])
        self.gate_margin = CLEARANCE + (~np.asarray(obs["gates_visited"], bool)) * gate_error
        self.pole_radius = (
            0.015 + POLE_CLEARANCE + (~np.asarray(obs["obstacles_visited"], bool)) * pole_error
        )
        limits = config.env.track.safety_limits
        self.low = np.maximum(np.asarray(limits.pos_limit_low), [-3, -3, 0.0])
        self.high = np.minimum(np.asarray(limits.pos_limit_high), [3, 3, 2.0]) - 0.1

    def free(self, points: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
        """Check drone-centre samples against inflated geometry and course bounds."""
        points = np.atleast_2d(points)
        free = np.all((points >= self.low) & (points <= self.high), axis=-1)
        if len(self.poles):
            distance = np.linalg.norm(points[:, None, :2] - self.poles[None, :, :2], axis=-1)
            below_top = points[:, None, 2] < self.poles[None, :, 2] + CLEARANCE + 0.05
            collision = (distance < self.pole_radius) & below_top
            if start is not None:
                initial = np.linalg.norm(start[:2] - self.poles[:, :2], axis=-1)
                for pole in range(len(self.poles)):
                    # Tracking error can put the drone just inside a planning
                    # margin. Allow an outward escape, never a closer approach.
                    if 0.08 < initial[pole] < self.pole_radius[pole]:
                        if np.all(distance[:, pole] >= initial[pole] - 0.002):
                            collision[:, pole] = False
            free &= ~np.any(collision, axis=-1)
        for centre, rotation, margin in zip(self.gates, self.rotations, self.gate_margin):
            local = (points - centre) @ rotation
            near_plane = np.abs(local[:, 0]) < 0.015 + margin
            inside_frame = np.all(np.abs(local[:, 1:]) < 0.36 + margin, axis=-1)
            inside_opening = np.all(np.abs(local[:, 1:]) < max(0, 0.2 - margin), axis=-1)
            free &= ~(near_plane & inside_frame & ~inside_opening)
        return free

    def clear_segment(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Sample a segment at at most 3 cm spacing."""
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / 0.03)) + 1)
        return bool(self.free(np.linspace(start, end, count), start=start).all())

    def route(self, start: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
        """A* over visible detour points; return a collision-checked polyline."""
        if not self.free(end)[0]:
            raise ValueError("Route endpoint lies inside an inflated obstacle")
        if self.clear_segment(start, end):
            return [start, end]
        levels = np.unique([max(0.3, start[2]), end[2]])
        candidates = [start, end]
        for pole, radius in zip(self.poles, self.pole_radius):
            for height in levels:
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    # Extra radius keeps chords between adjacent ring points clear.
                    offset = (radius + 0.08) * np.array([np.cos(angle), np.sin(angle)])
                    candidates.append(np.array([*(pole[:2] + offset), height]))
        for centre, rotation, margin in zip(self.gates, self.rotations, self.gate_margin):
            for x in [-margin - 0.10, margin + 0.10]:
                for y in [-0.44 - margin, 0.44 + margin]:
                    corner = centre + rotation @ [x, y, 0]
                    for height in levels:
                        candidates.append(np.array([corner[0], corner[1], height]))
            candidates.append(centre + [0, 0, 0.48 + margin])
        nodes = np.asarray(candidates)
        valid = self.free(nodes)
        valid[:2] = True  # A revealed object can make the current inflated margin overlap.
        nodes = nodes[valid]
        costs = np.full(len(nodes), np.inf)
        costs[0] = 0
        previous = {}
        queue = [(float(np.linalg.norm(end - start)), 0)]
        closed = set()
        while queue:
            _, current = heapq.heappop(queue)
            if current == 1:
                indices = [1]
                while indices[-1] != 0:
                    indices.append(previous[indices[-1]])
                return [nodes[index] for index in reversed(indices)]
            if current in closed:
                continue
            closed.add(current)
            distances = np.linalg.norm(nodes - nodes[current], axis=-1)
            for neighbor in np.argsort(distances):
                if neighbor in closed or costs[current] + distances[neighbor] >= costs[neighbor]:
                    continue
                if not self.clear_segment(nodes[current], nodes[neighbor]):
                    continue
                costs[neighbor] = costs[current] + distances[neighbor]
                previous[neighbor] = current
                estimate = costs[neighbor] + np.linalg.norm(nodes[neighbor] - end)
                heapq.heappush(queue, (float(estimate), int(neighbor)))
        raise ValueError("No clear route to the gate approach point")


def make_segment(
    position: np.ndarray, velocity: np.ndarray, goal: np.ndarray, speed_limit: float = SPEED
) -> tuple[BPoly, float]:
    """Quintic segment matching current position/velocity and stopping at the goal."""
    distance = np.linalg.norm(goal - position)
    duration = max(0.45, 1.875 * distance / speed_limit, np.sqrt(5.8 * distance / ACCELERATION))
    for _ in range(4):
        curve = BPoly.from_derivatives(
            [0, duration], [[position, velocity, np.zeros(3)], [goal, np.zeros(3), np.zeros(3)]]
        )
        samples = np.linspace(0, duration, 40)
        speed = np.linalg.norm(curve(samples, nu=1), axis=-1).max()
        acceleration = np.linalg.norm(curve(samples, nu=2), axis=-1).max()
        scale = max(1.0, speed / speed_limit, np.sqrt(acceleration / ACCELERATION))
        if scale < 1.02:
            return curve, duration
        duration *= scale * 1.02
    return curve, float(curve.x[-1])


def make_route(route: list[np.ndarray], velocity: np.ndarray, scene: Scene) -> tuple[BPoly, float]:
    """Blend detours into one C2 spline; reduce corner speeds if clearance is lost."""
    points = np.asarray(route)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    durations = np.maximum(0.5, 1.6 * lengths / SPEED)
    times = np.concatenate([[0], np.cumsum(durations)])
    for factor in [0.5, 0.25, 0.0]:
        velocities = np.zeros_like(points)
        velocities[0] = velocity
        for index in range(1, len(points) - 1):
            direction = points[index + 1] - points[index - 1]
            velocities[index] = factor * SPEED * direction / max(np.linalg.norm(direction), 1e-8)
        curve = BPoly.from_derivatives(
            times, [[point, speed, np.zeros(3)] for point, speed in zip(points, velocities)]
        )
        samples = np.linspace(0, times[-1], max(40, int(times[-1] / 0.02)))
        if scene.free(curve(samples), start=points[0]).all():
            return curve, float(times[-1])
    raise ValueError("Smooth route cannot maintain clearance")


class SplineController(Controller):
    """Fly through gates in order, replanning from public sensor updates."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Start with vertical takeoff; keep RL modules and models independent."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "state":
            raise ValueError('SplineController needs [env] control_mode = "state"')
        self.config = config
        self.dt = 1 / config.env.freq
        self.entry = int(obs["n_gates_passed"])
        self.phase = "takeoff"
        self.scene = Scene(obs, config)
        self.signature = self._signature(obs)
        self.replans = 0
        self.holding = False
        self.retry_delay = 0.0
        self.staging = False
        self.hold_position = None
        self.tracking_offset = np.zeros(3)
        goal = np.array(obs["pos"], copy=True)
        goal[2] = max(goal[2], 0.5)
        self._segment(obs, goal)

    @staticmethod
    def _signature(obs: dict) -> bytes:
        return b"".join(
            np.asarray(obs[key]).tobytes()
            for key in (
                "gates_pos",
                "gates_quat",
                "gates_visited",
                "obstacles_pos",
                "obstacles_visited",
            )
        )

    def _segment(self, obs: dict, goal: np.ndarray) -> None:
        self.curve, self.duration = make_segment(
            np.asarray(obs["pos"]), np.asarray(obs["vel"]), goal
        )
        self.elapsed = 0.0
        self.goal = np.asarray(goal)

    def _approach(self, obs: dict) -> None:
        self.phase = "approach"
        self.scene = Scene(obs, self.config)
        approach, centre, _ = gate_points(obs, self.entry)
        normal = (centre - approach) / APPROACH_DISTANCE
        for distance in [APPROACH_DISTANCE, 0.3, 0.2, 0.6]:
            candidate = centre - distance * normal
            if self.scene.free(candidate)[0]:
                approach = candidate
                break
        self.staging = False
        if not self.scene.free(approach)[0]:
            # Unknown object margins can overlap. Move to the nearest clear
            # observation point first; shrink margins only after actual sensing.
            start = np.asarray(obs["pos"])
            for fraction in np.linspace(0.95, 0.05, 19):
                candidate = start + fraction * (approach - start)
                if np.linalg.norm(candidate - start) > 0.15 and self.scene.free(candidate)[0]:
                    approach = candidate
                    self.staging = True
                    break
        route = self.scene.route(np.asarray(obs["pos"]), approach)
        self.curve, self.duration = make_route(route, np.asarray(obs["vel"]), self.scene)
        self.goal = approach
        self.elapsed = 0.0
        self.replans += 1
        self.holding = False

    def _cross(self, obs: dict) -> None:
        """Sense beyond a gate before committing if an unknown pole blocks the exit."""
        self.scene = Scene(obs, self.config)
        approach, centre, _ = gate_points(obs, self.entry)
        normal = (centre - approach) / APPROACH_DISTANCE
        for distance in [EXIT_DISTANCE, 0.15, 0.10, 0.06]:
            goal = centre + distance * normal
            if len(self.scene.poles):
                pole_distance = np.linalg.norm(goal[:2] - self.scene.poles[:, :2], axis=-1)
                if np.any(pole_distance < self.scene.pole_radius + 0.08):
                    continue
            curve, duration = make_segment(
                np.asarray(obs["pos"]), np.asarray(obs["vel"]), goal, speed_limit=0.8
            )
            if self.scene.free(
                curve(np.linspace(0, duration, 60)), start=np.asarray(obs["pos"])
            ).all():
                self.phase = "cross"
                self.curve, self.duration, self.goal, self.elapsed = curve, duration, goal, 0.0
                return
        inspect = centre - 0.16 * normal
        if np.linalg.norm(obs["pos"] - inspect) > 0.06 and self.scene.clear_segment(
            np.asarray(obs["pos"]), inspect
        ):
            self.phase = "inspect"
            self._segment(obs, inspect)
            return
        raise ValueError("No safe gate exit after sensing")

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Follow the current smooth segment; replan when object poses become known."""
        signature = self._signature(obs)
        changed = signature != self.signature
        self.signature = signature
        passed = int(obs["n_gates_passed"])
        self.retry_delay = max(0.0, self.retry_delay - self.dt)
        if self.elapsed >= self.duration and np.linalg.norm(obs["vel"]) < 0.35:
            # Learn steady bias at stopped references, not normal tracking lag.
            self.tracking_offset += 0.8 * (np.asarray(obs["pos"]) - self.goal) * self.dt
            self.tracking_offset = np.clip(self.tracking_offset, -0.12, 0.12)
        reached = (
            self.elapsed >= self.duration
            and np.linalg.norm(np.asarray(obs["pos"])[:2] - self.goal[:2]) < 0.12
            and abs(float(obs["pos"][2]) - self.goal[2]) < 0.15
            and np.linalg.norm(obs["vel"]) < 0.25
        )
        try:
            if self.phase == "takeoff":
                if reached:
                    self._approach(obs)
            elif self.phase == "approach" and passed > self.entry:
                self.entry = passed
                self._approach(obs)
            elif self.phase == "inspect" and (changed or reached):
                self._cross(obs)
            elif self.phase == "approach" and (changed or self.holding and self.retry_delay == 0):
                self._approach(obs)
            elif reached:
                if self.phase == "approach":
                    if self.staging:
                        self._approach(obs)
                        return self.compute_control(obs, info)
                    gate = int(obs["gate_sequence"][self.entry])
                    if not obs["gates_visited"][gate]:
                        # Get a little closer to reveal a displaced gate before committing.
                        _, centre, _ = gate_points(obs, self.entry)
                        self._segment(obs, 0.75 * self.goal + 0.25 * centre)
                    else:
                        local = (
                            Rotation.from_quat(np.asarray(obs["gates_quat"][gate]))
                            .inv()
                            .apply(np.asarray(obs["pos"]) - obs["gates_pos"][gate])
                        )
                        if np.max(np.abs(local[1:])) < 0.075:
                            self._cross(obs)
                elif self.phase == "cross":
                    if passed > self.entry:
                        self.entry = passed
                    self._approach(obs)
        except ValueError:
            # Stop and retry rather than execute an unchecked route.
            self.holding = True
            self.retry_delay = 0.2
            if self.hold_position is None:
                self.hold_position = np.array(obs["pos"], copy=True)
            self._segment(obs, self.hold_position)

        # Check the actual smooth curve, not just its straight waypoint links.
        times = np.linspace(min(self.elapsed, self.duration), self.duration, 30)
        if (
            self.holding
            or self.phase != "takeoff"
            and not self.scene.free(self.curve(times), start=np.asarray(obs["pos"])).all()
        ):
            self.holding = True
            if self.hold_position is None:
                self.hold_position = np.array(obs["pos"], copy=True)
            if self.phase == "cross":
                self.phase = "approach"
            position, velocity, acceleration = self.hold_position, np.zeros(3), np.zeros(3)
        else:
            self.hold_position = None
            speed_factor = 1.0 if self.phase == "takeoff" else getattr(self, "speed_factor", 1.0)
            time = min(self.elapsed + self.dt * speed_factor, self.duration)
            position = self.curve(time)
            # Match the repo's working state-controller interface: position
            # targets with zero velocity/acceleration references for damping.
            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            self.elapsed += self.dt * speed_factor
        if self.phase == "cross" and not self.holding:
            approach, centre, _ = gate_points(obs, self.entry)
            normal = (centre - approach) / APPROACH_DISTANCE
            offset = np.asarray(obs["pos"]) - centre
            lateral_error = offset - normal * np.dot(offset, normal)
            position = centre + normal * np.dot(position - centre, normal) - lateral_error
            velocity = normal * np.dot(self.curve(time, nu=1), normal) * speed_factor
        return np.concatenate(
            [position - self.tracking_offset, velocity, acceleration, [0, 0, 0, 1], np.zeros(3)]
        ).astype(np.float32)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Let the racing environment determine completion and failure."""
        return False

    def render_callback(self, sim: Sim):
        """Show the planned segment and its endpoint in the existing viewer."""
        from crazyflow.sim.visualize import draw_line, draw_points

        draw_line(sim, self.curve(np.linspace(0, self.duration, 40)), rgba=(0, 1, 0, 1))
        draw_points(sim, self.goal[None], rgba=(1, 0.3, 0, 1), size=0.03)
