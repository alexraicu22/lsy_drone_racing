"""Choose two gate crossings together using the installed acceleration/braking model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.profile_controller import retime
from lsy_drone_racing.control.racing_line_controller import crossing_offset, fit_line
from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
from lsy_drone_racing.control.submission_controller import SubmissionController

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from scipy.interpolate import BPoly

    from lsy_drone_racing.control.spline_controller import Scene


def flight_time(
    curve: BPoly, knots: np.ndarray, obs: dict, scene: Scene, config: ConfigDict
) -> float:
    """Score the full route with the same speed, curvature and braking limits used in flight."""
    samples = np.linspace(0, curve.x[-1], max(100, int(curve.x[-1] / 0.008)))
    if not scene.free(curve(samples), start=obs["pos"]).all():
        raise ValueError("Pair candidate loses clearance")
    settings = config.controller
    profile = SpeedProfile(
        curve,
        scene,
        np.linalg.norm(obs["vel"]),
        3.3,
        tuple(curve(curve.x[knots])),
        sensor_speed=1.8,
        gate_speed=1.8,
        acceleration_scale=2.5,
    )
    timed = retime(
        profile,
        knots,
        obs,
        scene,
        settings.profile_max_speed,
        settings.profile_acceleration,
        settings.profile_gate_speed,
        settings.profile_sensor_speed,
        settings.get("profile_vertical_acceleration"),
    )
    return float(timed.x[-1])


def change_crossing(
    curve: BPoly,
    knots: np.ndarray,
    offset: int,
    choice: tuple[float, float, float],
    obs: dict,
    scene: Scene,
) -> BPoly:
    """Adjust one crossing and its corridor, retaining start, end and other gate planes."""
    index = int(knots[offset])
    if index == 0 or index == len(curve.x) - 1:
        return curve
    entry = int(obs["n_gates_passed"]) + offset
    gate = int(obs["gate_sequence"][entry])
    centre = scene.gates[gate]
    normal = scene.rotations[gate][:, 0] * int(obs["gate_sequence_direction"][entry])
    plane = scene.rotations[gate][:, 1:]
    points, velocities = curve(curve.x), curve(curve.x, nu=1)
    incoming = points[0] if offset == 0 else points[knots[offset - 1]]
    outgoing = (
        obs["gates_pos"][int(obs["gate_sequence"][entry + 1])]
        if entry + 1 < len(obs["gate_sequence"])
        else centre + normal
    )
    offset_fraction, heading_fraction, tangent_scale = choice
    limit = min(0.065, max(0, 0.2 - scene.gate_margin[gate] - 0.025))
    shift = offset_fraction * crossing_offset(centre, plane, incoming, outgoing, limit)
    crossing = centre + plane @ shift
    direction = (crossing - incoming) / max(np.linalg.norm(crossing - incoming), 1e-8)
    direction += (outgoing - crossing) / max(np.linalg.norm(outgoing - crossing), 1e-8)
    slope = heading_fraction * (plane.T @ direction) / max(float(direction @ normal), 0.25)
    slope_limit = np.minimum(
        0.65,
        np.maximum(0, (0.2 - scene.gate_margin[gate] - 0.02 - np.abs(shift)))
        / (0.015 + scene.gate_margin[gate]),
    )
    heading = normal + plane @ np.clip(slope, -slope_limit, slope_limit)
    unit = heading / np.linalg.norm(heading)
    points[index] = crossing
    fixed = {0, len(points) - 1}
    for knot in knots:
        fixed.update(range(max(0, knot - 1), min(len(points), knot + 2)))
    for node in [index - 1, index + 1]:
        if node in {0, len(points) - 1} or node in knots:
            continue
        stand_off = float((points[node] - centre) @ normal)
        points[node] = crossing + stand_off * heading
        velocities[node] = unit * max(0.15, np.linalg.norm(velocities[node])) * tangent_scale
    velocities[index] = unit * max(0.15, np.linalg.norm(velocities[index])) * tangent_scale
    return fit_line(points, velocities, fixed)


def optimize_pair(
    original: BPoly, knots: np.ndarray, obs: dict, scene: Scene, config: ConfigDict
) -> tuple[BPoly, tuple, float]:
    """Keep two alternatives at the first gate so a joint improvement can beat a greedy choice."""
    if not len(knots):
        return original, (), 0.0
    initial = flight_time(original, knots, obs, scene, config)
    beam = [(initial, original, ())]
    # The first choice keeps the existing crossing. Other choices can trade a
    # slower first gate for a faster turn into the second gate.
    choices = [None, (0.0, 0.8, 1.0), (1.0, 0.8, 1.0), (-1.0, 0.8, 1.0), (0.0, 0.5, 1.5)]
    for offset in range(min(2, len(knots))):
        candidates = []
        for duration, curve, selected in beam:
            for choice in choices:
                try:
                    changed = (
                        curve
                        if choice is None
                        else change_crossing(curve, knots, offset, choice, obs, scene)
                    )
                    score = (
                        duration
                        if choice is None
                        else flight_time(changed, knots, obs, scene, config)
                    )
                except ValueError:
                    continue
                candidates.append((score, changed, (*selected, choice)))
        beam = sorted(candidates, key=lambda item: item[0])[:2]
    duration, curve, selected = min(beam, key=lambda item: item[0])
    if initial - duration < 0.02:
        return original, (), 0.0
    return curve, selected, initial - duration


class PairController(SubmissionController):
    """Keep the selected tracker and try a bounded search over the next two crossings."""

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Cache crossing choices until the ordered gates or their observed poses change."""
        self.pair_cache = {}
        self.pair_searches = 0
        self.pair_accepts = 0
        super().__init__(obs, info, config)

    def _geometry(self, obs: dict) -> tuple[BPoly, tuple[np.ndarray, ...], np.ndarray]:
        curve, centres, knots = super()._geometry(obs)
        if len(knots) < int(self.config.controller.get("pair_min_gates", 1)):
            return curve, centres, knots
        entries = np.asarray(obs["gate_sequence"])[self.entry : self.entry + min(2, len(knots))]
        key = (
            self.entry,
            entries.tobytes(),
            np.asarray(obs["gates_pos"])[entries].tobytes(),
            np.asarray(obs["gates_quat"])[entries].tobytes(),
        )
        try:
            if key not in self.pair_cache:
                self.pair_searches += 1
                best, choices, gain = optimize_pair(curve, knots, obs, self.scene, self.config)
                self.pair_cache[key] = choices
            else:
                best = curve
                for offset, choice in enumerate(self.pair_cache[key]):
                    if choice is not None:
                        best = change_crossing(best, knots, offset, choice, obs, self.scene)
                gain = (
                    flight_time(curve, knots, obs, self.scene, self.config)
                    - flight_time(best, knots, obs, self.scene, self.config)
                    if self.pair_cache[key]
                    else 0.0
                )
        except ValueError:
            return curve, centres, knots
        if gain >= 0.02:
            self.pair_accepts += 1
            return best, tuple(best(best.x[knots])), knots
        return curve, centres, knots
