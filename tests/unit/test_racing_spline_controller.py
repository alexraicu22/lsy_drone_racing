from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control.racing_spline_controller import (
    LONGITUDINAL_ACCELERATION,
    SpeedProfile,
    refine_route,
)
from lsy_drone_racing.control.spline_controller import Scene, make_segment
from lsy_drone_racing.utils import load_config


def scene(poles: np.ndarray | None = None) -> Scene:
    cfg = load_config(Path(__file__).resolve().parents[2] / "config/level0.toml")
    return Scene(
        {
            "gates_pos": np.array([[2, 1.4, 1.5]]),
            "gates_quat": np.array([[0, 0, 0, 1]]),
            "gates_visited": np.array([True]),
            "obstacles_pos": np.empty((0, 3)) if poles is None else poles,
            "obstacles_visited": np.ones(0 if poles is None else len(poles), dtype=bool),
        },
        cfg,
    )


def test_straight_profile_accelerates_then_brakes_and_is_faster_than_baseline():
    start, end = np.array([-1, 0, 0.7]), np.array([1, 0, 0.7])
    curve, baseline_time = make_segment(start, np.zeros(3), end)
    profile = SpeedProfile(curve, scene(), 0, 2.6)
    assert profile.x[-1] < baseline_time
    samples = np.linspace(0, profile.x[-1], 1000)
    speed = np.linalg.norm(profile(samples, nu=1), axis=-1)
    acceleration = np.linalg.norm(profile(samples, nu=2), axis=-1)
    np.testing.assert_allclose(profile(0), start, atol=1e-7)
    np.testing.assert_allclose(profile(profile.x[-1]), end, atol=1e-7)
    np.testing.assert_allclose(profile(profile.x[-1] + 1, nu=1), 0)
    assert speed.max() <= 2.6 + 1e-3
    assert speed[len(speed) // 2] > speed[10]
    assert acceleration.max() <= LONGITUDINAL_ACCELERATION + 0.01


def test_corner_speed_is_lower_than_straight_speed():
    angle = np.linspace(0, np.pi, 40)
    positions = np.column_stack([0.5 * np.sin(angle), 0.5 * np.cos(angle), np.full(40, 0.7)])
    tangent = np.column_stack([0.5 * np.cos(angle), -0.5 * np.sin(angle), np.zeros(40)])
    circle = SpeedProfile(CubicHermiteSpline(angle, positions, tangent), scene(), 0, 2.6)
    line, _ = make_segment(np.array([-0.8, 0, 0.7]), np.zeros(3), np.array([0.8, 0, 0.7]))
    straight = SpeedProfile(line, scene(), 0, 2.6)
    circle_speed = np.linalg.norm(circle(circle.x[-1] / 2, nu=1))
    straight_speed = np.linalg.norm(straight(straight.x[-1] / 2, nu=1))
    # v^2/r <= 2.5 m/s^2 gives about 1.12 m/s on a half-metre-radius bend.
    assert circle_speed < 1.2
    assert straight_speed > circle_speed + 0.5


def test_braking_distance_respects_acceleration_limit():
    curve, _ = make_segment(np.array([-1, 0, 0.7]), np.zeros(3), np.array([1, 0, 0.7]))
    profile = SpeedProfile(curve, scene(), 0, 2.6)
    energy_change = np.abs(np.diff(profile.speed**2))
    maximum_change = 2 * LONGITUDINAL_ACCELERATION * np.diff(profile.distance)
    assert np.all(energy_change <= maximum_change + 1e-8)


def test_line_refinement_keeps_obstacle_clearance_and_fixed_endpoints():
    world = scene(np.array([[0, 0, 1.55]]))
    start, end = np.array([-1, 0, 0.7]), np.array([1, 0, 0.7])
    original = world.route(start, end)
    refined = refine_route(np.array(original), world)
    np.testing.assert_allclose(refined[0], start)
    np.testing.assert_allclose(refined[-1], end)
    assert all(world.clear_segment(a, b) for a, b in zip(refined[:-1], refined[1:]))


def test_higher_speed_cap_can_be_tested_without_raising_acceleration():
    curve, _ = make_segment(np.array([-2, 0, 0.7]), np.zeros(3), np.array([2, 0, 0.7]))
    normal = SpeedProfile(curve, scene(), 0, 2.2, sensor_speed=1.2)
    faster = SpeedProfile(curve, scene(), 0, 3.0, sensor_speed=1.2)
    assert faster.x[-1] < normal.x[-1]
    time = np.linspace(0, faster.x[-1], 4000)
    assert np.linalg.norm(faster(time, nu=2), axis=-1).max() <= 1.8 + 0.01
