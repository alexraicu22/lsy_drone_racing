from pathlib import Path

import numpy as np

from lsy_drone_racing.control.spline_controller import (
    Scene,
    SplineController,
    gate_points,
    make_route,
    make_segment,
)
from lsy_drone_racing.utils import load_config


def observation() -> dict:
    return {
        "pos": np.array([-1.0, 0.0, 0.7]),
        "vel": np.zeros(3),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 0]),
        "gate_sequence_direction": np.array([1, -1]),
        "gates_pos": np.array([[0.0, 0.0, 0.7]]),
        "gates_quat": np.array([[0.0, 0.0, 0.0, 1.0]]),
        "gates_visited": np.array([True]),
        "obstacles_pos": np.array([[0.0, 1.0, 1.55]]),
        "obstacles_visited": np.array([True]),
    }


def config_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "config" / name


def test_gate_direction_changes_approach_and_exit():
    obs = observation()
    before, centre, after = gate_points(obs, 0)
    assert before[0] < centre[0] < after[0]
    before, centre, after = gate_points(obs, 1)
    assert before[0] > centre[0] > after[0]


def test_segment_preserves_position_velocity_and_acceleration_at_replan():
    start, velocity, goal = np.array([-1, 0, 0.7]), np.array([0.2, 0.1, 0]), np.array([1, 1, 1])
    curve, duration = make_segment(start, velocity, goal)
    np.testing.assert_allclose(curve(0), start)
    np.testing.assert_allclose(curve(0, nu=1), velocity)
    np.testing.assert_allclose(curve(0, nu=2), 0, atol=1e-12)
    np.testing.assert_allclose(curve(duration), goal)
    np.testing.assert_allclose(curve(duration, nu=1), 0, atol=1e-12)
    np.testing.assert_allclose(curve(duration, nu=2), 0, atol=1e-12)


def test_gate_frame_blocks_off_centre_crossing_but_opening_is_clear():
    scene = Scene(observation(), load_config(config_path("level0.toml")))
    assert scene.clear_segment(np.array([-0.5, 0, 0.7]), np.array([0.5, 0, 0.7]))
    assert not scene.clear_segment(np.array([-0.5, 0.25, 0.7]), np.array([0.5, 0.25, 0.7]))


def test_route_detours_around_pole_and_every_link_is_clear():
    obs = observation()
    obs["gates_pos"] += [0, -1, 0]
    obs["obstacles_pos"] = np.array([[0, 0, 1.55]])
    scene = Scene(obs, load_config(config_path("level0.toml")))
    start, end = np.array([-1, 0, 0.7]), np.array([1, 0, 0.7])
    assert not scene.clear_segment(start, end)
    route = scene.route(start, end)
    assert len(route) > 2
    assert all(scene.clear_segment(a, b) for a, b in zip(route[:-1], route[1:]))


def test_unrevealed_level2_objects_have_larger_clearance():
    obs = observation()
    cfg = load_config(config_path("level2.toml"))
    known = Scene(obs, cfg)
    obs["gates_visited"][:] = False
    obs["obstacles_visited"][:] = False
    unknown = Scene(obs, cfg)
    assert np.all(unknown.gate_margin > known.gate_margin)
    assert np.all(unknown.pole_radius > known.pole_radius)


def test_revelation_replans_from_current_position_and_velocity():
    obs = observation()
    obs["gates_visited"][:] = False
    cfg = load_config(config_path("level2.toml"))
    controller = SplineController(obs, {}, cfg)
    controller._approach(obs)
    before = controller.replans
    obs["pos"] = np.array([-0.8, 0.0, 0.7])
    obs["vel"] = np.array([0.1, 0, 0])
    obs["gates_pos"] += [0.1, -0.1, 0.05]
    obs["gates_visited"][:] = True
    controller.compute_control(obs)
    assert controller.replans == before + 1
    np.testing.assert_allclose(controller.curve(0), obs["pos"])
    np.testing.assert_allclose(controller.curve(0, nu=1), obs["vel"])
    np.testing.assert_allclose(controller.goal, gate_points(obs, 0)[0])


def test_margin_escape_only_allows_motion_away_from_pole():
    obs = observation()
    obs["gates_pos"] += [0, -1, 0]
    obs["obstacles_pos"] = np.array([[0, 0, 1.55]])
    scene = Scene(obs, load_config(config_path("level0.toml")))
    start = np.array([0.18, 0, 0.7])
    assert not scene.free(start)[0]
    assert scene.clear_segment(start, np.array([0.5, 0, 0.7]))
    assert not scene.clear_segment(start, np.array([-0.5, 0, 0.7]))


def test_smooth_detour_preserves_clearance():
    obs = observation()
    obs["gates_pos"] += [0, -1, 0]
    obs["obstacles_pos"] = np.array([[0, 0, 1.55]])
    scene = Scene(obs, load_config(config_path("level0.toml")))
    route = scene.route(np.array([-1, 0, 0.7]), np.array([1, 0, 0.7]))
    curve, duration = make_route(route, np.zeros(3), scene)
    assert scene.free(curve(np.linspace(0, duration, 500))).all()
