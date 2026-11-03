from pathlib import Path

import numpy as np
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.continuous_spline_controller import continuous_route
from lsy_drone_racing.control.racing_line_controller import MAX_GATE_OFFSET, optimize_line
from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
from lsy_drone_racing.control.spline_controller import Scene
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple[dict, ConfigDict]:
    cfg = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    obs = {
        "pos": np.array([-1.0, -0.2, 0.7]),
        "vel": np.array([0.3, 0, 0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, 1]),
        "gates_pos": np.array([[0, 0, 0.7], [1, 1, 0.9]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]]),
        "gates_visited": np.array([True, True]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    return obs, cfg


def test_optimized_turn_is_faster_and_preserves_start_state_and_gate_crossings():
    obs, cfg = course()
    scene = Scene(obs, cfg)
    geometry, _, knots = continuous_route(obs, scene, 0)
    optimized, gain = optimize_line(geometry, knots, obs, scene, 1.5, 2.25)
    assert gain > 0.005
    np.testing.assert_allclose(optimized(0), obs["pos"])
    np.testing.assert_allclose(optimized(0, nu=1), geometry(0, nu=1))
    np.testing.assert_allclose(optimized(optimized.x[-1]), geometry(geometry.x[-1]))
    for gate, knot in enumerate(knots):
        rotation = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
        local = (optimized(optimized.x[knot]) - obs["gates_pos"][gate]) @ rotation
        assert abs(local[0]) < 1e-6
        assert np.max(np.abs(local[1:])) <= MAX_GATE_OFFSET + 1e-6
        assert np.dot(optimized(optimized.x[knot], nu=1), rotation[:, 0]) > 0
    assert scene.free(optimized(np.linspace(0, optimized.x[-1], 3000))).all()


def test_optimization_keeps_unknown_gate_horizon_and_obstacle_clearance():
    obs, cfg = course()
    obs["gates_visited"][1] = False
    obs["obstacles_pos"] = np.array([[0.5, 0.25, 1.55]])
    obs["obstacles_visited"] = np.array([True])
    scene = Scene(obs, cfg)
    geometry, _, knots = continuous_route(obs, scene, 0)
    optimized, _ = optimize_line(geometry, knots, obs, scene, 1.5, 2.25)
    assert len(knots) == 1
    np.testing.assert_allclose(optimized(optimized.x[-1]), geometry(geometry.x[-1]))
    assert scene.free(optimized(np.linspace(0, optimized.x[-1], 3000))).all()


def test_timing_bound_rejects_slow_candidates_before_expensive_validation():
    obs, cfg = course()
    scene = Scene(obs, cfg)
    geometry, _, _ = continuous_route(obs, scene, 0)
    try:
        SpeedProfile(geometry, scene, 0.0, 0.01, max_duration=1.0)
    except ValueError as error:
        assert "time bound" in str(error)
    else:
        raise AssertionError("Slow candidate exceeded its optimization time bound")


def test_standard_simulator_loads_racing_line_controller():
    root = Path(__file__).resolve().parents[2]
    cls = load_controller(root / "lsy_drone_racing/control/racing_line_controller.py")
    assert cls.__name__ == "RacingLineController"


def test_optimized_reverse_crossings_keep_the_required_direction():
    obs, cfg = course()
    obs["pos"][:2] *= -1
    obs["vel"][:2] *= -1
    obs["gates_pos"][:, :2] *= -1
    obs["gate_sequence_direction"][:] = -1
    scene = Scene(obs, cfg)
    geometry, _, knots = continuous_route(obs, scene, 0)
    optimized, _ = optimize_line(geometry, knots, obs, scene, 1.5, 2.5)
    for gate, knot in enumerate(knots):
        normal = Rotation.from_quat(obs["gates_quat"][gate]).apply([-1, 0, 0])
        assert np.dot(optimized(optimized.x[knot], nu=1), normal) > 0
    assert scene.free(optimized(np.linspace(0, optimized.x[-1], 3000))).all()
