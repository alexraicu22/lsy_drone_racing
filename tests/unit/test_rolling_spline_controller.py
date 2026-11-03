from pathlib import Path

import numpy as np

from lsy_drone_racing.control.continuous_spline_controller import safe_sensing_speed
from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
from lsy_drone_racing.control.spline_controller import Scene, make_segment
from lsy_drone_racing.utils import load_config, load_controller


def unknown_gate_scene() -> Scene:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "config/level2.toml")
    return Scene(
        {
            "gates_pos": np.array([[0, 0, 0.7]]),
            "gates_quat": np.array([[0, 0, 0, 1]]),
            "gates_visited": np.array([False]),
            "obstacles_pos": np.empty((0, 3)),
            "obstacles_visited": np.empty(0, dtype=bool),
        },
        config,
    )


def test_sensing_speed_requires_clear_braking_distance_before_unknown_frame():
    scene = unknown_gate_scene()
    start = np.array([-1, 0, 0.7])
    clear, _ = make_segment(start, np.zeros(3), np.array([-0.48, 0, 0.7]))
    blocked, _ = make_segment(start, np.zeros(3), np.array([-0.28, 0, 0.7]))
    assert safe_sensing_speed(clear, scene, 0.6) == 0.6
    assert safe_sensing_speed(blocked, scene, 0.6) == 0.0


def test_requested_terminal_speed_does_not_force_a_stop_at_sensing_waypoint():
    scene = unknown_gate_scene()
    curve, _ = make_segment(np.array([-1, 0, 0.7]), np.zeros(3), np.array([-0.48, 0, 0.7]))
    profile = SpeedProfile(curve, scene, 0.0, 2.2, terminal_speed=0.6)
    np.testing.assert_allclose(profile(profile.x[-1], nu=1), [0.6, 0, 0], atol=1e-6)
    assert scene.free(profile(np.linspace(0, profile.x[-1], 200))).all()


def test_standard_simulator_can_load_rolling_controller():
    root = Path(__file__).resolve().parents[2]
    controller = load_controller(root / "lsy_drone_racing/control/rolling_spline_controller.py")
    assert controller.__name__ == "RollingSplineController"
