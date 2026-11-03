from pathlib import Path
from unittest.mock import patch

import numpy as np

from lsy_drone_racing.control.continuous_spline_controller import continuous_route
from lsy_drone_racing.control.profile_controller import ProfileController, retime
from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
from lsy_drone_racing.control.spline_controller import Scene
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple[dict, Scene, SpeedProfile, np.ndarray]:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    obs = {
        "pos": np.array([-1.0, -0.2, 0.7]),
        "vel": np.array([0.3, 0, 0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, 1]),
        "gates_pos": np.array([[0, 0, 0.7], [1, 1, 1.2]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]]),
        "gates_visited": np.array([True, True]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    scene = Scene(obs, config)
    geometry, centres, knots = continuous_route(obs, scene, 0)
    profile = SpeedProfile(geometry, scene, 0.3, 3.3, centres, sensor_speed=1.8)
    return obs, scene, profile, knots


def test_retiming_respects_total_vertical_and_gate_speed_limits():
    obs, scene, original, knots = course()
    old_speed, old_time = original.speed.copy(), original.time.copy()
    result = retime(original, knots, obs, scene, 5.0, 12.0, 1.8, 2.0, vertical_limit=4.5)
    times = np.linspace(0, result.x[-1], 10000)
    acceleration = result(times, nu=2)
    assert np.linalg.norm(acceleration, axis=-1).max() <= 12.05
    assert np.abs(acceleration[:, 2]).max() <= 4.52
    assert scene.free(result(times)).all()
    np.testing.assert_allclose(result(0), obs["pos"])
    np.testing.assert_allclose(result(result.x[-1], nu=1), 0, atol=1e-6)
    for knot in knots:
        distance = original.knot_distance[knot]
        assert np.interp(distance, result.distance, result.speed) <= 1.80001
    np.testing.assert_array_equal(original.speed, old_speed)
    np.testing.assert_array_equal(original.time, old_time)


def test_gate_speed_limit_is_anticipated_by_braking_upstream():
    obs, scene, original, knots = course()
    result = retime(original, knots, obs, scene, 5.0, 10.0, 1.0)
    for knot in knots:
        gate = original.knot_distance[knot]
        before = result.distance < gate
        maximum = np.sqrt(1.0 + 2 * 10.0 * (gate - result.distance[before]))
        assert np.all(result.speed[before] <= maximum + 1e-6)


def test_narrow_revealed_pole_gets_a_local_speed_limit():
    obs, scene, original, knots = course()
    scene.poles = np.array([[0.0, -0.21, 1.55]])
    scene.poles_known = np.array([True])
    scene.pole_radius = np.array([0.13])
    result = retime(original, knots, obs, scene, 5.0, 12.0, 2.2)
    gate_distance = original.knot_distance[knots[0]]
    assert np.interp(gate_distance, result.distance, result.speed) <= 1.05


def test_standard_simulator_loads_one_profile_controller():
    root = Path(__file__).resolve().parents[2]
    cls = load_controller(root / "lsy_drone_racing/control/profile_controller.py")
    assert cls.__name__ == "ProfileController"


def test_selected_configuration_keeps_the_original_level2_environment():
    root = Path(__file__).resolve().parents[2]
    original = load_config(root / "config/level2.toml").to_dict()
    selected = load_config(root / "config/level2_racing.toml").to_dict()
    assert selected.pop("controller")["file"] == "profile_controller.py"
    original.pop("controller")
    assert selected == original


def test_failed_moving_resume_keeps_the_active_approach():
    obs, _, _, _ = course()
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    planner = ProfileController(obs, {}, config)
    planner.phase = "approach"
    planner.elapsed = 0.1
    old_curve = planner.curve

    def rejected(self: ProfileController, obs: dict) -> None:
        self.curve = None
        self.holding = True
        self.elapsed = 0.0
        raise ValueError("Blocked")

    with patch.object(ProfileController, "_approach", rejected):
        command = planner.compute_control(obs)
    assert planner.curve is old_curve
    assert not planner.holding
    assert planner.elapsed >= 0.1
    assert np.isfinite(command).all()
