from pathlib import Path

import numpy as np

from lsy_drone_racing.control.continuous_spline_controller import (
    ContinuousSplineController,
    continuous_route,
)
from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
from lsy_drone_racing.control.spline_controller import Scene
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple[dict, object]:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    obs = {
        "pos": np.array([-1.0, 0.0, 0.7]),
        "vel": np.array([0.3, 0.0, 0.0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, 1]),
        "gates_pos": np.array([[0.0, 0.0, 0.7], [1.0, 0.0, 0.7]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=float),
        "gates_visited": np.array([True, True]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    return obs, config


def test_known_gates_are_crossed_in_order_without_stopping():
    obs, config = course()
    scene = Scene(obs, config)
    geometry, centres, _ = continuous_route(obs, scene, 0)
    profile = SpeedProfile(geometry, scene, 0.3, 2.6, centres)
    time = np.linspace(0, profile.x[-1], 3000)
    positions = profile(time)
    assert len(centres) == 2
    crossing_indices = []
    for centre in centres:
        index = np.argmin(np.linalg.norm(positions - centre, axis=-1))
        crossing_indices.append(index)
        assert np.linalg.norm(positions[index] - centre) < 0.005
        assert profile(time[index], nu=1)[0] > 0.5
    assert crossing_indices[0] < crossing_indices[1]
    assert scene.free(positions).all()


def test_unknown_gate_stops_horizon_before_its_plane():
    obs, config = course()
    obs["gates_visited"][1] = False
    geometry, centres, _ = continuous_route(obs, Scene(obs, config), 0)
    assert len(centres) == 1
    positions = geometry(np.linspace(0, geometry.x[-1], 1000))
    assert positions[:, 0].max() < obs["gates_pos"][1, 0] - 0.3


def test_revealed_pose_replans_from_current_state():
    obs, config = course()
    planner = ContinuousSplineController(obs, {}, config)
    planner._approach(obs)
    previous_replans = planner.replans
    obs["pos"] = np.array([-0.6, 0, 0.7])
    obs["gates_pos"][0] += [0.05, 0.04, 0.02]
    command = planner.compute_control(obs)
    assert planner.replans == previous_replans + 1
    np.testing.assert_allclose(planner.curve(0), obs["pos"])
    assert command.shape == (16,)
    assert np.isfinite(command).all()


def test_simulator_loads_exactly_one_controller():
    path = (
        Path(__file__).resolve().parents[2]
        / "lsy_drone_racing/control/continuous_spline_controller.py"
    )
    controller = load_controller(path)
    assert controller.__name__ == "ContinuousSplineController"


def test_progress_waits_for_environment_to_confirm_the_required_gate():
    obs, config = course()
    planner = ContinuousSplineController(obs, {}, config)
    planner._approach(obs)
    obs["pos"] = np.array([1.15, 0.0, 0.7])
    for _ in range(12):
        command = planner.compute_control(obs)
    assert planner.entry == 0
    assert command[0] < 0.25


def test_gate_revisited_later_in_lap_has_a_later_crossing_index():
    obs, config = course()
    obs["gate_sequence"] = np.array([0, 1, 0])
    obs["gate_sequence_direction"] = np.array([1, 1, -1])
    planner = ContinuousSplineController(obs, {}, config)
    planner._approach(obs)
    assert planner.phase == "race"
    assert len(planner.crossing_indices) == 3
    assert np.all(np.diff(planner.crossing_indices) > 0)


def test_speed_profile_bounds_acceleration_between_samples_on_a_turn():
    obs, config = course()
    obs["gates_pos"][1] = [1, 1, 0.9]
    obs["gates_quat"][1] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    scene = Scene(obs, config)
    geometry, centres, _ = continuous_route(obs, scene, 0)
    profile = SpeedProfile(geometry, scene, 0.3, 2.2, centres, sensor_speed=0.8)
    time = np.linspace(0, profile.x[-1], 10000)
    assert np.linalg.norm(profile(time, nu=2), axis=-1).max() < 3.5
    assert np.isfinite(profile(time)).all()


def test_crossing_advances_to_next_gate_while_still_moving():
    obs, config = course()
    obs["pos"] = np.array([-0.3, 0, 0.7])
    planner = ContinuousSplineController(obs, {}, config)
    planner._cross(obs)
    assert planner.phase == "cross"
    obs["pos"] = np.array([0.05, 0, 0.7])
    obs["vel"] = np.array([0.4, 0, 0])
    obs["n_gates_passed"] = 1
    planner.compute_control(obs)
    assert planner.entry == 1
    assert not planner.holding


def test_aligned_approach_enters_gate_without_waiting_for_hover():
    obs, config = course()
    obs["pos"] = np.array([-0.35, 0, 0.7])
    obs["vel"] = np.array([0.5, 0, 0])
    planner = ContinuousSplineController(obs, {}, config)
    planner._segment(obs, np.array([-0.4, 0, 0.7]))
    planner.phase = "approach"
    command = planner.compute_control(obs)
    assert planner.phase == "cross"
    assert command[3] > 0


def test_known_pole_behind_gate_does_not_trap_inspection():
    obs, config = course()
    obs["pos"] = np.array([-0.16, 0, 0.7])
    obs["vel"] = np.zeros(3)
    obs["obstacles_pos"] = np.array([[0.28, 0, 1.55]])
    obs["obstacles_visited"] = np.array([True])
    planner = ContinuousSplineController(obs, {}, config)
    planner._cross(obs)
    assert planner.phase == "race"
    assert planner.tight_corridor
    samples = planner.curve(np.linspace(0, planner.duration, 300))
    assert planner.scene.free(samples).all()


def test_tight_route_keeps_uncertainty_margin_for_unrevealed_poles():
    obs, config = course()
    obs["pos"] = np.array([-0.16, 0, 0.7])
    obs["vel"] = np.zeros(3)
    obs["obstacles_pos"] = np.array([[0.28, 0, 1.55]])
    obs["obstacles_visited"] = np.array([False])
    planner = ContinuousSplineController(obs, {}, config)
    assert not planner._tight_route(obs)
    assert planner.scene.pole_radius[0] > 0.3


def test_tight_crossing_keeps_its_escape_turn_until_clear_of_pole():
    obs, config = course()
    obs["pos"] = np.array([-0.16, 0, 0.7])
    obs["vel"] = np.zeros(3)
    obs["obstacles_pos"] = np.array([[0.28, 0, 1.55]])
    obs["obstacles_visited"] = np.array([True])
    planner = ContinuousSplineController(obs, {}, config)
    planner._cross(obs)
    checked_curve = planner.curve
    obs["pos"] = np.array([0.05, 0, 0.7])
    obs["vel"] = np.array([0.3, 0, 0])
    obs["n_gates_passed"] = 1
    planner.compute_control(obs)
    assert planner.curve is checked_curve
    assert not planner.holding
