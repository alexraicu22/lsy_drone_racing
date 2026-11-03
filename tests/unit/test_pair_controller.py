from pathlib import Path

import numpy as np

from lsy_drone_racing.control.continuous_spline_controller import continuous_route
from lsy_drone_racing.control.pair_controller import change_crossing, flight_time, optimize_pair
from lsy_drone_racing.control.spline_controller import Scene
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2_racing_fast.toml")
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
    curve, _, knots = continuous_route(obs, scene, 0)
    return obs, config, scene, curve, knots


def test_joint_search_improves_actual_timing_and_preserves_state_and_gate_order():
    obs, config, scene, curve, knots = course()
    original_points = curve(curve.x).copy()
    best, choices, gain = optimize_pair(curve, knots, obs, scene, config)
    assert len(choices) == 2 and gain > 0.02
    np.testing.assert_allclose(
        flight_time(curve, knots, obs, scene, config)
        - flight_time(best, knots, obs, scene, config),
        gain,
    )
    for derivative in [0, 1]:
        np.testing.assert_allclose(best(0, nu=derivative), curve(0, nu=derivative))
    np.testing.assert_allclose(best(best.x[-1]), curve(curve.x[-1]))
    assert scene.free(best(np.linspace(0, best.x[-1], 2000))).all()
    for entry, knot in enumerate(knots):
        gate = obs["gate_sequence"][entry]
        crossing = best(best.x[knot])
        local = scene.rotations[gate].T @ (crossing - scene.gates[gate])
        assert abs(local[0]) < 1e-8
        assert np.max(np.abs(local[1:])) < 0.075
        assert best(best.x[knot], nu=1) @ scene.rotations[gate][:, 0] > 0
    np.testing.assert_array_equal(curve(curve.x), original_points)


def test_unknown_gate_is_only_a_stopped_sensing_horizon():
    obs, config, _, _, _ = course()
    obs["gates_visited"][1] = False
    scene = Scene(obs, config)
    curve, _, knots = continuous_route(obs, scene, 0)
    assert len(knots) == 1
    best, _, _ = optimize_pair(curve, knots, obs, scene, config)
    np.testing.assert_allclose(best(best.x[-1]), curve(curve.x[-1]))
    assert scene.free(best(np.linspace(0, best.x[-1], 1000))).all()


def test_revealed_obstacle_rejects_an_old_crossing_choice():
    obs, config, _, curve, knots = course()
    obs["obstacles_pos"] = np.array([[0, 0, 1.5]])
    obs["obstacles_visited"] = np.array([True])
    scene = Scene(obs, config)
    changed = change_crossing(curve, knots, 0, (1.0, 0.8, 1.0), obs, scene)
    try:
        flight_time(changed, knots, obs, scene, config)
    except ValueError:
        pass
    else:
        raise AssertionError("A crossing through the newly revealed pole was accepted")


def test_pair_config_keeps_level2_environment_and_submission_unchanged():
    root = Path(__file__).resolve().parents[2]
    standard = load_config(root / "config/level2.toml").to_dict()
    trial = load_config(root / "config/level2_racing_pair.toml").to_dict()
    assert standard.pop("controller")["file"] == "submission_controller.py"
    assert trial.pop("controller")["file"] == "pair_controller.py"
    assert trial == standard
    cls = load_controller(root / "lsy_drone_racing/control/pair_controller.py")
    assert cls.__name__ == "PairController"
