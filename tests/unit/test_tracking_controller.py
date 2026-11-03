from pathlib import Path

import numpy as np

from lsy_drone_racing.control.calibrated_controller import CalibratedMPC
from lsy_drone_racing.control.predictive_spline_controller import HORIZON, PREDICTION_DT
from lsy_drone_racing.control.tracking_controller import TrackingController, TrackingMPC
from lsy_drone_racing.utils import load_config, load_controller


def test_default_weights_reproduce_selected_mpc_including_saturation():
    original = CalibratedMPC(acceleration_limit=20.0, vertical_limit=5.0)
    trial = TrackingMPC(4.0, 1.5)
    random = np.random.default_rng(7)
    state = random.normal(size=(3, 3)) * 0.2
    bias, previous = np.array([0.1, -0.2, 4.9]), np.array([0.2, 0.4, -2.5])
    for magnitude in [0.1, 4.0]:
        positions = random.normal(size=(HORIZON, 3)) * magnitude
        velocities = random.normal(size=(HORIZON, 3))
        for close_tracking in [False, True]:
            expected = original.solve(state, positions, velocities, bias, previous, close_tracking)
            actual = trial.solve(state, positions, velocities, bias, previous, close_tracking)
            for first, second in zip(expected, actual):
                np.testing.assert_allclose(first, second, atol=1e-9)


def test_changed_weights_keep_physical_acceleration_bounds():
    model = TrackingMPC(8.0, 0.6)
    model.gain[:] = 1.6
    bias = np.array([0.0, 0, 4.9])
    command, predicted = model.solve(
        np.zeros((3, 3)), np.full((HORIZON, 3), 10.0), np.full((HORIZON, 3), 4.0), bias, np.zeros(3)
    )
    physical = command * model.gain + bias
    assert np.linalg.norm(physical) <= 20.00001
    assert abs(physical[2]) <= 5.00001
    assert predicted.shape == (HORIZON, 3) and np.isfinite(predicted).all()


def test_reference_advances_on_same_route_and_stops_at_checked_endpoint():
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    config.controller.tracking_advance = 0.06
    obs = {
        "pos": np.array([-1.2, 0.0, 0.7]),
        "vel": np.array([0.5, 0, 0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0]),
        "gate_sequence_direction": np.array([1]),
        "gates_pos": np.array([[0.0, 0.0, 0.7]]),
        "gates_quat": np.array([[0, 0, 0, 1]]),
        "gates_visited": np.array([True]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    planner = TrackingController(obs, {}, config)
    assert planner.phase == "race"
    positions, velocities = planner._reference(obs)
    times = np.minimum(0.06 + PREDICTION_DT * np.arange(1, HORIZON + 1), planner.duration)
    np.testing.assert_allclose(positions, planner.curve(times))
    np.testing.assert_allclose(velocities, planner.curve(times, nu=1))
    np.testing.assert_allclose(positions[-1], planner.goal)
    np.testing.assert_allclose(velocities[-1], 0, atol=1e-7)


def test_standard_loader_finds_one_tracking_controller():
    root = Path(__file__).resolve().parents[2]
    assert load_controller(root / "lsy_drone_racing/control/tracking_controller.py").__name__ == (
        "TrackingController"
    )
