from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from lsy_drone_racing.control.calibrated_controller import (
    COMMAND_GAIN,
    CalibratedController,
    CalibratedMPC,
)
from lsy_drone_racing.control.predictive_spline_controller import HORIZON
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple[dict, ConfigDict]:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    obs = {
        "pos": np.array([-1.0, -0.2, 0.025]),
        "vel": np.zeros(3),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, 1]),
        "gates_pos": np.array([[0, 0, 0.7], [1, 1, 0.9]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]]),
        "gates_visited": np.array([False, False]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    return obs, config


def test_hover_compensates_affine_command_response_without_predicted_drift():
    mpc = CalibratedMPC()
    state = np.zeros((3, 3))
    state[0] = [0.1, 0.2, 0.8]
    bias = np.array([0.2, -0.1, 4.9])
    previous = -bias / COMMAND_GAIN
    position = np.broadcast_to(state[0], (HORIZON, 3))
    command, predicted = mpc.solve(state, position, np.zeros_like(position), bias, previous)
    np.testing.assert_allclose(command, previous, atol=1e-8)
    np.testing.assert_allclose(predicted, position, atol=1e-8)


def test_faster_vertical_model_and_physical_acceleration_bounds():
    mpc = CalibratedMPC()
    assert mpc.vertical.transition[2, 2] < mpc.transition[2, 2]
    state = np.zeros((3, 3))
    target = np.full((HORIZON, 3), 2.0)
    bias = np.array([0, 0, 4.9])
    command, predicted = mpc.solve(state, target, np.zeros_like(target), bias, -bias / COMMAND_GAIN)
    effective = COMMAND_GAIN * command + bias
    assert np.linalg.norm(effective) <= 8.00001
    assert abs(effective[2]) <= 5.00001
    assert predicted.shape == (HORIZON, 3)
    assert np.isfinite(predicted).all()


def test_merged_climb_starts_at_measured_position_without_vertical_stop():
    obs, config = course()
    planner = CalibratedController(obs, {}, config)
    assert planner.phase in {"race", "approach"}
    np.testing.assert_allclose(planner.curve(0), obs["pos"])
    command = planner.compute_control(obs)
    assert command.shape == (16,)
    assert np.isfinite(command).all()


def test_pace_remains_cautious_at_an_unseen_gate():
    obs, config = course()
    obs["pos"] = np.array([-0.8, 0, 0.7])
    planner = CalibratedController(obs, {}, config)
    planner.compute_control(obs)
    assert planner.reference_pace == 1.0
    assert planner.goal[0] < 0


def test_simulator_loads_calibrated_controller():
    root = Path(__file__).resolve().parents[2]
    cls = load_controller(root / "lsy_drone_racing/control/calibrated_controller.py")
    assert cls.__name__ == "CalibratedController"


def test_online_thrust_estimate_recovers_a_public_motion_response():
    obs, config = course()
    obs["pos"][2] = 0.7
    obs["quat"] = np.array([0.0, 0, 0, 1])
    config.controller.calibrated_online_thrust = True
    config.controller.calibrated_merge_takeoff = False
    planner = CalibratedController(obs, {}, config)
    dt, response = planner.dt, 1.7
    # Include the state loop's within-step position and velocity damping.
    damping = 1.25 / 0.0393 * dt**2 / 4 + 0.5 / 0.0393 * dt / 2
    acceleration = 9.81 * (response - 1) / (1 + response * damping)
    planner.was_predictive = True
    planner.previous_input[:] = 0.0
    obs["vel"][2] += acceleration * dt
    obs["pos"][2] += acceleration * dt**2 / 2
    planner.compute_control(obs)
    expected = 1.5 + dt / (0.15 + dt) * (response - 1.5)
    np.testing.assert_allclose(planner.force_gain, expected, atol=1e-8)
