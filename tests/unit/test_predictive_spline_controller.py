from pathlib import Path

import numpy as np

from lsy_drone_racing.control.predictive_spline_controller import (
    ACCELERATION_LIMIT,
    HORIZON,
    PREDICTION_DT,
    AccelerationMPC,
)
from lsy_drone_racing.utils import load_controller


def test_predictive_hover_compensates_a_known_disturbance():
    model = AccelerationMPC()
    state = np.array([[0, 0, 0.7], [0, 0, 0], [0, 0, 0]], dtype=float)
    bias = np.array([0.5, -0.3, 1.0])
    command, predicted = model.solve(
        state, np.tile(state[0], (HORIZON, 1)), np.zeros((HORIZON, 3)), bias, -bias
    )
    np.testing.assert_allclose(command, -bias, atol=0.02)
    np.testing.assert_allclose(predicted, np.tile(state[0], (HORIZON, 1)), atol=0.005)


def test_predictive_controller_brakes_and_steers_before_future_corner():
    model = AccelerationMPC()
    state = np.array([[0, 0, 0.7], [1, 0, 0], [0, 0, 0]], dtype=float)
    time = PREDICTION_DT * np.arange(1, HORIZON + 1)
    straight = np.column_stack([time, np.zeros(HORIZON), np.full(HORIZON, 0.7)])
    straight_velocity = np.tile([1, 0, 0], (HORIZON, 1))
    nominal, _ = model.solve(state, straight, straight_velocity, np.zeros(3), np.zeros(3))
    turn = straight.copy()
    turn[:, 0] = np.minimum(time, 0.3)
    turn[:, 1] = np.maximum(time - 0.2, 0)
    turn_velocity = np.column_stack([time < 0.3, time > 0.2, np.zeros(HORIZON)])
    command, _ = model.solve(state, turn, turn_velocity, np.zeros(3), np.zeros(3))
    assert command[0] < nominal[0]
    assert command[1] > nominal[1]
    assert np.linalg.norm(command) <= ACCELERATION_LIMIT + 1e-6


def test_standard_simulator_can_load_predictive_controller():
    root = Path(__file__).resolve().parents[2]
    controller = load_controller(root / "lsy_drone_racing/control/predictive_spline_controller.py")
    assert controller.__name__ == "PredictiveSplineController"
