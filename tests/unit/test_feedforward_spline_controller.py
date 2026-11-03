from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control.feedforward_spline_controller import (
    MAX_FEEDFORWARD_ACCELERATION,
    path_acceleration,
)
from lsy_drone_racing.utils import load_controller


def test_straight_constant_speed_needs_no_feedforward_acceleration():
    distance = np.array([0.0, 2.0])
    path = CubicHermiteSpline(distance, [[0, 0, 0.7], [2, 0, 0.7]], [[1, 0, 0], [1, 0, 0]])
    profile = SimpleNamespace(distance=distance, path=path, speed=np.ones(2))
    np.testing.assert_allclose(path_acceleration(profile, 1, np.array([1, 0, 0])), 0)


def test_feedforward_turn_acceleration_points_toward_circle_centre():
    angle = np.linspace(0, np.pi, 50)
    distance = 0.5 * angle
    points = np.column_stack([0.5 * np.cos(angle), 0.5 * np.sin(angle), np.full(50, 0.7)])
    tangent = np.column_stack([-np.sin(angle), np.cos(angle), np.zeros(50)])
    profile = SimpleNamespace(
        distance=distance, path=CubicHermiteSpline(distance, points, tangent), speed=np.ones(50)
    )
    acceleration = path_acceleration(profile, np.pi / 4, np.array([-1, 0, 0]))
    np.testing.assert_allclose(acceleration, [0, -1.8, 0], atol=0.02)
    assert (
        np.linalg.norm(path_acceleration(profile, np.pi / 4, np.array([-50, 0, 0])))
        <= MAX_FEEDFORWARD_ACCELERATION + 1e-6
    )


def test_standard_simulator_can_load_feedforward_controller():
    root = Path(__file__).resolve().parents[2]
    controller = load_controller(root / "lsy_drone_racing/control/feedforward_spline_controller.py")
    assert controller.__name__ == "FeedforwardSplineController"
