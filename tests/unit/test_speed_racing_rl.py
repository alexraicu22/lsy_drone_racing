from pathlib import Path
from unittest.mock import patch

import numpy as np
from ml_collections import ConfigDict

from lsy_drone_racing.control.continuous_spline_controller import ContinuousSplineController
from lsy_drone_racing.control.speed_racing_controller import SpeedRacingController
from lsy_drone_racing.control.speed_racing_rl import (
    GAMMA,
    SpeedSetting,
    pace_reward,
    speed_features,
)
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple[dict, ConfigDict]:
    obs = {
        "pos": np.array([-1.0, 0.0, 0.7]),
        "vel": np.array([0.3, 0, 0]),
        "quat": np.array([0, 0, 0, 1]),
        "ang_vel": np.zeros(3),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, 1]),
        "gates_pos": np.array([[0, 0, 0.7], [1, 0, 0.7]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
        "gates_visited": np.array([True, True]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    root = Path(__file__).resolve().parents[2]
    return obs, load_config(root / "config/level2.toml")


def test_speed_action_is_bounded_and_changes_smoothly():
    setting = SpeedSetting()
    assert abs(setting.update(np.ones(1), 0.02) - 1.03) < 1e-10
    for _ in range(100):
        setting.update(np.ones(1), 0.02)
    assert abs(setting.factor - 1.75) < 1e-10
    for _ in range(100):
        setting.update(-np.ones(1), 0.02)
    assert abs(setting.factor - 0.25) < 1e-10


def test_successful_lap_scores_elapsed_time_without_navigation_bonuses():
    takeoff = 1.0
    total = 0.0
    for index in range(1, 701):
        terms = pace_reward(0.02, takeoff + index * 0.02, False, takeoff if index == 1 else 0)
        assert set(terms) == {"time", "failure"}
        total += sum(terms.values())
    assert abs(total - (-15.0)) < 1e-8


def test_crashing_early_does_not_avoid_the_failed_lap_cost():
    for decisions in [1, 100, 700, 1450]:
        takeoff = 1.0
        total = 0.0
        for index in range(1, decisions + 1):
            total += sum(
                pace_reward(
                    0.02, takeoff + index * 0.02, index == decisions, takeoff if index == 1 else 0
                ).values()
            )
        assert abs(total - (-35.0)) < 1e-8


def test_discounted_objective_prefers_fast_valid_laps_to_failure():
    def score(seconds: float, failed: bool) -> float:
        takeoff = 1.0
        decisions = round((seconds - takeoff) / 0.02)
        total = 0.0
        for index in range(1, decisions + 1):
            terms = pace_reward(
                0.02,
                takeoff + index * 0.02,
                failed and index == decisions,
                takeoff if index == 1 else 0,
            )
            total += GAMMA ** (index - 1) * sum(terms.values())
        return total

    assert score(10, False) > score(20, False)
    assert score(20, False) > max(score(seconds, True) for seconds in [2, 15, 30])


def test_speed_features_are_finite_and_have_same_shape_across_planner_phases():
    obs, config = course()
    planner = ContinuousSplineController(obs, {}, config)
    takeoff = speed_features(obs, planner, 1.0, 0.0)
    planner._approach(obs)
    racing = speed_features(obs, planner, 1.0, 2.0)
    assert takeoff.shape == racing.shape
    assert np.isfinite(racing).all()
    assert np.max(np.abs(racing)) <= 10


def test_zero_policy_action_preserves_the_classical_commands():
    class NeutralPolicy:
        speed_policy_frequency = 50

        def predict(
            self, features: np.ndarray, deterministic: bool = True
        ) -> tuple[np.ndarray, None]:
            return np.zeros(1), None

    obs, config = course()
    classical = ContinuousSplineController(obs, {}, config)
    with patch(
        "lsy_drone_racing.control.speed_racing_controller.load_policy", return_value=NeutralPolicy()
    ):
        learned = SpeedRacingController(obs, {}, config)
    classical._approach(obs)
    learned._approach(obs)
    for _ in range(5):
        np.testing.assert_allclose(
            classical.compute_control(obs), learned.compute_control(obs), atol=1e-7
        )


def test_speed_adapter_is_loadable_by_standard_simulator():
    root = Path(__file__).resolve().parents[2]
    controller = load_controller(root / "lsy_drone_racing/control/speed_racing_controller.py")
    assert controller.__name__ == "SpeedRacingController"
