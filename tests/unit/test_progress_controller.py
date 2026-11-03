from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control.calibrated_controller import CalibratedMPC
from lsy_drone_racing.control.predictive_spline_controller import HORIZON, PREDICTION_DT
from lsy_drone_racing.control.progress_controller import ProgressMPC
from lsy_drone_racing.control.spline_controller import Scene
from lsy_drone_racing.utils import load_config, load_controller


def straight_course() -> tuple:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    obs = {
        "gates_pos": np.array([[0, 2, 0.7]]),
        "gates_quat": np.array([[0, 0, 0, 1]]),
        "gates_visited": np.array([True]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    scene = Scene(obs, config)
    route = SimpleNamespace(
        path=CubicHermiteSpline([0, 2], [[0, 0, 0.7], [2, 0, 0.7]], [[1, 0, 0], [1, 0, 0]]),
        distance=np.linspace(0, 2, 81),
        time=np.array([0.0, 10.0]),
    )
    state = np.array([[0, 0, 0.7], [0.5, 0, 0], [0, 0, 0.0]])
    return config, scene, route, state


def test_progress_is_optimized_independently_of_the_timed_schedule():
    config, scene, route, state = straight_course()
    results = []
    for duration in [1.0, 100.0]:
        route.time[-1] = duration
        model = ProgressMPC(config.controller)
        model.set_route(route, 0, scene)
        control, prediction = model._optimize(state, np.zeros(3))
        assert control[0] > 0  # Make progress faster than passive coasting.
        assert prediction[-1, 0] > 0.5
        assert scene.free(prediction).all()
        assert np.linalg.norm(control) <= 20.00001 and abs(control[2]) <= 5.00001
        increments = np.diff(np.r_[0, model.last_progress])
        assert increments.min() >= -1e-5
        assert increments.max() <= 6 * PREDICTION_DT + 1e-5
        assert model.last_progress[-1] <= route.distance[-1] + 1e-5
        results.append((control, prediction))
    for first, second in zip(*results):
        np.testing.assert_allclose(first, second, atol=1e-8)


def test_replanning_discards_old_progress_and_controls():
    config, scene, route, state = straight_course()
    model = ProgressMPC(config.controller)
    model.set_route(route, 0, scene)
    model._optimize(state, np.zeros(3))
    assert model.warm_progress is not None
    new_route = SimpleNamespace(path=route.path, distance=route.distance.copy())
    model.set_route(new_route, 0.4, scene)
    assert model.warm_progress is model.warm_controls is model.last_progress is None
    assert model.start == 0.4


def test_grouped_controls_keep_bounds_and_valid_predictions():
    config, scene, route, state = straight_course()
    config.controller.progress_control_blocks = 9
    config.controller.progress_iterations = 15
    model = ProgressMPC(config.controller)
    model.set_route(route, 0, scene)
    command, prediction = model._optimize(state, np.zeros(3))
    assert model.control_size < 3 * HORIZON
    assert np.linalg.norm(command) <= 20.00001 and abs(command[2]) <= 5.00001
    assert scene.free(prediction).all()
    assert np.diff(np.r_[0, model.last_progress]).min() >= -1e-5


def test_failed_progress_solve_uses_the_calibrated_tracker():
    config, scene, route, state = straight_course()
    model = ProgressMPC(config.controller)
    model.set_route(route, 0, scene)
    times = PREDICTION_DT * np.arange(1, HORIZON + 1)
    positions = state[0] + times[:, None] * state[1]
    velocities = np.tile(state[1], (HORIZON, 1))
    bias, previous = np.array([0, 0, 4.9]), np.zeros(3)
    original = CalibratedMPC(acceleration_limit=20.0, vertical_limit=5.0)
    expected = original.solve(state, positions, velocities, bias, previous)
    with patch.object(ProgressMPC, "_optimize", side_effect=ValueError("Blocked prediction")):
        actual = model.solve(state, positions, velocities, bias, previous)
    for first, second in zip(expected, actual):
        np.testing.assert_allclose(first, second, atol=1e-9)
    assert model.attempts == 1 and model.accepts == 0


def test_config_keeps_level2_and_loader_finds_the_controller():
    root = Path(__file__).resolve().parents[2]
    original = load_config(root / "config/level2.toml").to_dict()
    trial = load_config(root / "config/level2_racing_progress.toml").to_dict()
    original.pop("controller")
    trial.pop("controller")
    assert original == trial
    assert load_controller(root / "lsy_drone_racing/control/progress_controller.py").__name__ == (
        "ProgressController"
    )
