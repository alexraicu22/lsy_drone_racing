from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from lsy_drone_racing.control.hybrid_controller import HybridController, clear_section
from lsy_drone_racing.control.predictive_spline_controller import HORIZON
from lsy_drone_racing.control.progress_controller import ProgressMPC
from lsy_drone_racing.control.tracking_controller import TrackingMPC
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2_racing_hybrid.toml")
    obs = {
        "pos": np.array([-1.2, 0, 0.7]),
        "vel": np.array([0.5, 0, 0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0]),
        "gate_sequence_direction": np.array([1]),
        "gates_pos": np.array([[0, 0, 0.7]]),
        "gates_quat": np.array([[0, 0, 0, 1]]),
        "gates_visited": np.array([True]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    return obs, config


def test_unknown_gate_tight_route_and_tracking_error_disable_progress():
    obs, config = course()
    controller = HybridController(obs, {}, config)
    assert clear_section(controller, obs)
    obs["gates_visited"][:] = False
    assert not clear_section(controller, obs)
    obs["gates_visited"][:] = True
    controller.tight_corridor = True
    assert not clear_section(controller, obs)
    controller.tight_corridor = False
    obs["pos"] += [0, 0.1, 0]
    assert not clear_section(controller, obs)


def test_entry_hold_and_immediate_exit_prevent_chattering():
    obs, config = course()
    controller = HybridController(obs, {}, config)
    model, route, scene = controller.mpc, controller.curve, controller.scene
    for _ in range(20):
        model.update_mode(route, 0, scene, True, 0.02)
    assert model.enabled
    model.update_mode(route, 0, scene, False, 0.02)
    assert not model.enabled
    for _ in range(4):
        model.update_mode(route, 0, scene, True, 0.02)
    assert not model.enabled
    for _ in range(20):
        model.update_mode(route, 0, scene, True, 0.02)
    assert model.enabled
    model.update_mode(SimpleNamespace(), 0, scene, True, 0.02)
    assert not model.enabled and model.warm_progress is None


def test_unknown_gate_option_requires_clear_braking_space():
    obs, config = course()
    obs["gates_visited"][:] = False
    config.controller.hybrid_unknown_braking = True
    config.controller.hybrid_lookahead_distance = 0.3
    config.controller.hybrid_lookahead_seconds = 0.25
    config.controller.hybrid_unknown_gate_distance = 0.5
    controller = HybridController(obs, {}, config)
    assert clear_section(controller, obs)
    obs["pos"] = np.array([-0.35, 0, 0.7])
    obs["vel"] = np.array([2.0, 0, 0])
    assert not clear_section(controller, obs)


def test_rejected_prediction_returns_exact_tracker_command_and_starts_cooldown():
    obs, config = course()
    controller = HybridController(obs, {}, config)
    model = controller.mpc
    route = SimpleNamespace(
        path=CubicHermiteSpline([0, 1], [[0, 0, 0.7], [1, 0, 0.7]], [[1, 0, 0], [1, 0, 0]])
    )
    for _ in range(20):
        model.update_mode(route, 0, controller.scene, True, 0.02)

    def off_route(self: ProgressMPC, state: np.ndarray, previous: np.ndarray) -> tuple:
        self.last_progress = np.linspace(0, 1, HORIZON)
        predicted = self.route.path(self.last_progress) + [0, 0.2, 0]
        return np.zeros(3), predicted

    state = np.array([obs["pos"], obs["vel"], np.zeros(3)])
    positions = np.tile(obs["pos"], (HORIZON, 1))
    velocities = np.zeros_like(positions)
    bias, previous = np.array([0, 0, 4.9]), np.zeros(3)
    expected = TrackingMPC(4.0, 1.5).solve(state, positions, velocities, bias, previous)
    with patch.object(ProgressMPC, "_optimize", off_route):
        actual = model.solve(state, positions, velocities, bias, previous)
    for first, second in zip(expected, actual):
        np.testing.assert_allclose(first, second, atol=1e-9)
    assert not model.enabled and model.cooldown > 0
    assert model.accepts == 0


def test_guarded_scene_does_not_change_planner_clearance():
    obs, config = course()
    controller = HybridController(obs, {}, config)
    original = controller.scene.gate_margin.copy()
    controller.mpc.update_mode(controller.curve, 0, controller.scene, True, 0.02)
    np.testing.assert_array_equal(controller.scene.gate_margin, original)
    assert np.all(controller.mpc.guarded_scene.gate_margin > original)


def test_short_extra_guard_retains_full_horizon_collision_rejection():
    obs, config = course()
    config.controller.hybrid_prediction_seconds = 0.12
    controller = HybridController(obs, {}, config)
    scene = controller.scene
    scene.poles = np.array([[0.7, 0, 1.5]])
    scene.pole_radius = np.array([0.20])
    scene.poles_known = np.array([True])
    route = SimpleNamespace(
        path=CubicHermiteSpline([0, 2], [[0, 0, 0.7], [2, 0, 0.7]], [[1, 0, 0], [1, 0, 0]]),
        distance=np.linspace(0, 2, 81),
    )
    for _ in range(20):
        controller.mpc.update_mode(route, 0, scene, True, 0.02)
    state = np.array([[0, 0, 0.7], [0.5, 0, 0], [0, 0, 0.0]])
    try:
        controller.mpc._optimize(state, np.zeros(3))
    except ValueError as error:
        assert "loses observed clearance" in str(error)
    else:
        raise AssertionError("The pole beyond the short extra guard was ignored")
    assert not controller.mpc.enabled


def test_hybrid_config_keeps_standard_level2_environment():
    root = Path(__file__).resolve().parents[2]
    original = load_config(root / "config/level2.toml").to_dict()
    trial = load_config(root / "config/level2_racing_hybrid.toml").to_dict()
    original.pop("controller")
    trial.pop("controller")
    assert trial == original
    assert load_controller(root / "lsy_drone_racing/control/hybrid_controller.py").__name__ == (
        "HybridController"
    )
