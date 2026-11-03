from pathlib import Path

import numpy as np

from lsy_drone_racing.control.recovery_controller import RecoveryController
from lsy_drone_racing.utils import load_config


def fixture() -> tuple[RecoveryController, dict]:
    planner = RecoveryController.__new__(RecoveryController)
    planner.config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    planner.replans = 0
    planner.recovery_attempts = 0
    planner.phase = "approach"
    obs = {
        "pos": np.array([-0.6, 0.0, 1.2]),
        "vel": np.zeros(3),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0]),
        "gate_sequence_direction": np.array([1]),
        "gates_pos": np.array([[0.0, 0, 1.2]]),
        "gates_quat": np.array([[0.0, 0, 0, 1]]),
        "gates_visited": np.array([False]),
        "obstacles_pos": np.array([[-0.35, 0, 1.55]]),
        "obstacles_visited": np.array([True]),
    }
    return planner, obs


def test_recovery_routes_around_pole_into_guaranteed_sensing_range():
    planner, obs = fixture()
    assert planner._recover(obs)
    assert planner.recovery_attempts == 1
    assert planner.staging and not planner.holding
    assert planner.phase == "approach"
    # Even the largest allowed horizontal displacement of the unknown gate
    # leaves this viewpoint inside the original 0.7 m sensor range.
    error = np.linalg.norm([0.15, 0.15])
    assert np.linalg.norm(planner.goal[:2] - obs["gates_pos"][0, :2]) + error < 0.7
    positions = planner.curve(np.linspace(0, planner.duration, 2000))
    assert planner.scene.free(positions, start=obs["pos"]).all()
    assert positions[:, 0].max() < -0.25


def test_blocked_recovery_keeps_existing_reference():
    planner, obs = fixture()
    limits = planner.config.env.track.safety_limits
    limits.pos_limit_low = [-0.7, -0.1, 0.0]
    limits.pos_limit_high = [-0.4, 0.2, 2.0]
    original = object()
    planner.curve = original
    assert not planner._recover(obs)
    assert planner.curve is original
    assert planner.recovery_attempts == 0
