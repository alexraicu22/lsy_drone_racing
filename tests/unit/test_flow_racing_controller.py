from pathlib import Path
from unittest.mock import patch

import numpy as np

from lsy_drone_racing.control.flow_racing_controller import FlowRacingController
from lsy_drone_racing.control.racing_line_controller import RacingLineController
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple[dict, object]:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    obs = {
        "pos": np.array([-0.3, 0.0, 0.7]),
        "vel": np.array([0.5, 0.0, 0.0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, 1]),
        "gates_pos": np.array([[0.0, 0.0, 0.7], [1.0, 0.0, 0.7]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=float),
        "gates_visited": np.array([True, False]),
        "obstacles_pos": np.empty((0, 3)),
        "obstacles_visited": np.empty(0, dtype=bool),
    }
    return obs, config


def no_continuous_route(self: RacingLineController, obs: dict) -> None:
    self.phase = "approach"


def test_moving_inspection_resumes_without_waiting_for_zero_velocity():
    obs, config = course()
    obs["gates_visited"][:] = True
    planner = FlowRacingController(obs, {}, config)
    planner.phase = "inspect"
    command = planner.compute_control(obs)
    assert planner.phase == "race"
    assert planner.moving_resumes == 1
    assert command.shape == (16,)
    assert np.isfinite(command).all()
    np.testing.assert_allclose(planner.curve(0), obs["pos"])
    assert np.linalg.norm(planner.curve(0, nu=1)) > 0.4


def test_inspection_retries_checked_crossing_if_full_route_does_not_fit():
    obs, config = course()
    planner = FlowRacingController(obs, {}, config)
    planner.phase = "inspect"
    with patch.object(RacingLineController, "_approach", no_continuous_route):
        command = planner.compute_control(obs)
    assert planner.phase == "cross"
    assert planner.moving_resumes == 1
    assert np.isfinite(command).all()
    samples = planner.curve(np.linspace(0, planner.duration, 1000))
    assert planner.scene.free(samples, start=obs["pos"]).all()


def test_rejected_proposals_preserve_active_route_and_clock():
    obs, config = course()
    planner = FlowRacingController(obs, {}, config)
    planner.phase = "inspect"
    old_curve, old_goal = planner.curve, planner.goal
    planner.elapsed = 0.1

    def rejected(self: RacingLineController, obs: dict) -> None:
        self.phase = "approach"
        self.curve = None
        self.goal = np.zeros(3)
        self.elapsed = 0.0
        self.holding = True
        raise ValueError("Blocked")

    with patch.object(RacingLineController, "_approach", rejected):
        with patch.object(RacingLineController, "_cross", rejected):
            planner._resume(obs)
    assert planner.phase == "inspect"
    assert planner.curve is old_curve
    assert planner.goal is old_goal
    assert planner.elapsed == 0.1
    assert not planner.holding
    assert planner.moving_resumes == 0


def test_unseen_gate_does_not_trigger_a_moving_crossing_attempt():
    obs, config = course()
    obs["gates_visited"][:] = False
    planner = FlowRacingController(obs, {}, config)
    planner.phase = "inspect"
    with patch.object(planner, "_resume") as resume:
        planner.compute_control(obs)
    resume.assert_not_called()


def test_confirmed_crossing_advances_before_retrying_the_old_inspection():
    obs, config = course()
    obs["n_gates_passed"] = 1
    obs["gates_visited"][:] = True
    planner = FlowRacingController(obs, {}, config)
    planner.entry = 0
    planner.phase = "inspect"
    with patch.object(planner, "_resume") as resume:
        planner.compute_control(obs)
    resume.assert_not_called()
    assert planner.entry == 1


def test_newly_revealed_pole_blocks_short_crossing():
    obs, config = course()
    obs["obstacles_pos"] = np.array([[0.08, 0.0, 1.5]])
    obs["obstacles_visited"] = np.array([True])
    planner = FlowRacingController(obs, {}, config)
    planner.phase = "inspect"
    old_curve = planner.curve
    with patch.object(RacingLineController, "_approach", no_continuous_route):
        planner._resume(obs)
    assert planner.phase == "inspect"
    assert planner.curve is old_curve
    assert planner.moving_resumes == 0


def test_standard_simulator_loads_flow_controller():
    root = Path(__file__).resolve().parents[2]
    cls = load_controller(root / "lsy_drone_racing/control/flow_racing_controller.py")
    assert cls.__name__ == "FlowRacingController"
