from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from lsy_drone_racing.control.candidate3_controller import PARAMETERS, Candidate3Controller
from lsy_drone_racing.control.tracking_controller import TrackingController
from lsy_drone_racing.control.tuned_racing_controller import TunedRacingController
from lsy_drone_racing.utils import load_config
from scripts import tune_racing


def course() -> tuple:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2_racing_tracking.toml")
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


def test_adapter_baseline_reproduces_tracker_commands():
    obs, config = course()
    tracker = TrackingController(obs, {}, config)
    adapter = TunedRacingController(obs, {}, config)
    for _ in range(4):
        expected = tracker.compute_control(obs)
        actual = adapter.compute_control(obs)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        obs["pos"] += obs["vel"] * 0.02


def test_trial_parameters_reach_planner_and_solver_without_changing_input_config():
    obs, config = course()
    config.controller.profile_acceleration = 13.0
    config.controller.profile_mpc_acceleration = 17.0
    config.controller.profile_gate_speed = 2.1
    config.controller.tracking_advance = 0.01
    config.controller.tracking_velocity_weight = 6.0
    original = config.to_dict()
    controller = TunedRacingController(obs, {}, config)
    assert controller.config.controller.profile_acceleration == 13.0
    assert controller.config.controller.profile_gate_speed == 2.1
    assert controller.config.controller.tracking_advance == 0.01
    assert controller.mpc.acceleration_limit == 17.0
    assert controller.mpc.velocity_weight == 6.0
    assert config.to_dict() == original


def test_selected_entry_matches_the_actual_search_candidate():
    obs, config = course()
    original = config.to_dict()
    selected = Candidate3Controller(obs, {}, config)
    config.controller.update(PARAMETERS)
    expected = TunedRacingController(obs, {}, config)
    np.testing.assert_allclose(
        selected.compute_control(obs), expected.compute_control(obs), atol=1e-7
    )
    assert selected.mpc.acceleration_limit == PARAMETERS["profile_mpc_acceleration"]
    assert selected.config.env.to_dict() == original["env"]


def test_search_counts_only_success_times_and_resumes_finished_races():
    calls = []

    def simulated(parameters: dict, seed: int, candidate: int, stage: str) -> dict:
        calls.append(seed)
        return {"seed": seed, "success": seed != 0, "seconds": 0.1 if seed == 0 else 8.0}

    with TemporaryDirectory() as directory:
        output = Path(directory)
        (output / "races").mkdir()
        with (
            patch.object(tune_racing, "OUTPUT", output),
            patch.object(tune_racing, "race", simulated),
        ):
            first = tune_racing.evaluate_job(0, {}, [0, 1, 2, 3], 0.55, "test")
            second = tune_racing.evaluate_job(0, {}, [0, 1, 2, 3], 0.55, "test")
    assert first == second
    assert first["qualified"] and first["completed"] == 3
    assert first["successful_mean"] == 8.0
    assert calls == [0, 1, 2, 3]
