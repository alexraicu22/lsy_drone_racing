from pathlib import Path
from unittest.mock import patch

import numpy as np

from lsy_drone_racing.control.profile_controller import ProfileController
from lsy_drone_racing.control.sensing_controller import SensingController
from lsy_drone_racing.control.spline_controller import Scene
from lsy_drone_racing.control.submission_controller import SubmissionController
from lsy_drone_racing.utils import load_config, load_controller


def course() -> tuple[dict, object]:
    config = load_config(Path(__file__).resolve().parents[2] / "config/level2_racing_sensing.toml")
    obs = {
        "pos": np.array([-1.2, 0.0, 0.7]),
        "vel": np.array([0.5, 0.0, 0.0]),
        "n_gates_passed": 0,
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, 1]),
        "gates_pos": np.array([[0.0, 0.0, 0.7], [1.2, 0.0, 0.7]]),
        "gates_quat": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=float),
        "gates_visited": np.array([True, False]),
        "obstacles_pos": np.array([[-1.0, 1.5, 1.5]]),
        "obstacles_visited": np.array([False]),
    }
    return obs, config


def test_moving_join_matches_state_and_preserves_ordered_gate_planes():
    obs, config = course()
    planner = SensingController(obs, {}, config)
    joined = planner._moving_geometry(obs, 0.1)
    assert joined is not None
    curve, centres, knots = joined
    np.testing.assert_allclose(curve(0), obs["pos"])
    np.testing.assert_allclose(curve(0, nu=1), obs["vel"])
    join = curve.x[1]
    for derivative in [0, 1, 2]:
        np.testing.assert_allclose(
            curve(join - 1e-9, nu=derivative), curve(join + 1e-9, nu=derivative), atol=1e-6
        )
    assert planner.scene.free(curve(np.linspace(0, curve.x[-1], 1000))).all()
    assert len(knots) == len(centres) == 1
    np.testing.assert_allclose(curve(curve.x[knots]), centres, atol=1e-8)
    assert curve(curve.x[knots[0]], nu=1)[0] > 0
    # The unobserved second gate still lies beyond the stopped horizon.
    assert curve(curve.x[-1])[0] < obs["gates_pos"][1, 0] - 0.25


def test_new_pole_blocks_prefix_and_keeps_scene_on_failure():
    obs, config = course()
    planner = SensingController(obs, {}, config)
    obs["obstacles_visited"][:] = True
    obs["obstacles_pos"][0] = [-1.0, 0, 1.5]
    planner.scene = Scene(obs, config)
    original = planner.scene.pole_radius.copy()
    assert planner._moving_geometry(obs, 0.1) is None
    np.testing.assert_array_equal(planner.scene.pole_radius, original)


def test_rejected_plan_leaves_active_route_and_scene_intact():
    obs, config = course()
    planner = SensingController(obs, {}, config)
    curve, elapsed = planner.curve, planner.elapsed
    radii = planner.scene.pole_radius.copy()

    def blocked(self: ProfileController, future_obs: dict) -> None:
        self.scene.pole_radius[:] = 0
        raise ValueError("No clear route")

    with patch.object(ProfileController, "_geometry", blocked):
        assert planner._moving_geometry(obs, 0.1) is None
    assert planner.curve is curve and planner.elapsed == elapsed
    np.testing.assert_array_equal(planner.scene.pole_radius, radii)


def test_near_gate_or_stopped_state_uses_original_planner():
    obs, config = course()
    planner = SensingController(obs, {}, config)
    obs["vel"][:] = 0
    assert planner._moving_geometry(obs, 0.1) is None


def test_config_preserves_level2_and_loads_controller():
    root = Path(__file__).resolve().parents[2]
    original = load_config(root / "config/level2.toml").to_dict()
    candidate = load_config(root / "config/level2_racing_sensing.toml").to_dict()
    candidate.pop("controller")
    original.pop("controller")
    assert candidate == original
    cls = load_controller(root / "lsy_drone_racing/control/sensing_controller.py")
    assert cls.__name__ == "SensingController"


def test_submission_uses_fast_tuning_without_modifying_supplied_config():
    root = Path(__file__).resolve().parents[2]
    obs, _ = course()
    config = load_config(root / "config/level2.toml")
    original = config.to_dict()
    planner = SubmissionController(obs, {}, config)
    expected = load_config(root / "config/level2_racing_fast.toml").controller.to_dict()
    actual = planner.config.controller.to_dict()
    expected.pop("file")
    actual.pop("file")
    assert actual == expected
    assert config.to_dict() == original
