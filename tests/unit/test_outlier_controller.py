from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.interpolate import BPoly

from lsy_drone_racing.control.outlier_controller import OutlierController
from lsy_drone_racing.control.profile_controller import ProfileController
from lsy_drone_racing.control.racing_spline_controller import SpeedProfile
from lsy_drone_racing.control.spline_controller import Scene
from lsy_drone_racing.utils import load_config, load_controller


def route() -> tuple[Scene, SpeedProfile]:
    cfg = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    scene = Scene(
        {
            "gates_pos": np.array([[0.0, 0, 0.7]]),
            "gates_quat": np.array([[0.0, 0, 0, 1]]),
            "gates_visited": np.array([True]),
            "obstacles_pos": np.empty((0, 3)),
            "obstacles_visited": np.empty(0, dtype=bool),
        },
        cfg,
    )
    positions = np.array([[-1.0, 0, 0.7], [0, 0, 0.7], [0.5, 0, 0.7]])
    velocities = np.array([[0.0, 0, 0], [0.5, 0, 0], [0, 0, 0]])
    geometry = BPoly.from_derivatives(
        [0, 1, 1.5], [[p, v, np.zeros(3)] for p, v in zip(positions, velocities)]
    )
    return scene, SpeedProfile(geometry, scene, 0.0, 0.5)


def test_only_tight_route_is_faster_and_keeps_its_measured_start_speed():
    scene, original = route()
    planner = OutlierController.__new__(OutlierController)
    planner.scene = scene
    planner.planning_obs = {"vel": np.array([9.0, 0, 0])}
    with patch.object(ProfileController, "_follow") as install:
        planner._follow(original, np.array([1]), tight=True)
    faster = install.call_args.args[0]
    assert faster.x[-1] < original.x[-1]
    assert faster.path is original.path
    assert faster.speed.max() <= 0.85001
    assert faster.speed.max() > 0.75
    np.testing.assert_allclose(faster(0, nu=1), 0, atol=1e-8)
    times = np.linspace(0, faster.x[-1], 2000)
    assert np.linalg.norm(faster(times, nu=2), axis=-1).max() <= 4.01
    assert scene.free(faster(times)).all()
    assert original.speed.max() <= 0.50001
    with patch.object(ProfileController, "_follow") as install:
        planner._follow(original, np.array([1]), tight=False)
    assert install.call_args.args[0] is original


def test_experimental_config_changes_only_controller_selection():
    root = Path(__file__).resolve().parents[2]
    baseline = load_config(root / "config/level2_racing.toml").to_dict()
    trial = load_config(root / "config/level2_racing_outliers.toml").to_dict()
    baseline["controller"]["file"] = "outlier_controller.py"
    assert trial == baseline
    cls = load_controller(root / "lsy_drone_racing/control/outlier_controller.py")
    assert cls.__name__ == "OutlierController"
