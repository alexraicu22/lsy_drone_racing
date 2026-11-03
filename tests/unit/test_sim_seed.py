from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from lsy_drone_racing.utils import load_config
from scripts import sim


def test_simulation_logs_and_forwards_reproducible_seeds_and_full_step_time():
    cfg = load_config(Path(__file__).resolve().parents[2] / "config/level2.toml")
    obs = {"n_gates_passed": 5, "gate_sequence": np.arange(5)}
    env = SimpleNamespace(
        reset=Mock(return_value=(obs, {})),
        step=Mock(return_value=(obs, 0.0, True, False, {})),
        close=Mock(),
    )
    controller = Mock()
    controller.compute_control.return_value = np.zeros(16)
    controller.step_callback.return_value = False
    with patch.object(sim, "load_config", return_value=cfg):
        with patch.object(sim, "load_controller", return_value=Mock(return_value=controller)):
            with patch.object(sim.gymnasium, "make", return_value=env):
                with patch.object(sim, "JaxToNumpy", side_effect=lambda instance: instance):
                    with patch.object(sim.logger, "info") as log:
                        result = sim.simulate(n_runs=2, render=False, seed=123)
    assert [call.kwargs["seed"] for call in env.reset.call_args_list] == [123, 124]
    assert result == [0.02, 0.02]
    log.assert_any_call("Race seed: %d", 123)
    log.assert_any_call("Race seed: %d", 124)
    env.close.assert_called_once()
