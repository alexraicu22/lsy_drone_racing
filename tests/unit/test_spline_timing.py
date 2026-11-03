from types import SimpleNamespace

import numpy as np

from scripts.evaluate_spline import RaceTiming, ReferenceTiming


def test_sector_and_phase_times_account_for_actual_crossings_and_unfinished_section():
    timing = RaceTiming(3)
    timing.step(0.2, "takeoff", False, 0.1, 0)
    timing.step(0.3, "approach", False, 0.1, 0)
    timing.step(0.4, "cross", False, 0.5, 1)
    timing.step(0.5, "race", False, 0.7, 1)
    timing.step(0.1, "approach", True, 0.0, 1)
    row = timing.row()
    assert abs(sum(timing.phases.values()) - timing.elapsed) < 1e-10
    assert abs(row["sector_1_seconds"] - 0.9) < 1e-10
    assert row["sector_2_seconds"] == ""
    assert abs(row["unfinished_sector_seconds"] - 0.6) < 1e-10
    assert abs(row["stopped_seconds"] - 0.4) < 1e-10
    assert row["phase_hold_seconds"] == 0.1


def test_reference_timing_ignores_replanning_clock_resets_and_tracks_gate_forecast():
    timing = ReferenceTiming()
    curve = SimpleNamespace(
        distance=np.array([0.0, 1.0, 2.0]),
        time=np.array([0.0, 0.5, 1.0]),
        path=lambda distance: np.array([distance, 0, 0]),
    )
    planner = SimpleNamespace(
        phase="race", curve=curve, progress=0, planned_entry=0, crossing_indices=np.array([2])
    )
    obs = {"pos": np.array([0.0, 0, 0]), "n_gates_passed": 0}
    timing.observe(planner, obs, 0.0, 0.02)
    planner.progress = 1
    obs["pos"][0] = 1.0
    timing.observe(planner, obs, 0.02, 0.02)
    assert timing.planned == 0.5
    assert timing.actual == 0.02
    planner.curve = SimpleNamespace(**curve.__dict__)
    planner.progress = 0
    timing.observe(planner, obs, 0.04, 0.02)
    assert timing.planned == 0.5
    assert timing.actual == 0.02
    timing.crossing(0, 1, 1.2)
    assert timing.row()["forecast_gate_seconds"] == 1.0
    assert timing.row()["actual_forecast_gate_seconds"] == 1.2
