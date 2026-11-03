"""Evaluate spline planning on reproducible seeds with the course's 30-second limit.

pixi run python scripts/evaluate_spline.py --config level2.toml --runs 20
Add --render to watch, or --trace to print planner state changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.control.continuous_spline_controller import ContinuousSplineController


class RaceTiming:
    """Account for every simulation step and each actual ordered gate crossing."""

    def __init__(self, gates: int):
        """Start a race clock with one split per required gate."""
        self.elapsed = 0.0
        self.last_crossing = 0.0
        self.splits = []
        self.stopped = 0.0
        self.phases = dict.fromkeys(
            ["takeoff", "race", "approach", "inspect", "cross", "hold"], 0.0
        )
        self.gates = gates

    def step(self, dt: float, phase: str, holding: bool, speed: float, passed: int) -> None:
        """Charge one environment step to its phase and close confirmed sectors."""
        self.elapsed += dt
        self.phases["hold" if holding else phase] += dt
        if phase != "takeoff" and speed < 0.2:
            self.stopped += dt
        if passed > len(self.splits):
            self.splits.append(self.elapsed - self.last_crossing)
            self.last_crossing = self.elapsed

    def row(self) -> dict:
        """Return CSV fields; leave uncompleted gate sectors empty."""
        return {
            "stopped_seconds": self.stopped,
            "unfinished_sector_seconds": self.elapsed - self.last_crossing,
            **{f"phase_{phase}_seconds": value for phase, value in self.phases.items()},
            **{
                f"sector_{index + 1}_seconds": self.splits[index]
                if index < len(self.splits)
                else ""
                for index in range(self.gates)
            },
        }


class ReferenceTiming:
    """Compare progress through timed continuous routes with simulation time spent tracking."""

    def __init__(self):
        """Ignore discontinuous reference-clock resets when a new route is installed."""
        self.curve = None
        self.reference_time = 0.0
        self.planned = 0.0
        self.actual = 0.0
        self.error_squared = 0.0
        self.samples = 0
        self.forecasts = {}
        self.completed = []

    def observe(
        self, planner: ContinuousSplineController, obs: dict, now: float, dt: float
    ) -> None:
        """Count reference-clock advance only while following the same timed route."""
        if planner.phase != "race" or not hasattr(planner.curve, "distance"):
            self.curve = None
            return
        curve = planner.curve
        distance = curve.distance[planner.progress]
        reference_time = float(np.interp(distance, curve.distance, curve.time))
        if curve is self.curve:
            self.planned += max(0.0, reference_time - self.reference_time)
            self.actual += dt
            error = np.linalg.norm(np.asarray(obs["pos"]) - curve.path(distance))
            self.error_squared += error**2
            self.samples += 1
        self.curve, self.reference_time = curve, reference_time
        passed = int(obs["n_gates_passed"])
        index = passed - planner.planned_entry
        if passed not in self.forecasts and 0 <= index < len(planner.crossing_indices):
            gate_distance = curve.distance[planner.crossing_indices[index]]
            gate_time = float(np.interp(gate_distance, curve.distance, curve.time))
            self.forecasts[passed] = now, max(0.0, gate_time - reference_time)

    def crossing(self, previous: int, passed: int, now: float) -> None:
        """Compare the first available next-gate forecast with the actual remaining time."""
        if passed > previous and previous in self.forecasts:
            issued, planned = self.forecasts[previous]
            self.completed.append((planned, now - issued))

    def row(self) -> dict:
        """Report tracking separately from approach/inspection fallback time."""
        return {
            "continuous_reference_seconds": self.planned,
            "continuous_actual_seconds": self.actual,
            "continuous_tracking_rmse": np.sqrt(self.error_squared / max(1, self.samples)),
            "forecasted_gate_count": len(self.completed),
            "forecast_gate_seconds": sum(pair[0] for pair in self.completed),
            "actual_forecast_gate_seconds": sum(pair[1] for pair in self.completed),
        }


def evaluate(
    config: str,
    runs: int,
    seed: int,
    render: bool,
    trace: bool,
    controller: str = "spline_controller.py",
    label: str = "",
    speed_scale: float | None = None,
    acceleration_scale: float | None = None,
):
    """Measure actual ordered crossings, completion times, and controller latency."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / config)
    if speed_scale is not None:
        cfg.controller.speed_scale = speed_scale
    if acceleration_scale is not None:
        cfg.controller.acceleration_scale = acceleration_scale
    controller_class = load_controller(root / "lsy_drone_racing/control" / controller)
    env = DroneRaceEnv(
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
    )
    results = []
    try:
        for race_seed in range(seed, seed + runs):
            obs, info = env.reset(seed=race_seed)
            planner = controller_class(obs, info, cfg)
            if (speed_scale is not None or acceleration_scale is not None) and not hasattr(
                planner, "speed_scale"
            ):
                raise ValueError("Speed sweeps need a controller with speed_scale support")
            previous_state = None
            latencies = []
            gate_speeds = []
            continuous_steps = 0
            peak_speed = 0.0
            path_length = 0.0
            timing = RaceTiming(len(obs["gate_sequence"]))
            reference_timing = ReferenceTiming()
            for step in range(30 * cfg.env.freq):
                before = time.perf_counter()
                action = planner.compute_control(obs, info)
                latencies.append(time.perf_counter() - before)
                reference_timing.observe(planner, obs, step / cfg.env.freq, 1 / cfg.env.freq)
                continuous_steps += planner.phase == "race"
                state = (
                    planner.entry,
                    planner.phase,
                    planner.holding,
                    tuple(np.round(planner.goal, 2)),
                )
                if trace and state != previous_state:
                    print(
                        f"  t={step / cfg.env.freq:.2f} {state} pos={np.round(obs['pos'], 3)}",
                        flush=True,
                    )
                previous_state = state
                last_position = np.asarray(obs["pos"]).copy()
                previous_gates = int(obs["n_gates_passed"])
                obs, _, terminated, truncated, info = env.step(action)
                peak_speed = max(peak_speed, float(np.linalg.norm(obs["vel"])))
                # Disabled drones are warped to (-1, -1, -1), including after
                # successful completion. Do not count that teleport as flight.
                if not terminated:
                    path_length += float(np.linalg.norm(np.asarray(obs["pos"]) - last_position))
                timing.step(
                    1 / cfg.env.freq,
                    planner.phase,
                    planner.holding,
                    float(np.linalg.norm(obs["vel"])),
                    int(obs["n_gates_passed"]),
                )
                if int(obs["n_gates_passed"]) > previous_gates:
                    gate_speeds.append(float(np.linalg.norm(obs["vel"])))
                reference_timing.crossing(
                    previous_gates, int(obs["n_gates_passed"]), (step + 1) / cfg.env.freq
                )
                if render:
                    planner.render_callback(env.sim)
                    env.render()
                if terminated or truncated:
                    break
            gates = int(obs["n_gates_passed"])
            success = gates == len(obs["gate_sequence"])
            outcome = "success" if success else "terminated" if terminated else "timeout"
            last_valid_position = last_position if terminated else np.asarray(obs["pos"])
            results.append(
                {
                    "seed": race_seed,
                    "speed_scale": getattr(planner, "speed_scale", 1.0),
                    "acceleration_scale": getattr(planner, "acceleration_scale", 1.0),
                    "success": success,
                    "outcome": outcome,
                    "gates": gates,
                    "seconds": (step + 1) / cfg.env.freq,
                    "replans": planner.replans,
                    "mean_gate_speed": float(np.mean(gate_speeds)) if gate_speeds else 0.0,
                    "peak_speed": peak_speed,
                    "path_length": path_length,
                    "last_valid_x": float(last_valid_position[0]),
                    "last_valid_y": float(last_valid_position[1]),
                    "last_valid_z": float(last_valid_position[2]),
                    "continuous_fraction": continuous_steps / (step + 1),
                    "predictive_fraction": getattr(planner, "mpc_steps", 0) / (step + 1),
                    "line_optimizations": getattr(planner, "line_optimizations", 0),
                    "moving_resumes": getattr(planner, "moving_resumes", 0),
                    "trajectory_attempts": getattr(planner, "trajectory_attempts", 0),
                    "trajectory_accepts": getattr(planner, "trajectory_accepts", 0),
                    "trajectory_time_saved": getattr(planner, "trajectory_time_saved", 0.0),
                    "force_gain": getattr(planner, "force_gain", 1.0),
                    "controller_p95_ms": 1000 * float(np.quantile(latencies, 0.95)),
                    "controller_max_ms": 1000 * max(latencies),
                    "fallback_reasons": json.dumps(
                        getattr(planner, "fallback_counts", {}), sort_keys=True
                    ),
                    **timing.row(),
                    **reference_timing.row(),
                }
            )
            print(
                f"seed={race_seed}: gates={gates}, success={success}, "
                f"time={results[-1]['seconds']:.2f}s, replans={planner.replans}",
                flush=True,
            )
            if trace and not success:
                print(
                    f"  failure at {last_position}, phase={planner.phase}, "
                    f"goal={planner.goal}, holding={planner.holding}",
                    flush=True,
                )
        successes = [row for row in results if row["success"]]
        print(
            f"Success: {len(successes)}/{runs}; mean gates: "
            f"{np.mean([row['gates'] for row in results]):.2f}",
            flush=True,
        )
        if successes:
            mean_sectors = [
                np.mean([r[f"sector_{index + 1}_seconds"] for r in successes])
                for index in range(timing.gates)
            ]
            print(f"Mean successful flight time: {np.mean([r['seconds'] for r in successes]):.2f}s")
            print(f"Mean stopped time: {np.mean([r['stopped_seconds'] for r in successes]):.2f}s")
            print(
                "Mean gate sectors: "
                + ", ".join(
                    f"{index + 1}: {seconds:.2f}s" for index, seconds in enumerate(mean_sectors)
                )
            )
            print(
                "Mean phase times: "
                + ", ".join(
                    f"{phase}: {np.mean([r[f'phase_{phase}_seconds'] for r in successes]):.2f}s"
                    for phase in timing.phases
                )
            )
        if not label and (speed_scale is not None or acceleration_scale is not None):
            label = f"speed{speed_scale or 1:g}_accel{acceleration_scale or 1:g}"
        suffix = "_" + re.sub(r"[^a-zA-Z0-9_.-]", "_", label) if label else ""
        output = (
            root
            / "saves/spline"
            / f"{Path(config).stem}_{Path(controller).stem}_{seed}_{runs}{suffix}.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="level2.toml")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--controller", default="spline_controller.py")
    parser.add_argument("--label", default="", help="Suffix to keep experiment CSVs separate")
    parser.add_argument(
        "--speed-scale", type=float, help="Scale cruise, gate and sensing speed caps"
    )
    parser.add_argument(
        "--acceleration-scale", type=float, help="Scale acceleration and cornering limits"
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    for value in [args.speed_scale, args.acceleration_scale]:
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error("Speed and acceleration scales must be positive and finite")
    evaluate(
        args.config,
        args.runs,
        args.seed,
        args.render,
        args.trace,
        args.controller,
        args.label,
        args.speed_scale,
        args.acceleration_scale,
    )
