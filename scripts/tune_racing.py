"""Bounded simulator search with early rejection, checkpoints and untouched validation.

pixi run python scripts/tune_racing.py
Rerun the same command to resume from saved race CSVs.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from lsy_drone_racing.control.tuned_racing_controller import TunedRacingController
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "saves/parameter_search"
BOUNDS = {
    "profile_acceleration": (12.0, 22.0),
    "profile_mpc_acceleration": (16.0, 24.0),
    "profile_gate_speed": (2.0, 3.5),
    "profile_sensor_speed": (1.6, 2.5),
    "tracking_advance": (0.0, 0.04),
    "tracking_velocity_weight": (2.0, 7.0),
    "tracking_smoothing_weight": (1.0, 2.2),
    "profile_gate_offset": (0.0, 0.075),
}
BASELINE = dict(zip(BOUNDS, [16.0, 20.0, 2.8, 2.0, 0.025, 4.0, 1.5, 0.05]))
SCREEN_SEEDS = [10003, 10008, 20002, 20008, 30001, 30002]
TRAIN_SEEDS = [base + offset for base in [10000, 20000, 30000] for offset in range(20)]
MIDDLE_SEEDS = sorted(
    set(
        SCREEN_SEEDS
        + [base + offset for base in [10000, 20000, 30000] for offset in range(5)]
        + [10005, 20005, 30005]
    )
)
VALIDATION_SEEDS = list(range(60000, 60030))
ENV = CONFIG = None


def initialize_worker() -> None:
    """Reuse a single compiled simulator per worker; reseed every race explicitly."""
    global ENV, CONFIG
    CONFIG = load_config(ROOT / "config/level2_racing_tracking.toml")
    ENV = DroneRaceEnv(
        freq=CONFIG.env.freq,
        sim_config=CONFIG.sim,
        track=CONFIG.env.track,
        sensor_range=CONFIG.env.sensor_range,
        control_mode=CONFIG.env.control_mode,
        disturbances=CONFIG.env.get("disturbances"),
        randomizations=CONFIG.env.get("randomizations"),
        seed=0,
    )
    atexit.register(ENV.close)


def race(parameters: dict, seed: int, candidate: int, stage: str) -> dict:
    """Measure completion and actual simulated flight time; failures never count as finishes."""
    config = copy.deepcopy(CONFIG)
    config.controller.update(parameters)
    config.controller.file = "tuned_racing_controller.py"
    obs, info = ENV.reset(seed=seed)
    planner = TunedRacingController(obs, info, config)
    latencies = []
    stopped = 0.0
    started = time.perf_counter()
    terminated = truncated = False
    for step in range(30 * config.env.freq):
        before = time.perf_counter()
        action = planner.compute_control(obs, info)
        latencies.append(time.perf_counter() - before)
        obs, _, terminated, truncated, info = ENV.step(action)
        if planner.phase != "takeoff" and np.linalg.norm(obs["vel"]) < 0.2:
            stopped += 1 / config.env.freq
        if terminated or truncated:
            break
    gates = int(obs["n_gates_passed"])
    success = gates == len(obs["gate_sequence"])
    return {
        "candidate": candidate,
        "stage": stage,
        "seed": seed,
        "success": success,
        "gates": gates,
        "seconds": (step + 1) / config.env.freq,
        "outcome": "success" if success else "terminated" if terminated else "timeout",
        "stopped_seconds": stopped,
        "controller_p95_ms": np.percentile(latencies, 95) * 1000,
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_job(
    candidate: int, parameters: dict, seeds: list[int], minimum: float, stage: str
) -> dict:
    """Append each finished race immediately so interrupted searches can resume."""
    path = OUTPUT / "races" / f"candidate_{candidate:02d}.csv"
    cached = {}
    if path.exists():
        for row in csv.DictReader(path.open()):
            row["success"] = row["success"] == "True"
            row["seconds"] = float(row["seconds"])
            row["seed"] = int(row["seed"])
            cached[row["seed"]] = row
    required = math.ceil(len(seeds) * minimum)
    rows = []
    for seed in seeds:
        if seed not in cached:
            row = race(parameters, seed, candidate, stage)
            header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                if header:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
            cached[seed] = row
        rows.append(cached[seed])
        wins = sum(row["success"] for row in rows)
        if wins + len(seeds) - len(rows) < required:
            break
    successes = [float(row["seconds"]) for row in rows if row["success"]]
    return {
        "candidate": candidate,
        "stage": stage,
        "runs": len(rows),
        "planned": len(seeds),
        "completed": len(successes),
        "qualified": len(rows) == len(seeds) and len(successes) >= required,
        "successful_mean": float(np.mean(successes)) if successes else None,
        "parameters": parameters,
    }


def run_stage(
    pool: ProcessPoolExecutor, candidates: dict, seeds: list[int], minimum: float, name: str
) -> list:
    """Compare every surviving candidate on identical seeds and retain the whole result table."""
    futures = [
        pool.submit(evaluate_job, key, parameters, seeds, minimum, name)
        for key, parameters in candidates.items()
    ]
    results = []
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        mean = result["successful_mean"]
        print(
            f"{name}: {result['candidate']:02d} {result['completed']}/{result['runs']} "
            f"mean={mean:.3f}s qualified={result['qualified']}"
            if mean is not None
            else f"{name}: {result['candidate']:02d} no completions",
            flush=True,
        )
        (OUTPUT / f"{name}.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def fastest(results: list, count: int) -> list[int]:
    """Choose by speed only after applying the stage's completion requirement."""
    eligible = [r for r in results if r["qualified"] and r["candidate"] != 0]
    return [r["candidate"] for r in sorted(eligible, key=lambda r: r["successful_mean"])[:count]]


def main(workers: int = 3) -> None:
    """Screen 30 configurations, promote six and then two, and freeze one for fresh validation."""
    # Crazyflow initializes SciPy's array API before SciPy itself is imported.
    from scipy.stats import qmc

    (OUTPUT / "races").mkdir(parents=True, exist_ok=True)
    parameters_path = OUTPUT / "candidates.json"
    if parameters_path.exists():
        candidates = {int(k): v for k, v in json.loads(parameters_path.read_text()).items()}
    else:
        sample = qmc.LatinHypercube(d=len(BOUNDS), seed=20261103).random(29)
        limits = np.array(list(BOUNDS.values()))
        values = qmc.scale(sample, limits[:, 0], limits[:, 1])
        candidates = {
            0: BASELINE,
            **{
                index + 1: dict(zip(BOUNDS, map(float, point)))
                for index, point in enumerate(values)
            },
        }
        parameters_path.write_text(json.dumps(candidates, indent=2) + "\n")
    source = ROOT / "lsy_drone_racing/control/tuned_racing_controller.py"
    manifest = {
        "bounds": BOUNDS,
        "screen_seeds": SCREEN_SEEDS,
        "middle_seeds": MIDDLE_SEEDS,
        "training_seeds": TRAIN_SEEDS,
        "untouched_validation_seeds": VALIDATION_SEEDS,
        "controller_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError(
                "Search inputs changed; archive the old search before starting another"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        screen = run_stage(pool, candidates, SCREEN_SEEDS, 0.50, "screen")
        middle_ids = [0, *fastest(screen, 6)]
        middle = run_stage(
            pool, {i: candidates[i] for i in middle_ids}, MIDDLE_SEEDS, 0.50, "middle"
        )
        final_ids = [0, *fastest(middle, 2)]
        final = run_stage(
            pool, {i: candidates[i] for i in final_ids}, TRAIN_SEEDS, 0.55, "training"
        )
        qualified = [r for r in final if r["qualified"]]
        winner = min(qualified, key=lambda r: r["successful_mean"])
        (OUTPUT / "frozen_winner.json").write_text(json.dumps(winner, indent=2) + "\n")
        if winner["candidate"] == 0:
            print(
                "No searched configuration beat the incumbent at the completion target.", flush=True
            )
            return
        validation = run_stage(
            pool,
            {i: candidates[i] for i in [0, winner["candidate"]]},
            VALIDATION_SEEDS,
            0.55,
            "validation",
        )
        (OUTPUT / "result.json").write_text(
            json.dumps({"winner": winner, "validation": validation}, indent=2) + "\n"
        )
        print(
            "Search finished; inspect fresh validation before selecting a replacement.", flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    main(parser.parse_args().workers)
