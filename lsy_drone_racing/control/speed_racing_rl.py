"""PPO chooses speed; the classical planner chooses the route and controls flight.

pixi run python -m lsy_drone_racing.control.speed_racing_rl train --steps 350000
pixi run python -m lsy_drone_racing.control.speed_racing_rl evaluate --runs 20
Add --baseline for the classical speed rule or --render to watch.
Training uses time and failed-lap cost only. Existing 350k weights remain available.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.continuous_spline_controller import ContinuousSplineController
from lsy_drone_racing.control.spline_controller import APPROACH_DISTANCE, gate_points
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_MODEL = ROOT / "saves/speed_racing_350k/final.zip"
DEFAULT_TRAINING_MODEL = ROOT / "saves/speed_timing_v2/final.zip"
DEFAULT_MODEL = DEFAULT_TRAINING_MODEL if DEFAULT_TRAINING_MODEL.is_file() else PREVIOUS_MODEL
VERSION = 1
REWARD_VERSION = 2
GAMMA = 0.99
EPISODE_SECONDS = 30
FAILED_LAP_SECONDS = 35.0  # A failed race costs more than any on-time completed lap.
PHASES = ("takeoff", "race", "approach", "inspect", "cross", "hold")


class SpeedSetting:
    """Map one policy action to a smooth multiplier between 0.25 and 1.75."""

    def __init__(self):
        """Start at the classical speed."""
        self.factor = 1.0

    def update(self, action: np.ndarray, dt: float) -> float:
        """Limit multiplier changes to 1.5 per second."""
        target = 1.0 + 0.75 * float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
        self.factor += float(np.clip(target - self.factor, -1.5 * dt, 1.5 * dt))
        return self.factor


def pace_reward(
    dt: float, elapsed: float, failure: bool, takeoff_seconds: float = 0.0
) -> dict[str, float]:
    """Charge time; make every failed lap cost 35 seconds before discounting.

    Charging the remaining failure cost prevents crashing early to avoid time
    penalties. Takeoff is charged once because it runs automatically in reset().
    """
    return {
        "time": -dt - takeoff_seconds,
        "failure": -max(0.0, FAILED_LAP_SECONDS - elapsed) if failure else 0.0,
    }


def speed_features(
    obs: dict, planner: ContinuousSplineController, factor: float, seconds: float
) -> np.ndarray:
    """Encode public geometry, upcoming bends, tracking state and policy memory."""
    body = Rotation.from_quat(obs["quat"]).inv()
    position = np.asarray(obs["pos"])
    passed = int(obs["n_gates_passed"])
    gates = []
    for offset in range(2):
        entry = passed + offset
        if entry >= len(obs["gate_sequence"]):
            gates.extend(np.zeros(7))
            continue
        before, centre, _ = gate_points(obs, entry)
        gate = int(obs["gate_sequence"][entry])
        gates.extend(body.apply(centre - position) / 3)
        gates.extend(body.apply((centre - before) / APPROACH_DISTANCE))
        gates.append(float(obs["gates_visited"][gate]))

    poles = np.zeros((3, 4))
    relative = np.asarray(obs["obstacles_pos"]) - position
    nearest = np.argsort(np.linalg.norm(relative[:, :2], axis=-1))[:3]
    poles[: len(nearest), :3] = body.apply(relative[nearest]) / 3
    poles[: len(nearest), 3] = np.asarray(obs["obstacles_visited"])[nearest]

    if planner.phase == "race":
        curve = planner.curve
        distance = curve.distance[planner.progress]
        samples = np.minimum(distance + np.array([0.0, 0.25, 0.6, 1.2]), curve.distance[-1])
        points = curve.path(samples)
        velocity = curve.path(samples, nu=1)
        acceleration = curve.path(samples, nu=2)
        speeds = np.interp(samples, curve.distance, curve.speed)
    else:
        samples = np.minimum(planner.elapsed + np.array([0.0, 0.15, 0.4, 0.8]), planner.duration)
        points = planner.curve(samples)
        velocity = planner.curve(samples, nu=1)
        acceleration = planner.curve(samples, nu=2)
        speeds = np.linalg.norm(velocity, axis=-1)
    norm = np.linalg.norm(velocity, axis=-1)
    tangents = velocity / np.maximum(norm[:, None], 1e-6)
    curvature = np.linalg.norm(np.cross(velocity, acceleration), axis=-1) / np.maximum(
        norm**3, 1e-6
    )
    route = np.column_stack(
        [
            body.apply(points - position) / 2,
            body.apply(tangents),
            speeds / 4,
            np.minimum(curvature, 20) / 5,
        ]
    )
    phase = "hold" if planner.holding else planner.phase
    features = np.concatenate(
        [
            body.apply(obs["vel"]) / 4,
            np.asarray(obs["ang_vel"]) / 10,
            body.as_matrix().reshape(-1),
            gates,
            poles.reshape(-1),
            route.reshape(-1),
            [float(phase == value) for value in PHASES],
            body.apply(planner.goal - position) / 3,
            [(factor - 1) / 0.75, passed / len(obs["gate_sequence"]), seconds / 30],
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite speed-policy observation")
    return np.clip(features, -10, 10)


class SpeedRacingEnv(gym.Env):
    """Same 30-second level 2 race, with one speed action per 50 Hz control step."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config: str = "level2.toml", render: bool = False):
        """Build the simulator; leave all classical controller settings unchanged."""
        self.config = load_config(ROOT / "config" / config)
        cfg = self.config
        self.render_mode = "human" if render else None
        self.dt = 1 / cfg.env.freq
        self.env = DroneRaceEnv(
            freq=cfg.env.freq,
            sim_config=cfg.sim,
            track=cfg.env.track,
            sensor_range=cfg.env.sensor_range,
            control_mode="state",
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=0,
            max_episode_steps=EPISODE_SECONDS * cfg.env.freq,
        )
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        obs, info = self.env.reset(seed=0)
        self.planner = ContinuousSplineController(obs, info, cfg)
        features = speed_features(obs, self.planner, 1.0, 0.0)
        self.observation_space = gym.spaces.Box(-10, 10, features.shape, np.float32)

    @staticmethod
    def numpy_obs(obs: dict) -> dict:
        """Transfer each public observation to NumPy once."""
        return {key: np.asarray(value) for key, value in obs.items()}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Perform classical takeoff, then start policy decisions."""
        super().reset(seed=seed)
        race_seed = seed if seed is not None else int(self.np_random.integers(1, 2**31 - 1))
        obs, info = self.env.reset(seed=race_seed)
        obs = self.numpy_obs(obs)
        self.planner = ContinuousSplineController(obs, info, self.config)
        self.setting = SpeedSetting()
        self.steps = 0
        self.factor_sum = 0.0
        self.decisions = 0
        self.seed_used = race_seed
        while self.planner.phase == "takeoff":
            command = self.planner.compute_control(obs, info)
            obs, _, terminated, truncated, info = self.env.step(command)
            obs = self.numpy_obs(obs)
            self.steps += 1
            if self.render_mode:
                self.render()
            if terminated or truncated:
                raise RuntimeError("Classical takeoff failed before the speed policy started")
        self.obs = obs
        self.takeoff_seconds = self.steps * self.dt
        return speed_features(obs, self.planner, 1.0, self.steps * self.dt), {}

    def step(self, action: np.ndarray) -> tuple:
        """Apply speed and charge elapsed time, with a fixed failed-lap score."""
        factor = self.setting.update(action, self.dt)
        self.planner.speed_factor = factor
        command = self.planner.compute_control(self.obs)
        obs, _, terminated, truncated, _ = self.env.step(command)
        obs = self.numpy_obs(obs)
        self.steps += 1
        self.decisions += 1
        self.factor_sum += factor
        passed = int(obs["n_gates_passed"])
        success = passed == len(obs["gate_sequence"])
        failure = bool(terminated or truncated) and not success
        terms = pace_reward(
            self.dt,
            self.steps * self.dt,
            failure,
            self.takeoff_seconds if self.decisions == 1 else 0.0,
        )
        self.obs = obs
        info = dict(
            is_success=success,
            gates_passed=passed,
            flight_time=self.steps * self.dt,
            speed_factor=factor,
            mean_speed_factor=self.factor_sum / self.decisions,
            race_seed=self.seed_used,
            reward_terms=terms,
        )
        if self.render_mode:
            self.render()
        return (
            speed_features(obs, self.planner, factor, self.steps * self.dt),
            float(sum(terms.values())),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> None:
        """Draw the route and update the simulator viewer."""
        self.planner.render_callback(self.env.sim)
        self.env.render()

    def close(self) -> None:
        """Close the simulator and viewer."""
        self.env.close()


def load_policy(path: str | Path) -> PPO:
    """Reject checkpoints trained for a different policy observation format."""
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    model = PPO.load(path, device="cpu")
    if getattr(model, "speed_policy_version", None) != VERSION:
        raise ValueError("This is not a compatible speed-policy checkpoint")
    return model


def train(args: argparse.Namespace) -> None:
    """Train from scratch and retain checkpoints every 50,000 decisions."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(1)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    completed = deque(maxlen=100)
    started = time.monotonic()
    progress_path = args.model.parent / "progress.jsonl"

    class Progress(BaseCallback):
        """Log actual episode results without running evaluation during collection."""

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    completed.append(info)
            if self.num_timesteps % 10000 == 0:
                good = [value for value in completed if value["is_success"]]
                record = dict(
                    steps=self.num_timesteps,
                    seconds=time.monotonic() - started,
                    episodes_recent=len(completed),
                    success_recent=len(good) / len(completed) if completed else 0.0,
                    mean_success_seconds=float(np.mean([value["flight_time"] for value in good]))
                    if good
                    else None,
                    mean_gates=float(np.mean([value["gates_passed"] for value in completed]))
                    if completed
                    else 0.0,
                )
                line = json.dumps(record)
                print(line, flush=True)
                with progress_path.open("a") as handle:
                    handle.write(line + "\n")
            if self.num_timesteps % 50000 == 0:
                self.model.save(args.model.parent / f"speed_{self.num_timesteps}_steps.zip")
            return True

    env = DummyVecEnv([lambda: Monitor(SpeedRacingEnv(args.config))])
    try:
        settings = dict(
            reward_version=REWARD_VERSION,
            steps=args.steps,
            seed=args.seed,
            config=args.config,
            gamma=GAMMA,
            time_cost_per_second=1.0,
            failed_lap_seconds=FAILED_LAP_SECONDS,
            navigation_bonuses=False,
        )
        (args.model.parent / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n")
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            seed=args.seed,
            verbose=0,
            n_steps=1000,
            batch_size=100,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=GAMMA,
            gae_lambda=0.95,
            ent_coef=0.002,
            policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.7),
        )
        model.speed_policy_version = VERSION
        model.speed_reward_version = REWARD_VERSION
        model.speed_policy_frequency = env.envs[0].unwrapped.config.env.freq
        model.speed_policy_config = args.config
        model.learn(args.steps, callback=Progress())
        model.save(args.model)
        print(f"Saved {args.model}; trained decisions={model.num_timesteps}", flush=True)
    finally:
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate deterministic speed choices on explicit, reproducible race seeds."""
    fixed = args.baseline or args.constant_factor is not None
    model = None if fixed else load_policy(args.model)
    if model is not None:
        reward_version = getattr(model, "speed_reward_version", 1)
        print(f"Model: {args.model}; trained reward version: {reward_version}", flush=True)
    env = SpeedRacingEnv(args.config, render=args.render)
    rows = []
    try:
        for seed in range(args.seed, args.seed + args.runs):
            features, _ = env.reset(seed=seed)
            done = False
            while not done:
                if model is None:
                    factor = 1.0 if args.constant_factor is None else args.constant_factor
                    action = np.array([(factor - 1.0) / 0.75])
                else:
                    action = model.predict(features, deterministic=True)[0]
                features, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rows.append(
                dict(
                    seed=seed,
                    success=info["is_success"],
                    gates=info["gates_passed"],
                    seconds=info["flight_time"],
                    mean_speed_factor=info["mean_speed_factor"],
                )
            )
            print(rows[-1], flush=True)
        good = [row for row in rows if row["success"]]
        print(f"Success: {len(good)}/{len(rows)}", flush=True)
        if good:
            mean = np.mean([row["seconds"] for row in good])
            best = min(row["seconds"] for row in good)
            print(f"Successful laps: mean={mean:.2f}s, best={best:.2f}s")
        name = "baseline" if args.baseline else args.model.stem
        if args.constant_factor is not None:
            name = f"constant_{args.constant_factor:g}"
        output = args.model.parent / f"eval_{name}_{args.seed}_{args.runs}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results: {output}", flush=True)
    finally:
        env.close()


def main() -> None:
    """Run the environment check, training or evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "train", "evaluate"])
    parser.add_argument("--config", default="level2.toml")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--steps", type=int, default=350000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--constant-factor", type=float, help="Compare against a fixed speed multiplier"
    )
    args = parser.parse_args()
    if args.model is None:
        # New training must not overwrite the previous, differently rewarded run.
        args.model = DEFAULT_TRAINING_MODEL if args.command == "train" else DEFAULT_MODEL
    if args.steps <= 0 or args.runs <= 0:
        parser.error("Step and run counts must be positive")
    if args.constant_factor is not None and not 0.25 <= args.constant_factor <= 1.75:
        parser.error("Constant speed factor must be between 0.25 and 1.75")
    if args.command == "train" and args.steps % 1000:
        parser.error("Training steps must be a multiple of the 1,000-step rollout")
    if args.seed is None:
        args.seed = 42 if args.command == "train" else 10000
    if args.command == "check":
        from stable_baselines3.common.env_checker import check_env

        env = SpeedRacingEnv(args.config)
        try:
            check_env(env, warn=True)
            print("Speed-policy environment check passed")
        finally:
            env.close()
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
