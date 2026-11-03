"""PPO chooses velocity; the existing state controller handles flight.

Install: pixi run python -m pip install stable-baselines3==2.9.2
Check:   pixi run python -m lsy_drone_racing.control.direct_racing_rl check
Train:   pixi run python -m lsy_drone_racing.control.direct_racing_rl train
Watch:   pixi run python -m lsy_drone_racing.control.direct_racing_rl evaluate --render

Use --config level0.toml for an easier first experiment, then --resume to
continue that checkpoint on level2.toml. No course configuration files change.
Training starts from scratch; the existing trajectory-tracking weights differ.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "saves/velocity_simple_level0_2m/final.zip"
HISTORY = 3
UPCOMING_GATES = 2
NEAREST_OBSTACLES = 4
OBSERVATION_VERSION = 5
GAMMA = 0.99
MAX_SPEED = np.array([2.0, 2.0, 1.0], dtype=np.float32)  # World x/y/z, metres/second.
MAX_ACCELERATION = 2.0  # Metres/second squared; limits abrupt direction changes.
TAKEOFF_HEIGHT = 0.6
TAKEOFF_SECONDS = 2.0

# Starting values, not tuned racing parameters. Distances are in metres.
PROGRESS_REWARD = 2.0
GATE_REWARD = 50.0
FAILURE_COST = 2.0  # Simulator termination includes collisions and leaving bounds.
STALL_SECONDS = 40.0
STALL_IMPROVEMENT = 0.1  # A new best distance must improve by 10 cm to reset the timer.
STALL_COST = 10.0
EPISODE_SECONDS = 120.0  # Experiment limit; the standard course sim.py still uses 30 seconds.


class GateProgress:
    """Reward new distance records and detect persistent lack of progress."""

    def __init__(self, obs: dict):
        """Start a fresh progress timer for the current required gate."""
        self.passed = int(obs["n_gates_passed"])
        gate = int(obs["gate_sequence"][min(self.passed, len(obs["gate_sequence"]) - 1)])
        self.target = np.array(obs["gates_pos"][gate], copy=True)
        self.best_distance = float(np.linalg.norm(obs["pos"] - self.target))
        self.stall_distance = self.best_distance
        self.improvement = 0.0
        self.seconds = 0.0

    def update(self, obs: dict, dt: float) -> bool:
        """Return True after 40 seconds without a significant new best or a gate pass."""
        self.improvement = 0.0
        passed = int(obs["n_gates_passed"])
        if passed >= len(obs["gate_sequence"]):
            return False
        gate = int(obs["gate_sequence"][passed])
        if passed != self.passed or not np.array_equal(obs["gates_pos"][gate], self.target):
            # A sensor revelation changes the reference distance, not actual progress.
            self.__init__(obs)
            return False
        self.seconds += dt
        distance = float(np.linalg.norm(obs["pos"] - self.target))
        self.improvement = max(0.0, self.best_distance - distance)
        self.best_distance = min(self.best_distance, distance)
        # Small records earn proportionate reward, but the timeout resets only
        # after a cumulative 10 cm improvement, not for every millimetre of noise.
        if self.best_distance <= self.stall_distance - STALL_IMPROVEMENT:
            self.stall_distance = self.best_distance
            self.seconds = 0.0
        return self.seconds + 1e-8 >= STALL_SECONDS

    def state(self) -> np.ndarray:
        """Let the policy observe its progress reference and remaining stall budget."""
        return np.array(
            [self.best_distance / 5, self.stall_distance / 5, self.seconds / STALL_SECONDS],
            dtype=np.float32,
        )


class Observations:
    """Shared by training and the adapter; reads only public sensor observations."""

    def __init__(self):
        """Keep the last three drone states."""
        self.history = deque(maxlen=HISTORY)

    def encode(
        self,
        obs: dict,
        previous_action: np.ndarray,
        command_state: np.ndarray,
        progress_state: np.ndarray,
        reset: bool = False,
    ) -> np.ndarray:
        """Flatten visible geometry, state history, and the previous action."""
        rotation = Rotation.from_quat(obs["quat"])
        body_from_world = rotation.inv()
        state = np.concatenate(
            [
                np.asarray(obs["pos"]) / 5.0,
                rotation.as_matrix().reshape(-1),
                body_from_world.apply(obs["vel"]) / 5.0,
                np.asarray(obs["ang_vel"]) / 10.0,
            ]
        )
        if reset:
            self.history.clear()
            self.history.extend([state.copy() for _ in range(HISTORY)])
        else:
            self.history.append(state)

        passed = int(obs["n_gates_passed"])
        sequence = obs["gate_sequence"]
        gates = []
        for offset in range(UPCOMING_GATES):
            index = passed + offset
            if index >= len(sequence):
                gates.extend(np.zeros(11))
                continue
            gate = int(sequence[index])
            gate_rotation = body_from_world * Rotation.from_quat(obs["gates_quat"][gate])
            # First two rotation-matrix columns encode orientation without Euler jumps.
            gates.extend(body_from_world.apply(obs["gates_pos"][gate] - obs["pos"]) / 5.0)
            gates.extend(gate_rotation.as_matrix()[:, :2].reshape(-1))
            gates.extend([obs["gate_sequence_direction"][index], obs["gates_visited"][gate]])

        obstacles = np.zeros((NEAREST_OBSTACLES, 4))
        relative = np.asarray(obs["obstacles_pos"]) - obs["pos"]
        nearest = np.argsort(np.linalg.norm(relative[:, :2], axis=-1))[:NEAREST_OBSTACLES]
        if len(nearest):
            obstacles[: len(nearest), :3] = body_from_world.apply(relative[nearest]) / 5.0
            obstacles[: len(nearest), 3] = np.asarray(obs["obstacles_visited"])[nearest]

        features = np.concatenate(
            [
                *self.history,
                gates,
                obstacles.reshape(-1),
                previous_action,
                command_state,
                progress_state,
                [passed / len(sequence)],
            ]
        ).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite racing observation")
        return features


class VelocityCommands:
    """Turn velocity choices into smooth full-state commands. Zero means hold position."""

    def __init__(self, obs: dict, frequency: int):
        """Start with an automatic vertical takeoff at the initial x/y position."""
        self.dt = 1 / frequency
        self.takeoff_steps = round(TAKEOFF_SECONDS * frequency)
        self.position = np.array(obs["pos"], dtype=np.float32, copy=True)
        self.position[2] = max(TAKEOFF_HEIGHT, self.position[2])
        self.velocity = np.zeros(3, dtype=np.float32)

    def state(self, obs: dict) -> np.ndarray:
        """Expose the target error and filtered velocity so the policy sees controller memory."""
        return np.concatenate([(self.position - obs["pos"]) / 5, self.velocity / 5])

    def command(self) -> np.ndarray:
        """Position, velocity, zero acceleration, identity quaternion, zero body rates."""
        return np.concatenate(
            [self.position, self.velocity, np.zeros(3), [0, 0, 0, 1], np.zeros(3)]
        ).astype(np.float32)

    def step(self, action: np.ndarray, obs: dict) -> np.ndarray:
        """Integrate bounded world-frame velocity while keeping the target near the drone."""
        desired_velocity = np.clip(np.asarray(action, dtype=np.float32), -1, 1) * MAX_SPEED
        change = desired_velocity - self.velocity
        length = np.linalg.norm(change)
        self.velocity += change * min(1.0, MAX_ACCELERATION * self.dt / max(length, 1e-8))
        self.position += self.velocity * self.dt
        # A lagging drone must not accumulate a target far beyond its actual position.
        error = self.position - obs["pos"]
        self.position = np.asarray(obs["pos"]) + error * min(
            1.0, 0.35 / max(np.linalg.norm(error), 1e-8)
        )
        # Keep exploration airborne; obstacle collisions and track bounds still terminate.
        if self.position[2] < 0.25:
            self.position[2] = 0.25
            self.velocity[2] = max(0.0, self.velocity[2])
        return self.command()


class RacingEnv(gym.Env):
    """Single race. PPO handles episode resets and rollout collection."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config: str = "level0.toml", render: bool = False):
        """Build the existing race simulator with position and velocity control."""
        self.config = load_config(ROOT / "config" / config)
        cfg = self.config
        self.render_mode = "human" if render else None
        # Construct directly so the simulator's own limit changes too, rather
        # than only changing Gymnasium's outer TimeLimit wrapper.
        self.env = DroneRaceEnv(
            freq=cfg.env.freq,
            sim_config=cfg.sim,
            track=cfg.env.track,
            sensor_range=cfg.env.sensor_range,
            control_mode="state",
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=0,
            max_episode_steps=round(EPISODE_SECONDS * cfg.env.freq),
        )
        self.encoder = Observations()
        self.dt = 1 / cfg.env.freq
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        obs, _ = self.env.reset(seed=0)
        self.commands = VelocityCommands(obs, cfg.env.freq)
        self.progress = GateProgress(obs)
        features = self.encoder.encode(
            obs, np.zeros(3), self.commands.state(obs), self.progress.state(), reset=True
        )
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, features.shape, np.float32)

    @staticmethod
    def numpy_observation(obs: dict) -> dict:
        """Move public JAX observations to NumPy."""
        return {key: np.asarray(value) for key, value in obs.items()}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Start a new randomized race and clear policy history."""
        super().reset(seed=seed)
        # Draw a fresh reproducible seed instead of reusing the config's race seed.
        race_seed = int(self.np_random.integers(1, 2**31 - 1))
        obs, info = self.env.reset(seed=race_seed, options=options)
        self.commands = VelocityCommands(obs, self.config.env.freq)
        self.steps = 0
        # PPO starts after takeoff. The adapter performs the same commands before inference.
        for _ in range(self.commands.takeoff_steps):
            obs, _, terminated, truncated, info = self.env.step(self.commands.command())
            self.steps += 1
            if self.render_mode:
                self.render()
            if terminated or truncated:
                raise RuntimeError("Automatic takeoff failed before the policy started")
        self.obs = self.numpy_observation(obs)
        self.progress = GateProgress(self.obs)
        self.previous_action = np.zeros(3, dtype=np.float32)
        return self.encoder.encode(
            self.obs,
            self.previous_action,
            self.commands.state(self.obs),
            self.progress.state(),
            reset=True,
        ), info

    def step(self, action: np.ndarray) -> tuple:
        """Apply one command and calculate the racing reward."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        old = self.obs
        command = self.commands.step(action, old)
        obs, _, terminated, truncated, info = self.env.step(command)
        obs = self.numpy_observation(obs)
        self.steps += 1
        passed = int(obs["n_gates_passed"])
        new_gates = passed - int(old["n_gates_passed"])
        success = passed == len(obs["gate_sequence"])
        failure = bool(terminated) and not success
        stalled = False
        if not terminated and not truncated:
            stalled = self.progress.update(obs, self.dt)
        # Stalling is a task failure, not a time-limit truncation: PPO must not
        # bootstrap future value from a state where this task has already ended.
        terminated = bool(terminated) or stalled

        # Only a new closest point earns progress. Retreating and approaching
        # again or receiving a newly revealed gate position earns nothing.
        progress = 0.0
        if not failure and not success and not truncated and new_gates == 0:
            progress = self.progress.improvement
        terms = {
            "progress": PROGRESS_REWARD * float(progress),
            "gates": GATE_REWARD * new_gates,
            "failure": -FAILURE_COST * failure,
            "stall": -STALL_COST * stalled,
        }
        self.obs = obs
        self.previous_action = action.copy()
        info = dict(
            info,
            reward_terms=terms,
            is_success=success,
            gates_passed=passed,
            flight_time=self.steps * self.dt,
            stalled=stalled,
        )
        if self.render_mode:
            self.render()
        return (
            self.encoder.encode(obs, action, self.commands.state(obs), self.progress.state()),
            sum(terms.values()),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> None:
        """Show the simulator viewer."""
        return self.env.render()

    def close(self):
        """Release the simulator and viewer."""
        self.env.close()


def load_policy(path: str | Path) -> PPO:
    """Load a checkpoint and reject incompatible observation formats."""
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    if getattr(model, "racing_observation_version", None) != OBSERVATION_VERSION:
        raise ValueError(
            f"Checkpoint is incompatible: expected observation version {OBSERVATION_VERSION}"
        )
    return model


def train(args: argparse.Namespace):
    """Train PPO and save periodic and final checkpoints."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class TrainingProgress(CheckpointCallback):
        """Save checkpoints and log gates passed in completed training episodes."""

        def __init__(self):
            super().__init__(
                save_freq=max(1, 50_000 // args.envs),
                save_path=str(args.model.parent),
                name_prefix="racing",
            )
            self.gates = deque(maxlen=100)
            self.best_gates = 0

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.gates.append(info["gates_passed"])
                    self.best_gates = max(self.best_gates, info["gates_passed"])
            if self.gates:
                self.logger.record("rollout/gates_mean", np.mean(self.gates))
                self.logger.record("rollout/gates_best", self.best_gates)
            return super()._on_step()

    args.model.parent.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(RacingEnv(args.config)) for _ in range(args.envs)])
    try:
        if args.resume:
            model = load_policy(args.resume)
            if model.racing_frequency != env.envs[0].unwrapped.config.env.freq:
                raise ValueError("Checkpoint control frequency differs from environment")
            model.set_env(env)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                device="cpu",
                seed=args.seed,
                verbose=1,
                n_steps=args.rollout_steps,
                batch_size=64,
                learning_rate=3e-4,
                gamma=GAMMA,
                gae_lambda=0.95,
                ent_coef=0.005,
                policy_kwargs={"net_arch": [128, 128]},
            )
        model.racing_observation_version = OBSERVATION_VERSION
        model.racing_frequency = env.envs[0].unwrapped.config.env.freq
        model.racing_config = args.config
        checkpoints = TrainingProgress()
        try:
            model.learn(args.steps, callback=checkpoints, reset_num_timesteps=not bool(args.resume))
        except KeyboardInterrupt:
            print("Interrupted; saving current policy.")
        model.save(args.model)
        print(f"Saved {args.model}")
    finally:
        env.close()


def evaluate(args: argparse.Namespace):
    """Measure completion and successful flight times on separate seeds."""
    model = load_policy(args.model)
    env = RacingEnv(args.config, render=args.render)
    times, gates = [], []
    try:
        if model.racing_frequency != env.config.env.freq:
            raise ValueError("Checkpoint control frequency differs from environment")
        for run in range(args.runs):
            obs, _ = env.reset(seed=args.seed + run)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            gates.append(info["gates_passed"])
            if info["is_success"]:
                times.append(info["flight_time"])
            print(
                f"Run {run + 1}: success={info['is_success']}, gates={gates[-1]}, "
                f"time={info['flight_time']:.2f}s"
            )
        print(
            f"Success: {len(times)}/{args.runs} ({len(times) / args.runs:.0%}); "
            f"mean gates: {np.mean(gates):.2f}"
        )
        if times:
            print(f"Successful flight times: mean={np.mean(times):.2f}s, best={min(times):.2f}s")
    finally:
        env.close()


def check(args: argparse.Namespace):
    """Exercise the real environment, observation space, rewards, and reset path."""
    from stable_baselines3.common.env_checker import check_env

    env = RacingEnv(args.config)
    try:
        check_env(env, warn=True)
        print("Environment check passed.")
    finally:
        env.close()


def main():
    """Parse the training, evaluation, or environment-check command."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["check", "train", "evaluate"])
    parser.add_argument("--config", default="level0.toml")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument(
        "--envs", type=int, default=1, help="Sequential CPU environments; start small"
    )
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.envs, args.runs) < 1 or args.rollout_steps < 2:
        parser.error("Counts must be positive; --rollout-steps must be at least 2")
    if args.envs * args.rollout_steps % 64 and args.command == "train":
        parser.error("--envs * --rollout-steps must be a multiple of 64")
    if args.seed is None:
        args.seed = 10_000 if args.command == "evaluate" else 0
    {"train": train, "evaluate": evaluate, "check": check}[args.command](args)


if __name__ == "__main__":
    main()
