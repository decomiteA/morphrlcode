"""Shared constants and hyperparameters for MJX training."""

from dataclasses import dataclass


ENV_NAME = "custom_ant"


@dataclass
class PPOConfig:
    num_envs: int = 8192
    batch_size: int = 2048
    num_minibatches: int = 32
    num_timesteps: int = 1_500_000_000
    episode_length: int = 1500
    unroll_length: int = 30
    num_updates_per_batch: int = 16
    learning_rate: float = 0.0003
    discounting: float = 0.995
    entropy_cost: float = 0.01
    num_evals: int = 10
    normalize_observations: bool = True
    seed: int = 0

    runs_dir: str = "runs"
    rollout_steps: int = 1500

    speed_min: float = 1.0    # change range for greater variation in speed, but may require more training time
    speed_max: float = 2.0
