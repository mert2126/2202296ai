"""Central configuration for the autonomous driving PPO pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


ENV_ID: str = "highway-fast-v0"
LEARNING_RATE: float = 5e-4
GAMMA: float = 0.8
N_STEPS: int = 1024
BATCH_SIZE: int = 64

DEFAULT_SEED: int = 42
TOTAL_TIMESTEPS: int = 100_000
LOG_DIR: str = "logs"
CHECKPOINT_DIR: str = "checkpoints"
ASSETS_DIR: str = "assets"

FINAL_MODEL_NAME: str = "ppo_highway_final.zip"
HALF_TRAINED_MODEL_NAME: str = "ppo_highway_half_trained.zip"
MONITOR_FILE_NAME: str = "monitor.csv"
EVOLUTION_GIF_NAME: str = "evolution.gif"
REWARD_PLOT_NAME: str = "reward_plot.png"


@dataclass(frozen=True, slots=True)
class PPOHyperparameters:
    """Explicit PPO hyperparameters used across training and documentation."""

    learning_rate: float = LEARNING_RATE
    gamma: float = GAMMA
    n_steps: int = N_STEPS
    batch_size: int = BATCH_SIZE
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Return a stable-baselines3 compatible parameter mapping."""

        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Runtime settings for the highway environment."""

    env_id: str = ENV_ID
    seed: int = DEFAULT_SEED
    render_mode: str = "rgb_array"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Filesystem and training duration settings."""

    total_timesteps: int = TOTAL_TIMESTEPS
    log_dir: str = LOG_DIR
    checkpoint_dir: str = CHECKPOINT_DIR
    model_name: str = "ppo_highway"


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Output configuration for evaluation artifacts."""

    assets_dir: str = ASSETS_DIR
    gif_name: str = EVOLUTION_GIF_NAME
    plot_name: str = REWARD_PLOT_NAME
    video_fps: int = 12
    episodes_per_stage: int = 3
    max_episode_steps: int = 250


def get_ppo_hyperparameters() -> Dict[str, Any]:
    """Return PPO hyperparameters as a dictionary."""

    return PPOHyperparameters().to_dict()


def get_environment_config() -> EnvironmentConfig:
    """Return the default environment configuration."""

    return EnvironmentConfig()


def get_training_config() -> TrainingConfig:
    """Return the default training configuration."""

    return TrainingConfig()


def get_evaluation_config() -> EvaluationConfig:
    """Return the default evaluation configuration."""

    return EvaluationConfig()
