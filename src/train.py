"""Training entry point for the PPO autonomous driving agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import highway_env  # noqa: F401  # Registers highway-env environments.
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.config import (
    CHECKPOINT_DIR,
    ENV_ID,
    FINAL_MODEL_NAME,
    HALF_TRAINED_MODEL_NAME,
    LOG_DIR,
    MONITOR_FILE_NAME,
    TOTAL_TIMESTEPS,
    TrainingConfig,
    get_ppo_hyperparameters,
    get_training_config,
)


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Resolved paths produced by a training run."""

    monitor_path: Path
    checkpoint_dir: Path
    final_model_path: Path
    midpoint_model_path: Path


class MidpointCheckpointCallback(CheckpointCallback):
    """Checkpoint callback that saves the half-trained model once."""

    def __init__(
        self,
        midpoint_step: int,
        midpoint_path: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            save_freq=max(1, midpoint_step),
            save_path=str(midpoint_path.parent),
            name_prefix=midpoint_path.stem,
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=verbose,
        )
        self._midpoint_step = max(1, midpoint_step)
        self._midpoint_path = midpoint_path
        self._saved = False

    def _on_step(self) -> bool:
        """Persist the midpoint checkpoint the first time the threshold is hit."""

        if not self._saved and self.num_timesteps >= self._midpoint_step:
            self._midpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self._midpoint_path))
            self._saved = True
            if self.verbose > 0:
                print(
                    "[Checkpoint] Saved midpoint model at "
                    f"{self.num_timesteps} timesteps to {self._midpoint_path}"
                )

        return True


def build_training_environment(
    env_id: str,
    seed: int,
    monitor_path: Path,
) -> gym.Env:
    """Create the monitored training environment."""

    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return Monitor(env, filename=str(monitor_path))


def build_model(env: gym.Env, hyperparameters: Dict[str, Any]) -> PPO:
    """Instantiate PPO on CPU with the configured hyperparameters."""

    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        seed=42,
        **hyperparameters,
    )


def train_model() -> TrainingArtifacts:
    """Run the full training job and persist the expected artifacts."""

    training_config: TrainingConfig = get_training_config()
    hyperparameters = get_ppo_hyperparameters()

    log_dir = Path(training_config.log_dir)
    checkpoint_dir = Path(training_config.checkpoint_dir)
    monitor_path = log_dir / MONITOR_FILE_NAME
    midpoint_model_path = checkpoint_dir / HALF_TRAINED_MODEL_NAME
    final_model_path = checkpoint_dir / FINAL_MODEL_NAME

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = build_training_environment(
        env_id=ENV_ID,
        seed=42,
        monitor_path=monitor_path,
    )

    model = build_model(env=env, hyperparameters=hyperparameters)
    midpoint_callback = MidpointCheckpointCallback(
        midpoint_step=max(1, TOTAL_TIMESTEPS // 2),
        midpoint_path=midpoint_model_path,
        verbose=1,
    )

    print(
        "[Train] Starting PPO training for "
        f"{training_config.total_timesteps} timesteps on CPU."
    )
    try:
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=midpoint_callback,
            progress_bar=True,
        )
    finally:
        env.close()

    model.save(str(final_model_path))

    if not midpoint_model_path.exists():
        raise RuntimeError(
            "Midpoint checkpoint was not created. Check the callback logic."
        )

    print(f"[Train] Saved final model to {final_model_path}")
    print(f"[Train] Saved midpoint model to {midpoint_model_path}")
    print(f"[Train] Monitor log written to {monitor_path}")

    return TrainingArtifacts(
        monitor_path=monitor_path,
        checkpoint_dir=checkpoint_dir,
        final_model_path=final_model_path,
        midpoint_model_path=midpoint_model_path,
    )


def main() -> None:
    """CLI entry point for training."""

    train_model()


if __name__ == "__main__":
    main()
