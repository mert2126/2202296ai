"""Evaluation and reporting utilities for the PPO driving agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from stable_baselines3 import PPO

import highway_env  # noqa: F401  # Registers highway-env environments.

from src.config import (
    ASSETS_DIR,
    CHECKPOINT_DIR,
    ENV_ID,
    EVOLUTION_GIF_NAME,
    EvaluationConfig,
    FINAL_MODEL_NAME,
    HALF_TRAINED_MODEL_NAME,
    LOG_DIR,
    MONITOR_FILE_NAME,
    REWARD_PLOT_NAME,
    get_evaluation_config,
)


@dataclass(frozen=True, slots=True)
class StageSpec:
    """A single rollout stage used in the evolution GIF."""

    name: str
    checkpoint_path: Optional[Path]
    deterministic: bool


def create_eval_environment(env_id: str) -> gym.Env:
    """Create a rendering-enabled evaluation environment."""

    return gym.make(env_id, render_mode="rgb_array")


def render_title_frame(
    width: int,
    height: int,
    title: str,
    subtitle: str,
) -> np.ndarray:
    """Build a title card that matches the environment frame size."""

    dpi = 100
    fig: Figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("#0f172a")
    fig.set_facecolor("#0f172a")
    ax.text(
        0.5,
        0.60,
        title,
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
        color="white",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.40,
        subtitle,
        ha="center",
        va="center",
        fontsize=14,
        color="#cbd5e1",
        transform=ax.transAxes,
    )
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def collect_episode_frames(
    env: gym.Env,
    model: Optional[PPO],
    deterministic: bool,
    max_steps: int,
) -> Tuple[List[np.ndarray], float, int]:
    """Collect a single episode and return the rendered frames."""

    frames: List[np.ndarray] = []
    observation, _ = env.reset()
    total_reward = 0.0
    episode_length = 0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(observation, deterministic=deterministic)

        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        episode_length += 1

        if terminated or truncated:
            final_frame = env.render()
            if final_frame is not None:
                frames.append(np.asarray(final_frame))
            break

    return frames, total_reward, episode_length


def load_model(path: Path, env: gym.Env) -> PPO:
    """Load a trained PPO model with a helpful error message."""

    if not path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found: {path}. Run training first."
        )
    return PPO.load(path, env=env, device="cpu")


def generate_stage_frames(
    env: gym.Env,
    stage: StageSpec,
    max_steps: int,
    episodes: int,
) -> List[np.ndarray]:
    """Roll out one stage and collect title cards plus episode frames."""

    stage_frames: List[np.ndarray] = []
    model: Optional[PPO] = None

    if stage.checkpoint_path is not None:
        model = load_model(stage.checkpoint_path, env)

    env.reset()
    sample_frame = env.render()
    if sample_frame is None:
        raise RuntimeError("Evaluation environment did not return render frames.")

    height, width = sample_frame.shape[:2]
    stage_frames.append(
        render_title_frame(
            width=width,
            height=height,
            title=stage.name,
            subtitle=(
                "Random policy"
                if model is None
                else (
                    "Deterministic PPO rollout"
                    if stage.deterministic
                    else "Stochastic PPO rollout"
                )
            ),
        )
    )

    for episode_index in range(episodes):
        frames, reward, length = collect_episode_frames(
            env=env,
            model=model,
            deterministic=stage.deterministic,
            max_steps=max_steps,
        )
        if not frames:
            continue

        stage_frames.extend(frames)
        stage_frames.append(
            render_title_frame(
                width=width,
                height=height,
                title=f"{stage.name} | Episode {episode_index + 1}",
                subtitle=f"Reward: {reward:.2f} | Length: {length}",
            )
        )

    return stage_frames


def generate_evolution_video(
    checkpoint_dir: Path,
    env_id: str,
    output_path: Path,
    episodes_per_stage: int,
    max_episode_steps: int,
    fps: int,
) -> bool:
    """Generate the stage-by-stage evolution GIF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = create_eval_environment(env_id=env_id)

    stages: Sequence[StageSpec] = (
        StageSpec(name="Untrained", checkpoint_path=None, deterministic=False),
        StageSpec(
            name="Half-Trained",
            checkpoint_path=checkpoint_dir / HALF_TRAINED_MODEL_NAME,
            deterministic=True,
        ),
        StageSpec(
            name="Fully-Trained",
            checkpoint_path=checkpoint_dir / FINAL_MODEL_NAME,
            deterministic=True,
        ),
    )

    frames: List[np.ndarray] = []
    try:
        for stage in stages:
            print(f"[Eval] Generating stage: {stage.name}")
            frames.extend(
                generate_stage_frames(
                    env=env,
                    stage=stage,
                    max_steps=max_episode_steps,
                    episodes=episodes_per_stage,
                )
            )
    finally:
        env.close()

    if not frames:
        return False

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"[Eval] Saved evolution GIF to {output_path}")
    return True


def read_monitor_dataframe(log_dir: Path) -> pd.DataFrame:
    """Load the training monitor CSV from the logs directory."""

    preferred = log_dir / MONITOR_FILE_NAME
    if preferred.exists():
        monitor_path = preferred
    else:
        monitor_files = sorted(log_dir.glob("*.csv"))
        if not monitor_files:
            raise FileNotFoundError(
                f"No CSV monitor files found in {log_dir}. Run training first."
            )
        monitor_path = monitor_files[-1]

    df = pd.read_csv(monitor_path, comment="#")
    if df.empty:
        raise ValueError(f"Monitor file is empty: {monitor_path}")
    if "r" not in df.columns:
        raise ValueError(
            f"Monitor file does not contain rewards column: {monitor_path}"
        )
    return df


def plot_reward_history(log_dir: Path, output_path: Path) -> bool:
    """Plot reward versus episode index from the monitor logs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = read_monitor_dataframe(log_dir)

    episodes = np.arange(1, len(df) + 1)
    rewards = df["r"].astype(float).to_numpy()
    lengths = df["l"].astype(float).to_numpy() if "l" in df.columns else None
    moving_average = pd.Series(rewards).rolling(window=10, min_periods=1).mean()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        episodes,
        rewards,
        color="#2563eb",
        alpha=0.35,
        linewidth=1.2,
        label="Episode reward",
    )
    ax.plot(
        episodes,
        moving_average,
        color="#dc2626",
        linewidth=2.2,
        label="10-episode moving average",
    )
    ax.set_title("Reward vs. Episodes", fontsize=16, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    if lengths is not None:
        ax2 = ax.twinx()
        ax2.plot(
            episodes,
            lengths,
            color="#16a34a",
            alpha=0.18,
            linewidth=1.0,
            label="Episode length",
        )
        ax2.set_ylabel("Episode length", fontsize=12)
    ax.legend(frameon=True, loc="upper left")
    ax.text(
        0.01,
        0.02,
        f"Episodes: {len(df)} | Mean reward: {rewards.mean():.2f}",
        transform=ax.transAxes,
        fontsize=10,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] Saved reward plot to {output_path}")
    return True


def main() -> None:
    """Generate the GIF and reward plot artifacts."""

    eval_config: EvaluationConfig = get_evaluation_config()
    checkpoint_dir = Path(CHECKPOINT_DIR)
    log_dir = Path(LOG_DIR)
    assets_dir = Path(ASSETS_DIR)

    evolution_path = assets_dir / EVOLUTION_GIF_NAME
    reward_plot_path = assets_dir / REWARD_PLOT_NAME

    video_ok = generate_evolution_video(
        checkpoint_dir=checkpoint_dir,
        env_id=ENV_ID,
        output_path=evolution_path,
        episodes_per_stage=eval_config.episodes_per_stage,
        max_episode_steps=eval_config.max_episode_steps,
        fps=eval_config.video_fps,
    )
    plot_ok = plot_reward_history(log_dir=log_dir, output_path=reward_plot_path)

    if not video_ok or not plot_ok:
        raise RuntimeError("Evaluation did not complete successfully.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
