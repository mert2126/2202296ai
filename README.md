# Autonomous Driving Agent with PPO

![Evolution](assets/evolution.gif)

## Overview

This project implements a CPU-friendly autonomous driving agent for `highway-fast-v0` using Gymnasium, `highway-env`, and Stable-Baselines3 PPO. The pipeline is intentionally operational: training logs episode statistics with `Monitor`, saves a midpoint checkpoint at the half-way point of training, persists a final policy, and evaluation produces both an evolution GIF and a reward plot under `assets/`.

## Methodology

The policy is optimized with a compact multi-objective reward that rewards forward progress while strongly penalizing unsafe behavior:

$$
R_t = 0.4\,\hat{v}_t - 1.0\,\mathbb{1}(\text{collision}_t) + 0.2\,\mathbb{1}(\text{lane\_stable}_t) - 0.1\,\lvert \Delta a_t \rvert
$$

Here, $\hat{v}_t$ is normalized speed, $\mathbb{1}(\text{collision}_t)$ is a collision penalty, $\mathbb{1}(\text{lane\_stable}_t)$ rewards steady lane discipline, and $\lvert \Delta a_t \rvert$ discourages unnecessary action switching. The weighting is intentionally safety-biased so the agent learns to prefer stable traffic handling over aggressive lane changes.

PPO is a strong fit for this environment because the control problem is discrete, the state is structured, and the hardware target is CPU-only. The clipped policy update improves stability, the rollout-based training loop keeps memory usage predictable, and the method is reliable under the stochastic traffic generation used by `highway-env`.

The key hyperparameters are chosen for stable learning rather than raw aggressiveness:

- `learning_rate = 5e-4` keeps updates controlled on CPU.
- `gamma = 0.8` prioritizes near-term safety and reduces reward-horizon drift.
- `n_steps = 1024` provides a useful rollout batch without excessive memory pressure.
- `batch_size = 64` maintains efficient minibatch updates.

## Observation And Action Spaces

The environment uses a kinematic observation representation instead of raw pixels. In practice, the policy receives a compact numerical encoding of nearby traffic, including relative positions, velocities, and lane context around the ego vehicle. That structure is a good match for PPO because it is easy to process, lightweight on CPU, and expressive enough for lane selection and collision avoidance.

The action space is discrete and corresponds to the core highway-driving decisions: lane changes and speed control. This keeps the learning problem tractable while still forcing the agent to manage the essential trade-off between speed, spacing, and safety.

## Training Analysis

![Reward Plot](assets/reward_plot.png)

The reward curve should be interpreted as a stability trace rather than a simple upward line. Early episodes usually show high variance because the agent is still learning the traffic dynamics. As PPO updates accumulate, the moving average should smooth out and the spread between episodes should narrow, which is the practical sign that the policy is becoming safer and more consistent.

A healthy training run often shows this pattern: large oscillations at the start, reduced crash frequency in the middle, and longer, more stable episodes later on. In a driving task, lower variance is often more important than isolated reward spikes because stable trajectories usually indicate fewer abrupt lane changes and fewer terminal collisions.

## Challenges and Failures

The main early failure mode was cascading crashes. After a single unsafe maneuver, the policy often continued to swerve or brake too aggressively, compounding the mistake into a sequence of collisions. The most effective fix was to reduce the discount factor to `0.8`, which made PPO pay more attention to immediate safety consequences instead of overvaluing delayed reward.

That change improved the driving style in a practical way. The agent became less willing to gamble on risky lane switches, recovered more cleanly after near-misses, and learned to treat the next few seconds as the most important part of the trajectory.

Evaluation also uses deterministic inference for the learned policies. That removes sampling noise when comparing the half-trained and final checkpoints, which makes the evolution GIF easier to interpret.

## Outputs

- `checkpoints/ppo_highway_half_trained.zip`
- `checkpoints/ppo_highway_final.zip`
- `logs/monitor.csv`
- `assets/evolution.gif`
- `assets/reward_plot.png`

## Run Instructions

```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```

The training script writes the monitor CSV and both checkpoints. The evaluation script reads those artifacts and generates the GIF and reward plot automatically.

## Project Layout

```text
.
├── assets/
├── checkpoints/
├── logs/
├── src/
│   ├── config.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```