"""End-to-end Double DQN training workflow."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium
import gymnasium.spaces
import numpy as np
import tensorflow as tf

from ..agents.double_dqn import DoubleDQNAgent
from ..config import ExperimentConfig, save_config
from ..environments import make_environment
from ..reporting.plots import plot_training_metrics
from .evaluator import EvaluationSummary, evaluate_model


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    model_path: Path
    episodes_completed: int
    environment_steps: int
    best_validation_score: float
    final_validation: EvaluationSummary


def set_global_seed(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "gymnasium": gymnasium.__version__,
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    }


def _create_run_dir(config: ExperimentConfig, explicit: str | Path | None) -> Path:
    if explicit is not None:
        run_dir = Path(explicit)
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{timestamp}-{config.name}-s{config.training.seed}"
        run_dir = Path(config.training.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train(config: ExperimentConfig, *, run_dir: str | Path | None = None) -> TrainingResult:
    """Train an agent and persist a fully reproducible run."""
    set_global_seed(config.training.seed)
    output = _create_run_dir(config, run_dir)
    save_config(config, output / "config.yaml")
    (output / "runtime.json").write_text(
        json.dumps(_runtime_metadata(), indent=2), encoding="utf-8"
    )

    env = make_environment(config.environment, seed=config.training.seed)
    if not isinstance(env.observation_space, gymnasium.spaces.Box):
        raise TypeError("Double DQN requires a Box observation space")
    if not isinstance(env.action_space, gymnasium.spaces.Discrete):
        raise TypeError("Double DQN requires a Discrete action space")
    state_shape = tuple(int(value) for value in env.observation_space.shape)
    action_count = int(env.action_space.n)
    agent = DoubleDQNAgent(state_shape, action_count, config.agent, config.training.seed)
    writer = tf.summary.create_file_writer(str(output / "tensorboard"))
    metrics_path = output / "metrics.csv"
    model_path = output / "best_model.keras"
    best_score = -float("inf")
    evaluations_without_improvement = 0
    total_steps = 0
    last_evaluation: EvaluationSummary | None = None

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            fields = ["episode", "return", "length", "epsilon", "mean_loss", "validation_return"]
            csv_writer = csv.DictWriter(metrics_file, fieldnames=fields)
            csv_writer.writeheader()
            for episode in range(1, config.training.episodes + 1):
                state, _ = env.reset(seed=config.training.seed + episode)
                terminated = truncated = False
                episode_return = 0.0
                losses: list[float] = []
                episode_length = 0
                while not (terminated or truncated):
                    epsilon = agent.epsilon(total_steps)
                    action = agent.act(state, epsilon)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    agent.remember(state, action, float(reward), next_state, terminated)
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                    state = next_state
                    episode_return += float(reward)
                    episode_length += 1
                    total_steps += 1

                validation_score: float | None = None
                if episode % config.training.evaluation_interval == 0:
                    seeds = tuple(
                        config.training.seed + 100_000 + index
                        for index in range(config.training.evaluation_episodes)
                    )
                    last_evaluation = evaluate_model(
                        agent.online_network, config.environment, seeds
                    )
                    validation_score = last_evaluation.mean_return
                    if last_evaluation.mean_return > best_score:
                        best_score = last_evaluation.mean_return
                        agent.save(model_path)
                        evaluations_without_improvement = 0
                    else:
                        evaluations_without_improvement += 1
                    print(
                        f"[train] episode={episode}/{config.training.episodes} "
                        f"steps={total_steps} return={episode_return:.1f} "
                        f"validation={last_evaluation.mean_return:.1f} "
                        f"best={best_score:.1f}",
                        flush=True,
                    )

                mean_loss = float(np.mean(losses)) if losses else float("nan")
                csv_writer.writerow(
                    {
                        "episode": episode,
                        "return": episode_return,
                        "length": episode_length,
                        "epsilon": agent.epsilon(total_steps),
                        "mean_loss": mean_loss,
                        "validation_return": validation_score,
                    }
                )
                metrics_file.flush()
                with writer.as_default():
                    tf.summary.scalar("episode/return", episode_return, step=episode)
                    tf.summary.scalar("episode/length", episode_length, step=episode)
                    tf.summary.scalar("agent/epsilon", agent.epsilon(total_steps), step=episode)
                    if losses:
                        tf.summary.scalar("agent/loss", mean_loss, step=episode)
                    if validation_score is not None:
                        tf.summary.scalar("evaluation/mean_return", validation_score, step=episode)

                if (
                    best_score >= config.training.checkpoint_min_score
                    and evaluations_without_improvement
                    >= config.training.early_stopping_evaluations
                ):
                    break
            episodes_completed = episode
    finally:
        writer.close()
        env.close()

    if not model_path.exists():
        agent.save(model_path)
    final_model = tf.keras.models.load_model(model_path)
    final_seeds = tuple(
        config.training.seed + 200_000 + index
        for index in range(config.training.evaluation_episodes)
    )
    final_evaluation = evaluate_model(final_model, config.environment, final_seeds)
    summary = {
        "episodes_completed": episodes_completed,
        "environment_steps": total_steps,
        "best_validation_score": best_score,
        "final_validation": final_evaluation.to_dict(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_training_metrics(metrics_path, output / "training.png")
    return TrainingResult(
        run_dir=output,
        model_path=model_path,
        episodes_completed=episodes_completed,
        environment_steps=total_steps,
        best_validation_score=best_score,
        final_validation=final_evaluation,
    )
