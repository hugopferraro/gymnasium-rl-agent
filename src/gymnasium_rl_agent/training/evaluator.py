"""Deterministic policy evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import tensorflow as tf

from ..config import EnvironmentConfig
from ..environments import make_environment


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    episodes: int
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float
    confidence_low: float
    confidence_high: float
    success_rate: float
    returns: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _select_action(model: tf.keras.Model, state: np.ndarray) -> int:
    q_values = model(np.asarray(state, dtype=np.float32)[None, ...], training=False)
    return int(tf.argmax(q_values[0]).numpy())


def evaluate_model(
    model: tf.keras.Model,
    environment: EnvironmentConfig,
    seeds: Sequence[int],
    *,
    video_dir: str | Path | None = None,
) -> EvaluationSummary:
    """Evaluate a greedy policy on explicit episode seeds."""
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    render_mode = "rgb_array" if video_dir is not None else None
    env = make_environment(environment, render_mode=render_mode, video_dir=video_dir)
    returns: list[float] = []
    try:
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = _select_action(model, state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
            returns.append(episode_return)
    finally:
        env.close()

    values = np.asarray(returns, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    margin = NormalDist().inv_cdf(0.975) * std / np.sqrt(len(values))
    return EvaluationSummary(
        episodes=len(returns),
        mean_return=mean,
        median_return=float(np.median(values)),
        std_return=std,
        min_return=float(values.min()),
        max_return=float(values.max()),
        confidence_low=float(mean - margin),
        confidence_high=float(mean + margin),
        success_rate=float(np.mean(values >= 200.0)),
        returns=tuple(returns),
    )


def evaluate_random_policy(
    environment: EnvironmentConfig, seeds: Sequence[int]
) -> EvaluationSummary:
    """Evaluate a seeded random-action baseline."""
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    env = make_environment(environment)
    returns: list[float] = []
    try:
        for seed in seeds:
            _, _ = env.reset(seed=int(seed))
            env.action_space.seed(int(seed))
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                total += float(reward)
            returns.append(total)
    finally:
        env.close()
    values = np.asarray(returns, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    margin = NormalDist().inv_cdf(0.975) * std / np.sqrt(len(values))
    return EvaluationSummary(
        episodes=len(returns),
        mean_return=mean,
        median_return=float(np.median(values)),
        std_return=std,
        min_return=float(values.min()),
        max_return=float(values.max()),
        confidence_low=float(mean - margin),
        confidence_high=float(mean + margin),
        success_rate=float(np.mean(values >= 200.0)),
        returns=tuple(returns),
    )


def load_model(path: str | Path) -> tf.keras.Model:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)
