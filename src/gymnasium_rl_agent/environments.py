"""Gymnasium environment construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym

from .config import EnvironmentConfig


def make_environment(
    config: EnvironmentConfig,
    *,
    render_mode: str | None = None,
    seed: int | None = None,
    video_dir: str | Path | None = None,
) -> gym.Env[Any, Any]:
    """Create, optionally seed, and optionally record an environment."""
    env = gym.make(config.id, render_mode=render_mode, max_episode_steps=config.max_episode_steps)
    if video_dir is not None:
        target = Path(video_dir)
        target.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(target),
            episode_trigger=lambda _: True,
            name_prefix="lunar-lander",
        )
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def environment_factory(config: EnvironmentConfig) -> Callable[..., gym.Env[Any, Any]]:
    return lambda **kwargs: make_environment(config, **kwargs)
