"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    id: str = "LunarLander-v3"
    max_episode_steps: int = 1_000


@dataclass(frozen=True, slots=True)
class AgentConfig:
    hidden_layers: tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    target_update_steps: int = 1_000
    gradient_clip_norm: float = 10.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    episodes: int = 1_000
    evaluation_interval: int = 25
    evaluation_episodes: int = 10
    checkpoint_min_score: float = 200.0
    early_stopping_evaluations: int = 5
    output_dir: str = "artifacts/runs"


@dataclass(frozen=True, slots=True)
class TuningConfig:
    trials: int = 12
    seed: int = 2026
    episodes_per_trial: int = 500
    validation_seeds: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str = "lunar-lander-double-dqn"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], values: dict[str, Any], section: str) -> Any:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"Unknown keys in '{section}': {', '.join(sorted(unknown))}")
    if cls in (AgentConfig,) and "hidden_layers" in values:
        values["hidden_layers"] = tuple(values["hidden_layers"])
    if cls in (TuningConfig,) and "validation_seeds" in values:
        values["validation_seeds"] = tuple(values["validation_seeds"])
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("The root YAML value must be a mapping")
    allowed = {"name", "environment", "agent", "training", "tuning"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"Unknown root keys: {', '.join(sorted(unknown))}")
    config = ExperimentConfig(
        name=str(raw.get("name", "lunar-lander-double-dqn")),
        environment=_construct(EnvironmentConfig, raw.get("environment", {}), "environment"),
        agent=_construct(AgentConfig, raw.get("agent", {}), "agent"),
        training=_construct(TrainingConfig, raw.get("training", {}), "training"),
        tuning=_construct(TuningConfig, raw.get("tuning", {}), "tuning"),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """Validate cross-field invariants."""
    agent = config.agent
    training = config.training
    positive = {
        "batch_size": agent.batch_size,
        "replay_capacity": agent.replay_capacity,
        "warmup_steps": agent.warmup_steps,
        "epsilon_decay_steps": agent.epsilon_decay_steps,
        "target_update_steps": agent.target_update_steps,
        "episodes": training.episodes,
        "evaluation_interval": training.evaluation_interval,
        "evaluation_episodes": training.evaluation_episodes,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ConfigError(f"Values must be positive: {', '.join(invalid)}")
    if agent.replay_capacity < agent.batch_size:
        raise ConfigError("replay_capacity must be at least batch_size")
    if not 0 < agent.gamma <= 1:
        raise ConfigError("gamma must be in (0, 1]")
    if not 0 <= agent.epsilon_end <= agent.epsilon_start <= 1:
        raise ConfigError("epsilon values must satisfy 0 <= end <= start <= 1")
    if not agent.hidden_layers or any(width <= 0 for width in agent.hidden_layers):
        raise ConfigError("hidden_layers must contain positive widths")


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
