"""Reproducible random-search orchestration."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..config import AgentConfig, ExperimentConfig, save_config
from .evaluator import evaluate_model
from .trainer import train


def _sample_config(
    base: ExperimentConfig, rng: np.random.Generator, trial: int
) -> ExperimentConfig:
    hidden = ((128, 128), (256, 128), (256, 256))
    agent = replace(
        base.agent,
        hidden_layers=hidden[int(rng.integers(len(hidden)))],
        learning_rate=float(10 ** rng.uniform(-4, -3)),
        batch_size=int(rng.choice([64, 128])),
        gamma=float(rng.choice([0.99, 0.995])),
        target_update_steps=int(rng.choice([500, 1_000, 2_000])),
        epsilon_decay_steps=int(rng.choice([50_000, 100_000, 150_000])),
    )
    training = replace(
        base.training,
        seed=base.tuning.seed + trial,
        episodes=base.tuning.episodes_per_trial,
        early_stopping_evaluations=3,
    )
    return replace(base, name=f"{base.name}-trial-{trial:02d}", agent=agent, training=training)


def tune(config: ExperimentConfig, *, output_dir: str | Path | None = None) -> Path:
    """Run random search and return the tuning directory."""
    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = Path(config.training.output_dir) / f"{timestamp}-{config.name}-tuning"
    else:
        root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.tuning.seed)
    rows: list[dict[str, object]] = []
    best_score = -float("inf")
    best_trial = -1
    best_agent: AgentConfig | None = None

    for trial in range(config.tuning.trials):
        trial_config = _sample_config(config, rng, trial)
        result = train(trial_config, run_dir=root / f"trial-{trial:02d}")
        model = tf.keras.models.load_model(result.model_path)
        evaluation = evaluate_model(
            model, config.environment, config.tuning.validation_seeds
        )
        row = {
            "trial": trial,
            "mean_return": evaluation.mean_return,
            "learning_rate": trial_config.agent.learning_rate,
            "hidden_layers": "x".join(map(str, trial_config.agent.hidden_layers)),
            "batch_size": trial_config.agent.batch_size,
            "gamma": trial_config.agent.gamma,
            "target_update_steps": trial_config.agent.target_update_steps,
            "epsilon_decay_steps": trial_config.agent.epsilon_decay_steps,
            "model_path": str(result.model_path),
        }
        rows.append(row)
        if evaluation.mean_return > best_score:
            best_score = evaluation.mean_return
            best_trial = trial
            best_agent = trial_config.agent

    if best_agent is None:  # Protected by configuration validation in normal CLI usage.
        raise ValueError("Tuning requires at least one trial")

    results_path = root / "tuning_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row["mean_return"]), reverse=True))
    (root / "best_trial.json").write_text(
        json.dumps({"trial": best_trial, "mean_return": best_score}, indent=2), encoding="utf-8"
    )
    # Restore the full final-training budget while retaining only the winning agent parameters.
    final_config = replace(config, agent=best_agent)
    save_config(final_config, root / "best_config.yaml")
    return root
