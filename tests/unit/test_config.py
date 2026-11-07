from pathlib import Path

import pytest

from gymnasium_rl_agent.config import ConfigError, load_config


def test_loads_project_configuration() -> None:
    config = load_config(Path("configs/lunar_lander.yaml"))
    assert config.environment.id == "LunarLander-v3"
    assert config.agent.hidden_layers == (128, 128)
    assert config.training.seed == 42


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("agent:\n  mystery: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(path)


def test_rejects_invalid_epsilon(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("agent:\n  epsilon_start: 0.1\n  epsilon_end: 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="epsilon"):
        load_config(path)
