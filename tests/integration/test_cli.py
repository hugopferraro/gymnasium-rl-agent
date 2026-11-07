import pytest

from gymnasium_rl_agent.main import build_parser, main


def test_train_command_parses() -> None:
    args = build_parser().parse_args(["train", "--config", "example.yaml"])
    assert args.command == "train"
    assert args.config == "example.yaml"


def test_missing_model_returns_usage_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "gymnasium-rl-agent",
            "evaluate",
            "--config",
            "configs/lunar_lander.yaml",
            "--model",
            "missing.keras",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Model not found" in capsys.readouterr().err
