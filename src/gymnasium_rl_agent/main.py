"""Command-line interface for all project workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente Double DQN para LunarLander-v3")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train", help="Treinar um agente")
    train_parser.add_argument("--config", default="configs/lunar_lander.yaml")
    train_parser.add_argument("--output-dir")

    tune_parser = subcommands.add_parser("tune", help="Executar random search")
    tune_parser.add_argument("--config", default="configs/tuning.yaml")
    tune_parser.add_argument("--output-dir")

    evaluate_parser = subcommands.add_parser("evaluate", help="Avaliar um modelo salvo")
    evaluate_parser.add_argument("--config", default="configs/lunar_lander.yaml")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--episodes", type=int, default=100)
    evaluate_parser.add_argument("--seed", type=int, default=300_000)
    evaluate_parser.add_argument("--output")
    evaluate_parser.add_argument("--include-random-baseline", action="store_true")

    demo_parser = subcommands.add_parser("demo", help="Executar e opcionalmente gravar o agente")
    demo_parser.add_argument("--config", default="configs/lunar_lander.yaml")
    demo_parser.add_argument("--model", required=True)
    demo_parser.add_argument("--episodes", type=int, default=3)
    demo_parser.add_argument("--seed", type=int, default=400_000)
    demo_parser.add_argument("--record", action="store_true")
    demo_parser.add_argument("--video-dir", default="artifacts/final/videos")
    return parser


def _print_json(data: object, output: str | None = None) -> None:
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    print(rendered)
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.command == "train":
        from .training.trainer import train

        result = train(config, run_dir=args.output_dir)
        _print_json(
            {
                "run_dir": str(result.run_dir),
                "model": str(result.model_path),
                "best_validation_score": result.best_validation_score,
            }
        )
    elif args.command == "tune":
        from .training.tuner import tune

        print(tune(config, output_dir=args.output_dir))
    elif args.command in {"evaluate", "demo"}:
        from .training.evaluator import evaluate_model, evaluate_random_policy, load_model

        if args.episodes <= 0:
            raise ValueError("episodes must be positive")
        seeds = tuple(args.seed + index for index in range(args.episodes))
        video_dir = args.video_dir if args.command == "demo" and args.record else None
        summary = evaluate_model(
            load_model(args.model), config.environment, seeds, video_dir=video_dir
        )
        payload: dict[str, object] = {"agent": summary.to_dict()}
        if args.command == "evaluate" and args.include_random_baseline:
            payload["random_baseline"] = evaluate_random_policy(
                config.environment, seeds
            ).to_dict()
        _print_json(payload, getattr(args, "output", None))
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (ConfigError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
