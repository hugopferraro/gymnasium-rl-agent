"""Non-interactive experiment plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_training_metrics(metrics_csv: str | Path, output_path: str | Path) -> None:
    data = pd.read_csv(metrics_csv)
    if data.empty:
        return
    rolling = data["return"].rolling(window=50, min_periods=1).mean()
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(data["episode"], data["return"], alpha=0.25, label="Retorno")
    axes[0].plot(data["episode"], rolling, linewidth=2, label="Média móvel (50)")
    axes[0].set_ylabel("Retorno")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(data["episode"], data["epsilon"], label="Epsilon")
    axes[1].set_xlabel("Episódio")
    axes[1].set_ylabel("Exploração")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
