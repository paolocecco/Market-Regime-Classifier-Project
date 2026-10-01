"""Visualise target labels and engineered features.

Run from the project root, for example:
    python -m src.visualise --ticker SPY --start 2015-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.features import make_dataset

REGIME_COLORS = {"low": "#2a9d8f", "medium": "#e9c46a", "high": "#e76f51"}


def make_label_figure(dataset: pd.DataFrame, ticker: str, horizon: int) -> plt.Figure:
    """Create a figure that makes target-label construction inspectable."""
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, layout="constrained")

    axes[0].plot(dataset.index, dataset["Close"], color="#264653", linewidth=1, label="Close")
    for regime, color in REGIME_COLORS.items():
        rows = dataset[dataset["regime"].astype(str) == regime]
        axes[0].scatter(rows.index, rows["Close"], s=10, color=color, label=regime.title(), zorder=2)
    axes[0].set_title(f"{ticker}: price coloured by future {horizon}-day volatility regime")
    axes[0].set_ylabel("Adjusted close")
    axes[0].legend(title="Label", ncols=4)

    axes[1].plot(dataset.index, dataset["future_volatility"], color="#457b9d", linewidth=1)
    low_cut, high_cut = dataset["future_volatility"].quantile([1 / 3, 2 / 3])
    axes[1].axhline(low_cut, color=REGIME_COLORS["low"], linestyle="--", label="Low / medium cutoff")
    axes[1].axhline(high_cut, color=REGIME_COLORS["high"], linestyle="--", label="Medium / high cutoff")
    axes[1].set_title("Target: realised volatility in the following trading days")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()

    axes[2].plot(dataset.index, dataset["volatility_20d"], label="20-day trailing volatility", color="#6a4c93")
    axes[2].plot(dataset.index, dataset["momentum_20d"], label="20-day momentum", color="#f4a261", alpha=0.8)
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_title("Two example features calculated from information available on each date")
    axes[2].set_ylabel("Feature value")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    return figure


def main() -> None:
    from src.data import download_prices

    parser = argparse.ArgumentParser(description="Plot market-regime labels and starter features.")
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker, e.g. SPY or ^FTSE")
    parser.add_argument("--start", default="2015-01-01", help="First date in YYYY-MM-DD form")
    parser.add_argument("--horizon", default=10, type=int, help="Future trading days used by the label")
    parser.add_argument("--show", action="store_true", help="Open the figure after saving it")
    args = parser.parse_args()

    dataset = make_dataset(download_prices(args.ticker, args.start), horizon=args.horizon)
    figure = make_label_figure(dataset, args.ticker, args.horizon)
    output_path = Path("outputs") / f"label_exploration_{args.ticker.replace('^', '')}.png"
    output_path.parent.mkdir(exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot to {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()