"""Visualise target labels and engineered features.

Run from the project root, for example:
    python -m src.visualise --ticker SPY --start 2015-01-01
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.features import make_dataset
from src.features import FEATURE_COLUMNS
from src.models import MODELS

REGIME_COLORS = {"low": "#2a9d8f", "medium": "#e9c46a", "high": "#e76f51"}
REGIMES = list(REGIME_COLORS)
TEST_FRACTION = 0.20


def make_evaluation_figure(dataset: pd.DataFrame, ticker: str, horizon: int) -> Figure:
    """Compare models on a chronological hold-out set in one figure."""
    split_index = int(len(dataset) * (1 - TEST_FRACTION))
    train, test = dataset.iloc[:split_index], dataset.iloc[split_index:]
    actual = test["regime"].astype(str)
    predictions: dict[str, pd.Series] = {}
    scores: dict[str, tuple[float, float]] = {}

    for model_name, model in MODELS.items():
        fitted_model = clone(model)
        fitted_model.fit(train[FEATURE_COLUMNS], train["regime"])
        predicted = pd.Series(fitted_model.predict(test[FEATURE_COLUMNS]), index=test.index).astype(str)
        predictions[model_name] = predicted
        scores[model_name] = (
            float(balanced_accuracy_score(actual, predicted)),
            float(f1_score(actual, predicted, average="macro")),
        )

    model_names = list(predictions)
    figure = plt.figure(figsize=(16, 17), layout="constrained")
    timeline_rows = 1 + len(model_names)
    grid = figure.add_gridspec(
        2 + timeline_rows,
        max(3, len(model_names)),
        height_ratios=[1.1, 2.1] + [1.3] * timeline_rows,
    )

    metrics_axis = figure.add_subplot(grid[0, :])
    y_positions = np.arange(len(model_names))
    bar_height = 0.35
    metrics_axis.barh(y_positions - bar_height / 2, [scores[name][0] for name in model_names], height=bar_height, label="Balanced accuracy", color="#264653")
    metrics_axis.barh(y_positions + bar_height / 2, [scores[name][1] for name in model_names], height=bar_height, label="Macro F1", color="#e76f51")
    metrics_axis.set_yticks(y_positions, model_names)
    metrics_axis.set_xlim(0, 1)
    metrics_axis.set_xlabel("Score")
    metrics_axis.set_title(f"{ticker}: model performance on chronological test set")
    metrics_axis.legend(loc="lower right")
    metrics_axis.grid(axis="x", alpha=0.25)

    confusion_axes = [figure.add_subplot(grid[1, column]) for column in range(len(model_names))]
    for axis, model_name in zip(confusion_axes, model_names):
        matrix = confusion_matrix(actual, predictions[model_name], labels=REGIMES)
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row in range(len(REGIMES)):
            for column in range(len(REGIMES)):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        axis.set_title(model_name)
        axis.set_xticks(range(len(REGIMES)), [regime.title() for regime in REGIMES])
        axis.set_yticks(range(len(REGIMES)), [regime.title() for regime in REGIMES])
        axis.set_xlabel("Predicted", fontweight="bold")
        axis.set_ylabel("Actual", fontweight="bold")
    figure.text(0.05, 0.65, "Confusion matrices", ha="center", va="bottom", fontsize=14, fontweight="bold", rotation=90)

    timeline_names = ["Actual"] + model_names
    timeline_values = {"Actual": actual, **predictions}
    timeline_axes = []
    for row, name in enumerate(timeline_names, start=2):
        timeline_axis = figure.add_subplot(
            grid[row, :],
            sharex=timeline_axes[0] if timeline_axes else None,
        )
        timeline_axes.append(timeline_axis)
        timeline_axis.plot(test.index, test["Close"], color="#264653", linewidth=1, alpha=0.75)
        values = timeline_values[name]
        for regime, color in REGIME_COLORS.items():
            regime_rows = test[values == regime]
            timeline_axis.scatter(
                regime_rows.index,
                regime_rows["Close"],
                s=12,
                color=color,
                label=regime.title(),
                zorder=2,
            )
        timeline_axis.set_ylabel(name)
        timeline_axis.grid(axis="x", alpha=0.25)
        timeline_axis.legend(loc="upper left", ncols=3, fontsize="small")
    timeline_axes[0].set_title(
        f"Index price and regimes across the test period "
        f"({test.index.min().date()} to {test.index.max().date()})"
    )
    timeline_axes[-1].set_xlabel("Date")
    figure.suptitle(f"{ticker} volatility-regime model evaluation ({horizon}-day horizon)", fontsize=16)
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
    figure = make_evaluation_figure(dataset, args.ticker, args.horizon)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path("outputs") / f"model_evaluation_{args.ticker.replace('^', '')}_{timestamp}.png"
    output_path.parent.mkdir(exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot to {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()