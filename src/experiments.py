"""Run chronological model experiments and persist their evaluation results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report

from src.features import FEATURE_COLUMNS, make_dataset
from src.models import MODELS

DEFAULT_OUTPUT_PATH = Path("outputs") / "experiments.csv"


def _serialise(value: Any) -> str:
    """Represent nested sklearn parameters consistently in the CSV log."""
    return json.dumps(value, sort_keys=True, default=str)


def evaluate_experiments(
    dataset: pd.DataFrame,
    ticker: str,
    horizon: int,
    test_fraction: float = 0.20,
    feature_sets: dict[str, list[str]] | None = None,
    models: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate model/feature combinations on one chronological hold-out set."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    selected_feature_sets = feature_sets or {"all": FEATURE_COLUMNS}
    selected_models = models or MODELS
    split_index = int(len(dataset) * (1 - test_fraction))
    train, test = dataset.iloc[:split_index], dataset.iloc[split_index:]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rows: list[dict[str, Any]] = []

    for feature_set_name, features in selected_feature_sets.items():
        missing_features = sorted(set(features) - set(dataset.columns))
        if missing_features:
            raise ValueError(f"Feature set {feature_set_name!r} contains missing columns: {missing_features}")

        for model_name, model in selected_models.items():
            fitted_model = clone(model)
            fitted_model.fit(train[features], train["regime"])
            predictions = fitted_model.predict(test[features])
            report: Any = classification_report(
                test["regime"],
                predictions,
                labels=["low", "medium", "high"],
                output_dict=True,
                zero_division=0,
            )
            model_parameters = fitted_model.get_params(deep=True)
            rows.append({
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ticker": ticker,
                "horizon": horizon,
                "test_fraction": test_fraction,
                "feature_set": feature_set_name,
                "features": _serialise(features),
                "model": model_name,
                "parameters": _serialise(model_parameters),
                "train_start": train.index.min().date().isoformat(),
                "train_end": train.index.max().date().isoformat(),
                "test_start": test.index.min().date().isoformat(),
                "test_end": test.index.max().date().isoformat(),
                "accuracy": report["accuracy"],
                "balanced_accuracy": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "low_f1": report["low"]["f1-score"],
                "medium_f1": report["medium"]["f1-score"],
                "high_f1": report["high"]["f1-score"],
            })

    return pd.DataFrame(rows)


def append_experiment_results(results: pd.DataFrame, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Append experiment rows to a CSV, writing headers only for a new file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)


def main() -> None:
    """Run the configured models and append their results to the experiment log."""
    parser = argparse.ArgumentParser(description="Run chronological model experiments.")
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker, e.g. SPY or ^FTSE")
    parser.add_argument("--start", default="2005-01-01", help="First date in YYYY-MM-DD form")
    parser.add_argument("--horizon", default=10, type=int, help="Future trading days used by the label")
    parser.add_argument("--test-fraction", default=0.20, type=float, help="Fraction of observations reserved for testing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="CSV file for appended experiment results")
    args = parser.parse_args()

    from src.data import download_prices

    dataset = make_dataset(download_prices(args.ticker, args.start), horizon=args.horizon)
    results = evaluate_experiments(
        dataset,
        ticker=args.ticker,
        horizon=args.horizon,
        test_fraction=args.test_fraction,
    )
    append_experiment_results(results, args.output)
    print(f"Saved {len(results)} evaluations to {args.output}")
    print(results[["model", "feature_set", "accuracy", "balanced_accuracy", "macro_f1", "parameters"]].to_string(index=False))


if __name__ == "__main__":
    main()