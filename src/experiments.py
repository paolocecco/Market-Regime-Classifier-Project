"""Run chronological model experiments and persist their evaluation results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.features import FEATURE_COLUMNS, make_dataset
from src.models import MODELS

DEFAULT_OUTPUT_PATH = Path("outputs") / "experiments.csv"
REGIME_LABELS = ["low", "medium", "high"]


def _serialise(value: Any) -> str:
    """Represent nested sklearn parameters consistently in the CSV log."""
    return json.dumps(value, sort_keys=True, default=str)


def _labels_from_thresholds(volatility: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    """Assign labels using cutoffs known at the start of a fold."""
    low_cut, high_cut = thresholds
    return pd.cut(
        volatility,
        [-float("inf"), low_cut, high_cut, float("inf")],
        labels=REGIME_LABELS,
    )


def _probability_frame(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Return probabilities with stable class names for portfolio consumers."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not provide predict_proba().")
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    result = pd.DataFrame(0.0, index=features.index, columns=REGIME_LABELS)
    for column, class_name in enumerate(classes):
        result[class_name] = probabilities[:, column]
    return result


def _prediction_rows(
    model: Any,
    model_name: str,
    feature_set_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    fold: int,
) -> pd.DataFrame:
    """Fit one fold and return predictions plus class confidence values."""
    fitted_model = clone(model)
    fitted_model.fit(train[features], train["regime"])
    probabilities = _probability_frame(fitted_model, test[features])
    predictions = fitted_model.predict(test[features])
    result = test[["regime", "future_volatility"]].copy()
    result["prediction"] = predictions
    for label in REGIME_LABELS:
        result[f"probability_{label}"] = probabilities[label]
    result["confidence"] = probabilities.max(axis=1)
    result["confidence_margin"] = probabilities.apply(lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1], axis=1)
    result["model"] = model_name
    result["feature_set"] = feature_set_name
    result["fold"] = fold
    result.index.name = "date"
    return result.reset_index()


def _summary_row(predictions: pd.DataFrame, model: Any, **metadata: Any) -> dict[str, Any]:
    """Summarise predictions while retaining the probability columns in the output."""
    report: Any = classification_report(
        predictions["regime"],
        predictions["prediction"],
        labels=REGIME_LABELS,
        output_dict=True,
        zero_division=0,
    )
    row = {
        **metadata,
        "accuracy": report["accuracy"],
        "balanced_accuracy": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "low_f1": report["low"]["f1-score"],
        "medium_f1": report["medium"]["f1-score"],
        "high_f1": report["high"]["f1-score"],
        "mean_confidence": predictions["confidence"].mean(),
        "mean_confidence_margin": predictions["confidence_margin"].mean(),
        "log_loss": log_loss(
            predictions["regime"],
            predictions[["probability_high", "probability_low", "probability_medium"]],
            labels=["high", "low", "medium"],
        ),
        "parameters": _serialise(model.get_params(deep=True)),
    }
    return row


def evaluate_sliding_windows(
    dataset: pd.DataFrame,
    ticker: str,
    horizon: int,
    train_window: int = 756,
    test_window: int = 63,
    step: int | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    models: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models with rolling train/test windows and fold-local cutoffs.

    Returns a summary table and row-level out-of-sample predictions. Each fold's
    volatility thresholds are computed from its training data only.
    """
    if train_window < 1 or test_window < 1:
        raise ValueError("train_window and test_window must be positive.")
    step = test_window if step is None else step
    if step < 1:
        raise ValueError("step must be positive.")
    if len(dataset) < train_window + test_window:
        raise ValueError("dataset is shorter than train_window + test_window.")

    selected_feature_sets = feature_sets or {"all": FEATURE_COLUMNS}
    selected_models = models or MODELS
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summaries: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold = 0

    for start in range(0, len(dataset) - train_window - test_window + 1, step):
        fold += 1
        train = dataset.iloc[start : start + train_window].copy()
        test = dataset.iloc[start + train_window : start + train_window + test_window].copy()
        thresholds = (
            float(train["future_volatility"].quantile(1 / 3)),
            float(train["future_volatility"].quantile(2 / 3)),
        )
        train["regime"] = _labels_from_thresholds(train["future_volatility"], thresholds)
        test["regime"] = _labels_from_thresholds(test["future_volatility"], thresholds)

        for feature_set_name, features in selected_feature_sets.items():
            missing_features = sorted(set(features) - set(dataset.columns))
            if missing_features:
                raise ValueError(f"Feature set {feature_set_name!r} contains missing columns: {missing_features}")
            for model_name, model in selected_models.items():
                predictions = _prediction_rows(model, model_name, feature_set_name, train, test, features, fold)
                prediction_frames.append(predictions)
                fitted_model = clone(model).fit(train[features], train["regime"])
                summaries.append(_summary_row(
                    predictions,
                    fitted_model,
                    run_id=run_id,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    ticker=ticker,
                    horizon=horizon,
                    phase="sliding",
                    fold=fold,
                    feature_set=feature_set_name,
                    features=_serialise(features),
                    train_start=train.index.min().date().isoformat(),
                    train_end=train.index.max().date().isoformat(),
                    test_start=test.index.min().date().isoformat(),
                    test_end=test.index.max().date().isoformat(),
                ))

    return pd.DataFrame(summaries), pd.concat(prediction_frames, ignore_index=True)


def evaluate_sliding_baselines(
    dataset: pd.DataFrame,
    horizon: int,
    train_window: int = 756,
    test_window: int = 63,
    step: int | None = None,
) -> pd.DataFrame:
    """Evaluate simple non-ML baselines on the same sliding windows.

    ``persistence`` carries the latest observed regime forward. ``majority``
    always predicts the most common training regime. Both use fold-local labels.
    """
    if train_window < 1 or test_window < 1:
        raise ValueError("train_window and test_window must be positive.")
    step = test_window if step is None else step
    rows: list[dict[str, Any]] = []
    for fold, start in enumerate(
        range(0, len(dataset) - train_window - test_window + 1, step),
        start=1,
    ):
        train = dataset.iloc[start : start + train_window]
        test = dataset.iloc[start + train_window : start + train_window + test_window]
        thresholds = (
            float(train["future_volatility"].quantile(1 / 3)),
            float(train["future_volatility"].quantile(2 / 3)),
        )
        train_labels = _labels_from_thresholds(train["future_volatility"], thresholds)
        test_labels = _labels_from_thresholds(test["future_volatility"], thresholds)
        for name, prediction in {
            "PersistenceBaseline": pd.Series(train_labels.iloc[-1], index=test.index),
            "MajorityBaseline": pd.Series(train_labels.mode().iloc[0], index=test.index),
        }.items():
            prediction_frame = pd.DataFrame({"regime": test_labels, "prediction": prediction})
            report: Any = classification_report(
                prediction_frame["regime"],
                prediction_frame["prediction"],
                labels=REGIME_LABELS,
                output_dict=True,
                zero_division=0,
            )
            rows.append({
                "ticker": None,
                "horizon": horizon,
                "phase": "sliding_baseline",
                "fold": fold,
                "model": name,
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
                "phase": "test",
                "fold": None,
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


def evaluate_validation_windows(
    dataset: pd.DataFrame,
    ticker: str,
    horizon: int,
    validation_splits: int = 3,
    feature_sets: dict[str, list[str]] | None = None,
    models: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate models over rolling time-series validation windows."""
    if validation_splits < 2:
        raise ValueError("validation_splits must be at least 2.")

    selected_feature_sets = feature_sets or {"all": FEATURE_COLUMNS}
    selected_models = models or MODELS
    splitter = TimeSeriesSplit(n_splits=validation_splits)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rows: list[dict[str, Any]] = []

    for feature_set_name, features in selected_feature_sets.items():
        for model_name, model in selected_models.items():
            for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(dataset), start=1):
                train = dataset.iloc[train_idx]
                valid = dataset.iloc[valid_idx]
                fitted_model = clone(model)
                fitted_model.fit(train[features], train["regime"])
                predictions = fitted_model.predict(valid[features])
                report: Any = classification_report(
                    valid["regime"],
                    predictions,
                    labels=["low", "medium", "high"],
                    output_dict=True,
                    zero_division=0,
                )
                rows.append({
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "ticker": ticker,
                    "horizon": horizon,
                    "phase": "validation",
                    "fold": fold_index,
                    "feature_set": feature_set_name,
                    "features": _serialise(features),
                    "model": model_name,
                    "parameters": _serialise(fitted_model.get_params(deep=True)),
                    "train_start": train.index.min().date().isoformat(),
                    "train_end": train.index.max().date().isoformat(),
                    "valid_start": valid.index.min().date().isoformat(),
                    "valid_end": valid.index.max().date().isoformat(),
                    "accuracy": report["accuracy"],
                    "balanced_accuracy": report["macro avg"]["recall"],
                    "macro_f1": report["macro avg"]["f1-score"],
                    "weighted_f1": report["weighted avg"]["f1-score"],
                    "low_f1": report["low"]["f1-score"],
                    "medium_f1": report["medium"]["f1-score"],
                    "high_f1": report["high"]["f1-score"],
                })

    return pd.DataFrame(rows)


def main() -> None:
    """Run the configured models and append their results to the experiment log."""
    parser = argparse.ArgumentParser(description="Run chronological model experiments.")
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker, e.g. SPY or ^FTSE")
    parser.add_argument("--start", default="2005-01-01", help="First date in YYYY-MM-DD form")
    parser.add_argument("--horizon", default=10, type=int, help="Future trading days used by the label")
    parser.add_argument("--test-fraction", default=0.20, type=float, help="Fraction of observations reserved for testing")
    parser.add_argument("--validation-splits", type=int, default=0, help="Use rolling time-series validation folds before the final hold-out set")
    parser.add_argument("--sliding", action="store_true", help="Use sliding train/test windows and save probabilities")
    parser.add_argument("--train-window", type=int, default=756, help="Training observations per sliding fold")
    parser.add_argument("--test-window", type=int, default=63, help="Test observations per sliding fold")
    parser.add_argument("--step", type=int, default=None, help="Observations to advance between sliding folds")
    parser.add_argument("--predictions-output", type=Path, default=Path("outputs") / "sliding_predictions.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="CSV file for appended experiment results")
    args = parser.parse_args()

    from src.data import download_prices

    dataset = make_dataset(download_prices(args.ticker, args.start), horizon=args.horizon)
    if args.sliding:
        results, predictions = evaluate_sliding_windows(
            dataset,
            ticker=args.ticker,
            horizon=args.horizon,
            train_window=args.train_window,
            test_window=args.test_window,
            step=args.step,
        )
        baselines = evaluate_sliding_baselines(
            dataset,
            horizon=args.horizon,
            train_window=args.train_window,
            test_window=args.test_window,
            step=args.step,
        )
        results = pd.concat([results, baselines], ignore_index=True, sort=False)
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions_output, index=False)
        print(f"Saved {len(predictions)} out-of-sample predictions to {args.predictions_output}")
    elif args.validation_splits:
        results = evaluate_validation_windows(
            dataset,
            ticker=args.ticker,
            horizon=args.horizon,
            validation_splits=args.validation_splits,
        )
    else:
        results = evaluate_experiments(
            dataset,
            ticker=args.ticker,
            horizon=args.horizon,
            test_fraction=args.test_fraction,
        )
    append_experiment_results(results, args.output)
    print(f"Saved {len(results)} evaluations to {args.output}")
    print(results[["model", "feature_set", "phase", "fold", "accuracy", "balanced_accuracy", "macro_f1", "parameters"]].to_string(index=False))


if __name__ == "__main__":
    main()