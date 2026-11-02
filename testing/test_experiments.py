import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments import evaluate_sliding_baselines, evaluate_sliding_windows


def _dataset(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    values = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "feature": np.sin(values / 5),
            "future_volatility": (values % 9) / 10 + 0.01,
            "regime": pd.cut(
                (values % 9) / 10 + 0.01,
                [-np.inf, 0.3, 0.6, np.inf],
                labels=["low", "medium", "high"],
            ),
        },
        index=dates,
    )
    return frame


def test_sliding_windows_return_probabilities_and_confidence() -> None:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    summaries, predictions = evaluate_sliding_windows(
        _dataset(),
        ticker="TEST",
        horizon=5,
        train_window=30,
        test_window=10,
        feature_sets={"test": ["feature"]},
        models={"LogisticRegression": model},
    )

    assert len(summaries) == 5
    assert len(predictions) == 50
    probability_columns = ["probability_low", "probability_medium", "probability_high"]
    assert np.allclose(predictions[probability_columns].sum(axis=1), 1.0)
    assert predictions["confidence"].between(0.0, 1.0).all()
    assert predictions["confidence_margin"].between(0.0, 1.0).all()
    assert "log_loss" in summaries


def test_sliding_baselines_use_same_folds() -> None:
    results = evaluate_sliding_baselines(_dataset(), horizon=5, train_window=30, test_window=10)

    assert set(results["model"]) == {"PersistenceBaseline", "MajorityBaseline"}
    assert results["fold"].nunique() == 5
