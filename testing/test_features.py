import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, make_dataset


def test_make_dataset_has_features_and_regimes() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    prices = pd.DataFrame({"Close": np.linspace(100, 140, len(dates)) + np.sin(np.arange(len(dates))), "Volume": np.linspace(1_000, 2_000, len(dates))}, index=dates)
    dataset = make_dataset(prices, horizon=5)
    assert set(FEATURE_COLUMNS).issubset(dataset.columns)
    assert set(dataset["regime"].unique()).issubset({"low", "medium", "high"})