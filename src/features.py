"""Feature and target construction."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["return_1d", "return_5d", "volatility_20d", "momentum_20d", "distance_from_ma_50d", "volume_ratio_20d"]


def make_dataset(prices: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Build a labelled feature set for future realised-volatility regimes."""
    frame = prices.copy()
    close = frame["Close"]
    daily_returns = close.pct_change()
    frame["return_1d"] = daily_returns
    frame["return_5d"] = close.pct_change(5)
    frame["volatility_20d"] = daily_returns.rolling(20).std()
    frame["momentum_20d"] = close.pct_change(20)
    ma_50 = close.rolling(50).mean()
    frame["distance_from_ma_50d"] = close / ma_50 - 1
    frame["volume_ratio_20d"] = frame["Volume"] / frame["Volume"].rolling(20).mean()

    # This is the prediction target; all feature columns above use only past data.
    future_volatility = daily_returns.rolling(horizon).std().shift(-horizon)
    low_cut, high_cut = future_volatility.quantile([1 / 3, 2 / 3])
    frame["regime"] = pd.cut(future_volatility, [-np.inf, low_cut, high_cut, np.inf], labels=["low", "medium", "high"])
    return frame.dropna(subset=FEATURE_COLUMNS + ["regime"])