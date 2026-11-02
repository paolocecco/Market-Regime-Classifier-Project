"""Feature and target construction."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["return_1d", "return_5d", "volatility_20d", "momentum_20d", "distance_from_ma_50d", "volume_ratio_20d"]


def compute_regime_thresholds(future_volatility: pd.Series) -> tuple[float, float]:
    """Compute the low/medium/high boundaries from a training-period volatility series."""
    low_cut, high_cut = future_volatility.quantile([1 / 3, 2 / 3])
    return float(low_cut), float(high_cut)


def make_dataset(prices: pd.DataFrame, horizon: int = 10, thresholds: tuple[float, float] | None = None) -> pd.DataFrame:
    """Build a labelled feature set for future realised-volatility regimes.

    If thresholds are provided, they are used directly so the same cutoffs can be
    applied consistently across a validation or test split without leaking future
    information from the full dataset.
    """
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

    future_volatility = daily_returns.rolling(horizon).std().shift(-horizon)
    if thresholds is None:
        low_cut, high_cut = compute_regime_thresholds(future_volatility)
    else:
        low_cut, high_cut = thresholds
    frame["future_volatility"] = future_volatility
    frame["regime"] = pd.cut(future_volatility, [-np.inf, low_cut, high_cut, np.inf], labels=["low", "medium", "high"])
    return frame.dropna(subset=FEATURE_COLUMNS + ["regime"])