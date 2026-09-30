"""Market-data acquisition."""

import pandas as pd
import yfinance as yf


def download_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily OHLCV data and return a flat, non-empty frame."""
    prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if prices is None or prices.empty:
        raise ValueError(f"No price data returned for {ticker!r}.")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    return prices.sort_index().dropna(subset=["Close"])