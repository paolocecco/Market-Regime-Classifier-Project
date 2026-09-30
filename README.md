# Market Regime Classifier

Classify future market volatility as `low`, `medium`, or `high` using only information available at the prediction date.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

The first run downloads S&P 500 ETF (`SPY`) data, creates features, trains a logistic-regression baseline, and prints a chronological hold-out evaluation.

## Project layout

- `src/data.py` downloads price data.
- `src/features.py` creates features and forward-looking labels.
- `src/train.py` trains and evaluates the first model.
- `notebooks/` is for exploratory work; move stable logic into `src/`.
- `tests/` is for checks as the project grows.

## First experiments

Change `TICKER`, `START_DATE`, feature windows, or the label horizon in `src/train.py`. Keep the time-ordered train/test split: do not randomly shuffle financial time series.

## Important limitation

This is an educational risk-regime classifier, not trading advice. Financial relationships change over time; a good test score does not establish a profitable strategy.
