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

For rolling out-of-sample predictions, use:

```bash
python -m src.experiments --sliding --train-window 756 --test-window 63 --step 63
```

This trains each model on a moving 756-observation window and tests it on the
following 63 observations. It writes summary metrics to
`outputs/experiments.csv` and row-level predictions to
`outputs/sliding_predictions.csv`. Each prediction includes
`probability_low`, `probability_medium`, `probability_high`, `confidence`, and
`confidence_margin`. The regime cutoffs are recomputed from each fold's
training observations only, avoiding future-information leakage. Persistence
and majority-class baselines are included in the summary.

The class probabilities are model-estimated probabilities, not automatically
Bayesian posterior probabilities. Logistic regression estimates class
probabilities from a likelihood-based discriminative model; tree models
estimate them from class frequencies in their leaves. `confidence` is the
largest class probability, while `confidence_margin` is the difference
between the largest and second-largest probabilities. These values should be
calibrated and evaluated out of sample before being used to change portfolio
exposure.

## Important limitation

This is an educational risk-regime classifier, not trading advice. Financial relationships change over time; a good test score does not establish a profitable strategy.
