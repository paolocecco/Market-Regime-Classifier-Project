"""Train and evaluate a chronological baseline classifier."""

from sklearn.metrics import classification_report

from src.data import download_prices
from src.features import FEATURE_COLUMNS, make_dataset
from src.models import MODELS

TICKER = "SPY"
START_DATE = "2005-01-01"
TEST_FRACTION = 0.20

def main() -> None:
    prices = download_prices(TICKER, START_DATE)
    dataset = make_dataset(prices)
    split_index = int(len(dataset) * (1 - TEST_FRACTION))
    train, test = dataset.iloc[:split_index], dataset.iloc[split_index:]
    for model_name, model in MODELS.items():
        model.fit(train[FEATURE_COLUMNS], train["regime"])
        predictions = model.predict(test[FEATURE_COLUMNS])
        print(f"Model: {model_name}")
        print(f"Ticker: {TICKER}")
        print(f"Training: {train.index.min().date()} to {train.index.max().date()}")
        print(f"Test: {test.index.min().date()} to {test.index.max().date()}\n")
        print(classification_report(test["regime"], predictions, digits=3, zero_division=0))


if __name__ == "__main__":
    main()