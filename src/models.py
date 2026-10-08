"""Model definitions used by training and evaluation scripts."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5_000, class_weight="balanced")),
    ]),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", max_depth=8),
    "KNeighborsClassifier": Pipeline([
        ("scale", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "HistGradientBoostingClassifier": Pipeline([
        ("scale", StandardScaler()),
        ("classifier", HistGradientBoostingClassifier(max_iter=100, random_state=42, learning_rate=0.05)),
    ]),
}