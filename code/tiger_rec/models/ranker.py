"""A shared logistic ranker for recall-system comparisons."""
from __future__ import annotations

from typing import Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["rrf_score", "source_score", "popularity_log", "is_long_tail"]


class LogisticRanker:
    def __init__(self, c_value: float = 1.0, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=float(c_value),
            max_iter=1000,
            random_state=int(random_state),
            class_weight="balanced",
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticRanker":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        self.scaler.fit(features)
        self.model.fit(self.scaler.transform(features), labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features, dtype=np.float64))
        return self.model.predict_proba(transformed)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(
            {"scaler": self.scaler, "model": self.model, "feature_names": FEATURE_NAMES},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LogisticRanker":
        checkpoint = joblib.load(path)
        ranker = cls()
        ranker.scaler = checkpoint["scaler"]
        ranker.model = checkpoint["model"]
        return ranker


def rank_candidates(
    item_ids: Sequence[str],
    features: np.ndarray,
    ranker: LogisticRanker,
    k: int,
) -> list[str]:
    scores = ranker.predict(features)
    order = np.argsort(-scores)[:k]
    return [str(item_ids[int(index)]) for index in order]
