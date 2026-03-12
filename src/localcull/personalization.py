"""
PU-calibrated personalization head (Phase 2 — NOT IMPLEMENTED).

Photographer corrections are positive-unlabeled: they primarily
promote missed keepers (false negatives) and rarely demote false
positives. This is a PU learning problem.

The data ingestion path (reading photographer-modified star ratings
from XMP sidecars and diffing against localcull:PercentileRank) is
not implemented. Implement after accumulating ≥50 confirmed
corrections across multiple shoots.
"""

import logging
import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier

from localcull.constants import PU_C, PU_MIN_EXAMPLES

logger = logging.getLogger(__name__)


class PUPersonalizationHead:
    """
    Positive-Unlabeled learning for personalization.

    'Positive' = user explicitly rated/kept (confirmed keeper)
    'Unlabeled' = pipeline-selected but not explicitly confirmed

    Uses PU estimator: P(positive) = P(labeled) / c
    where c ≈ 0.8 (photographer confirms ~80% of true positives).

    Activates only after min_examples confirmed positives.
    """

    def __init__(self, c: float = PU_C, min_examples: int = PU_MIN_EXAMPLES):
        self.c = c
        self.min_examples = min_examples
        self.model = SGDClassifier(
            loss="log_loss",
            class_weight={0: 1.0, 1: 1.0 / c},
            warm_start=True,
            random_state=42,
        )
        self.n_positives = 0
        self.fitted = False

    def update(self, features: np.ndarray, labels: np.ndarray):
        """
        features: [n, d] array of composite feature vectors
        labels: 1 = confirmed positive, 0 = unlabeled
        """
        self.model.partial_fit(features, labels, classes=[0, 1])
        self.n_positives += int(labels.sum())
        self.fitted = self.n_positives >= self.min_examples
        if self.fitted:
            logger.info(
                f"PU head activated: {self.n_positives} positives"
            )

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return PU-calibrated P(positive | features)."""
        if not self.fitted:
            return np.full(len(features), 0.5)
        raw_prob = self.model.predict_proba(features)[:, 1]
        return np.clip(raw_prob / self.c, 0, 1)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "n_positives": self.n_positives,
                    "fitted": self.fitted,
                    "c": self.c,
                },
                f,
            )
        logger.info(f"PU head saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PUPersonalizationHead":
        with open(path, "rb") as f:
            data = pickle.load(f)
        head = cls(c=data["c"])
        head.model = data["model"]
        head.n_positives = data["n_positives"]
        head.fitted = data["fitted"]
        logger.info(
            f"PU head loaded: {head.n_positives} positives, "
            f"fitted={head.fitted}"
        )
        return head


def load_pu_head(path: str | None) -> PUPersonalizationHead | None:
    """
    Load PU head from disk if path exists, otherwise return None.
    Convenience wrapper for pipeline orchestration.
    """
    if path is None:
        return None
    if not os.path.exists(path):
        logger.info(f"No PU head at {path} — running without personalization")
        return None
    return PUPersonalizationHead.load(path)
