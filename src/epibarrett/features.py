"""
Differentially-methylated-probe (DMP) feature selection.

The precedent literature selects a compact CpG panel with a moderated t-statistic
(limma-style) followed by LASSO (Oncotarget 2019, PMC6932928). We mirror that:
``ModeratedTSelector`` is a scikit-learn transformer that ranks probes by an
empirical-Bayes-shrunken t-statistic between cases and controls and keeps the
top-k. Being a transformer, it is fit *inside* each cross-validation fold, so
feature selection never sees held-out labels (a common source of optimistic bias
in omics classifiers).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def moderated_t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Empirical-Bayes moderated two-sample t-statistic per column.

    A limma-style approximation: the per-probe residual variance is shrunk
    toward the pooled median variance, which stabilises rankings when the number
    of samples is small relative to the number of probes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    g1, g0 = X[y == 1], X[y == 0]
    n1, n0 = len(g1), len(g0)
    m1, m0 = g1.mean(0), g0.mean(0)
    v1 = g1.var(0, ddof=1) if n1 > 1 else np.zeros(X.shape[1])
    v0 = g0.var(0, ddof=1) if n0 > 1 else np.zeros(X.shape[1])
    pooled = ((n1 - 1) * v1 + (n0 - 1) * v0) / max(n1 + n0 - 2, 1)
    # Empirical-Bayes shrinkage toward the median pooled variance.
    prior = np.median(pooled[pooled > 0]) if np.any(pooled > 0) else 1.0
    d0 = 2.0  # prior degrees of freedom
    post = (d0 * prior + (n1 + n0 - 2) * pooled) / (d0 + (n1 + n0 - 2))
    se = np.sqrt(post * (1.0 / max(n1, 1) + 1.0 / max(n0, 1)))
    se[se == 0] = np.nan
    t = (m1 - m0) / se
    return np.nan_to_num(t, nan=0.0)


class ModeratedTSelector(BaseEstimator, TransformerMixin):
    """Keep the top-k probes by |moderated t| (fit on training fold only)."""

    def __init__(self, k: int = 200):
        self.k = k

    def fit(self, X, y):
        t = moderated_t(np.asarray(X), np.asarray(y))
        k = min(self.k, X.shape[1])
        self.support_ = np.argsort(-np.abs(t))[:k]
        self.scores_ = t
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        return self.support_ if indices else _mask(self.support_, len(self.scores_))


def _mask(idx: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m
