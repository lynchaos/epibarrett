"""
Preprocessing for HM450-style methylation matrices.

Implements the standard, leakage-safe steps used before modelling:
* sample QC (drop "quantity not sufficient" / high-missingness samples),
* probe QC (drop probes with excessive missingness or ~zero variance),
* imputation of residual missing values,
* beta <-> M-value conversion (M-values are ~homoscedastic and preferred for
  linear modelling; Du et al., BMC Bioinformatics 2010).

All fitting statistics (medians, variances) are computed on the training split
only, then applied to held-out data, so nothing about preprocessing leaks label
information across the train/test boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-4


def beta_to_m(beta: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Convert beta values in (0,1) to M-values = log2(beta/(1-beta))."""
    b = np.clip(np.asarray(beta, dtype=float), _EPS, 1 - _EPS)
    m = np.log2(b / (1 - b))
    if isinstance(beta, pd.DataFrame):
        return pd.DataFrame(m, index=beta.index, columns=beta.columns)
    return m


def m_to_beta(m: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Inverse of :func:`beta_to_m`."""
    x = np.asarray(m, dtype=float)
    b = 2.0**x / (1.0 + 2.0**x)
    if isinstance(m, pd.DataFrame):
        return pd.DataFrame(b, index=m.index, columns=m.columns)
    return b


@dataclass
class QCReport:
    n_samples_in: int
    n_samples_out: int
    n_probes_in: int
    n_probes_out: int
    dropped_samples: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_samples_in": self.n_samples_in,
            "n_samples_out": self.n_samples_out,
            "n_probes_in": self.n_probes_in,
            "n_probes_out": self.n_probes_out,
            "n_samples_dropped": len(self.dropped_samples),
        }


def sample_qc(
    X: pd.DataFrame, max_missing_frac: float = 0.20
) -> tuple[pd.DataFrame, list[str]]:
    """Drop samples whose fraction of missing probes exceeds ``max_missing_frac``
    (the "quantity not sufficient" analogue used by clinical labs)."""
    miss = X.isna().mean(axis=1)
    keep = miss[miss <= max_missing_frac].index
    dropped = [s for s in X.index if s not in set(keep)]
    return X.loc[keep], dropped


class Preprocessor:
    """Leakage-safe preprocessor.

    Usage::

        pp = Preprocessor().fit(X_train)
        Mtr = pp.transform(X_train)
        Mte = pp.transform(X_test)
    """

    def __init__(
        self,
        probe_max_missing: float = 0.10,
        min_variance: float = 1e-3,
    ) -> None:
        self.probe_max_missing = probe_max_missing
        self.min_variance = min_variance
        self.keep_probes_: pd.Index | None = None
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        miss = X.isna().mean(axis=0)
        candidate = miss[miss <= self.probe_max_missing].index
        M = beta_to_m(X[candidate])
        var = M.var(axis=0, ddof=1)
        keep = var[var >= self.min_variance].index
        self.keep_probes_ = keep
        self.medians_ = M[keep].median(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.keep_probes_ is None or self.medians_ is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        M = beta_to_m(X.reindex(columns=self.keep_probes_))
        M = M.fillna(self.medians_)
        return M[self.keep_probes_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
