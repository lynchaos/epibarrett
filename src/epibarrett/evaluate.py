"""
Evaluation utilities with a clinical-screening emphasis.

Beyond AUROC we report the quantities a diagnostics team actually negotiates:
sensitivity at a fixed high specificity (and vice versa), calibration (Brier),
and net benefit via decision-curve analysis. Confidence intervals are obtained
by stratified bootstrap so point estimates are never reported bare.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass
class Metrics:
    auroc: float
    auprc: float
    brier: float
    sensitivity_at_spec90: float
    specificity_at_sens90: float
    n: int
    prevalence: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _sens_at_spec(y, s, target_spec):
    y = np.asarray(y)
    thr = np.unique(s)[::-1]
    best = 0.0
    for t in thr:
        pred = (s >= t).astype(int)
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if spec >= target_spec:
            best = max(best, sens)
    return best


def _spec_at_sens(y, s, target_sens):
    y = np.asarray(y)
    thr = np.unique(s)[::-1]
    best = 0.0
    for t in thr:
        pred = (s >= t).astype(int)
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if sens >= target_sens:
            best = max(best, spec)
    return best


def compute_metrics(y_true, y_score) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    return Metrics(
        auroc=float(roc_auc_score(y_true, y_score)),
        auprc=float(average_precision_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        sensitivity_at_spec90=float(_sens_at_spec(y_true, y_score, 0.90)),
        specificity_at_sens90=float(_spec_at_sens(y_true, y_score, 0.90)),
        n=int(len(y_true)),
        prevalence=float(np.mean(y_true)),
    )


def bootstrap_auroc(
    y_true, y_score, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Return (auroc, lo95, hi95) via stratified bootstrap resampling."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True),
             rng.choice(neg, len(neg), replace=True)]
        )
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(roc_auc_score(y_true, y_score)), float(lo), float(hi)


def decision_curve(
    y_true, y_score, thresholds: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Decision-curve analysis: net benefit of the model vs treat-all/treat-none.

    Net benefit = (TP/n) - (FP/n) * (pt / (1 - pt)) where pt is the
    threshold probability. Returns arrays for model, treat-all and treat-none.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    prev = y_true.mean()
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 60)
    nb_model, nb_all = [], []
    for pt in thresholds:
        pred = (y_score >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        w = pt / (1 - pt)
        nb_model.append(tp / n - fp / n * w)
        nb_all.append(prev - (1 - prev) * w)
    return {
        "thresholds": thresholds,
        "net_benefit_model": np.array(nb_model),
        "net_benefit_all": np.array(nb_all),
        "net_benefit_none": np.zeros_like(thresholds),
    }
