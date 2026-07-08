"""
Publication-style figures. All functions take pre-computed arrays and write a
PNG; nothing here recomputes model fits, so plots are cheap and deterministic.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    precision_recall_curve,
    roc_curve,
)

_C = {"lasso": "#1b6ca8", "gbm": "#c1666b", "targeted": "#8a8d91", "mm": "#2a9d8f"}


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str, title: str):
    """curves: name -> (y_true, y_score)."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        from sklearn.metrics import auc

        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})",
                color=_C.get(name.split()[0].lower(), None))
    ax.plot([0, 1], [0, 1], "--", color="0.7", lw=1)
    ax.set_xlabel("1 - specificity (false positive rate)")
    ax.set_ylabel("sensitivity (true positive rate)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, (y, s) in curves.items():
        prec, rec, _ = precision_recall_curve(y, s)
        from sklearn.metrics import average_precision_score

        ax.plot(rec, prec, lw=2,
                label=f"{name} (AP={average_precision_score(y, s):.3f})",
                color=_C.get(name.split()[0].lower(), None))
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision (PPV)")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_calibration(y, s, path: str, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    frac_pos, mean_pred = calibration_curve(y, np.clip(s, 0, 1), n_bins=10,
                                            strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=_C["lasso"], label="model")
    ax.plot([0, 1], [0, 1], "--", color="0.7", label="ideal")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_decision_curve(dca: dict[str, np.ndarray], path: str, title: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    t = dca["thresholds"]
    ax.plot(t, dca["net_benefit_model"], lw=2, color=_C["lasso"], label="model")
    ax.plot(t, dca["net_benefit_all"], lw=1.5, color="0.5", label="biopsy all")
    ax.plot(t, dca["net_benefit_none"], "--", lw=1, color="0.7", label="biopsy none")
    ax.set_ylim(bottom=min(-0.02, np.min(dca["net_benefit_model"])))
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_coefficients(genes, coefs, path: str, title: str, top: int = 20):
    order = np.argsort(-np.abs(coefs))[:top]
    g = np.array(genes)[order][::-1]
    c = np.array(coefs)[order][::-1]
    colors = [_C["lasso"] if v >= 0 else _C["gbm"] for v in c]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(g))))
    ax.barh(range(len(g)), c, color=colors)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g, fontsize=8)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("standardised logistic coefficient")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_progression(meta, X_beta, marker_probes, path: str, title: str):
    """Mean marker methylation (beta) by histology stage — the biology check."""
    from .data.simulate import STAGES

    stages = [s for s in STAGES if s in set(meta["stage"])]
    present = [p for p in marker_probes if p in X_beta.columns]
    mat = np.array(
        [[X_beta.loc[meta["stage"] == s, p].mean() for s in stages]
         for p in present]
    )
    fig, ax = plt.subplots(figsize=(1.1 * len(stages) + 2, 0.4 * len(present) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean beta (methylation)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
