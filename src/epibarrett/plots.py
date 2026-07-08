"""
Publication-style figures. All functions take pre-computed arrays and write a
PNG; nothing here recomputes model fits, so plots are cheap and deterministic.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

_C = {
    "lasso": "#1b6ca8",
    "gbm": "#c1666b",
    "targeted": "#8a8d91",
    "mm": "#2a9d8f",
    "clinical": "#e9c46a",
    "multimodal": "#2a9d8f",
}

_LABEL_COHORT = "#264653"
_LABEL_CASE = "#c1666b"
_LABEL_CTRL = "#2a9d8f"


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


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
    _save(fig, path)


def plot_confusion_matrix(
    y_true, y_score, path: str, title: str, specificity: float = 0.90
):
    """Confusion matrix at the threshold that first achieves `specificity`."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    thr = np.unique(y_score)[::-1]
    chosen_thr = 0.5
    for t in thr:
        pred = (y_score >= t).astype(int)
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        spec = tn / max(tn + fp, 1)
        if spec >= specificity:
            chosen_thr = t
            break
    pred = (y_score >= chosen_thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control", "Case"])
    ax.set_yticklabels(["Control", "Case"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title}\n(threshold={chosen_thr:.3f}, spec≥{specificity:.0%})")
    for i in range(2):
        for j in range(2):
            text = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
            ax.text(j, i, text, ha="center", va="center", fontsize=12,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="count")
    _save(fig, path)


def plot_shap_summary(
    fitted_gbm_pipeline,
    X: pd.DataFrame,
    feature_names,
    path: str,
    title: str,
    max_display: int = 20,
):
    """Beeswarm-style SHAP summary for the GBM, if SHAP is installed."""
    select = fitted_gbm_pipeline.named_steps.get("select")
    clf = fitted_gbm_pipeline.named_steps["clf"]
    Xsel = select.transform(np.asarray(X)) if select is not None else np.asarray(X)
    names = (
        np.array(feature_names)[select.support_]
        if select is not None and hasattr(select, "support_")
        else np.array(feature_names)
    )
    try:
        import shap
    except Exception:  # pragma: no cover - optional dependency
        warnings.warn("shap not installed; skipping SHAP summary plot.")
        return
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xsel)
    vals = sv[1] if isinstance(sv, list) else sv
    # Build a DataFrame for a matplotlib beeswarm
    df = pd.DataFrame(vals, columns=names)
    mean_abs = df.abs().mean().sort_values(ascending=False)
    top = mean_abs.head(max_display).index[::-1]
    df_top = df[top]

    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(top))))
    cmap = plt.get_cmap("RdBu_r")
    for i, feat in enumerate(top):
        y = np.full(len(df_top), i)
        x = df_top[feat].to_numpy()
        feat_vals = Xsel[:, list(names).index(feat)]
        colors = cmap(np.clip(feat_vals, 0, 1))
        # jitter y for visibility
        y = y + np.random.default_rng(42 + i).uniform(-0.15, 0.15, size=len(y))
        ax.scatter(x, y, c=colors, s=15, alpha=0.6, edgecolors="none")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(title)
    ax.axvline(0, color="0.4", lw=0.8)
    _save(fig, path)


def plot_model_radar(results: dict[str, dict], path: str, title: str):
    """Radar chart comparing models on AUROC, AUPRC, sens@spec90, specificity."""
    metrics = ["auroc", "auprc", "sensitivity_at_spec90", "specificity_at_sens90"]
    labels = ["AUROC", "AUPRC", "Sens@Spec90", "Spec@Sens90"]
    # collect per-model within-cohort results
    series: dict[str, list[float]] = {}
    for key, v in results.items():
        if not key.startswith("within."):
            continue
        name = key.split(".", 1)[1]
        series[name] = [float(v.get(m, 0)) for m in metrics]
    if not series:
        return
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for name, vals in series.items():
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, label=name,
                color=_C.get(name.split("_")[0], None))
        ax.fill(angles, vals, alpha=0.10,
                color=_C.get(name.split("_")[0], None))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    _save(fig, path)


def plot_feature_stability(
    fold_features: list[list[str]], path: str, title: str, top_n: int = 25
):
    """Selection frequency of features across CV folds."""
    from collections import Counter

    counts = Counter()
    for fold in fold_features:
        counts.update(fold)
    n_folds = len(fold_features)
    # keep features selected at least once, sorted by frequency
    items = [(f, c / n_folds) for f, c in counts.items() if c > 0]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    items = items[:top_n][::-1]
    if not items:
        return
    feats, freqs = zip(*items)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(feats))))
    colors = [_C["lasso"] if f >= 0.8 else _C["gbm"] for f in freqs]
    ax.barh(range(len(feats)), freqs, color=colors)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("selection frequency across CV folds")
    ax.set_title(title)
    ax.axvline(1.0, color="0.5", ls="--", lw=0.8)
    _save(fig, path)


def plot_cohort_embedding(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    path: str,
    title: str,
    method: str = "pca",
):
    """2-D embedding of samples coloured by cohort and label (PCA or t-SNE)."""
    from sklearn.decomposition import PCA

    Xmat = np.asarray(X)
    # fill NaN with feature means for embedding only
    fill = np.nanmean(Xmat, axis=0)
    Xmat = np.where(np.isnan(Xmat), fill, Xmat)
    if method == "tsne":
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, random_state=7, perplexity=min(30, len(Xmat) - 1)).fit_transform(Xmat)
    else:
        emb = PCA(n_components=2).fit_transform(Xmat)
    emb = pd.DataFrame(emb, columns=["D1", "D2"], index=X.index)
    df = pd.concat([emb, meta[["cohort", "label"]]], axis=1)
    cohorts = df["cohort"].unique()
    markers = {"0": "o", "1": "X"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    # by cohort
    ax = axes[0]
    for c in cohorts:
        sub = df[df["cohort"] == c]
        ax.scatter(sub["D1"], sub["D2"], label=str(c), s=35, alpha=0.7)
    ax.set_title("coloured by cohort")
    ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    ax.legend(fontsize=8, title="cohort")
    # by label
    ax = axes[1]
    for lab, name, color in ((0, "control", _LABEL_CTRL), (1, "case", _LABEL_CASE)):
        sub = df[df["label"] == lab]
        ax.scatter(sub["D1"], sub["D2"], label=name, s=35, alpha=0.7, color=color,
                   marker=markers[str(lab)])
    ax.set_title("coloured by true label")
    ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    ax.legend(fontsize=8, title="label")
    fig.suptitle(title)
    _save(fig, path)


def plot_prediction_distribution(
    y_true, y_score, groups, path: str, title: str
):
    """Violin plot of predicted probabilities by true label and cohort."""
    df = pd.DataFrame({
        "score": np.asarray(y_score),
        "label": np.asarray(y_true).astype(str),
        "cohort": np.asarray(groups),
    })
    df["label"] = df["label"].map({"0": "control", "1": "case"})
    cohorts = sorted(df["cohort"].unique())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.2 * len(cohorts), 4),
                             sharey=True)
    if len(cohorts) == 1:
        axes = [axes]
    for ax, cohort in zip(axes, cohorts):
        sub = df[df["cohort"] == cohort]
        parts = ax.violinplot(
            [sub.loc[sub["label"] == "control", "score"].values,
             sub.loc[sub["label"] == "case", "score"].values],
            positions=[1, 2], showmeans=True, showmedians=False,
        )
        for pc, color in zip(parts["bodies"], [_LABEL_CTRL, _LABEL_CASE]):
            pc.set_facecolor(color); pc.set_alpha(0.5)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["control", "case"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("predicted probability")
        ax.set_title(cohort)
    fig.suptitle(title)
    _save(fig, path)


def plot_calibration_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str, title: str
):
    """Calibration curves for multiple models on one axes."""
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.7", lw=1, label="ideal")
    for name, (y, s) in curves.items():
        frac_pos, mean_pred = calibration_curve(
            np.asarray(y), np.clip(np.asarray(s), 0, 1), n_bins=10,
            strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, "o-", lw=2, label=name,
                color=_C.get(name.split()[0].lower(), None))
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, path)


def plot_decision_curve_comparison(
    dca_curves: dict[str, dict], path: str, title: str
):
    """Overlay decision curves for several models."""
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot([0, 1], [0, 0], "--", color="0.7", lw=1, label="biopsy none")
    # treat-all is prevalence-dependent; plot once from first curve
    first = next(iter(dca_curves.values()))
    ax.plot(first["thresholds"], first["net_benefit_all"], lw=1.5,
            color="0.5", label="biopsy all")
    for name, dca in dca_curves.items():
        ax.plot(dca["thresholds"], dca["net_benefit_model"], lw=2,
                label=name, color=_C.get(name.split()[0].lower(), None))
    ax.set_xlim(0, 0.6); ax.set_ylim(bottom=min(-0.05, min(
        np.min(d["net_benefit_model"]) for d in dca_curves.values()
    )))
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_missing_data(
    X: pd.DataFrame, path: str, title: str, max_probes: int = 100
):
    """Heatmap of missing beta values (samples x probes)."""
    missing = X.isna().astype(int)
    if missing.shape[1] > max_probes:
        # choose probes with highest missingness
        col_missing = missing.sum().sort_values(ascending=False).head(max_probes).index
        missing = missing[col_missing]
    if missing.sum().sum() == 0:
        # nothing missing: render a small informative plot
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        ax.set_title(title)
        _save(fig, path)
        return
    fig, ax = plt.subplots(figsize=(8, 0.15 * missing.shape[0] + 1.5))
    im = ax.imshow(missing.to_numpy(), aspect="auto", cmap="Greys",
                   interpolation="nearest")
    ax.set_xlabel("probes"); ax.set_ylabel("samples")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="missing", ticks=[0, 1])
    _save(fig, path)
