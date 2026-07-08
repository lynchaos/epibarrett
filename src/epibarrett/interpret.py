"""
Model interpretability.

For the sparse L1 model the coefficients *are* the explanation: which CpGs
survived selection and how each shifts BE/EAC odds. For the gradient-boosted
comparator we use SHAP (if installed) to attribute predictions to probes, with a
permutation-importance fallback so the pipeline never hard-depends on SHAP.
"""

from __future__ import annotations

import numpy as np


def lasso_selected_features(fitted_pipeline, feature_names) -> list[tuple[str, float]]:
    """Return (probe, coefficient) for probes with non-zero L1 weight.

    Handles the ModeratedT selection step so names map back to original probes.
    """
    select = fitted_pipeline.named_steps.get("select")
    clf = fitted_pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()
    if select is not None and hasattr(select, "support_"):
        names = np.array(feature_names)[select.support_]
    else:
        names = np.array(feature_names)
    pairs = [(n, float(c)) for n, c in zip(names, coef) if abs(c) > 1e-8]
    pairs.sort(key=lambda kv: -abs(kv[1]))
    return pairs


def gbm_importance(fitted_pipeline, X, feature_names, max_display: int = 20):
    """SHAP mean-|value| importance for the GBM, or permutation fallback.

    Returns list of (probe, importance) sorted descending.
    """
    select = fitted_pipeline.named_steps.get("select")
    clf = fitted_pipeline.named_steps["clf"]
    Xsel = select.transform(np.asarray(X)) if select is not None else np.asarray(X)
    names = (
        np.array(feature_names)[select.support_]
        if select is not None and hasattr(select, "support_")
        else np.array(feature_names)
    )
    try:
        import shap

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xsel)
        vals = sv[1] if isinstance(sv, list) else sv
        imp = np.abs(vals).mean(axis=0)
    except Exception:  # pragma: no cover - fallback path
        from sklearn.inspection import permutation_importance

        # permutation importance needs y; approximate with model's own preds
        y_hat = clf.predict(Xsel)
        r = permutation_importance(
            clf, Xsel, y_hat, scoring="accuracy", n_repeats=5, random_state=0
        )
        imp = r.importances_mean
    order = np.argsort(-imp)[:max_display]
    return [(str(names[i]), float(imp[i])) for i in order]


def cv_selected_features(estimator, X, y, feature_names, cv) -> list[list[str]]:
    """Return the list of selected feature names for each CV fold.

    The estimator must expose a final classifier with `.coef_` and an optional
    selector step with `.support_`. This is used to visualise feature-selection
    stability across folds.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    if not hasattr(cv, "split"):
        cv = StratifiedKFold(cv, shuffle=True, random_state=0)
    folds = []
    Xa = np.asarray(X)
    ya = np.asarray(y)
    names = np.array(feature_names)
    for trn, _ in cv.split(Xa, ya):
        est = clone(estimator).fit(Xa[trn], ya[trn])
        select = est.named_steps.get("select")
        clf = est.named_steps["clf"]
        coef = clf.coef_.ravel()
        if select is not None and hasattr(select, "support_"):
            fold_names = names[select.support_]
        else:
            fold_names = names
        selected = [str(n) for n, c in zip(fold_names, coef) if abs(c) > 1e-8]
        folds.append(selected)
    return folds
