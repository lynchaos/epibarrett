"""
Classifier definitions for BE/EAC detection.

Three model families, each returned as a self-contained scikit-learn Pipeline so
that preprocessing, feature selection and calibration all happen inside
cross-validation folds (no leakage):

1. ``lasso_panel``   -- ModeratedT DMP selection -> standardise -> L1-penalised
   logistic regression. L1 yields a sparse CpG panel, directly analogous to the
   moderated-t + LASSO classifier in the precedent literature and to the compact
   panels used by deployed assays.
2. ``gbm``           -- ModeratedT selection -> histogram gradient-boosted trees.
   A non-linear comparator; also the model we explain with SHAP.
3. ``targeted_panel``-- restrict to the VIM+CCNA1 anchor probes only, then a
   small logistic model. Emulates an EsoGuard-style targeted assay and lets us
   quantify how much a genome-wide model adds over the validated two-gene panel.

``build_multimodal`` wraps any methylation model with tabular clinical features
(age, sex, BMI, smoking, GERD) via a ColumnTransformer, demonstrating the
multi-modal genomic + clinical integration the role calls for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ModeratedTSelector

CLINICAL_FEATURES = ["age", "sex_male", "bmi", "smoker", "gerd"]


def lasso_panel(k: int = 200, C: float = 0.15, random_state: int = 0) -> Pipeline:
    """Moderated-t selection -> standardise -> L1 logistic (sparse CpG panel)."""
    return Pipeline(
        steps=[
            ("select", ModeratedTSelector(k=k)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def gbm(k: int = 500, random_state: int = 0) -> Pipeline:
    """Moderated-t selection -> histogram gradient-boosted trees (non-linear)."""
    return Pipeline(
        steps=[
            ("select", ModeratedTSelector(k=k)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.06,
                    max_iter=300,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def targeted_panel(anchor_probes: list[str], C: float = 1.0) -> Pipeline:
    """EsoGuard-like model restricted to VIM+CCNA1 anchor CpGs.

    The selector here is a fixed column subset applied upstream (see
    ``epibarrett.pipeline``); this pipeline just standardises + fits a small
    logistic model on those columns.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )


def calibrated(estimator: Pipeline, method: str = "sigmoid", cv: int | object = 3):
    """Wrap an estimator in probability calibration.

    Calibrated posterior probabilities matter clinically: a screening report
    should say "this sample's probability of BE/EAC is 0.8", not just "positive".
    Sigmoid (Platt) calibration is the default because it is more robust than
    isotonic regression for the modest sample sizes typical of HM450 cohorts.
    """
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


class MethylationScoreTransformer(BaseEstimator, TransformerMixin):
    """Wrap a methylation classifier and emit its case probability as a feature.

    The wrapped model is cloned and fit inside each cross-validation fold, so
    the methylation score is generated without leaking held-out labels.
    """

    def __init__(self, methyl_model: Pipeline):
        self.methyl_model = methyl_model

    def fit(self, X, y=None):
        self.methyl_model_ = clone(self.methyl_model)
        self.methyl_model_.fit(X, y)
        return self

    def transform(self, X):
        proba = self.methyl_model_.predict_proba(X)
        return proba[:, 1].reshape(-1, 1)


def build_multimodal(
    methyl_model: Pipeline,
    methyl_cols: list[str],
    clinical_cols: list[str] = CLINICAL_FEATURES,
) -> Pipeline:
    """Combine a methylation model's risk score with standardized clinical features.

    The returned estimator expects a DataFrame containing both the probe
    columns (``methyl_cols``) and the clinical columns (``clinical_cols``).
    The methylation branch selects the probe block, runs the supplied methylation
    classifier, and outputs a single risk score; the clinical branch selects,
    median-imputes and standardises the clinical covariates. The two feature
    blocks are concatenated and fed to an L2 logistic meta-learner.
    """
    methyl_branch = Pipeline(
        steps=[
            (
                "select",
                ColumnTransformer(
                    [("methyl", "passthrough", methyl_cols)], remainder="drop"
                ),
            ),
            ("score", MethylationScoreTransformer(methyl_model)),
        ]
    )

    clinical_branch = Pipeline(
        steps=[
            (
                "select",
                ColumnTransformer(
                    [("clin", "passthrough", clinical_cols)], remainder="drop"
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    return Pipeline(
        steps=[
            (
                "union",
                FeatureUnion(
                    [
                        ("methyl_score", methyl_branch),
                        ("clinical", clinical_branch),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l2", C=1.0, class_weight="balanced", max_iter=2000
                ),
            ),
        ]
    )


def clinical_only(clinical_cols: list[str] = CLINICAL_FEATURES) -> Pipeline:
    """Standardised logistic baseline using only clinical risk factors."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2", C=1.0, class_weight="balanced", max_iter=2000
                ),
            ),
        ]
    )


def ensure_clinical_columns(
    df: pd.DataFrame, clinical_cols: list[str] = CLINICAL_FEATURES
) -> pd.DataFrame:
    """Ensure expected clinical columns exist, filling sensible defaults.

    Binary indicators default to 0.0 (absent); continuous variables default to
    NaN so the imputer can estimate them from the data.
    """
    df = df.copy()
    defaults = {"sex_male": 0.0, "smoker": 0.0, "gerd": 0.0}
    for col in clinical_cols:
        if col not in df.columns:
            df[col] = defaults.get(col, np.nan)
    return df


def operating_threshold(
    y_true: np.ndarray, y_score: np.ndarray, target_specificity: float = 0.90
) -> float:
    """Return the score threshold that achieves ``target_specificity``.

    Screening favours a high-specificity operating point (few false positives
    triggering unnecessary endoscopy); we pick the lowest threshold whose
    specificity on the given data is >= target.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_score)
    scores = y_score[order]
    thresholds = np.unique(scores)[::-1]
    best = thresholds[-1]
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        spec = tn / max(tn + fp, 1)
        if spec >= target_specificity:
            best = thr
        else:
            break
    return float(best)
