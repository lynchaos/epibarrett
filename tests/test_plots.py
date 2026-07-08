"""Tests for the publication-style plotting utilities."""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from epibarrett import plots
from epibarrett.data.simulate import SimConfig, simulate_cohort
from epibarrett.evaluate import decision_curve
from epibarrett.interpret import cv_selected_features
from epibarrett.models import lasso_panel


def _tmp_png():
    return os.path.join(tempfile.mkdtemp(), "fig.png")


def test_plot_confusion_matrix():
    y = np.array([0] * 40 + [1] * 10)
    s = np.array([0.1] * 35 + [0.6] * 5 + [0.2] * 2 + [0.8] * 8)
    path = _tmp_png()
    plots.plot_confusion_matrix(y, s, path, "Test CM", specificity=0.90)
    assert os.path.getsize(path) > 0


def test_plot_calibration_comparison():
    y = np.array([0] * 50 + [1] * 50)
    s1 = np.array([0.1] * 50 + [0.9] * 50)
    s2 = np.array([0.2] * 50 + [0.8] * 50)
    path = _tmp_png()
    plots.plot_calibration_comparison({"A": (y, s1), "B": (y, s2)}, path, "Test")
    assert os.path.getsize(path) > 0


def test_plot_decision_curve_comparison():
    y = np.array([0] * 50 + [1] * 50)
    s = np.array([0.1] * 50 + [0.9] * 50)
    path = _tmp_png()
    dca = {"A": decision_curve(y, s), "B": decision_curve(y, s * 0.9 + 0.05)}
    plots.plot_decision_curve_comparison(dca, path, "Test")
    assert os.path.getsize(path) > 0


def test_plot_model_radar():
    results = {
        "within.lasso": {"auroc": 0.9, "auprc": 0.7,
                         "sensitivity_at_spec90": 0.8,
                         "specificity_at_sens90": 0.75},
        "within.gbm": {"auroc": 0.88, "auprc": 0.68,
                       "sensitivity_at_spec90": 0.78,
                       "specificity_at_sens90": 0.73},
    }
    path = _tmp_png()
    plots.plot_model_radar(results, path, "Test radar")
    assert os.path.getsize(path) > 0


def test_plot_feature_stability():
    folds = [["p1", "p2", "p3"], ["p2", "p3", "p4"], ["p1", "p2", "p5"]]
    path = _tmp_png()
    plots.plot_feature_stability(folds, path, "Test stability")
    assert os.path.getsize(path) > 0


def test_plot_cohort_embedding():
    X, meta = simulate_cohort(60, cfg=SimConfig(seed=10))
    path = _tmp_png()
    plots.plot_cohort_embedding(X, meta, path, "Test embedding", method="pca")
    assert os.path.getsize(path) > 0


def test_plot_prediction_distribution():
    y = np.array([0] * 30 + [1] * 30)
    s = np.array([0.2] * 30 + [0.8] * 30)
    groups = np.array(["A"] * 15 + ["B"] * 15 + ["A"] * 15 + ["B"] * 15)
    path = _tmp_png()
    plots.plot_prediction_distribution(y, s, groups, path, "Test dist")
    assert os.path.getsize(path) > 0


def test_plot_prediction_distribution_empty_group():
    y = np.array([0] * 30 + [1] * 30)
    s = np.array([0.2] * 30 + [0.8] * 30)
    groups = np.array(["A"] * 20 + ["B"] * 40)  # cohort A has only controls
    path = _tmp_png()
    plots.plot_prediction_distribution(y, s, groups, path, "Test dist empty")
    assert os.path.getsize(path) > 0


def test_plot_missing_data():
    X = pd.DataFrame(np.random.rand(20, 10), columns=[f"p{i}" for i in range(10)])
    X.iloc[0, 0] = np.nan
    path = _tmp_png()
    plots.plot_missing_data(X, path, "Test missing")
    assert os.path.getsize(path) > 0


def test_plot_missing_data_no_missing():
    X = pd.DataFrame(np.random.rand(20, 10), columns=[f"p{i}" for i in range(10)])
    path = _tmp_png()
    plots.plot_missing_data(X, path, "Test no missing")
    assert os.path.getsize(path) > 0


def test_cv_selected_features():
    X, meta = simulate_cohort(120, cfg=SimConfig(seed=11))
    y = meta["label"].to_numpy()
    model = lasso_panel(k=20, C=0.5, random_state=0)
    from sklearn.model_selection import StratifiedKFold
    folds = cv_selected_features(model, X.to_numpy(), y, list(X.columns),
                                 StratifiedKFold(3, shuffle=True, random_state=0))
    assert len(folds) == 3
    assert all(isinstance(f, list) and len(f) > 0 for f in folds)
