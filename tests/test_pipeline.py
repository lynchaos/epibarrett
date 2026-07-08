"""Fast, deterministic tests exercised in CI (no network, no real data)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from epibarrett.data.download import _classify_stage, _parse_clinical
from epibarrett.data.preprocess import Preprocessor, beta_to_m, m_to_beta, sample_qc
from epibarrett.data.simulate import SimConfig, simulate_cohort, simulate_multicohort
from epibarrett.evaluate import compute_metrics, decision_curve
from epibarrett.features import ModeratedTSelector, moderated_t
from epibarrett.models import build_multimodal, clinical_only, ensure_clinical_columns
from epibarrett.panels import primary_panel_probes, probe_to_gene


# --- data / simulator ---------------------------------------------------------
def test_simulate_shapes_and_labels():
    X, meta = simulate_cohort(80, cohort="t", case_fraction=0.5, cfg=SimConfig(seed=1))
    assert X.shape[0] == 80 == meta.shape[0]
    assert set(meta["label"].unique()) <= {0, 1}
    assert {"age", "bmi", "sex_male", "smoker", "gerd"}.issubset(meta.columns)


def test_markers_hypermethylated_in_cases():
    X, meta = simulate_cohort(400, cohort="t", case_fraction=0.5, cfg=SimConfig(seed=2))
    probes = [p for p in primary_panel_probes() if p in X.columns]
    case_mean = X.loc[meta["label"] == 1, probes].mean().mean()
    ctrl_mean = X.loc[meta["label"] == 0, probes].mean().mean()
    assert case_mean > ctrl_mean + 0.1  # BE/EAC hypermethylated at markers


def test_multicohort_two_batches():
    X, meta = simulate_multicohort(SimConfig(seed=3))
    assert meta["cohort"].nunique() == 2
    cohorts = meta["cohort"].unique()
    # Both cohorts must share the exact same probe set (inner-join alignment).
    for c in cohorts:
        assert set(X.columns) == set(X.loc[meta["cohort"] == c].columns)
    # Cohorts are distinct and the external set has lower case prevalence.
    prev_a = meta.loc[meta["cohort"] == cohorts[0], "label"].mean()
    prev_b = meta.loc[meta["cohort"] == cohorts[1], "label"].mean()
    assert prev_b < prev_a


# --- preprocess ---------------------------------------------------------------
def test_beta_m_roundtrip():
    b = np.array([[0.1, 0.5, 0.9]])
    assert np.allclose(m_to_beta(beta_to_m(b)), b, atol=1e-6)


def test_preprocessor_no_leakage_columns():
    X, meta = simulate_cohort(60, cfg=SimConfig(seed=4))
    pp = Preprocessor().fit(X)
    M1 = pp.transform(X)
    # transforming a subset yields the same columns in the same order
    M2 = pp.transform(X.iloc[:10])
    assert list(M1.columns) == list(M2.columns)
    assert not M1.isna().any().any()


def test_sample_qc_drops_qns():
    X, meta = simulate_cohort(120, cfg=SimConfig(seed=5, qns_rate=0.1))
    Xq, dropped = sample_qc(X, max_missing_frac=0.2)
    assert len(dropped) >= 1
    assert Xq.shape[0] == X.shape[0] - len(dropped)


# --- features -----------------------------------------------------------------
def test_moderated_t_separates_markers_from_background():
    X, meta = simulate_cohort(300, cfg=SimConfig(seed=6))
    y = meta["label"].to_numpy()
    t = np.abs(moderated_t(X.to_numpy(), y))
    from epibarrett.panels import all_marker_probes

    marker_cols = [X.columns.get_loc(p) for p in all_marker_probes()
                   if p in X.columns]
    bg_cols = [i for i, c in enumerate(X.columns) if c.startswith("bg_")]
    # Informative markers carry substantially larger |t| than pure background.
    assert t[marker_cols].mean() > 2.0 * t[bg_cols].mean()


def test_selector_shape():
    X, meta = simulate_cohort(60, cfg=SimConfig(seed=7))
    sel = ModeratedTSelector(k=50).fit(X.to_numpy(), meta["label"].to_numpy())
    assert sel.transform(X.to_numpy()).shape[1] == 50


# --- evaluate -----------------------------------------------------------------
def test_metrics_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    m = compute_metrics(y, s)
    assert 0 <= m.auroc <= 1 and 0 <= m.auprc <= 1
    assert 0 <= m.sensitivity_at_spec90 <= 1


def test_perfect_separation_metrics():
    y = np.array([0] * 50 + [1] * 50)
    s = np.array([0.1] * 50 + [0.9] * 50)
    m = compute_metrics(y, s)
    assert m.auroc == pytest.approx(1.0)
    assert m.sensitivity_at_spec90 == pytest.approx(1.0)


def test_decision_curve_keys():
    y = np.array([0, 1] * 50)
    s = np.random.default_rng(1).random(100)
    dca = decision_curve(y, s)
    assert {"thresholds", "net_benefit_model", "net_benefit_all"} <= set(dca)


# --- models -----------------------------------------------------------------
def test_build_multimodal_uses_methylation_and_clinical():
    X, meta = simulate_cohort(120, cohort="t", case_fraction=0.5, cfg=SimConfig(seed=8, qns_rate=0.0))
    y = meta["label"].to_numpy()
    methyl_cols = [c for c in X.columns if c.startswith("cg_")]
    Z = pd.concat([X[methyl_cols], meta[["age", "sex_male", "bmi", "smoker", "gerd"]]], axis=1)
    model = build_multimodal(
        methyl_model=__import__("epibarrett.models", fromlist=["lasso_panel"]).lasso_panel(k=20, C=0.5, random_state=0),
        methyl_cols=methyl_cols,
        clinical_cols=["age", "sex_male", "bmi", "smoker", "gerd"],
    )
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    s = cross_val_predict(model, Z, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    assert len(s) == len(y)
    assert 0 <= s.min() <= s.max() <= 1


def test_clinical_only_pipeline():
    X, meta = simulate_cohort(60, cfg=SimConfig(seed=9))
    y = meta["label"].to_numpy()
    clin = meta[["age", "sex_male", "bmi", "smoker", "gerd"]]
    model = clinical_only(clinical_cols=["age", "sex_male", "bmi", "smoker", "gerd"])
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    s = cross_val_predict(model, clin, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    assert len(s) == len(y)


def test_ensure_clinical_columns_fills_defaults():
    df = pd.DataFrame({"age": [55.0, 60.0]})
    df = ensure_clinical_columns(df)
    assert df["sex_male"].tolist() == [0.0, 0.0]
    assert df["smoker"].tolist() == [0.0, 0.0]
    assert df["gerd"].tolist() == [0.0, 0.0]
    assert "bmi" in df.columns and df["bmi"].isna().all()


# --- real-data parsing --------------------------------------------------------
def test_parse_clinical_complete():
    out = _parse_clinical("age: 55; sex: male; bmi: 28.5; smoker: yes; gerd: yes")
    assert out["age"] == 55.0
    assert out["sex_male"] == 1.0
    assert out["bmi"] == 28.5
    assert out["smoker"] == 1.0
    assert out["gerd"] == 1.0


def test_parse_clinical_missing_binary_defaults():
    out = _parse_clinical("age: 60; sex: female")
    assert out["sex_male"] == 0.0
    assert out["smoker"] == 0.0
    assert out["gerd"] == 0.0


def test_parse_clinical_gerd_synonyms():
    for text in ("reflux: yes", "heartburn: yes", "gord: yes"):
        assert _parse_clinical(text)["gerd"] == 1.0


def test_classify_stage():
    assert _classify_stage("Barrett's esophagus") == "NDBE"
    assert _classify_stage("normal squamous") == "SQ"
    assert _classify_stage("esophageal adenocarcinoma") == "EAC"
    assert _classify_stage("high-grade dysplasia") == "HGD"
    assert _classify_stage("low-grade dysplasia") == "LGD"
    assert _classify_stage("some unrelated tissue") is None


# --- panels -------------------------------------------------------------------
def test_probe_gene_map_covers_primary():
    p2g = probe_to_gene()
    for p in primary_panel_probes():
        assert p2g[p] in {"VIM", "CCNA1"}
