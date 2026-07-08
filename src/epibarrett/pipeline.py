"""
End-to-end orchestration: data -> QC -> selection -> models -> validation ->
figures + metrics. The same code runs on simulated data (offline/CI) or on real
GEO cohorts; only the loader differs.

Two validation regimes are reported, because they answer different questions:

* Within-cohort (stratified k-fold on the discovery cohort) -- the optimistic,
  standard number.
* Leave-one-cohort-out (train on discovery, test on a *held-out cohort* with its
  own batch effect) -- the honest test of whether a methylation classifier
  generalises to a new lab/site, which is exactly what matters for deployment.
"""

from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import models as M
from . import plots
from .config import RunConfig
from .data.preprocess import Preprocessor, sample_qc
from .evaluate import bootstrap_auroc, compute_metrics, decision_curve
from .interpret import gbm_importance, lasso_selected_features
from .panels import all_marker_probes, primary_panel_probes, probe_to_gene


def _oof_scores(estimator, X, y, cv) -> np.ndarray:
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    return proba[:, 1]


def run(
    load_fn: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
    cfg: RunConfig | None = None,
) -> dict:
    cfg = cfg or RunConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    figdir = os.path.join(cfg.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    # ---- 1. load + sample QC ------------------------------------------------
    Xbeta, meta = load_fn()
    Xbeta, dropped = sample_qc(Xbeta, max_missing_frac=0.20)
    meta = meta.loc[Xbeta.index]
    meta = M.ensure_clinical_columns(meta, cfg.clinical_features)

    cohorts = list(dict.fromkeys(meta["cohort"]))
    discovery = cohorts[0]
    external = cohorts[1] if len(cohorts) > 1 else None

    disc = meta["cohort"] == discovery
    Xd_beta, yd = Xbeta.loc[disc], meta.loc[disc, "label"].to_numpy()

    # ---- 2. preprocess (fit on discovery only) ------------------------------
    pp = Preprocessor().fit(Xd_beta)
    Md = pp.transform(Xd_beta)           # discovery M-values
    probe_names = list(Md.columns)

    cv = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    # ---- 3. within-cohort out-of-fold scores for each model -----------------
    results: dict[str, dict] = {}
    roc_within: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    lasso = M.lasso_panel(k=cfg.lasso_k, C=cfg.lasso_C, random_state=cfg.seed)
    gbm = M.gbm(k=cfg.gbm_k, random_state=cfg.seed)

    for name, est in (("lasso", lasso), ("gbm", gbm)):
        s = _oof_scores(M.calibrated(est, cv=cv), Md.to_numpy(), yd, cv)
        m = compute_metrics(yd, s)
        auc, lo, hi = bootstrap_auroc(yd, s, cfg.n_bootstrap, cfg.seed)
        results[f"within.{name}"] = {**m.as_dict(), "auroc_ci95": [lo, hi]}
        roc_within[name] = (yd, s)

    # targeted EsoGuard-like panel (VIM+CCNA1 anchors only) --------------------
    anchor = [p for p in primary_panel_probes() if p in Md.columns]
    if anchor:
        s = _oof_scores(
            M.calibrated(M.targeted_panel(anchor), cv=cv),
            Md[anchor].to_numpy(), yd, cv,
        )
        m = compute_metrics(yd, s)
        auc, lo, hi = bootstrap_auroc(yd, s, cfg.n_bootstrap, cfg.seed)
        results["within.targeted"] = {**m.as_dict(), "auroc_ci95": [lo, hi]}
        roc_within["targeted (VIM+CCNA1)"] = (yd, s)

    # ---- 4. multi-modal: methylation score + clinical features --------------
    Zd = pd.concat([Md, meta.loc[disc, cfg.clinical_features]], axis=1)
    mm = M.calibrated(
        M.build_multimodal(lasso, probe_names, cfg.clinical_features), cv=cv
    )
    s_mm = _oof_scores(mm, Zd, yd, cv)
    m = compute_metrics(yd, s_mm)
    auc, lo, hi = bootstrap_auroc(yd, s_mm, cfg.n_bootstrap, cfg.seed)
    results["within.multimodal"] = {**m.as_dict(), "auroc_ci95": [lo, hi]}
    roc_within["multimodal (methyl+clinical)"] = (yd, s_mm)

    # clinical-only baseline (context for how much methylation adds) ----------
    clin_df = meta.loc[disc, cfg.clinical_features]
    s_clin = _oof_scores(M.calibrated(M.clinical_only(cfg.clinical_features), cv=cv), clin_df, yd, cv)
    results["within.clinical_only"] = compute_metrics(yd, s_clin).as_dict()

    # ---- 5. leave-one-cohort-out external validation ------------------------
    roc_external: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if external is not None:
        ext = meta["cohort"] == external
        Xe_beta, ye = Xbeta.loc[ext], meta.loc[ext, "label"].to_numpy()
        Me = pp.transform(Xe_beta)          # SAME preprocessor (no refit)
        for name, est in (("lasso", M.lasso_panel(cfg.lasso_k, cfg.lasso_C, cfg.seed)),
                          ("gbm", M.gbm(cfg.gbm_k, cfg.seed))):
            est = M.calibrated(est, cv=cv)
            est.fit(Md.to_numpy(), yd)
            s = est.predict_proba(Me.to_numpy())[:, 1]
            m = compute_metrics(ye, s)
            auc, lo, hi = bootstrap_auroc(ye, s, cfg.n_bootstrap, cfg.seed)
            results[f"external.{name}"] = {**m.as_dict(), "auroc_ci95": [lo, hi]}
            roc_external[name] = (ye, s)

    # ---- 6. fit final LASSO on all discovery for interpretability -----------
    lasso_final = M.lasso_panel(cfg.lasso_k, cfg.lasso_C, cfg.seed).fit(
        Md.to_numpy(), yd
    )
    selected = lasso_selected_features(lasso_final, probe_names)
    p2g = probe_to_gene()
    results["selected_panel"] = [
        {"probe": p, "gene": p2g.get(p, "background"), "coef": c}
        for p, c in selected[:25]
    ]
    gbm_final = M.gbm(cfg.gbm_k, cfg.seed).fit(Md.to_numpy(), yd)
    results["gbm_top_features"] = [
        {"probe": p, "gene": p2g.get(p, "background"), "importance": v}
        for p, v in gbm_importance(gbm_final, Md.to_numpy(), probe_names)
    ]

    # ---- 7. figures ---------------------------------------------------------
    plots.plot_roc(roc_within, os.path.join(figdir, "roc_within_cohort.png"),
                   f"Within-cohort ROC ({discovery})")
    plots.plot_pr({k: v for k, v in roc_within.items() if "targeted" not in k},
                  os.path.join(figdir, "pr_within_cohort.png"),
                  "Within-cohort precision-recall")
    plots.plot_calibration(*roc_within["lasso"],
                           os.path.join(figdir, "calibration_lasso.png"),
                           "Calibration (LASSO panel)")
    dca = decision_curve(*roc_within["lasso"])
    plots.plot_decision_curve(dca, os.path.join(figdir, "decision_curve.png"),
                              "Decision-curve analysis (LASSO panel)")
    genes = [f"{p2g.get(p, 'bg')}:{p}" for p, _ in selected[:20]]
    coefs = [c for _, c in selected[:20]]
    if genes:
        plots.plot_coefficients(genes, coefs,
                                os.path.join(figdir, "lasso_coefficients.png"),
                                "Selected CpG panel (L1 logistic)")
    plots.plot_progression(meta.loc[disc], Xd_beta, all_marker_probes(),
                           os.path.join(figdir, "marker_progression.png"),
                           "Marker methylation by histology stage")
    if roc_external:
        overlay = {f"within {k}": roc_within[k] for k in ("lasso", "gbm")}
        overlay.update({f"external {k}": v for k, v in roc_external.items()})
        plots.plot_roc(overlay, os.path.join(figdir, "roc_within_vs_external.png"),
                       "Within- vs external-cohort ROC (generalisation)")

    # ---- 8. persist metrics + human-readable table --------------------------
    meta_summary = {
        "discovery_cohort": discovery,
        "external_cohort": external,
        "n_discovery": int(disc.sum()),
        "n_external": int((meta["cohort"] == external).sum()) if external else 0,
        "n_probes_after_qc": len(probe_names),
        "n_samples_dropped_qc": len(dropped),
        "discovery_prevalence": float(np.mean(yd)),
        "external_prevalence": float(np.mean(ye)) if external is not None else None,
    }
    results["_summary"] = meta_summary
    with open(os.path.join(cfg.outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_table(results, os.path.join(cfg.outdir, "RESULTS.md"), meta_summary)
    return results


def _write_table(results: dict, path: str, summary: dict) -> None:
    rows = []
    for key, v in results.items():
        if not key.startswith(("within.", "external.")):
            continue
        regime, model = key.split(".", 1)
        ci = v.get("auroc_ci95")
        ci_s = f"[{ci[0]:.3f}-{ci[1]:.3f}]" if ci else ""
        rows.append(
            (regime, model, v["auroc"], ci_s, v["auprc"],
             v["sensitivity_at_spec90"], v["brier"])
        )
    rows.sort(key=lambda r: (r[0], -r[2]))
    with open(path, "w") as f:
        f.write("# Results (auto-generated)\n\n")
        f.write(f"Discovery: **{summary['discovery_cohort']}** "
                f"(n={summary['n_discovery']}, "
                f"prevalence={summary['discovery_prevalence']:.2f})  \n")
        ext_prev = summary.get("external_prevalence")
        ext_prev_s = f", prevalence={ext_prev:.2f}" if ext_prev is not None else ""
        f.write(f"External: **{summary['external_cohort']}** "
                f"(n={summary['n_external']}{ext_prev_s})  \n")
        f.write(f"Probes after QC: {summary['n_probes_after_qc']}; "
                f"QNS samples dropped: {summary['n_samples_dropped_qc']}\n\n")
        f.write("| regime | model | AUROC | 95% CI | AUPRC | "
                "sens@spec90 | Brier |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for regime, model, auroc, ci_s, auprc, sens, brier in rows:
            f.write(f"| {regime} | {model} | {auroc:.3f} | {ci_s} | "
                    f"{auprc:.3f} | {sens:.3f} | {brier:.3f} |\n")
        f.write("\n_Simulated-data demo run; numbers illustrate the pipeline, "
                "not a validated assay. Replace the loader with real GEO cohorts "
                "(see docs/DATA.md) to reproduce on peer-reviewed data._\n")


def run_offline_demo(cfg: RunConfig | None = None) -> dict:
    """Run the whole pipeline on the biologically-calibrated simulator."""
    from .data.simulate import SimConfig, simulate_multicohort

    cfg = cfg or RunConfig()

    def _load():
        return simulate_multicohort(SimConfig(seed=cfg.seed))

    return run(_load, cfg)
