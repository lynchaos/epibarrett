# Methods

## Pipeline overview

```
load (GEO or simulator)
  -> sample QC (drop QNS / high-missingness)
  -> Preprocessor: beta->M, probe QC, median-impute   (fit on discovery only)
  -> feature selection: moderated-t DMP ranking        (fit inside CV folds)
  -> model: L1 logistic  |  HistGB  |  targeted VIM+CCNA1  |  +clinical
  -> validation: within-cohort k-fold  &  leave-one-cohort-out external
  -> calibration, decision-curve analysis, bootstrap CIs
  -> interpretability: L1 coefficients + SHAP
  -> figures + metrics.json + RESULTS.md
```

## Feature selection

Per-probe **empirical-Bayes moderated t-statistic** (limma-style) between cases
and controls; the residual variance is shrunk toward the pooled median to
stabilise rankings in the small-n / large-p regime. Implemented as a
scikit-learn transformer (`ModeratedTSelector`) so it is re-fit inside every
cross-validation fold — feature selection never sees held-out labels. This
avoids the "selection bias" that inflates many published omics classifiers.

## Models

- **L1 logistic (`lasso_panel`)** — the primary model. L1 yields a sparse CpG
  panel, mirroring the moderated-t + LASSO precedent and the compact panels used
  by deployed assays. `class_weight="balanced"` handles screening-prevalence
  skew.
- **HistGradientBoosting (`gbm`)** — non-linear comparator and the model we
  explain with SHAP.
- **Targeted panel (`targeted_panel`)** — restricted to VIM+CCNA1 anchor CpGs,
  emulating an EsoGuard-style two-gene assay; quantifies the marginal value of a
  genome-wide model.
- **Multi-modal (`build_multimodal`)** — a methylation classifier branch that
  outputs a single risk score, concatenated with median-imputed and standardised
  clinical covariates, then fed to an L2 logistic meta-learner. The methylation
  branch is re-fit inside each cross-validation fold to avoid leakage.
- **Clinical-only (`clinical_only`)** — standardised logistic baseline using
  only age, sex, BMI, smoking and GERD.

All final classifiers are wrapped in `CalibratedClassifierCV` (sigmoid/Platt by
default) so reported probabilities are clinically interpretable and Brier score
reflects true calibration.

## Validation regimes

1. **Within-cohort**: stratified k-fold on the discovery cohort. Out-of-fold
   predictions are pooled for metrics; nothing is scored on data used to fit.
2. **Leave-one-cohort-out (external)**: fit on the entire discovery cohort,
   evaluate on a *different* cohort with its own batch/quality profile. This is
   the deployment-relevant number.

## Metrics

AUROC and AUPRC; **sensitivity at fixed 90% specificity** and specificity at
fixed 90% sensitivity (the operating points a screening programme negotiates);
Brier score for calibration of the sigmoid-calibrated probabilities; 95% CIs by
stratified bootstrap. Decision-curve analysis reports **net benefit** versus
biopsy-all / biopsy-none across threshold probabilities. PPV and NPV at
realistic screening prevalence are reported in the model card.

## Interpretability

The L1 model's non-zero coefficients are reported as the selected CpG panel
(mapped to genes), signed by direction of effect (hypermethylation → higher
BE/EAC odds). The GBM is explained with SHAP mean-|value| importances (with a
permutation-importance fallback when SHAP is absent).

## Reproducibility

Fixed seeds throughout; deterministic figures; pinned dependency ranges; CI runs
tests + a smoke pipeline on Python 3.10–3.12 and uploads the result figures as
build artifacts.
