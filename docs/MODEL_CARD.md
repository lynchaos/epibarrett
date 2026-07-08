# Model card — epibarrett BE/EAC methylation classifier

## Intended use

Research demonstration of a methylation-based classifier for **non-endoscopic
early detection of Barrett's esophagus / esophageal adenocarcinoma**, in the
spirit of capsule-sponge + methylation assays. It is a portfolio/reference
implementation, **not** a medical device and not for clinical use.

## Inputs

Per-sample HM450 (or EPIC) beta values, optionally with clinical covariates
(age, sex, BMI, smoking, reflux). The targeted arm uses only VIM+CCNA1 CpGs.

## Output

A calibrated probability of BE/EAC and a binary call at a chosen operating
threshold (default: the threshold achieving 90% specificity, a screening-
appropriate, false-positive-averse point).

## Training / evaluation data

Intended: public HM450 esophageal cohorts (GSE81334 discovery; GSE104707 /
GSE72874 / TCGA-ESCA external; see DATA.md). The shipped demo uses a
literature-calibrated simulator so the pipeline is runnable offline.

## Performance (simulated demo — illustrative only)

Discovery cohort n=250 (balanced, 50% cases); external cohort n=192 (~8% cases,
screening-like). Probabilities are sigmoid-calibrated within each regime.

| Regime | Model | AUROC | sens@spec90 | Brier |
|---|---|---|---|---|
| within | genome-wide L1 | 0.950 | 0.873 | 0.098 |
| within | targeted VIM+CCNA1 | 0.914 | 0.754 | 0.121 |
| within | multimodal (+clinical) | 0.957 | 0.889 | 0.091 |
| within | clinical only | 0.669 | 0.270 | 0.227 |
| external | genome-wide L1 | 0.943 | 0.800 | 0.419 |
| external | gradient boosting | 0.943 | 0.867 | 0.280 |

Numbers will differ on real data; see RESULTS.md produced by each run.

### PPV / NPV at realistic screening prevalence

At the within-cohort LASSO operating point (90% specificity, 87.3% sensitivity):

| Prevalence | PPV | NPV |
|---|---|---|
| 5% | ~31% | ~99% |
| 10% | ~49% | ~98% |

These illustrate why a high-specificity operating point is chosen for screening:
NPV is excellent, but even with 90% specificity the PPV remains modest at low
prevalence, so positive tests require confirmatory endoscopy.

## Ethical & safety considerations

- **Screening context**: false negatives miss a treatable precursor; false
  positives trigger unnecessary endoscopy. The default operating point favours
  specificity, and decision-curve analysis makes the trade-off explicit.
- **Population shift**: methylation varies with age, sex, BMI, smoking and cell
  composition; performance must be checked per intended-use population.
- **Site/batch transfer**: additive batch effects preserve ranking (AUROC) but
  break fixed thresholds — recalibrate per site before deployment.
- **Sample quality**: FFPE/sponge cellularity attenuates signal; QNS handling
  and per-sample QC are part of the pipeline, not afterthoughts.

## Limitations

Illustrative sample sizes; no true external clinical validation; not a
prospective study; multi-class/ordinal progression modelling is scaffolded but
not the headline result.

## Maintainer

Kemal Yaylali — https://github.com/lynchaos
