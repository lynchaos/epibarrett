# epibarrett

**Epigenetic early detection of Barrett's esophagus and esophageal adenocarcinoma
from DNA methylation.**

A compact, reproducible machine-learning pipeline that classifies BE/EAC from
Illumina HumanMethylation450 (HM450) profiles — mirroring the non-endoscopic
*capsule-sponge + methylation* diagnostic paradigm (e.g. Cyted's EndoSign; the
FDA-cleared EsoGuard VIM+CCNA1 assay). It does moderated-*t* differentially-
methylated-probe selection, LASSO panel construction, cross-cohort validation,
probability calibration, decision-curve analysis and SHAP/coefficient
interpretability — the full path from raw methylation to a clinically framed,
deployable classifier.

![ci](https://img.shields.io/badge/CI-passing-brightgreen) ![python](https://img.shields.io/badge/python-3.10--3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green)

> **Data note.** The default `make demo` runs on a **biologically-calibrated
> simulator** so the whole pipeline executes offline and in CI. The pipeline is
> built to run on **real, peer-reviewed GEO/TCGA HM450 cohorts** (GSE81334,
> GSE104707, TCGA-ESCA; see [`docs/DATA.md`](docs/DATA.md)) — that path is one
> function call away (`scripts/run_real.py`). Demo numbers illustrate the method,
> not a validated assay.

---

## Why this exists / scientific grounding

- Barrett's esophagus is the only recognised precursor to esophageal
  adenocarcinoma; catching it early enables surveillance/therapy before invasive
  cancer. Endoscopy doesn't scale for screening, hence non-endoscopic cell
  collection + a molecular readout.
- The clinically validated anchor is the **VIM + CCNA1** two-gene methylation
  panel — ~90% sensitivity / ~92% specificity vs endoscopy (Moinova et al.,
  *Sci Transl Med* 2018), the basis of the FDA-cleared EsoGuard test (31 CpGs).
- The modelling recipe (moderated-*t* → LASSO, cross-cohort validation) follows
  established esophageal-methylation classifier work (AUC ≈ 0.98 external;
  Oncotarget 2019).

Full literature review with DOIs: [`docs/SCIENCE.md`](docs/SCIENCE.md).

## How it maps to the Computational Biologist remit

| Requirement | Where in this repo |
|---|---|
| Algorithms interpreting **epigenetic profiles** for early GI-cancer detection | `models.py` (L1 panel, GBM), `features.py` (moderated-*t* DMP selection) |
| Integrate **molecular + clinical/demographic** data for **risk stratification** | multi-modal arm in `pipeline.py`; `build_multimodal` |
| Validate on **real-world/clinical datasets** | `data/download.py` (GEO/TCGA), leave-one-cohort-out external validation |
| **Reproducible, deployable** code | installable package, pinned deps, tests, CI, model card |
| **Interpretability** of outputs | `interpret.py` (L1 coefficients → genes; SHAP for GBM) |
| Clinically meaningful **operating points** | sensitivity@fixed-specificity, calibration, decision-curve analysis |

## Install

```bash
pip install -e ".[dev,interpret]"     # core + tests + SHAP
# optional real-data path (network + GEOparse):
pip install -e ".[real]"
```

## Quickstart

```bash
make demo        # end-to-end on simulated data -> results/
make test        # 12 fast, deterministic tests
make real        # same pipeline on real GEO cohorts (needs .[real] + network)
```

`make demo` writes `results/metrics.json`, `results/RESULTS.md` and the figures
below.

## Results (simulated demo, seed 7)

Discovery cohort n=250 (balanced); external cohort n=192 (~8% case prevalence,
screening-like, distinct batch/quality). Probabilities are sigmoid-calibrated
within each validation regime. Illustrative only.

| Regime | Model | AUROC | 95% CI | sens@spec90 | Brier |
|---|---|---|---|---|---|
| within | genome-wide **L1 panel** | **0.950** | 0.925–0.974 | 0.873 | 0.098 |
| within | multimodal (methyl+clinical) | **0.957** | 0.933–0.976 | 0.889 | 0.091 |
| within | targeted **VIM+CCNA1** | 0.914 | 0.879–0.944 | 0.754 | 0.121 |
| within | gradient boosting | 0.897 | 0.859–0.932 | 0.714 | 0.140 |
| within | clinical only | 0.669 | — | 0.270 | 0.227 |
| external | genome-wide **L1 panel** | **0.943** | 0.882–0.986 | 0.800 | 0.419 |
| external | gradient boosting | 0.943 | 0.885–0.989 | 0.867 | 0.280 |

Reading it: the validated two-gene panel already reaches ~0.91; a genome-wide
sparse panel adds a few points; adding clinical covariates gives a small but
consistent lift; clinical risk factors alone are weak; the L1 model transfers
well to a new cohort; and the L1 solution **recovers the true biology** — VIM,
CCNA1, TFPI2, ZNF345, TAC1, NELL1 are selected with positive
(hypermethylation→case) coefficients. The high external Brier for the LASSO
panel reflects calibration trained on a 50%-prevalence discovery cohort and
applied to an ~8%-prevalence screening population — a realistic reminder that
operating thresholds and calibration must be refreshed per intended-use
population.

### Figures

| | |
|---|---|
| ![ROC](results/figures/roc_within_cohort.png) | ![within vs external](results/figures/roc_within_vs_external.png) |
| ![calibration](results/figures/calibration_lasso.png) | ![decision curve](results/figures/decision_curve.png) |
| ![panel](results/figures/lasso_coefficients.png) | ![progression](results/figures/marker_progression.png) |

## Repository layout

```
src/epibarrett/
  panels.py         VIM/CCNA1 + supporting markers, gene<->probe maps
  data/
    simulate.py     biologically-calibrated HM450 simulator (offline/CI)
    download.py     real GEO/TCGA loaders (network; pip install .[real])
    preprocess.py   beta<->M, QC, leakage-safe Preprocessor
  features.py       moderated-t DMP selector (fit inside CV folds)
  models.py         L1 panel, GBM, targeted VIM+CCNA1, multimodal, calibration
  evaluate.py       AUROC/AUPRC, sens@spec, bootstrap CIs, decision curve
  plots.py          ROC/PR/calibration/DCA/coefficient/progression figures
  interpret.py      L1 coefficients + SHAP importances
  pipeline.py       orchestration; within + leave-one-cohort-out validation
scripts/            run_demo.py (offline), run_real.py (GEO)
tests/              pytest suite (CI)
docs/               SCIENCE.md, DATA.md, METHODS.md, MODEL_CARD.md
```

## Design choices worth noting

- **No leakage**: feature selection and preprocessing are fit inside CV folds /
  on the discovery split only.
- **Cross-cohort is the headline test**: additive batch effects preserve AUROC
  ranking but break fixed thresholds — the repo shows both and recalibrates.
- **Screening-first metrics**: sensitivity at fixed specificity, calibration and
  net benefit, not just AUROC.
- **Honest simulation**: effect sizes tuned to published operating points; the
  demo never claims to be a validated result.

## References

Key sources (full list in `docs/SCIENCE.md`): Moinova et al., *Sci Transl Med*
2018 (VIM+CCNA1); Ghosal et al., *Diagnostics* 2024 (EsoGuard analytical
validation, 31 CpGs); Kaz et al., *Clin Epigenetics* 2016 and *Epigenetics* 2011
(HM450 across the BE→EAC spectrum); Oncotarget 2019 (moderated-*t* + LASSO HM450
classifier). GEO: GSE81334, GSE104707, GSE72874; TCGA-ESCA.

## License

MIT — see [LICENSE](LICENSE). Research/portfolio code; not a medical device.
