<div align="center">

# epibarrett

**Epigenetic early detection of Barrett's esophagus and esophageal adenocarcinoma from DNA methylation.**

[![CI](https://img.shields.io/github/actions/workflow/status/lynchaos/epibarrett/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/lynchaos/epibarrett/actions)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow?style=flat-square)](https://huggingface.co/kmlyyll/epibarrett-model)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue?style=flat-square)](https://huggingface.co/datasets/kmlyyll/epibarrett-simulated-cohort)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-purple?style=flat-square)](https://huggingface.co/spaces/kmlyyll/epibarrett-demo)

</div>

> **⚠️ Research software — not a medical device.**
> This repository is a reproducible, open-source reference implementation of a
> methylation-based BE/EAC classifier. The default demo runs on a
> biologically-calibrated simulator; real-data validation on peer-reviewed GEO
> cohorts is one command away.

---

## Overview

`epibarrett` is a compact, end-to-end machine-learning pipeline that classifies
**Barrett's esophagus (BE)** and **esophageal adenocarcinoma (EAC)** from
Illumina **HumanMethylation450 (HM450)** profiles. It mirrors the non-endoscopic
*capsule-sponge + methylation* diagnostic paradigm used by clinically validated
assays such as Cyted's EndoSign and the FDA-cleared **EsoGuard VIM+CCNA1**
test.

The pipeline covers the full path from raw methylation beta values to a
clinically framed, deployable classifier:

- **Moderated-*t* differentially-methylated-probe (DMP) selection**
- **LASSO panel construction** for sparse, interpretable CpG signatures
- **Cross-cohort validation** (within-cohort + leave-one-cohort-out external)
- **Probability calibration** (sigmoid/Platt) for clinically interpretable scores
- **Decision-curve analysis** and screening-first operating points
- **SHAP / coefficient interpretability**

---

## Scientific grounding

- Barrett's esophagus is the only recognized precursor to esophageal
  adenocarcinoma. Early detection enables surveillance or endoscopic therapy
  before progression to invasive cancer.
- Endoscopy is accurate but invasive and costly; non-endoscopic cell collection
  (capsule-sponge or balloon) combined with a molecular readout is the leading
  screening alternative.
- The **VIM + CCNA1** two-gene methylation panel is the clinically validated
  anchor (~90% sensitivity / ~92% specificity vs endoscopy; Moinova et al.,
  *Sci Transl Med* 2018) and underpins the FDA-cleared EsoGuard assay.
- The modelling recipe (moderated-*t* → LASSO, cross-cohort validation) follows
  established esophageal-methylation classifier literature (AUC ≈ 0.98 external;
  Oncotarget 2019).

Full literature review with DOIs: [`docs/SCIENCE.md`](docs/SCIENCE.md).

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/lynchaos/epibarrett.git
cd epibarrett

# Install with development and interpretability extras
pip install -e ".[dev,interpret]"

# Run the offline, simulated-data demo
make demo

# Run the test suite
make test

# Run on real GEO cohorts (requires network + GEOparse)
pip install -e ".[real]"
make real
```

`make demo` writes `results/metrics.json`, `results/RESULTS.md`, and
publication-style figures to `results/figures/`.

---

## Installation options

| Extra | Command | Purpose |
|---|---|---|
| Core | `pip install -e .` | Pipeline, models, evaluation |
| Development | `pip install -e ".[dev]"` | + pytest, ruff |
| Interpretability | `pip install -e ".[interpret]"` | + SHAP for GBM explanations |
| Real data | `pip install -e ".[real]"` | + GEOparse for GEO/TCGA loaders |
| Hugging Face | `pip install -e ".[hf]"` | + `huggingface-hub` for `load_model()` |

---

## Results (simulated demo, seed 7)

Discovery cohort **n=250** (balanced); external cohort **n=192** (~8% case
prevalence, screening-like, distinct batch/quality). Probabilities are
sigmoid-calibrated within each validation regime. **Illustrative only.**

| Regime | Model | AUROC | 95% CI | sens@spec90 | Brier |
|---|---|---|---|---|---|
| Within | Genome-wide **L1 panel** | **0.950** | 0.925–0.974 | 0.873 | 0.098 |
| Within | **Multimodal** (methyl + clinical) | **0.957** | 0.933–0.976 | 0.889 | 0.091 |
| Within | Targeted **VIM+CCNA1** | 0.914 | 0.879–0.944 | 0.754 | 0.121 |
| Within | Gradient boosting | 0.897 | 0.859–0.932 | 0.714 | 0.140 |
| Within | Clinical only | 0.669 | — | 0.270 | 0.227 |
| External | Genome-wide **L1 panel** | **0.943** | 0.882–0.986 | 0.800 | 0.419 |
| External | Gradient boosting | 0.943 | 0.885–0.989 | 0.867 | 0.280 |

**Reading the table:** the validated two-gene panel already reaches ~0.91 AUROC;
a genome-wide sparse panel adds a few points; adding clinical covariates gives a
small but consistent lift; clinical risk factors alone are weak; and the L1
model transfers well to a new cohort. The L1 solution recovers the expected
biology — **VIM, CCNA1, TFPI2, ZNF345, TAC1, NELL1** are selected with positive
(hypermethylation → case) coefficients.

> **Calibration note:** the high external Brier for the LASSO panel reflects
> sigmoid calibration trained on a 50%-prevalence discovery cohort and applied
> to an ~8%-prevalence screening population. This is a realistic illustration
> that operating thresholds and calibration must be refreshed per intended-use
> population.

### Figures

| | |
|---|---|
| ![ROC](results/figures/roc_within_cohort.png) | ![Within vs external](results/figures/roc_within_vs_external.png) |
| ![Calibration](results/figures/calibration_lasso.png) | ![Decision curve](results/figures/decision_curve.png) |
| ![Panel coefficients](results/figures/lasso_coefficients.png) | ![Progression](results/figures/marker_progression.png) |

---

## Hugging Face integration

Pre-trained artifacts, a simulated dataset, and an interactive demo are
published on the Hugging Face Hub.

| Resource | Link | Description |
|---|---|---|
| 🤗 Model | [`kmlyyll/epibarrett-model`](https://huggingface.co/kmlyyll/epibarrett-model) | `lasso`, `targeted`, `multimodal` pipelines + preprocessor |
| 🤗 Dataset | [`kmlyyll/epibarrett-simulated-cohort`](https://huggingface.co/datasets/kmlyyll/epibarrett-simulated-cohort) | Simulated HM450-style cohort (n=460) |
| 🤗 Space | [`kmlyyll/epibarrett-demo`](https://huggingface.co/spaces/kmlyyll/epibarrett-demo) | Gradio UI: upload a CSV, get BE/EAC probabilities |

### Load the model in one line

```python
import epibarrett

model = epibarrett.load_model("kmlyyll/epibarrett-model")
```

Or load the full bundle:

```python
bundle = epibarrett.load_bundle("kmlyyll/epibarrett-model")
model = bundle["lasso"]
preprocessor = bundle["preprocessor"]
probe_names = bundle["probe_names"]
```

---

## Repository layout

```
epibarrett/
├── src/epibarrett/
│   ├── panels.py          # VIM/CCNA1 + supporting markers
│   ├── data/
│   │   ├── simulate.py    # Biologically-calibrated HM450 simulator
│   │   ├── download.py    # Real GEO/TCGA loaders
│   │   └── preprocess.py  # beta→M, QC, leakage-safe Preprocessor
│   ├── features.py        # Moderated-t DMP selector
│   ├── models.py          # L1 panel, GBM, targeted, multimodal, calibration
│   ├── evaluate.py        # AUROC/AUPRC, sens@spec, bootstrap CIs, DCA
│   ├── plots.py           # Publication-style figures
│   ├── interpret.py       # L1 coefficients + SHAP importances
│   ├── pipeline.py        # End-to-end orchestration
│   └── hf_hub.py          # Hugging Face Hub helpers
├── hf_space/              # Gradio demo source
├── scripts/
│   ├── run_demo.py        # Offline simulated-data pipeline
│   └── run_real.py        # Real GEO cohort pipeline
├── tests/                 # pytest suite (CI)
├── docs/                  # SCIENCE.md, DATA.md, METHODS.md, MODEL_CARD.md
└── configs/
    └── default.yaml
```

---

## Design principles

- **No information leakage.** Feature selection and preprocessing statistics are
  fit inside cross-validation folds or on the discovery split only.
- **Cross-cohort generalisation is the headline test.** Additive batch effects
  preserve AUROC ranking but break fixed thresholds; the pipeline reports both
  and recalibrates probabilities.
- **Screening-first metrics.** Sensitivity at fixed specificity, calibration
  (Brier), and net benefit — not just AUROC.
- **Honest simulation.** Effect sizes are tuned to published operating points;
  the demo never claims to be a validated assay.
- **Reproducibility.** Fixed seeds, pinned dependency ranges, CI on Python
  3.10–3.12, and a published model card.

---

## How it maps to a computational-biology workflow

| Requirement | Where in this repo |
|---|---|
| Algorithms interpreting epigenetic profiles for early GI-cancer detection | [`src/epibarrett/models.py`](src/epibarrett/models.py), [`src/epibarrett/features.py`](src/epibarrett/features.py) |
| Integration of molecular + clinical/demographic data for risk stratification | [`src/epibarrett/models.py`](src/epibarrett/models.py) (`build_multimodal`), [`src/epibarrett/pipeline.py`](src/epibarrett/pipeline.py) |
| Validation on real-world/clinical datasets | [`src/epibarrett/data/download.py`](src/epibarrett/data/download.py), leave-one-cohort-out external validation |
| Reproducible, deployable code | Installable package, pinned deps, tests, CI, model card, Hugging Face artifacts |
| Interpretability of outputs | [`src/epibarrett/interpret.py`](src/epibarrett/interpret.py) (L1 coefficients; SHAP for GBM) |
| Clinically meaningful operating points | [`src/epibarrett/evaluate.py`](src/epibarrett/evaluate.py) (sens@spec, calibration, DCA) |

---

## Citation

If you use this software, please cite it:

```bibtex
@software{yaylali_epibarrett_2026,
  author = {Yaylali, Kemal},
  title = {epibarrett: epigenetic early detection of Barrett's esophagus},
  url = {https://github.com/lynchaos/epibarrett},
  year = {2026},
}
```

---

## References

Key sources (full list in [`docs/SCIENCE.md`](docs/SCIENCE.md)):

- Moinova et al., *Sci Transl Med* 2018 — VIM+CCNA1 panel.
- Ghosal et al., *Diagnostics* 2024 — EsoGuard analytical validation (31 CpGs).
- Kaz et al., *Clin Epigenetics* 2016 / *Epigenetics* 2011 — HM450 across the
  BE→EAC spectrum.
- Oncotarget 2019 — moderated-*t* + LASSO HM450 classifier.

GEO cohorts: GSE81334, GSE104707, GSE72874; TCGA-ESCA.

---

## License

MIT — see [LICENSE](LICENSE). Research/portfolio code; not a medical device.
