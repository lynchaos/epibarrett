# epibarrett model card

Epigenetic early detection of Barrett's esophagus / esophageal adenocarcinoma
from DNA methylation.

## Model description

This repository contains three scikit-learn pipelines trained on a biologically
calibrated HM450-style simulator:

- `lasso`: genome-wide moderated-t + L1 logistic panel
- `targeted`: VIM+CCNA1 two-gene assay analogue
- `multimodal`: methylation risk score + clinical covariates

All models are accompanied by a fitted `Preprocessor` (beta→M, probe QC,
median imputation) and the list of probe names expected at inference time.

## Intended use

Research demonstration only. Not a medical device. The intended input is a
samples × probes beta-value DataFrame (HM450 or EPIC) plus optional clinical
covariates (age, sex_male, bmi, smoker, gerd).

## How to use

```python
import joblib
import pandas as pd

bundle = joblib.load("epibarrett_model.joblib")
lasso = bundle["lasso"]
preprocessor = bundle["preprocessor"]
probe_names = bundle["probe_names"]

# X_beta is a DataFrame of beta values with the same probe columns
M = preprocessor.transform(X_beta[probe_names])
proba = lasso.predict_proba(M)[:, 1]
```

## Training data

Trained on the simulator in `epibarrett.data.simulate` (seed 7). Replace with
real GEO cohorts (GSE81334, GSE104707, etc.) for a scientific study.

## Performance (simulated demo)

| Regime | Model | AUROC | sens@spec90 | Brier |
|---|---|---|---|---|
| within | L1 panel | 0.950 | 0.873 | 0.098 |
| within | targeted VIM+CCNA1 | 0.914 | 0.754 | 0.121 |
| within | multimodal | 0.957 | 0.889 | 0.091 |
| external | L1 panel | 0.943 | 0.800 | 0.419 |

## License

MIT — see the GitHub repository for details.
