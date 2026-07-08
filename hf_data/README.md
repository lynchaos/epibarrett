# epibarrett simulated cohort

A biologically calibrated, synthetic HumanMethylation450-style dataset for
Barrett's esophagus / esophageal adenocarcinoma detection.

## Files

- `X.parquet` — samples × probes beta-value matrix (460 samples, 2074 probes)
- `meta.parquet` — sample metadata: label, stage, cohort, age, sex_male, bmi,
  smoker, gerd

## Cohorts

- `GSE81334_like` — balanced discovery cohort (n=250, 50% cases)
- `GSE104707_like` — screening-like external cohort (n=192, ~8% cases)

## Usage

```python
import pandas as pd

X = pd.read_parquet("X.parquet")
meta = pd.read_parquet("meta.parquet")
```

## Note

This is synthetic data for pipeline demonstration and CI. It is not a
validated clinical dataset. For real GEO cohorts see the epibarrett GitHub
repository.
