"""Central configuration for the demo/real pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    outdir: str = "results"
    seed: int = 7
    cv_folds: int = 5
    lasso_k: int = 200          # candidate probes into L1 selection
    lasso_C: float = 0.15
    gbm_k: int = 500
    target_specificity: float = 0.90
    n_bootstrap: int = 1000
    clinical_features: list[str] = field(
        default_factory=lambda: ["age", "sex_male", "bmi", "smoker", "gerd"]
    )
