#!/usr/bin/env python3
"""
Offline, reproducible end-to-end demo on biologically-calibrated simulated data.

    python scripts/run_demo.py --outdir results --seed 7

Produces results/metrics.json, results/RESULTS.md and results/figures/*.png.
This is the target of `make demo` and of the CI workflow.
"""

from __future__ import annotations

import argparse
import warnings

# The pinned scikit-learn range (see pyproject) uses the stable penalty= API.
# Newer pre-release sklearn emits a forward-looking deprecation for it; silence
# it so demo output stays readable without changing behaviour.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from epibarrett.config import RunConfig
from epibarrett.pipeline import run_offline_demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    cfg = RunConfig(
        outdir=args.outdir,
        seed=args.seed,
        cv_folds=args.cv_folds,
        n_bootstrap=args.bootstrap,
    )
    res = run_offline_demo(cfg)
    s = res["_summary"]
    print(f"Discovery {s['discovery_cohort']} (n={s['n_discovery']}), "
          f"external {s['external_cohort']} (n={s['n_external']}).")
    for k in ("within.lasso", "within.targeted", "within.multimodal",
              "external.lasso"):
        if k in res:
            v = res[k]
            print(f"  {k:24s} AUROC={v['auroc']:.3f}  "
                  f"sens@spec90={v['sensitivity_at_spec90']:.3f}")
    print(f"Wrote metrics + figures to {args.outdir}/")


if __name__ == "__main__":
    main()
