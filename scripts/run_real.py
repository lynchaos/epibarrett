#!/usr/bin/env python3
"""
Run the identical pipeline on real GEO methylation cohorts.

Requires network access and the optional extra::

    pip install ".[real]"       # brings in GEOparse

Example::

    python scripts/run_real.py --discovery GSE81334 --external GSE104707 \
        --outdir results_real

This downloads HM450 beta matrices, harmonises them on their shared probe set,
resolves VIM/CCNA1 to real cg probe IDs for the targeted panel, and produces the
same metrics.json + figures as the offline demo. See docs/DATA.md for provenance
and licences of each cohort.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from epibarrett.config import RunConfig
from epibarrett.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discovery", default="GSE81334")
    ap.add_argument("--external", nargs="*", default=["GSE104707"])
    ap.add_argument("--outdir", default="results_real")
    ap.add_argument("--cache-dir", default="data/geo")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from epibarrett.data.download import load_multicohort

    def _load():
        return load_multicohort(
            discovery=args.discovery,
            external=tuple(args.external),
            cache_dir=args.cache_dir,
        )

    cfg = RunConfig(outdir=args.outdir, seed=args.seed)
    res = run(_load, cfg)
    print(f"Done. Wrote metrics + figures to {args.outdir}/")
    for k in ("within.lasso", "within.targeted", "external.lasso"):
        if k in res:
            print(f"  {k}: AUROC={res[k]['auroc']:.3f}")


if __name__ == "__main__":
    main()
