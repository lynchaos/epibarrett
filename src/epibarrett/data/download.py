"""
Real-data loaders for peer-reviewed esophageal methylation cohorts.

These functions pull HumanMethylation450 (HM450) beta matrices and sample
metadata from the Gene Expression Omnibus (GEO). They require network access
and the optional dependency ``GEOparse`` (``pip install epibarrett[real]``), so
they are intentionally NOT exercised in CI. The offline demo uses
``epibarrett.data.simulate`` instead; every downstream function accepts the same
``(X, meta)`` contract, so switching to real data is a one-line change.

Datasets (see docs/DATA.md for full provenance and licences)
------------------------------------------------------------
GSE81334   HM450, esophagus: BE (non-dysplastic, cancer-free), EAC, normal
           squamous + fundus. Yu/Grady, Fred Hutchinson. Primary discovery set.
GSE104707  HM450, esophagus (BE / EAC / normal). Cross-cohort validation.
GSE72874   HM450, esophagus. Additional external cohort.
TCGA-ESCA  HM450, esophageal carcinoma + adjacent normal (via GDC / cBioPortal).

The clinical anchor genes VIM and CCNA1 are resolved to real cg probe IDs via
``epibarrett.panels.HM450_ANCHOR_PROBES`` so the targeted-panel model can be run
on the exact loci used by deployed assays.

Label convention (matches the simulator): ``label`` = 1 for case
(BE/dysplasia/EAC), 0 for normal squamous control.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

GEO_COHORTS = {
    "GSE81334": "Discovery: BE (cancer-free) / EAC / normal squamous + fundus (HM450)",
    "GSE104707": "External validation: BE / EAC / normal (HM450)",
    "GSE72874": "External validation: esophageal HM450",
}


def _classify_stage(text: str) -> str | None:
    """Map a free-text GEO characteristics string to a histology stage."""
    t = text.lower()
    if re.search(r"squamous|normal esoph|control", t) and "carcinoma" not in t:
        return "SQ"
    if "adenocarcinoma" in t or re.search(r"\beac\b", t):
        return "EAC"
    if "high-grade" in t or "high grade" in t or "hgd" in t:
        return "HGD"
    if "low-grade" in t or "low grade" in t or "lgd" in t:
        return "LGD"
    if "barrett" in t or re.search(r"\bbe\b|\bndbe\b|metaplasia", t):
        return "NDBE"
    return None


def load_geo(
    accession: str,
    *,
    cache_dir: str = "data/geo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download a GEO series and return ``(X_beta, meta)``.

    Parameters
    ----------
    accession : GEO series accession, e.g. ``"GSE81334"``.
    cache_dir : local directory for GEOparse's SOFT cache.

    Returns
    -------
    X : samples x probes beta-value DataFrame (NaNs allowed).
    meta : DataFrame with columns ``label``, ``stage``, ``cohort`` and any
        parsed clinical covariates.
    """
    try:
        import GEOparse  # noqa: PLC0415  (optional, network-only path)
    except ImportError as exc:  # pragma: no cover - exercised only with extras
        raise ImportError(
            "load_geo requires GEOparse and network access. "
            "Install with `pip install epibarrett[real]`."
        ) from exc

    import os

    os.makedirs(cache_dir, exist_ok=True)
    gse = GEOparse.get_GEO(geo=accession, destdir=cache_dir, silent=True)

    # Beta matrix: GSMs x probes.
    frames, stages, meta_rows, index = [], [], [], []
    for name, gsm in gse.gsms.items():
        table = gsm.table
        if table is None or table.empty:
            continue
        # HM450 processed tables usually expose ID_REF + VALUE (beta).
        cols = {c.upper(): c for c in table.columns}
        id_col = cols.get("ID_REF")
        val_col = cols.get("VALUE")
        if id_col is None or val_col is None:
            continue
        s = pd.Series(
            pd.to_numeric(table[val_col], errors="coerce").values,
            index=table[id_col].values,
            name=name,
        )
        frames.append(s)
        index.append(name)

        chars = " ; ".join(gsm.metadata.get("characteristics_ch1", []))
        title = " ".join(gsm.metadata.get("title", []))
        stage = _classify_stage(chars + " " + title)
        stages.append(stage)
        meta_rows.append(_parse_clinical(chars))

    X = pd.concat(frames, axis=1).T
    X.index = index

    meta = pd.DataFrame(meta_rows, index=index)
    meta["stage"] = stages
    meta["cohort"] = accession
    meta = meta[meta["stage"].notna()]
    meta["label"] = meta["stage"].isin(["NDBE", "LGD", "HGD", "EAC"]).astype(int)
    X = X.loc[meta.index]
    return X, meta


def _parse_clinical(chars: str) -> dict[str, float]:
    """Best-effort extraction of age/sex/BMI/smoking/GERD from GEO characteristics.

    Binary indicators are explicitly set to 0.0 when not mentioned, so the
    downstream clinical model always receives a complete feature vector.
    """
    out: dict[str, float] = {"sex_male": 0.0, "smoker": 0.0, "gerd": 0.0}
    m = re.search(r"age[:=]\s*(\d+)", chars, re.I)
    if m:
        out["age"] = float(m.group(1))
    if re.search(r"sex[:=]\s*m|gender[:=]\s*m|\bmale\b", chars, re.I):
        out["sex_male"] = 1.0
    elif re.search(r"sex[:=]\s*f|gender[:=]\s*f|\bfemale\b", chars, re.I):
        out["sex_male"] = 0.0
    m = re.search(r"bmi[:=]\s*([\d.]+)", chars, re.I)
    if m:
        out["bmi"] = float(m.group(1))
    if re.search(r"smoker|smoking[:=]\s*(yes|ever|current|former)", chars, re.I):
        out["smoker"] = 1.0
    if re.search(r"gerd|reflux|heartburn|gord", chars, re.I):
        out["gerd"] = 1.0
    return out


def load_multicohort(
    discovery: str = "GSE81334",
    external: tuple[str, ...] = ("GSE104707",),
    *,
    cache_dir: str = "data/geo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a discovery cohort plus one or more external cohorts and align them
    on their shared probe set (inner join), mirroring ``simulate_multicohort``.
    """
    Xs, metas = [], []
    for acc in (discovery, *external):
        X, meta = load_geo(acc, cache_dir=cache_dir)
        Xs.append(X)
        metas.append(meta)
    shared = Xs[0].columns
    for X in Xs[1:]:
        shared = shared.intersection(X.columns)
    shared = np.array(sorted(shared))
    X = pd.concat([x[shared] for x in Xs], axis=0)
    meta = pd.concat(metas, axis=0)
    return X, meta
