"""
Biologically-calibrated *simulator* of HM450-style esophageal methylation data.

WHY THIS EXISTS
---------------
The modelling pipeline in this repo is designed to run on real, peer-reviewed
Illumina HumanMethylation450 data (GSE81334 and companion cohorts; see
docs/DATA.md). Continuous-integration runners and reviewers cloning the repo do
not have that data on disk, and public omics repositories are frequently
unreachable from sandboxed/CI environments. This module produces a synthetic
methylation matrix whose structure and effect sizes are *calibrated to the
published literature* so the entire pipeline can be exercised end-to-end offline
and in CI.

The output is clearly synthetic and is never presented as a scientific result.
Its only jobs are (a) to prove the code path works and (b) to let a reviewer see
the shapes of the analyses the model performs. Swap `simulate_cohort` for
`epibarrett.data.download.load_geo` to obtain identical downstream behaviour on
real data.

CALIBRATION TARGETS (from the literature)
-----------------------------------------
* Normal squamous esophagus has low methylation at marker loci
  (beta ~ 0.05-0.15); BE/EAC shows progressive hypermethylation
  (beta ~ 0.45-0.80), increasing along SQ -> NDBE -> LGD -> HGD -> EAC.
* VIM+CCNA1 alone separates BE/EAC from controls at ~90% sens / ~92% spec
  (Moinova 2018), i.e. strong but not perfect -> we tune marker sigma so a
  targeted panel lands near that operating point.
* Genome-wide panels reach AUC ~0.98 (e.g. 12-CpG LASSO classifier, Oncotarget
  2019), so a genome-wide model here should modestly outperform the targeted one.
* Cohorts differ by processing/FFPE batch effects, so naive cross-cohort
  transfer should lose a few points of AUC unless batch is handled.
* BMI, sex, smoking, GERD are risk factors and carry weak independent signal
  (multi-modal integration adds a little over methylation alone).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..panels import ALL_MARKERS

# Histology stages along the BE -> EAC progression axis.
STAGES = ("SQ", "NDBE", "LGD", "HGD", "EAC")
CONTROL_STAGES = ("SQ",)
CASE_STAGES = ("NDBE", "LGD", "HGD", "EAC")

# Relative progression "severity" used to scale marker hypermethylation.
# Non-dysplastic BE (NDBE) is deliberately closer to controls: it is the
# screening-relevant, hardest-to-separate class, and keeping it modest prevents
# the demo from producing implausibly perfect separation.
_STAGE_SEVERITY = {"SQ": 0.0, "NDBE": 0.45, "LGD": 0.62, "HGD": 0.82, "EAC": 1.00}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log2(p / (1 - p))  # M-value (log2 scale, Du et al. 2010)


def _expit_m(m: np.ndarray) -> np.ndarray:
    return 2.0**m / (1.0 + 2.0**m)  # M -> beta


@dataclass
class SimConfig:
    n_background: int = 2000          # non-marker probes
    n_weak_background: int = 60       # weakly class-associated nuisance probes
    marker_control_beta: float = 0.12  # baseline methylation at markers (SQ)
    marker_case_beta_max: float = 0.72  # EAC-level methylation at markers
    marker_sigma_m: float = 1.70       # M-space noise at markers (controls AUC)
    background_sigma_m: float = 0.9
    cellularity_mean: float = 0.70     # mean BE/tumour cell fraction in a sample
    cellularity_sd: float = 0.18       # heterogeneity in cell fraction (FFPE)
    qns_rate: float = 0.04             # "quantity not sufficient" sample fraction
    seed: int = 7


def _marker_probe_names() -> list[str]:
    names: list[str] = []
    for m in ALL_MARKERS:
        names.extend(m.probes)
    return names


def simulate_cohort(
    n: int,
    *,
    cohort: str = "cohortA",
    case_fraction: float = 0.5,
    batch_shift_m: float = 0.0,
    noise_gain: float = 1.0,
    stage_mix: dict[str, float] | None = None,
    cfg: SimConfig | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one cohort of esophageal methylation samples.

    Parameters
    ----------
    n : number of samples.
    cohort : cohort label (drives a reproducible batch effect).
    case_fraction : fraction of samples that are cases (BE/LGD/HGD/EAC).
    batch_shift_m : systematic M-space offset applied to this cohort's probes,
        emulating array/FFPE batch effects that make cross-cohort transfer hard.
    stage_mix : optional dict giving relative frequencies of the four case
        stages; defaults to a screening-like mix dominated by non-dysplastic BE.
    cfg : SimConfig with calibration constants.
    seed : overrides cfg.seed if given.

    Returns
    -------
    (X, meta) : X is a samples x probes beta-value DataFrame; meta holds
        labels, stage, cohort and clinical covariates.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    stage_mix = stage_mix or {"NDBE": 0.55, "LGD": 0.20, "HGD": 0.15, "EAC": 0.10}
    case_stages = list(stage_mix)
    case_probs = np.array([stage_mix[s] for s in case_stages], dtype=float)
    case_probs /= case_probs.sum()

    n_cases = int(round(n * case_fraction))
    n_ctrl = n - n_cases
    stages = np.array(
        ["SQ"] * n_ctrl + list(rng.choice(case_stages, size=n_cases, p=case_probs))
    )
    rng.shuffle(stages)
    y = np.isin(stages, CASE_STAGES).astype(int)
    severity = np.array([_STAGE_SEVERITY[s] for s in stages])

    # Sample-specific BE/tumour cell fraction attenuates the marker signal,
    # emulating cellular heterogeneity in FFPE/sponge material. This is the main
    # reason real assays fall short of perfect separation, especially for the
    # low-severity non-dysplastic BE class.
    cellularity = np.clip(
        rng.normal(cfg.cellularity_mean, cfg.cellularity_sd, size=n), 0.15, 1.0
    )
    severity = severity * cellularity

    # Per-cohort reproducible batch offsets per probe.
    batch_rng = np.random.default_rng(abs(hash(cohort)) % (2**32))

    marker_probes = _marker_probe_names()
    bg_probes = [f"bg_{i:05d}" for i in range(cfg.n_background)]
    weak_probes = [f"bgw_{i:04d}" for i in range(cfg.n_weak_background)]
    all_probes = marker_probes + weak_probes + bg_probes

    X = np.empty((n, len(all_probes)), dtype=float)

    # --- marker probes: hypermethylation scales with progression severity -----
    m_ctrl = float(_logit(np.array([cfg.marker_control_beta]))[0])
    m_case_max = float(_logit(np.array([cfg.marker_case_beta_max]))[0])
    delta = m_case_max - m_ctrl
    col = 0
    for _ in marker_probes:
        mean_m = m_ctrl + severity * delta
        batch = batch_shift_m + batch_rng.normal(0, 0.25)
        X[:, col] = _expit_m(
            mean_m + batch + rng.normal(0, cfg.marker_sigma_m * noise_gain, size=n)
        )
        col += 1

    # --- weak nuisance probes: small class effect, adds realistic FDR burden --
    for _ in weak_probes:
        eff = rng.normal(0, 0.35)  # tiny, mostly noise
        base = rng.uniform(-2.5, 2.5)
        batch = batch_shift_m + batch_rng.normal(0, 0.25)
        X[:, col] = _expit_m(
            base + eff * severity + batch + rng.normal(0, cfg.background_sigma_m, n)
        )
        col += 1

    # --- background probes: no class signal, mixture of methylation states ----
    for _ in bg_probes:
        base = rng.choice([-3.0, -0.5, 2.5], p=[0.5, 0.2, 0.3])  # un/partly/methylated
        batch = batch_shift_m + batch_rng.normal(0, 0.20)
        X[:, col] = _expit_m(base + batch + rng.normal(0, cfg.background_sigma_m, n))
        col += 1

    X = np.clip(X, 1e-4, 1 - 1e-4)
    Xdf = pd.DataFrame(X, columns=all_probes)
    Xdf.index = [f"{cohort}_{i:04d}" for i in range(n)]

    # --- clinical covariates (risk factors carry weak independent signal) -----
    age = rng.normal(60 + 3.0 * y, 10).clip(30, 90)
    male = rng.random(n) < (0.52 + 0.12 * y)          # male-skewed in cases
    bmi = rng.normal(27 + 1.2 * y, 4.8).clip(16, 55)
    smoker = rng.random(n) < (0.30 + 0.12 * y)
    gerd = rng.random(n) < (0.42 + 0.14 * y)          # reflux symptoms

    meta = pd.DataFrame(
        {
            "label": y,
            "stage": stages,
            "cohort": cohort,
            "age": age.round(1),
            "sex_male": male.astype(int),
            "bmi": bmi.round(1),
            "smoker": smoker.astype(int),
            "gerd": gerd.astype(int),
        },
        index=Xdf.index,
    )

    # --- inject "quantity not sufficient" samples (missingness for QC demo) ----
    n_qns = int(round(n * cfg.qns_rate))
    if n_qns:
        qns_idx = rng.choice(n, size=n_qns, replace=False)
        # blank out most probes for these samples
        mask_cols = rng.random(len(all_probes)) < 0.85
        Xdf.iloc[qns_idx, np.where(mask_cols)[0]] = np.nan

    return Xdf, meta


def simulate_multicohort(
    cfg: SimConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two cohorts with different batch effects and case mixes.

    cohortA behaves like a discovery cohort (balanced), cohortB like an external
    validation cohort (screening-like prevalence, stronger batch shift). This
    enables an honest leave-one-cohort-out external validation.
    """
    cfg = cfg or SimConfig()
    Xa, ma = simulate_cohort(
        260, cohort="GSE81334_like", case_fraction=0.5,
        batch_shift_m=0.0, cfg=cfg, seed=cfg.seed,
    )
    Xb, mb = simulate_cohort(
        200, cohort="GSE104707_like", case_fraction=0.08,
        batch_shift_m=0.6, noise_gain=1.30, cfg=cfg, seed=cfg.seed + 1,
    )
    X = pd.concat([Xa, Xb], axis=0)
    meta = pd.concat([ma, mb], axis=0)
    return X, meta
