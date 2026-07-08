# Scientific background

## Clinical problem

Esophageal adenocarcinoma (EAC) is usually diagnosed late and has poor survival.
Its only recognised precursor is Barrett's esophagus (BE), a metaplastic
replacement of normal squamous esophageal epithelium by columnar epithelium,
which can progress through low-grade dysplasia (LGD) and high-grade dysplasia
(HGD) to EAC. Detecting BE early — and stratifying who is likely to progress —
creates a window for surveillance or endoscopic therapy before invasive cancer
develops.

Endoscopy is the diagnostic standard but is invasive, requires sedation, and is
not cost-effective for population screening. This has motivated **non-endoscopic
cell collection** (a swallowable capsule-sponge or balloon that samples cells
across the esophagus as it is withdrawn) combined with a **molecular readout**.
Cyted's EndoSign is a capsule-sponge device paired with a methylation-based
laboratory assay and machine-learning analytics for exactly this purpose.

## Why DNA methylation

Aberrant DNA methylation arises early and pervasively along the
SQ → BE → dysplasia → EAC sequence, and is measurable from the small amount of
DNA a sponge/balloon recovers. Promoter/CpG-island hypermethylation of specific
tumour-suppressor and lineage genes is a recurrent, quantitative signal that
distinguishes BE/EAC from normal squamous epithelium.

## Marker evidence used in this repo

**Primary, clinically validated anchor — VIM + CCNA1.**
Moinova et al. (*Science Translational Medicine*, 2018;
[10.1126/scitranslmed.aao5848](https://doi.org/10.1126/scitranslmed.aao5848))
identified a two-gene methylation panel — vimentin (*VIM*) and cyclin A1
(*CCNA1*) — that, from balloon-collected samples, detected BE with roughly
**90% sensitivity and 92% specificity** versus endoscopy. This panel underpins
the EsoGuard assay, the first FDA-cleared (510(k) K183262) non-endoscopic
molecular test for BE/EAC, which interrogates a defined set of 31 CpG sites
across *VIM* and *CCNA1* by targeted bisulfite next-generation sequencing
(analytical validation: Ghosal et al., *Diagnostics*, 2024;
[10.3390/diagnostics14161784](https://doi.org/10.3390/diagnostics14161784)).

**Supporting progression markers.** Additional loci are repeatedly reported as
hypermethylated across BE and its progression, and are used here as the
"supporting" marker block: *TFPI2*, *ZNF345* (both high-specificity markers in
Cytosponge-based studies), *TAC1*, *SST*, *NELL1*, and *CDKN2A/p16* (an early
event in the BE→EAC sequence). See Kaz et al. (*Epigenetics*, 2011;
[10.4161/epi.6.12.18199](https://doi.org/10.4161/epi.6.12.18199)) and the
progression-focused reviews cited in DATA.md.

## Methodological precedent

The modelling recipe here deliberately mirrors an established esophageal
methylation-classifier design: moderated *t*-statistic feature selection
followed by LASSO-penalised logistic regression, trained and validated across
independent cohorts. A representative example integrated ~1,700 HM450 samples
from TCGA and GEO and built a compact 12-CpG diagnostic classifier that
separated BE/EAC/ESCC from normal tissue with AUC ≈ 0.99 in training and
≈ 0.98 in external validation (Oncotarget, 2019;
[PMC6932928](https://pmc.ncbi.nlm.nih.gov/articles/PMC6932928/)). We reproduce
the *shape* of that pipeline — sparse panel selection, cross-cohort validation,
calibration and interpretability — as the core deliverable.

## How the simulated demo is calibrated to this literature

Public omics repositories are frequently unreachable from CI/sandbox
environments, so the runnable demo uses a synthetic HM450-style matrix whose
effect sizes are tuned to the numbers above (see `data/simulate.py`):

- normal squamous epithelium is lowly methylated at marker loci; BE/EAC shows
  progressive hypermethylation increasing along SQ → NDBE → LGD → HGD → EAC;
- the targeted VIM+CCNA1 panel lands near the ~90% sens / ~92% spec operating
  point of Moinova et al.;
- a genome-wide sparse panel modestly outperforms the targeted panel, as the
  precedent literature shows;
- cohorts differ by batch/FFPE quality, so cross-cohort transfer loses a little
  AUROC and (more importantly) breaks fixed operating thresholds;
- BMI, sex, smoking and reflux are weak independent risk factors, so adding
  clinical covariates to methylation yields only a small lift.

None of the demo numbers are presented as a validated assay. Swap the loader for
`epibarrett.data.download.load_multicohort` to reproduce the analysis on the real
GEO cohorts.
