# Data provenance

This project is engineered to run on **real, peer-reviewed** Illumina
HumanMethylation450 (HM450) cohorts spanning the BE→EAC spectrum. Because CI and
fresh clones have no data on disk (and NCBI/GDC are often unreachable from
sandboxes), the default demo runs on a biologically-calibrated simulator; the
real-data path is one function call away.

## Real cohorts (the intended inputs)

| Accession | Platform | Content | Role | Reference |
|---|---|---|---|---|
| **GSE81334** | HM450 | Non-dysplastic BE (cancer-free), EAC, normal squamous + fundus | Discovery | Yu / Grady, Fred Hutchinson; deposited with the BE/EAC methylation-subtype study, *Clin Cancer Res* / [PMC6565505](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565505/) |
| **GSE104707** | HM450 | BE / EAC / normal esophageal tissue | External validation | Used as an independent esophageal HM450 cohort in downstream classifier work |
| **GSE72874** | HM450 | Esophageal tissue methylation | External validation | Additional independent cohort |
| **TCGA-ESCA** | HM450 | Esophageal carcinoma (EAC + ESCC) + adjacent normal | Optional large cohort | The Cancer Genome Atlas (GDC / cBioPortal) |

Companion HM450 esophageal series that appear in the literature and can be added
as further external sets: GSE52826, GSE74693, GSE79366.

The Kaz et al. progression cohort (21 BE / 18 LGD / 18 HGD / 24 EAC / 12 SQ on
HM450; *Clinical Epigenetics* 2016,
[10.1186/s13148-016-0273-7](https://doi.org/10.1186/s13148-016-0273-7)) is the
canonical description of an HM450 series covering the full histological spectrum
and informs the class structure of the simulator.

## Labels

Samples are labelled `1` (case: NDBE / LGD / HGD / EAC) vs `0` (control: normal
squamous esophagus). The loader parses each GEO sample's `characteristics_ch1`
to assign a histology stage, then derives the binary label. Stage is retained so
the progression figure and any future multi-class/ordinal analysis can use it.

## Clinical covariates

Where GEO metadata expose them, age, sex, BMI and smoking status are parsed into
a tabular block and fused with the methylation model (see `models.build_multimodal`
and the multi-modal arm of the pipeline). BMI, sex and smoking are established
BE/EAC risk factors (Kaz et al. 2016), which is why the simulator encodes them as
weak independent signal.

## Preprocessing choices (real data)

- keep processed `VALUE` (beta) per probe; convert to M-values for modelling
  (Du et al., *BMC Bioinformatics* 2010);
- drop samples with excessive missingness (the lab "quantity not sufficient"
  analogue) and probes with high missingness or ~zero variance;
- fit all preprocessing statistics on the **training/discovery split only**;
- batch/site effects across cohorts are expected — the leave-one-cohort-out
  regime is the honest test of cross-site generalisation. (ComBat-style batch
  correction can be slotted in before modelling if desired.)

## Licences

GEO series are publicly available for research; check each series' stated terms.
TCGA/GDC data follow the NIH Genomic Data Sharing policy (open-access level-3
methylation used here). No controlled-access data are required for this repo.

## Reproducing on real data

```bash
pip install ".[real]"      # GEOparse + network
python scripts/run_real.py --discovery GSE81334 --external GSE104707 \
    --outdir results_real
```
