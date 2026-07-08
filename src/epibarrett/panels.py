"""
Marker panel definitions for methylation-based detection of Barrett's esophagus
(BE) and esophageal adenocarcinoma (EAC).

The CpG identifiers used here are illustrative anchors that map onto genes with
peer-reviewed evidence as non-endoscopic BE/EAC methylation markers. When the
real-data path is used (`epibarrett.data.download`), these gene symbols are
resolved to actual Illumina HumanMethylation450 (HM450) probe IDs via the array
manifest, so the "targeted panel" model can be evaluated on the exact CpGs that
underpin deployed assays.

Primary clinical anchor
-----------------------
VIM (vimentin) + CCNA1 (cyclin A1) is the two-gene methylation panel that
underpins the only FDA-cleared non-endoscopic BE/EAC molecular assay
(EsoGuard, FDA 510(k) K183262). The panel interrogates 31 CpG sites across VIM
and CCNA1 and was first reported by Moinova et al., Sci Transl Med 2018
(~90% sensitivity / ~92% specificity vs endoscopy). Cyted's EndoSign capsule
sponge uses a conceptually equivalent methylation readout.

Supporting markers
------------------
TFPI2, ZNF345, TAC1, SST, NELL1, NDRG4, p16/CDKN2A, AKAP12, TFF1 all have
published support as hypermethylated loci across the BE -> dysplasia -> EAC
progression spectrum (Cytosponge / balloon / brushing studies).

References are collected in docs/SCIENCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Marker:
    """A methylation marker: a gene and the CpG probe anchors used for it."""

    gene: str
    probes: tuple[str, ...]
    role: str  # "primary" | "supporting"
    citation: str


# --- Clinically validated two-gene anchor (EsoGuard / Moinova 2018) -----------
PRIMARY_MARKERS: tuple[Marker, ...] = (
    Marker(
        gene="VIM",
        probes=("cg_VIM_1", "cg_VIM_2", "cg_VIM_3"),
        role="primary",
        citation="Moinova et al., Sci Transl Med 2018; 10.1126/scitranslmed.aao5848",
    ),
    Marker(
        gene="CCNA1",
        probes=("cg_CCNA1_1", "cg_CCNA1_2", "cg_CCNA1_3"),
        role="primary",
        citation="Moinova et al., Sci Transl Med 2018; 10.1126/scitranslmed.aao5848",
    ),
)

# --- Supporting progression markers ------------------------------------------
SUPPORTING_MARKERS: tuple[Marker, ...] = (
    Marker("TFPI2", ("cg_TFPI2_1", "cg_TFPI2_2"), "supporting",
           "Cytosponge methylation panels; multiple BE/EAC cohorts"),
    Marker("ZNF345", ("cg_ZNF345_1", "cg_ZNF345_2"), "supporting",
           "High-specificity BE marker (Cytosponge)"),
    Marker("TAC1", ("cg_TAC1_1",), "supporting",
           "Jin et al.; hypermethylated in EAC"),
    Marker("SST", ("cg_SST_1",), "supporting",
           "Somatostatin; BE/EAC hypermethylation"),
    Marker("NELL1", ("cg_NELL1_1",), "supporting",
           "Progression-associated hypermethylation"),
    Marker("CDKN2A", ("cg_CDKN2A_1",), "supporting",
           "p16; early loss in BE->EAC (Reid; Eads et al.)"),
)

ALL_MARKERS: tuple[Marker, ...] = PRIMARY_MARKERS + SUPPORTING_MARKERS


def primary_panel_probes() -> list[str]:
    """CpG anchors for the EsoGuard-like VIM+CCNA1 targeted panel."""
    probes: list[str] = []
    for m in PRIMARY_MARKERS:
        probes.extend(m.probes)
    return probes


def all_marker_probes() -> list[str]:
    """CpG anchors for every biologically informative marker in the simulator."""
    probes: list[str] = []
    for m in ALL_MARKERS:
        probes.extend(m.probes)
    return probes


def probe_to_gene() -> dict[str, str]:
    """Map each CpG anchor to its gene symbol (for interpretability plots)."""
    mapping: dict[str, str] = {}
    for m in ALL_MARKERS:
        for p in m.probes:
            mapping[p] = m.gene
    return mapping


# Real HM450 probe IDs for the clinical anchor genes, used by the real-data
# path to subset the array to the deployed assay's loci. These are stable
# Illumina cg identifiers annotated to VIM / CCNA1 promoter regions.
HM450_ANCHOR_PROBES: dict[str, list[str]] = {
    "VIM": ["cg16409452", "cg21051458", "cg13872923", "cg24998692"],
    "CCNA1": ["cg11429292", "cg17301223", "cg13176022"],
}
