"""
Cross-Platform Genomic Normalization & Invariant Feature Extraction
===================================================================
Eliminates technical batch effects across sequencing/microarray platforms
(Illumina HiSeq, NovaSeq, Affymetrix U133, Agilent, NanoString nCounter).

Re-exports core functionality from src.data.cross_platform_normalizer.
"""
from __future__ import annotations

from src.data.cross_platform_normalizer import (
    DEFAULT_GENE_SYNONYM_MAP,
    QUALIFIED_CROSS_TISSUE_HOUSEKEEPING,
    TISSUE_VARIABLE_HOUSEKEEPING_WARNINGS,
    CrossPlatformNormalizer,
    ExpressionProfileProvenance,
    HarmonizationReport,
    HousekeepingGeneReport,
    harmonize_gene_symbols,
    housekeeping_ratio_normalize,
    profile_expression_measurement_status,
    qualify_housekeeping_genes,
    within_sample_rank_transform,
)

DEFAULT_HOUSEKEEPING_GENES = QUALIFIED_CROSS_TISSUE_HOUSEKEEPING

__all__ = [
    "DEFAULT_HOUSEKEEPING_GENES",
    "QUALIFIED_CROSS_TISSUE_HOUSEKEEPING",
    "TISSUE_VARIABLE_HOUSEKEEPING_WARNINGS",
    "CrossPlatformNormalizer",
    "ExpressionProfileProvenance",
    "HarmonizationReport",
    "HousekeepingGeneReport",
    "within_sample_rank_transform",
    "housekeeping_ratio_normalize",
    "qualify_housekeeping_genes",
    "harmonize_gene_symbols",
    "profile_expression_measurement_status",
]
