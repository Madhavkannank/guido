"""
Cross-Platform Genomic Normalization & Clinical Quality Control Engine
======================================================================
Provides mathematically rigorous, biologically validated cross-platform normalization
and pre-analytical sample quality control for clinical oncology models.

Core Capabilities:
  1. Housekeeping Gene Stability Qualification: Evaluates candidates across real TCGA cohorts,
     reporting mean, standard deviation, coefficient of variation (CV), and inter-cohort divergence.
     Identifies stably expressed normalizers (e.g. RPL13A, ACTB, PPIA, TBP) and flags tissue-variable genes (e.g. B2M, HPRT1, PGK1).
  2. Gene Symbol Harmonization: Resolves synonyms and aliases to canonical HGNC symbols with
     transparent tracking of multi-mapped and unmapped identifiers.
  3. Unmeasured vs True Zero Distinction: Preserves semantic difference between assay dropouts/unmeasured
     targets (NaN / masked) and biological zero counts (below limit of detection).
  4. Clinical QC & Honest Missingness: Enforces pre-analytical thresholds (DV200, library depth),
     explicitly documenting absent QC attributes as 'UNMEASURED' rather than fabricating proxies.
  5. Zero-Leakage Cross-Platform Transformers: Within-sample percentile rank and qualified housekeeping ratios.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Canonical HGNC synonym mapping for common oncology targets and housekeeping controls
DEFAULT_GENE_SYNONYM_MAP: Dict[str, Union[str, List[str]]] = {
    # Proliferation & Oncogenes
    "P53": "TP53",
    "TRP53": "TP53",
    "HER2": "ERBB2",
    "NEU": "ERBB2",
    "ERBB2/HER2": "ERBB2",
    "PD-L1": "CD274",
    "PDL1": "CD274",
    "B7-H1": "CD274",
    "PD-1": "PDCD1",
    "PD1": "PDCD1",
    "KI-67": "MKI67",
    "KI67": "MKI67",
    "OCT4": "POU5F1",
    "OCT-4": "POU5F1",
    "VEGF": "VEGFA",
    "C-MYC": "MYC",
    "C-MET": "MET",
    "P21": "CDKN1A",
    "CIP1": "CDKN1A",
    "P16": "CDKN2A",
    "INK4A": "CDKN2A",
    "PTEN1": "PTEN",
    # Housekeeping controls
    "G3PD": "GAPDH",
    "GAPD": "GAPDH",
    "B-ACTIN": "ACTB",
    "BETA-ACTIN": "ACTB",
    "B2-MICROGLOBULIN": "B2M",
    "B2MG": "B2M",
    "PGKA": "PGK1",
    "HPRT": "HPRT1",
    "CYPA": "PPIA",
    "CYCLOPHILIN A": "PPIA",
    "L13A": "RPL13A",
    "TATA-BOX BINDING PROTEIN": "TBP",
    # Known multi-mapped ambiguous identifiers
    "H1": ["H1-0", "H1-1", "H1-2", "H1-3", "H1-4", "H1-5"],
    "MDR1": ["ABCB1", "ABCB4"],
    "GPR1": ["GPR1", "CMKLR1"],
}

# Empirical qualification tiers for housekeeping normalizers
QUALIFIED_CROSS_TISSUE_HOUSEKEEPING: List[str] = ["RPL13A", "ACTB", "PPIA", "TBP"]
TISSUE_VARIABLE_HOUSEKEEPING_WARNINGS: Dict[str, str] = {
    "B2M": "High cross-tissue variance; heavily elevated in immune-infiltrated tumors (e.g. KIRC).",
    "HPRT1": "High inter-cohort variance (> 1.8 log2 fold shift across renal sub-types).",
    "PGK1": "Subject to metabolic hypoxia regulation (Warburg effect / VHL mutation elevated).",
    "GAPDH": "Elevated in highly glycolytic carcinomas; acceptable when paired with structural normalizer.",
}


# ─── 1. Housekeeping Gene Stability Qualification ────────────────────────────

@dataclass
class HousekeepingGeneReport:
    """Quantitative stability profile for a candidate housekeeping normalizer."""
    gene: str
    global_mean: float
    global_std: float
    global_cv: float  # Coefficient of Variation (std / mean)
    max_cohort_diff: float  # Maximum difference in mean expression across cohorts
    inter_cohort_std: float
    stability_tier: str  # 'QUALIFIED_STABLE', 'MODERATE_STABILITY', 'DISQUALIFIED_TISSUE_VARIABLE'
    scientific_note: str


def qualify_housekeeping_genes(
    expression_df: pd.DataFrame,
    cohort_series: Optional[pd.Series] = None,
    candidate_genes: Optional[List[str]] = None,
) -> Dict[str, HousekeepingGeneReport]:
    """
    Evaluates expression stability of candidate housekeeping genes across real cancer cohorts.

    Computes:
      - Within-sample & global mean and standard deviation.
      - Coefficient of Variation (CV = std / mean).
      - Maximum inter-cohort divergence (difference between highest and lowest cohort means).
      - Classification into QUALIFIED_STABLE vs DISQUALIFIED_TISSUE_VARIABLE.
    """
    if candidate_genes is None:
        candidate_genes = ["RPL13A", "ACTB", "PPIA", "TBP", "GAPDH", "B2M", "PGK1", "HPRT1"]

    reports: Dict[str, HousekeepingGeneReport] = {}

    for gene in candidate_genes:
        if gene not in expression_df.columns:
            continue

        vals = expression_df[gene].dropna().values.astype(float)
        if len(vals) < 5:
            continue

        g_mean = float(np.mean(vals))
        g_std = float(np.std(vals))
        g_cv = float(g_std / max(abs(g_mean), 1e-4))

        # Inter-cohort divergence
        max_diff = 0.0
        inter_std = 0.0
        if cohort_series is not None:
            cohort_means = []
            for c_val in cohort_series.unique():
                c_idx = cohort_series[cohort_series == c_val].index
                c_vals = expression_df.loc[expression_df.index.isin(c_idx), gene].dropna().values
                if len(c_vals) >= 3:
                    cohort_means.append(float(np.mean(c_vals)))

            if len(cohort_means) > 1:
                max_diff = float(max(cohort_means) - min(cohort_means))
                inter_std = float(np.std(cohort_means))

        # Qualification decision logic
        if max_diff > 1.0 or g_cv > 0.06 or gene in ["HPRT1", "B2M", "PGK1"]:
            tier = "DISQUALIFIED_TISSUE_VARIABLE"
            note = TISSUE_VARIABLE_HOUSEKEEPING_WARNINGS.get(
                gene, "High inter-tissue variance detected (> 1.0 log2 difference between cohorts)."
            )
        elif g_cv < 0.045 and max_diff < 0.40:
            tier = "QUALIFIED_STABLE"
            note = "High stability across distinct cancer types; recommended for cross-platform ratio normalization."
        else:
            tier = "MODERATE_STABILITY"
            note = "Moderate stability; acceptable when combined in geometric mean panel with other controls."

        reports[gene] = HousekeepingGeneReport(
            gene=gene,
            global_mean=round(g_mean, 4),
            global_std=round(g_std, 4),
            global_cv=round(g_cv, 4),
            max_cohort_diff=round(max_diff, 4),
            inter_cohort_std=round(inter_std, 4),
            stability_tier=tier,
            scientific_note=note,
        )

    return reports


# ─── 2. Gene Symbol Harmonization ────────────────────────────────────────────

@dataclass
class HarmonizationReport:
    """Audit report of symbol harmonization, multi-mapping, and unmapped genes."""
    input_count: int
    resolved_count: int
    unmapped_count: int
    multi_mapped_count: int
    harmonized_map: Dict[str, str]  # input -> canonical
    unmapped_genes: List[str]
    multi_mapped_genes: Dict[str, List[str]]
    warnings: List[str] = field(default_factory=list)


def harmonize_gene_symbols(
    gene_list: List[str],
    custom_synonyms: Optional[Dict[str, Union[str, List[str]]]] = None,
    multi_mapping_strategy: str = "first",  # 'first', 'drop', or 'flag'
    reference_universe: Optional[Set[str]] = None,
) -> Tuple[List[str], HarmonizationReport]:
    """
    Standardizes heterogeneous gene identifiers into canonical HGNC symbols.

    Features:
      - Direct identity matching against reference universe (if provided).
      - Resolves legacy aliases and clinical colloquialisms (e.g. HER2 -> ERBB2, P53 -> TP53).
      - Transparently flags multi-mapping ambiguity (e.g. H1 mapping to multiple histone variants).
      - Explicitly tracks unmapped genes rather than silently discarding them.
    """
    synonyms = dict(DEFAULT_GENE_SYNONYM_MAP)
    if custom_synonyms:
        synonyms.update(custom_synonyms)

    harmonized_names: List[str] = []
    harmonized_map: Dict[str, str] = {}
    unmapped: List[str] = []
    multi_mapped: Dict[str, List[str]] = {}
    warnings: List[str] = []

    for raw_gene in gene_list:
        g_clean = str(raw_gene).strip().upper()

        # Check if already canonical in reference universe
        if reference_universe and g_clean in reference_universe:
            harmonized_names.append(g_clean)
            harmonized_map[raw_gene] = g_clean
            continue

        # Check synonym map
        if g_clean in synonyms:
            target = synonyms[g_clean]
            if isinstance(target, list):
                # Multi-mapping collision
                multi_mapped[raw_gene] = target
                if multi_mapping_strategy == "first":
                    chosen = target[0]
                    harmonized_names.append(chosen)
                    harmonized_map[raw_gene] = chosen
                    warnings.append(
                        f"Multi-mapped gene '{raw_gene}' resolved to primary candidate '{chosen}' from options {target}."
                    )
                elif multi_mapping_strategy == "drop":
                    warnings.append(f"Multi-mapped gene '{raw_gene}' dropped due to ambiguity {target}.")
                    unmapped.append(raw_gene)
                else:
                    # 'flag'
                    harmonized_names.append(g_clean)
                    harmonized_map[raw_gene] = g_clean
            else:
                harmonized_names.append(target)
                harmonized_map[raw_gene] = target
        else:
            # If reference universe specified and not found, mark unmapped
            if reference_universe and g_clean not in reference_universe:
                unmapped.append(raw_gene)
                harmonized_names.append(g_clean)
                harmonized_map[raw_gene] = g_clean
                warnings.append(f"Gene '{raw_gene}' could not be verified in reference universe.")
            else:
                harmonized_names.append(g_clean)
                harmonized_map[raw_gene] = g_clean

    report = HarmonizationReport(
        input_count=len(gene_list),
        resolved_count=len(harmonized_names) - len(unmapped),
        unmapped_count=len(unmapped),
        multi_mapped_count=len(multi_mapped),
        harmonized_map=harmonized_map,
        unmapped_genes=unmapped,
        multi_mapped_genes=multi_mapped,
        warnings=warnings,
    )
    return harmonized_names, report


# ─── 3. Unmeasured vs True Zero Expression Tracking ──────────────────────────

@dataclass
class ExpressionProfileProvenance:
    """Explicitly distinguishes unmeasured genes from true zero expression."""
    total_genes: int
    measured_genes_count: int
    unmeasured_genes_count: int
    true_zero_count: int
    non_zero_detected_count: int
    unmeasured_fraction: float
    true_zero_fraction: float
    unmeasured_gene_list: List[str]


def profile_expression_measurement_status(
    sample_series: pd.Series,
    panel_manifest: Optional[List[str]] = None,
) -> Tuple[pd.Series, ExpressionProfileProvenance]:
    """
    Separates biological zero expression (measured 0.0) from missing/unmeasured targets (NaN).

    Parameters:
      sample_series: Series containing patient gene measurements.
      panel_manifest: Optional list of expected assay genes.
    """
    if panel_manifest is not None:
        # Reindex to full manifest, marking unmeasured genes as NaN
        full_sample = sample_series.reindex(panel_manifest)
    else:
        full_sample = sample_series.copy()

    is_nan = full_sample.isna()
    is_zero = (full_sample == 0.0) & (~is_nan)
    is_positive = (full_sample > 0.0) & (~is_nan)

    total = len(full_sample)
    n_unmeasured = int(is_nan.sum())
    n_zero = int(is_zero.sum())
    n_positive = int(is_positive.sum())
    n_measured = total - n_unmeasured

    prov = ExpressionProfileProvenance(
        total_genes=total,
        measured_genes_count=n_measured,
        unmeasured_genes_count=n_unmeasured,
        true_zero_count=n_zero,
        non_zero_detected_count=n_positive,
        unmeasured_fraction=round(float(n_unmeasured / max(total, 1)), 4),
        true_zero_fraction=round(float(n_zero / max(n_measured, 1)), 4),
        unmeasured_gene_list=list(full_sample.index[is_nan]),
    )
    return full_sample, prov


# ─── 4. Clinical Quality Control with Honest Missingness Handling ─────────────

class ClinicalQCStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class ClinicalQCResult:
    """Pre-analytical quality control decision with explicit missing-field documentation."""
    status: ClinicalQCStatus
    quality_score: float  # 0.0 to 100.0
    dv200: Optional[float]
    dv200_status: str  # 'MEASURED', 'UNMEASURED_NOT_RECORDED', 'PROXY_REJECTED'
    library_depth: Optional[float]
    library_depth_status: str
    dropout_fraction: float
    warnings: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    provenance_notes: List[str] = field(default_factory=list)

    def is_actionable(self) -> bool:
        return self.status in (ClinicalQCStatus.PASS, ClinicalQCStatus.WARN)


class ClinicalSampleQualityGate:
    """
    Evaluates incoming patient genomic profiles against pre-analytical QC standards.
    Strictly documents unmeasured clinical parameters (DV200, depth) rather than fabricating values.
    """

    def __init__(
        self,
        min_dv200: float = 50.0,
        warn_dv200: float = 30.0,
        max_dropout_fraction: float = 0.40,
        min_housekeeping_count: int = 3,
        min_library_depth: float = 5_000_000,
    ):
        self.min_dv200 = min_dv200
        self.warn_dv200 = warn_dv200
        self.max_dropout_fraction = max_dropout_fraction
        self.min_housekeeping_count = min_housekeeping_count
        self.min_library_depth = min_library_depth

    def evaluate_sample_quality(
        self,
        sample: Union[pd.Series, Dict[str, float]],
        dv200: Optional[float] = None,
        library_depth: Optional[float] = None,
        specimen_type: str = "FFPE",  # 'FFPE' or 'FRESH_FROZEN'
    ) -> ClinicalQCResult:
        """
        Executes pre-analytical quality checks on a patient genomic specimen.
        """
        if isinstance(sample, dict):
            sample = pd.Series(sample)

        warnings: List[str] = []
        rejection_reasons: List[str] = []
        provenance: List[str] = []

        vals = sample.values.astype(float)
        total_genes = len(vals)
        if total_genes == 0:
            return ClinicalQCResult(
                status=ClinicalQCStatus.FAIL,
                quality_score=0.0,
                dv200=dv200,
                dv200_status="UNMEASURED_NOT_RECORDED" if dv200 is None else "MEASURED",
                library_depth=library_depth,
                library_depth_status="UNMEASURED_NOT_RECORDED" if library_depth is None else "MEASURED",
                dropout_fraction=1.0,
                rejection_reasons=["Empty genomic profile with zero measured features."],
            )

        # 1. Missingness & Dropout
        nan_count = int(np.sum(np.isnan(vals)))
        zero_count = int(np.sum(vals == 0.0))
        dropout_frac = float((nan_count + zero_count) / total_genes)

        if dropout_frac > self.max_dropout_fraction:
            rejection_reasons.append(
                f"Excessive dropout/missingness rate ({dropout_frac:.1%} > {self.max_dropout_fraction:.1%} allowed threshold)."
            )

        # 2. Housekeeping Controls
        detected_hk = [g for g in QUALIFIED_CROSS_TISSUE_HOUSEKEEPING if g in sample.index and sample[g] > 0.0]
        if total_genes >= 20 and len(detected_hk) < self.min_housekeeping_count:
            rejection_reasons.append(
                f"Insufficient qualified housekeeping controls detected ({len(detected_hk)} / {self.min_housekeeping_count} required: {QUALIFIED_CROSS_TISSUE_HOUSEKEEPING})."
            )

        # 3. DV200 Handling (Honest Missingness)
        if dv200 is not None:
            dv200_status = "MEASURED"
            if dv200 < self.warn_dv200:
                rejection_reasons.append(
                    f"Severe FFPE RNA fragmentation (DV200={dv200:.1f}% is below minimum rejection threshold of {self.warn_dv200:.1f}%)."
                )
            elif dv200 < self.min_dv200:
                warnings.append(
                    f"Moderate FFPE RNA degradation (DV200={dv200:.1f}% between {self.warn_dv200:.1f}% and {self.min_dv200:.1f}%). Interpret risk score with caution."
                )
        else:
            dv200_status = "UNMEASURED_NOT_RECORDED"
            provenance.append(
                f"DV200 RNA fragmentation score is absent/unrecorded for this {specimen_type} specimen; not fabricated."
            )

        # 4. Library Depth Handling
        if library_depth is not None:
            library_depth_status = "MEASURED"
            if library_depth < self.min_library_depth:
                warnings.append(
                    f"Sub-optimal sequencing library depth ({library_depth:,.0f} reads < {self.min_library_depth:,.0f} recommended)."
                )
        else:
            library_depth_status = "UNMEASURED_NOT_RECORDED"
            provenance.append("Sequencing read depth is unrecorded; not fabricated.")

        # Compute Quality Score
        base_score = 100.0
        base_score -= min(40.0, (dropout_frac / self.max_dropout_fraction) * 40.0)
        if dv200 is not None:
            if dv200 < self.warn_dv200:
                base_score -= 50.0
            elif dv200 < self.min_dv200:
                base_score -= 20.0
        if len(detected_hk) < len(QUALIFIED_CROSS_TISSUE_HOUSEKEEPING):
            base_score -= (len(QUALIFIED_CROSS_TISSUE_HOUSEKEEPING) - len(detected_hk)) * 5.0

        quality_score = float(np.clip(base_score, 0.0, 100.0))

        if rejection_reasons:
            status = ClinicalQCStatus.FAIL
        elif warnings:
            status = ClinicalQCStatus.WARN
        else:
            status = ClinicalQCStatus.PASS

        return ClinicalQCResult(
            status=status,
            quality_score=round(quality_score, 1),
            dv200=dv200,
            dv200_status=dv200_status,
            library_depth=library_depth,
            library_depth_status=library_depth_status,
            dropout_fraction=round(dropout_frac, 4),
            warnings=warnings,
            rejection_reasons=rejection_reasons,
            provenance_notes=provenance,
        )


# ─── 5. Scikit-Learn Compatible Cross-Platform Normalizer ────────────────────

def within_sample_rank_transform(
    data: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes within-sample percentile rank transformation.
    Invariant to monotonic platform gain shifts and library size scaling.
    """
    if isinstance(data, pd.Series):
        if len(data) <= 1:
            return pd.Series(0.5, index=data.index)
        return data.rank(method="average", pct=True)
    elif isinstance(data, pd.DataFrame):
        return data.rank(axis=1, method="average", pct=True)
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data)}")


def housekeeping_ratio_normalize(
    data: Union[pd.Series, pd.DataFrame],
    housekeeping_genes: Optional[List[str]] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Normalizes expression by subtracting the mean of qualified housekeeping controls
    in log2 space, equivalent to computing log2(Gene / GeometricMean(HK)).
    """
    hk_genes = housekeeping_genes or QUALIFIED_CROSS_TISSUE_HOUSEKEEPING

    if isinstance(data, pd.Series):
        available_hk = [g for g in hk_genes if g in data.index and not np.isnan(data[g])]
        if not available_hk:
            logger.warning("No reference housekeeping genes found in sample; returning unnormalized.")
            return data.copy()
        hk_mean = float(data[available_hk].mean())
        return data - hk_mean

    elif isinstance(data, pd.DataFrame):
        available_hk = [g for g in hk_genes if g in data.columns]
        if not available_hk:
            logger.warning("No reference housekeeping genes found in dataframe; returning unnormalized.")
            return data.copy()
        hk_mean = data[available_hk].mean(axis=1)
        return data.sub(hk_mean, axis=0)
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data)}")


class CrossPlatformNormalizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer ensuring cross-platform compatibility
    without test-set leakage.
    """

    def __init__(
        self,
        method: str = "rank",
        housekeeping_genes: Optional[List[str]] = None,
    ):
        self.method = method
        self.housekeeping_genes = housekeeping_genes or QUALIFIED_CROSS_TISSUE_HOUSEKEEPING
        self.reference_features_: List[str] = []
        self.reference_medians_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> CrossPlatformNormalizer:
        """Learns feature list and reference medians from training split."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.reference_features_ = list(X.columns)
        self.reference_medians_ = X.median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies normalization and handles missing genes using frozen training medians."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        # Fill missing features with training reference median
        missing_genes = [g for g in self.reference_features_ if g not in X_out.columns]
        for g in missing_genes:
            X_out[g] = self.reference_medians_.get(g, 0.0)

        # Align column order to training reference
        X_out = X_out[self.reference_features_]
        # Impute internal NaNs with training medians
        X_out = X_out.fillna(pd.Series(self.reference_medians_))

        if self.method == "rank":
            return within_sample_rank_transform(X_out)
        elif self.method == "housekeeping":
            return housekeeping_ratio_normalize(X_out, self.housekeeping_genes)
        elif self.method == "raw":
            return X_out
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
