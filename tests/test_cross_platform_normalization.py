"""
Tests for Cross-Platform Normalization, Gene Harmonization & Clinical QC
========================================================================
Verifies:
  1. Housekeeping gene stability qualification across multiple cohorts (mean, std, CV, inter-cohort divergence).
  2. Qualification tiers (e.g. RPL13A, ACTB as stable vs B2M, HPRT1 as tissue-variable).
  3. Gene symbol harmonization handling canonical symbols, aliases, multi-mapping, and unmapped targets.
  4. Explicit tracking of unmeasured genes vs biological true zero expression.
  5. Honest missingness handling in Clinical Sample QC (absent DV200 is unmeasured, never fabricated).
  6. Within-sample percentile rank transformation invariance to platform shifts.
  7. CrossPlatformNormalizer fit/transform with zero-leakage training medians.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.cross_platform_normalizer import (
    QUALIFIED_CROSS_TISSUE_HOUSEKEEPING,
    TISSUE_VARIABLE_HOUSEKEEPING_WARNINGS,
    ClinicalQCResult,
    ClinicalQCStatus,
    ClinicalSampleQualityGate,
    CrossPlatformNormalizer,
    HarmonizationReport,
    HousekeepingGeneReport,
    harmonize_gene_symbols,
    housekeeping_ratio_normalize,
    profile_expression_measurement_status,
    qualify_housekeeping_genes,
    within_sample_rank_transform,
)


@pytest.fixture
def multi_cohort_expression_data():
    """Generates synthetic expression matrix across 3 cancer cohorts with distinct HK stability."""
    np.random.seed(42)
    n = 150
    cohorts = ["TCGA-COHORT_A"] * 50 + ["TCGA-COHORT_B"] * 50 + ["TCGA-COHORT_C"] * 50

    # RPL13A: stable across cohorts (mean ~13.8)
    rpl13a = np.random.normal(13.8, 0.4, size=n)
    # ACTB: stable across cohorts (mean ~16.4)
    actb = np.random.normal(16.4, 0.5, size=n)
    # HPRT1: unstable between cohorts (11.0 in A, 9.0 in B, 9.1 in C)
    hprt1 = np.concatenate([
        np.random.normal(11.0, 0.5, size=50),
        np.random.normal(9.0, 0.5, size=50),
        np.random.normal(9.1, 0.5, size=50),
    ])
    # B2M: unstable immune-regulated (16.2 in A, 16.3 in B, 14.8 in C)
    b2m = np.concatenate([
        np.random.normal(16.2, 0.6, size=50),
        np.random.normal(16.3, 0.6, size=50),
        np.random.normal(14.8, 0.6, size=50),
    ])

    df = pd.DataFrame({
        "RPL13A": rpl13a,
        "ACTB": actb,
        "HPRT1": hprt1,
        "B2M": b2m,
        "TP53": np.random.normal(7.5, 1.2, size=n),
        "MKI67": np.random.normal(5.0, 1.5, size=n),
    })
    cohort_s = pd.Series(cohorts, index=df.index)
    return df, cohort_s


def test_housekeeping_gene_qualification(multi_cohort_expression_data):
    """Verifies that housekeeping gene qualification computes CV and flags tissue-variable genes."""
    df, cohorts = multi_cohort_expression_data
    reports = qualify_housekeeping_genes(df, cohort_series=cohorts)

    assert "RPL13A" in reports
    assert "ACTB" in reports
    assert "HPRT1" in reports
    assert "B2M" in reports

    rpl_rep = reports["RPL13A"]
    assert isinstance(rpl_rep, HousekeepingGeneReport)
    assert rpl_rep.global_cv < 0.05
    assert rpl_rep.max_cohort_diff < 0.50
    assert rpl_rep.stability_tier == "QUALIFIED_STABLE"

    hprt_rep = reports["HPRT1"]
    assert hprt_rep.max_cohort_diff > 1.5  # Large inter-cohort divergence
    assert hprt_rep.stability_tier == "DISQUALIFIED_TISSUE_VARIABLE"


def test_gene_symbol_harmonization():
    """Verifies resolution of aliases, multi-mapping detection, and unmapped tracking."""
    raw_genes = ["HER2", "p53", "PD-L1", "Ki-67", "H1", "UNKNOWN_GENE_XYZ", "GAPDH"]
    harmonized, report = harmonize_gene_symbols(
        raw_genes,
        reference_universe={"ERBB2", "TP53", "CD274", "MKI67", "GAPDH", "H1-0", "H1-1"}
    )

    assert isinstance(report, HarmonizationReport)
    assert "ERBB2" in harmonized
    assert "TP53" in harmonized
    assert "CD274" in harmonized
    assert "MKI67" in harmonized

    # Check multi-mapping of H1
    assert "H1" in report.multi_mapped_genes
    assert len(report.multi_mapped_genes["H1"]) > 1

    # Check unmapped gene
    assert "UNKNOWN_GENE_XYZ" in report.unmapped_genes
    assert report.unmapped_count == 1


def test_unmeasured_vs_true_zero_distinction():
    """Verifies that missing/unmeasured targets are not conflated with biological zero expression."""
    sample = pd.Series({
        "TP53": 8.5,
        "MKI67": 0.0,  # Measured zero (below detection limit)
        "EGFR": 6.2,
    })
    manifest = ["TP53", "MKI67", "EGFR", "CD274", "BRCA1"]  # CD274 and BRCA1 are unmeasured

    full_sample, prov = profile_expression_measurement_status(sample, panel_manifest=manifest)

    assert prov.total_genes == 5
    assert prov.measured_genes_count == 3
    assert prov.unmeasured_genes_count == 2
    assert prov.true_zero_count == 1
    assert "CD274" in prov.unmeasured_gene_list
    assert "BRCA1" in prov.unmeasured_gene_list
    # The measured zero (MKI67) must NOT be in unmeasured list
    assert "MKI67" not in prov.unmeasured_gene_list


def test_clinical_qc_honest_missingness_no_fabricated_dv200():
    """Verifies that absent DV200 and library depth are explicitly recorded as UNMEASURED, not fabricated."""
    gate = ClinicalSampleQualityGate(min_dv200=50.0)
    sample = pd.Series({
        "RPL13A": 13.5,
        "ACTB": 16.2,
        "PPIA": 12.1,
        "TBP": 7.8,
        "TP53": 8.0,
    })

    # Specimen without DV200 measurement
    qc_res = gate.evaluate_sample_quality(
        sample=sample,
        dv200=None,
        library_depth=None,
        specimen_type="FRESH_FROZEN",
    )

    assert isinstance(qc_res, ClinicalQCResult)
    assert qc_res.dv200 is None
    assert qc_res.dv200_status == "UNMEASURED_NOT_RECORDED"
    assert qc_res.library_depth_status == "UNMEASURED_NOT_RECORDED"
    assert any("absent/unrecorded" in p for p in qc_res.provenance_notes)
    # Without artificial fabrication, the sample can be clinically actionable with fresh-frozen notice
    assert qc_res.is_actionable()


def test_within_sample_rank_transform_invariance():
    """Verifies rank transformation invariance to platform gain and scale shifts."""
    profile_orig = pd.Series({
        "TP53": 7.0, "MKI67": 5.0, "CD274": 4.0, "VEGFA": 8.0, "EGFR": 6.5
    })
    # Simulated platform shift: 2.5x gain + 3.0 background offset
    profile_shifted = profile_orig * 2.5 + 3.0

    ranks_orig = within_sample_rank_transform(profile_orig)
    ranks_shifted = within_sample_rank_transform(profile_shifted)

    np.testing.assert_allclose(ranks_orig.values, ranks_shifted.values, atol=1e-5)


def test_cross_platform_normalizer_zero_leakage():
    """Verifies CrossPlatformNormalizer handles missing genes using training reference medians."""
    train_df = pd.DataFrame([
        {"G1": 5.0, "G2": 10.0, "G3": 15.0},
        {"G1": 7.0, "G2": 12.0, "G3": 17.0},
    ])
    normalizer = CrossPlatformNormalizer(method="rank")
    normalizer.fit(train_df)

    assert normalizer.reference_medians_["G2"] == 11.0

    # Test sample missing G2
    test_sample = pd.DataFrame([{"G1": 6.0, "G3": 16.0}])
    transformed = normalizer.transform(test_sample)

    assert "G2" in transformed.columns
    assert not transformed.isna().any().any()
