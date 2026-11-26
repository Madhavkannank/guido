"""
Tests for Cross-Platform Normalization & Clinical Sample Quality Control (QC)
=============================================================================
Verifies:
  1. Within-sample percentile rank transformation invariance across platform shifts.
  2. Housekeeping gene ratio normalization.
  3. Handling of platform-specific missing genes without data leakage.
  4. Clinical sample QC gating (DV200 thresholds, dropout rate, pass/warn/fail).
  5. Degraded RNA simulation and model risk stability measurement.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.platform_normalization import (
    CrossPlatformNormalizer,
    within_sample_rank_transform,
    housekeeping_ratio_normalize,
)
from src.preprocessing.sample_qc import (
    ClinicalSampleQualityGate,
    QCResult,
    QCStatus,
    simulate_ffpe_degradation,
)


@pytest.fixture
def sample_patient_expression():
    """Generates a realistic single-patient gene expression profile with housekeeping genes."""
    np.random.seed(42)
    genes = [
        "TP53", "MKI67", "CD274", "VEGFA", "EGFR", "MYC", "PTEN", "BRCA1",
        "GAPDH", "ACTB", "PGK1", "B2M", "HPRT1"
    ]
    # Expression in log2(count + 1)
    values = [7.2, 5.1, 4.3, 8.0, 6.5, 9.1, 5.8, 4.0, 12.5, 13.0, 11.8, 12.0, 10.5]
    return pd.Series(values, index=genes)


def test_within_sample_rank_transform_invariance(sample_patient_expression):
    """Verifies that within-sample rank transform is invariant to platform gain/offset."""
    profile_orig = sample_patient_expression.copy()

    # Simulate platform shift: microarray vs RNA-seq (scale factor + offset)
    profile_shifted = profile_orig * 1.8 + 4.5

    ranks_orig = within_sample_rank_transform(profile_orig)
    ranks_shifted = within_sample_rank_transform(profile_shifted)

    # Ranks must be identical despite platform scale/offset shift
    np.testing.assert_allclose(ranks_orig.values, ranks_shifted.values, atol=1e-5)
    assert np.all(ranks_orig.values >= 0.0)
    assert np.all(ranks_orig.values <= 1.0)


def test_housekeeping_ratio_normalize(sample_patient_expression):
    """Verifies normalization against geometric mean of reference housekeeping genes."""
    profile = sample_patient_expression.copy()
    norm_profile = housekeeping_ratio_normalize(
        profile,
        housekeeping_genes=["GAPDH", "ACTB", "PGK1"]
    )

    assert "TP53" in norm_profile
    assert "MKI67" in norm_profile
    # Shift entire profile by +3 (global gain shift)
    profile_gain = profile + 3.0
    norm_gain = housekeeping_ratio_normalize(
        profile_gain,
        housekeeping_genes=["GAPDH", "ACTB", "PGK1"]
    )
    # The relative difference from housekeeping mean is invariant
    np.testing.assert_allclose(norm_profile.values, norm_gain.values, atol=1e-5)


def test_cross_platform_normalizer_pipeline(sample_patient_expression):
    """Verifies CrossPlatformNormalizer fit/transform and missing gene handling."""
    df = pd.DataFrame([sample_patient_expression.to_dict(), sample_patient_expression.to_dict()])
    normalizer = CrossPlatformNormalizer(method="rank")
    normalizer.fit(df)

    # Transform with a missing gene
    test_sample = sample_patient_expression.drop("BRCA1").to_frame().T
    transformed = normalizer.transform(test_sample)

    assert "BRCA1" in transformed.columns  # Imputed with reference median
    assert not transformed.isna().any().any()


def test_sample_qc_pass_on_high_quality(sample_patient_expression):
    """Verifies that high-quality fresh frozen sample passes QC."""
    gate = ClinicalSampleQualityGate(min_dv200=50.0, max_zero_fraction=0.30)
    res = gate.evaluate_sample_quality(
        sample_patient_expression,
        dv200=78.5,
        library_depth=25_000_000,
    )

    assert isinstance(res, QCResult)
    assert res.status == QCStatus.PASS
    assert res.quality_score >= 80.0
    assert len(res.rejection_reasons) == 0


def test_sample_qc_warn_on_moderate_ffpe(sample_patient_expression):
    """Verifies warning status for moderate FFPE degradation."""
    gate = ClinicalSampleQualityGate(min_dv200=50.0, warn_dv200=30.0)
    res = gate.evaluate_sample_quality(
        sample_patient_expression,
        dv200=38.0,  # Between 30 and 50
    )

    assert res.status == QCStatus.WARN
    assert len(res.warnings) > 0
    assert any("DV200" in w for w in res.warnings)


def test_sample_qc_reject_on_severe_degradation(sample_patient_expression):
    """Verifies explicit rejection when sample fails critical QC threshold."""
    gate = ClinicalSampleQualityGate(min_dv200=50.0, warn_dv200=30.0)
    res = gate.evaluate_sample_quality(
        sample_patient_expression,
        dv200=18.0,  # Below 30% -> Rejection
    )

    assert res.status == QCStatus.FAIL
    assert len(res.rejection_reasons) > 0
    assert any("DV200" in r for r in res.rejection_reasons)


def test_sample_qc_reject_on_high_dropout():
    """Verifies rejection when too many genes have zero or missing expression."""
    gate = ClinicalSampleQualityGate(max_zero_fraction=0.40)
    # Series with 70% zeros
    bad_sample = pd.Series([0.0] * 7 + [5.0] * 3, index=[f"G_{i}" for i in range(10)])
    res = gate.evaluate_sample_quality(bad_sample)

    assert res.status == QCStatus.FAIL
    assert any("dropout" in r.lower() or "zero" in r.lower() for r in res.rejection_reasons)


def test_ffpe_degradation_simulation(sample_patient_expression):
    """Verifies FFPE degradation simulator produces expected signal attenuation."""
    orig = sample_patient_expression.copy()
    degraded = simulate_ffpe_degradation(orig, dv200=25.0, seed=42)

    assert len(degraded) == len(orig)
    # Degraded sample should have lower total expression / higher variance
    assert degraded.mean() < orig.mean()
