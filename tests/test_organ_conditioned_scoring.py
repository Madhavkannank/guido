"""
Tests for Deployable Organ-Conditioned Clinical Risk Scorer
===========================================================
Verifies:
  1. Zero Transductive Leakage: Patient scores depend strictly on frozen training references.
  2. N=1 Inference Parity: Single-patient prediction equals batch prediction identically.
  3. Batch Independence: Composition/ordering of test batch does not alter individual scores.
  4. Monotonicity: Higher risk scores yield monotonically non-decreasing percentiles.
  5. Fallback Mechanics: Unknown organ types trigger transparent global reference fallback.
  6. Serialization Fidelity: Scorer can be saved to/loaded from JSON without numerical drift.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.organ_conditioned_scoring import (
    DeployableOrganScorer,
    OrganReferenceDistribution,
)


@pytest.fixture
def training_cohort_scores():
    """Generates synthetic multi-organ training dataset with distinct organ baseline distributions."""
    np.random.seed(42)
    n = 1000
    cohorts = np.random.choice(["TCGA-BRCA", "TCGA-KIRC", "TCGA-LUAD"], size=n)

    # Different baseline risk distributions per organ
    base_scores = []
    for c in cohorts:
        if c == "TCGA-BRCA":
            base_scores.append(np.random.beta(2, 5))  # Lower baseline risk
        elif c == "TCGA-KIRC":
            base_scores.append(np.random.beta(3, 3))  # Intermediate baseline risk
        else:
            base_scores.append(np.random.beta(5, 2))  # Higher baseline risk (LUAD)

    df_train = pd.DataFrame({
        "cohort": cohorts,
        "risk_score": base_scores,
    })
    return df_train


def test_scorer_fitting_and_quantiles(training_cohort_scores):
    """Verifies that scorer computes frozen quantiles, medians, and IQRs per organ and globally."""
    scorer = DeployableOrganScorer(method="percentile")
    scorer.fit(training_cohort_scores, cohort_col="cohort", score_col="risk_score")

    assert scorer.is_fitted
    assert len(scorer.organ_references) == 3
    assert "TCGA-BRCA" in scorer.organ_references
    assert "TCGA-KIRC" in scorer.organ_references
    assert "TCGA-LUAD" in scorer.organ_references
    assert scorer.global_reference is not None

    brca_ref = scorer.organ_references["TCGA-BRCA"]
    assert isinstance(brca_ref, OrganReferenceDistribution)
    assert brca_ref.min_score < brca_ref.median < brca_ref.max_score
    assert len(brca_ref.quantiles) > 10


def test_n_equals_1_inference_parity(training_cohort_scores):
    """Verifies that scoring a single patient (N=1) matches batch scoring identically."""
    scorer = DeployableOrganScorer(method="percentile")
    scorer.fit(training_cohort_scores, cohort_col="cohort", score_col="risk_score")

    test_patient_score = 0.45
    test_patient_cohort = "TCGA-BRCA"

    # Single patient scoring
    single_res = scorer.score_single(raw_score=test_patient_score, organ_cohort=test_patient_cohort)

    # Batch containing that patient and 20 other diverse patients
    np.random.seed(99)
    other_cohorts = np.random.choice(["TCGA-BRCA", "TCGA-KIRC", "TCGA-LUAD"], size=20)
    other_scores = np.random.uniform(0.05, 0.95, size=20)

    batch_df = pd.DataFrame({
        "cohort": [test_patient_cohort] + list(other_cohorts),
        "risk_score": [test_patient_score] + list(other_scores),
    })

    batch_res = scorer.score_batch(batch_df, cohort_col="cohort", score_col="risk_score")

    # Patient in position 0 of batch must match single patient prediction exactly
    p0 = batch_res.iloc[0]
    assert p0["raw_score"] == single_res["raw_score"]
    assert p0["organ_conditioned_score"] == single_res["organ_conditioned_score"]
    assert p0["organ_percentile"] == single_res["organ_percentile"]
    assert p0["is_fallback"] == single_res["is_fallback"]


def test_batch_independence_against_transductive_leakage(training_cohort_scores):
    """Verifies that adding extreme outliers to the test batch does NOT shift an existing patient's score."""
    scorer = DeployableOrganScorer(method="percentile")
    scorer.fit(training_cohort_scores, cohort_col="cohort", score_col="risk_score")

    target_score = 0.50
    target_cohort = "TCGA-KIRC"

    # Baseline score in benign batch
    batch_benign = pd.DataFrame({
        "cohort": [target_cohort, "TCGA-KIRC", "TCGA-KIRC"],
        "risk_score": [target_score, 0.48, 0.52],
    })
    res_benign = scorer.score_batch(batch_benign).iloc[0]

    # Score in contaminated batch containing extreme outliers
    batch_contaminated = pd.DataFrame({
        "cohort": [target_cohort, "TCGA-KIRC", "TCGA-KIRC", "TCGA-KIRC"],
        "risk_score": [target_score, 0.001, 0.002, 0.999],
    })
    res_contaminated = scorer.score_batch(batch_contaminated).iloc[0]

    # Scores must be 100% identical because reference distributions are frozen
    assert res_benign["organ_conditioned_score"] == res_contaminated["organ_conditioned_score"]
    assert res_benign["organ_percentile"] == res_contaminated["organ_percentile"]


def test_monotonicity_of_percentiles(training_cohort_scores):
    """Verifies that percentiles are strictly non-decreasing with increasing risk score."""
    scorer = DeployableOrganScorer(method="percentile")
    scorer.fit(training_cohort_scores, cohort_col="cohort", score_col="risk_score")

    scores_grid = np.linspace(0.05, 0.95, 50)
    percentiles = [
        scorer.score_single(s, organ_cohort="TCGA-BRCA")["organ_percentile"]
        for s in scores_grid
    ]

    diffs = np.diff(percentiles)
    # Monotonic: no negative step
    assert np.all(diffs >= 0.0)


def test_unknown_organ_fallback(training_cohort_scores):
    """Verifies that presenting an unseen organ triggers fallback to global reference with explicit reason."""
    scorer = DeployableOrganScorer(method="percentile")
    scorer.fit(training_cohort_scores, cohort_col="cohort", score_col="risk_score")

    res = scorer.score_single(raw_score=0.60, organ_cohort="TCGA-CHOL")

    assert res["is_fallback"] is True
    assert res["fallback_reason"] is not None
    assert "Unknown organ" in res["fallback_reason"]
    assert res["reference_used"] == "GLOBAL_PAN_CANCER"
    assert 0.0 <= res["organ_percentile"] <= 100.0


def test_serialization_fidelity(training_cohort_scores, tmp_path):
    """Verifies that save_json and load_json preserve exact numerical outputs."""
    scorer = DeployableOrganScorer(method="percentile")
    scorer.fit(training_cohort_scores, cohort_col="cohort", score_col="risk_score")

    json_path = tmp_path / "organ_reference_distributions.json"
    scorer.save_json(json_path)

    assert json_path.exists()

    loaded_scorer = DeployableOrganScorer.load_json(json_path)
    assert loaded_scorer.is_fitted
    assert len(loaded_scorer.organ_references) == len(scorer.organ_references)

    score_orig = scorer.score_single(0.55, "TCGA-LUAD")
    score_loaded = loaded_scorer.score_single(0.55, "TCGA-LUAD")

    assert score_orig == score_loaded
