"""
Tests for Organ-Stratified Clinical Evaluation & Stage-Adjusted Cox Regression
==============================================================================
Verifies:
  1. Harrell's C-index with comparable pair counts and patient-level cluster bootstrap 95% CI.
  2. Uno's C-index with IPCW weighting for censoring distribution.
  3. Unevaluable cohort handling (returns None and unevaluable status, not C-index 0.5).
  4. Benjamini-Hochberg FDR multiple testing adjustment.
  5. Organ-stratified evaluation across multi-organ cohorts.
  6. Deployable Organ-Conditioned Risk Adjuster (frozen quantiles for N=1 clinical inference).
  7. Multivariate stage- and age-adjusted cohort-stratified Cox proportional hazards modeling
     with Likelihood Ratio Test (LRT) and Grambsch-Therneau PH diagnostics.
  8. TCGA cohort availability audit cataloging all 33 official cohorts.
"""
import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from src.validation.organ_stratified_evaluation import (
    ALL_33_TCGA_COHORTS,
    OrganConditionedRiskAdjuster,
    adjust_p_values_benjamini_hochberg,
    audit_tcga_cohort_availability,
    compute_harrell_c_index_with_ci,
    compute_unos_c_index,
    evaluate_organ_stratified_performance,
    fit_stage_adjusted_cox_regression,
    organ_conditioned_risk_adjustment,
)


@pytest.fixture
def synthetic_survival_cohort():
    """Generates a synthetic multi-organ survival dataset with clinical covariates."""
    np.random.seed(42)
    n = 300
    cohorts = np.random.choice(["TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC"], size=n)

    # Generate true prognostic biomarker
    gene_signal = np.random.normal(0, 1, size=n)
    stages = np.random.choice([1, 2, 3, 4], size=n, p=[0.4, 0.3, 0.2, 0.1])
    ages = np.random.normal(60, 10, size=n).clip(30, 85)

    # Organ baseline hazard: KIRC has lower hazard, LUAD has higher hazard
    organ_hazard = {"TCGA-BRCA": 0.0, "TCGA-LUAD": 0.5, "TCGA-KIRC": -0.5}
    hazard = np.array([organ_hazard[c] for c in cohorts]) + 0.6 * gene_signal + 0.4 * (stages - 1)

    # Generate exponential survival time
    base_time = np.random.exponential(scale=1000.0, size=n)
    true_time = base_time / np.exp(hazard)

    # Censoring time
    cens_time = np.random.uniform(200, 2500, size=n)

    observed_time = np.minimum(true_time, cens_time).clip(min=1.0)
    event = (true_time <= cens_time).astype(int)

    # Model risk score: strongly correlated with gene signal + noise
    risk_score = 1.0 / (1.0 + np.exp(-(0.8 * gene_signal + np.random.normal(0, 0.3, size=n))))

    patient_ids = [f"PATIENT_{i // 2:04d}" for i in range(n)]  # 2 samples per patient

    df = pd.DataFrame({
        "sample_id": [f"SAMPLE_{i:04d}" for i in range(n)],
        "patient_id": patient_ids,
        "cohort": cohorts,
        "time": observed_time,
        "event": event,
        "risk_score": risk_score,
        "stage": [f"Stage {s}" for s in stages],
        "age": ages,
    })
    return df


def test_harrell_c_index_with_ci_and_pairs():
    """Verifies Harrell's C-index calculation, pair counts, and confidence intervals."""
    times = np.array([100, 200, 300, 400, 500], dtype=float)
    events = np.array([1, 1, 0, 1, 0], dtype=int)
    # Higher risk score -> shorter survival time
    scores = np.array([0.9, 0.8, 0.5, 0.3, 0.1], dtype=float)

    res = compute_harrell_c_index_with_ci(times, events, scores, n_bootstrap=100, seed=42)
    assert res["status"] == "evaluable"
    assert res["c_index"] == 1.0
    assert res["comparable_pairs"] > 0
    assert 0.0 <= res["ci_lower"] <= res["c_index"]
    assert res["c_index"] <= res["ci_upper"] <= 1.0


def test_patient_level_cluster_bootstrap():
    """Verifies cluster bootstrap at patient level when patient_ids are supplied."""
    times = np.array([100, 120, 200, 210, 400, 420], dtype=float)
    events = np.array([1, 1, 1, 1, 0, 0], dtype=int)
    scores = np.array([0.9, 0.85, 0.6, 0.55, 0.2, 0.15], dtype=float)
    patients = np.array(["P1", "P1", "P2", "P2", "P3", "P3"])

    res = compute_harrell_c_index_with_ci(
        times, events, scores,
        patient_ids=patients,
        n_bootstrap=50,
        seed=42,
    )
    assert res["status"] == "evaluable"
    assert res["c_index"] == 1.0
    assert res["ci_lower"] is not None
    assert res["ci_upper"] is not None


def test_unevaluable_cohort_returns_none_not_half():
    """Verifies that cohorts with 0 events or no comparable pairs return None and status='unevaluable'."""
    times = np.array([100, 200, 300], dtype=float)
    events = np.array([0, 0, 0], dtype=int)  # Zero events
    scores = np.array([0.9, 0.5, 0.2], dtype=float)

    res = compute_harrell_c_index_with_ci(times, events, scores)
    assert res["status"] == "unevaluable"
    assert res["c_index"] is None
    assert res["comparable_pairs"] == 0
    assert res["ci_lower"] is None
    assert res["ci_upper"] is None

    # Uno C must also be None
    uno_c = compute_unos_c_index(times, events, scores)
    assert uno_c is None


def test_unos_c_index_calculation():
    """Verifies Uno's IPCW-weighted C-index calculation."""
    np.random.seed(42)
    times = np.array([50, 120, 200, 350, 500, 700, 900], dtype=float)
    events = np.array([1, 1, 0, 1, 0, 1, 0], dtype=int)
    scores = np.array([0.85, 0.70, 0.40, 0.60, 0.30, 0.50, 0.20], dtype=float)

    uno_c = compute_unos_c_index(times, events, scores)
    assert uno_c is not None
    assert 0.0 <= uno_c <= 1.0
    assert uno_c > 0.60


def test_benjamini_hochberg_fdr():
    """Verifies Benjamini-Hochberg FDR q-value calculation."""
    raw_p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]
    q_vals = adjust_p_values_benjamini_hochberg(raw_p)
    assert len(q_vals) == len(raw_p)
    # q-values must be >= p-values
    assert np.all(q_vals >= np.array(raw_p))
    # q-values must be monotonically increasing for sorted p
    assert np.all(np.diff(q_vals) >= -1e-8)


def test_evaluate_organ_stratified_performance(synthetic_survival_cohort):
    """Verifies organ-stratified evaluation across all cohorts including FDR correction."""
    df = synthetic_survival_cohort
    report = evaluate_organ_stratified_performance(
        df,
        cohort_col="cohort",
        time_col="time",
        event_col="event",
        score_col="risk_score",
        patient_col="patient_id",
        n_bootstrap=50,
    )

    assert "cohort_results" in report
    assert "summary" in report

    cohort_results = report["cohort_results"]
    assert "TCGA-BRCA" in cohort_results
    assert "TCGA-LUAD" in cohort_results
    assert "TCGA-KIRC" in cohort_results

    for c_name, c_res in cohort_results.items():
        assert c_res["n_samples"] > 0
        assert c_res["n_events"] >= 0
        if c_res["status"] == "evaluable":
            assert c_res["c_index"] is not None
            assert c_res["comparable_pairs"] > 0
            assert "fdr_q_value" in c_res
            assert c_res["fdr_q_value"] is not None

    summary = report["summary"]
    assert summary["total_cohorts_checked"] == 3
    assert summary["evaluable_cohorts_count"] > 0
    assert summary["weighted_intra_cohort_c_index"] is not None
    assert "scientific_limitation" in summary


def test_deployable_organ_conditioned_adjuster(synthetic_survival_cohort):
    """Verifies OrganConditionedRiskAdjuster fits frozen distributions for N=1 clinical scoring."""
    df = synthetic_survival_cohort
    adjuster = OrganConditionedRiskAdjuster(method="percentile")
    adjuster.fit(df, cohort_col="cohort", score_col="risk_score")

    assert adjuster.is_fitted_
    assert "TCGA-BRCA" in adjuster.reference_distributions_
    brca_dist = adjuster.reference_distributions_["TCGA-BRCA"]
    assert "median" in brca_dist
    assert "q_50" in brca_dist

    # Deterministic single-patient (N=1) inference
    single_pct = adjuster.transform_single(score=0.75, cohort="TCGA-BRCA")
    assert 0.0 <= single_pct <= 1.0

    # Batch transform
    batch_adj = adjuster.transform(df, cohort_col="cohort", score_col="risk_score")
    assert len(batch_adj) == len(df)
    assert np.all(batch_adj >= 0.0) and np.all(batch_adj <= 1.0)


def test_stage_and_age_adjusted_cox_with_strata(synthetic_survival_cohort):
    """Verifies multivariate Cox regression with cohort strata and LRT test."""
    df = synthetic_survival_cohort
    cox_res = fit_stage_adjusted_cox_regression(
        df,
        time_col="time",
        event_col="event",
        score_col="risk_score",
        stage_col="stage",
        age_col="age",
        strata_col="cohort",
    )

    assert cox_res["status"] == "evaluable"
    assert cox_res["strata_used"] == "cohort"
    assert "risk_score" in cox_res

    rs = cox_res["risk_score"]
    assert rs["hazard_ratio"] > 1.0
    assert rs["p_value"] < 0.05

    # Likelihood Ratio Test comparing vs clinical-only model
    lrt = cox_res["likelihood_ratio_test"]
    assert lrt["df"] == 1
    assert "chi2_stat" in lrt
    assert "p_value" in lrt

    # Proportional Hazards assumption test
    ph = cox_res["proportional_hazards_test"]
    assert "score_p_value" in ph
    assert "assumption_met" in ph
    assert "covariate_diagnostics" in ph


def test_tcga_cohort_availability_registry():
    """Verifies that all 33 official TCGA cohorts are cataloged and audited."""
    res = audit_tcga_cohort_availability("data")
    assert res["total_expected"] == 33
    assert len(res["expected_33_tcga_cohorts"]) == 33
    assert "locally_available_genomic_cohorts" in res
    assert "missing_genomic_cohorts" in res
    assert res["locally_available_count"] + res["missing_count"] == 33
    assert "reporting_integrity_notice" in res
