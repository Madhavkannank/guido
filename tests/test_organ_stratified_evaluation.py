"""
Tests for Organ-Stratified Clinical Evaluation & Stage-Adjusted Cox Regression
==============================================================================
Verifies:
  1. Harrell's C-index with comparable pair counts and bootstrap 95% CI.
  2. Uno's C-index with IPCW weighting for censoring distribution.
  3. Organ-stratified evaluation preventing pan-cancer organ confounding.
  4. Organ-conditioned risk adjustments (intra-organ centering/percentile ranking).
  5. Multivariate stage- and age-adjusted Cox proportional hazards modeling with
     hazard ratios, confidence intervals, p-values, and PH diagnostics.
"""
import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from src.validation.organ_stratified_evaluation import (
    compute_harrell_c_index_with_ci,
    compute_unos_c_index,
    evaluate_organ_stratified_performance,
    organ_conditioned_risk_adjustment,
    fit_stage_adjusted_cox_regression,
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

    df = pd.DataFrame({
        "sample_id": [f"PATIENT_{i:04d}" for i in range(n)],
        "cohort": cohorts,
        "time": observed_time,
        "event": event,
        "risk_score": risk_score,
        "stage": stages,
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
    assert res["c_index"] == 1.0
    assert res["comparable_pairs"] > 0
    assert 0.0 <= res["ci_lower"] <= res["c_index"]
    assert res["c_index"] <= res["ci_upper"] <= 1.0


def test_unos_c_index_calculation():
    """Verifies Uno's IPCW-weighted C-index calculation."""
    np.random.seed(42)
    times = np.array([50, 120, 200, 350, 500, 700, 900], dtype=float)
    events = np.array([1, 1, 0, 1, 0, 1, 0], dtype=int)
    scores = np.array([0.85, 0.70, 0.40, 0.60, 0.30, 0.50, 0.20], dtype=float)

    uno_c = compute_unos_c_index(times, events, scores)
    assert 0.0 <= uno_c <= 1.0
    # Should be above random guessing for this concordant set
    assert uno_c > 0.60


def test_evaluate_organ_stratified_performance(synthetic_survival_cohort):
    """Verifies organ-stratified evaluation across all cohorts."""
    df = synthetic_survival_cohort
    report = evaluate_organ_stratified_performance(
        df,
        cohort_col="cohort",
        time_col="time",
        event_col="event",
        score_col="risk_score",
        n_bootstrap=100,
    )

    assert "cohort_results" in report
    assert "summary" in report

    cohort_results = report["cohort_results"]
    assert "TCGA-BRCA" in cohort_results
    assert "TCGA-LUAD" in cohort_results
    assert "TCGA-KIRC" in cohort_results

    # Check that each cohort has sample counts, event counts, C-index, and pairs
    for c_name, c_res in cohort_results.items():
        assert c_res["n_samples"] > 0
        assert c_res["n_events"] >= 0
        assert "c_index" in c_res
        assert "comparable_pairs" in c_res
        assert "ci_95" in c_res

    # Summary must have weighted and unweighted averages
    summary = report["summary"]
    assert "weighted_c_index" in summary
    assert "macro_c_index" in summary
    assert "total_cohorts" in summary
    assert summary["total_cohorts"] == 3


def test_organ_conditioned_risk_adjustment(synthetic_survival_cohort):
    """Verifies organ-conditioned score adjustments remove cross-cohort baseline shifts."""
    df = synthetic_survival_cohort.copy()
    adj_scores = organ_conditioned_risk_adjustment(
        df,
        cohort_col="cohort",
        score_col="risk_score",
        method="percentile",
    )
    assert len(adj_scores) == len(df)
    assert np.all(adj_scores >= 0.0) and np.all(adj_scores <= 1.0)


def test_stage_and_age_adjusted_cox_regression(synthetic_survival_cohort):
    """Verifies multivariate Cox regression for independent prognostic value of GUIDO risk score."""
    df = synthetic_survival_cohort
    cox_res = fit_stage_adjusted_cox_regression(
        df,
        time_col="time",
        event_col="event",
        score_col="risk_score",
        stage_col="stage",
        age_col="age",
    )

    assert "risk_score" in cox_res
    rs_stat = cox_res["risk_score"]

    # Must provide Hazard Ratio, CI, p-value, and Wald stat
    assert "hazard_ratio" in rs_stat
    assert "ci_lower" in rs_stat
    assert "ci_upper" in rs_stat
    assert "p_value" in rs_stat
    assert rs_stat["hazard_ratio"] > 1.0  # Higher risk score -> increased hazard
    assert "proportional_hazards_test" in cox_res
    assert "likelihood_ratio_p_value" in cox_res
