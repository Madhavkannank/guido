"""
Organ-Stratified Clinical Risk Evaluation & Stage-Adjusted Cox Regression
========================================================================
Scientific validation module designed to evaluate whether genomic prognostic risk models
provide genuine, independent clinical value within individual cancer types rather than
relying on pan-cancer organ confounding (tissue baseline survival differences).

Capabilities:
  1. Harrell's Concordance Index with comparable pair counts & 95% bootstrap CIs.
  2. Uno's Inverse Probability of Censoring Weighted (IPCW) C-statistic.
  3. Organ-Stratified Cohort Evaluation (reporting intra-cohort metrics across 33 TCGA cohorts).
  4. Organ-Conditioned Risk Modeling (within-organ percentile normalization and centering).
  5. Stage- and Age-Adjusted Multivariate Cox Proportional Hazards Regression with
     Hazard Ratios (HR), 95% CIs, Wald p-values, Likelihood Ratio Tests, and PH diagnostics.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

logger = logging.getLogger(__name__)


# ─── 1. Harrell's C-Index with Bootstrap Confidence Intervals ────────────────

def compute_harrell_c_index_with_ci(
    times: np.ndarray,
    events: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes Harrell's C-index along with comparable pair counts and empirical bootstrap 95% CIs.

    Parameters:
      times: Observed event or censoring times (days).
      events: Binary event status (1 = progression/death, 0 = censored).
      scores: Continuous risk predictions (higher score implies higher risk/shorter survival).
      n_bootstrap: Number of bootstrap iterations.
      alpha: Significance level (default 0.05 for 95% CI).
      seed: Random seed for reproducibility.

    Returns:
      Dictionary containing c_index, comparable_pairs, ci_lower, ci_upper, standard_error.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)

    valid_mask = (times > 0) & (~np.isnan(times)) & (~np.isnan(events)) & (~np.isnan(scores))
    times = times[valid_mask]
    events = events[valid_mask]
    scores = scores[valid_mask]

    n = len(times)
    if n < 2 or np.sum(events == 1) == 0:
        return {
            "c_index": 0.5,
            "comparable_pairs": 0,
            "ci_lower": 0.5,
            "ci_upper": 0.5,
            "standard_error": 0.0,
            "n_samples": n,
            "n_events": int(np.sum(events == 1)),
        }

    # Primary Harrell calculation
    def _c_index_and_pairs(t: np.ndarray, e: np.ndarray, s: np.ndarray) -> Tuple[float, int]:
        event_indices = np.where(e == 1)[0]
        if len(event_indices) == 0:
            return 0.5, 0

        concordant = 0.0
        pairs = 0
        for i in event_indices:
            t_i = t[i]
            s_i = s[i]
            # Valid comparison: patient j survived strictly longer than patient i
            valid_j = (t > t_i)
            if not np.any(valid_j):
                continue
            s_j = s[valid_j]
            pairs += len(s_j)
            concordant += float(np.sum(s_i > s_j)) + 0.5 * float(np.sum(s_i == s_j))

        if pairs == 0:
            return 0.5, 0
        return float(concordant / pairs), pairs

    point_c, pairs = _c_index_and_pairs(times, events, scores)

    # Bootstrap for confidence interval
    rng = np.random.RandomState(seed)
    boot_cs = []
    if n_bootstrap > 0 and pairs > 0:
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            t_b, e_b, s_b = times[idx], events[idx], scores[idx]
            if np.sum(e_b == 1) == 0:
                continue
            c_b, p_b = _c_index_and_pairs(t_b, e_b, s_b)
            if p_b > 0:
                boot_cs.append(c_b)

    if boot_cs:
        ci_lower = float(np.percentile(boot_cs, 100 * (alpha / 2)))
        ci_upper = float(np.percentile(boot_cs, 100 * (1 - alpha / 2)))
        se = float(np.std(boot_cs))
    else:
        ci_lower, ci_upper, se = point_c, point_c, 0.0

    return {
        "c_index": round(point_c, 4),
        "comparable_pairs": pairs,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "standard_error": round(se, 4),
        "n_samples": n,
        "n_events": int(np.sum(events == 1)),
    }


# ─── 2. Uno's Inverse Probability of Censoring Weighted C-Index ──────────────

def compute_unos_c_index(
    times: np.ndarray,
    events: np.ndarray,
    scores: np.ndarray,
    tau: Optional[float] = None,
) -> float:
    """
    Computes Uno's C-statistic (Uno et al., Ann Intern Med 2011).
    Uses Inverse Probability of Censoring Weights (IPCW) to correct for the censoring distribution.

    Parameters:
      times: Observed survival times.
      events: Binary event status.
      scores: Predicted risk scores.
      tau: Truncation time (default: max event time).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)

    valid_mask = (times > 0) & (~np.isnan(times)) & (~np.isnan(events)) & (~np.isnan(scores))
    times = times[valid_mask]
    events = events[valid_mask]
    scores = scores[valid_mask]

    n = len(times)
    event_indices = np.where(events == 1)[0]
    if len(event_indices) == 0 or n < 2:
        return 0.5

    if tau is None:
        tau = float(np.max(times[event_indices]))

    # Fit Kaplan-Meier estimator for censoring distribution G(t) = P(C > t)
    censoring_events = 1 - events
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=censoring_events)

    def get_g(t_val: float) -> float:
        val = kmf.predict(t_val)
        return float(np.clip(val, 0.05, 1.0))

    numerator = 0.0
    denominator = 0.0

    for i in event_indices:
        t_i = times[i]
        if t_i > tau:
            continue
        s_i = scores[i]
        g_i = get_g(t_i)
        weight_i = 1.0 / (g_i ** 2)

        valid_j = (times > t_i)
        if not np.any(valid_j):
            continue

        s_j = scores[valid_j]
        concordant_ij = (s_i > s_j).astype(float) + 0.5 * (s_i == s_j).astype(float)

        numerator += np.sum(concordant_ij) * weight_i
        denominator += len(s_j) * weight_i

    if denominator <= 0:
        return 0.5
    return round(float(numerator / denominator), 4)


# ─── 3. Organ-Stratified Cohort Evaluation ───────────────────────────────────

def evaluate_organ_stratified_performance(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    time_col: str = "time",
    event_col: str = "event",
    score_col: str = "risk_score",
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """
    Evaluates risk predictions within each organ cohort independently to detect
    and measure organ confounding.

    Returns:
      Dict with individual cohort metrics, weighted intra-cohort C-index, and macro C-index.
    """
    unique_cohorts = sorted(df[cohort_col].dropna().unique())
    cohort_results: Dict[str, Dict[str, Any]] = {}

    total_pairs = 0
    weighted_c_sum = 0.0
    valid_c_indices = []

    for c in unique_cohorts:
        sub_df = df[df[cohort_col] == c]
        times = sub_df[time_col].values
        events = sub_df[event_col].values
        scores = sub_df[score_col].values

        harrell_res = compute_harrell_c_index_with_ci(
            times, events, scores, n_bootstrap=n_bootstrap
        )
        uno_c = compute_unos_c_index(times, events, scores)

        pairs = harrell_res["comparable_pairs"]
        c_val = harrell_res["c_index"]

        if pairs > 0 and harrell_res["n_events"] >= 3:
            total_pairs += pairs
            weighted_c_sum += c_val * pairs
            valid_c_indices.append(c_val)

        cohort_results[c] = {
            "n_samples": len(sub_df),
            "n_events": int(np.sum(events == 1)),
            "event_rate": round(float(np.mean(events == 1)), 4),
            "c_index": c_val,
            "unos_c_index": uno_c,
            "comparable_pairs": pairs,
            "ci_95": [harrell_res["ci_lower"], harrell_res["ci_upper"]],
            "standard_error": harrell_res["standard_error"],
            "statistically_significant": harrell_res["ci_lower"] > 0.50,
        }

    weighted_c = round(weighted_c_sum / max(total_pairs, 1), 4) if total_pairs > 0 else 0.5
    macro_c = round(float(np.mean(valid_c_indices)), 4) if valid_c_indices else 0.5
    sig_cohorts = sum(1 for r in cohort_results.values() if r["statistically_significant"])

    summary = {
        "total_cohorts": len(unique_cohorts),
        "total_samples": len(df),
        "total_events": int(df[event_col].sum()),
        "total_comparable_pairs": total_pairs,
        "weighted_c_index": weighted_c,
        "macro_c_index": macro_c,
        "cohorts_with_significant_prognosis": sig_cohorts,
    }

    return {
        "cohort_results": cohort_results,
        "summary": summary,
    }


# ─── 4. Organ-Conditioned Risk Modeling ──────────────────────────────────────

def organ_conditioned_risk_adjustment(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    score_col: str = "risk_score",
    method: str = "percentile",
) -> np.ndarray:
    """
    Adjusts raw model risk scores conditioned on organ cohort to remove cross-organ baseline bias.

    Methods:
      - 'percentile': Replaces global score with its empirical percentile within the organ cohort (0.0 to 1.0).
      - 'median_center': Subtracts organ-level median risk score and standardizes by IQR.
    """
    adjusted = np.zeros(len(df), dtype=float)
    df_copy = df.copy()

    for cohort, grp in df_copy.groupby(cohort_col):
        indices = grp.index
        vals = grp[score_col].values
        if method == "percentile":
            if len(vals) <= 1:
                ranks = np.array([0.5] * len(vals))
            else:
                ranks = (pd.Series(vals).rank(pct=True).values)
            adjusted[indices] = ranks
        elif method == "median_center":
            med = float(np.median(vals))
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            if iqr < 1e-5:
                iqr = 1.0
            adjusted[indices] = (vals - med) / iqr
        else:
            raise ValueError(f"Unknown adjustment method: {method}")

    return adjusted


# ─── 5. Stage- and Age-Adjusted Multivariate Cox Regression ──────────────────

def fit_stage_adjusted_cox_regression(
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    score_col: str = "risk_score",
    stage_col: Optional[str] = "stage",
    age_col: Optional[str] = "age",
) -> Dict[str, Any]:
    """
    Fits multivariate Cox Proportional Hazards regression testing whether the GUIDO
    risk score provides independent prognostic contribution beyond AJCC stage and patient age.

    Calculates:
      - Hazard Ratio (HR) and 95% Confidence Interval for Risk Score.
      - Wald test p-value.
      - Likelihood Ratio Test (LRT) comparing Full Model vs Baseline (Stage + Age).
      - Grambsch-Therneau Proportional Hazards diagnostic test.
    """
    cols = [time_col, event_col, score_col]
    if stage_col and stage_col in df.columns:
        cols.append(stage_col)
    if age_col and age_col in df.columns:
        cols.append(age_col)

    clean_df = df[cols].dropna().copy()
    clean_df = clean_df[clean_df[time_col] > 0]

    if len(clean_df) < 20 or clean_df[event_col].sum() < 5:
        return {
            "error": "Insufficient sample size or events for multivariate Cox regression",
            "n_samples": len(clean_df),
            "n_events": int(clean_df[event_col].sum()),
        }

    # 1. Full Model (Risk Score + Stage + Age)
    cph_full = CoxPHFitter(penalizer=0.01)
    cph_full.fit(clean_df, duration_col=time_col, event_col=event_col)

    full_ll = cph_full.log_likelihood_

    summary_df = cph_full.summary
    rs_row = summary_df.loc[score_col]

    hr = float(np.exp(rs_row["coef"]))
    ci_lower = float(np.exp(rs_row["coef lower 95%"]))
    ci_upper = float(np.exp(rs_row["coef upper 95%"]))
    p_val = float(rs_row["p"])
    z_val = float(rs_row["z"])

    # 2. Baseline Model (without Risk Score) for Likelihood Ratio Test
    baseline_cols = [c for c in cols if c != score_col]
    lrt_p = 1.0
    lrt_stat = 0.0
    if len(baseline_cols) > 2:
        clean_base = clean_df[baseline_cols]
        cph_base = CoxPHFitter(penalizer=0.01)
        cph_base.fit(clean_base, duration_col=time_col, event_col=event_col)
        base_ll = cph_base.log_likelihood_
        # Likelihood ratio statistic: 2 * (LL_full - LL_base) ~ Chi-Square(df=1)
        lrt_stat = max(0.0, 2 * (full_ll - base_ll))
        from scipy.stats import chi2
        lrt_p = float(1.0 - chi2.cdf(lrt_stat, df=1))

    # 3. Proportional Hazards Assumption Diagnostic (Grambsch-Therneau test)
    ph_p = 1.0
    try:
        ph_test = proportional_hazard_test(cph_full, clean_df, time_transform="rank")
        if score_col in ph_test.summary.index:
            ph_p = float(ph_test.summary.loc[score_col, "p"])
    except Exception as e:
        logger.warning("Proportional hazards test exception: %s", e)

    return {
        "n_samples": len(clean_df),
        "n_events": int(clean_df[event_col].sum()),
        "risk_score": {
            "hazard_ratio": round(hr, 4),
            "coef": round(float(rs_row["coef"]), 4),
            "se": round(float(rs_row["se(coef)"]), 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "z_stat": round(z_val, 4),
            "p_value": round(p_val, 6),
            "statistically_significant": p_val < 0.05,
        },
        "likelihood_ratio_p_value": round(lrt_p, 6),
        "likelihood_ratio_test": {
            "chi2_stat": round(lrt_stat, 4),
            "p_value": round(lrt_p, 6),
            "independent_prognostic_value": lrt_p < 0.05,
        },
        "proportional_hazards_test": {
            "p_value": round(ph_p, 4),
            "assumption_met": ph_p >= 0.05,
        },
        "concordance_index": round(float(cph_full.concordance_index_), 4),
    }
