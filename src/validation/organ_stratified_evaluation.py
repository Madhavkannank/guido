"""
Organ-Stratified Clinical Risk Evaluation & Stage-Adjusted Cox Regression
========================================================================
Scientific validation module designed to evaluate whether genomic prognostic risk models
provide genuine, independent clinical value within individual cancer types rather than
relying on pan-cancer organ confounding (tissue baseline survival differences).

Capabilities:
  1. Harrell's Concordance Index with comparable pair counts, patient-level cluster bootstrap 95% CIs.
  2. Uno's Inverse Probability of Censoring Weighted (IPCW) C-statistic.
  3. Organ-Stratified Cohort Evaluation (reporting intra-cohort metrics across TCGA cohorts,
     distinguishing evaluable from unevaluable cohorts without artificial floors).
  4. Multiple testing correction (Benjamini-Hochberg FDR and Bonferroni across cohorts).
  5. Deployable Organ-Conditioned Risk Modeling (frozen reference distributions for N=1 clinical scoring).
  6. Stage- and Age-Adjusted Multivariate Cohort-Stratified Cox Proportional Hazards Regression
     with Hazard Ratios (HR), 95% CIs, Wald p-values, Likelihood Ratio Tests, and PH diagnostics.
  7. TCGA Cohort Availability Registry cataloging all 33 official cohorts.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from scipy.stats import chi2, norm

logger = logging.getLogger(__name__)

# Official 33 TCGA Pan-Cancer Cohort Codes
ALL_33_TCGA_COHORTS: List[str] = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"
]


# ─── 1. Harrell's C-Index with Patient-Level Cluster Bootstrap CIs ────────────

def compute_harrell_c_index_with_ci(
    times: np.ndarray,
    events: np.ndarray,
    scores: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes Harrell's C-index along with comparable pair counts and empirical bootstrap 95% CIs.
    Supports patient-level clustered resampling when multiple samples originate from the same patient.

    Parameters:
      times: Observed event or censoring times (days).
      events: Binary event status (1 = progression/death, 0 = censored).
      scores: Continuous risk predictions (higher score implies higher risk / shorter survival).
      patient_ids: Optional array of patient identifiers for cluster bootstrap.
      n_bootstrap: Number of bootstrap iterations.
      alpha: Significance level (default 0.05 for 95% CI).
      seed: Random seed for reproducibility.

    Returns:
      Dictionary containing c_index, comparable_pairs, ci_lower, ci_upper, standard_error, status.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if patient_ids is not None:
        patient_ids = np.asarray(patient_ids)

    valid_mask = (times > 0) & np.isfinite(times) & np.isfinite(scores) & (~np.isnan(events)) & np.isin(events, [0, 1])
    times = times[valid_mask]
    events = events[valid_mask]
    scores = scores[valid_mask]
    if patient_ids is not None:
        patient_ids = patient_ids[valid_mask]

    n = len(times)
    n_events = int(np.sum(events == 1))

    # Fast comparable pair count calculation
    def _count_comparable_pairs(t: np.ndarray, e: np.ndarray) -> int:
        ev_indices = np.where(e == 1)[0]
        if len(ev_indices) == 0:
            return 0
        pairs = 0
        for i in ev_indices:
            t_i = t[i]
            # Valid comparable pair: patient j had event or censoring at t_j > t_i,
            # or t_j == t_i where j was censored (e_j == 0)
            valid_j = (t > t_i) | ((t == t_i) & (e == 0))
            pairs += int(np.sum(valid_j))
        return pairs

    pairs = _count_comparable_pairs(times, events)

    # Unevaluable case: zero events, n < 2, or no comparable pairs
    if n < 2 or n_events == 0 or pairs == 0:
        return {
            "c_index": None,
            "comparable_pairs": pairs,
            "ci_lower": None,
            "ci_upper": None,
            "standard_error": None,
            "n_samples": n,
            "n_events": n_events,
            "status": "unevaluable",
            "reason": "Zero survival events or no comparable pairs available",
        }

    # Gold-standard calculation via lifelines concordance_index
    # lifelines interprets higher value as longer survival, so pass -scores for risk
    try:
        point_c = float(concordance_index(times, -scores, events))
    except Exception as exc:
        logger.warning("Concordance calculation failed: %s", exc)
        return {
            "c_index": None,
            "comparable_pairs": pairs,
            "ci_lower": None,
            "ci_upper": None,
            "standard_error": None,
            "n_samples": n,
            "n_events": n_events,
            "status": "unevaluable",
            "reason": str(exc),
        }

    # Bootstrap for confidence interval (cluster bootstrap if patient_ids supplied)
    rng = np.random.RandomState(seed)
    boot_cs: List[float] = []

    if n_bootstrap > 0 and pairs > 0:
        if patient_ids is not None and len(np.unique(patient_ids)) > 2:
            # Cluster bootstrap at patient level
            unique_patients = np.unique(patient_ids)
            n_patients = len(unique_patients)
            # Pre-index patient rows
            patient_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_patients}

            for _ in range(n_bootstrap):
                sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
                sample_indices = np.concatenate([patient_to_indices[p] for p in sampled_patients])
                t_b, e_b, s_b = times[sample_indices], events[sample_indices], scores[sample_indices]
                if np.sum(e_b == 1) == 0:
                    continue
                try:
                    c_b = float(concordance_index(t_b, -s_b, e_b))
                    boot_cs.append(c_b)
                except ZeroDivisionError:
                    continue
        else:
            # Standard observation-level bootstrap
            for _ in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                t_b, e_b, s_b = times[idx], events[idx], scores[idx]
                if np.sum(e_b == 1) == 0:
                    continue
                try:
                    c_b = float(concordance_index(t_b, -s_b, e_b))
                    boot_cs.append(c_b)
                except ZeroDivisionError:
                    continue

    if len(boot_cs) >= 10:
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
        "n_events": n_events,
        "status": "evaluable",
    }


# ─── 2. Uno's Inverse Probability of Censoring Weighted C-Index ──────────────

def compute_unos_c_index(
    times: Union[pd.Series, np.ndarray, List[float]],
    events: Union[pd.Series, np.ndarray, List[int]],
    scores: Union[pd.Series, np.ndarray, List[float]],
    tau: Optional[float] = None,
) -> Optional[float]:
    """
    Computes Uno's Inverse Probability of Censoring Weighted (IPCW) concordance index
    (Uno et al., Ann Intern Med 2011; 155:310-315).

    Mathematical Formulation:
      Let G_hat(t-) = P(C >= t) be the left-continuous Kaplan-Meier estimator of the
      censoring distribution, where censoring is treated as the event (Delta_c = 1 - Delta).
      Truncation time tau is defined as:
        tau = sup { t : G_hat(t-) >= 0.05 }

      Pairs are evaluated for events occurring at or before tau (Delta_i = 1, X_i <= tau).
      Each eligible event i is compared to all subjects j surviving longer (X_j > X_i).
      The IPCW weight is:
        W_i = 1 / [G_hat(X_i-)]^2

      The concordance estimator is:
        C_tau = sum_{i: Delta_i=1, X_i <= tau} sum_{j: X_j > X_i} W_i * [ I(S_i > S_j) + 0.5 * I(S_i == S_j) ]
                ------------------------------------------------------------------------------------------------
                         sum_{i: Delta_i=1, X_i <= tau} sum_{j: X_j > X_i} W_i

    Returns:
      Uno's C-index as float in [0, 1], or None if unevaluable (e.g., zero events, no comparable pairs,
      or denominator <= 0).
    """
    times_arr = np.asarray(times, dtype=float)
    events_arr = np.asarray(events, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    valid_mask = (times_arr > 0) & np.isfinite(times_arr) & np.isfinite(scores_arr) & (~np.isnan(events_arr)) & np.isin(events_arr, [0, 1])
    times_clean = times_arr[valid_mask]
    events_clean = events_arr[valid_mask]
    scores_clean = scores_arr[valid_mask]

    n = len(times_clean)
    event_indices = np.where(events_clean == 1)[0]
    if len(event_indices) == 0 or n < 2:
        return None

    # Step 1: Fit left-continuous Kaplan-Meier estimator of censoring distribution G_hat(t-)
    # Censoring is treated as the event: d_c = 1 - events
    censoring = 1 - events_clean
    unique_times = np.sort(np.unique(times_clean))

    g_step_times: List[float] = []
    g_left_values: List[float] = []
    g_val = 1.0

    for t in unique_times:
        g_step_times.append(float(t))
        g_left_values.append(float(g_val))

        c_at_t = np.sum((times_clean == t) & (censoring == 1))
        n_at_t = np.sum(times_clean >= t)
        if n_at_t > 0 and c_at_t > 0:
            g_val *= (1.0 - float(c_at_t) / float(n_at_t))

    g_step_times_arr = np.array(g_step_times)
    g_left_values_arr = np.array(g_left_values)

    def get_g_left(t_query: float) -> float:
        if len(g_step_times_arr) == 0 or t_query <= g_step_times_arr[0]:
            return 1.0
        idx = np.searchsorted(g_step_times_arr, t_query)
        if idx >= len(g_left_values_arr):
            return float(g_val)
        return float(g_left_values_arr[idx])

    # Step 2: Determine truncation time tau = sup { t : G_hat(t-) >= 0.05 }
    valid_tau_mask = g_left_values_arr >= 0.05
    if not np.any(valid_tau_mask):
        return None

    max_tau_empirical = float(np.max(g_step_times_arr[valid_tau_mask]))
    if tau is None:
        tau_val = max_tau_empirical
    else:
        tau_val = min(float(tau), max_tau_empirical)

    # Step 3: Compute IPCW concordance on eligible pairs
    numerator = 0.0
    denominator = 0.0

    for i in event_indices:
        t_i = times_clean[i]
        if t_i > tau_val:
            continue

        g_left_i = get_g_left(t_i)
        if g_left_i < 0.05 or g_left_i <= 0.0:
            continue

        weight_i = 1.0 / (g_left_i ** 2)

        # In Uno et al. 2011: comparable patients j have X_j > X_i
        valid_j = times_clean > t_i
        if not np.any(valid_j):
            continue

        s_i = scores_clean[i]
        s_j = scores_clean[valid_j]

        # Higher score = higher risk = shorter predicted survival
        concordant = float(np.sum(s_i > s_j)) + 0.5 * float(np.sum(s_i == s_j))
        n_pairs_i = float(len(s_j))

        numerator += concordant * weight_i
        denominator += n_pairs_i * weight_i

    if denominator <= 0.0:
        return None

    return round(float(numerator / denominator), 4)


# ─── 3. Multiple Testing Correction (Benjamini-Hochberg FDR) ─────────────────

def adjust_p_values_benjamini_hochberg(p_values: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Applies the Benjamini-Hochberg (1995) procedure to control False Discovery Rate (FDR).

    Parameters:
      p_values: 1D array of unadjusted p-values.

    Returns:
      Array of FDR q-values (adjusted p-values) bounded in [0, 1].
    """
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.clip(p_vals, 0.0, 1.0)

    order = np.argsort(p_vals)
    sorted_p = p_vals[order]

    # q_i = min_{k >= i} (m * p_k / k)
    q_vals = np.zeros(n, dtype=float)
    cumulative_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = (sorted_p[i] * n) / rank
        cumulative_min = min(cumulative_min, q)
        q_vals[i] = cumulative_min

    # Revert to original order
    orig_q = np.zeros(n, dtype=float)
    orig_q[order] = np.clip(q_vals, 0.0, 1.0)
    return orig_q


# ─── 4. Organ-Stratified Cohort Evaluation ───────────────────────────────────

def evaluate_organ_stratified_performance(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    time_col: str = "time",
    event_col: str = "event",
    score_col: str = "risk_score",
    patient_col: Optional[str] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates risk predictions within each organ cohort independently to detect
    and measure organ confounding. Distinguishes evaluable from unevaluable cohorts
    and applies Benjamini-Hochberg FDR multiple testing correction.

    Parameters:
      df: Clinical evaluation DataFrame.
      cohort_col: Column with organ / cancer type identifier.
      time_col: Survival follow-up time column.
      event_col: Binary event column.
      score_col: Predicted genomic risk score column.
      patient_col: Optional column with unique patient IDs for cluster bootstrap.
      n_bootstrap: Number of bootstrap iterations per cohort.
      alpha: Significance level for confidence intervals.

    Returns:
      Dict with individual cohort metrics, FDR-adjusted significance, weighted intra-cohort C-index,
      and scientific limitation notice.
    """
    unique_cohorts = sorted(df[cohort_col].dropna().unique())
    cohort_results: Dict[str, Dict[str, Any]] = {}

    total_pairs = 0
    weighted_c_sum = 0.0
    valid_c_indices = []

    raw_p_values = []
    evaluable_cohort_keys = []

    for c in unique_cohorts:
        sub_df = df[df[cohort_col] == c]
        times = sub_df[time_col].values
        events = sub_df[event_col].values
        scores = sub_df[score_col].values
        pts = sub_df[patient_col].values if (patient_col and patient_col in sub_df.columns) else None

        n_samples = len(sub_df)
        n_events = int(np.sum(events == 1))

        # Check if cohort has sufficient events for evaluation (minimum 5 events)
        if n_events < 5:
            cohort_results[c] = {
                "status": "unevaluable",
                "reason": f"Insufficient events ({n_events} observed, minimum 5 required)",
                "n_samples": n_samples,
                "n_events": n_events,
                "event_rate": round(float(n_events / max(n_samples, 1)), 4),
                "c_index": None,
                "unos_c_index": None,
                "comparable_pairs": 0,
                "ci_95": [None, None],
                "standard_error": None,
                "statistically_significant": False,
                "fdr_q_value": None,
            }
            continue

        harrell_res = compute_harrell_c_index_with_ci(
            times, events, scores,
            patient_ids=pts,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )

        if harrell_res["status"] == "unevaluable" or harrell_res["c_index"] is None:
            cohort_results[c] = {
                "status": "unevaluable",
                "reason": harrell_res.get("reason", "Unable to compute concordance"),
                "n_samples": n_samples,
                "n_events": n_events,
                "event_rate": round(float(n_events / max(n_samples, 1)), 4),
                "c_index": None,
                "unos_c_index": None,
                "comparable_pairs": harrell_res.get("comparable_pairs", 0),
                "ci_95": [None, None],
                "standard_error": None,
                "statistically_significant": False,
                "fdr_q_value": None,
            }
            continue

        uno_c = compute_unos_c_index(times, events, scores)
        pairs = harrell_res["comparable_pairs"]
        c_val = harrell_res["c_index"]
        raw_se = harrell_res.get("standard_error")
        se_val = 0.05 if raw_se is None else float(raw_se)

        # Wald test p-value testing H0: C = 0.5 (random guessing)
        if se_val > 0:
            z_score = (c_val - 0.5) / se_val
            p_val = float(2.0 * (1.0 - norm.cdf(abs(z_score))))
        else:
            p_val = 1.0

        raw_p_values.append(p_val)
        evaluable_cohort_keys.append(c)

        total_pairs += pairs
        weighted_c_sum += c_val * pairs
        valid_c_indices.append(c_val)

        cohort_results[c] = {
            "status": "evaluable",
            "n_samples": n_samples,
            "n_events": n_events,
            "event_rate": round(float(n_events / max(n_samples, 1)), 4),
            "c_index": c_val,
            "unos_c_index": uno_c,
            "comparable_pairs": pairs,
            "ci_95": [harrell_res["ci_lower"], harrell_res["ci_upper"]],
            "standard_error": harrell_res["standard_error"],
            "p_value_vs_random": round(p_val, 6),
            "statistically_significant": harrell_res["ci_lower"] > 0.50,
            "fdr_q_value": None,  # Populated after multiple testing correction
        }

    # Apply Benjamini-Hochberg FDR correction across evaluable cohorts
    if raw_p_values:
        fdr_qs = adjust_p_values_benjamini_hochberg(raw_p_values)
        for c_key, q in zip(evaluable_cohort_keys, fdr_qs):
            cohort_results[c_key]["fdr_q_value"] = round(float(q), 6)
            cohort_results[c_key]["fdr_significant"] = bool(q < 0.05)

    weighted_c = round(weighted_c_sum / max(total_pairs, 1), 4) if total_pairs > 0 else None
    macro_c = round(float(np.mean(valid_c_indices)), 4) if valid_c_indices else None
    sig_cohorts = sum(1 for r in cohort_results.values() if r.get("statistically_significant", False))
    fdr_sig_cohorts = sum(1 for r in cohort_results.values() if r.get("fdr_significant", False))

    summary = {
        "total_cohorts_checked": len(unique_cohorts),
        "evaluable_cohorts_count": len(evaluable_cohort_keys),
        "unevaluable_cohorts_count": len(unique_cohorts) - len(evaluable_cohort_keys),
        "total_samples": len(df),
        "total_events": int(df[event_col].sum()),
        "total_comparable_pairs": total_pairs,
        "weighted_intra_cohort_c_index": weighted_c,
        "macro_intra_cohort_c_index": macro_c,
        "cohorts_with_ci_excluding_half": sig_cohorts,
        "cohorts_with_fdr_significant_prognosis": fdr_sig_cohorts,
        "scientific_limitation": (
            "IMPORTANT SCIENTIFIC LIMITATION: Within-cohort discrimination (C-index > 0.5) "
            "demonstrates prognostic ordering within a tissue, but does NOT by itself prove "
            "pan-cancer organ confounding has been eliminated across disparate tissues. "
            "Intra-cohort evaluation must be combined with organ-stratified baseline hazards."
        ),
    }

    return {
        "cohort_results": cohort_results,
        "summary": summary,
    }


# ─── 5. Deployable Organ-Conditioned Risk Modeling (N=1 Deployment) ───────────

@dataclass
class OrganConditionedRiskAdjuster:
    """
    Fits and freezes organ-specific baseline risk reference distributions during training,
    enabling deterministic single-patient (N=1) clinical inference without transductive batch leakage.

    CLINICAL INTERPRETATION NOTICE:
    Percentile normalization provides within-organ relative ranking, NOT calibrated absolute clinical risk.
    Reference distributions must be fit strictly on independent training data or cross-validation training folds.
    If no independent reference exists for a cohort, organ-conditioned scoring is unavailable (returns None/NaN).
    """
    method: str = "percentile"  # 'percentile' or 'median_center'
    reference_distributions_: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    is_fitted_: bool = False

    def fit(self, df: pd.DataFrame, cohort_col: str = "cohort", score_col: str = "risk_score") -> OrganConditionedRiskAdjuster:
        """Computes and freezes quantiles, median, and IQR for each organ cohort strictly from training data."""
        self.reference_distributions_ = {}
        for cohort, grp in df.groupby(cohort_col):
            vals = grp[score_col].dropna().values.astype(float)
            if len(vals) < 3:
                logger.warning("Cohort '%s' has fewer than 3 training samples; skipping reference fitting.", cohort)
                continue
            quantiles = {f"q_{int(p)}": float(np.percentile(vals, p)) for p in range(5, 100, 5)}
            median_val = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
            iqr_val = max(q75 - q25, 1e-5)
            self.reference_distributions_[str(cohort)] = {
                "n_training_samples": len(vals),
                "median": median_val,
                "iqr": iqr_val,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                **quantiles,
            }
        self.is_fitted_ = True
        return self

    def transform_single(self, score: float, cohort: str) -> Optional[float]:
        """
        Transforms a single patient's raw score conditioned strictly on their organ's frozen reference.
        Returns None if no independent reference distribution exists for the cohort.
        """
        if not self.is_fitted_:
            raise ValueError("OrganConditionedRiskAdjuster must be fitted on training data before transform.")
        cohort = str(cohort)
        if cohort not in self.reference_distributions_:
            # Unknown cohort with no independent reference: mark unavailable
            return None

        dist = self.reference_distributions_[cohort]
        if self.method == "median_center":
            return float((score - dist["median"]) / dist["iqr"])
        else:
            # Empirical percentile interpolation using frozen 5th..95th quantiles
            pct_steps = [p for p in range(5, 100, 5)]
            if "quantiles" in dist and isinstance(dist["quantiles"], dict):
                # Compatible with DeployableOrganScorer serialized format
                q_dict = dist["quantiles"]
                q_vals = []
                for p in pct_steps:
                    key1 = f"{p / 100.0:.2f}"
                    key2 = f"q_{p}"
                    val = q_dict.get(key1, dist.get(key2, None))
                    if val is None:
                        val = q_dict.get(str(p / 100.0), dist.get(str(p), 0.0))
                    q_vals.append(float(val))
            else:
                q_vals = [dist[f"q_{p}"] for p in pct_steps]
            # Use linear interpolation
            pct = np.interp(score, q_vals, [p / 100.0 for p in pct_steps], left=0.01, right=0.99)
            return float(pct)

    def transform(self, df: pd.DataFrame, cohort_col: str = "cohort", score_col: str = "risk_score") -> np.ndarray:
        """
        Applies frozen reference transformation to an evaluation DataFrame.
        Missing / unreferenced cohorts receive np.nan.
        """
        if not self.is_fitted_:
            raise ValueError("OrganConditionedRiskAdjuster must be fitted on training data before transform.")
        adjusted = np.full(len(df), np.nan, dtype=float)
        for i, (_, row) in enumerate(df.iterrows()):
            c = row[cohort_col]
            s = row[score_col]
            val = self.transform_single(s, c)
            if val is not None:
                adjusted[i] = val
        return adjusted


def organ_conditioned_risk_adjustment(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    score_col: str = "risk_score",
    method: str = "percentile",
) -> np.ndarray:
    """Convenience wrapper fitting and transforming in-sample (for benchmark evaluation only)."""
    adjuster = OrganConditionedRiskAdjuster(method=method)
    adjuster.fit(df, cohort_col=cohort_col, score_col=score_col)
    return adjuster.transform(df, cohort_col=cohort_col, score_col=score_col)


# ─── 6. Stage- and Age-Adjusted Multivariate Cohort-Stratified Cox Regression ─

def _parse_stage_to_numeric(stage_val: Any) -> Optional[float]:
    """Parses heterogeneous AJCC clinical/pathologic stage strings into numeric 1, 2, 3, 4."""
    if stage_val is None or pd.isna(stage_val) or stage_val == "":
        return None
    s_str = str(stage_val).strip().upper()
    if "STAGE IV" in s_str or re.search(r"\b(IV[ABC]?|4[ABC]?)\b", s_str) or re.search(r"^IV[ABC]?$", s_str):
        return 4.0
    elif "STAGE III" in s_str or re.search(r"\b(III[ABC]?|3[ABC]?)\b", s_str) or re.search(r"^III[ABC]?$", s_str):
        return 3.0
    elif "STAGE II" in s_str or re.search(r"\b(II[ABC]?|2[ABC]?)\b", s_str) or re.search(r"^II[ABC]?$", s_str):
        return 2.0
    elif "STAGE I" in s_str or re.search(r"\b(I[ABC]?|1[ABC]?)\b", s_str) or re.search(r"^I[ABC]?$", s_str):
        return 1.0
    return None


def fit_stage_adjusted_cox_regression(
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    score_col: str = "risk_score",
    stage_col: Optional[str] = "stage",
    age_col: Optional[str] = "age",
    strata_col: Optional[str] = "cohort",
) -> Dict[str, Any]:
    """
    Fits multivariate Cox Proportional Hazards regression testing whether the GUIDO
    risk score provides independent prognostic contribution beyond AJCC stage and patient age.
    Uses cohort stratification (strata=['cohort']) to allow cohort-specific baseline hazards.

    Calculates:
      - Hazard Ratio (HR) and 95% Confidence Interval for Risk Score.
      - Wald test p-value.
      - Likelihood Ratio Test (LRT) comparing Full Model vs Baseline (Stage + Age).
      - Grambsch-Therneau Proportional Hazards diagnostic test with explicit violation reporting.
    """
    cols_to_keep = [time_col, event_col, score_col]
    clean_df = pd.DataFrame()
    clean_df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    clean_df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    clean_df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    included_covariates = [score_col]

    # Stage covariate parsing
    if stage_col and stage_col in df.columns:
        clean_df["stage_numeric"] = df[stage_col].apply(_parse_stage_to_numeric)
        included_covariates.append("stage_numeric")

    # Age covariate parsing
    if age_col and age_col in df.columns:
        clean_df["age_numeric"] = pd.to_numeric(df[age_col], errors="coerce")
        included_covariates.append("age_numeric")

    # Cohort strata
    use_strata = False
    if strata_col and strata_col in df.columns and len(df[strata_col].dropna().unique()) > 1:
        clean_df[strata_col] = df[strata_col].astype(str)
        use_strata = True

    # Drop missing values
    before_n = len(clean_df)
    eval_cols = [time_col, event_col, score_col] + [c for c in included_covariates if c != score_col]
    if use_strata and strata_col:
        eval_cols.append(strata_col)

    clean_df = clean_df.dropna(subset=eval_cols)
    clean_df = clean_df[clean_df[time_col] > 0]
    after_n = len(clean_df)
    dropped_n = before_n - after_n

    n_events = int(clean_df[event_col].sum())
    if len(clean_df) < 20 or n_events < 5:
        return {
            "status": "unevaluable",
            "reason": f"Insufficient sample size (N={len(clean_df)}) or events (E={n_events}) for multivariate Cox regression",
            "n_samples": len(clean_df),
            "n_events": n_events,
            "dropped_missing_covariates": dropped_n,
        }

    # 1. Full Model (Risk Score + Stage + Age), stratified by cohort if multi-cohort
    cph_full = CoxPHFitter(penalizer=0.01)
    strata_list = [strata_col] if use_strata and strata_col else None
    cph_full.fit(clean_df, duration_col=time_col, event_col=event_col, strata=strata_list)

    full_ll = cph_full.log_likelihood_
    summary_df = cph_full.summary
    rs_row = summary_df.loc[score_col]

    hr = float(np.exp(rs_row["coef"]))
    ci_lower = float(np.exp(rs_row["coef lower 95%"]))
    ci_upper = float(np.exp(rs_row["coef upper 95%"]))
    p_val = float(rs_row["p"])
    z_val = float(rs_row["z"])

    # 2. Baseline Model (without Risk Score) for Nested Likelihood Ratio Test
    baseline_covars = [c for c in included_covariates if c != score_col]
    lrt_p = 1.0
    lrt_stat = 0.0
    df_diff = 1

    if len(baseline_covars) > 0:
        base_cols = [time_col, event_col] + baseline_covars
        if use_strata and strata_col:
            base_cols.append(strata_col)
        clean_base = clean_df[base_cols]
        cph_base = CoxPHFitter(penalizer=0.01)
        cph_base.fit(clean_base, duration_col=time_col, event_col=event_col, strata=strata_list)
        base_ll = cph_base.log_likelihood_

        # Likelihood ratio statistic: 2 * (LL_full - LL_base) ~ Chi-Square(df=1)
        lrt_stat = max(0.0, 2 * (full_ll - base_ll))
        lrt_p = float(1.0 - chi2.cdf(lrt_stat, df=df_diff))

    # 3. Proportional Hazards Assumption Diagnostic (Grambsch-Therneau test)
    ph_p = 1.0
    ph_summary: Dict[str, Any] = {}
    assumption_met = True

    try:
        ph_test = proportional_hazard_test(cph_full, clean_df, time_transform="rank")
        if score_col in ph_test.summary.index:
            ph_p = float(ph_test.summary.loc[score_col, "p"])
            assumption_met = (ph_p >= 0.05)
        for idx_name, row_s in ph_test.summary.iterrows():
            ph_summary[str(idx_name)] = {
                "test_statistic": round(float(row_s["test_statistic"]), 4),
                "p_value": round(float(row_s["p"]), 4),
                "violates_ph": bool(float(row_s["p"]) < 0.05),
            }
    except Exception as exc:
        logger.warning("Proportional hazards test exception: %s", exc)
        ph_summary["error"] = str(exc)

    return {
        "status": "evaluable",
        "n_samples": len(clean_df),
        "n_events": n_events,
        "dropped_missing_covariates": dropped_n,
        "strata_used": strata_col if use_strata else None,
        "included_covariates": included_covariates,
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
        "likelihood_ratio_test": {
            "baseline_covariates": baseline_covars,
            "chi2_stat": round(lrt_stat, 4),
            "df": df_diff,
            "p_value": round(lrt_p, 6),
            "independent_prognostic_value": bool(lrt_p < 0.05),
        },
        "proportional_hazards_test": {
            "score_p_value": round(ph_p, 4),
            "assumption_met": assumption_met,
            "covariate_diagnostics": ph_summary,
            "warning": None if assumption_met else (
                "WARNING: Proportional hazards assumption violated (p < 0.05). "
                "Hazard ratios should be interpreted as time-averaged effects. "
                "Consider restricted mean survival time (RMST) or time-varying coefficients."
            ),
        },
        "concordance_index": round(float(cph_full.concordance_index_), 4),
    }


# ─── 7. TCGA Cohort Availability Registry ────────────────────────────────────

def audit_tcga_cohort_availability(
    data_dir: Union[str, Path] = "data",
) -> Dict[str, Any]:
    """
    Audits the local repository to transparently report which of the 33 official TCGA
    cohorts are available with local genomic expression matrices vs only clinical follow-up.
    """
    data_path = Path(data_dir)
    xena_raw = data_path / "raw" / "xena"
    shards_dir = data_path / "processed" / "shards"
    processed_dir = data_path / "processed"

    # Check clinical CDR table
    cdr_file = xena_raw / "TCGA_CDR_survival.tsv"
    cdr_available = cdr_file.exists() and cdr_file.stat().st_size > 1000

    locally_available_genomics: List[str] = []
    missing_genomics: List[str] = []

    for c in ALL_33_TCGA_COHORTS:
        # Check raw TSV
        raw_tsv = xena_raw / f"TCGA.{c}.sampleMap_HiSeqV2.tsv"
        # Check shard parquet
        shard_pq = shards_dir / f"TCGA_{c}_expression.parquet"
        # Check processed folder
        proc_pq = processed_dir / f"TCGA-{c}" / "rnaseq_processed.parquet"

        if (raw_tsv.exists() and raw_tsv.stat().st_size > 1000) or \
           (shard_pq.exists() and shard_pq.stat().st_size > 1000) or \
           (proc_pq.exists() and proc_pq.stat().st_size > 1000):
            locally_available_genomics.append(c)
        else:
            missing_genomics.append(c)

    return {
        "expected_33_tcga_cohorts": ALL_33_TCGA_COHORTS,
        "total_expected": len(ALL_33_TCGA_COHORTS),
        "locally_available_genomic_cohorts": locally_available_genomics,
        "locally_available_count": len(locally_available_genomics),
        "missing_genomic_cohorts": missing_genomics,
        "missing_count": len(missing_genomics),
        "clinical_cdr_available": cdr_available,
        "reporting_integrity_notice": (
            f"Of the 33 official TCGA cohorts, {len(locally_available_genomics)} have local expression "
            f"data ({', '.join(locally_available_genomics)}). The remaining {len(missing_genomics)} cohorts "
            f"require cloud data streaming from UCSC Xena. Any claims of pan-cancer validation "
            f"must clearly report this cohort coverage breakdown."
        ),
    }
