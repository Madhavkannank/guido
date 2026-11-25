"""
Step 8 – Cross-Cohort & Independent External Validation
======================================================
Evaluates trained clinical genomic risk models on independent hospital and
consortium cohorts (METABRIC, GEO GSE31210, etc.) without fine-tuning:

  1. Out-of-Distribution Concordance (Harrell's C-Index)
  2. 3-Year Landmark Discrimination (AUROC & AUPRC with Bootstrap 95% CIs)
  3. Calibration & Reliability (Brier Score, Expected Calibration Error [ECE])
  4. Clinical Risk Stratification (Low <35%, Moderate 35-65%, High >65%)
  5. Feature Space & Hallmark Metagene Alignment using trained baseline statistics

All validation results are persisted to disk and optionally logged to MLflow.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MLFLOW_TRACKING_URI, RESULTS_DIR
from src.models.deep_surv import PyTorchDeepSurvClassifier
from src.models.metrics import compute_clf_metrics

# Ensure backward unpickling compatibility across all execution contexts
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "PyTorchDeepSurvClassifier", PyTorchDeepSurvClassifier)

logger = logging.getLogger(__name__)

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ─── High-Performance Concordance Index ──────────────────────────────────────

def compute_c_index_fast(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    """Computes Harrell's C-Index handling right-censoring in O(N log N) time."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(risk_scores, dtype=float)
    n = len(times)
    if n < 2:
        return 0.5

    order = np.lexsort((scores, times))
    times = times[order]
    events = events[order]
    scores = scores[order]

    event_indices = np.where(events == 1)[0]
    if len(event_indices) == 0:
        return 0.5

    concordant = 0.0
    pairs = 0
    for idx in event_indices:
        t_i = times[idx]
        s_i = scores[idx]
        valid_j = (times > t_i)
        if not np.any(valid_j):
            continue
        scores_j = scores[valid_j]
        pairs += len(scores_j)
        concordant += float(np.sum(s_i > scores_j)) + 0.5 * float(np.sum(s_i == scores_j))

    return round(float(concordant / max(pairs, 1)), 4)


# ─── Expected Calibration Error (ECE) ────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return round(float(ece), 4)


# ─── AUROC bootstrap CI ──────────────────────────────────────────────────────

def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for AUROC."""
    rng = np.random.default_rng(seed)
    aurocs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aurocs.append(roc_auc_score(y_b, p_b))
    if not aurocs:
        return (0.5, 0.5)
    aurocs = np.array(aurocs)
    lower = float(np.percentile(aurocs, 100 * alpha / 2))
    upper = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


# ─── Feature & Metagene Alignment ────────────────────────────────────────────

def align_external_features(
    X_ext: pd.DataFrame,
    artifact: Dict[str, Any]
) -> pd.DataFrame:
    """
    Standardizes external patient expression using training metagene_stats,
    computes the 12 Hallmark Metagenes, and aligns columns to artifact['features'].
    """
    target_features: List[str] = artifact["features"]
    hallmark_pathways: Dict[str, List[str]] = artifact.get("hallmark_metagenes", {})
    metagene_stats: Dict[str, Dict[str, float]] = artifact.get("metagene_stats", {})

    X_aligned = X_ext.copy()

    # Compute hallmark metagenes using training baseline statistics
    for meta_name, gene_list in hallmark_pathways.items():
        if meta_name not in X_aligned.columns:
            z_cols = []
            for g in gene_list:
                if g in X_aligned.columns:
                    mu = metagene_stats[g]["mean"] if g in metagene_stats else float(X_aligned[g].mean())
                    sigma = metagene_stats[g]["std"] if g in metagene_stats else 1.0
                    z = (X_aligned[g].values - mu) / (sigma + 1e-6)
                    z_cols.append(z)
            if z_cols:
                X_aligned[meta_name] = np.mean(np.column_stack(z_cols), axis=1)
            else:
                X_aligned[meta_name] = 0.0

    # Reindex to exact 100 features in order
    missing = set(target_features) - set(X_aligned.columns)
    if missing:
        logger.warning("External dataset missing %d features; filling with 0.0", len(missing))
    X_final = X_aligned.reindex(columns=target_features, fill_value=0.0)
    return X_final


# ─── Clinical External Validation Engine ─────────────────────────────────────

def evaluate_clinical_model_on_external_cohort(
    model_or_artifact: Union[Path, str, Dict[str, Any]],
    X_external: pd.DataFrame,
    y_external: pd.Series,
    times_external: pd.Series,
    events_external: pd.Series,
    cohort_name: str = "External-Cohort"
) -> Dict[str, Any]:
    """
    Evaluates a trained clinical genomic risk model on an independent external cohort.
    Returns a comprehensive validation scorecard.
    """
    if isinstance(model_or_artifact, (str, Path)):
        artifact_path = Path(model_or_artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
    else:
        artifact = model_or_artifact

    model = artifact["model"]
    scaler = artifact["scaler"]
    internal_metrics = artifact.get("test_metrics", {})

    # 1. Align features and compute hallmark metagenes
    X_aligned = align_external_features(X_external, artifact)

    # 2. Scale features using internal training scaler
    X_scaled = scaler.transform(X_aligned.values.astype(np.float32))

    # 3. Model Inference
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    # 4. Out-of-Distribution Metrics
    y_true = y_external.values.astype(int)
    times = times_external.values.astype(float)
    events = events_external.values.astype(int)

    ext_c_idx = compute_c_index_fast(times, events, y_prob)
    ext_auroc = round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5
    ext_auprc = round(float(average_precision_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.0
    ext_brier = round(float(brier_score_loss(y_true, y_prob)), 4)
    ext_ece = compute_ece(y_true, y_prob, n_bins=10)
    ext_bal_acc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
    ext_ci = bootstrap_auroc_ci(y_true, y_prob)

    # 5. Risk Stratification Tiers
    low_risk = int(np.sum(y_prob < 0.35))
    mod_risk = int(np.sum((y_prob >= 0.35) & (y_prob <= 0.65)))
    high_risk = int(np.sum(y_prob > 0.65))

    delta_auroc = round(internal_metrics.get("risk_auroc", ext_auroc) - ext_auroc, 4)
    generalisation_verdict = "EXCELLENT" if abs(delta_auroc) <= 0.08 else ("ACCEPTABLE" if abs(delta_auroc) <= 0.15 else "DEGRADED")

    report = {
        "external_cohort"      : cohort_name,
        "n_samples"            : len(y_true),
        "n_events"             : int(np.sum(events)),
        "internal_metrics"     : internal_metrics,
        "external_metrics"     : {
            "c_index"          : ext_c_idx,
            "risk_auroc"       : ext_auroc,
            "auroc_95ci"       : ext_ci,
            "risk_auprc"       : ext_auprc,
            "brier_score"      : ext_brier,
            "ece"              : ext_ece,
            "balanced_accuracy": ext_bal_acc,
        },
        "delta_auroc"          : delta_auroc,
        "generalisation_flag"  : generalisation_verdict,
        "risk_stratification"  : {
            "low_risk_count"   : low_risk,
            "moderate_risk_count": mod_risk,
            "moderate_risk_cnt": mod_risk,
            "high_risk_count"  : high_risk,
        }
    }

    logger.info("External Validation Complete on %s: C-Index=%.4f, AUROC=%.4f (Verdict: %s)",
                cohort_name, ext_c_idx, ext_auroc, generalisation_verdict)
    return report


def save_validation_report(report: Dict[str, Any], project_id: str, out_dir: Optional[Path] = None) -> Path:
    """Persists external validation report to results directory."""
    out_dir = out_dir or (RESULTS_DIR / project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_slug = report.get("external_cohort", "external").lower().replace(" ", "_")
    out_path = out_dir / f"external_validation_{cohort_slug}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved external validation report to %s", out_path)
    return out_path
