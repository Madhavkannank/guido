"""
Minimal Gene Panel Distillation Engine (Penalized Cox & Stability Selection)
=============================================================================
Distills high-dimensional transcriptomic models down to minimal,
clinically assayable candidate panels (15, 25, 35, 50 genes) using genuine
survival analysis (ElasticNet/L1-penalized Cox Proportional Hazards regression).

Scientific Safeguards & Methodology:
  1. True Survival Likelihood: Preserves continuous survival times and right-censoring
     status. Does NOT binarize survival times or substitute a binary classifier.
  2. Zero-Leakage Nested CV: Preprocessing (scaling), feature screening, stability selection,
     and model fitting are executed strictly within outer training folds.
  3. Patient-Grouped Cross-Validation: Evaluates multi-sample cohorts using GroupKFold
     grouped by 12-character patient barcode to eliminate patient leakage.
  4. Out-of-Fold Evaluation Integrity: Out-of-fold prediction buffers are initialized
     to NaN and monitored with explicit coverage tracking. Failed folds never inject
     fabricated zero-risk scores.
  5. Comparator Transparency: Explicitly reports original feature universe, screened
     feature universe, and comparator model type ("screened_high_dimensional_comparator").
     Retention is computed strictly against this fitted comparator.
  6. Stability Selection (Meinshausen & Bühlmann): Resampling within training partitions
     measures empirical feature selection probability and coefficient direction.
  7. Provenance Discipline: Explicitly records whether input data is synthetic or real TCGA,
     preserves endpoint definitions (PFI/OS), and flags in-silico validation status.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from src.validation.organ_stratified_evaluation import (
    compute_harrell_c_index_with_ci,
    compute_unos_c_index,
)

logger = logging.getLogger(__name__)

# Canonical Hallmark Gene Sets (Functional Biological Annotation)
HALLMARK_ANNOTATIONS: Dict[str, List[str]] = {
    "Sustained Proliferation": ["MKI67", "CCND1", "CDK4", "MYC", "EGFR", "ERBB2", "E2F1", "AURKA", "PLK1"],
    "Evading Growth Suppressors": ["TP53", "RB1", "CDKN2A", "PTEN", "SMAD4", "BRCA1", "BRCA2", "ATM"],
    "Resisting Cell Death": ["BCL2", "BCL2L1", "BAX", "MCL1", "CASP3", "FAS", "BAD", "BIRC5"],
    "Enabling Replicative Immortality": ["TERT", "TERC", "DKC1", "TINF2", "POT1", "ACD"],
    "Inducing Angiogenesis": ["VEGFA", "VEGFB", "KDR", "FLT1", "ANGPT1", "ANGPT2", "HIF1A"],
    "Activating Invasion & Metastasis": ["CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "MMP2", "MMP9"],
    "Reprogramming Energy Metabolism": ["SLC2A1", "HK2", "LDHA", "PKM", "IDH1", "GLS", "FASN"],
    "Evading Immune Destruction": ["CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "B2M", "HLA-A", "CXCL9"],
    "Tumor-Promoting Inflammation": ["IL6", "TNF", "CXCL8", "STAT3", "NFKB1", "PTGS2", "CSF1"],
    "Genome Instability & Mutation": ["MLH1", "MSH2", "MSH6", "PMS2", "PARP1", "RAD51", "ERCC1"],
}

# Standard Housekeeping Candidate Genes (Reference Controls)
# Empirically qualified across TCGA cohorts in src/data/cross_platform_normalizer.py
QUALIFIED_HOUSEKEEPING_CONTROLS: List[str] = ["RPL13A", "ACTB", "PPIA", "TBP"]
TISSUE_VARIABLE_HOUSEKEEPING_CONTROLS: List[str] = ["HPRT1", "B2M", "PGK1"]
HOUSEKEEPING_CONTROLS: List[str] = QUALIFIED_HOUSEKEEPING_CONTROLS


def extract_tcga_patient_ids(sample_ids: Union[List[str], pd.Index, np.ndarray]) -> np.ndarray:
    """
    Extracts 12-character patient barcode from TCGA sample IDs
    (e.g., 'TCGA-BP-4965-01A-...' -> 'TCGA-BP-4965').
    Preserves non-TCGA sample IDs unchanged.
    """
    pts = []
    for s in sample_ids:
        s_str = str(s).strip()
        parts = s_str.split("-")
        if len(parts) >= 3 and parts[0] == "TCGA":
            pts.append("-".join(parts[:3]))
        elif len(s_str) >= 12 and s_str.startswith("TCGA-"):
            pts.append(s_str[:12])
        else:
            pts.append(s_str)
    return np.array(pts)


@dataclass
class CandidatePanel:
    """Represents a distilled candidate clinical gene panel."""
    panel_size: int
    genes: List[str]
    weights: Dict[str, float]
    c_index: Optional[float]  # Genuine out-of-fold Harrell's C-index
    c_index_ci: Optional[List[float]]  # Out-of-fold 95% bootstrap confidence interval
    unos_c_index: Optional[float]  # Out-of-fold Uno's IPCW C-index
    retention_ratio: float  # Out-of-fold retention relative to comparator model
    hallmark_representation: Dict[str, int]
    biotypes: Dict[str, int]
    assay_type_recommendation: str
    validation_status: str = "in_silico_candidate"
    evidence_level: str = "in_silico_unvalidated"
    sample_count: int = 0
    event_count: int = 0
    fitted_feature_count: int = 0
    comparator_feature_count: int = 0
    comparator_type: str = "screened_high_dimensional_comparator"
    oof_coverage_ratio: float = 1.0
    oof_missing_count: int = 0
    panel_evaluation_status: str = "complete"
    panel_fit_fold_success_rate: float = 1.0


@dataclass
class DistillationReport:
    """Comprehensive report detailing the minimal gene panel distillation process."""
    data_source: str
    is_synthetic: bool
    endpoint: str
    evaluation_method: str
    n_samples: int
    n_events: int
    censoring_rate: float
    full_model_gene_count: int  # Comparator feature count (backward compatibility alias)
    full_model_c_index: Optional[float]  # Comparator out-of-fold C-index (backward compatibility alias)
    full_model_c_index_ci: Optional[List[float]]
    full_model_unos_c_index: Optional[float]
    candidate_panels: Dict[int, CandidatePanel]
    retention_curve: List[Dict[str, Any]]
    stability_rankings: List[Dict[str, Any]]
    limitations: List[str] = field(default_factory=list)
    # Scientific integrity audit fields
    original_feature_count: int = 0
    screened_feature_count: int = 0
    comparator_feature_count: int = 0
    comparator_type: str = "screened_high_dimensional_comparator"
    comparator_c_index: Optional[float] = None
    comparator_c_index_ci: Optional[List[float]] = None
    comparator_unos_c_index: Optional[float] = None
    cv_strategy: str = "row_level_kfold"
    patient_count: int = 0
    patient_grouping_status: str = "unverified"
    oof_coverage_ratio: float = 1.0
    oof_missing_count: int = 0
    fold_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_status: str = "complete"


def _compute_univariate_survival_concordance(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> np.ndarray:
    """
    Computes univariate survival prognostic strength |C - 0.5| for each feature.
    Used for pre-screening high-dimensional spaces strictly within training folds.
    """
    n, p = X.shape
    scores = np.zeros(p, dtype=float)
    event_idx = np.where(events == 1)[0]
    if len(event_idx) == 0:
        return scores

    for j in range(p):
        xj = X[:, j]
        concordant = 0.0
        pairs = 0
        for i in event_idx:
            ti = times[i]
            xi = xj[i]
            valid_k = times > ti
            if not np.any(valid_k):
                continue
            xk = xj[valid_k]
            pairs += len(xk)
            concordant += float(np.sum(xi > xk)) + 0.5 * float(np.sum(xi == xk))
        c_val = concordant / pairs if pairs > 0 else 0.5
        scores[j] = abs(c_val - 0.5)
    return scores


class MinimalGenePanelDistiller:
    """
    Distills transcriptomic risk models into minimal panels using regularized
    Cox Proportional Hazards regression and cross-validated stability selection.
    """

    def __init__(
        self,
        panel_sizes: Optional[List[int]] = None,
        n_folds: int = 4,
        penalizer: float = 0.1,
        l1_ratio: float = 0.7,
        n_subsamples: int = 3,
        max_screen_features: int = 60,
        random_state: int = 42,
        require_complete_oof: bool = False,
    ):
        self.panel_sizes = panel_sizes or [15, 25, 35, 50]
        self.n_folds = n_folds
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.n_subsamples = n_subsamples
        self.max_screen_features = max(max_screen_features, max(self.panel_sizes))
        self.random_state = random_state
        self.require_complete_oof = require_complete_oof

        self.feature_stability_scores_: Dict[str, float] = {}
        self.feature_weights_: Dict[str, float] = {}
        self.distillation_report_: Optional[DistillationReport] = None

    def fit(
        self,
        df_X: pd.DataFrame,
        times: Union[pd.Series, np.ndarray],
        events: Union[pd.Series, np.ndarray],
        patient_ids: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
        sample_ids: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
        data_source: str = "synthetic",
        is_synthetic: bool = True,
        endpoint: str = "PFI",
        require_complete_oof: Optional[bool] = None,
    ) -> DistillationReport:
        """
        Runs stability selection across CV folds to rank genes and extract candidate panels.
        All validation metrics are computed strictly OUT-OF-FOLD.
        """
        feature_names = list(df_X.columns)
        n_features = len(feature_names)
        n_samples_orig = len(df_X)

        # ─── Step 0a: Input Sequence and Length Validation ────────────────────
        try:
            n_times = len(times)
        except TypeError:
            raise ValueError(f"times must be a sequence or array, got {type(times)}")
        try:
            n_events = len(events)
        except TypeError:
            raise ValueError(f"events must be a sequence or array, got {type(events)}")

        if n_times != n_samples_orig or n_events != n_samples_orig:
            raise ValueError(
                f"Array length mismatch: df_X has {n_samples_orig} rows, "
                f"times has {n_times} elements, and events has {n_events} elements. "
                "All input arrays must have identical length."
            )

        if patient_ids is not None and len(patient_ids) != n_samples_orig:
            raise ValueError(
                f"Patient IDs length mismatch: df_X has {n_samples_orig} rows, "
                f"patient_ids has {len(patient_ids)} elements. Patient identifiers must match sample count."
            )

        if sample_ids is not None and len(sample_ids) != n_samples_orig:
            raise ValueError(
                f"Sample IDs length mismatch: df_X has {n_samples_orig} rows, "
                f"sample_ids has {len(sample_ids)} elements. Sample identifiers must match sample count."
            )

        # ─── Step 0b: Validate Raw Survival Inputs Prior to Numeric Casting ──
        raw_times_s = pd.to_numeric(pd.Series(times).reset_index(drop=True), errors="coerce")
        raw_events_s = pd.to_numeric(pd.Series(events).reset_index(drop=True), errors="coerce")

        # Missing and invalid checks:
        # Times must be non-NaN, finite (rejecting np.inf / -np.inf), and strictly positive (t > 0)
        # Events must be non-NaN and binary in {0, 1}
        valid_times = raw_times_s.notna() & np.isfinite(raw_times_s) & (raw_times_s > 0)
        valid_events = raw_events_s.notna() & raw_events_s.isin([0, 1])
        valid_mask = (valid_times & valid_events).to_numpy()

        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            reasons = []
            if (~valid_times).any():
                reasons.append(f"{int((~valid_times).sum())} non-positive/non-finite/missing times")
            if (~valid_events).any():
                reasons.append(f"{int((~valid_events).sum())} invalid/missing events (must be in {{0, 1}})")
            logger.warning(
                "Excluding %d invalid or missing sample(s) prior to Cox distillation: %s",
                n_invalid, "; ".join(reasons)
            )

        if not np.any(valid_mask):
            raise ValueError(
                "No valid survival records remain after excluding non-finite, non-positive, or missing times and events."
            )

        # Synchronously filter df_X, X, times, and events to preserve strict alignment
        df_X = df_X.iloc[valid_mask].copy()
        X = df_X.values.astype(float)
        times = raw_times_s[valid_mask].to_numpy(dtype=float)
        events = raw_events_s[valid_mask].to_numpy(dtype=int)
        n_samples = len(times)
        n_events = int(np.sum(events == 1))

        if n_samples < 20 or n_events < 5:
            raise ValueError(
                f"Insufficient survival data for Cox distillation: N={n_samples}, events={n_events}. "
                "At least 20 samples and 5 events are required for survival modeling."
            )

        censoring_rate = round(float(1.0 - (n_events / n_samples)), 4)
        logger.info(
            "Fitting MinimalGenePanelDistiller (N=%d, Events=%d, Censoring=%.1f%%, P=%d features)",
            n_samples, n_events, censoring_rate * 100, n_features
        )

        # ─── Step 0c: Patient Identification & Cross-Validation Strategy ──────
        extracted_patients: Optional[np.ndarray] = None
        if patient_ids is not None:
            raw_pt = np.asarray(extract_tcga_patient_ids(patient_ids))
            extracted_patients = raw_pt[valid_mask]
        elif sample_ids is not None:
            raw_sample = np.asarray(extract_tcga_patient_ids(sample_ids))
            if any(str(s).startswith("TCGA-") for s in sample_ids):
                extracted_patients = raw_sample[valid_mask]
            else:
                logger.info(
                    "Sample IDs provided do not contain TCGA patient barcodes; "
                    "preserving ungrouped status as patient identifiers are unavailable."
                )
        elif hasattr(df_X, "index") and len(df_X.index) > 0:
            first_idx = str(df_X.index[0])
            if first_idx.startswith("TCGA-"):
                # df_X was already filtered by valid_mask, so df_X.index is already aligned
                extracted_patients = np.asarray(extract_tcga_patient_ids(df_X.index))
                logger.info(
                    "Auto-extracted TCGA patient barcodes from sample index (N=%d samples -> %d unique patients)",
                    n_samples, len(np.unique(extracted_patients))
                )

        if extracted_patients is not None:
            unique_patients = np.unique(extracted_patients)
            n_patients = len(unique_patients)
            if n_patients < 2:
                raise ValueError(
                    f"Patient-level cross-validation cannot be performed: only {n_patients} unique patient(s) "
                    "available. At least 2 unique patients are required to partition data into disjoint training and validation sets."
                )

            effective_folds = min(self.n_folds, n_patients)
            gkf = GroupKFold(n_splits=effective_folds)
            splits = list(gkf.split(X, groups=extracted_patients))

            if effective_folds < self.n_folds:
                cv_strategy = (
                    f"Patient-Grouped GroupKFold (N={n_patients} patients, "
                    f"adjusted to {effective_folds} folds from requested {self.n_folds} due to patient count)"
                )
            else:
                cv_strategy = f"Patient-Grouped GroupKFold (N={n_patients} patients, {effective_folds} folds)"
            patient_grouping_status = "patient_grouped_strictly_partitioned"

            # Adversarial verification: verify 0 patient overlap across all folds
            for f_i, (tr_i, val_i) in enumerate(splits):
                overlap = set(extracted_patients[tr_i]).intersection(set(extracted_patients[val_i]))
                if len(overlap) > 0:
                    raise ValueError(f"Patient overlap detected in fold {f_i}: {overlap}")
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X))
            cv_strategy = f"Row-Level KFold ({self.n_folds} folds, unverified patient grouping - patient IDs unavailable)"
            patient_grouping_status = "row_level_ungrouped_patient_ids_unavailable"
            n_patients = n_samples

        # ─── Step 1: Comparator Definition & OOF Buffer Initialization ───────
        original_feature_count = n_features
        if n_features > self.max_screen_features:
            comparator_type = "screened_high_dimensional_comparator"
            comparator_feature_count = self.max_screen_features
            screened_feature_count = self.max_screen_features
        else:
            comparator_type = "complete_feature_universe_model"
            comparator_feature_count = n_features
            screened_feature_count = n_features

        # Strict integrity: Initialize OOF predictions to NaN (never zero)
        oof_comparator_scores = np.full(n_samples, np.nan, dtype=float)
        max_possible_size = min(n_features, self.max_screen_features)
        curve_sizes = sorted(list(set(
            [s for s in self.panel_sizes if s <= max_possible_size] +
            [s for s in [5, 10, min(50, max_possible_size)] if s <= max_possible_size]
        )))
        oof_panel_scores: Dict[int, np.ndarray] = {
            s: np.full(n_samples, np.nan, dtype=float) for s in curve_sizes
        }

        selection_counts = np.zeros(n_features, dtype=float)
        summed_coefs = np.zeros(n_features, dtype=float)
        fold_diagnostics: List[Dict[str, Any]] = []

        effective_require_complete_oof = self.require_complete_oof if require_complete_oof is None else require_complete_oof

        # ─── Step 2: Out-of-Fold Nested Cross-Validation Loop ────────────────
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X[train_idx], X[val_idx]
            t_tr, t_val = times[train_idx], times[val_idx]
            e_tr, e_val = events[train_idx], events[val_idx]

            fold_diag = {
                "fold_idx": fold_idx,
                "n_train_samples": len(train_idx),
                "n_val_samples": len(val_idx),
                "n_train_events": int(np.sum(e_tr == 1)),
                "n_val_events": int(np.sum(e_val == 1)),
                "fit_success": False,
                "comparator_fit_success": False,
                "panel_fit_success": {size: False for size in oof_panel_scores.keys()},
                "failure_reason": None,
                "failure_reasons": [],
                "panel_failure_reasons": {size: None for size in oof_panel_scores.keys()},
                "fitted_comparator_features": 0,
                "patient_overlap_count": 0,
                "patient_overlap_detected": False,
                "n_train_patients": len(train_idx),
                "n_val_patients": len(val_idx),
            }

            if extracted_patients is not None:
                tr_pts = set(extracted_patients[train_idx])
                val_pts = set(extracted_patients[val_idx])
                overlap = tr_pts.intersection(val_pts)
                fold_diag["patient_overlap_count"] = len(overlap)
                fold_diag["patient_overlap_detected"] = len(overlap) > 0
                fold_diag["n_train_patients"] = len(tr_pts)
                fold_diag["n_val_patients"] = len(val_pts)

            if np.sum(e_tr == 1) < 3:
                reason = "Insufficient training events (<3)"
                fold_diag["failure_reason"] = reason
                fold_diag["failure_reasons"].append(reason)
                fold_diagnostics.append(fold_diag)
                logger.warning("Fold %d has < 3 events in training set; skipping.", fold_idx)
                continue

            # Scale strictly on training split
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)

            # Feature screening strictly within training fold if p > max_screen_features
            if n_features > self.max_screen_features:
                univ_scores = _compute_univariate_survival_concordance(X_tr_sc, t_tr, e_tr)
                screened_indices = np.argsort(-univ_scores)[:self.max_screen_features]
            else:
                screened_indices = np.arange(n_features)

            n_screened = len(screened_indices)
            fold_counts = np.zeros(n_screened, dtype=float)
            fold_coefs = np.zeros(n_screened, dtype=float)

            rng = np.random.RandomState(self.random_state + fold_idx * 17)
            n_tr = len(X_tr)

            for b in range(self.n_subsamples):
                sub_idx = rng.choice(n_tr, size=int(0.85 * n_tr), replace=False)
                if np.sum(e_tr[sub_idx] == 1) < 2:
                    continue

                df_sub = pd.DataFrame(
                    X_tr_sc[sub_idx][:, screened_indices],
                    columns=[f"F_{k}" for k in range(n_screened)]
                )
                df_sub["time"] = t_tr[sub_idx]
                df_sub["event"] = e_tr[sub_idx]

                try:
                    cph_sub = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
                    cph_sub.fit(df_sub, duration_col="time", event_col="event")
                    b_coefs = cph_sub.params_.values
                    active = np.abs(b_coefs) > 1e-4
                    fold_counts += active.astype(float)
                    fold_coefs += b_coefs
                except Exception:
                    continue

            # Record selection counts for global stability
            effective_subs = max(1.0, float(self.n_subsamples))
            for local_k, orig_k in enumerate(screened_indices):
                selection_counts[orig_k] += fold_counts[local_k] / effective_subs
                summed_coefs[orig_k] += fold_coefs[local_k] / effective_subs

            # Fold ranking of features
            local_scores = fold_counts / effective_subs + 0.3 * np.abs(fold_coefs / effective_subs)
            local_ranked = screened_indices[np.argsort(-local_scores)]

            # 1. Fitted Comparator Model out-of-fold prediction
            try:
                df_tr_full = pd.DataFrame(
                    X_tr_sc[:, screened_indices],
                    columns=[f"F_{k}" for k in range(n_screened)]
                )
                df_tr_full["time"] = t_tr
                df_tr_full["event"] = e_tr
                cph_full = CoxPHFitter(penalizer=0.05, l1_ratio=0.1)
                cph_full.fit(df_tr_full, duration_col="time", event_col="event")
                full_b = cph_full.params_.values
                oof_comparator_scores[val_idx] = np.dot(X_val_sc[:, screened_indices], full_b)
                fold_diag["comparator_fit_success"] = True
                fold_diag["fitted_comparator_features"] = n_screened
            except Exception as e:
                err_msg = f"Comparator fit failed: {str(e)}"
                fold_diag["comparator_fit_success"] = False
                fold_diag["failure_reasons"].append(err_msg)
                logger.warning("Comparator fold %d fit exception: %s", fold_idx, e)

            # 2. Candidate Panel out-of-fold predictions
            for size in oof_panel_scores.keys():
                top_k = local_ranked[:size]
                if len(top_k) < size:
                    fold_diag["panel_fit_success"][size] = False
                    err_msg = f"Insufficient ranked features for panel size {size}: {len(top_k)} < {size}"
                    fold_diag["panel_failure_reasons"][size] = err_msg
                    fold_diag["failure_reasons"].append(err_msg)
                    continue
                try:
                    df_tr_panel = pd.DataFrame(
                        X_tr_sc[:, top_k],
                        columns=[f"P_{k}" for k in range(len(top_k))]
                    )
                    df_tr_panel["time"] = t_tr
                    df_tr_panel["event"] = e_tr
                    cph_panel = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
                    cph_panel.fit(df_tr_panel, duration_col="time", event_col="event")
                    panel_b = cph_panel.params_.values
                    oof_panel_scores[size][val_idx] = np.dot(X_val_sc[:, top_k], panel_b)
                    fold_diag["panel_fit_success"][size] = True
                except Exception as e:
                    err_msg = f"Panel size {size} fit failed: {str(e)}"
                    fold_diag["panel_fit_success"][size] = False
                    fold_diag["panel_failure_reasons"][size] = str(e)
                    fold_diag["failure_reasons"].append(err_msg)
                    logger.warning("Panel size %d fold %d fit exception: %s", size, fold_idx, e)

            all_panels_ok = all(fold_diag["panel_fit_success"].values()) if len(fold_diag["panel_fit_success"]) > 0 else True
            fold_diag["fit_success"] = bool(fold_diag["comparator_fit_success"] and all_panels_ok)
            if not fold_diag["fit_success"]:
                fold_diag["failure_reason"] = "; ".join(fold_diag["failure_reasons"]) if fold_diag["failure_reasons"] else "Fold fit failed"

            fold_diagnostics.append(fold_diag)

        # ─── Step 3: Out-of-Fold Prediction Coverage & Metric Honesty ────────
        valid_comp_mask = ~np.isnan(oof_comparator_scores)
        missing_comparator_count = int(np.sum(~valid_comp_mask))
        oof_coverage_ratio = round(float(np.sum(valid_comp_mask)) / max(n_samples, 1), 4)

        any_panel_missing = any(np.isnan(oof_panel_scores[s]).any() for s in oof_panel_scores)
        failed_folds = [f for f in fold_diagnostics if not f["fit_success"]]
        has_missing = (missing_comparator_count > 0) or any_panel_missing or (len(failed_folds) > 0)

        if has_missing:
            evaluation_status = "incomplete_missing_predictions"
            logger.warning(
                "Incomplete OOF predictions: %d/%d samples lack valid comparator predictions. "
                "Any panel missing: %s. Failed folds: %d/%d",
                missing_comparator_count, n_samples, any_panel_missing, len(failed_folds), len(splits)
            )
            if effective_require_complete_oof:
                raise RuntimeError(
                    f"Cross-validation incomplete: {missing_comparator_count}/{n_samples} samples lack valid comparator predictions, "
                    f"panel missing: {any_panel_missing}. Failed folds: {len(failed_folds)}. "
                    f"Reasons: {[f['failure_reason'] for f in failed_folds]}"
                )
        else:
            evaluation_status = "complete"

        # Evaluate comparator model on valid OOF predictions
        if np.sum(valid_comp_mask) > 0 and np.sum(events[valid_comp_mask] == 1) > 0:
            full_res = compute_harrell_c_index_with_ci(
                times[valid_comp_mask], events[valid_comp_mask], oof_comparator_scores[valid_comp_mask], n_bootstrap=100
            )
            full_c = full_res["c_index"]
            full_ci = [full_res["ci_lower"], full_res["ci_upper"]]
            full_uno = compute_unos_c_index(
                scores=oof_comparator_scores[valid_comp_mask],
                times=times[valid_comp_mask],
                events=events[valid_comp_mask]
            )
        else:
            full_c, full_ci, full_uno = None, None, None

        # ─── Step 4: Evaluate Candidate Panels & Retention vs Comparator ─────
        retention_curve: List[Dict[str, Any]] = []
        for size in sorted(oof_panel_scores.keys()):
            p_scores = oof_panel_scores[size]
            valid_p_mask = ~np.isnan(p_scores)
            n_valid_p = int(np.sum(valid_p_mask))

            if n_valid_p > 0 and np.sum(events[valid_p_mask] == 1) > 0:
                sub_res = compute_harrell_c_index_with_ci(
                    times[valid_p_mask], events[valid_p_mask], p_scores[valid_p_mask], n_bootstrap=100
                )
                sub_c = sub_res["c_index"]
                sub_ci = [sub_res["ci_lower"], sub_res["ci_upper"]]
                sub_uno = compute_unos_c_index(
                    scores=p_scores[valid_p_mask],
                    times=times[valid_p_mask],
                    events=events[valid_p_mask]
                )
            else:
                sub_c, sub_ci, sub_uno = None, None, None

            if full_c is not None and full_c > 0.0 and sub_c is not None:
                retention = round(float(sub_c / full_c), 4)
            else:
                retention = 0.0

            n_folds_total = max(1, len(splits))
            fold_panel_successes = sum(
                1 for f in fold_diagnostics
                if f.get("panel_fit_success", {}).get(size, False)
            )
            panel_fold_rate = round(float(fold_panel_successes) / float(n_folds_total), 4)

            retention_curve.append({
                "panel_size": size,
                "c_index": sub_c,
                "ci_95": sub_ci,
                "unos_c_index": sub_uno,
                "retention_ratio": retention,
                "oof_valid_samples": n_valid_p,
                "oof_missing_samples": n_samples - n_valid_p,
                "oof_coverage_ratio": round(float(n_valid_p) / max(n_samples, 1), 4),
                "evaluation_status": "complete" if n_valid_p == n_samples else "incomplete_missing_predictions",
                "panel_fit_fold_success_rate": panel_fold_rate,
            })

        # ─── Step 5: Global Consensus Stability Ranking on Full Data ─────────
        total_folds = max(1.0, float(len(splits)))
        stability_prob = selection_counts / total_folds
        mean_coefs = summed_coefs / total_folds

        self.feature_stability_scores_ = {
            feature_names[i]: round(float(stability_prob[i]), 4) for i in range(n_features)
        }
        self.feature_weights_ = {
            feature_names[i]: round(float(mean_coefs[i]), 4) for i in range(n_features)
        }

        composite_score = stability_prob * 0.7 + (np.abs(mean_coefs) / (np.max(np.abs(mean_coefs)) + 1e-8)) * 0.3
        ranked_indices = np.argsort(-composite_score)

        stability_rankings = [
            {
                "rank": rank + 1,
                "gene": feature_names[idx],
                "stability_score": round(float(stability_prob[idx]), 4),
                "mean_coefficient": round(float(mean_coefs[idx]), 4),
            }
            for rank, idx in enumerate(ranked_indices[:min(100, n_features)])
        ]

        # ─── Step 6: Final Candidate Panels Annotation ───────────────────────
        candidate_panels: Dict[int, CandidatePanel] = {}

        # Scale full dataset for consensus panel weight calculation
        sc_full = StandardScaler()
        X_full_sc = sc_full.fit_transform(X)

        for size in self.panel_sizes:
            if size > n_features:
                continue
            top_idx = ranked_indices[:size]
            sub_genes = [feature_names[i] for i in top_idx]

            # Fit consensus weights
            sub_weights: Dict[str, float] = {}
            try:
                df_consensus = pd.DataFrame(
                    X_full_sc[:, top_idx],
                    columns=[f"G_{k}" for k in range(size)]
                )
                df_consensus["time"] = times
                df_consensus["event"] = events
                cph_cons = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
                cph_cons.fit(df_consensus, duration_col="time", event_col="event")
                sub_weights = {
                    sub_genes[i]: round(float(cph_cons.params_.values[i]), 4)
                    for i in range(len(sub_genes))
                }
            except Exception:
                sub_weights = {
                    sub_genes[i]: round(float(mean_coefs[top_idx[i]]), 4)
                    for i in range(len(sub_genes))
                }

            # Extract corresponding out-of-fold metrics from retention curve
            curve_entry = next((entry for entry in retention_curve if entry["panel_size"] == size), None)
            if curve_entry:
                panel_c = curve_entry["c_index"]
                panel_ci = curve_entry["ci_95"]
                panel_uno = curve_entry["unos_c_index"]
                retention = curve_entry["retention_ratio"]
            else:
                panel_c, panel_ci, panel_uno, retention = None, None, None, 0.0

            hallmarks = self._annotate_hallmarks(sub_genes)
            biotypes = self._annotate_biotypes(sub_genes)
            assay_rec = self._recommend_assay(size)

            candidate_panels[size] = CandidatePanel(
                panel_size=size,
                genes=sub_genes,
                weights=sub_weights,
                c_index=panel_c,
                c_index_ci=panel_ci,
                unos_c_index=panel_uno,
                retention_ratio=retention,
                hallmark_representation=hallmarks,
                biotypes=biotypes,
                assay_type_recommendation=assay_rec,
                validation_status="in_silico_candidate",
                evidence_level="in_silico_unvalidated",
                sample_count=n_samples,
                event_count=n_events,
                fitted_feature_count=len(sub_genes),
                comparator_feature_count=comparator_feature_count,
                comparator_type=comparator_type,
                oof_coverage_ratio=curve_entry["oof_coverage_ratio"] if curve_entry else 0.0,
                oof_missing_count=curve_entry["oof_missing_samples"] if curve_entry else n_samples,
                panel_evaluation_status=curve_entry["evaluation_status"] if curve_entry else "incomplete_missing_predictions",
                panel_fit_fold_success_rate=curve_entry["panel_fit_fold_success_rate"] if curve_entry else 0.0,
            )

        limitations = [
            "In-silico candidate gene panels require prospective analytical and clinical validation.",
            "Primer and probe efficiency has not been measured experimentally.",
            f"Performance retention ({retention * 100:.1f}%) is calculated strictly relative to the "
            f"{comparator_feature_count}-gene {comparator_type}, NOT against the complete {original_feature_count}-gene universe.",
            f"Cross-validation executed via {cv_strategy}; patient grouping status: {patient_grouping_status}.",
        ]
        if is_synthetic:
            limitations.append("EVALUATED ON SYNTHETIC BENCHMARK DATASET - NOT REAL PATIENT BIOLOGY.")
        if evaluation_status != "complete":
            limitations.append(f"WARNING: Out-of-fold evaluation status is '{evaluation_status}'.")

        report = DistillationReport(
            data_source=data_source,
            is_synthetic=is_synthetic,
            endpoint=endpoint,
            evaluation_method=f"Nested Cross-Validation ({cv_strategy})",
            n_samples=n_samples,
            n_events=n_events,
            censoring_rate=censoring_rate,
            full_model_gene_count=comparator_feature_count,
            full_model_c_index=full_c,
            full_model_c_index_ci=full_ci,
            full_model_unos_c_index=full_uno,
            candidate_panels=candidate_panels,
            retention_curve=retention_curve,
            stability_rankings=stability_rankings,
            limitations=limitations,
            original_feature_count=original_feature_count,
            screened_feature_count=screened_feature_count,
            comparator_feature_count=comparator_feature_count,
            comparator_type=comparator_type,
            comparator_c_index=full_c,
            comparator_c_index_ci=full_ci,
            comparator_unos_c_index=full_uno,
            cv_strategy=cv_strategy,
            patient_count=n_patients,
            patient_grouping_status=patient_grouping_status,
            oof_coverage_ratio=oof_coverage_ratio,
            oof_missing_count=missing_comparator_count,
            fold_diagnostics=fold_diagnostics,
            evaluation_status=evaluation_status,
        )
        self.oof_comparator_scores_ = oof_comparator_scores
        self.oof_panel_scores_ = oof_panel_scores
        self.distillation_report_ = report
        return report

    def _annotate_hallmarks(self, genes: List[str]) -> Dict[str, int]:
        """Maps panel genes to biological cancer hallmark pathways."""
        hallmark_counts: Dict[str, int] = {}
        gene_set = set(genes)
        for hallmark, hallmark_genes in HALLMARK_ANNOTATIONS.items():
            overlap = len(gene_set.intersection(hallmark_genes))
            if overlap > 0:
                hallmark_counts[hallmark] = overlap
        if not hallmark_counts:
            hallmark_counts["Multi-Pathway Prognosis"] = len(genes)
        return hallmark_counts

    def _annotate_biotypes(self, genes: List[str]) -> Dict[str, int]:
        """Annotates genes with genomic biotypes (Protein-coding, lncRNA, pseudogene)."""
        biotypes = {"protein_coding": 0, "lncRNA": 0, "other": 0}
        for g in genes:
            if g.startswith("LINC") or g.endswith("-AS1") or g.endswith("-IT1"):
                biotypes["lncRNA"] += 1
            elif g.startswith("MIR") or g.startswith("SNORD"):
                biotypes["other"] += 1
            else:
                biotypes["protein_coding"] += 1
        return biotypes

    def _recommend_assay(self, panel_size: int) -> str:
        """Recommends targeted diagnostic assay technology candidate based on gene count."""
        if panel_size <= 20:
            return "RT-qPCR TaqMan Low-Density Array (TLDA) / Multiplex PCR (Candidate Protocol - Unvalidated)"
        elif panel_size <= 40:
            return "NanoString nCounter Elements / Digital Spatial Profiling (Candidate Protocol - Unvalidated)"
        else:
            return "Illumina AmpliSeq Targeted RNA Library Prep (Candidate Protocol - Unvalidated)"

    def export_assay_specifications(
        self,
        panel_size: int,
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Exports in-silico assay design specifications to disk with explicit validation status.
        """
        if not self.distillation_report_ or panel_size not in self.distillation_report_.candidate_panels:
            raise ValueError(f"Panel size {panel_size} not available in fitted distillation report.")

        panel = self.distillation_report_.candidate_panels[panel_size]
        report = self.distillation_report_
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        oof_c_desc = f"{report.comparator_c_index:.4f}" if report.comparator_c_index is not None else "N/A"
        spec = {
            "title": f"GUIDO In-Silico Assay Design Specification — {panel_size}-Gene Candidate Panel",
            "validation_status": "in_silico_candidate",
            "regulatory_status": "Research Use Only (RUO) - Unvalidated Assay Design",
            "is_synthetic_training": report.is_synthetic,
            "training_data_source": report.data_source,
            "survival_endpoint": report.endpoint,
            "panel_size": panel_size,
            "fitted_gene_count": len(panel.genes),
            "target_genes": panel.genes,
            "gene_weights": panel.weights,
            "original_feature_universe_count": report.original_feature_count,
            "screened_feature_count": report.screened_feature_count,
            "comparator_model_feature_count": report.comparator_feature_count,
            "comparator_model_type": report.comparator_type,
            "retention_against_comparator": panel.retention_ratio,
            "comparator_retention_basis": (
                f"Retention ratio ({panel.retention_ratio * 100:.1f}%) is calculated strictly relative "
                f"to the {report.comparator_feature_count}-gene {report.comparator_type} (OOF C-index: {oof_c_desc}), "
                f"NOT against the complete {report.original_feature_count}-gene universe."
            ),
            "cv_strategy": report.cv_strategy,
            "patient_grouping_status": report.patient_grouping_status,
            "oof_prediction_coverage": report.oof_coverage_ratio,
            "evaluation_status": report.evaluation_status,
            "panel_oof_coverage_ratio": panel.oof_coverage_ratio,
            "panel_oof_missing_count": panel.oof_missing_count,
            "panel_evaluation_status": panel.panel_evaluation_status,
            "panel_fit_fold_success_rate": panel.panel_fit_fold_success_rate,
            "housekeeping_candidate_controls": {
                "qualified_stable_controls": QUALIFIED_HOUSEKEEPING_CONTROLS,
                "tissue_variable_controls_to_avoid": TISSUE_VARIABLE_HOUSEKEEPING_CONTROLS,
                "empirical_basis": (
                    "Empirically evaluated across real TCGA cohorts in cross_platform_normalizer.py; "
                    "RPL13A, ACTB, PPIA, and TBP demonstrate cross-tissue CV < 0.05, while HPRT1, B2M, "
                    "and PGK1 exhibit significant tissue-specific variation (>0.9 log2 shift)."
                ),
            },
            "recommended_platform_candidate": panel.assay_type_recommendation,
            "out_of_fold_metrics": {
                "c_index": panel.c_index,
                "c_index_95_ci": panel.c_index_ci,
                "unos_c_index": panel.unos_c_index,
                "retention_ratio": panel.retention_ratio,
                "n_samples": panel.sample_count,
                "n_events": panel.event_count,
            },
            "hallmark_coverage": panel.hallmark_representation,
            "biotype_distribution": panel.biotypes,
            "in_silico_design_guidelines": {
                "amplicon_length_bp": "60-120 bp (recommended for FFPE tissue RNA)",
                "primer_melting_temp_c": "58-62 deg C",
                "normalization_protocol": "Geometric mean of qualified housekeeping controls (RPL13A, ACTB, PPIA, TBP); avoid solitary reliance on HPRT1/B2M.",
            },
            "missing_evidence": [
                "Primer/probe amplification efficiency (must be verified between 90% and 110% with R^2 >= 0.98).",
                "Limit of Detection (LoD) and Limit of Blank (LoB) on titrated FFPE RNA reference materials.",
                "Analytical specificity and cross-hybridization assessment across homologous transcript isoforms.",
                "Linear dynamic range across >= 4 orders of magnitude.",
                "Intra-assay and inter-assay repeatability/reproducibility (CV < 8% across technical replicates).",
                "Turnaround time (24-48h target) and reagent costs (<$200 target) are aspirational estimates and have not been validated with an accredited laboratory partner.",
                "Prospective clinical utility (impact on adjuvant treatment decisions and patient outcome) must be demonstrated in multi-center clinical trials prior to diagnostic use.",
            ],
            "notice": "This document is an in-silico research specification and must not be used as an approved clinical assay protocol."
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)

        logger.info("Exported assay design specification to: %s", out_p)
        return spec
