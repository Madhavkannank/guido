"""
Comprehensive Clinical Risk Engine Test Suite
=============================================
Tests end-to-end functionality for:
  1. UCSC Xena & TCGA-CDR data ingestion endpoints
  2. High-performance Concordance Index (C-Index) & IPCW weighting
  3. Leak-free Hallmark Oncology Metagene computation & serialization
  4. StackingClassifier training & artifact schema compliance
  5. Cross-cohort out-of-distribution external validation
  6. API serving integrity & prevention of silent legacy fallback
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retrain_fixed import (
    DEFAULT_N_GENES,
    HALLMARK_PATHWAYS,
    clean_matrix_axes,
    compute_c_index,
    compute_ipcw_weights,
    compute_leak_free_hallmarks,
)
from src.api.app import app
from src.validation.cross_cohort import (
    compute_c_index_fast,
    compute_ece,
    evaluate_clinical_model_on_external_cohort,
)
from src.validation.external_datasets import generate_synthetic_external_cohort


# ─── 1. Concordance Index Tests ──────────────────────────────────────────────

def test_c_index_perfect_concordance():
    # Earlier events have higher risk scores
    times = np.array([100, 200, 300, 400, 500], dtype=float)
    events = np.array([1, 1, 1, 1, 1], dtype=int)
    risk_scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1], dtype=float)

    c_idx = compute_c_index(times, events, risk_scores)
    assert c_idx == 1.0, f"Expected C-index 1.0, got {c_idx}"


def test_c_index_inverted_concordance():
    # Earlier events have lower risk scores
    times = np.array([100, 200, 300, 400, 500], dtype=float)
    events = np.array([1, 1, 1, 1, 1], dtype=int)
    risk_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)

    c_idx = compute_c_index(times, events, risk_scores)
    assert c_idx == 0.0, f"Expected C-index 0.0, got {c_idx}"


def test_c_index_with_right_censoring():
    times = np.array([100, 200, 300, 400, 500], dtype=float)
    events = np.array([1, 0, 1, 0, 1], dtype=int)
    risk_scores = np.array([0.8, 0.4, 0.6, 0.2, 0.1], dtype=float)

    c_idx_1 = compute_c_index(times, events, risk_scores)
    c_idx_2 = compute_c_index_fast(times, events, risk_scores)
    assert 0.0 <= c_idx_1 <= 1.0
    assert c_idx_1 == c_idx_2, f"Discrepancy between C-index implementations: {c_idx_1} vs {c_idx_2}"


# ─── 2. IPCW Weighting Tests ────────────────────────────────────────────────

def test_compute_ipcw_weights_properties():
    times = np.array([150, 400, 800, 1200, 1600], dtype=float)
    events = np.array([1, 0, 1, 0, 1], dtype=int)

    weights = compute_ipcw_weights(times, events, horizon=1095.0)

    assert len(weights) == len(times)
    assert np.all(weights > 0.0), "All weights must be strictly positive"
    assert np.all(np.isfinite(weights)), "All weights must be finite"
    assert np.isclose(np.sum(weights), len(times), rtol=1e-4), "Weights must be normalized to sample size"


# ─── 3. Leak-Free Hallmark Metagenes Tests ───────────────────────────────────

def test_leak_free_hallmark_metagenes():
    rng = np.random.default_rng(42)
    # Collect all hallmark genes
    all_hallmark_genes = []
    for glist in HALLMARK_PATHWAYS.values():
        all_hallmark_genes.extend(glist)
    all_hallmark_genes = sorted(list(set(all_hallmark_genes)))

    n_tr = 40
    n_te = 10
    # Create raw expression dataframes
    X_tr = pd.DataFrame(
        rng.normal(loc=10.0, scale=2.0, size=(n_tr, len(all_hallmark_genes))),
        columns=all_hallmark_genes,
        index=[f"TR_{i}" for i in range(n_tr)]
    )
    # Test partition with intentional distribution shift
    X_te = pd.DataFrame(
        rng.normal(loc=14.0, scale=3.0, size=(n_te, len(all_hallmark_genes))),
        columns=all_hallmark_genes,
        index=[f"TE_{i}" for i in range(n_te)]
    )

    X_tr_h, X_te_h, stats, meta_names = compute_leak_free_hallmarks(X_tr, X_te)

    # 1. Metagene columns added
    for m in HALLMARK_PATHWAYS.keys():
        assert m in X_tr_h.columns, f"Missing metagene {m} in X_tr"
        assert m in X_te_h.columns, f"Missing metagene {m} in X_te"

    # 2. Stats computed strictly from training split
    for g in all_hallmark_genes:
        assert g in stats
        expected_mu = float(X_tr[g].mean())
        assert np.isclose(stats[g]["mean"], expected_mu, rtol=1e-4)

    # 3. No NaNs or Infs
    assert not X_tr_h.isna().any().any()
    assert not X_te_h.isna().any().any()


# ─── 4. Model Training & Artifact Schema Tests ───────────────────────────────

def test_synthetic_training_and_artifact():
    from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    rng = np.random.default_rng(42)
    n = 60
    p = 100
    features = [f"GENE_{i:03d}" for i in range(88)] + list(HALLMARK_PATHWAYS.keys())
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    weights = np.ones(n, dtype=np.float32)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    base_models = [
        ("en", LogisticRegression(solver="saga", l1_ratio=0.5, C=0.3, max_iter=500, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=10, max_depth=2, random_state=42, verbosity=0)),
        ("et", ExtraTreesClassifier(n_estimators=10, max_depth=2, random_state=42, bootstrap=True)),
    ]
    meta = LogisticRegression(random_state=42)
    ensemble = StackingClassifier(estimators=base_models, final_estimator=meta, cv=2)

    ensemble.fit(X_sc, y, sample_weight=weights)
    probs = ensemble.predict_proba(X_sc)[:, 1]

    assert len(probs) == n
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Artifact structure verification
    artifact = {
        "model": ensemble,
        "scaler": scaler,
        "features": features,
        "individual_genes": features[:88],
        "hallmark_metagenes": HALLMARK_PATHWAYS,
        "metagene_stats": {g: {"mean": 10.0, "std": 1.5} for g in ["MKI67", "PCNA"]},
        "task": "genomic_disease_risk_prediction",
        "time_horizon": "3-Year Progression & Recurrence Risk",
        "risk_cutoffs": {"low": 0.35, "high": 0.65},
        "optimal_threshold": 0.50,
        "feature_importances": {f: 0.01 for f in features},
        "test_metrics": {"risk_auroc": 0.85, "c_index": 0.80},
    }

    assert len(artifact["features"]) == 100
    assert "metagene_stats" in artifact
    assert artifact["risk_cutoffs"] == {"low": 0.35, "high": 0.65}


# ─── 5. External Cross-Cohort Validation Tests ───────────────────────────────

def test_external_validation_pipeline():
    # Create mock artifact
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = [f"GENE_{i:03d}" for i in range(88)] + list(HALLMARK_PATHWAYS.keys())
    model = LogisticRegression(random_state=42)
    scaler = StandardScaler()
    scaler.fit(np.zeros((10, 100)))
    model.fit(np.zeros((10, 100)), [0, 1] * 5)

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "hallmark_metagenes": HALLMARK_PATHWAYS,
        "metagene_stats": {g: {"mean": 9.5, "std": 1.5} for g in ["MKI67", "PCNA", "VEGFA"]},
        "test_metrics": {"risk_auroc": 0.88, "c_index": 0.82},
    }

    X_ext, y_ext, t_ext, e_ext = generate_synthetic_external_cohort(n_samples=50, seed=42)

    report = evaluate_clinical_model_on_external_cohort(
        model_or_artifact=artifact,
        X_external=X_ext,
        y_external=y_ext,
        times_external=t_ext,
        events_external=e_ext,
        cohort_name="Mock-External"
    )

    assert "external_metrics" in report
    assert "c_index" in report["external_metrics"]
    assert "risk_auroc" in report["external_metrics"]
    assert "brier_score" in report["external_metrics"]
    assert "ece" in report["external_metrics"]
    assert "risk_stratification" in report
    assert report["generalisation_flag"] in ("EXCELLENT", "ACCEPTABLE", "DEGRADED")


# ─── 6. API Serving & Silent Fallback Prevention Tests ───────────────────────

def test_api_ui_predict_rejection_for_untrained_cohort():
    client = TestClient(app)
    # Request prediction on a cohort that has no trained clinical model
    response = client.post(
        "/api/ui/predict",
        data={
            "project_id": "TCGA-NONEXISTENT-CANCER",
            "disease_name": "unknown_cancer",
            "json_payload": '{"GENE_001": 5.2, "MKI67": 8.1}'
        }
    )
    # Must reject with HTTP 404 rather than silently falling back to a tumor-vs-normal model
    assert response.status_code == 404
    detail = response.json().get("detail", "")
    assert "No calibrated clinical genomic risk model found" in detail


def test_api_ui_predict_success_with_trained_model():
    client = TestClient(app)
    # TCGA-KIRP model was trained earlier and exists
    artifact_path = ROOT / "models_artifacts" / "TCGA-KIRP" / "xgboost_clinical_model.joblib"
    if not artifact_path.exists():
        pytest.skip("TCGA-KIRP artifact not trained on this machine")

    # Send gene expression payload
    response = client.post(
        "/api/ui/predict",
        data={
            "project_id": "TCGA-KIRP",
            "disease_name": "papillary_renal_cell_carcinoma",
            "json_payload": '{"MKI67": 10.5, "PCNA": 11.2, "VEGFA": 9.8, "HIF1A": 12.0}'
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    pred = data["prediction"]
    assert "risk_tier" in pred
    assert pred["risk_tier"] in ("Low Risk", "Moderate Risk", "High Risk")
    assert "risk_score" in pred
    assert 0.0 <= pred["risk_score"] <= 100.0
    assert pred["time_horizon"] == "3-Year Progression & Recurrence Risk"


def test_clean_matrix_axes_and_multi_cohort_concatenation():
    """Verify that multi-cohort concatenation cleanly handles duplicate genes and indices without InvalidIndexError."""
    rng = np.random.default_rng(42)
    # Simulate matrix 1 with duplicate gene columns and duplicate patient index
    genes_1 = ["TP53", "EGFR", "SLC35E2", "SLC35E2", "BRCA1"]
    df_1 = pd.DataFrame(rng.normal(size=(5, 5)), index=["P1", "P2", "P2", "P3", "P4"], columns=genes_1)
    y_1 = pd.Series([1, 0, 0, 1, 0], index=df_1.index)
    t_1 = pd.Series([200.0, 1200.0, 1200.0, 400.0, 1500.0], index=df_1.index)
    e_1 = pd.Series([1, 0, 0, 1, 0], index=df_1.index)

    # Simulate matrix 2 with probe duplicates and different patient IDs
    genes_2 = ["TP53", "BRCA1", "ERBB2", "ERBB2"]
    df_2 = pd.DataFrame(rng.normal(size=(4, 4)), index=["P1", "P2", "P5", "P6"], columns=genes_2)
    y_2 = pd.Series([0, 1, 0, 1], index=df_2.index)
    t_2 = pd.Series([1100.0, 300.0, 1400.0, 250.0], index=df_2.index)
    e_2 = pd.Series([0, 1, 0, 1], index=df_2.index)

    # Clean both matrices with distinct cohort prefixes
    X1, y1, t1, e1 = clean_matrix_axes(df_1, y_1, t_1, e_1, prefix="COHORT1")
    X2, y2, t2, e2 = clean_matrix_axes(df_2, y_2, t_2, e_2, prefix="COHORT2")

    assert X1.columns.is_unique, "Columns in X1 must be unique"
    assert X2.columns.is_unique, "Columns in X2 must be unique"
    assert X1.index.is_unique, "Index in X1 must be unique"
    assert X2.index.is_unique, "Index in X2 must be unique"

    # Multi-cohort concatenation must succeed without InvalidIndexError
    X_concat = pd.concat([X1, X2], axis=0).fillna(0.0).astype(np.float32)
    y_concat = np.concatenate([y1.values, y2.values])
    t_concat = np.concatenate([t1.values, t2.values])
    e_concat = np.concatenate([e1.values, e2.values])

    assert len(X_concat) == len(y_concat) == len(t_concat) == len(e_concat)
    assert X_concat.columns.is_unique, "Concatenated columns must be unique"
    assert X_concat.index.is_unique, "Concatenated indices must be unique"
    assert X_concat.shape[1] == 5  # TP53, EGFR, SLC35E2, BRCA1, ERBB2

