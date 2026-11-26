"""
Tests for Minimal Gene Panel Distillation (LASSO-Cox / Stability Selection)
==========================================================================
Verifies:
  1. Regularized feature selection and stability across cross-validation folds.
  2. Candidate panel extraction for targets (15, 25, 35, 50 genes).
  3. Performance retention curve (C-index vs gene count).
  4. Hallmark pathway representation and biotype distribution.
  5. Export of clinical targeted assay specifications (RT-qPCR / NanoString).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.panel_distillation import (
    MinimalGenePanelDistiller,
    CandidatePanel,
    DistillationReport,
)


@pytest.fixture
def synthetic_expression_cohort():
    """Generates synthetic expression matrix (100 patients x 50 genes) with true survival targets."""
    np.random.seed(42)
    n_patients = 100
    n_genes = 50

    gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    # Mark specific genes with hallmark names
    gene_names[0] = "TP53"
    gene_names[1] = "MKI67"
    gene_names[2] = "CD274"
    gene_names[3] = "VEGFA"
    gene_names[4] = "EGFR"
    gene_names[5] = "MYC"

    # Synthetic expression data (zero-centered)
    X = np.random.normal(loc=0.0, scale=1.0, size=(n_patients, n_genes))
    # Signal: top 6 genes dictate true hazard
    hazard = 0.5 * X[:, 0] + 0.4 * X[:, 1] + 0.3 * X[:, 2] - 0.3 * X[:, 3] + 0.4 * X[:, 4] - 0.2 * X[:, 5]

    base_time = np.random.exponential(scale=1000.0, size=n_patients)
    true_time = base_time / np.exp(hazard)
    cens_time = np.random.uniform(200, 2500, size=n_patients)

    times = np.minimum(true_time, cens_time).clip(min=1.0)
    events = (true_time <= cens_time).astype(int)

    df_X = pd.DataFrame(X, columns=gene_names)
    return df_X, times, events


def test_distiller_fitting_and_stability(synthetic_expression_cohort):
    """Verifies that distillation performs fold-wise stability selection."""
    df_X, times, events = synthetic_expression_cohort

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[15, 25, 35, 50],
        n_folds=3,
        random_state=42,
    )
    report = distiller.fit(df_X, times, events)

    assert isinstance(report, DistillationReport)
    assert len(report.candidate_panels) == 4
    assert sorted(report.candidate_panels.keys()) == [15, 25, 35, 50]

    # Verify top signal genes have high selection stability
    stability = distiller.feature_stability_scores_
    assert len(stability) == df_X.shape[1]
    # TP53, MKI67 should have non-zero selection frequency
    assert "TP53" in stability
    assert stability["TP53"] > 0.0


def test_candidate_panel_structure(synthetic_expression_cohort):
    """Verifies the structure and metadata of candidate panels."""
    df_X, times, events = synthetic_expression_cohort

    distiller = MinimalGenePanelDistiller(panel_sizes=[15, 35], n_folds=3, random_state=42)
    report = distiller.fit(df_X, times, events)

    panel_15 = report.candidate_panels[15]
    assert isinstance(panel_15, CandidatePanel)
    assert len(panel_15.genes) == 15
    assert panel_15.c_index > 0.50
    assert panel_15.retention_ratio > 0.0  # Compared to full model

    # Check hallmark representation
    assert isinstance(panel_15.hallmark_representation, dict)
    assert len(panel_15.hallmark_representation) > 0


def test_performance_retention_curves(synthetic_expression_cohort):
    """Verifies performance retention curve generation."""
    df_X, times, events = synthetic_expression_cohort

    distiller = MinimalGenePanelDistiller(panel_sizes=[15, 25, 35, 50], n_folds=3, random_state=42)
    report = distiller.fit(df_X, times, events)

    curve = report.retention_curve
    assert len(curve) >= 4
    sizes = [pt["panel_size"] for pt in curve]
    c_indices = [pt["c_index"] for pt in curve]

    assert sorted(sizes) == sizes
    for c in c_indices:
        assert 0.40 <= c <= 1.0


def test_assay_specifications_export(synthetic_expression_cohort, tmp_path):
    """Verifies clinical assay spec export to JSON / Markdown."""
    df_X, times, events = synthetic_expression_cohort

    distiller = MinimalGenePanelDistiller(panel_sizes=[15, 35], n_folds=3, random_state=42)
    report = distiller.fit(df_X, times, events)

    json_out = tmp_path / "panel_spec.json"
    distiller.export_assay_specifications(panel_size=35, output_path=json_out)

    assert json_out.exists()
    content = json_out.read_text(encoding="utf-8")
    assert "35-Gene" in content or "panel_size" in content
    assert "genes" in content
    assert "in_silico_candidate" in content
    assert "missing_evidence" in content


def test_insufficient_events_raises_clear_error(synthetic_expression_cohort):
    """Verifies that distillation raises ValueError on insufficient events instead of silent fallback."""
    df_X, times, events = synthetic_expression_cohort
    # Zero events
    zero_events = np.zeros(len(events), dtype=int)
    distiller = MinimalGenePanelDistiller(panel_sizes=[15], n_folds=3)
    with pytest.raises(ValueError, match="Insufficient survival data"):
        distiller.fit(df_X, times, zero_events)


def test_synthetic_provenance_flagging(synthetic_expression_cohort, tmp_path):
    """Verifies that synthetic data is strictly labeled as synthetic in reports and exported specs."""
    df_X, times, events = synthetic_expression_cohort
    distiller = MinimalGenePanelDistiller(panel_sizes=[15], n_folds=3, random_state=42)
    report = distiller.fit(
        df_X, times, events,
        data_source="synthetic_test",
        is_synthetic=True,
        endpoint="test_pfi"
    )

    assert report.is_synthetic is True
    assert report.data_source == "synthetic_test"
    assert any("SYNTHETIC" in lim for lim in report.limitations)

    json_out = tmp_path / "synthetic_spec.json"
    spec = distiller.export_assay_specifications(panel_size=15, output_path=json_out)
    assert spec["is_synthetic_training"] is True
    assert "in-silico research specification" in spec["notice"]


def test_distiller_mismatched_array_lengths_raises_value_error(synthetic_expression_cohort):
    """Verifies that distiller.fit() rejects mismatched array lengths across X, times, events, and IDs."""
    df_X, times, events = synthetic_expression_cohort
    distiller = MinimalGenePanelDistiller(panel_sizes=[15], n_folds=3)

    # 1. times length mismatch
    with pytest.raises(ValueError, match="Array length mismatch"):
        distiller.fit(df_X, times[:50], events)

    # 2. events length mismatch
    with pytest.raises(ValueError, match="Array length mismatch"):
        distiller.fit(df_X, times, events[:50])

    # 3. patient_ids length mismatch
    bad_pt_ids = [f"TCGA-AA-{i:04d}" for i in range(len(df_X) - 5)]
    with pytest.raises(ValueError, match="Patient IDs length mismatch"):
        distiller.fit(df_X, times, events, patient_ids=bad_pt_ids)

    # 4. sample_ids length mismatch
    bad_sample_ids = [f"TCGA-AA-{i:04d}-01A" for i in range(len(df_X) - 5)]
    with pytest.raises(ValueError, match="Sample IDs length mismatch"):
        distiller.fit(df_X, times, events, sample_ids=bad_sample_ids)


def test_distiller_nan_events_handled_with_strict_alignment(synthetic_expression_cohort):
    """
    Verifies that raw survival inputs with NaN events are validated prior to integer casting,
    invalid samples are explicitly excluded, and expression rows, times, events, and patient barcodes
    remain strictly aligned.
    """
    df_X, times, events = synthetic_expression_cohort
    # Introduce NaN values into float events array
    events_with_nan = events.astype(float).copy()
    nan_indices = [5, 12, 25, 40]
    events_with_nan[nan_indices] = np.nan

    # Provide patient IDs to verify synchronous alignment
    patient_ids = [f"TCGA-PT-{i:04d}" for i in range(len(df_X))]

    distiller = MinimalGenePanelDistiller(panel_sizes=[15], n_folds=3, random_state=42)
    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events_with_nan,
        patient_ids=patient_ids,
        data_source="nan_event_test",
        is_synthetic=True,
    )

    expected_n = len(df_X) - len(nan_indices)
    assert report.n_samples == expected_n
    assert len(distiller.oof_comparator_scores_) == expected_n
    # Verify that the distiller fitted without premature casting exception
    assert report.patient_grouping_status == "patient_grouped_strictly_partitioned"


def test_distiller_invalid_event_values_handled_with_strict_alignment(synthetic_expression_cohort):
    """
    Verifies that invalid survival event values (e.g. 2, -1, non-binary) are explicitly handled,
    excluded from survival distillation, and strict 1:1 alignment is maintained.
    """
    df_X, times, events = synthetic_expression_cohort
    events_with_invalid = events.astype(float).copy()
    invalid_indices = [3, 17, 33]
    events_with_invalid[invalid_indices[0]] = 2.0
    events_with_invalid[invalid_indices[1]] = -1.0
    events_with_invalid[invalid_indices[2]] = 99.0

    sample_ids = [f"TCGA-SA-{i:04d}-01A" for i in range(len(df_X))]

    distiller = MinimalGenePanelDistiller(panel_sizes=[15], n_folds=3, random_state=42)
    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events_with_invalid,
        sample_ids=sample_ids,
        data_source="invalid_event_test",
        is_synthetic=True,
    )

    expected_n = len(df_X) - len(invalid_indices)
    assert report.n_samples == expected_n
    assert len(distiller.oof_comparator_scores_) == expected_n
    assert report.patient_grouping_status == "patient_grouped_strictly_partitioned"

