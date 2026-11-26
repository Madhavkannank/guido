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
    """Generates synthetic expression matrix (150 patients x 120 genes) with true survival targets."""
    np.random.seed(42)
    n_patients = 150
    n_genes = 120

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
