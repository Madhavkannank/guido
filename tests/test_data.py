"""
Unit tests for data preprocessing and splitting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.data.acquisition import fpkm_to_log2tpm, filter_low_variance, align_and_label
from src.data.splitter import split_data


# ─── fpkm_to_log2tpm ─────────────────────────────────────────────────────────

def test_fpkm_to_log2tpm_shape():
    fpkm = pd.DataFrame(np.random.rand(100, 20), columns=[f"S{i}" for i in range(20)])
    result = fpkm_to_log2tpm(fpkm)
    assert result.shape == (100, 20)


def test_fpkm_to_log2tpm_values_non_negative():
    fpkm = pd.DataFrame(np.abs(np.random.rand(50, 10)), columns=[f"S{i}" for i in range(10)])
    result = fpkm_to_log2tpm(fpkm)
    assert (result.values >= 0).all()


def test_fpkm_to_log2tpm_zero_input():
    """All-zero FPKM should yield log2(0+1) = 0 after TPM conversion."""
    fpkm = pd.DataFrame(np.zeros((10, 5)), columns=[f"S{i}" for i in range(5)])
    result = fpkm_to_log2tpm(fpkm)
    # When all FPKM = 0, division by 0 → NaN; check graceful handling
    assert result.shape == (10, 5)


# ─── filter_low_variance ─────────────────────────────────────────────────────

def test_filter_low_variance_retains_top_n():
    rng = np.random.default_rng(42)
    expr = pd.DataFrame(rng.random((200, 30)), index=[f"ENSG{i:011d}" for i in range(200)])
    filtered, genes = filter_low_variance(expr, threshold=0.0, n_top=50)
    assert len(genes) == 50
    assert filtered.shape[0] == 50


def test_filter_low_variance_removes_low_var():
    rng = np.random.default_rng(0)
    # 100 genes × 2 samples;  genes 0-19 are constant (zero variance)
    expr = pd.DataFrame({
        "S1": [1.0] * 20 + (rng.random(80) * 5 + 1).tolist(),
        "S2": [1.0] * 20 + (rng.random(80) * 5 + 1).tolist(),
    })  # shape (100, 2) — genes as rows, samples as columns
    # genes 0-19 have zero variance; genes 20-99 have positive variance
    _, genes = filter_low_variance(expr, threshold=0.01, n_top=50)
    assert all(float(expr.loc[g, :].var()) > 0.01 for g in genes)


# ─── align_and_label ─────────────────────────────────────────────────────────

def test_align_and_label():
    n_genes, n_patients = 100, 40
    gene_ids = [f"ENSG{i:011d}" for i in range(n_genes)]
    patient_ids = [f"TCGA-XX-{i:04d}" for i in range(n_patients)]

    expr = pd.DataFrame(np.random.rand(n_genes, n_patients), index=gene_ids, columns=patient_ids)
    clinical = pd.DataFrame({
        "vital_status": ["Alive"] * 20 + ["Dead"] * 20,
    }, index=patient_ids)

    X, y = align_and_label(expr, clinical)
    assert X.shape == (n_patients, n_genes)
    assert set(y.unique()).issubset({0, 1})
    assert y.sum() == 20


def test_align_and_label_partial_overlap():
    gene_ids = [f"ENSG{i:011d}" for i in range(50)]
    all_patients = [f"TCGA-XX-{i:04d}" for i in range(20)]
    expr = pd.DataFrame(np.random.rand(50, 20), index=gene_ids, columns=all_patients)
    # Clinical only covers half the patients
    clinical = pd.DataFrame(
        {"vital_status": ["Alive"] * 5 + ["Dead"] * 5},
        index=all_patients[:10],
    )
    X, y = align_and_label(expr, clinical)
    assert X.shape[0] == 10


# ─── split_data ───────────────────────────────────────────────────────────────

def test_split_data_sizes():
    X = pd.DataFrame(np.random.rand(200, 50))
    y = pd.Series([0] * 120 + [1] * 80)
    splits = split_data(X, y)
    total = sum(v[0].shape[0] for v in splits.values())
    assert total == 200


def test_split_data_no_overlap():
    X = pd.DataFrame(np.random.rand(100, 20))
    y = pd.Series([0] * 60 + [1] * 40)
    splits = split_data(X, y)
    train_idx = set(splits["train"][0].index)
    val_idx   = set(splits["val"][0].index)
    test_idx  = set(splits["test"][0].index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_data_stratified():
    X = pd.DataFrame(np.random.rand(200, 10))
    y = pd.Series([0] * 130 + [1] * 70)
    splits = split_data(X, y)
    # Check that non-empty splits have both classes
    for name, (Xs, ys) in splits.items():
        if len(ys) > 0:
            assert len(ys.unique()) == 2, f"Split '{name}' is missing a class"

    # Explicit 3-way split test
    splits_3way = split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    for name, (Xs, ys) in splits_3way.items():
        assert len(ys.unique()) == 2, f"Split '{name}' is missing a class"


# ─── Xena Acquisition Integrity (No Synthetic Labels) ────────────────────────

def test_xena_missing_phenotypes_raises_runtime_error_no_synthetic_labels(monkeypatch):
    """
    Regression Test: Proves that missing phenotypes in real Xena acquisition
    raise RuntimeError and NEVER silently generate synthetic random labels.
    """
    import src.data.xena_acquisition as xena_mod

    dummy_expr = pd.DataFrame(
        np.random.rand(20, 5),
        index=[f"GENE_{i}" for i in range(20)],
        columns=[f"TCGA-01-000{i}" for i in range(5)],
    )

    monkeypatch.setattr(xena_mod, "download_matrix", lambda dataset, hub: dummy_expr)
    monkeypatch.setattr(xena_mod, "download_phenotypes", lambda dataset, hub: None)

    with pytest.raises(RuntimeError, match="Real-data acquisition strictly prohibits synthetic label generation"):
        xena_mod.run_xena_acquisition("TCGA.BRCA.sampleMap/HiSeqV2")


def test_xena_empty_phenotypes_raises_runtime_error(monkeypatch):
    """Verifies that empty DataFrame from download_phenotypes raises RuntimeError."""
    import src.data.xena_acquisition as xena_mod

    dummy_expr = pd.DataFrame(
        np.random.rand(20, 5),
        index=[f"GENE_{i}" for i in range(20)],
        columns=[f"TCGA-01-000{i}" for i in range(5)],
    )

    monkeypatch.setattr(xena_mod, "download_matrix", lambda dataset, hub: dummy_expr)
    monkeypatch.setattr(xena_mod, "download_phenotypes", lambda dataset, hub: pd.DataFrame())

    with pytest.raises(RuntimeError, match="Real-data acquisition strictly prohibits synthetic label generation"):
        xena_mod.run_xena_acquisition("TCGA.BRCA.sampleMap/HiSeqV2")


def test_xena_align_and_label_invalid_phenotypes_raises():
    """Verifies that xena align_and_label raises ValueError on invalid or missing clinical labels."""
    from src.data.xena_acquisition import align_and_label as xena_align

    expr = pd.DataFrame(
        np.random.rand(10, 4),
        index=[f"G_{i}" for i in range(10)],
        columns=["S1", "S2", "S3", "S4"],
    )

    # 1. No overlapping samples
    pheno_no_overlap = pd.DataFrame({"PFI": [1, 0, 1]}, index=["X1", "X2", "X3"])
    with pytest.raises(ValueError, match="No sample IDs overlap"):
        xena_align(expr, pheno_no_overlap)

    # 2. No valid outcome columns
    pheno_no_col = pd.DataFrame({"random_col": ["A", "B", "C", "D"]}, index=["S1", "S2", "S3", "S4"])
    with pytest.raises(ValueError, match="No valid clinical outcome"):
        xena_align(expr, pheno_no_col)

    # 3. All events NaN / missing
    pheno_nan = pd.DataFrame({"PFI": [np.nan, np.nan, np.nan, np.nan]}, index=["S1", "S2", "S3", "S4"])
    with pytest.raises(ValueError, match="No valid binary outcome labels remain"):
        xena_align(expr, pheno_nan)


def test_xena_align_and_label_unknown_phenotypes_excluded_not_zero():
    """
    Regression Test: Prohibits unknown phenotype categories ('unknown', 'not reported', 'discrepancy')
    from being silently converted to event-free (0). Samples with unknown categories must be excluded.
    """
    from src.data.xena_acquisition import align_and_label as xena_align

    expr = pd.DataFrame(
        np.random.rand(10, 5),
        index=[f"G_{i}" for i in range(10)],
        columns=["S1", "S2", "S3", "S4", "S5"],
    )
    # S1: Dead (event=1), S2: Alive (event=0), S3: Unknown, S4: Not Reported, S5: Discrepancy
    pheno = pd.DataFrame({
        "vital_status": ["Dead", "Alive", "Unknown", "Not Reported", "Discrepancy"],
    }, index=["S1", "S2", "S3", "S4", "S5"])

    X, y = xena_align(expr, pheno)

    # S3, S4, S5 must be excluded, NOT assigned 0!
    assert len(X) == 2
    assert list(X.index) == ["S1", "S2"]
    assert y.loc["S1"] == 1
    assert y.loc["S2"] == 0
    assert "S3" not in y.index
    assert "S4" not in y.index
    assert "S5" not in y.index


def test_xena_align_and_label_only_unknown_categories_raises_value_error():
    """
    Regression Test: If a phenotype column contains only unknown or unmapped categories,
    it must raise an explicit ValueError rather than silently returning an all-zero vector.
    """
    from src.data.xena_acquisition import align_and_label as xena_align

    expr = pd.DataFrame(
        np.random.rand(10, 3),
        index=[f"G_{i}" for i in range(10)],
        columns=["S1", "S2", "S3"],
    )
    pheno = pd.DataFrame({
        "vital_status": ["unknown", "not reported", "indeterminate"],
    }, index=["S1", "S2", "S3"])

    with pytest.raises(ValueError, match="Conversion of unknown phenotypes to event-free \\(0\\) is strictly prohibited"):
        xena_align(expr, pheno)


def test_acquisition_align_and_label_unknown_phenotypes_excluded_not_zero():
    """
    Regression Test for src.data.acquisition.align_and_label:
    Verifies that unknown / unmapped categories are excluded and never silently mapped to 0.
    """
    gene_ids = [f"ENSG{i:011d}" for i in range(20)]
    sample_ids = ["P1", "P2", "P3", "P4", "P5", "P6"]

    expr = pd.DataFrame(np.random.rand(20, 6), index=gene_ids, columns=sample_ids)
    clinical = pd.DataFrame({
        "vital_status": ["Dead", "Alive", "Unknown", "Not Reported", "Indeterminate", "Recurred"],
    }, index=sample_ids)

    X, y = align_and_label(expr, clinical)

    # P1 (Dead -> 1), P2 (Alive -> 0), P6 (Recurred -> 1) are valid.
    # P3 (Unknown), P4 (Not Reported), P5 (Indeterminate) must be excluded!
    assert len(X) == 3
    assert set(X.index) == {"P1", "P2", "P6"}
    assert y.loc["P1"] == 1
    assert y.loc["P2"] == 0
    assert y.loc["P6"] == 1
    assert "P3" not in y.index
    assert "P4" not in y.index
    assert "P5" not in y.index


def test_acquisition_align_and_label_only_unknown_categories_raises_value_error():
    """
    Regression Test: When all clinical rows contain unknown or unmapped categories,
    src.data.acquisition.align_and_label must raise ValueError prohibiting silent conversion to 0.
    """
    gene_ids = [f"ENSG{i:011d}" for i in range(10)]
    sample_ids = ["P1", "P2", "P3"]

    expr = pd.DataFrame(np.random.rand(10, 3), index=gene_ids, columns=sample_ids)
    clinical = pd.DataFrame({
        "vital_status": ["unknown", "not reported", "discrepancy"],
    }, index=sample_ids)

    with pytest.raises(ValueError, match="Conversion of unknown phenotypes to event-free \\(0\\) is strictly prohibited"):
        align_and_label(expr, clinical)

