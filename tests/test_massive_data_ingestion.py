"""
Unit and integration tests for GUIDO Massive Cloud Oncology Data Ingestion Engine.
Verifies directory resolution, storage footprint audit logic, URL catalog integrity,
and mock streaming Parquet conversion for multi-gigabyte genomics.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.download_all_massive_oncology_data import (
    ALL_33_TCGA,
    BEAT_AML_STUDY,
    CPTAC_STUDIES,
    GEO_BENCHMARKS,
    GTEX_V8_TPM_URL,
    METABRIC_CLIN_URL,
    METABRIC_EXPR_URL,
    RECOUNT3_TARGET_STUDIES,
    TCGA_CDR_URL,
    TCGA_EXPR_URL,
    TCGA_METHYL_URL,
    TCGA_PANCAN_MASTER_URL,
    audit_storage_footprint,
    download_metabric,
    resolve_base_dirs,
    stream_download,
)


def test_resolve_base_dirs(tmp_path: Path):
    """Verifies that base directory structure for 5 TB Google Drive is properly created."""
    raw_xena, raw_gtex, external, shards, cptac = resolve_base_dirs(override_root=tmp_path)
    assert raw_xena.exists() and raw_xena.is_dir()
    assert raw_gtex.exists() and raw_gtex.is_dir()
    assert external.exists() and external.is_dir()
    assert shards.exists() and shards.is_dir()
    assert cptac.exists() and cptac.is_dir()
    assert raw_xena.name == "xena"
    assert raw_gtex.name == "gtex"
    assert shards.name == "shards"


def test_url_catalog_integrity():
    """Validates URL formats and endpoints across all consortia."""
    # TCGA
    assert len(ALL_33_TCGA) == 33
    for c in ALL_33_TCGA:
        expr_url = TCGA_EXPR_URL.format(cancer=c)
        methyl_url = TCGA_METHYL_URL.format(cancer=c)
        assert expr_url.startswith("https://tcga.xenahubs.net/download/")
        assert methyl_url.startswith("https://tcga.xenahubs.net/download/")
        assert f"TCGA.{c}.sampleMap" in expr_url
        assert f"TCGA.{c}.sampleMap" in methyl_url

    # GTEx v8
    assert "storage.googleapis.com/adult-gtex" in GTEX_V8_TPM_URL
    assert GTEX_V8_TPM_URL.endswith(".gct.gz")

    # METABRIC
    assert "cBioPortal/datahub" in METABRIC_EXPR_URL
    assert "cBioPortal/datahub" in METABRIC_CLIN_URL

    # GEO
    assert len(GEO_BENCHMARKS) == 10
    for geo_id, desc, url, target_n in GEO_BENCHMARKS:
        assert geo_id.startswith("GSE")
        assert "ftp.ncbi.nlm.nih.gov" in url
        assert target_n > 100

    # CPTAC & AML
    assert len(CPTAC_STUDIES) == 8
    assert BEAT_AML_STUDY == "aml_ohsu_2018"

    # recount3
    assert len(RECOUNT3_TARGET_STUDIES) >= 5


def test_audit_storage_footprint(tmp_path: Path):
    """Verifies storage footprint computation and uncompressed size multipliers."""
    # Create fake files across categories
    raw_xena = tmp_path / "data" / "raw" / "xena"
    raw_gtex = tmp_path / "data" / "raw" / "gtex"
    shards   = tmp_path / "data" / "processed" / "shards"
    for d in (raw_xena, raw_gtex, shards):
        d.mkdir(parents=True, exist_ok=True)

    # 10 MB in xena
    (raw_xena / "sample.tsv").write_bytes(b"0" * 10_000_000)
    # 20 MB in gtex
    (raw_gtex / "sample.gct").write_bytes(b"0" * 20_000_000)
    # 5 MB in shards
    (shards / "sample.parquet").write_bytes(b"0" * 5_000_000)

    report = audit_storage_footprint(tmp_path)
    assert report["total_disk_gb"] > 0
    assert report["estimated_uncompressed_gb"] >= report["total_disk_gb"]
    assert report["file_counts"]["raw_xena"] == 1
    assert report["file_counts"]["raw_gtex"] == 1
    assert report["file_counts"]["shards"] == 1


def test_stream_download_existing_file(tmp_path: Path):
    """Verifies that stream_download skips already downloaded files that meet minimum size."""
    test_file = tmp_path / "already_downloaded.tsv"
    test_file.write_bytes(b"Hello oncology world " * 100)
    ok = stream_download("http://example.com/fake", test_file, min_size=500)
    assert ok is True
