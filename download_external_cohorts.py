#!/usr/bin/env python3
"""
Multi-Cohort External Dataset Downloader & Parquet Harmonizer
============================================================
Fetches, harmonizes, and caches independent international clinical oncology cohorts
into Snappy-compressed Apache Parquet shards for high-throughput training and validation:

  1. METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)
     - 1,980 breast cancer patients with Relapse-Free Survival (RFS) and Overall Survival (OS)
  2. NCBI GEO GSE31210 (Okayama et al. Lung Adenocarcinoma)
     - 246 Stage I/II lung adenocarcinoma patients with Relapse-Free Survival
  3. CPTAC / International Benchmarks

Usage:
  py -3.12 download_external_cohorts.py metabric
  py -3.12 download_external_cohorts.py gse31210
  py -3.12 download_external_cohorts.py all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.validation.external_datasets import (
    EXTERNAL_DATA_DIR,
    fetch_geo_gse31210,
    fetch_metabric,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download & Harmonize External Oncology Cohorts")
    parser.add_argument("cohorts", nargs="+", choices=["metabric", "gse31210", "all"], help="Cohorts to download and cache")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for harmonized Parquet shards")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.cohorts
    if "all" in targets:
        targets = ["metabric", "gse31210"]

    log.info("Starting external cohort ingestion into: %s", out_dir)

    for cohort in targets:
        if cohort == "metabric":
            log.info("=" * 60)
            log.info("Ingesting METABRIC Breast Cancer Cohort (~1,980 patients)...")
            X_df, y_ser, t_ser, e_ser = fetch_metabric(out_dir=out_dir)
            log.info("METABRIC Ingestion Complete: %d patients, %d genes, %d recurrence events (3-yr risk=%d)",
                     X_df.shape[0], X_df.shape[1], int(e_ser.sum()), int(y_ser.sum()))

        elif cohort == "gse31210":
            log.info("=" * 60)
            log.info("Ingesting NCBI GEO GSE31210 Lung Adenocarcinoma Cohort (~246 patients)...")
            X_df, y_ser, t_ser, e_ser = fetch_geo_gse31210(out_dir=out_dir)
            log.info("GSE31210 Ingestion Complete: %d patients, %d genes, %d recurrence events (3-yr risk=%d)",
                     X_df.shape[0], X_df.shape[1], int(e_ser.sum()), int(y_ser.sum()))

    log.info("=" * 60)
    log.info("All requested external datasets are cached and ready for training & cross-cohort validation.")


if __name__ == "__main__":
    main()
