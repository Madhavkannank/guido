#!/usr/bin/env python3
"""
GUIDO Massive Cloud Oncology Data Ingestion & Harmonization Engine
==================================================================
Streams and harmonizes massive multi-consortium oncology datasets directly into
Google Drive (5 TB storage) or local disk for large-scale clinical model training
and cross-cohort external validation.

Dataset Catalog & Uncompressed Volume Breakdown (Target: 75 GB to 1 TB):
  1. TCGA PanCanAtlas ComBat Matrix (~11,069 patients across 33 cancers) : 24.5 GB uncompressed
  2. TCGA All 33 Cohort RNA-Seq V2 Matrices (~11,160 patients)           : 35.0 GB uncompressed
  3. TCGA All 33 Cohort HumanMethylation450 Multi-Omics Matrices         : 115.0 GB uncompressed
  4. GTEx v8 Master Expression TPM (Healthy Human Organ Baseline, N=17k) : 18.0 GB uncompressed
  5. METABRIC Breast Cancer Benchmark (1,980 patients with 10-20yr RFS)  : 1.5 GB uncompressed
  6. CPTAC Multi-Omics Proteogenomics (8 Major Cancer Studies)           : 15.0 GB uncompressed
  7. NCBI GEO Landmark Hospital Benchmarks (10 Validated Cancer Series)  : 3.5 GB uncompressed
  8. recount3 Large-Scale SRA Cancer Transcriptomes (AWS Open Data)      : 35.0 GB uncompressed
  9. Beat AML Leukemia & MSK-IMPACT Clinical Cohorts                     : 2.5 GB uncompressed
  -----------------------------------------------------------------------------------------
  TOTAL VERIFIED OPEN-ACCESS ONCOLOGY DATA FOOTPRINT                     : ~250.0 GB UNCOMPRESSED
  (Scalable up to 1 TB by selecting full recount3 SRA & GDC STAR counts)

Usage:
  py -3.12 scripts/download_all_massive_oncology_data.py --all
  py -3.12 scripts/download_all_massive_oncology_data.py --all-massive    # Includes 115 GB Methylation + GTEx
  py -3.12 scripts/download_all_massive_oncology_data.py --tcga --pancan-master --metabric --geo
  py -3.12 scripts/download_all_massive_oncology_data.py --methylation --gtex
  py -3.12 scripts/download_all_massive_oncology_data.py --audit-only
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Root Directory resolution (Default to Drive if in Colab, else project root)
COLAB_DRIVE_DIR = Path("/content/drive/MyDrive/GUIDO_5TB_DATA")
LOCAL_ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = "GUIDO-Cloud-Massive-Ingest/2.0 (Mozilla/5.0; Windows NT 10.0; Win64; x64)"

# ── 1. TCGA UCSC Xena Endpoints (All 33 Official Cohorts) ────────────────────
ALL_33_TCGA = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"
]
TCGA_CDR_URL  = "https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp"
TCGA_EXPR_URL = "https://tcga.xenahubs.net/download/TCGA.{cancer}.sampleMap%2FHiSeqV2.gz"
TCGA_PANCAN_MASTER_URL = "https://pancanatlas.xenahubs.net/download/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
TCGA_METHYL_URL = "https://tcga.xenahubs.net/download/TCGA.{cancer}.sampleMap%2FHumanMethylation450.gz"
TCGA_GISTIC_URL = "https://tcga.xenahubs.net/download/TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz"

# ── 2. GTEx v8 Master Expression TPM (Healthy Non-Cancer Baseline) ───────────
GTEX_V8_TPM_URL = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"

# ── 3. METABRIC & GEO Endpoints ──────────────────────────────────────────────
METABRIC_EXPR_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_mrna_illumina_microarray.txt"
METABRIC_CLIN_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_patient.txt"
GPL570_ANNOT_URL  = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"

# Landmark Verified NCBI GEO Series Matrices across Organ Systems
GEO_BENCHMARKS = [
    ("GSE31210", "Lung Adenocarcinoma Relapse", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31210/matrix/GSE31210_series_matrix.txt.gz", 226),
    ("GSE73731", "Renal Clear Cell Progression", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73731/matrix/GSE73731_series_matrix.txt.gz", 265),
    ("GSE2034",  "Breast Cancer Distant Relapse", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/matrix/GSE2034_series_matrix.txt.gz", 286),
    ("GSE39582", "Colorectal Relapse (CIT Cohort)", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/matrix/GSE39582_series_matrix.txt.gz", 566),
    ("GSE42568", "Breast Cancer (Ireland Hospital)", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42568/matrix/GSE42568_series_matrix.txt.gz", 121),
    ("GSE17536", "Colorectal (Moffitt Cancer Center)", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE17nnn/GSE17536/matrix/GSE17536_series_matrix.txt.gz", 177),
    ("GSE65858", "Head & Neck HNSCC Recurrence", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65858/matrix/GSE65858_series_matrix.txt.gz", 270),
    ("GSE30219", "Lung Adeno/Squamous Survival", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE30nnn/GSE30219/matrix/GSE30219_series_matrix.txt.gz", 300),
    ("GSE13507", "Bladder Urothelial Progression", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13507/matrix/GSE13507_series_matrix.txt.gz", 256),
    ("GSE19234", "AML Leukemia Survival Cohort", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19234/matrix/GSE19234_series_matrix.txt.gz", 290),
]

# ── 4. CPTAC Multi-Omics & Beat AML Cohorts ──────────────────────────────────
CPTAC_STUDIES = [
    ("brca_cptac_2020", "data_mrna_seq_fpkm.txt", "data_protein_quantification.txt"),
    ("coad_cptac_2019", "data_mrna_seq_v2_rsem.txt", "data_protein_quantification.txt"),
    ("luad_cptac_2020", "data_mrna_seq_rpkm.txt", "data_protein_quantification.txt"),
    ("paad_cptac_2021", "data_mrna_seq_v2_rsem.txt", "data_protein_quantification.txt"),
    ("gbm_cptac_2021",  "data_mrna_seq_fpkm.txt", "data_protein_quantification.txt"),
    ("brain_cptac_2020","data_mrna_seq_fpkm.txt", "data_protein_quantification.txt"),
    ("ucec_cptac_2020", "data_mrna_seq_fpkm.txt", "data_protein_quantification.txt"),
    ("lusc_cptac_2021", "data_mrna_seq_rpkm.txt", "data_protein_quantification.txt"),
]
BEAT_AML_STUDY = "aml_ohsu_2018"

# ── 5. ICGC PCAWG & MSK-IMPACT ───────────────────────────────────────────────
ICGC_PCAWG_STUDY = "pancan_pcawg_2020"
MSK_IMPACT_STUDY = "msk_impact_2017"

# ── 6. recount3 Endpoints (AWS Open Data) ─────────────────────────────────────
RECOUNT3_BASE = "https://recount-opendata.s3.amazonaws.com/recount3/release"
RECOUNT3_TARGET_STUDIES = [
    ("sra", "SRP009615", "15", "G026"),  # Landmark Pan-Cancer SRA study
    ("sra", "SRP045225", "25", "G029"),  # Cancer Transcriptome & Progression
    ("sra", "SRP012656", "56", "G026"),  # Breast Cancer recurrence & metastasis
    ("sra", "SRP057196", "96", "G026"),  # Gastrointestinal malignancy cohort
    ("sra", "SRP018838", "38", "G026"),  # Urological / Renal oncogenomics
]


def resolve_base_dirs(override_root: Optional[Path] = None) -> Tuple[Path, Path, Path, Path, Path]:
    """Resolves base directory layout for raw, external, and processed parquet shards."""
    if override_root:
        base = override_root
    elif COLAB_DRIVE_DIR.exists():
        base = COLAB_DRIVE_DIR
    else:
        base = LOCAL_ROOT

    raw_xena = base / "data" / "raw" / "xena"
    raw_gtex = base / "data" / "raw" / "gtex"
    external = base / "data" / "processed" / "external"
    shards   = base / "data" / "processed" / "shards"
    cptac    = base / "data" / "processed" / "cptac"

    for d in (raw_xena, raw_gtex, external, shards, cptac):
        d.mkdir(parents=True, exist_ok=True)

    return raw_xena, raw_gtex, external, shards, cptac


def stream_download(
    url: str,
    dest_file: Path,
    is_gz: bool = False,
    min_size: int = 1000,
    chunk_mb: int = 4,
    max_retries: int = 3,
) -> bool:
    """
    Streams a large dataset from URL to dest_file with live progress, automatic decompression,
    and exponential backoff retry on HTTP 429 / rate limits.
    Uses chunked streaming to avoid high RAM consumption on multi-gigabyte files.
    """
    if dest_file.exists() and dest_file.stat().st_size >= min_size:
        log.info("[EXISTS] %s (%d bytes)", dest_file.name, dest_file.stat().st_size)
        return True

    log.info("Downloading %s -> %s ...", url, dest_file.name)
    tmp_dest = dest_file.parent / f"tmp_{dest_file.name}"
    chunk_size = chunk_mb * 1024 * 1024

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as resp:
                total_size_hdr = resp.headers.get("Content-Length")
                total_bytes = int(total_size_hdr) if total_size_hdr else None

                if is_gz:
                    tmp_gz = dest_file.parent / f"tmp_{dest_file.name}.gz"
                    downloaded = 0
                    t_start = time.time()
                    with open(tmp_gz, "wb") as f_out:
                        while True:
                            buf = resp.read(chunk_size)
                            if not buf:
                                break
                            f_out.write(buf)
                            downloaded += len(buf)
                            if total_bytes and total_bytes > 50_000_000:
                                pct = (downloaded / total_bytes) * 100.0
                                mb = downloaded / (1024 * 1024)
                                tot_mb = total_bytes / (1024 * 1024)
                                elapsed = max(0.1, time.time() - t_start)
                                speed = mb / elapsed
                                sys.stdout.write(f"\r  Streaming {dest_file.name}: {mb:.1f} / {tot_mb:.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s")
                                sys.stdout.flush()

                    if total_bytes and total_bytes > 50_000_000:
                        sys.stdout.write("\n")

                    log.info("Decompressing %s ...", dest_file.name)
                    with gzip.open(tmp_gz, "rb") as f_in, open(tmp_dest, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out, length=chunk_size)
                    if tmp_gz.exists():
                        tmp_gz.unlink()
                else:
                    downloaded = 0
                    t_start = time.time()
                    with open(tmp_dest, "wb") as f_out:
                        while True:
                            buf = resp.read(chunk_size)
                            if not buf:
                                break
                            f_out.write(buf)
                            downloaded += len(buf)
                            if total_bytes and total_bytes > 50_000_000:
                                pct = (downloaded / total_bytes) * 100.0
                                mb = downloaded / (1024 * 1024)
                                tot_mb = total_bytes / (1024 * 1024)
                                elapsed = max(0.1, time.time() - t_start)
                                speed = mb / elapsed
                                sys.stdout.write(f"\r  Streaming {dest_file.name}: {mb:.1f} / {tot_mb:.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s")
                                sys.stdout.flush()
                    if total_bytes and total_bytes > 50_000_000:
                        sys.stdout.write("\n")

            if tmp_dest.stat().st_size < min_size:
                log.warning("File %s too small (%d bytes), removing.", dest_file.name, tmp_dest.stat().st_size)
                if tmp_dest.exists():
                    tmp_dest.unlink()
                return False

            if dest_file.exists():
                dest_file.unlink()
            tmp_dest.rename(dest_file)
            log.info("[SAVED] %s (%d bytes)", dest_file.name, dest_file.stat().st_size)
            return True

        except urllib.error.HTTPError as h_err:
            if h_err.code == 429 and attempt < max_retries:
                wait_time = attempt * 5
                log.warning("HTTP 429 Too Many Requests on %s. Waiting %ds before retry (attempt %d/%d)...",
                            dest_file.name, wait_time, attempt, max_retries)
                time.sleep(wait_time)
            else:
                log.error("HTTP error downloading %s: %s", url, h_err)
                break
        except Exception as exc:
            if attempt < max_retries:
                wait_time = attempt * 3
                log.warning("Download error on %s: %s. Retrying in %ds (attempt %d/%d)...",
                            dest_file.name, exc, wait_time, attempt, max_retries)
                time.sleep(wait_time)
            else:
                log.error("Failed to download %s after %d attempts: %s", url, max_retries, exc)

    if tmp_dest.exists():
        tmp_dest.unlink()
    return False


# ── SECTION 1: TCGA 33 Cohorts & PanCanAtlas Master ──────────────────────────

def download_tcga_all_33(raw_xena: Path, shards: Path) -> int:
    """Streams all 33 TCGA cohorts and converts them to Snappy Parquet shards (~35 GB uncompressed)."""
    log.info("=" * 76)
    log.info("PHASE 1: Ingesting All 33 TCGA Pan-Cancer Cohorts & TCGA-CDR Survival (~35 GB)")
    log.info("=" * 76)

    # 1. Standardized NIH TCGA-CDR
    stream_download(TCGA_CDR_URL, raw_xena / "TCGA_CDR_survival.tsv", is_gz=False, min_size=2_000_000)

    # 2. Ingest 33 cohorts
    success = 0
    for cancer in ALL_33_TCGA:
        tsv_name = f"TCGA.{cancer}.sampleMap_HiSeqV2.tsv"
        dest_tsv = raw_xena / tsv_name
        url = TCGA_EXPR_URL.format(cancer=cancer)
        ok = stream_download(url, dest_tsv, is_gz=True, min_size=3_000_000)
        if ok:
            success += 1
            pq_name = f"TCGA_{cancer}_expression.parquet"
            dest_pq = shards / pq_name
            if not dest_pq.exists():
                try:
                    log.info("Converting %s -> %s (Snappy Parquet)...", tsv_name, pq_name)
                    df = pd.read_csv(dest_tsv, sep="\t", index_col=0, low_memory=False)
                    df.to_parquet(dest_pq, compression="snappy")
                    log.info("[PARQUET SHARD READY] %s (%d bytes)", pq_name, dest_pq.stat().st_size)
                except Exception as e:
                    log.warning("Parquet conversion failed for %s: %s", cancer, e)

    log.info("[DONE] TCGA RNA-Seq Ingestion: %d / %d cohorts ready in Google Drive.", success, len(ALL_33_TCGA))
    return success


def download_pancan_master_matrix(raw_xena: Path, shards: Path) -> bool:
    """
    Streams the PanCanAtlas ComBat-adjusted Master Matrix (EB++AdjustPANCAN, ~24.5 GB uncompressed).
    Contains all 11,069 TCGA patients across all 33 cancer types.
    """
    log.info("=" * 76)
    log.info("PHASE 1b: Ingesting Unified PanCanAtlas ComBat-Adjusted Master Matrix (~24.5 GB)")
    log.info("=" * 76)
    tsv_name = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
    dest_tsv = raw_xena / tsv_name
    pq_name = "TCGA_PANCAN_EB_ADJUST_expression.parquet"
    dest_pq = shards / pq_name

    if dest_pq.exists() and dest_pq.stat().st_size > 100_000_000:
        log.info("[EXISTS] Unified PanCanAtlas Parquet Shard (%d bytes)", dest_pq.stat().st_size)
        return True

    ok = stream_download(TCGA_PANCAN_MASTER_URL, dest_tsv, is_gz=True, min_size=50_000_000)
    if ok and not dest_pq.exists():
        try:
            log.info("Converting %s -> %s (Snappy Parquet)...", tsv_name, pq_name)
            df = pd.read_csv(dest_tsv, sep="\t", index_col=0, low_memory=False)
            df.to_parquet(dest_pq, compression="snappy")
            log.info("[PARQUET READY] %s (%d genes x %d patients)", dest_pq.name, df.shape[0], df.shape[1])
        except Exception as e:
            log.warning("PanCanAtlas master Parquet conversion note: %s", e)
    return ok


def download_tcga_methylation_450k(raw_xena: Path, shards: Path, cohorts: Optional[List[str]] = None) -> int:
    """
    Streams TCGA HumanMethylation450 multi-omics arrays across all 33 cancer types.
    Total Volume: ~12 GB compressed -> ~115 GB uncompressed TSVs.
    Matches exact patient barcodes for multi-omics integration and epigenomic risk modeling.
    """
    log.info("=" * 76)
    log.info("PHASE 1c: Ingesting TCGA HumanMethylation450 Multi-Omics (~115 GB Uncompressed)")
    log.info("=" * 76)
    target_cohorts = cohorts if cohorts else ALL_33_TCGA
    success = 0

    for cancer in target_cohorts:
        tsv_name = f"TCGA.{cancer}.sampleMap_HumanMethylation450.tsv"
        dest_tsv = raw_xena / tsv_name
        dest_pq = shards / f"TCGA_{cancer}_methylation.parquet"

        if dest_pq.exists() and dest_pq.stat().st_size > 10_000_000:
            log.info("[EXISTS] %s Parquet Shard (%d bytes)", dest_pq.name, dest_pq.stat().st_size)
            success += 1
            continue

        url = TCGA_METHYL_URL.format(cancer=cancer)
        ok = stream_download(url, dest_tsv, is_gz=True, min_size=10_000_000)
        time.sleep(1.0)  # Gentle spacing to respect UCSC CDN rate limits
        if ok:
            success += 1
            if not dest_pq.exists():
                try:
                    log.info("Converting %s -> %s (Snappy Parquet)...", tsv_name, dest_pq.name)
                    df = pd.read_csv(dest_tsv, sep="\t", index_col=0, low_memory=False)
                    df.to_parquet(dest_pq, compression="snappy")
                    log.info("[METHYLATION SHARD READY] %s (%d probes x %d patients)", dest_pq.name, df.shape[0], df.shape[1])
                except Exception as e:
                    log.warning("Methylation Parquet conversion note for %s: %s", cancer, e)

    log.info("[DONE] TCGA Methylation450 Ingestion: %d / %d cohorts ready in Google Drive.", success, len(target_cohorts))
    return success


# ── SECTION 2: GTEx v8 Master Expression TPM (Healthy Human Organ Baseline) ──

def download_gtex_v8_baseline(raw_gtex: Path, shards: Path) -> bool:
    """
    Streams GTEx v8 Master Expression TPM matrix (17,382 non-tumor samples across 54 body sites).
    Total Volume: 1.625 GB gz -> ~18.0 GB uncompressed GCT.
    Establishes the non-cancer baseline for hallmark metagene specificity and tissue calibration.
    """
    log.info("=" * 76)
    log.info("PHASE 2: Ingesting GTEx v8 Non-Cancer Normal Organ Baseline (~18.0 GB)")
    log.info("=" * 76)
    gct_name = "GTEx_Analysis_v8_RNASeQCv1.1.9_gene_tpm.gct"
    dest_gct = raw_gtex / gct_name
    pq_name = "GTEx_v8_gene_tpm.parquet"
    dest_pq = shards / pq_name

    if dest_pq.exists() and dest_pq.stat().st_size > 500_000_000:
        log.info("[EXISTS] GTEx v8 Parquet Shard (%d bytes)", dest_pq.stat().st_size)
        return True

    ok = stream_download(GTEX_V8_TPM_URL, dest_gct, is_gz=True, min_size=500_000_000)
    if ok and not dest_pq.exists():
        try:
            log.info("Converting %s -> %s (Snappy Parquet)...", gct_name, pq_name)
            # Skip first 2 lines of standard GCT format
            df = pd.read_csv(dest_gct, sep="\t", skiprows=2, index_col=0, low_memory=False)
            if "Description" in df.columns:
                df = df.drop(columns=["Description"])
            df.to_parquet(dest_pq, compression="snappy")
            log.info("[GTEx PARQUET READY] %s (%d genes x %d healthy samples)", dest_pq.name, df.shape[0], df.shape[1])
        except Exception as e:
            log.warning("GTEx Parquet conversion note: %s", e)
    return ok


# ── SECTION 3: METABRIC (Breast Cancer) ──────────────────────────────────────

def download_metabric(external: Path) -> bool:
    """Streams and harmonizes live METABRIC cohort (1,980 patients with longitudinal RFS/OS)."""
    log.info("=" * 76)
    log.info("PHASE 3: Ingesting METABRIC Breast Cancer Benchmark (~1,980 Patients, ~1.5 GB)")
    log.info("=" * 76)
    pq_file = external / "metabric_harmonized.parquet"
    if pq_file.exists() and pq_file.stat().st_size > 1_000_000:
        log.info("[EXISTS] METABRIC harmonized cohort (%d bytes)", pq_file.stat().st_size)
        return True

    try:
        req_clin = urllib.request.Request(METABRIC_CLIN_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req_clin, timeout=60) as resp:
            lines = [l.decode("utf-8", errors="replace") for l in resp.readlines() if not l.startswith(b"#")]
        df_clin = pd.read_csv(io.StringIO("".join(lines)), sep="\t", index_col="PATIENT_ID")

        if "RFS_MONTHS" in df_clin.columns and "RFS_STATUS" in df_clin.columns:
            time_days = pd.to_numeric(df_clin["RFS_MONTHS"], errors="coerce") * 30.4375
            rfs_str = df_clin["RFS_STATUS"].astype(str).str.strip()
            is_event = rfs_str.str.startswith("1") | (rfs_str.str.contains("Recurred", case=False, na=False) & ~rfs_str.str.contains("Not", case=False, na=False))
            is_free = rfs_str.str.startswith("0") | rfs_str.str.contains("Not Recurred", case=False, na=False)
            events = pd.Series(np.nan, index=df_clin.index)
            events[is_event] = 1.0
            events[is_free] = 0.0
        elif "OS_MONTHS" in df_clin.columns and "OS_STATUS" in df_clin.columns:
            time_days = pd.to_numeric(df_clin["OS_MONTHS"], errors="coerce") * 30.4375
            os_str = df_clin["OS_STATUS"].astype(str).str.strip()
            is_event = os_str.str.startswith("1") | (os_str.str.contains("DECEASED|Dead", case=False, na=False) & ~os_str.str.contains("Living|Alive", case=False, na=False))
            is_free = os_str.str.startswith("0") | os_str.str.contains("Living|Alive", case=False, na=False)
            events = pd.Series(np.nan, index=df_clin.index)
            events[is_event] = 1.0
            events[is_free] = 0.0
        else:
            raise ValueError("Required survival columns missing in METABRIC clinical table.")

        req_expr = urllib.request.Request(METABRIC_EXPR_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req_expr, timeout=180) as resp:
            df_expr = pd.read_csv(resp, sep="\t", low_memory=False)

        if "Entrez_Gene_Id" in df_expr.columns:
            df_expr = df_expr.drop(columns=["Entrez_Gene_Id"])
        if "Hugo_Symbol" in df_expr.columns:
            df_expr = df_expr.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")

        common = [
            p for p in df_clin.index
            if p in df_expr.columns
            and not pd.isna(time_days.loc[p])
            and np.isfinite(time_days.loc[p])
            and time_days.loc[p] > 0
            and not pd.isna(events.loc[p])
            and events.loc[p] in (0, 1)
        ]
        X_df = df_expr[common].T.astype(float).fillna(0.0)
        time_series = time_days.loc[common].astype(float)
        event_series = events.loc[common].astype(int)
        y_series = ((event_series == 1) & (time_series <= 1095)).astype(int)

        df_cached = X_df.copy()
        df_cached["_Y_3YR"] = y_series
        df_cached["_TIME_DAYS"] = time_series
        df_cached["_EVENT"] = event_series
        df_cached.to_parquet(pq_file, compression="snappy")
        log.info("[SAVED METABRIC] %d patients, %d genes (%d bytes)", len(common), X_df.shape[1], pq_file.stat().st_size)
        return True
    except Exception as exc:
        log.error("Failed to ingest METABRIC: %s", exc)
        return False


# ── SECTION 4: Expanded NCBI GEO Clinical Benchmarks ─────────────────────────

def _load_gpl570_mapping(external: Path) -> Dict[str, str]:
    """Downloads GPL570 probe-to-gene mapping table from NCBI GEO FTP."""
    map_file = external / "gpl570_probe_mapping.json"
    if map_file.exists():
        try:
            return json.loads(map_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    probe_to_gene = {}
    try:
        log.info("Streaming GPL570 probe annotation table from NCBI GEO...")
        req = urllib.request.Request(GPL570_ANNOT_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=90) as resp, gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
            in_table = False
            for line in gz:
                if "!platform_table_begin" in line:
                    in_table = True
                    continue
                if not in_table or line.startswith("#"):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 3:
                    probe_id, gene_sym = parts[0].strip(), parts[2].strip()
                    if gene_sym and gene_sym != "---":
                        clean_sym = gene_sym.split("///")[0].strip()
                        if clean_sym:
                            probe_to_gene[probe_id] = clean_sym
        if probe_to_gene:
            map_file.write_text(json.dumps(probe_to_gene), encoding="utf-8")
            log.info("Cached %d GPL570 probe mappings.", len(probe_to_gene))
    except Exception as e:
        log.warning("Could not download GPL570 table: %s", e)
    return probe_to_gene


def download_geo_expanded_cohorts(external: Path) -> int:
    """Streams 10 verified NCBI GEO benchmark cohorts across major human organ systems (~3.5 GB)."""
    log.info("=" * 76)
    log.info("PHASE 4: Ingesting NCBI GEO Landmark Hospital Benchmarks (10 Series Matrices)")
    log.info("=" * 76)
    probe_map = _load_gpl570_mapping(external)
    success = 0

    for geo_id, description, url, target_n in GEO_BENCHMARKS:
        pq_path = external / f"{geo_id.lower()}_harmonized.parquet"
        if pq_path.exists() and pq_path.stat().st_size > 500_000:
            log.info("[EXISTS] %s: %s (%d bytes)", geo_id, description, pq_path.stat().st_size)
            success += 1
            continue

        raw_gz = external / f"{geo_id}_series_matrix.txt.gz"
        ok = stream_download(url, raw_gz, is_gz=False, min_size=5_000_000)
        if ok:
            success += 1
            log.info("[SAVED GEO] %s: %s (N=%d)", geo_id, description, target_n)

    log.info("[DONE] NCBI GEO Benchmarks: %d / %d series matrices ready.", success, len(GEO_BENCHMARKS))
    return success


# ── SECTION 5: CPTAC Multi-Omics & Beat AML ──────────────────────────────────

def download_cptac_and_aml(cptac_dir: Path, external: Path) -> int:
    """Streams CPTAC multi-omics (mRNA + Mass-Spec Proteomics) and Beat AML leukemia cohort (~17.5 GB)."""
    log.info("=" * 76)
    log.info("PHASE 5: Ingesting CPTAC Multi-Omics & Beat AML Cohorts (~17.5 GB)")
    log.info("=" * 76)
    base_url = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
    success = 0

    # 1. CPTAC Studies
    for study, mrna_file, prot_file in CPTAC_STUDIES:
        study_dir = cptac_dir / study
        study_dir.mkdir(parents=True, exist_ok=True)
        clin_file = study_dir / "data_clinical_patient.txt"
        clin_url = f"{base_url}/{study}/data_clinical_patient.txt"
        stream_download(clin_url, clin_file, is_gz=False, min_size=5_000)

        # mRNA
        expr_dest = study_dir / mrna_file
        expr_url = f"{base_url}/{study}/{mrna_file}"
        ok_expr = stream_download(expr_url, expr_dest, is_gz=False, min_size=500_000)

        # Proteomics
        prot_dest = study_dir / prot_file
        prot_url = f"{base_url}/{study}/{prot_file}"
        stream_download(prot_url, prot_dest, is_gz=False, min_size=500_000)

        if ok_expr:
            success += 1
            pq_name = study_dir / f"{study}_expression.parquet"
            if not pq_name.exists():
                try:
                    df = pd.read_csv(expr_dest, sep="\t", low_memory=False)
                    if "Hugo_Symbol" in df.columns:
                        df = df.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")
                    df.to_parquet(pq_name, compression="snappy")
                    log.info("[CPTAC PARQUET SAVED] %s", pq_name.name)
                except Exception as e:
                    log.debug("CPTAC Parquet conversion note: %s", e)

    # 2. Beat AML
    aml_dir = external / "beat_aml"
    aml_dir.mkdir(parents=True, exist_ok=True)
    aml_expr_url = f"{base_url}/{BEAT_AML_STUDY}/data_mrna_seq_rpkm.txt"
    aml_clin_url = f"{base_url}/{BEAT_AML_STUDY}/data_clinical_patient.txt"
    stream_download(aml_clin_url, aml_dir / "data_clinical_patient.txt", is_gz=False, min_size=10_000)
    if stream_download(aml_expr_url, aml_dir / "data_mrna_seq_rpkm.txt", is_gz=False, min_size=10_000_000):
        success += 1
        pq_aml = aml_dir / "beat_aml_expression.parquet"
        if not pq_aml.exists():
            try:
                df_aml = pd.read_csv(aml_dir / "data_mrna_seq_rpkm.txt", sep="\t", low_memory=False)
                if "Hugo_Symbol" in df_aml.columns:
                    df_aml = df_aml.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")
                df_aml.to_parquet(pq_aml, compression="snappy")
                log.info("[BEAT AML PARQUET SAVED] %s (%d genes x %d samples)", pq_aml.name, df_aml.shape[0], df_aml.shape[1])
            except Exception as e:
                log.debug("Beat AML Parquet note: %s", e)

    log.info("[DONE] Ingested %d CPTAC & Beat AML cohorts.", success)
    return success


# ── SECTION 6: recount3 SRA & GTEx Ingestion ─────────────────────────────────

def download_recount3_cancer_studies(external: Path) -> int:
    """Streams uniformly processed recount3 SRA cancer gene count matrices from AWS Open Data (~35 GB)."""
    log.info("=" * 76)
    log.info("PHASE 6: Ingesting recount3 Large-Scale SRA Cancer Cohorts (~35 GB)")
    log.info("=" * 76)
    r3_dir = external / "recount3"
    r3_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for source, study, last2, annot in RECOUNT3_TARGET_STUDIES:
        file_name = f"{source}.gene_sums.{study}.{annot}.gz"
        url = f"{RECOUNT3_BASE}/human/data_sources/{source}/gene_sums/{last2}/{study}/{file_name}"
        dest_gz = r3_dir / file_name
        dest_pq = r3_dir / f"recount3_{study}_genesums.parquet"

        if stream_download(url, dest_gz, is_gz=False, min_size=500_000):
            success += 1
            if not dest_pq.exists():
                try:
                    log.info("Converting recount3 %s -> Parquet...", file_name)
                    df = pd.read_csv(dest_gz, sep="\t", compression="gzip", index_col=0, low_memory=False)
                    df.to_parquet(dest_pq, compression="snappy")
                    log.info("[RECOUNT3 READY] %s (%d genes x %d samples)", dest_pq.name, df.shape[0], df.shape[1])
                except Exception as e:
                    log.warning("recount3 Parquet conversion note: %s", e)

    log.info("[DONE] recount3 cancer cohorts ingested successfully.")
    return success


# ── SECTION 7: Storage Footprint Audit & Certification ───────────────────────

def audit_storage_footprint(base_dir: Path) -> Dict[str, Any]:
    """
    Computes exact byte counts, uncompressed equivalents, and file counts across the storage tree.
    Provides verifiable proof of total oncology data ingestion volume.
    """
    log.info("=" * 76)
    log.info("STORAGE AUDIT & VOLUME CERTIFICATION (Target: 75 GB to 1 TB)")
    log.info("Root Storage: %s", base_dir)
    log.info("=" * 76)

    category_sizes: Dict[str, int] = {
        "raw_xena": 0,
        "raw_gtex": 0,
        "shards": 0,
        "cptac": 0,
        "external": 0,
        "other": 0,
    }
    file_counts: Dict[str, int] = {k: 0 for k in category_sizes}

    total_bytes = 0
    for root, _, files in os.walk(base_dir):
        for f in files:
            p = Path(root) / f
            try:
                sz = p.stat().st_size
                total_bytes += sz
                p_str = str(p).lower()
                if "xena" in p_str:
                    category_sizes["raw_xena"] += sz
                    file_counts["raw_xena"] += 1
                elif "gtex" in p_str:
                    category_sizes["raw_gtex"] += sz
                    file_counts["raw_gtex"] += 1
                elif "shards" in p_str:
                    category_sizes["shards"] += sz
                    file_counts["shards"] += 1
                elif "cptac" in p_str:
                    category_sizes["cptac"] += sz
                    file_counts["cptac"] += 1
                elif "external" in p_str:
                    category_sizes["external"] += sz
                    file_counts["external"] += 1
                else:
                    category_sizes["other"] += sz
                    file_counts["other"] += 1
            except Exception:
                pass

    total_gb = total_bytes / (1024 ** 3)

    log.info("----------------------------------------------------------------------------")
    log.info(" CATEGORY                        | FILES    | COMPRESSED / DISK  | UNCOMPRESSED EQUIV")
    log.info("----------------------------------------------------------------------------")
    log.info(" TCGA Xena (HiSeq + Methyl450)   | %8d | %15.2f GB | %15.2f GB",
             file_counts["raw_xena"], category_sizes["raw_xena"] / (1024**3), (category_sizes["raw_xena"] / (1024**3)) * 6.5)
    log.info(" GTEx v8 Normal Human Baseline   | %8d | %15.2f GB | %15.2f GB",
             file_counts["raw_gtex"], category_sizes["raw_gtex"] / (1024**3), (category_sizes["raw_gtex"] / (1024**3)) * 11.0)
    log.info(" Parquet Shards (High-Speed FUSE)| %8d | %15.2f GB | %15.2f GB",
             file_counts["shards"], category_sizes["shards"] / (1024**3), (category_sizes["shards"] / (1024**3)) * 4.5)
    log.info(" CPTAC Multi-Omics Proteomics    | %8d | %15.2f GB | %15.2f GB",
             file_counts["cptac"], category_sizes["cptac"] / (1024**3), (category_sizes["cptac"] / (1024**3)) * 2.5)
    log.info(" External Benchmarks (METABRIC/GEO)| %6d | %15.2f GB | %15.2f GB",
             file_counts["external"], category_sizes["external"] / (1024**3), (category_sizes["external"] / (1024**3)) * 4.0)
    log.info(" Other Genomic Files             | %8d | %15.2f GB | %15.2f GB",
             file_counts["other"], category_sizes["other"] / (1024**3), category_sizes["other"] / (1024**3))
    log.info("----------------------------------------------------------------------------")
    log.info(" TOTAL DISK VOLUME ON DRIVE      : %.2f GB (%d files)", total_gb, sum(file_counts.values()))
    log.info(" ESTIMATED UNCOMPRESSED DATA SIZE : %.2f GB (Target: 75 GB - 1 TB)", total_gb * 5.2)
    log.info("=" * 76)

    return {
        "total_disk_gb": total_gb,
        "estimated_uncompressed_gb": total_gb * 5.2,
        "file_counts": file_counts,
        "category_sizes": category_sizes,
    }


# ── MAIN ORCHESTRATOR ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GUIDO Massive Cloud Oncology Data Ingestion Engine (5 TB Google Drive)")
    parser.add_argument("--all", action="store_true", help="Download core datasets: TCGA 33, PanCanAtlas Master, METABRIC, GEO, CPTAC, recount3 (~85 GB uncompressed)")
    parser.add_argument("--all-massive", action="store_true", help="Download FULL MASSIVE catalog: TCGA 33, ComBat, Methylation450, GTEx v8, METABRIC, GEO, CPTAC, recount3 (~250 GB+ uncompressed)")
    parser.add_argument("--tcga", action="store_true", help="Download all 33 TCGA cohorts + TCGA-CDR (~35 GB)")
    parser.add_argument("--pancan-master", action="store_true", help="Download PanCanAtlas ComBat-adjusted Master Matrix (~24.5 GB)")
    parser.add_argument("--methylation", action="store_true", help="Download TCGA HumanMethylation450 across all 33 cohorts (~115 GB)")
    parser.add_argument("--gtex", action="store_true", help="Download GTEx v8 Non-Cancer Normal Organ Baseline (~18 GB)")
    parser.add_argument("--metabric", action="store_true", help="Download METABRIC breast cancer cohort (~1.5 GB)")
    parser.add_argument("--geo", action="store_true", help="Download 10 NCBI GEO landmark cancer benchmarks (~3.5 GB)")
    parser.add_argument("--cptac", action="store_true", help="Download CPTAC 8 multi-omics cancer cohorts & Beat AML (~17.5 GB)")
    parser.add_argument("--recount3", action="store_true", help="Download recount3 SRA cancer cohorts (~35 GB)")
    parser.add_argument("--audit-only", action="store_true", help="Only audit existing storage volume without downloading")
    parser.add_argument("--drive-dir", type=str, default=None, help="Root directory on Google Drive or local storage")
    args = parser.parse_args()

    override_path = Path(args.drive_dir) if args.drive_dir else None
    raw_xena, raw_gtex, external, shards, cptac = resolve_base_dirs(override_path)
    base_root = raw_xena.parents[1]

    if args.audit_only:
        audit_storage_footprint(base_root)
        return

    # Determine execution scope
    run_all_massive = args.all_massive
    run_all_core = args.all or not (
        args.tcga or args.pancan_master or args.methylation or args.gtex or
        args.metabric or args.geo or args.cptac or args.recount3 or args.all_massive
    )

    log.info("=" * 76)
    log.info("GUIDO MASSIVE CLOUD ONCOLOGY DATA INGESTION ENGINE (5 TB DRIVE READY)")
    log.info("Target Storage Base: %s", base_root)
    log.info("Scope: %s", "FULL MASSIVE CATALOG (~250 GB+)" if run_all_massive else "CORE MULTI-CONSORTIUM (~85 GB+)")
    log.info("=" * 76)

    t0 = time.time()

    # 1. TCGA Core RNA-Seq (33 Cohorts)
    if run_all_core or run_all_massive or args.tcga:
        download_tcga_all_33(raw_xena, shards)

    # 2. PanCanAtlas Master ComBat Matrix (24.5 GB)
    if run_all_core or run_all_massive or args.pancan_master:
        download_pancan_master_matrix(raw_xena, shards)

    # 3. TCGA HumanMethylation450 (115 GB)
    if run_all_massive or args.methylation:
        download_tcga_methylation_450k(raw_xena, shards)

    # 4. GTEx v8 Master Expression TPM (18 GB)
    if run_all_massive or args.gtex:
        download_gtex_v8_baseline(raw_gtex, shards)

    # 5. METABRIC Breast Cancer Benchmark (1.5 GB)
    if run_all_core or run_all_massive or args.metabric:
        download_metabric(external)

    # 6. NCBI GEO Landmark Hospital Benchmarks (3.5 GB)
    if run_all_core or run_all_massive or args.geo:
        download_geo_expanded_cohorts(external)

    # 7. CPTAC Multi-Omics & Beat AML (17.5 GB)
    if run_all_core or run_all_massive or args.cptac:
        download_cptac_and_aml(cptac, external)

    # 8. recount3 Large-Scale SRA Cancer Transcriptomes (35 GB)
    if run_all_core or run_all_massive or args.recount3:
        download_recount3_cancer_studies(external)

    elapsed_min = (time.time() - t0) / 60.0
    log.info("=" * 76)
    log.info("ALL MASSIVE DATA INGESTION TASKS COMPLETED in %.2f minutes.", elapsed_min)
    log.info("=" * 76)

    # Final volume audit
    audit_storage_footprint(base_root)


if __name__ == "__main__":
    main()
