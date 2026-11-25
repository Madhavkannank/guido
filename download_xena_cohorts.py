#!/usr/bin/env python3
"""
Download TCGA Expression & Standardized Clinical Survival Matrices from UCSC Xena Hub
for Genomic Disease Risk Prediction (PFS / OS Recurrence & Progression Risk).

Key Endpoints:
  - RNA-Seq HiSeqV2: https://tcga.xenahubs.net/download/TCGA.{cancer}.sampleMap%2FHiSeqV2.gz
  - TCGA-CDR Survival: https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp

Usage:
    py -3.12 download_xena_cohorts.py KICH              # single cohort (fast)
    py -3.12 download_xena_cohorts.py KIRP BRCA LUAD    # specific cohorts
    py -3.12 download_xena_cohorts.py PAN-RENAL         # pan-organ group
    py -3.12 download_xena_cohorts.py all               # all clinical cohorts
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "raw" / "xena"

ALL_TCGA_COHORTS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"
]

def normalize_cancer_code(cancer: str) -> str:
    """Normalize 'TCGA-BRCA', 'tcga-brca', 'tcga.brca', or 'brca' to 'BRCA'."""
    return cancer.strip().upper().replace("TCGA-", "").replace("TCGA.", "")

def get_xena_tsv_name(cancer: str) -> str:
    """Return canonical TSV filename for a cancer type."""
    code = normalize_cancer_code(cancer)
    return f"TCGA.{code}.sampleMap_HiSeqV2.tsv"

# Universal mapping covering all 33 TCGA cohorts
XENA_FILE = {c: get_xena_tsv_name(c) for c in ALL_TCGA_COHORTS}

COHORT_GROUPS = {
    "PAN-RENAL"        : ["KIRC", "KIRP", "KICH"],
    "PAN-LUNG"         : ["LUAD", "LUSC"],
    "PAN-GI"           : ["COAD", "STAD", "READ", "ESCA"],
    "PAN-HEPATOBILIARY": ["LIHC", "CHOL"],
    "PAN-BREAST"       : ["BRCA"],
    "PAN-BRAIN"        : ["LGG", "GBM"],
    "PAN-GYN"          : ["OV", "UCEC", "CESC", "UCS"],
    "PAN-UROLOGICAL"   : ["BLCA", "PRAD", "TGCT"],
    "PAN-HEADNECK"     : ["HNSC"],
    "PAN-ENDOCRINE"    : ["THCA", "ACC", "PCPG"],
    "PAN-SARCOMA"      : ["SARC"],
    "PAN-MELANOMA"     : ["SKCM", "UVM"],
    "PAN-HEMATOLOGIC"  : ["LAML", "DLBC", "THYM"],
    "PAN-MESO"         : ["MESO"],
}

# Verified UCSC Xena CDN Endpoints (HTTP 200 OK)
BASE_EXPR_URL = "https://tcga.xenahubs.net/download/TCGA.{cancer}.sampleMap%2FHiSeqV2.gz"
BASE_CLIN_URL = "https://tcga.xenahubs.net/download/TCGA.{cancer}.sampleMap%2F{cancer}_clinicalMatrix"
TCGA_CDR_URL  = "https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp"
TCGA_CDR_FILE = "TCGA_CDR_survival.tsv"

USER_AGENT = "GUIDO-Clinical-Risk-Pipeline/1.0 (Mozilla/5.0; Windows NT 10.0; Win64; x64)"


def download_file(url: str, out_path: Path, is_gzip: bool = False, min_size: int = 1000) -> bool:
    """Streams and validates a file download from UCSC Xena Hub."""
    if out_path.exists() and out_path.stat().st_size >= min_size:
        log.info("Already downloaded: %s (%d bytes)", out_path.name, out_path.stat().st_size)
        return True

    log.info("Downloading from %s ...", url)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_out = OUT_DIR / f"temp_{out_path.name}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if is_gzip:
            gz_path = OUT_DIR / f"temp_{out_path.name}.gz"
            with urllib.request.urlopen(req, timeout=120) as response, open(gz_path, "wb") as f_gz:
                shutil.copyfileobj(response, f_gz)

            log.info("Decompressing %s -> %s ...", gz_path.name, out_path.name)
            with gzip.open(gz_path, "rb") as f_in, open(tmp_out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            if gz_path.exists():
                gz_path.unlink()
        else:
            with urllib.request.urlopen(req, timeout=120) as response, open(tmp_out, "wb") as f_out:
                shutil.copyfileobj(response, f_out)

        # Validate download size before replacing destination
        if tmp_out.stat().st_size < min_size:
            log.warning("Downloaded file too small (%d bytes < %d bytes min): %s",
                        tmp_out.stat().st_size, min_size, out_path.name)
            if tmp_out.exists():
                tmp_out.unlink()
            return False

        if out_path.exists():
            out_path.unlink()
        tmp_out.rename(out_path)
        log.info("Successfully saved: %s (%d bytes)", out_path.name, out_path.stat().st_size)
        return True

    except Exception as e:
        log.error("Failed to download %s: %s", url, e)
        if tmp_out.exists():
            tmp_out.unlink()
        return False


def download_tcga_cdr(force: bool = False) -> bool:
    """Downloads the standardized NIH Pan-Cancer Clinical Data Resource (TCGA-CDR)."""
    cdr_path = OUT_DIR / TCGA_CDR_FILE
    if cdr_path.exists() and cdr_path.stat().st_size > 2_000_000 and not force:
        log.info("Standardized TCGA-CDR matrix already present (%d bytes)", cdr_path.stat().st_size)
        return True

    log.info("Fetching standardized NIH Pan-Cancer Clinical Data Resource (TCGA-CDR)...")
    ok = download_file(TCGA_CDR_URL, cdr_path, is_gzip=False, min_size=2_000_000)
    if ok:
        # Validate that header has required columns
        with open(cdr_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split("\t")
        required_cols = {"sample", "_PATIENT", "PFI", "PFI.time", "OS", "OS.time"}
        missing = required_cols - set(header)
        if missing:
            log.error("TCGA-CDR table missing critical survival columns: %s", missing)
            return False
        log.info("TCGA-CDR matrix validated successfully (contains %d columns including PFI & OS)", len(header))
    return ok


def download_cohort(cancer: str) -> bool:
    """Downloads RNA-Seq expression matrix and standardized survival outcomes for a cohort."""
    cancer = normalize_cancer_code(cancer)
    if cancer not in ALL_TCGA_COHORTS and cancer not in XENA_FILE:
        log.warning("Cancer '%s' is not in standard 33 TCGA cohorts, but attempting dynamic fetch.", cancer)

    # 1. Download standardized Pan-Cancer TCGA-CDR (covers all cohorts)
    ok_cdr = download_tcga_cdr()

    # 2. Download RNA-Seq Expression matrix (HiSeqV2)
    expr_filename = XENA_FILE.get(cancer, get_xena_tsv_name(cancer))
    expr_file = OUT_DIR / expr_filename
    expr_url = BASE_EXPR_URL.format(cancer=cancer)
    ok_expr = download_file(expr_url, expr_file, is_gzip=True, min_size=5_000_000)

    # 3. Download cohort-specific clinicalMatrix (for legacy backward compatibility if available)
    clin_file = OUT_DIR / f"TCGA.{cancer}.sampleMap_clinicalMatrix.tsv"
    clin_url = BASE_CLIN_URL.format(cancer=cancer)
    download_file(clin_url, clin_file, is_gzip=False, min_size=10_000)

    return ok_expr and ok_cdr


def main():
    parser = argparse.ArgumentParser(description="Download TCGA Expression & Standardized Clinical Survival Datasets")
    parser.add_argument("cancers", nargs="+", help="Cancer or group names (e.g. PAN-RENAL KICH all)")
    parser.add_argument("--force-cdr", action="store_true", help="Force re-download of TCGA-CDR survival matrix")
    args = parser.parse_args()

    if args.force_cdr:
        download_tcga_cdr(force=True)

    requested = [c.upper().replace("TCGA-", "") for c in args.cancers]
    target_cancers = []

    if "ALL" in requested:
        target_cancers = sorted(list(XENA_FILE.keys()))
    else:
        for item in requested:
            if item in COHORT_GROUPS:
                target_cancers.extend(COHORT_GROUPS[item])
            else:
                target_cancers.append(item)

    # Deduplicate while preserving order
    target_cancers = list(dict.fromkeys(target_cancers))
    log.info("Target cohorts to download (%d total): %s", len(target_cancers), target_cancers)

    # Always ensure TCGA-CDR is downloaded first
    download_tcga_cdr()

    success_count = 0
    for c in target_cancers:
        if download_cohort(c):
            success_count += 1

    log.info("Downloaded %d / %d cohorts successfully.", success_count, len(target_cancers))


if __name__ == "__main__":
    main()
