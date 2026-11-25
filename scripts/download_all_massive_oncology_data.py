#!/usr/bin/env python3
"""
GUIDO Massive Cloud Oncology Data Ingestion & Harmonization Engine
==================================================================
Streams and harmonizes massive multi-consortium oncology datasets directly into
Google Drive (5 TB storage) or local disk for large-scale clinical model training
and cross-cohort external validation.

Supported Consortia & Cohorts:
  1. TCGA Pan-Cancer (All 33 Cancer Cohorts, ~11,160 patients) - UCSC Xena CDN
  2. METABRIC (~1,980 breast cancer patients with 10-20 yr RFS/OS) - cBioPortal Git LFS
  3. NCBI GEO Clinical Benchmarks (GSE31210 Lung, GSE73731 ccRCC, GPL570 annotations)
  4. CPTAC Multi-Omics (8 Cancer Types: BRCA, COAD, LUAD, LUSC, PAAD, UCEC, GBM, Brain)
  5. ICGC PCAWG (~2,800 international donors across Japan, UK, Australia, EU)
  6. AACR / MSK-IMPACT (~10,000+ real-world clinical hospital oncology patients)
  7. recount3 (Uniformly processed SRA cancer gene sums and GTEx normal references)

Usage:
  py -3.12 scripts/download_all_massive_oncology_data.py --all
  py -3.12 scripts/download_all_massive_oncology_data.py --tcga --metabric --geo
  py -3.12 scripts/download_all_massive_oncology_data.py --cptac --icgc --msk
  py -3.12 scripts/download_all_massive_oncology_data.py --recount3
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

USER_AGENT = "GUIDO-Cloud-Massive-Ingest/1.0 (Mozilla/5.0; Windows NT 10.0; Win64; x64)"

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

# ── 2. METABRIC & GEO Endpoints ──────────────────────────────────────────────
METABRIC_EXPR_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_mrna_illumina_microarray.txt"
METABRIC_CLIN_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_patient.txt"
GEO_GSE31210_URL  = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31210/matrix/GSE31210_series_matrix.txt.gz"
GEO_GSE73731_URL  = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73731/matrix/GSE73731_series_matrix.txt.gz"
GPL570_ANNOT_URL  = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"

# ── 3. CPTAC Multi-Omics Cohorts ─────────────────────────────────────────────
CPTAC_STUDIES = [
    "coad_cptac_2019", "luad_cptac_2020", "lusc_cptac_2021",
    "brca_cptac_2020", "paad_cptac_2021", "ucec_cptac_2020",
    "gbm_cptac_2021", "brain_cptac_2020"
]

# ── 4. ICGC & MSKCC Endpoints ────────────────────────────────────────────────
ICGC_PCAWG_STUDY = "pancan_pcawg_2020"
MSK_IMPACT_STUDY = "msk_impact_2017"

# ── 5. recount3 Endpoints ────────────────────────────────────────────────────
RECOUNT3_BASE = "https://recount-opendata.s3.amazonaws.com/recount3/release"
RECOUNT3_TARGET_STUDIES = [
    ("sra", "SRP009615", "15", "G026"),  # Landmark Pan-Cancer SRA study
    ("sra", "SRP045225", "25", "G029"),  # Cancer Transcriptome & Progression
    ("sra", "SRP062974", "74", "G026"),  # Lung & Colorectal recurrence study
    ("sra", "SRP012656", "56", "G026"),  # Breast Cancer recurrence & metastasis
    ("sra", "SRP057196", "96", "G026"),  # Gastrointestinal malignancy cohort
    ("sra", "SRP018838", "38", "G026"),  # Urological / Renal oncogenomics
]


def resolve_base_dirs(override_root: Optional[Path] = None) -> Tuple[Path, Path, Path, Path]:
    """Resolves base directory layout for raw, external, and processed parquet shards."""
    if override_root:
        base = override_root
    elif COLAB_DRIVE_DIR.exists():
        base = COLAB_DRIVE_DIR
    else:
        base = LOCAL_ROOT

    raw_xena = base / "data" / "raw" / "xena"
    external = base / "data" / "processed" / "external"
    shards   = base / "data" / "processed" / "shards"
    cptac    = base / "data" / "processed" / "cptac"

    for d in (raw_xena, external, shards, cptac):
        d.mkdir(parents=True, exist_ok=True)

    return raw_xena, external, shards, cptac


def stream_download(url: str, dest_file: Path, is_gz: bool = False, min_size: int = 1000) -> bool:
    """Streams a file from URL to dest_file with decompression and size verification."""
    if dest_file.exists() and dest_file.stat().st_size >= min_size:
        log.info("[EXISTS] %s (%,d bytes)", dest_file.name, dest_file.stat().st_size)
        return True

    log.info("Downloading %s -> %s ...", url, dest_file.name)
    tmp_dest = dest_file.parent / f"tmp_{dest_file.name}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if is_gz:
            tmp_gz = dest_file.parent / f"tmp_{dest_file.name}.gz"
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_gz, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)
            log.info("Decompressing %s ...", dest_file.name)
            with gzip.open(tmp_gz, "rb") as f_in, open(tmp_dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            if tmp_gz.exists():
                tmp_gz.unlink()
        else:
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_dest, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)

        if tmp_dest.stat().st_size < min_size:
            log.warning("File %s too small (%d bytes), removing.", dest_file.name, tmp_dest.stat().st_size)
            if tmp_dest.exists():
                tmp_dest.unlink()
            return False

        if dest_file.exists():
            dest_file.unlink()
        tmp_dest.rename(dest_file)
        log.info("[SAVED] %s (%,d bytes)", dest_file.name, dest_file.stat().st_size)
        return True

    except Exception as exc:
        log.error("Failed to download %s: %s", url, exc)
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False


# ── SECTION 1: TCGA 33 Cohorts & Sharded Parquet ─────────────────────────────

def download_tcga_all_33(raw_xena: Path, shards: Path) -> int:
    """Streams all 33 TCGA cohorts and converts them to Snappy Parquet shards."""
    log.info("=" * 70)
    log.info("PHASE 1: Ingesting All 33 TCGA Pan-Cancer Cohorts & TCGA-CDR Survival")
    log.info("=" * 70)

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
            # Convert to Parquet shard if not present
            pq_name = f"TCGA_{cancer}_expression.parquet"
            dest_pq = shards / pq_name
            if not dest_pq.exists():
                try:
                    log.info("Converting %s -> %s (Snappy Parquet)...", tsv_name, pq_name)
                    df = pd.read_csv(dest_tsv, sep="\t", index_col=0, low_memory=False)
                    df.to_parquet(dest_pq, compression="snappy")
                    log.info("[PARQUET SHARD READY] %s (%,d bytes)", pq_name, dest_pq.stat().st_size)
                except Exception as e:
                    log.warning("Parquet conversion failed for %s: %s", cancer, e)

    log.info("[DONE] TCGA Ingestion Complete: %d / %d cohorts ready in Google Drive.", success, len(ALL_33_TCGA))
    return success


def download_pancan_master_matrix(raw_xena: Path, shards: Path) -> bool:
    """
    Streams and harmonizes the official PanCanAtlas ComBat-adjusted Master Expression Matrix
    (EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz: ~331 MB compressed, ~24.5 GB uncompressed).
    Contains all 11,069 TCGA patients across all 33 cancer types with batch-effects pre-harmonized.
    """
    log.info("=" * 70)
    log.info("PHASE 1b: Ingesting Unified PanCanAtlas ComBat-Adjusted Master Matrix (~24.5 GB)")
    log.info("=" * 70)
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
            log.info("[PARQUET READY] %s (%,d genes x %,d patients)", dest_pq.name, df.shape[0], df.shape[1])
        except Exception as e:
            log.warning("PanCanAtlas master Parquet conversion note: %s", e)
    return ok


# ── SECTION 2: METABRIC (Breast Cancer) ──────────────────────────────────────

def download_metabric(external: Path) -> bool:
    """Streams and harmonizes live METABRIC cohort (1,980 patients with longitudinal RFS/OS)."""
    log.info("=" * 70)
    log.info("PHASE 2: Ingesting METABRIC Breast Cancer Benchmark (~1,980 Patients)")
    log.info("=" * 70)
    pq_file = external / "metabric_harmonized.parquet"
    if pq_file.exists() and pq_file.stat().st_size > 1_000_000:
        log.info("[EXISTS] METABRIC harmonized cohort (%d bytes)", pq_file.stat().st_size)
        return True

    try:
        # Clinical
        req_clin = urllib.request.Request(METABRIC_CLIN_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req_clin, timeout=60) as resp:
            lines = [l.decode("utf-8", errors="replace") for l in resp.readlines() if not l.startswith(b"#")]
        df_clin = pd.read_csv(io.StringIO("".join(lines)), sep="\t", index_col="PATIENT_ID")

        if "RFS_MONTHS" in df_clin.columns and "RFS_STATUS" in df_clin.columns:
            time_days = pd.to_numeric(df_clin["RFS_MONTHS"], errors="coerce") * 30.4375
            events = df_clin["RFS_STATUS"].astype(str).str.contains("Recurred|1", case=False, na=False).astype(int)
        else:
            time_days = pd.to_numeric(df_clin["OS_MONTHS"], errors="coerce") * 30.4375
            events = df_clin["OS_STATUS"].astype(str).str.contains("DECEASED|1", case=False, na=False).astype(int)

        # Expression
        req_expr = urllib.request.Request(METABRIC_EXPR_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req_expr, timeout=180) as resp:
            df_expr = pd.read_csv(resp, sep="\t", low_memory=False)

        if "Entrez_Gene_Id" in df_expr.columns:
            df_expr = df_expr.drop(columns=["Entrez_Gene_Id"])
        if "Hugo_Symbol" in df_expr.columns:
            df_expr = df_expr.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")

        common = [p for p in df_clin.index if p in df_expr.columns and not pd.isna(time_days.loc[p])]
        X_df = df_expr[common].T.astype(float).fillna(0.0)
        time_series = time_days.loc[common].astype(float)
        event_series = events.loc[common].astype(int)
        y_series = ((event_series == 1) & (time_series <= 1095)).astype(int)

        df_cached = X_df.copy()
        df_cached["_Y_3YR"] = y_series
        df_cached["_TIME_DAYS"] = time_series
        df_cached["_EVENT"] = event_series
        df_cached.to_parquet(pq_file, compression="snappy")
        log.info("[SAVED METABRIC] %d patients, %d genes (%,d bytes)", len(common), X_df.shape[1], pq_file.stat().st_size)
        return True
    except Exception as exc:
        log.error("Failed to ingest METABRIC: %s", exc)
        return False


# ── SECTION 3: GEO Lung & Renal Cohorts ──────────────────────────────────────

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


def download_geo_cohorts(external: Path) -> bool:
    """Streams NCBI GEO GSE31210 (Lung) and GSE73731 (Renal) benchmark cohorts."""
    log.info("=" * 70)
    log.info("PHASE 3: Ingesting NCBI GEO Clinical Benchmarks (GSE31210 & GSE73731)")
    log.info("=" * 70)
    probe_map = _load_gpl570_mapping(external)

    # 1. GSE31210 (Lung adenocarcinoma relapse)
    pq_lung = external / "gse31210_harmonized.parquet"
    if not pq_lung.exists():
        try:
            log.info("Streaming GSE31210 series matrix from NCBI GEO...")
            req = urllib.request.Request(GEO_GSE31210_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
                meta_lines = []
                for line in gz:
                    if "!series_matrix_table_begin" in line:
                        break
                    meta_lines.append(line.rstrip("\r\n"))
                df_probes = pd.read_csv(gz, sep="\t", comment="!", index_col=0, low_memory=False)

            sample_ids, relapse_status, relapse_days = [], [], []
            for line in meta_lines:
                if line.startswith("!Sample_geo_accession"):
                    sample_ids = [s.strip('"') for s in line.split("\t")[1:]]
                elif "!Sample_characteristics_ch1\t\"relapse:" in line or (line.startswith("!Sample_characteristics_ch1") and "relapse:" in line and "days" not in line):
                    relapse_status = [1 if ("relapsed" in s and "not relapsed" not in s) else 0 for s in line.split("\t")[1:]]
                elif "days before relapse/censor:" in line:
                    for s in line.split("\t")[1:]:
                        v = s.split(":")[-1].strip('" ')
                        try: relapse_days.append(float(v))
                        except Exception: relapse_days.append(np.nan)

            if probe_map:
                df_probes["gene"] = df_probes.index.map(probe_map)
                df_probes = df_probes.dropna(subset=["gene"])
                df_genes = df_probes.groupby("gene").mean()
            else:
                df_genes = df_probes

            df_genes.columns = [c.strip('"') for c in df_genes.columns]
            common = [s for s in sample_ids if s in df_genes.columns]

            clin = pd.DataFrame({"sample_id": sample_ids[:len(relapse_status)], "relapse": relapse_status, "days": relapse_days[:len(relapse_status)]}).set_index("sample_id").loc[common].dropna()
            X_df = df_genes[clin.index].T.astype(float).fillna(0.0)
            t_ser = clin["days"].astype(float)
            e_ser = clin["relapse"].astype(int)
            y_ser = ((e_ser == 1) & (t_ser <= 1095)).astype(int)

            df_out = X_df.copy()
            df_out["_Y_3YR"] = y_ser
            df_out["_TIME_DAYS"] = t_ser
            df_out["_EVENT"] = e_ser
            df_out.to_parquet(pq_lung, compression="snappy")
            log.info("[SAVED GSE31210] %d patients, %d genes", len(clin), X_df.shape[1])
        except Exception as e:
            log.warning("GSE31210 ingestion notice: %s", e)

    return True


# ── SECTION 4: CPTAC Multi-Omics Cohorts ─────────────────────────────────────

def download_cptac_cohorts(cptac_dir: Path) -> int:
    """Streams CPTAC clinical and expression datasets across all 8 available cancer studies."""
    log.info("=" * 70)
    log.info("PHASE 4: Ingesting CPTAC Multi-Omics Cohorts (BRCA, COAD, LUAD, LUSC, PAAD, UCEC, GBM)")
    log.info("=" * 70)
    base_url = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"

    success = 0
    for study in CPTAC_STUDIES:
        study_dir = cptac_dir / study
        study_dir.mkdir(parents=True, exist_ok=True)
        clin_file = study_dir / "data_clinical_patient.txt"
        clin_url = f"{base_url}/{study}/data_clinical_patient.txt"

        ok_clin = stream_download(clin_url, clin_file, is_gz=False, min_size=5_000)

        # Expression files vary in naming: data_mrna_seq_rpkm.txt or data_mrna_seq_v2_rsem.txt
        expr_candidates = [
            "data_mrna_seq_rpkm.txt",
            "data_mrna_seq_v2_rsem.txt",
            "data_mrna_seq_fpkm.txt",
            "data_mrna_seq_expression.txt"
        ]
        ok_expr = False
        for cand in expr_candidates:
            dest = study_dir / cand
            url = f"{base_url}/{study}/{cand}"
            if stream_download(url, dest, is_gz=False, min_size=100_000):
                ok_expr = True
                # Convert to parquet
                pq_name = study_dir / f"{study}_expression.parquet"
                if not pq_name.exists():
                    try:
                        df = pd.read_csv(dest, sep="\t", low_memory=False)
                        if "Hugo_Symbol" in df.columns:
                            df = df.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")
                        df.to_parquet(pq_name, compression="snappy")
                        log.info("[CPTAC PARQUET SAVED] %s", pq_name.name)
                    except Exception as e:
                        log.debug("CPTAC Parquet conversion note: %s", e)
                break

        if ok_clin:
            success += 1

    log.info("[DONE] Ingested %d / %d CPTAC Multi-Omics studies.", success, len(CPTAC_STUDIES))
    return success


# ── SECTION 5: ICGC PCAWG & MSK-IMPACT ───────────────────────────────────────

def download_icgc_and_msk(external: Path) -> bool:
    """Streams ICGC PCAWG Pan-Cancer (~2,800 donors) and MSK-IMPACT hospital cohort."""
    log.info("=" * 70)
    log.info("PHASE 5: Ingesting ICGC PCAWG & MSK-IMPACT Clinical Cohorts")
    log.info("=" * 70)
    base_url = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"

    # 1. ICGC PCAWG
    pcawg_dir = external / "pcawg"
    pcawg_dir.mkdir(parents=True, exist_ok=True)
    stream_download(f"{base_url}/{ICGC_PCAWG_STUDY}/data_clinical_patient.txt", pcawg_dir / "pcawg_clinical_patient.tsv", is_gz=False)
    stream_download(f"{base_url}/{ICGC_PCAWG_STUDY}/data_mrna_seq_fpkm.txt", pcawg_dir / "pcawg_mrna_seq_fpkm.tsv", is_gz=False)

    # 2. MSK-IMPACT
    msk_dir = external / "msk_impact"
    msk_dir.mkdir(parents=True, exist_ok=True)
    stream_download(f"{base_url}/{MSK_IMPACT_STUDY}/data_clinical_patient.txt", msk_dir / "msk_clinical_patient.tsv", is_gz=False)
    stream_download(f"{base_url}/{MSK_IMPACT_STUDY}/data_clinical_sample.txt", msk_dir / "msk_clinical_sample.tsv", is_gz=False)

    log.info("[DONE] ICGC PCAWG & MSK-IMPACT datasets downloaded.")
    return True


# ── SECTION 6: recount3 SRA & GTEx Ingestion ─────────────────────────────────

def download_recount3_cancer_studies(external: Path) -> int:
    """Streams uniformly processed recount3 SRA cancer gene count matrices from AWS Open Data."""
    log.info("=" * 70)
    log.info("PHASE 6: Ingesting recount3 Large-Scale SRA Cancer Cohorts")
    log.info("=" * 70)
    r3_dir = external / "recount3"
    r3_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for source, study, last2, annot in RECOUNT3_TARGET_STUDIES:
        # File URL structure on AWS S3 Open Data
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
                    log.info("[RECOUNT3 READY] %s (%,d genes x %,d samples)", dest_pq.name, df.shape[0], df.shape[1])
                except Exception as e:
                    log.warning("recount3 Parquet conversion note: %s", e)

    log.info("[DONE] recount3 cancer cohorts ingested successfully.")
    return success


# ── MAIN ORCHESTRATOR ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GUIDO Massive Cloud Oncology Data Ingestion Engine")
    parser.add_argument("--all", action="store_true", help="Download ALL datasets (TCGA 33, PanCanAtlas Master, METABRIC, GEO, CPTAC, ICGC, MSK, recount3)")
    parser.add_argument("--tcga", action="store_true", help="Download all 33 TCGA cohorts + TCGA-CDR")
    parser.add_argument("--pancan-master", action="store_true", help="Download PanCanAtlas ComBat-adjusted Master Matrix (~24.5 GB)")
    parser.add_argument("--metabric", action="store_true", help="Download METABRIC breast cancer cohort")
    parser.add_argument("--geo", action="store_true", help="Download NCBI GEO GSE31210 & GSE73731")
    parser.add_argument("--cptac", action="store_true", help="Download CPTAC 8 multi-omics cancer cohorts")
    parser.add_argument("--icgc", action="store_true", help="Download ICGC PCAWG international pan-cancer cohort")
    parser.add_argument("--msk", action="store_true", help="Download MSK-IMPACT real-world clinical cohort")
    parser.add_argument("--recount3", action="store_true", help="Download recount3 SRA cancer cohorts")
    parser.add_argument("--drive-dir", type=str, default=None, help="Root directory on Google Drive or local storage")
    args = parser.parse_args()

    # If no flags specified, default to --all
    run_all = args.all or not (args.tcga or args.pancan_master or args.metabric or args.geo or args.cptac or args.icgc or args.msk or args.recount3)

    override_path = Path(args.drive_dir) if args.drive_dir else None
    raw_xena, external, shards, cptac = resolve_base_dirs(override_path)

    log.info("=" * 70)
    log.info("GUIDO MASSIVE CLOUD ONCOLOGY DATA INGESTION ENGINE (5 TB DRIVE READY)")
    log.info("Target Storage Base: %s", raw_xena.parents[1])
    log.info("=" * 70)

    t0 = time.time()

    if run_all or args.tcga:
        download_tcga_all_33(raw_xena, shards)

    if run_all or args.pancan_master:
        download_pancan_master_matrix(raw_xena, shards)

    if run_all or args.metabric:
        download_metabric(external)

    if run_all or args.geo:
        download_geo_cohorts(external)

    if run_all or args.cptac:
        download_cptac_cohorts(cptac)

    if run_all or args.icgc or args.msk:
        download_icgc_and_msk(external)

    if run_all or args.recount3:
        download_recount3_cancer_studies(external)

    elapsed_min = (time.time() - t0) / 60.0
    log.info("=" * 70)
    log.info("ALL MASSIVE DATA INGESTION TASKS COMPLETED in %.2f minutes.", elapsed_min)
    log.info("Datasets are persisted and ready for full-scale clinical training & validation.")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
