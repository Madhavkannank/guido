"""
External Clinical Cohort Ingestion & Harmonization Module
========================================================

Provides automated acquisition, probe-to-symbol mapping, and outcome harmonization
for independent external benchmark datasets:

  1. METABRIC (Molecular Taxonomy of Breast Cancer International Consortium):
     - Source: cBioPortal Datahub (1,980 patients with longitudinal RFS and OS data)
     - Platform: Illumina HumanHT-12 v3 microarray

  2. GEO GSE31210 (Okayama et al. Lung Adenocarcinoma):
     - Source: NCBI GEO FTP (226 Stage I/II lung adenocarcinoma patients)
     - Platform: Affymetrix Human Genome U133 Plus 2.0 Array (GPL570)
     - Target: Relapse-Free Survival (RFS)

  3. Synthetic Benchmark Cohort:
     - Deterministic generator for offline testing and continuous integration
"""
from __future__ import annotations

import gzip
import io
import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXTERNAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "external"

METABRIC_EXPR_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_mrna_illumina_microarray.txt"
METABRIC_CLIN_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_patient.txt"
GEO_GSE31210_URL  = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31210/matrix/GSE31210_series_matrix.txt.gz"
GPL570_ANNOT_URL  = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"


def generate_synthetic_external_cohort(
    n_samples: int = 150,
    seed: int = 42,
    hallmark_pathways: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Generates a deterministic synthetic external cohort for validation and unit tests.
    Returns:
        (expression_df, binary_3yr_risk, times, events)
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"EXT-SAMPLE-{i:04d}" for i in range(n_samples)]

    # Generate synthetic gene expressions for hallmark genes and top markers
    all_genes = set()
    if hallmark_pathways:
        for genes in hallmark_pathways.values():
            all_genes.update(genes)
    else:
        all_genes = {
            "MKI67", "PCNA", "AURKA", "BIRC5", "VIM", "CDH2", "VEGFA", "HIF1A",
            "PKM", "CD274", "PDCD1", "SOX2", "PARP1", "BCL2", "IL6", "COL1A1", "NFE2L2"
        }
    # Add filler genes
    filler = [f"GENE_{i:03d}" for i in range(100)]
    gene_list = sorted(list(all_genes.union(filler)))

    # Synthetic expression matrix: samples x genes
    expr_matrix = rng.normal(loc=9.5, scale=1.5, size=(n_samples, len(gene_list)))
    X_df = pd.DataFrame(expr_matrix, index=sample_ids, columns=gene_list)

    # Synthetic survival dynamics: follow-up up to 5 years (1825 days)
    times = rng.uniform(60, 2000, size=n_samples).astype(float)
    # Proliferation-correlated risk score
    prolif_score = expr_matrix[:, 0] if len(gene_list) > 0 else rng.normal(size=n_samples)
    event_prob = 1.0 / (1.0 + np.exp(-(prolif_score - 9.5)))
    events = (rng.uniform(size=n_samples) < event_prob).astype(int)

    # 3-year landmark recurrence target
    is_high_risk = (events == 1) & (times <= 1095)
    y_series = pd.Series(is_high_risk.astype(int), index=sample_ids)
    time_series = pd.Series(times, index=sample_ids)
    event_series = pd.Series(events, index=sample_ids)

    return X_df, y_series, time_series, event_series


def fetch_metabric(
    out_dir: Optional[Path] = None,
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Fetches and harmonizes the METABRIC breast cancer cohort (~1,980 patients).
    Raises RuntimeError if network is unavailable or download fails (no silent synthetic fallback).
    For offline testing, use generate_synthetic_external_cohort() explicitly.

    Returns:
        (expression_df, binary_3yr_risk, times_days, events)
    """
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "metabric_harmonized.parquet"

    if cache_file.exists():
        logger.info("Loading cached METABRIC cohort from %s", cache_file)
        df_all = pd.read_parquet(cache_file)
        # Check if cache is corrupted by the legacy '0:Not Recurred' substring bug (e.g. event rate > 0.80)
        if df_all["_EVENT"].mean() > 0.80:
            logger.warning("Cached METABRIC cohort has corrupted event labels (mean event rate %.2f > 0.80). Re-harmonizing...", float(df_all["_EVENT"].mean()))
            try:
                cache_file.unlink()
            except Exception:
                pass
        else:
            if max_samples:
                df_all = df_all.iloc[:max_samples]
            meta_cols = ["_Y_3YR", "_TIME_DAYS", "_EVENT"]
            gene_cols = [c for c in df_all.columns if c not in meta_cols]
            return (
                df_all[gene_cols],
                df_all["_Y_3YR"],
                df_all["_TIME_DAYS"],
                df_all["_EVENT"]
            )

    logger.info("Attempting to fetch METABRIC dataset from cBioPortal datahub...")
    try:
        # Download clinical data
        req_clin = urllib.request.Request(METABRIC_CLIN_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_clin, timeout=45) as resp:
            lines = [line.decode("utf-8", errors="replace") for line in resp.readlines() if not line.startswith(b"#")]
        df_clin = pd.read_csv(io.StringIO("".join(lines)), sep="\t", index_col="PATIENT_ID")

        # Extract survival/recurrence endpoints: prefer Relapse-Free Survival (RFS) over Overall Survival (OS)
        # Strict parsing: avoid '0:Not Recurred' matching 'Recurred'
        if "RFS_MONTHS" in df_clin.columns and "RFS_STATUS" in df_clin.columns:
            time_days = pd.to_numeric(df_clin["RFS_MONTHS"], errors="coerce") * 30.4375
            rfs_str = df_clin["RFS_STATUS"].astype(str).str.strip()
            events = (rfs_str.str.startswith("1") | (rfs_str.str.contains("Recurred", case=False, na=False) & ~rfs_str.str.contains("Not", case=False, na=False))).astype(int)
        elif "OS_MONTHS" in df_clin.columns and "OS_STATUS" in df_clin.columns:
            time_days = pd.to_numeric(df_clin["OS_MONTHS"], errors="coerce") * 30.4375
            os_str = df_clin["OS_STATUS"].astype(str).str.strip()
            events = (os_str.str.startswith("1") | (os_str.str.contains("DECEASED|Dead", case=False, na=False) & ~os_str.str.contains("Living|Alive", case=False, na=False))).astype(int)
        else:
            raise ValueError("Required survival columns missing in METABRIC clinical table.")

        # Download expression data (microarray)
        req_expr = urllib.request.Request(METABRIC_EXPR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_expr, timeout=180) as resp:
            df_expr = pd.read_csv(resp, sep="\t", low_memory=False)

        if "Entrez_Gene_Id" in df_expr.columns:
            df_expr = df_expr.drop(columns=["Entrez_Gene_Id"])

        # Clean gene index
        if "Hugo_Symbol" in df_expr.columns:
            df_expr = df_expr.dropna(subset=["Hugo_Symbol"])
            df_expr = df_expr.drop_duplicates(subset=["Hugo_Symbol"], keep="first")
            df_expr = df_expr.set_index("Hugo_Symbol")

        common_patients = [
            p for p in df_clin.index
            if p in df_expr.columns and not pd.isna(time_days.loc[p]) and time_days.loc[p] > 0 and events.loc[p] in (0, 1)
        ]
        if len(common_patients) < 50:
            raise ValueError(f"Insufficient overlapping patients in METABRIC ({len(common_patients)})")

        X_df = df_expr[common_patients].T.astype(float).fillna(0.0)
        time_series = time_days.loc[common_patients].astype(float)
        event_series = events.loc[common_patients].astype(int)
        y_series = ((event_series == 1) & (time_series <= 1095)).astype(int)

        if max_samples:
            X_df = X_df.iloc[:max_samples]
            time_series = time_series.iloc[:max_samples]
            event_series = event_series.iloc[:max_samples]
            y_series = y_series.iloc[:max_samples]

        # Cache harmonized dataframe
        df_cached = X_df.copy()
        df_cached["_Y_3YR"] = y_series
        df_cached["_TIME_DAYS"] = time_series
        df_cached["_EVENT"] = event_series
        df_cached.to_parquet(cache_file, compression="snappy")
        logger.info("Successfully cached METABRIC cohort (%d patients)", len(common_patients))

        return X_df, y_series, time_series, event_series

    except Exception as e:
        logger.error("Failed to download live METABRIC cohort: %s", e)
        raise RuntimeError(f"Failed to download live METABRIC cohort from cBioPortal: {e}") from e


def _load_gpl570_probe_mapping() -> Dict[str, str]:
    """Downloads and caches GPL570 probe-to-gene mapping table from NCBI GEO FTP."""
    cache_path = EXTERNAL_DATA_DIR / "gpl570_probe_mapping.json"
    if cache_path.exists():
        import json
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    probe_to_gene: Dict[str, str] = {}
    try:
        logger.info("Fetching GPL570 annotation table from NCBI GEO...")
        req = urllib.request.Request(GPL570_ANNOT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
                in_table = False
                for line in gz:
                    if "!platform_table_begin" in line:
                        in_table = True
                        continue
                    if not in_table or line.startswith("#"):
                        continue
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 3:
                        probe_id, gene_symbol = parts[0].strip(), parts[2].strip()
                        if gene_symbol and gene_symbol != "---":
                            clean_sym = gene_symbol.split("///")[0].strip()
                            if clean_sym:
                                probe_to_gene[probe_id] = clean_sym

        if probe_to_gene:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            cache_path.write_text(json.dumps(probe_to_gene), encoding="utf-8")
            logger.info("Cached %d GPL570 probe-to-gene mappings", len(probe_to_gene))
    except Exception as exc:
        logger.warning("Could not download GPL570 annotation table: %s", exc)

    return probe_to_gene


def fetch_geo_gse31210(
    out_dir: Optional[Path] = None,
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Fetches and harmonizes NCBI GEO GSE31210 lung adenocarcinoma cohort (226 patients).
    Extracts recurrence endpoints (days before relapse/censor, relapse status) and
    maps Affymetrix HG-U133_Plus_2 probes to official HGNC gene symbols.
    Raises RuntimeError if network is unavailable or download fails (no silent synthetic fallback).
    """
    out_dir = out_dir or EXTERNAL_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "gse31210_harmonized.parquet"

    if cache_file.exists():
        logger.info("Loading cached GSE31210 cohort from %s", cache_file)
        df_all = pd.read_parquet(cache_file)
        if max_samples:
            df_all = df_all.iloc[:max_samples]
        meta_cols = ["_Y_3YR", "_TIME_DAYS", "_EVENT"]
        gene_cols = [c for c in df_all.columns if c not in meta_cols]
        return (
            df_all[gene_cols],
            df_all["_Y_3YR"],
            df_all["_TIME_DAYS"],
            df_all["_EVENT"]
        )

    logger.info("Attempting to fetch GSE31210 from NCBI GEO FTP...")
    try:
        probe_to_gene = _load_gpl570_probe_mapping()
        req = urllib.request.Request(GEO_GSE31210_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            with gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
                meta_lines = []
                for line in gz:
                    if "!series_matrix_table_begin" in line:
                        break
                    meta_lines.append(line.rstrip("\r\n"))

                df_probes = pd.read_csv(gz, sep="\t", comment="!", index_col=0, low_memory=False)

        # Parse clinical metadata
        sample_ids = []
        relapse_status = []
        relapse_days = []
        for line in meta_lines:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [s.strip('"') for s in line.split("\t")[1:]]
            elif "!Sample_characteristics_ch1\t\"relapse:" in line or (
                line.startswith("!Sample_characteristics_ch1") and "relapse:" in line and "days" not in line and "months" not in line
            ):
                relapse_status = [1 if ("relapsed" in s and "not relapsed" not in s) else 0 for s in line.split("\t")[1:]]
            elif "days before relapse/censor:" in line:
                for s in line.split("\t")[1:]:
                    val = s.split(":")[-1].strip('" ')
                    try:
                        relapse_days.append(float(val))
                    except Exception:
                        relapse_days.append(np.nan)

        if not sample_ids or not relapse_status or not relapse_days:
            raise ValueError("Failed to extract complete clinical relapse endpoints from GSE31210 metadata.")

        # Map probes to genes
        if probe_to_gene:
            df_probes["gene"] = df_probes.index.map(probe_to_gene)
            df_probes = df_probes.dropna(subset=["gene"])
            df_genes = df_probes.groupby("gene").mean()
        else:
            df_genes = df_probes

        df_genes.columns = [c.strip('"') for c in df_genes.columns]
        common = [s for s in sample_ids if s in df_genes.columns]

        clin_df = pd.DataFrame({
            "sample_id": sample_ids[:len(relapse_status)],
            "relapse": relapse_status,
            "days": relapse_days[:len(relapse_status)],
        }).set_index("sample_id").loc[common].dropna()
        clin_df = clin_df[(clin_df["days"] > 0) & clin_df["relapse"].isin([0, 1])]

        X_df = df_genes[clin_df.index].T.astype(float).fillna(0.0)
        time_series = clin_df["days"].astype(float)
        event_series = clin_df["relapse"].astype(int)
        y_series = ((event_series == 1) & (time_series <= 1095)).astype(int)

        if max_samples:
            X_df = X_df.iloc[:max_samples]
            time_series = time_series.iloc[:max_samples]
            event_series = event_series.iloc[:max_samples]
            y_series = y_series.iloc[:max_samples]

        df_cached = X_df.copy()
        df_cached["_Y_3YR"] = y_series
        df_cached["_TIME_DAYS"] = time_series
        df_cached["_EVENT"] = event_series
        df_cached.to_parquet(cache_file, compression="snappy")
        logger.info("Successfully cached GSE31210 cohort (%d patients, %d genes)", len(clin_df), X_df.shape[1])

        return X_df, y_series, time_series, event_series

    except Exception as exc:
        logger.error("Failed to download live GSE31210 cohort: %s", exc)
        raise RuntimeError(f"Failed to download live GSE31210 cohort from NCBI GEO: {exc}") from exc
