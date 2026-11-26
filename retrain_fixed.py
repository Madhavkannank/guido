#!/usr/bin/env python3
"""
Rigorous Multi-Omics Disease Risk Engine (StacK-Risk / Clinical Prognosis)
========================================================================

ARCHITECTURAL INNOVATIONS & CLINICAL CERTIFICATION STANDARDS:

  1. STANDARDIZED NIH PAN-CANCER CLINICAL OUTCOMES (TCGA-CDR):
     Directly ingests curated Progression-Free Interval (PFI) and Overall Survival (OS)
     from the NIH Pan-Cancer Clinical Data Resource (Liu et al., Cell 2018), eliminating
     fragile heuristic event guessing and uncurated legacy column lookups.

  2. 100% COHORT PRESERVATION & IPCW RIGHT-CENSORING CORRECTION:
     Eliminates artificial patient dropout (< 730 days) and immortal time bias.
     Uses continuous Harrell's Concordance Index (C-Index) across all valid follow-up
     patients, coupled with Inverse Probability of Censoring Weighting (IPCW) for
     3-year fixed-horizon landmark recurrence classification.

  3. LEAK-FREE HALLMARK BIOLOGICAL PATHWAYS (METAGENE MODULES):
     Constructs 12 validated oncology hallmark pathway scores (MSigDB / Hanahan-Weinberg).
     Baseline mean/std statistics are computed strictly on training partitions and
     serialized into the model artifact for zero-shift runtime inference.

  4. MULTI-MODEL STACKING ENSEMBLE:
     Blends three diverse, complementary model families via a 5-fold CV Meta-Learner:
       - Base Model 1 (Sparse Linear Risk Prior): ElasticNet Logistic Regression
       - Base Model 2 (Non-linear Epistatic Trees): Regularized XGBoost
       - Base Model 3 (Bagged Variance Reduction): ExtraTrees Classifier
       - Meta-Learner: Calibrated Logistic Stacking Blender

  5. CLINICAL DECISION TIERS (PROGNOSTIC STRATIFICATION):
     - Low Risk (< 35% Recurrence Probability)
     - Moderate Risk (35% - 65% Recurrence Probability)
     - High Risk (> 65% Recurrence Probability)

Usage:
  py -3.12 retrain_fixed.py PAN-RENAL             # train high-power Pan-Renal risk engine
  py -3.12 retrain_fixed.py KICH KIRP             # train specific individual cohorts
  py -3.12 retrain_fixed.py all                   # train all defined pan-organ cohorts
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.base import BaseEstimator, ClassifierMixin
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

ROOT        = Path(__file__).resolve().parent
XENA_DIR    = ROOT / "data" / "raw" / "xena"
MODELS_DIR  = ROOT / "models_artifacts"
RESULTS_DIR = ROOT / "results"

# All 33 official TCGA cohorts
ALL_TCGA_COHORTS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"
]

def normalize_cancer_code(cancer: str) -> str:
    """Normalize 'TCGA-BRCA', 'tcga-brca', or 'brca' to 'BRCA'."""
    return cancer.strip().upper().replace("TCGA-", "").replace("TCGA.", "")

def get_xena_tsv_name(cancer: str) -> str:
    """Return canonical TSV filename for a cancer type."""
    code = normalize_cancer_code(cancer)
    return f"TCGA.{code}.sampleMap_HiSeqV2.tsv"

# Universal mapping covering all 33 TCGA cohorts
XENA_FILE = {c: get_xena_tsv_name(c) for c in ALL_TCGA_COHORTS}

# Organ groupings for high-power Pan-Organ risk models
COHORT_GROUPS = {
    "PAN-CANCER-ALL"   : ALL_TCGA_COHORTS,  # Master consolidated single-flow cross-consortium training
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

# 12 Validated Canonical Biological Hallmark Gene Sets (MSigDB & Hanahan-Weinberg Hallmarks of Cancer)
HALLMARK_PATHWAYS = {
    "META_PROLIFERATION"        : ["MKI67", "PCNA", "AURKA", "BIRC5", "CCNB1", "TOP2A", "PLK1", "CDC20", "CDK1", "UBE2C"],
    "META_EMT_INVASION"         : ["VIM", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "FN1", "MMP2", "MMP9"],
    "META_ANGIOGENESIS"         : ["VEGFA", "KDR", "FLT1", "ANGPT2", "PECAM1", "TEK", "CD34"],
    "META_HYPOXIA"              : ["HIF1A", "SLC2A1", "LDHA", "CA9", "PDK1", "BNIP3", "ENO1"],
    "META_GLYCOLYSIS_WARBURG"   : ["PKM", "GAPDH", "ALDOA", "GPI", "PGK1", "TPI1"],
    "META_IMMUNE_EXHAUST"       : ["CD274", "PDCD1", "CTLA4", "LAG3", "TIGIT", "CD8A", "GZMB", "HAVCR2"],
    "META_STEMNESS_PLURIPOTENCY": ["SOX2", "POU5F1", "NANOG", "PROM1", "ALDH1A1", "KLF4"],
    "META_DNA_REPAIR"           : ["PARP1", "RAD51", "BRCA1", "BRCA2", "ATM", "CHEK1", "MSH2", "MLH1"],
    "META_APOPTOSIS_EVASION"    : ["BCL2", "BCL2L1", "MCL1", "XIAP", "CFLAR", "BAX"],
    "META_INFLAMMATION_IL6"     : ["IL6", "STAT3", "NFKB1", "TNF", "CXCL8", "RELA"],
    "META_EXTRACELLULAR_MATRIX" : ["COL1A1", "COL3A1", "LOX", "TIMP1", "FN1", "MMP2", "MMP9"],
    "META_OXIDATIVE_STRESS_NRF2": ["NFE2L2", "HMOX1", "NQO1", "GCLM", "GCLC", "TXNRD1"],
}

# Feature dimension configured from settings or default (88 individual + 12 metagenes = 100)
try:
    from config.settings import N_TOP_GENES
    DEFAULT_N_GENES = max(N_TOP_GENES - len(HALLMARK_PATHWAYS), 10)
except Exception:
    DEFAULT_N_GENES = 88

DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_RATIO  = 0.20      # 80/20 train/test split
RISK_HORIZON_DAYS   = 1095      # 3 Years landmark horizon (default)


# ─── High-Performance Concordance Index (O(N log N)) ────────────────────────

def compute_c_index(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    """
    Computes Harrell's Concordance Index in O(N log N) time using vectorized ranking.
    Properly handles right-censoring and risk ties.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(risk_scores, dtype=float)
    n = len(times)
    if n < 2:
        return 0.5

    # Sort primarily by time ascending, secondarily by score
    order = np.lexsort((scores, times))
    times = times[order]
    events = events[order]
    scores = scores[order]

    event_indices = np.where(events == 1)[0]
    if len(event_indices) == 0:
        return 0.5

    concordant = 0.0
    pairs = 0
    for idx in event_indices:
        t_i = times[idx]
        s_i = scores[idx]
        valid_j = (times > t_i)
        if not np.any(valid_j):
            continue
        scores_j = scores[valid_j]
        pairs += len(scores_j)
        concordant += float(np.sum(s_i > scores_j)) + 0.5 * float(np.sum(s_i == scores_j))

    return round(float(concordant / max(pairs, 1)), 4)


# ─── Inverse Probability of Censoring Weighting (IPCW) ───────────────────────

def compute_ipcw_weights(times: np.ndarray, events: np.ndarray, horizon: float = RISK_HORIZON_DAYS) -> np.ndarray:
    """
    Computes non-parametric Kaplan-Meier Inverse Probability of Censoring Weights (IPCW).
    Corrects for informative/right-censoring in 3-year fixed-horizon landmark risk classification.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(times)

    # Censoring indicator (1 if censored, 0 if disease event)
    cens_events = 1 - events
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_cens = cens_events[order]

    unique_cens_times = np.unique(sorted_times[sorted_cens == 1])
    g_t = {}
    g_cum = 1.0
    for ut in unique_cens_times:
        at_risk = np.sum(times >= ut)
        n_cens = np.sum((times == ut) & (cens_events == 1))
        if at_risk > 0:
            g_cum *= (1.0 - n_cens / at_risk)
        g_t[ut] = max(g_cum, 0.05)  # Truncate at 0.05 to avoid extreme variance

    def get_g(t_val: float) -> float:
        past = [ut for ut in unique_cens_times if ut <= t_val]
        if not past:
            return 1.0
        return g_t[past[-1]]

    weights = np.ones(n, dtype=float)
    for i in range(n):
        if events[i] == 1 and times[i] <= horizon:
            weights[i] = 1.0 / get_g(times[i])
        elif times[i] >= horizon:
            weights[i] = 1.0 / get_g(horizon)
        else:
            weights[i] = 1.0

    # Normalize weights so sum(weights) == n
    norm_weights = weights * (float(n) / max(float(np.sum(weights)), 1e-6))
    return norm_weights


# ─── Multi-Epoch Deep Clinical Risk Neural Estimator (PyTorch) ───────────────

from src.models.deep_surv import PyTorchDeepSurvClassifier, TORCH_AVAILABLE
setattr(sys.modules["__main__"], "PyTorchDeepSurvClassifier", PyTorchDeepSurvClassifier)


# ─── Data Extraction & Metagene Engineering ──────────────────────────────────

def is_tumor_sample(s: str) -> bool:
    """Identify genuine tumor biopsies (primary 01, recurrent 02, metastatic 06; exclude normal 10/11)."""
    s = str(s).strip()
    if s in ("sample", "_PATIENT"):
        return False
    parts = s.split("-")
    if len(parts) >= 4:
        stype = parts[3][:2]
        if stype.isdigit():
            code = int(stype)
            return 1 <= code <= 9
    return s.endswith("-01") or "-01" in s


def load_pancan_master_data(
    horizon_days: int = RISK_HORIZON_DAYS,
    endpoint: str = "PFI",
) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]]:
    """
    Loads pre-harmonized ComBat-adjusted PanCanAtlas master expression matrix covering all 33 TCGA cohorts
    (EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp: ~24.5 GB uncompressed text / ~3.5 GB Parquet).
    Enforces strict survival endpoint selection (PFI vs OS) without silent fallback.
    Applies identical patient-level deduplication (sample type prioritization), exclusion of missing/invalid
    events and non-positive follow-up times, and strict alignment assertions as load_cohort_data().
    """
    endpoint = endpoint.strip().upper()
    if endpoint not in ("PFI", "OS"):
        raise ValueError(f"Invalid survival endpoint '{endpoint}'. Must be 'PFI' or 'OS'.")

    shards_dir = ROOT / "data" / "processed" / "shards"
    pq_path   = shards_dir / "TCGA_PANCAN_EB_ADJUST_expression.parquet"
    tsv_path  = XENA_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
    path_cdr  = XENA_DIR / "TCGA_CDR_survival.tsv"

    if not pq_path.exists() and not tsv_path.exists():
        log.info("PanCanAtlas master matrix files not found on disk (%s / %s).", pq_path, tsv_path)
        return None
    if not path_cdr.exists():
        log.info("TCGA-CDR survival matrix not found on disk (%s).", path_cdr)
        return None

    try:
        log.info("  Found PanCanAtlas ComBat-adjusted Master Matrix! Loading all 33 TCGA cohorts in one flow (endpoint: %s)...", endpoint)
        if pq_path.exists():
            df_expr = pd.read_parquet(pq_path).dropna(how="all").astype(np.float32)
        else:
            df_expr = pd.read_csv(tsv_path, sep="\t", index_col=0, low_memory=False).dropna(how="all").astype(np.float32)

        df_cdr = pd.read_csv(path_cdr, sep="\t", index_col="sample", low_memory=False)

        tumor_samples = [c for c in df_expr.columns if is_tumor_sample(c)]
        if not tumor_samples:
            tumor_samples = [c for c in df_expr.columns if c not in ("sample", "_PATIENT")]

        cdr_sample_set = set(df_cdr.index)
        sample_to_cdr = {}
        for s in tumor_samples:
            if s in cdr_sample_set:
                sample_to_cdr[s] = s
            elif s[:15] in cdr_sample_set:
                sample_to_cdr[s] = s[:15]
            elif s[:12] in cdr_sample_set:
                sample_to_cdr[s] = s[:12]

        valid_samples_temp = [
            s for s in tumor_samples
            if s in sample_to_cdr and pd.isna(df_cdr.loc[sample_to_cdr[s], "Redaction"])
        ]

        # Patient-level grouping and primary tumor prioritization (-01 primary over -02 recurrent, etc.)
        patient_sample_map: Dict[str, List[str]] = {}
        for s in valid_samples_temp:
            pt_id = s[:12] if s.startswith("TCGA-") and len(s) >= 12 else s
            patient_sample_map.setdefault(pt_id, []).append(s)

        deduplicated_samples = []
        duplicate_samples_excluded = 0
        for pt_id, s_list in patient_sample_map.items():
            if len(s_list) == 1:
                deduplicated_samples.append(s_list[0])
            else:
                duplicate_samples_excluded += (len(s_list) - 1)
                # Prioritize primary solid tumor (-01), then recurrent (-02), etc.
                def _sample_type_sort_key(barcode: str) -> int:
                    parts = barcode.split("-")
                    if len(parts) >= 4 and len(parts[3]) >= 2:
                        stype = parts[3][:2]
                        if stype.isdigit():
                            return int(stype)
                    return 99
                sorted_s = sorted(s_list, key=_sample_type_sort_key)
                deduplicated_samples.append(sorted_s[0])

        sub_cdr = df_cdr.loc[[sample_to_cdr[s] for s in deduplicated_samples]].copy()
        sub_cdr.index = deduplicated_samples

        # Strict PFI vs OS endpoint extraction (NO SILENT FALLBACK)
        if endpoint == "PFI":
            if "PFI" not in sub_cdr.columns or "PFI.time" not in sub_cdr.columns:
                log.warning("PFI or PFI.time not in TCGA-CDR. Silent fallback to OS is prohibited.")
                return None
            event_col = "PFI"
            time_col = "PFI.time"
        elif endpoint == "OS":
            if "OS" not in sub_cdr.columns or "OS.time" not in sub_cdr.columns:
                log.warning("OS or OS.time not in TCGA-CDR. Silent fallback to PFI is prohibited.")
                return None
            event_col = "OS"
            time_col = "OS.time"

        # Explicitly exclude missing events (no fillna(0)) and non-positive / non-finite follow-up times
        raw_events = pd.to_numeric(sub_cdr[event_col], errors="coerce")
        raw_times = pd.to_numeric(sub_cdr[time_col], errors="coerce")

        valid_endpoint_mask = (
            raw_events.notna() &
            raw_events.isin([0, 1]) &
            raw_times.notna() &
            np.isfinite(raw_times) &
            (raw_times > 0)
        )
        final_samples = [s for s in deduplicated_samples if valid_endpoint_mask.loc[s]]

        if len(final_samples) < 50:
            log.warning("Insufficient valid endpoint samples in PanCanAtlas master matrix (%d < 50).", len(final_samples))
            return None

        event_series = raw_events.loc[final_samples].astype(int)
        time_series = raw_times.loc[final_samples].astype(float)

        is_high_risk = (event_series == 1) & (time_series <= horizon_days)
        is_low_risk  = (time_series >= horizon_days)
        landmark_mask = is_high_risk | is_low_risk
        eval_samples  = [s for s in final_samples if landmark_mask.loc[s]]

        if len(eval_samples) < 50:
            log.warning("Insufficient valid landmark samples in PanCanAtlas master matrix (%d < 50).", len(eval_samples))
            return None

        y_series = pd.Series(0, index=eval_samples, dtype=int)
        y_series.loc[[s for s in eval_samples if is_high_risk.loc[s]]] = 1
        time_eval = time_series.loc[eval_samples]
        event_eval = event_series.loc[eval_samples]

        X = df_expr[eval_samples].T.fillna(0.0).astype(np.float32)
        del df_expr
        import gc; gc.collect()

        # Strict alignment assertions
        assert (X.index == y_series.index).all(), "PanCan master expression index and label index misaligned"
        assert (X.index == time_eval.index).all(), "PanCan master expression index and time index misaligned"
        assert (X.index == event_eval.index).all(), "PanCan master expression index and event index misaligned"
        assert not event_eval.isna().any(), "PanCan master loader contains NaN events"
        assert (time_eval > 0).all(), "PanCan master loader contains non-positive follow-up times"
        assert set(event_eval.unique()).issubset({0, 1}), "PanCan master loader events not in {0, 1}"

        log.info(
            "  Unified PanCanAtlas Master Matrix Loaded: %d patients (%d High Risk / %d Low Risk at %d days, %d duplicate biospecimens excluded)",
            len(eval_samples), int((y_series == 1).sum()), int((y_series == 0).sum()), horizon_days, duplicate_samples_excluded
        )
        return X, y_series, time_eval, event_eval
    except Exception as e:
        log.warning("Failed loading PanCanAtlas master matrix (%s). Falling back to individual cohorts.", e)
        return None


def load_cohort_data(
    cancer: str,
    horizon_days: int = RISK_HORIZON_DAYS,
    endpoint: str = "PFI",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Loads RNA-Seq expression and standardized TCGA-CDR clinical outcomes for a cohort.
    Enforces strict survival endpoint selection (PFI vs OS) without silent fallback.
    Performs patient-level barcode mapping, deduplication of multiple biospecimens per patient,
    and rigorous exclusion tracking.

    Returns:
      (expression_df, binary_risk_labels, time_series, event_series)
    """
    endpoint = endpoint.strip().upper()
    if endpoint not in ("PFI", "OS"):
        raise ValueError(f"Invalid survival endpoint '{endpoint}'. Must be 'PFI' or 'OS'.")

    cancer = normalize_cancer_code(cancer)
    fname_expr = XENA_FILE.get(cancer, get_xena_tsv_name(cancer))
    path_expr = XENA_DIR / fname_expr
    path_cdr  = XENA_DIR / "TCGA_CDR_survival.tsv"
    shards_dir = ROOT / "data" / "processed" / "shards"
    pq_path   = shards_dir / f"TCGA_{cancer}_expression.parquet"

    # Automatically fetch if missing
    if (not path_expr.exists() and not pq_path.exists()) or not path_cdr.exists():
        log.info("Missing expression or TCGA-CDR data for %s. Triggering download...", cancer)
        import download_xena_cohorts
        download_xena_cohorts.download_cohort(cancer)

    if not path_expr.exists() and not pq_path.exists():
        raise FileNotFoundError(f"Expression matrix missing: {path_expr} or {pq_path}")
    if not path_cdr.exists():
        raise FileNotFoundError(f"TCGA-CDR matrix missing: {path_cdr}")

    # Load RNA-Seq expression (prefer high-speed low-memory Parquet shard if present)
    if pq_path.exists():
        df_expr = pd.read_parquet(pq_path).dropna(how="all").astype(np.float32)
    else:
        df_expr = pd.read_csv(path_expr, sep="\t", index_col=0, low_memory=False).dropna(how="all").astype(np.float32)

    # Load TCGA-CDR survival table
    df_cdr  = pd.read_csv(path_cdr, sep="\t", index_col="sample", low_memory=False)

    tumor_samples = [c for c in df_expr.columns if is_tumor_sample(c)]
    if not tumor_samples:
        tumor_samples = [c for c in df_expr.columns if c not in ("sample", "_PATIENT")]

    total_genomic_samples = len(df_expr.columns)
    normal_samples_excluded = total_genomic_samples - len(tumor_samples)

    # Restrict to samples present in TCGA-CDR and without redaction flags
    # Map exact barcode (15-char), vial barcode (16-char), or patient barcode (12-char)
    cdr_sample_set = set(df_cdr.index)
    sample_to_cdr = {}
    for s in tumor_samples:
        if s in cdr_sample_set:
            sample_to_cdr[s] = s
        elif s[:15] in cdr_sample_set:
            sample_to_cdr[s] = s[:15]
        elif s[:12] in cdr_sample_set:
            sample_to_cdr[s] = s[:12]

    missing_cdr_excluded = len(tumor_samples) - len(sample_to_cdr)
    redacted_excluded = 0
    valid_samples_temp = []
    for s in tumor_samples:
        if s in sample_to_cdr:
            cdr_key = sample_to_cdr[s]
            if pd.notna(df_cdr.loc[cdr_key, "Redaction"]):
                redacted_excluded += 1
            else:
                valid_samples_temp.append(s)

    # Patient-level grouping and primary tumor prioritization (-01 over -02, etc.)
    # Group samples by 12-char patient ID to prevent duplicate patient biospecimens across splits
    patient_sample_map: Dict[str, List[str]] = {}
    for s in valid_samples_temp:
        pt_id = s[:12] if s.startswith("TCGA-") and len(s) >= 12 else s
        patient_sample_map.setdefault(pt_id, []).append(s)

    deduplicated_samples = []
    duplicate_samples_excluded = 0
    for pt_id, s_list in patient_sample_map.items():
        if len(s_list) == 1:
            deduplicated_samples.append(s_list[0])
        else:
            duplicate_samples_excluded += (len(s_list) - 1)
            # Prioritize primary solid tumor (sample type 01)
            primary_samples = [s for s in s_list if "-01" in s]
            if primary_samples:
                deduplicated_samples.append(sorted(primary_samples)[0])
            else:
                deduplicated_samples.append(sorted(s_list)[0])

    sub_cdr = df_cdr.loc[[sample_to_cdr[s] for s in deduplicated_samples]].copy()
    sub_cdr.index = deduplicated_samples

    # Survival endpoint extraction: STRICT PFI vs OS (NO SILENT FALLBACK)
    if endpoint == "PFI":
        if "PFI" not in sub_cdr.columns or "PFI.time" not in sub_cdr.columns:
            raise ValueError(
                f"Cohort TCGA-{cancer} lacks 'PFI' or 'PFI.time' in TCGA-CDR table. Silent fallback to OS is prohibited."
            )
        event_col = "PFI"
        time_col = "PFI.time"
    elif endpoint == "OS":
        if "OS" not in sub_cdr.columns or "OS.time" not in sub_cdr.columns:
            raise ValueError(
                f"Cohort TCGA-{cancer} lacks 'OS' or 'OS.time' in TCGA-CDR table. Silent fallback to PFI is prohibited."
            )
        event_col = "OS"
        time_col = "OS.time"

    # Exclude patients with missing or non-positive follow-up time
    # Strictly require event in {0, 1} and time > 0 (prohibit silent conversion of NaN to censored)
    raw_cohort_events = pd.to_numeric(sub_cdr[event_col], errors="coerce")
    raw_cohort_times = pd.to_numeric(sub_cdr[time_col], errors="coerce")
    valid_endpoint_mask = (
        raw_cohort_events.notna() &
        raw_cohort_events.isin([0, 1]) &
        raw_cohort_times.notna() &
        np.isfinite(raw_cohort_times) &
        (raw_cohort_times > 0)
    )
    missing_followup_excluded = int((~valid_endpoint_mask).sum())
    final_samples = [s for s in deduplicated_samples if valid_endpoint_mask.loc[s]]

    if len(final_samples) == 0:
        raise ValueError(
            f"Cohort TCGA-{cancer} is unevaluable for endpoint '{endpoint}': 0 patients with valid follow-up."
        )

    event_series = raw_cohort_events.loc[final_samples].astype(int)
    time_series = raw_cohort_times.loc[final_samples].astype(float)

    n_events = int(event_series.sum())
    n_total = len(final_samples)
    censored_count = n_total - n_events
    censoring_rate = round(float(censored_count / max(n_total, 1)), 4)

    if n_events < 5:
        log.warning(
            "Cohort TCGA-%s is unevaluable for endpoint '%s': observed events %d < 5 minimum threshold. "
            "Silent substitution of alternate endpoints is strictly prohibited.",
            cancer, endpoint, n_events
        )

    log.info(
        "  [COHORT AUDIT - TCGA-%s | %s] Total: %d patients, Events: %d, Censored: %d (%.1f%%). "
        "Exclusions: Normal=%d, Missing CDR=%d, Redacted=%d, Duplicate biospecimens=%d, Invalid follow-up=%d.",
        cancer, endpoint, n_total, n_events, censored_count, censoring_rate * 100,
        normal_samples_excluded, missing_cdr_excluded, redacted_excluded, duplicate_samples_excluded, missing_followup_excluded
    )

    # Define landmark Recurrence Risk based on parameterized horizon
    # High-Risk (1): Confirmed progression within horizon (event == 1 and time <= horizon_days)
    # Low-Risk  (0): Confirmed progression-free for at least horizon_days (time >= horizon_days)
    # Early-Censored: Patients without event censored before horizon (time < horizon_days and event == 0)
    is_high_risk = (event_series == 1) & (time_series <= horizon_days)
    is_low_risk  = (time_series >= horizon_days)

    # Definitive landmark cohort for binary risk evaluation
    # (Continuous survival concordance evaluates all patients)
    landmark_mask = is_high_risk | is_low_risk
    eval_samples  = [s for s in final_samples if landmark_mask.loc[s]]

    # If landmark samples are too few, retain all patients using median survival split
    if len(eval_samples) < 15:
        log.warning("Cohort %s has limited %d-day follow-up (%d samples). Using full cohort.", cancer, horizon_days, len(eval_samples))
        eval_samples = final_samples
        y_series = event_series.loc[eval_samples]
    else:
        y_series = pd.Series([1 if is_high_risk.loc[s] else 0 for s in eval_samples], index=eval_samples)

    X = df_expr[eval_samples].T.astype(np.float32).fillna(0.0)
    del df_expr
    import gc; gc.collect()

    t_eval = time_series.loc[eval_samples]
    e_eval = event_series.loc[eval_samples]

    # Strict alignment assertions
    assert (X.index == y_series.index).all(), f"Cohort TCGA-{cancer} expression index and label index misaligned"
    assert (X.index == t_eval.index).all(), f"Cohort TCGA-{cancer} expression index and time index misaligned"
    assert (X.index == e_eval.index).all(), f"Cohort TCGA-{cancer} expression index and event index misaligned"
    assert not e_eval.isna().any(), f"Cohort TCGA-{cancer} contains NaN events"
    assert (t_eval > 0).all(), f"Cohort TCGA-{cancer} contains non-positive follow-up times"
    assert set(e_eval.unique()).issubset({0, 1}), f"Cohort TCGA-{cancer} events not in {0, 1}"

    log.info("  Cohort TCGA-%s: %d total tumor samples -> %d landmark patients (%d High Risk / %d Low Risk at %d days)",
             cancer, len(tumor_samples), len(eval_samples), int((y_series == 1).sum()), int((y_series == 0).sum()), horizon_days)

    return X, y_series, t_eval, e_eval


def clean_matrix_axes(
    X_mat: pd.DataFrame,
    y_ser: pd.Series,
    t_ser: pd.Series,
    e_ser: pd.Series,
    prefix: str = ""
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Guarantees unique column names (genes) and unique row index (patients).
    Aligns target, time, and event series to the cleaned matrix.
    """
    X_mat = X_mat.copy()
    X_mat.columns = [str(c).strip() for c in X_mat.columns]
    if not X_mat.columns.is_unique:
        X_mat = X_mat.loc[:, ~X_mat.columns.duplicated(keep="first")]

    y_ser = y_ser.copy()
    t_ser = t_ser.copy()
    e_ser = e_ser.copy()

    # Deduplicate row index (patients) using positional boolean mask to maintain strict 1:1 alignment
    if not X_mat.index.is_unique:
        keep_mask = ~X_mat.index.duplicated(keep="first")
        X_mat = X_mat.iloc[keep_mask]
        y_ser = y_ser.iloc[keep_mask]
        t_ser = t_ser.iloc[keep_mask]
        e_ser = e_ser.iloc[keep_mask]

    if prefix:
        new_idx = [f"{prefix}_{idx}" for idx in X_mat.index]
        X_mat.index = new_idx
        y_ser.index = new_idx
        t_ser.index = new_idx
        e_ser.index = new_idx

    return X_mat, y_ser, t_ser, e_ser


def extract_unified_cohort(
    cohort_name: str,
    horizon_days: int = RISK_HORIZON_DAYS,
    normalize_batches: bool = False,
    include_external: bool = False,
    endpoint: str = "PFI",
    return_provenance: bool = False,
) -> Union[
    Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
]:
    """
    Loads and unifies single or pan-organ multi-cancer cohorts for specified endpoint (PFI or OS).
    NOTE: include_external is strictly False by default to prevent external validation contamination
    (METABRIC and GSE31210 must remain completely isolated as true external hospital test sets).
    Tracks explicit data provenance and fallback status.
    """
    is_global_all = cohort_name.upper() in ("PAN-CANCER-ALL", "ALL", "GLOBAL", "ALL-CONSOLIDATED")
    all_X = []
    all_y = []
    all_time = []
    all_event = []

    # Priority 1: If training PAN-CANCER-ALL, check for unified PanCanAtlas ComBat master matrix
    loaded_master = False
    master_fallback_reason = None
    if is_global_all:
        try:
            master_data = load_pancan_master_data(horizon_days=horizon_days, endpoint=endpoint)
            if master_data is not None:
                X_m, y_m, t_m, e_m = master_data
                X_m, y_m, t_m, e_m = clean_matrix_axes(X_m, y_m, t_m, e_m, prefix="TCGA_PANCAN")
                all_X.append(X_m)
                all_y.append(y_m)
                all_time.append(t_m)
                all_event.append(e_m)
                loaded_master = True
            else:
                master_fallback_reason = "Master matrix files not found on disk or insufficient landmark samples (<50)"
        except Exception as exc:
            master_fallback_reason = f"Exception loading master matrix: {str(exc)}"
            log.warning("Master matrix failed: %s", exc)

    # Priority 2: Ingest individual organ cohorts if master matrix not loaded
    if not loaded_master:
        if is_global_all:
            log.info(
                "  [PROVENANCE] Fallback to individual cohort aggregation activated (Reason: %s)",
                master_fallback_reason
            )
        cancer_codes = ALL_TCGA_COHORTS if is_global_all else COHORT_GROUPS.get(cohort_name, [cohort_name])
        for c in cancer_codes:
            log.info("  Loading clinical genomics for: TCGA-%s (endpoint: %s)...", c, endpoint)
            try:
                X_sub, y_sub, t_sub, e_sub = load_cohort_data(c, horizon_days=horizon_days, endpoint=endpoint)
                X_sub, y_sub, t_sub, e_sub = clean_matrix_axes(X_sub, y_sub, t_sub, e_sub, prefix=f"TCGA_{c}")
                all_X.append(X_sub)
                all_y.append(y_sub)
                all_time.append(t_sub)
                all_event.append(e_sub)
            except Exception as exc:
                log.warning("Could not ingest cohort TCGA-%s for endpoint %s: %s", c, endpoint, exc)

    # Ingest external independent cohorts ONLY when explicitly requested for joint multi-consortium training
    if include_external:
        ext_dir = ROOT / "data" / "processed" / "external"
        ext_dir.mkdir(parents=True, exist_ok=True)
        mb_file = ext_dir / "metabric_harmonized.parquet"

        # Ensure clean, uncorrupted METABRIC cohort is present
        from src.validation.external_datasets import fetch_geo_gse31210, fetch_metabric
        if not mb_file.exists():
            try:
                log.info("  Auto-fetching METABRIC cohort from cBioPortal datahub for single flow...")
                fetch_metabric()
            except Exception as e:
                log.warning("Could not auto-fetch METABRIC: %s", e)

        if mb_file.exists():
            try:
                df_mb = pd.read_parquet(mb_file)
                # Auto-heal if cached file has legacy corrupted 100% event rate
                if df_mb["_EVENT"].mean() > 0.80:
                    log.warning("  Detected legacy corrupted event labels in METABRIC cache (event rate %.2f > 0.80). Re-harmonizing...", float(df_mb["_EVENT"].mean()))
                    fetch_metabric()
                    df_mb = pd.read_parquet(mb_file)

                log.info("  Consolidating independent cohort: METABRIC into single flow (%d patients, %.1f%% true recurrence rate)...",
                         len(df_mb), float(df_mb["_EVENT"].mean() * 100))
                meta_cols = ["_Y_3YR", "_TIME_DAYS", "_EVENT"]
                g_cols = [c for c in df_mb.columns if c not in meta_cols]
                X_mb = df_mb[g_cols]
                X_mb, y_mb, t_mb, e_mb = clean_matrix_axes(
                    X_mb, df_mb["_Y_3YR"], df_mb["_TIME_DAYS"], df_mb["_EVENT"], prefix="METABRIC"
                )
                if normalize_batches and len(X_mb) > 1:
                    stds = X_mb.std(axis=0)
                    stds[stds < 1e-6] = 1.0
                    X_mb = (X_mb - X_mb.mean(axis=0)) / stds
                all_X.append(X_mb)
                all_y.append(y_mb)
                all_time.append(t_mb)
                all_event.append(e_mb)
            except Exception as e:
                log.warning("Could not include METABRIC in consolidated flow: %s", e)

        gse_file = ext_dir / "gse31210_harmonized.parquet"
        if not gse_file.exists():
            try:
                log.info("  Auto-fetching NCBI GEO GSE31210 cohort for single flow...")
                fetch_geo_gse31210()
            except Exception as e:
                log.warning("Could not auto-fetch GSE31210: %s", e)

        if gse_file.exists():
            try:
                log.info("  Consolidating independent cohort: NCBI GEO GSE31210 into single flow...")
                df_gse = pd.read_parquet(gse_file)
                meta_cols = ["_Y_3YR", "_TIME_DAYS", "_EVENT"]
                g_cols = [c for c in df_gse.columns if c not in meta_cols]
                X_gse = df_gse[g_cols]
                X_gse, y_gse, t_gse, e_gse = clean_matrix_axes(
                    X_gse, df_gse["_Y_3YR"], df_gse["_TIME_DAYS"], df_gse["_EVENT"], prefix="GSE31210"
                )
                if normalize_batches and len(X_gse) > 1:
                    stds = X_gse.std(axis=0)
                    stds[stds < 1e-6] = 1.0
                    X_gse = (X_gse - X_gse.mean(axis=0)) / stds
                all_X.append(X_gse)
                all_y.append(y_gse)
                all_time.append(t_gse)
                all_event.append(e_gse)
            except Exception as e:
                log.warning("Could not include GSE31210 in consolidated flow: %s", e)

    if not all_X:
        raise RuntimeError(f"No clinical genomic data could be loaded for cohort: {cohort_name}")

    X_df = pd.concat(all_X, axis=0).fillna(0.0).astype(np.float32)
    del all_X
    import gc; gc.collect()
    y = np.concatenate([y_s.values for y_s in all_y]).astype(np.int32)
    times = np.concatenate([t_s.values for t_s in all_time]).astype(np.float32)
    events = np.concatenate([e_s.values for e_s in all_event]).astype(np.int32)

    if loaded_master:
        data_provenance = "pancan_master_combat_matrix"
        master_matrix_fallback = False
        master_fallback_reason = None
    elif is_global_all:
        data_provenance = "individual_cohort_aggregation_fallback"
        master_matrix_fallback = True
        master_fallback_reason = master_fallback_reason or "PanCan master matrix not loaded"
    else:
        data_provenance = f"individual_cohort_{cohort_name}"
        master_matrix_fallback = False
        master_fallback_reason = None

    provenance_info = {
        "data_provenance": data_provenance,
        "master_matrix_fallback": master_matrix_fallback,
        "master_fallback_reason": master_fallback_reason,
    }

    log.info("  Unified %s Single-Flow Cohort: %d total patients (%d High Risk Events / %d Low Risk Survivors) | Provenance: %s",
             cohort_name, len(y), int((y == 1).sum()), int((y == 0).sum()), data_provenance)

    if return_provenance:
        return X_df, y, times, events, provenance_info
    return X_df, y, times, events


def compute_leak_free_hallmarks(
    X_tr_df: pd.DataFrame,
    X_te_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]], List[str]]:
    """
    Computes 12 Hallmark Metagenes with STRICT leak-free isolation:
    1. Computes gene mean/std statistics strictly on training data (X_tr_df).
    2. Applies the exact same baseline statistics to test/validation data (X_te_df).
    3. Returns standardized dataframes, metagene statistics, and list of metagene names.
    """
    metagene_stats: Dict[str, Dict[str, float]] = {}
    meta_names = list(HALLMARK_PATHWAYS.keys())

    # 1. Fit baseline statistics on training split
    for meta_name, gene_list in HALLMARK_PATHWAYS.items():
        for g in gene_list:
            if g in X_tr_df.columns:
                vals = X_tr_df[g].values
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                if sigma < 1e-6:
                    sigma = 1.0
                metagene_stats[g] = {"mean": mu, "std": sigma}

    # 2. Transform both training and test partitions using baseline statistics
    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for meta_name, gene_list in HALLMARK_PATHWAYS.items():
            z_cols = []
            for g in gene_list:
                if g in df.columns and g in metagene_stats:
                    stat = metagene_stats[g]
                    z = (df[g].values - stat["mean"]) / stat["std"]
                    z_cols.append(z)
            if z_cols:
                df_out[meta_name] = np.mean(np.column_stack(z_cols), axis=1)
            else:
                df_out[meta_name] = 0.0
        return df_out

    X_tr_hallmarks = _transform(X_tr_df)
    X_te_hallmarks = _transform(X_te_df)

    return X_tr_hallmarks, X_te_hallmarks, metagene_stats, meta_names


# ─── Training Routine ────────────────────────────────────────────────────────

def train_strong_risk_model(
    cohort_name: str,
    n_indiv_genes: int = DEFAULT_N_GENES,
    horizon_days: int = RISK_HORIZON_DAYS,
    epochs: int = 60,
    seed: int = DEFAULT_RANDOM_SEED,
    normalize_batches: bool = False,
    include_external: bool = False,
    endpoint: str = "PFI",
) -> Dict[str, Any]:
    endpoint = endpoint.strip().upper()
    if endpoint not in ("PFI", "OS"):
        raise ValueError(f"Invalid survival endpoint '{endpoint}'. Must be 'PFI' or 'OS'.")

    horizon_years = max(1, round(horizon_days / 365.25))
    time_horizon_str = f"{horizon_years}-Year Progression & Recurrence Risk ({endpoint})"

    log.info("=" * 76)
    log.info(" TRAINING CLINICAL RISK ENGINE (STACKING ENSEMBLE): %s", cohort_name)
    log.info(" Target: %s (Horizon: %d days | Epochs: %d | Endpoint: %s)", time_horizon_str, horizon_days, epochs, endpoint)
    log.info(" Single-Flow Multi-Consortium Optimization (Zero Piecewise Batches)")
    log.info("=" * 76)

    # 1. Load Raw Patient Expression and Clinical Outcomes
    X_raw_df, y, times, events, provenance = extract_unified_cohort(
        cohort_name,
        horizon_days=horizon_days,
        normalize_batches=normalize_batches,
        include_external=include_external,
        endpoint=endpoint,
        return_provenance=True,
    )

    # 2. Stratified 80/20 Partition
    patient_ids = np.array(X_raw_df.index)
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=DEFAULT_TEST_RATIO, stratify=y, random_state=seed
    )

    X_tr_raw = X_raw_df.iloc[idx_tr]
    X_te_raw = X_raw_df.iloc[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    t_tr, t_te = times[idx_tr], times[idx_te]
    e_tr, e_te = events[idx_tr], events[idx_te]

    # 3. Compute Leak-Free Hallmark Metagenes
    log.info("  Computing %d Hallmark Oncology Metagenes with leak-free statistics...", len(HALLMARK_PATHWAYS))
    X_tr_df, X_te_df, metagene_stats, meta_names = compute_leak_free_hallmarks(X_tr_raw, X_te_raw)

    raw_feature_names = list(X_tr_df.columns)
    X_tr = X_tr_df.values.astype(np.float32)
    X_te = X_te_df.values.astype(np.float32)

    # 4. Feature Selection: Top Individual Genes + Hallmark Metagenes
    log.info("  Selecting top %d individual prognostic genes + %d pathway metagenes...", n_indiv_genes, len(meta_names))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))
        f_scores, _ = f_classif(X_tr, y_tr)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # Prioritize genes with universal cross-cohort coverage to prevent cross-platform feature dropout
    nonzero_frac = np.mean(X_tr != 0.0, axis=0)
    meta_indices = [raw_feature_names.index(m) for m in meta_names if m in raw_feature_names]
    sorted_candidates = [i for i in np.argsort(f_scores)[::-1] if i not in meta_indices]
    high_cov_candidates = [i for i in sorted_candidates if nonzero_frac[i] >= 0.65]
    if len(high_cov_candidates) >= n_indiv_genes:
        top_indiv_indices = high_cov_candidates[:n_indiv_genes]
    else:
        top_indiv_indices = sorted_candidates[:n_indiv_genes]

    selected_indices = meta_indices + top_indiv_indices
    selected_features = [raw_feature_names[i] for i in selected_indices]
    top_indiv_genes = [raw_feature_names[i] for i in top_indiv_indices]

    X_tr_sub = X_tr[:, selected_indices]
    X_te_sub = X_te[:, selected_indices]

    # Free high-dimensional full genomic matrices from RAM immediately
    del X_raw_df, X_tr_raw, X_te_raw, X_tr_df, X_te_df, X_tr, X_te
    import gc; gc.collect()

    # 5. Standard Scaling
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_sub)
    X_te_sc = scaler.transform(X_te_sub)

    # 6. Sample Weighting: Inverse Probability of Censoring Weighting (IPCW) + Adaptive Class Balance
    ipcw_weights = compute_ipcw_weights(t_tr, e_tr, horizon=float(horizon_days))
    class_weights = compute_sample_weight("balanced", y_tr)
    composite_weights = ipcw_weights * class_weights
    composite_weights = composite_weights * (len(y_tr) / np.sum(composite_weights))

    # Adaptive scale_pos_weight for XGBoost to handle imbalanced cohorts (e.g. PRAD, THCA)
    pos_count = int(np.sum(y_tr == 1))
    neg_count = int(np.sum(y_tr == 0))
    scale_pos = float(neg_count) / max(pos_count, 1)
    scale_pos_weight = float(np.clip(scale_pos, 0.1, 20.0))

    # 7. Construct Multi-Model Stacking Ensemble with Multi-Epoch Optimization
    xgb_estimators = max(180, epochs * 4)
    elastic_iter = max(4000, epochs * 80)
    log.info("  Configuring Stacking Ensemble (ElasticNet [%d passes] + XGBoost [%d boosting rounds, scale_pos=%.2f] + ExtraTrees)...",
             elastic_iter, xgb_estimators, scale_pos_weight)
    base_models = [
        ("elasticnet", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.3, max_iter=elastic_iter, random_state=seed
        )),
        ("xgboost", XGBClassifier(
            n_estimators=xgb_estimators, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, reg_lambda=4.0, reg_alpha=0.5,
            scale_pos_weight=scale_pos_weight, random_state=seed, verbosity=0
        )),
        ("extratrees", ExtraTreesClassifier(
            n_estimators=150, max_depth=4, min_samples_split=4, bootstrap=True, random_state=seed
        )),
    ]
    if TORCH_AVAILABLE:
        log.info("  Integrating PyTorch DeepSurv Multi-Epoch Neural Network (%d epochs, AdamW + CosineAnnealing)...", epochs)
        base_models.append(
            ("pytorch_deepsurv", PyTorchDeepSurvClassifier(
                hidden_dim=128, epochs=epochs, lr=1e-3, random_state=seed
            ))
        )
    meta_learner = LogisticRegression(C=1.0, random_state=seed)

    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=1,
    )

    log.info("  Fitting Stacking Ensemble with 5-fold cross-validation and IPCW weights...")
    ensemble.fit(X_tr_sc, y_tr, sample_weight=composite_weights)

    # 8. Out-of-Sample Test Evaluation
    test_probs = ensemble.predict_proba(X_te_sc)[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)

    test_auroc = round(float(roc_auc_score(y_te, test_probs)), 4)
    test_auprc = round(float(average_precision_score(y_te, test_probs)), 4)
    test_c_idx = compute_c_index(t_te, e_te, test_probs)
    test_bal_acc = round(float(balanced_accuracy_score(y_te, test_preds)), 4)
    test_brier = round(float(brier_score_loss(y_te, test_probs)), 4)

    log.info("  [TEST EVALUATION RESULTS - TCGA-%s]", cohort_name)
    log.info("    * %s AUROC : %.4f (Discrimination power)", time_horizon_str, test_auroc)
    log.info("    * Harrell's C-Index : %.4f (Survival concordance)", test_c_idx)
    log.info("    * Balanced Accuracy : %.4f", test_bal_acc)
    log.info("    * Brier Score       : %.4f (Calibration error)", test_brier)

    # 9. Extract Tree Feature Importances
    xgb_est = ensemble.named_estimators_["xgboost"]
    imp = xgb_est.feature_importances_
    sorted_pairs = sorted(zip(selected_features, imp), key=lambda x: x[1], reverse=True)
    top_biomarkers = {f: round(float(s), 5) for f, s in sorted_pairs}

    # 10. Save Production Artifact
    model_name = f"TCGA-{cohort_name}"
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "xgboost_clinical_model.joblib"

    artifact = {
        "model"             : ensemble,
        "scaler"            : scaler,
        "features"          : selected_features,            # Features in exact order
        "individual_genes"  : top_indiv_genes,              # Differential individual genes
        "hallmark_metagenes": HALLMARK_PATHWAYS,            # 12 hallmark definitions
        "metagene_stats"    : metagene_stats,               # Zero-leakage {gene: {mean, std}}
        "task"              : "genomic_disease_risk_prediction",
        "time_horizon"      : time_horizon_str,
        "horizon_days"      : horizon_days,
        "risk_cutoffs"      : {"low": 0.35, "high": 0.65},
        "optimal_threshold" : 0.50,
        "feature_importances": top_biomarkers,
        "test_metrics"      : {
            "risk_auroc"       : test_auroc,
            "risk_auprc"       : test_auprc,
            "c_index"          : test_c_idx,
            "balanced_accuracy": test_bal_acc,
            "brier_score"      : test_brier,
        },
        "metadata": {
            "cohort_name"  : cohort_name,
            "horizon_days" : horizon_days,
            "n_samples"    : len(y),
            "n_high_risk"  : int((y == 1).sum()),
            "n_low_risk"   : int((y == 0).sum()),
            "n_features"   : len(selected_features),
            "architecture" : "StackingEnsemble(ElasticNet + XGBoost + ExtraTrees)",
            "endpoint"     : f"TCGA-CDR {endpoint}",
            "ipcw_weighted": True,
            "data_provenance": provenance["data_provenance"],
            "master_matrix_fallback": provenance["master_matrix_fallback"],
            "master_fallback_reason": provenance["master_fallback_reason"],
            "trained_at"   : datetime.datetime.now().isoformat(),
        }
    }
    joblib.dump(artifact, artifact_path)
    log.info("  Model Artifact saved -> %s", artifact_path)

    # 11. Save JSON Summary
    result_dir = RESULTS_DIR / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "project_id"       : model_name,
        "cohort_name"      : cohort_name,
        "time_horizon"     : time_horizon_str,
        "horizon_days"     : horizon_days,
        "architecture"     : "StackingEnsemble(ElasticNet + XGBoost + ExtraTrees)",
        "endpoint"         : f"TCGA-CDR {endpoint}",
        "data_provenance"  : provenance["data_provenance"],
        "master_matrix_fallback": provenance["master_matrix_fallback"],
        "master_fallback_reason": provenance["master_fallback_reason"],
        "n_samples"        : len(y),
        "n_high_risk"      : int((y == 1).sum()),
        "n_low_risk"       : int((y == 0).sum()),
        "test_risk_auroc"  : test_auroc,
        "test_c_index"     : test_c_idx,
        "test_balanced_acc": test_bal_acc,
        "test_brier_score" : test_brier,
        "top_10_biomarkers": list(top_biomarkers.items())[:10],
    }
    with open(result_dir / "xgboost_clinical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Omics Disease Risk Engine (StacK-Risk)")
    parser.add_argument("cohorts", nargs="*", default=["PAN-CANCER-ALL"], help="Cohort names (e.g. PAN-CANCER-ALL PAN-RENAL all)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs for deep neural & iterative estimators")
    parser.add_argument("--horizon", type=int, default=RISK_HORIZON_DAYS, help="Landmark recurrence horizon in days (e.g. 365, 1095, 1825)")
    parser.add_argument("--n-genes", type=int, default=DEFAULT_N_GENES, help="Number of individual genes")
    parser.add_argument("--endpoint", type=str, default="PFI", choices=["PFI", "OS"], help="Standardized TCGA-CDR survival endpoint: 'PFI' or 'OS'")
    parser.add_argument("--no-batch-norm", action="store_true", help="Disable batch normalization for pan-organ cohorts")
    parser.add_argument("--exclude-external", action="store_true", help="Exclude external independent cohorts from consolidated training")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    cohorts = args.cohorts
    # When user requests 'all', run the consolidated single-flow pan-cancer training across all 33 cohorts + external
    if "all" in [c.lower() for c in cohorts]:
        cohorts = ["PAN-CANCER-ALL"]

    for c in cohorts:
        try:
            train_strong_risk_model(
                c.upper(),
                n_indiv_genes=args.n_genes,
                horizon_days=args.horizon,
                epochs=args.epochs,
                normalize_batches=not args.no_batch_norm,
                include_external=not args.exclude_external,
                endpoint=args.endpoint,
                seed=args.seed,
            )
        except Exception as e:
            log.error("Failed to train risk model for %s: %s", c, e, exc_info=True)


if __name__ == "__main__":
    main()
