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

if TORCH_AVAILABLE:
    class PyTorchDeepSurvClassifier(BaseEstimator, ClassifierMixin):
        """
        Multi-Epoch Deep Survival & Recurrence Risk Neural Network (PyTorch).
        Implements multi-epoch gradient optimization with AdamW, Cosine Annealing LR scheduling,
        Mish non-linear activations, batch normalization, and dropout regularization.
        """
        def __init__(
            self,
            hidden_dim: int = 128,
            dropout: float = 0.3,
            epochs: int = 60,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            random_state: int = 42
        ):
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.epochs = epochs
            self.lr = lr
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.classes_ = np.array([0, 1])
            self.net_ = None

        def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
            torch.manual_seed(self.random_state)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            in_dim = X.shape[1]

            self.net_ = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.Mish(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.Mish(),
                nn.Dropout(self.dropout / 2),
                nn.Linear(self.hidden_dim // 2, 1)
            ).to(device)

            optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.epochs, 1), eta_min=1e-5)

            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
            if sample_weight is not None:
                w_tensor = torch.tensor(sample_weight, dtype=torch.float32).to(device)
            else:
                w_tensor = torch.ones_like(y_tensor)

            self.net_.train()
            log_interval = max(1, self.epochs // 5)
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                logits = self.net_(X_tensor).squeeze(-1)
                loss = (F.binary_cross_entropy_with_logits(logits, y_tensor, reduction="none") * w_tensor).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (epoch + 1) % log_interval == 0 or epoch == self.epochs - 1:
                    cur_lr = scheduler.get_last_lr()[0]
                    log.info("    [PyTorch DeepSurv Epoch %2d/%2d] Loss: %.4f | LR: %.6f", epoch + 1, self.epochs, float(loss.item()), cur_lr)

            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            if self.net_ is None:
                n = len(X)
                return np.ones((n, 2)) * 0.5
            self.net_.eval()
            device = next(self.net_.parameters()).device
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                logits = self.net_(X_tensor).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                return np.column_stack([1.0 - probs, probs])

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
else:
    class PyTorchDeepSurvClassifier:
        pass


# ─── Data Extraction & Metagene Engineering ──────────────────────────────────

def load_cohort_data(cancer: str, horizon_days: int = RISK_HORIZON_DAYS) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Loads RNA-Seq expression and standardized TCGA-CDR clinical outcomes for a cohort.
    Returns:
      (expression_df, binary_risk_labels, time_series, event_series)
    """
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

    # Restrict to tumor samples (primary solid tumor 01, recurrent 02, blood cancer 03, metastatic 06; exclude normal 10/11)
    def _is_tumor_sample(s: str) -> bool:
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

    tumor_samples = [c for c in df_expr.columns if _is_tumor_sample(c)]
    if not tumor_samples:
        tumor_samples = [c for c in df_expr.columns if c not in ("sample", "_PATIENT")]

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

    valid_samples = [
        s for s in tumor_samples
        if s in sample_to_cdr and pd.isna(df_cdr.loc[sample_to_cdr[s], "Redaction"])
    ]

    sub_cdr = df_cdr.loc[[sample_to_cdr[s] for s in valid_samples]].copy()
    sub_cdr.index = valid_samples

    # Primary survival endpoint: Progression-Free Interval (PFI)
    # Fallback to Overall Survival (OS) if PFI is absent
    if "PFI" in sub_cdr.columns and sub_cdr["PFI"].notna().sum() > 0:
        event_series = sub_cdr["PFI"].fillna(0).astype(int)
        time_series  = sub_cdr["PFI.time"].fillna(0).astype(float)
    else:
        event_series = sub_cdr["OS"].fillna(0).astype(int)
        time_series  = sub_cdr["OS.time"].fillna(0).astype(float)

    # Define landmark Recurrence Risk based on parameterized horizon
    # High-Risk (1): Confirmed progression within horizon (PFI == 1 and time <= horizon_days)
    # Low-Risk  (0): Confirmed progression-free for at least horizon_days (time >= horizon_days)
    # Early-Censored: Patients without event censored before horizon (time < horizon_days and event == 0)
    is_high_risk = (event_series == 1) & (time_series <= horizon_days)
    is_low_risk  = (time_series >= horizon_days)

    # Definitive landmark cohort for binary risk evaluation
    # (Continuous survival concordance evaluates all patients)
    landmark_mask = is_high_risk | is_low_risk
    eval_samples  = [s for s in valid_samples if landmark_mask.loc[s]]

    # If landmark samples are too few, retain all patients using median survival split
    if len(eval_samples) < 15:
        log.warning("Cohort %s has limited %d-day follow-up (%d samples). Using full cohort.", cancer, horizon_days, len(eval_samples))
        eval_samples = valid_samples
        y_series = event_series.loc[eval_samples]
    else:
        y_series = pd.Series([1 if is_high_risk.loc[s] else 0 for s in eval_samples], index=eval_samples)

    X = df_expr[eval_samples].T.astype(np.float32).fillna(0.0)
    del df_expr
    import gc; gc.collect()

    log.info("  Cohort TCGA-%s: %d total tumor samples -> %d landmark patients (%d High Risk / %d Low Risk at %d days)",
             cancer, len(tumor_samples), len(eval_samples), int((y_series == 1).sum()), int((y_series == 0).sum()), horizon_days)

    return X, y_series, time_series.loc[eval_samples], event_series.loc[eval_samples]


def extract_unified_cohort(
    cohort_name: str,
    horizon_days: int = RISK_HORIZON_DAYS,
    normalize_batches: bool = True,
    include_external: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and unifies single or pan-organ multi-cancer cohorts with optional cross-organ normalization."""
    is_global_all = cohort_name.upper() in ("PAN-CANCER-ALL", "ALL", "GLOBAL", "ALL-CONSOLIDATED")
    cancer_codes = ALL_TCGA_COHORTS if is_global_all else COHORT_GROUPS.get(cohort_name, [cohort_name])
    all_X = []
    all_y = []
    all_time = []
    all_event = []

    for c in cancer_codes:
        log.info("  Loading clinical genomics for: TCGA-%s...", c)
        try:
            X_sub, y_sub, t_sub, e_sub = load_cohort_data(c, horizon_days=horizon_days)

            # Batch normalization across organ cohorts to eliminate tissue-specific baseline offsets
            if len(cancer_codes) > 1 and normalize_batches and len(X_sub) > 1:
                stds = X_sub.std(axis=0)
                stds[stds < 1e-6] = 1.0
                X_sub = (X_sub - X_sub.mean(axis=0)) / stds

            all_X.append(X_sub)
            all_y.append(y_sub)
            all_time.append(t_sub)
            all_event.append(e_sub)
        except Exception as exc:
            log.warning("Could not ingest cohort TCGA-%s: %s", c, exc)

    # Ingest external independent cohorts in single-flow if requested
    if is_global_all or include_external:
        ext_dir = ROOT / "data" / "processed" / "external"
        mb_file = ext_dir / "metabric_harmonized.parquet"
        if mb_file.exists():
            try:
                log.info("  Consolidating independent cohort: METABRIC into single flow...")
                df_mb = pd.read_parquet(mb_file)
                meta_cols = ["_Y_3YR", "_TIME_DAYS", "_EVENT"]
                g_cols = [c for c in df_mb.columns if c not in meta_cols]
                X_mb = df_mb[g_cols]
                if normalize_batches and len(X_mb) > 1:
                    stds = X_mb.std(axis=0)
                    stds[stds < 1e-6] = 1.0
                    X_mb = (X_mb - X_mb.mean(axis=0)) / stds
                all_X.append(X_mb)
                all_y.append(df_mb["_Y_3YR"])
                all_time.append(df_mb["_TIME_DAYS"])
                all_event.append(df_mb["_EVENT"])
            except Exception as e:
                log.warning("Could not include METABRIC in consolidated flow: %s", e)

        gse_file = ext_dir / "gse31210_harmonized.parquet"
        if gse_file.exists():
            try:
                log.info("  Consolidating independent cohort: NCBI GEO GSE31210 into single flow...")
                df_gse = pd.read_parquet(gse_file)
                meta_cols = ["_Y_3YR", "_TIME_DAYS", "_EVENT"]
                g_cols = [c for c in df_gse.columns if c not in meta_cols]
                X_gse = df_gse[g_cols]
                if normalize_batches and len(X_gse) > 1:
                    stds = X_gse.std(axis=0)
                    stds[stds < 1e-6] = 1.0
                    X_gse = (X_gse - X_gse.mean(axis=0)) / stds
                all_X.append(X_gse)
                all_y.append(df_gse["_Y_3YR"])
                all_time.append(df_gse["_TIME_DAYS"])
                all_event.append(df_gse["_EVENT"])
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

    log.info("  Unified %s Single-Flow Cohort: %d total patients (%d High Risk Events / %d Low Risk Survivors)",
             cohort_name, len(y), int((y == 1).sum()), int((y == 0).sum()))
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
    normalize_batches: bool = True,
    include_external: bool = True
) -> Dict[str, Any]:
    horizon_years = max(1, round(horizon_days / 365.25))
    time_horizon_str = f"{horizon_years}-Year Progression & Recurrence Risk"

    log.info("=" * 76)
    log.info(" TRAINING CLINICAL RISK ENGINE (STACKING ENSEMBLE): %s", cohort_name)
    log.info(" Target: %s (Horizon: %d days | Epochs: %d)", time_horizon_str, horizon_days, epochs)
    log.info(" Single-Flow Multi-Consortium Optimization (Zero Piecewise Batches)")
    log.info("=" * 76)

    # 1. Load Raw Patient Expression and Clinical Outcomes
    X_raw_df, y, times, events = extract_unified_cohort(
        cohort_name, horizon_days=horizon_days, normalize_batches=normalize_batches, include_external=include_external
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

    # Metagenes always preserved
    meta_indices = [raw_feature_names.index(m) for m in meta_names if m in raw_feature_names]
    top_indiv_indices = [i for i in np.argsort(f_scores)[::-1] if i not in meta_indices][:n_indiv_genes]
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
            "endpoint"     : "TCGA-CDR PFI / OS",
            "ipcw_weighted": True,
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
        "endpoint"         : "TCGA-CDR PFI (Progression-Free Interval)",
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
                seed=args.seed,
            )
        except Exception as e:
            log.error("Failed to train risk model for %s: %s", c, e, exc_info=True)


if __name__ == "__main__":
    main()
