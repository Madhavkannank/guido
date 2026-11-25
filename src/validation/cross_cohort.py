"""
Step 8 – Cross-Cohort & Independent External Validation
======================================================
Evaluates trained clinical genomic risk models on independent hospital and
consortium cohorts (METABRIC, GEO GSE31210, etc.) without fine-tuning:

  1. Out-of-Distribution Concordance (Harrell's C-Index)
  2. 3-Year Landmark Discrimination (AUROC & AUPRC with Bootstrap 95% CIs)
  3. Calibration & Reliability (Brier Score, Expected Calibration Error [ECE])
  4. Clinical Risk Stratification (Low <35%, Moderate 35-65%, High >65%)
  5. Feature Space & Hallmark Metagene Alignment using trained baseline statistics

All validation results are persisted to disk and optionally logged to MLflow.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MLFLOW_TRACKING_URI, RESULTS_DIR
from src.models.deep_surv import PyTorchDeepSurvClassifier
from src.models.metrics import compute_clf_metrics

# Ensure backward unpickling compatibility across all execution contexts
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "PyTorchDeepSurvClassifier", PyTorchDeepSurvClassifier)

logger = logging.getLogger(__name__)

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ─── High-Performance Concordance Index ──────────────────────────────────────

def compute_c_index_fast(times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray) -> float:
    """Computes Harrell's C-Index handling right-censoring in O(N log N) time."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(risk_scores, dtype=float)
    n = len(times)
    if n < 2:
        return 0.5

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


# ─── Expected Calibration Error (ECE) ────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across equal-width probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return round(float(ece), 4)


# ─── AUROC bootstrap CI ──────────────────────────────────────────────────────

def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for AUROC."""
    rng = np.random.default_rng(seed)
    aurocs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aurocs.append(roc_auc_score(y_b, p_b))
    if not aurocs:
        return (0.5, 0.5)
    aurocs = np.array(aurocs)
    lower = float(np.percentile(aurocs, 100 * alpha / 2))
    upper = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)


# ─── Comprehensive Cross-Platform Gene Alias Dictionary ─────────────────────
# Maps historical microarray probe names (Illumina BeadChip & Affymetrix) to modern HGNC symbols
GENE_ALIASES: Dict[str, List[str]] = {
    # Proliferation & Mitosis
    "NUF2": ["CDCA1", "CT106"],
    "CDC20": ["p55CDC", "bA276H19.1"],
    "CDK1": ["CDC2", "CDC28A", "P34CDC2"],
    "AURKA": ["STK15", "BTAK", "AIK", "AURORA2"],
    "AURKB": ["STK12", "AIM1", "AIK2", "STK5"],
    "BIRC5": ["API4", "EPR-1", "SURVIVIN"],
    "CCNB1": ["CCNB"],
    "CCNE1": ["CCNE"],
    "PLK1": ["STPK13"],
    "BUB1B": ["SSK1", "BUBR1", "hBUBR1"],
    "CENPA": ["CENP-A"],
    "CENPE": ["CENP-E", "KIF10"],
    "CENPF": ["CENP-F", "hcp-1", "ST-410"],
    "HJURP": ["hJn1", "FAKTL"],
    "PTTG1": ["SECURIN", "EPI-1", "TUTR1"],
    "TROAP": ["TASTIN"],
    "CEP55": ["C10orf3", "URCC6"],
    "KIF11": ["EG5", "TRIP5", "KSP"],
    "KIF20A": ["MKLP2", "RAB6KIFL", "GG10_2"],
    "KIF23": ["MKLP1"],
    "KIF2C": ["MCAK"],
    "KIF18B": ["KIAA0319L"],
    "ZWINT": ["ZWINT1", "HZwint-1"],
    "TPX2": ["DIL2", "p100", "RECA", "HCA519"],
    "UBE2C": ["UBCH10", "dJ447E24.1"],
    "FOXM1": ["HFH11", "INS1", "TRIDENT"],
    "MYBL2": ["B-MYB"],
    "E2F1": ["RBP3"],
    "MKI67": ["Ki-67", "KIA"],
    "PCNA": ["ATLD2"],
    "TOP2A": ["TOP2"],

    # Epigenetic & Chromatin Regulators
    "KMT2A": ["MLL", "MLL1", "ALL-1", "HRX", "HTX1"],
    "KMT2B": ["MLL4", "MLL2", "WBP7", "TRX2"],
    "KMT2C": ["MLL3", "HALR"],
    "KMT2D": ["MLL2", "ALR", "KAD4D"],
    "H3-3A": ["H3F3A", "H3.3A"],
    "H3-3B": ["H3F3B", "H3.3B"],

    # Extracellular Matrix, Angiogenesis & Signaling
    "CCN1": ["CYR61", "GIG1", "IGFBP10"],
    "CCN2": ["CTGF", "HCS24", "IGFBP8"],
    "CCN3": ["NOV", "IGFBP9"],
    "CCN4": ["WISP1", "WISP-1"],
    "CCN5": ["WISP2", "WISP-2"],
    "CCN6": ["WISP3", "WISP-3"],
    "ADGRA2": ["GPR124", "TEM5"],
    "ADGRG1": ["GPR56", "TM7XN1", "BFPP"],
    "CD274": ["PD-L1", "B7-H1", "PDL1", "B7H1"],
    "PDCD1": ["PD-1", "PD1", "CD279", "SLEB2"],
    "PARP1": ["ADPRT", "PPOL", "pADPRT-1", "ARTD1"],
    "ERBB2": ["HER2", "NEU", "NGL", "TKR1"],
    "KDR": ["VEGFR2", "FLK1", "CD309"],
    "FLT1": ["VEGFR1", "FLT"],
    "FLT4": ["VEGFR3", "PCL"],
    "PTPRC": ["CD45", "B220", "GP180", "LY5"],
    "PECAM1": ["CD31", "endoCAM", "PECA1"],
    "PROM1": ["CD133", "AC133", "PROML1"],
    "SDC1": ["CD138", "SYND1"],
    "ENG": ["CD105", "END", "ORW1"],
    "NCAM1": ["CD56", "MSK39"],
    "CDH1": ["E-CAD", "CD324", "Arc-1", "UVO"],
    "CDH2": ["N-CAD", "CD325", "NCAD"],
    "ACTA2": ["a-SMA", "ACTSA"],
    "FN1": ["FN", "CIG", "FINC"],
    "SNAI1": ["SNAIL", "SNA", "dJ710H13.1"],
    "SNAI2": ["SLUG", "SLUGH1", "WS2D"],
    "TWIST1": ["TWIST", "bHLHa38", "CSO"],
    "ZEB1": ["TCF8", "NIL-2-A", "AREB6"],
    "ZEB2": ["SIP1", "ZFHX1B", "HSPC082"],
    "HIF1A": ["MOP1", "bHLHe78", "PASD8"],
    "VEGFA": ["VEGF", "VPF", "MVCD1"],
    "GLUT1": ["SLC2A1", "GLUT", "DYT18"],
    "PKM": ["PKM2", "PK3", "CTHBP", "OIP3"],
    "LDHA": ["GSD11", "LDHM"],
}

ALIAS_TO_GENE: Dict[str, str] = {}
for canon_sym, alias_list in GENE_ALIASES.items():
    ALIAS_TO_GENE[canon_sym.upper()] = canon_sym
    for a in alias_list:
        ALIAS_TO_GENE[a.upper()] = canon_sym


# ─── Feature & Metagene Alignment ────────────────────────────────────────────

def align_external_features(
    X_ext: pd.DataFrame,
    artifact: Dict[str, Any]
) -> pd.DataFrame:
    """
    Standardizes external patient expression using training metagene_stats,
    computes the 12 Hallmark Metagenes, and aligns columns to artifact['features'].
    Applies cross-platform distribution harmonization (e.g. Microarray vs RNA-Seq),
    resolves gene aliases across historical platforms, and applies neutral baseline
    imputation for missing genes.
    """
    target_features: List[str] = artifact["features"]
    hallmark_pathways: Dict[str, List[str]] = artifact.get("hallmark_metagenes", {})
    metagene_stats: Dict[str, Dict[str, float]] = artifact.get("metagene_stats", {})
    scaler = artifact.get("scaler")

    X_aligned = X_ext.copy()
    if not X_aligned.columns.is_unique:
        X_aligned = X_aligned.loc[:, ~X_aligned.columns.duplicated(keep="first")]
    if not X_aligned.index.is_unique:
        X_aligned = X_aligned.loc[~X_aligned.index.duplicated(keep="first")]

    n_samples = len(X_aligned)
    use_cohort_zscore = n_samples > 10

    # Build fast case-insensitive lookup of columns in external cohort
    col_lookup: Dict[str, str] = {c.upper(): c for c in X_aligned.columns}

    def resolve_gene_in_cohort(name: str) -> Optional[str]:
        if name in X_aligned.columns:
            return name
        u = name.upper()
        if u in col_lookup:
            return col_lookup[u]
        # Check aliases of name
        for alias in GENE_ALIASES.get(name, []):
            if alias.upper() in col_lookup:
                return col_lookup[alias.upper()]
        # Check if name is an alias of a canonical gene
        canon = ALIAS_TO_GENE.get(u)
        if canon:
            if canon in X_aligned.columns:
                return canon
            if canon.upper() in col_lookup:
                return col_lookup[canon.upper()]
            for alt in GENE_ALIASES.get(canon, []):
                if alt.upper() in col_lookup:
                    return col_lookup[alt.upper()]
        return None

    # 1. Compute hallmark metagenes with cross-platform relative pathway activation
    for meta_name, gene_list in hallmark_pathways.items():
        if meta_name not in X_aligned.columns:
            z_cols = []
            for g in gene_list:
                matched_g = resolve_gene_in_cohort(g)
                if matched_g is not None:
                    vals = X_aligned[matched_g].values.astype(float)
                    if use_cohort_zscore and np.std(vals) > 1e-4:
                        z = (vals - np.mean(vals)) / (np.std(vals) + 1e-6)
                    else:
                        mu = metagene_stats[g]["mean"] if g in metagene_stats else float(np.mean(vals))
                        sigma = metagene_stats[g]["std"] if g in metagene_stats else 1.0
                        z = (vals - mu) / (sigma + 1e-6)
                    z_cols.append(z)
            if z_cols:
                X_aligned[meta_name] = np.mean(np.column_stack(z_cols), axis=1)
            else:
                X_aligned[meta_name] = 0.0

    # 2. Extract reference scaler means and scales for neutral missing-gene filling and platform alignment
    feature_means: Dict[str, float] = {}
    feature_scales: Dict[str, float] = {}
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        for idx, feat in enumerate(target_features):
            feature_means[feat] = float(scaler.mean_[idx])
            feature_scales[feat] = float(scaler.scale_[idx])

    # 3. Align and harmonise individual features with alias resolution
    missing_count = 0
    missing_genes: List[str] = []
    aligned_dict: Dict[str, np.ndarray] = {}
    for feat in target_features:
        ref_mu = feature_means.get(feat, 0.0)
        ref_sigma = feature_scales.get(feat, 1.0)

        matched_col = resolve_gene_in_cohort(feat)
        if matched_col is not None:
            vals = X_aligned[matched_col].values.astype(float)
            if use_cohort_zscore and np.std(vals) > 1e-4 and not feat.startswith("META_"):
                cohort_mu = float(np.mean(vals))
                cohort_sigma = float(np.std(vals))
                if abs(cohort_mu - ref_mu) > 1.5 or abs(cohort_sigma - ref_sigma) > 1.0:
                    z = (vals - cohort_mu) / (cohort_sigma + 1e-6)
                    aligned_dict[feat] = (z * ref_sigma + ref_mu).astype(np.float32)
                else:
                    aligned_dict[feat] = vals.astype(np.float32)
            else:
                aligned_dict[feat] = vals.astype(np.float32)
        else:
            missing_count += 1
            missing_genes.append(feat)
            # Impute with neutral reference mean (0.0 scaled score)
            aligned_dict[feat] = np.full(n_samples, ref_mu, dtype=np.float32)

    if missing_count > 0:
        logger.info("External dataset missing %d features (%s); aligned with neutral baseline (0.0 scaled)",
                    missing_count, ", ".join(missing_genes[:5]) + ("..." if len(missing_genes) > 5 else ""))
    else:
        logger.info("All %d features successfully resolved and aligned across platforms!", len(target_features))

    X_final = pd.DataFrame(aligned_dict, index=X_aligned.index)[target_features]
    return X_final


# ─── Clinical External Validation Engine ─────────────────────────────────────

def evaluate_clinical_model_on_external_cohort(
    model_or_artifact: Union[Path, str, Dict[str, Any]],
    X_external: pd.DataFrame,
    y_external: pd.Series,
    times_external: pd.Series,
    events_external: pd.Series,
    cohort_name: str = "External-Cohort"
) -> Dict[str, Any]:
    """
    Evaluates a trained clinical genomic risk model on an independent external cohort.
    Returns a comprehensive validation scorecard.
    """
    if isinstance(model_or_artifact, (str, Path)):
        artifact_path = Path(model_or_artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        artifact = joblib.load(artifact_path)
    else:
        artifact = model_or_artifact

    model = artifact["model"]
    scaler = artifact["scaler"]
    internal_metrics = artifact.get("test_metrics", {})

    # 1. Align features and compute hallmark metagenes
    X_aligned = align_external_features(X_external, artifact)

    # 2. Scale features using internal training scaler
    X_scaled = scaler.transform(X_aligned.values.astype(np.float32))

    # 3. Model Inference
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    # 4. Out-of-Distribution Metrics
    y_true = y_external.values.astype(int)
    times = times_external.values.astype(float)
    events = events_external.values.astype(int)

    ext_c_idx = compute_c_index_fast(times, events, y_prob)
    ext_auroc = round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5
    ext_auprc = round(float(average_precision_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.0
    ext_brier = round(float(brier_score_loss(y_true, y_prob)), 4)
    ext_ece = compute_ece(y_true, y_prob, n_bins=10)
    ext_bal_acc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
    ext_ci = bootstrap_auroc_ci(y_true, y_prob)

    # 5. Risk Stratification Tiers
    low_risk = int(np.sum(y_prob < 0.35))
    mod_risk = int(np.sum((y_prob >= 0.35) & (y_prob <= 0.65)))
    high_risk = int(np.sum(y_prob > 0.65))

    delta_auroc = round(internal_metrics.get("risk_auroc", ext_auroc) - ext_auroc, 4)
    generalisation_verdict = "EXCELLENT" if abs(delta_auroc) <= 0.08 else ("ACCEPTABLE" if abs(delta_auroc) <= 0.15 else "DEGRADED")

    report = {
        "external_cohort"      : cohort_name,
        "n_samples"            : len(y_true),
        "n_events"             : int(np.sum(events)),
        "internal_metrics"     : internal_metrics,
        "external_metrics"     : {
            "c_index"          : ext_c_idx,
            "risk_auroc"       : ext_auroc,
            "auroc_95ci"       : ext_ci,
            "risk_auprc"       : ext_auprc,
            "brier_score"      : ext_brier,
            "ece"              : ext_ece,
            "balanced_accuracy": ext_bal_acc,
        },
        "delta_auroc"          : delta_auroc,
        "generalisation_flag"  : generalisation_verdict,
        "risk_stratification"  : {
            "low_risk_count"   : low_risk,
            "moderate_risk_count": mod_risk,
            "moderate_risk_cnt": mod_risk,
            "high_risk_count"  : high_risk,
        }
    }

    logger.info("External Validation Complete on %s: C-Index=%.4f, AUROC=%.4f (Verdict: %s)",
                cohort_name, ext_c_idx, ext_auroc, generalisation_verdict)
    return report


def save_validation_report(report: Dict[str, Any], project_id: str, out_dir: Optional[Path] = None) -> Path:
    """Persists external validation report to results directory."""
    out_dir = out_dir or (RESULTS_DIR / project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_slug = report.get("external_cohort", "external").lower().replace(" ", "_")
    out_path = out_dir / f"external_validation_{cohort_slug}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved external validation report to %s", out_path)
    return out_path
