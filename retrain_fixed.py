#!/usr/bin/env python3
"""
Retrain all XGBoost clinical models with three fixes applied:

  FIX 1 – No feature-selection leakage
          Feature selection (variance filter + top-k by F-statistic) is
          performed ONLY on the training split, never on val/test data.

  FIX 2 – Class-imbalance correction
          scale_pos_weight = n_negative / n_positive, computed from the
          training split only.

  FIX 3 – Regularisation
          Lighter tree (max_depth=4), stronger L2 (reg_lambda=2),
          min_child_weight=5 to resist fitting individual samples.

Outputs per cancer:
  models_artifacts/<CANCER>/xgboost_clinical_model.joblib
  results/<CANCER>/evaluation_clinical.json
  results/<CANCER>/xgboost_clinical_summary.json

Usage:
  python retrain_fixed.py                        # all available cancers
  python retrain_fixed.py BRCA LUAD KIRC         # specific subset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
XENA_DIR   = ROOT / "data" / "raw" / "xena"
MODELS_DIR = ROOT / "models_artifacts"
RESULTS_DIR= ROOT / "results"

# ─── Xena dataset filenames ──────────────────────────────────────────────────
XENA_FILE = {
    "BRCA": "TCGA.BRCA.sampleMap_HiSeqV2.tsv",
    "LUAD": "TCGA.LUAD.sampleMap_HiSeqV2.tsv",
    "LUSC": "TCGA.LUSC.sampleMap_HiSeqV2.tsv",
    "KIRC": "TCGA.KIRC.sampleMap_HiSeqV2.tsv",
    "KIRP": "TCGA.KIRP.sampleMap_HiSeqV2.tsv",
    "KICH": "TCGA.KICH.sampleMap_HiSeqV2.tsv",
    "LIHC": "TCGA.LIHC.sampleMap_HiSeqV2.tsv",
    "COAD": "TCGA.COAD.sampleMap_HiSeqV2.tsv",
    "LGG" : "TCGA.LGG.sampleMap_HiSeqV2.tsv",
    "GBM" : "TCGA.GBM.sampleMap_HiSeqV2.tsv",
    "PAAD": "TCGA.PAAD.sampleMap_HiSeqV2.tsv",
    "PRAD": "TCGA.PRAD.sampleMap_HiSeqV2.tsv",
    "SKCM": "TCGA.SKCM.sampleMap_HiSeqV2.tsv",
}

N_TOP_GENES = 500   # features to keep
RANDOM_SEED = 42
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_xena(cancer: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Load a Xena TSV and return X (samples × genes), y, gene_names."""
    fname = XENA_FILE.get(cancer)
    if not fname:
        raise ValueError(f"No Xena file mapped for cancer {cancer}")
    path = XENA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Xena file not found: {path}")

    log.info("  Loading %s ...", path.name)
    df = pd.read_csv(path, sep="\t", index_col=0)          # genes × samples
    df = df.dropna(how="all")

    # Label: -01 = tumor (1), -11 = normal (0)
    tumor_cols  = [c for c in df.columns if c.endswith("-01")]
    normal_cols = [c for c in df.columns if c.endswith("-11")]
    if len(normal_cols) < 3:
        raise ValueError(f"{cancer}: only {len(normal_cols)} normal samples — insufficient")

    keep_cols = tumor_cols + normal_cols
    X = df[keep_cols].T.fillna(0).values.astype(np.float32)   # samples × genes
    y = np.array([1]*len(tumor_cols) + [0]*len(normal_cols), dtype=np.int32)
    genes = list(df.index)

    log.info("  %s: %d tumor, %d normal, %d genes",
             cancer, len(tumor_cols), len(normal_cols), len(genes))
    return X, y, genes


def select_features_on_train(
    X_train: np.ndarray, y_train: np.ndarray, genes: list, k: int = N_TOP_GENES
) -> list:
    """
    FIX 1 — feature selection strictly inside train split.
    Uses univariate F-statistic (ANOVA) to rank genes, returns top-k names.
    """
    # Step 1: drop near-zero-variance genes first (cheap)
    var = X_train.var(axis=0)
    nzv_mask = var > np.percentile(var, 10)   # keep top 90% by variance
    X_nzv = X_train[:, nzv_mask]
    genes_nzv = [g for g, m in zip(genes, nzv_mask) if m]

    # Step 2: ANOVA F-test on training data only
    fscores, _ = f_classif(X_nzv, y_train)
    fscores = np.nan_to_num(fscores, nan=0.0)
    top_idx = np.argsort(fscores)[::-1][:k]
    selected = [genes_nzv[i] for i in top_idx]
    return selected


def compute_metrics(y_true, y_prob) -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel().tolist() + [0,0,0,0])[:4]
    return {
        "auroc"      : float(roc_auc_score(y_true, y_prob)),
        "auprc"      : float(average_precision_score(y_true, y_prob)),
        "accuracy"   : float(accuracy_score(y_true, y_pred)),
        "precision"  : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"     : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"         : float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def cv_auroc(X_train, y_train, genes, k=N_TOP_GENES, n_splits=5) -> float:
    """5-fold CV AUROC on training data to monitor overfitting."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xva, yva = X_train[va_idx], y_train[va_idx]

        # Feature selection inside fold (leak-free)
        feats_fold = select_features_on_train(Xtr, ytr, genes, k)
        gene_idx = [genes.index(g) for g in feats_fold]

        Xtr_f = Xtr[:, gene_idx]
        Xva_f = Xva[:, gene_idx]

        scaler = StandardScaler()
        Xtr_f = scaler.fit_transform(Xtr_f)
        Xva_f = scaler.transform(Xva_f)

        n_neg_fold = int((ytr == 0).sum())
        n_pos_fold = int((ytr == 1).sum())
        spw   = n_neg_fold / max(n_pos_fold, 1) if n_neg_fold >= 10 else 1.0
        mcw   = max(1, min(5, n_neg_fold // 8))

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=0.1,
            min_child_weight=mcw,
            scale_pos_weight=spw,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        clf.fit(Xtr_f, ytr)
        probs = clf.predict_proba(Xva_f)[:, 1]
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(yva, probs))

    cv_mean = float(np.mean(aucs))
    log.info("  5-fold CV AUROC = %.4f (±%.4f)", cv_mean, float(np.std(aucs)))
    return cv_mean


# ─── Main training function ──────────────────────────────────────────────────

def train_cancer(cancer: str) -> dict:
    log.info("=" * 60)
    log.info("Training  TCGA-%s", cancer)
    log.info("=" * 60)

    X, y, genes = load_xena(cancer)

    # ── Stratified 70/15/15 split ────────────────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y, random_state=RANDOM_SEED
    )
    rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test,
        stratify=y_tmp, random_state=RANDOM_SEED
    )
    log.info("  Split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))

    # ── FIX 1: Feature selection on training data ONLY ──────────────────────
    log.info("  Selecting top %d genes from training split ...", N_TOP_GENES)
    selected_genes = select_features_on_train(X_train, y_train, genes, N_TOP_GENES)
    gene_idx = [genes.index(g) for g in selected_genes]

    X_tr_f  = X_train[:, gene_idx]
    X_val_f = X_val[:,   gene_idx]
    X_te_f  = X_test[:,  gene_idx]

    # ── FIX 2: Scale from training split only ───────────────────────────────
    scaler = StandardScaler()
    X_tr_f  = scaler.fit_transform(X_tr_f)
    X_val_f = scaler.transform(X_val_f)
    X_te_f  = scaler.transform(X_te_f)

    # ── FIX 2: Class-imbalance weight (from training split only) ───────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    # scale_pos_weight = n_neg/n_pos balances gradients.
    # Guard: if n_neg < 10, too few samples for meaningful balancing — keep 1.0
    spw = n_neg / max(n_pos, 1) if n_neg >= 10 else 1.0
    log.info("  Class balance train: %d pos / %d neg  → scale_pos_weight=%.3f",
             n_pos, n_neg, spw)

    # ── FIX 3: Regularisation — adaptive to cohort size ────────────────────
    # min_child_weight must be << n_neg so splits on the minority class
    # are always possible. Use n_neg // 8, clamped to [1, 5].
    mcw = max(1, min(5, n_neg // 8))
    log.info("  min_child_weight=%d  (n_neg=%d)", mcw, n_neg)

    # ── Optional: 5-fold CV on training data to check for overfit ───────────
    cv_auroc(X_train, y_train, genes)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,           # was 5
        learning_rate=0.05,    # was 0.1 (slower, less overfit)
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,        # L2 ridge on leaf weights
        reg_alpha=0.1,         # L1 on leaf weights
        min_child_weight=mcw,  # adaptive to cohort size
        scale_pos_weight=spw,  # FIX 2
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    clf.fit(X_tr_f, y_train,
            eval_set=[(X_val_f, y_val)],
            verbose=False)

    # ── Evaluate ────────────────────────────────────────────────────────────
    val_metrics  = compute_metrics(y_val,  clf.predict_proba(X_val_f)[:, 1])
    test_metrics = compute_metrics(y_test, clf.predict_proba(X_te_f)[:, 1])
    train_metrics= compute_metrics(y_train,clf.predict_proba(X_tr_f)[:, 1])

    log.info("  Train AUROC = %.4f  (watch for overfit if >> val)",
             train_metrics["auroc"])
    log.info("  Val   AUROC = %.4f", val_metrics["auroc"])
    log.info("  Test  AUROC = %.4f  ← honest number", test_metrics["auroc"])

    # ── Save artifact ───────────────────────────────────────────────────────
    artifact = {
        "model"   : clf,
        "scaler"  : scaler,
        "features": selected_genes,
    }
    model_dir = MODELS_DIR / f"TCGA-{cancer}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / "xgboost_clinical_model.joblib")
    log.info("  Saved → %s/xgboost_clinical_model.joblib", model_dir.name)

    # ── Save evaluation JSON ────────────────────────────────────────────────
    result_dir = RESULTS_DIR / f"TCGA-{cancer}"
    result_dir.mkdir(parents=True, exist_ok=True)

    eval_record = {
        "project_id"          : f"TCGA-{cancer}",
        "cancer_type"         : cancer,
        "model_type"          : "clinical",
        "n_samples"           : len(y),
        "n_features"          : N_TOP_GENES,
        "n_tumor"             : int((y == 1).sum()),
        "n_normal"            : int((y == 0).sum()),
        "split_train"         : len(y_train),
        "split_val"           : len(y_val),
        "split_test"          : len(y_test),
        "scale_pos_weight"    : round(spw, 4),
        "fixes_applied"       : ["leak_free_feature_selection",
                                  "class_imbalance_correction",
                                  "regularised_xgboost"],
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}"  : v for k, v in val_metrics.items()},
        **{f"test_{k}" : v for k, v in test_metrics.items()},
    }
    with open(result_dir / "evaluation_clinical.json", "w") as f:
        json.dump(eval_record, f, indent=2)

    summary = {
        "project_id"      : f"TCGA-{cancer}",
        "cancer_type"     : cancer,
        "n_samples"       : len(y),
        "n_tumor"         : int((y == 1).sum()),
        "n_normal"        : int((y == 0).sum()),
        "n_features"      : N_TOP_GENES,
        "val_auroc"       : val_metrics["auroc"],
        "test_auroc"      : test_metrics["auroc"],
        "train_auroc"     : train_metrics["auroc"],
        "scale_pos_weight": round(spw, 4),
        "split_train"     : len(y_train),
        "split_val"       : len(y_val),
        "split_test"      : len(y_test),
    }
    with open(result_dir / "xgboost_clinical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("  Saved evaluation results → %s", result_dir.name)
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    all_cancers = list(XENA_FILE.keys())

    parser = argparse.ArgumentParser(description="Retrain XGBoost models with fixes")
    parser.add_argument("cancers", nargs="*", default=[],
                        help="Cancer codes to train (default: all available)")
    args = parser.parse_args()

    requested = [c.upper() for c in args.cancers] if args.cancers else all_cancers

    # Only run cancers for which the Xena file actually exists
    available = [c for c in requested if (XENA_DIR / XENA_FILE.get(c, "")).exists()]
    missing   = [c for c in requested if c not in available]
    if missing:
        log.warning("Skipping (no Xena file): %s", missing)

    if not available:
        log.error("No valid cancers to train. Check %s", XENA_DIR)
        sys.exit(1)

    log.info("Will train: %s", available)
    summaries = {}
    for cancer in available:
        try:
            summaries[cancer] = train_cancer(cancer)
        except Exception as e:
            log.error("FAILED %s: %s", cancer, e)

    log.info("\n%s", "=" * 60)
    log.info("SUMMARY")
    log.info("%-6s  %7s  %7s  %7s  %6s", "Cancer", "Train", "Val", "Test", "SPW")
    log.info("-" * 46)
    for cancer, s in summaries.items():
        log.info("%-6s  %7.4f  %7.4f  %7.4f  %6.3f",
                 cancer,
                 s["train_auroc"],
                 s["val_auroc"],
                 s["test_auroc"],
                 s["scale_pos_weight"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
