#!/usr/bin/env python3
"""
CLI Runner for Minimal Gene Panel Distillation (Survival-Based Feature Selection)
================================================================================
Distills high-dimensional genomic models down to clinically assayable candidate
panels (15, 25, 35, 50 genes) using ElasticNet/L1-penalized Cox Proportional Hazards
and nested cross-validation out-of-fold evaluation.

Outputs:
  - Stability selection rankings (reports/minimal_gene_panels.json)
  - Genuine out-of-fold performance retention curves (C-index vs panel size)
  - In-silico candidate assay design specifications (reports/assay_specs/panel_{size}.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.panel_distillation import MinimalGenePanelDistiller

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("distill")


def generate_synthetic_survival_cohort(
    n_patients: int = 150,
    n_genes: int = 80,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generates synthetic survival benchmark cohort with true underlying hazards."""
    np.random.seed(seed)
    gene_names = [
        "MKI67", "TP53", "CD274", "VEGFA", "EGFR", "MYC", "PTEN", "BRCA1",
        "CDH1", "SLC2A1", "IL6", "MLH1", "AURKA", "PLK1", "BCL2", "TERT",
        "VIM", "HK2", "PDCD1", "TNF"
    ]
    gene_names += [f"GENE_{i:03d}" for i in range(len(gene_names), n_genes)]

    X = np.random.normal(loc=0.0, scale=1.0, size=(n_patients, n_genes))
    # True biological hazard driven by first 4 genes
    hazard = 0.45 * X[:, 0] + 0.40 * X[:, 1] + 0.35 * X[:, 2] + 0.30 * X[:, 3] - 0.25 * X[:, 4]

    base_time = np.random.exponential(scale=1100.0, size=n_patients)
    true_time = base_time / np.exp(hazard)
    cens_time = np.random.uniform(150, 2800, size=n_patients)

    times = np.minimum(true_time, cens_time).clip(min=1.0)
    events = (true_time <= cens_time).astype(int)

    df_X = pd.DataFrame(X, columns=gene_names)
    return df_X, times, events


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill GUIDO Genomic Models into Minimal Clinical Gene Panels")
    parser.add_argument(
        "--cohort",
        type=str,
        default="synthetic",
        help="Real TCGA cohort code (e.g. 'KICH', 'KIRC', 'KIRP') or 'synthetic'"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicitly run on synthetic benchmark data"
    )
    parser.add_argument("--n-patients", type=int, default=150, help="Number of benchmark patients if synthetic")
    parser.add_argument("--n-genes", type=int, default=80, help="Number of input genes if synthetic")
    parser.add_argument("--n-folds", type=int, default=4, help="Number of outer cross-validation folds")
    parser.add_argument("--output-dir", type=str, default="reports/assay_specs", help="Output directory for assay specs")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    is_synthetic = args.synthetic or (args.cohort.lower() == "synthetic")

    if not is_synthetic:
        # Real cohort execution path
        cohort_name = args.cohort.upper().replace("TCGA-", "")
        log.info("[REAL DATA] Loading real TCGA cohort: TCGA-%s", cohort_name)
        try:
            from retrain_fixed import load_cohort_data
            df_X, y_series, time_series, event_series = load_cohort_data(cohort_name)
            times = time_series.values.astype(float)
            events = event_series.values.astype(int)
            data_source = f"TCGA-{cohort_name}"
            endpoint = "PFI"
            log.info(
                "[REAL DATA] Loaded TCGA-%s: N=%d patients, %d events (%.1f%% event rate), P=%d genes",
                cohort_name, len(df_X), int(np.sum(events == 1)),
                float(np.mean(events == 1)) * 100, df_X.shape[1]
            )
        except Exception as e:
            log.error(
                "[ERROR] Failed loading real TCGA cohort '%s': %s. "
                "Ensure TCGA-CDR and expression matrices exist in data/raw/xena/ or data/processed/shards/.",
                cohort_name, e
            )
            return 1
    else:
        # Synthetic benchmark execution path
        log.warning(
            "[SYNTHETIC BENCHMARK] Generating synthetic survival cohort (N=%d patients, P=%d genes). "
            "RESULTS REFLECT IN-SILICO METHODOLOGICAL BENCHMARKING ONLY — NOT REAL PATIENT BIOLOGY.",
            args.n_patients, args.n_genes
        )
        df_X, times, events = generate_synthetic_survival_cohort(
            n_patients=args.n_patients,
            n_genes=args.n_genes,
            seed=42,
        )
        data_source = "synthetic_benchmark"
        endpoint = "simulated_survival_time"

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[15, 25, 35, 50],
        n_folds=args.n_folds,
        penalizer=0.1,
        l1_ratio=0.7,
        n_subsamples=5,
        random_state=42,
    )

    log.info("Executing ElasticNet-Cox stability selection and out-of-fold nested cross-validation...")
    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events,
        data_source=data_source,
        is_synthetic=is_synthetic,
        endpoint=endpoint,
    )

    # Save individual candidate panel assay specifications
    for size in report.candidate_panels.keys():
        spec_path = out_dir / f"clinical_assay_spec_{size}_genes.json"
        distiller.export_assay_specifications(panel_size=size, output_path=spec_path)

    # Save master distillation summary
    master_report = {
        "title": "Minimal Gene Panel Distillation Report",
        "data_source": report.data_source,
        "is_synthetic": report.is_synthetic,
        "survival_endpoint": report.endpoint,
        "evaluation_method": report.evaluation_method,
        "sample_size": report.n_samples,
        "event_count": report.n_events,
        "censoring_rate": report.censoring_rate,
        "full_model_gene_count": report.full_model_gene_count,
        "full_model_out_of_fold_c_index": round(report.full_model_c_index, 4),
        "full_model_c_index_95_ci": report.full_model_c_index_ci,
        "full_model_unos_c_index": round(report.full_model_unos_c_index, 4),
        "candidate_panels": {
            str(k): {
                "panel_size": v.panel_size,
                "out_of_fold_c_index": round(v.c_index, 4),
                "c_index_95_ci": v.c_index_ci,
                "unos_c_index": round(v.unos_c_index, 4),
                "retention_ratio": round(v.retention_ratio, 4),
                "assay_recommendation": v.assay_type_recommendation,
                "genes": v.genes,
                "hallmark_coverage": v.hallmark_representation,
                "validation_status": v.validation_status,
                "evidence_level": v.evidence_level,
            }
            for k, v in report.candidate_panels.items()
        },
        "retention_curve": report.retention_curve,
        "top_20_stable_genes": report.stability_rankings[:20],
        "limitations": report.limitations,
    }

    master_path = ROOT / "reports" / "minimal_gene_panels.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    log.info("Master distillation report written to: %s", master_path)

    print("\n" + "=" * 80)
    print("MINIMAL GENE PANEL DISTILLATION RESULTS (OUT-OF-FOLD EVALUATION)")
    print("=" * 80)
    print(f"Data Source:       {report.data_source} (Synthetic: {report.is_synthetic})")
    print(f"Survival Endpoint: {report.endpoint} (N={report.n_samples}, Events={report.n_events}, Censored={report.censoring_rate:.1%})")
    print(f"Full Model:        {report.full_model_gene_count} genes | Out-of-Fold C-Index: {report.full_model_c_index:.4f} "
          f"[95% CI: {report.full_model_c_index_ci[0]:.4f} - {report.full_model_c_index_ci[1]:.4f}]\n")
    print(f"{'Panel Size':<12} | {'OOF C-Index':<12} | {'95% CI':<18} | {'Retention':<10} | {'Status':<15}")
    print("-" * 80)
    for k in sorted(report.candidate_panels.keys()):
        p = report.candidate_panels[k]
        ci_str = f"[{p.c_index_ci[0]:.3f}, {p.c_index_ci[1]:.3f}]"
        print(f"{p.panel_size:<12} | {p.c_index:<12.4f} | {ci_str:<18} | {p.retention_ratio * 100:<9.1f}% | {p.validation_status:<15}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
