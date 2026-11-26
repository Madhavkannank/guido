#!/usr/bin/env python3
"""
CLI Runner for Minimal Gene Panel Distillation
==============================================
Distills high-dimensional genomic models down to clinically assayable
15, 25, 35, and 50-gene panels for targeted diagnostic platforms (RT-qPCR, NanoString).

Outputs:
  - Stability selection rankings (reports/minimal_gene_panels.json)
  - Performance retention curve (C-index vs panel size)
  - Clinical Assay Design Specifications (reports/assay_specs/panel_{size}.json)
  - Detailed Markdown Summary (reports/MINIMAL_GENE_PANELS.md)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill GUIDO Genomic Models into Minimal Clinical Gene Panels")
    parser.add_argument("--cohort", type=str, default="synthetic", help="Cohort name or 'synthetic'")
    parser.add_argument("--n-patients", type=int, default=200, help="Number of benchmark patients if synthetic")
    parser.add_argument("--n-genes", type=int, default=200, help="Number of input genes if synthetic")
    parser.add_argument("--output-dir", type=str, default="reports/assay_specs", help="Output directory for assay specs")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating/loading genomic survival cohort (N=%d patients, P=%d genes)...", args.n_patients, args.n_genes)
    np.random.seed(42)
    gene_names = [
        "MKI67", "TP53", "CD274", "VEGFA", "EGFR", "MYC", "PTEN", "BRCA1",
        "CDH1", "SLC2A1", "IL6", "MLH1", "AURKA", "PLK1", "BCL2", "TERT",
        "VIM", "HK2", "PDCD1", "TNF"
    ]
    # Fill remaining
    gene_names += [f"GENE_{i:03d}" for i in range(len(gene_names), args.n_genes)]

    X = np.random.normal(loc=0.0, scale=1.0, size=(args.n_patients, args.n_genes))
    # Biological prognostic hazard
    hazard = 0.45 * X[:, 0] + 0.40 * X[:, 1] + 0.35 * X[:, 2] + 0.30 * X[:, 3] - 0.25 * X[:, 4]

    base_time = np.random.exponential(scale=1100.0, size=args.n_patients)
    true_time = base_time / np.exp(hazard)
    cens_time = np.random.uniform(150, 2800, size=args.n_patients)

    times = np.minimum(true_time, cens_time).clip(min=1.0)
    events = (true_time <= cens_time).astype(int)

    df_X = pd.DataFrame(X, columns=gene_names)

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[15, 25, 35, 50],
        n_folds=4,
        random_state=42,
    )

    log.info("Executing stability selection and cross-validated panel distillation...")
    report = distiller.fit(df_X, times, events)

    # Save individual panel assay specifications
    for size in [15, 25, 35, 50]:
        spec_path = out_dir / f"clinical_assay_spec_{size}_genes.json"
        distiller.export_assay_specifications(panel_size=size, output_path=spec_path)

    # Save master distillation summary
    master_report = {
        "full_model_gene_count": report.full_model_gene_count,
        "full_model_c_index": round(report.full_model_c_index, 4),
        "candidate_panels": {
            str(k): {
                "panel_size": v.panel_size,
                "c_index": round(v.c_index, 4),
                "retention_ratio": round(v.retention_ratio, 4),
                "assay_recommendation": v.assay_type_recommendation,
                "genes": v.genes,
                "hallmark_coverage": v.hallmark_representation,
            }
            for k, v in report.candidate_panels.items()
        },
        "retention_curve": report.retention_curve,
        "top_20_stable_genes": report.stability_rankings[:20],
    }

    master_path = ROOT / "reports" / "minimal_gene_panels.json"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    log.info("Master distillation report written to: %s", master_path)

    print("\n" + "=" * 75)
    print("MINIMAL GENE PANEL DISTILLATION RESULTS")
    print("=" * 75)
    print(f"Full Genome Model: {report.full_model_gene_count} genes | Baseline C-Index: {report.full_model_c_index:.4f}\n")
    print(f"{'Panel Size':<12} | {'C-Index':<10} | {'Retention':<10} | {'Platform':<25}")
    print("-" * 75)
    for k in sorted(report.candidate_panels.keys()):
        p = report.candidate_panels[k]
        plat_short = p.assay_type_recommendation.split(" ")[0]
        print(f"{p.panel_size:<12} | {p.c_index:<10.4f} | {p.retention_ratio * 100:<9.1f}% | {p.assay_type_recommendation[:35]}...")
    print("=" * 75)
    print("Targeted assay designs exported to:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
