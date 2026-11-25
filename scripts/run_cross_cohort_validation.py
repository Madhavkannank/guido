#!/usr/bin/env python3
"""
CLI Runner for Cross-Cohort External Clinical Validation
========================================================

Evaluates trained TCGA clinical genomic risk models on independent datasets:
  - METABRIC (cBioPortal breast cancer cohort)
  - GSE31210 (NCBI GEO lung adenocarcinoma cohort)
  - Synthetic benchmark cohort

Usage:
  py -3.12 scripts/run_cross_cohort_validation.py --model TCGA-KIRP --dataset synthetic
  py -3.12 scripts/run_cross_cohort_validation.py --model TCGA-KICH --dataset synthetic
  py -3.12 scripts/run_cross_cohort_validation.py --model TCGA-BRCA --dataset metabric
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deep_surv import PyTorchDeepSurvClassifier
setattr(sys.modules["__main__"], "PyTorchDeepSurvClassifier", PyTorchDeepSurvClassifier)

from src.validation.cross_cohort import (
    evaluate_clinical_model_on_external_cohort,
    save_validation_report,
)
from src.validation.external_datasets import (
    fetch_geo_gse31210,
    fetch_metabric,
    generate_synthetic_external_cohort,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cross-Cohort External Clinical Validation Runner")
    parser.add_argument("--model", type=str, default="TCGA-KIRP", help="Model cohort identifier (e.g. TCGA-KIRP, TCGA-KICH)")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "metabric", "gse31210"], help="External dataset")
    args = parser.parse_args()

    project_id = args.model if args.model.startswith("TCGA-") else f"TCGA-{args.model}"
    artifact_path = ROOT / "models_artifacts" / project_id / "xgboost_clinical_model.joblib"

    if not artifact_path.exists():
        log.error("Artifact not found: %s. Please train the model first with retrain_fixed.py.", artifact_path)
        sys.exit(1)

    log.info("Loading external dataset: %s ...", args.dataset.upper())
    if args.dataset == "metabric":
        X_ext, y_ext, t_ext, e_ext = fetch_metabric()
        cohort_name = "METABRIC (Breast Cancer)"
    elif args.dataset == "gse31210":
        X_ext, y_ext, t_ext, e_ext = fetch_geo_gse31210()
        cohort_name = "GSE31210 (Lung Adenocarcinoma)"
    else:
        X_ext, y_ext, t_ext, e_ext = generate_synthetic_external_cohort(n_samples=180, seed=42)
        cohort_name = "Independent Synthetic Benchmark"

    log.info("Running out-of-distribution clinical risk validation against %s ...", cohort_name)
    report = evaluate_clinical_model_on_external_cohort(
        model_or_artifact=artifact_path,
        X_external=X_ext,
        y_external=y_ext,
        times_external=t_ext,
        events_external=e_ext,
        cohort_name=cohort_name,
    )

    out_file = save_validation_report(report, project_id=project_id)
    log.info("Validation report saved to: %s", out_file)
    print("\n" + "=" * 60)
    print(f" EXTERNAL VALIDATION SUMMARY: {project_id} on {cohort_name}")
    print("=" * 60)
    print(f" Harrell's C-Index       : {report['external_metrics']['c_index']:.4f}")
    print(f" 3-Year Risk AUROC       : {report['external_metrics']['risk_auroc']:.4f} (95% CI: {report['external_metrics']['auroc_95ci']})")
    print(f" Brier Score (Loss)      : {report['external_metrics']['brier_score']:.4f}")
    print(f" Expected Calib. Error   : {report['external_metrics']['ece']:.4f}")
    print(f" Generalisation Verdict  : {report['generalisation_flag']}")
    print(f" Risk Stratification     : Low={report['risk_stratification']['low_risk_count']}, Mod={report['risk_stratification']['moderate_risk_cnt']}, High={report['risk_stratification']['high_risk_count']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
