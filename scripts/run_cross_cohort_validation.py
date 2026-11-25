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


def find_model_artifact(requested_model: str) -> Tuple[Optional[Path], str, List[str]]:
    """
    Resolves the model artifact path by checking local and Google Drive locations,
    handling aliases, and falling back to enclosing pan-cancer/pan-organ models.
    """
    search_dirs = [
        ROOT / "models_artifacts",
        Path("/content/drive/MyDrive/GUIDO_5TB_DATA/models_artifacts"),
    ]
    search_dirs = [d for d in search_dirs if d.exists()]

    available_models: Dict[str, Path] = {}
    for d in search_dirs:
        for p in d.glob("*/xgboost_clinical_model.joblib"):
            m_id = p.parent.name
            if m_id not in available_models:
                available_models[m_id] = p

    all_ids = sorted(list(available_models.keys()))
    clean_req = requested_model.strip()

    if clean_req.lower() == "auto":
        if "TCGA-PAN-CANCER-ALL" in available_models:
            return available_models["TCGA-PAN-CANCER-ALL"], "TCGA-PAN-CANCER-ALL", all_ids
        if available_models:
            chosen = all_ids[0]
            return available_models[chosen], chosen, all_ids
        return None, "", all_ids

    project_id = clean_req if clean_req.startswith("TCGA-") else f"TCGA-{clean_req}"
    if project_id in available_models:
        return available_models[project_id], project_id, all_ids

    # Case-insensitive match
    for m_id, path in available_models.items():
        if m_id.upper() == project_id.upper():
            return path, m_id, all_ids

    # Organ and Pan-Cancer fallbacks
    ORGAN_FALLBACKS = {
        "KIRP": "TCGA-PAN-RENAL", "KIRC": "TCGA-PAN-RENAL", "KICH": "TCGA-PAN-RENAL",
        "LUAD": "TCGA-PAN-LUNG",  "LUSC": "TCGA-PAN-LUNG",
        "BRCA": "TCGA-PAN-BREAST",
        "COAD": "TCGA-PAN-GI",    "READ": "TCGA-PAN-GI",  "ESCA": "TCGA-PAN-GI", "STAD": "TCGA-PAN-GI",
    }
    short_code = clean_req.upper().replace("TCGA-", "").replace("TCGA.", "")
    if short_code in ORGAN_FALLBACKS:
        pan_id = ORGAN_FALLBACKS[short_code]
        if pan_id in available_models:
            log.info("Model '%s' not found, but found enclosing pan-organ model '%s'. Evaluating '%s'...", project_id, pan_id, pan_id)
            return available_models[pan_id], pan_id, all_ids

    if "TCGA-PAN-CANCER-ALL" in available_models:
        log.info("Model '%s' not found, but found master Pan-Cancer model 'TCGA-PAN-CANCER-ALL'. Evaluating master model...", project_id)
        return available_models["TCGA-PAN-CANCER-ALL"], "TCGA-PAN-CANCER-ALL", all_ids

    return None, project_id, all_ids


def main():
    parser = argparse.ArgumentParser(description="Cross-Cohort External Clinical Validation Runner")
    parser.add_argument("--model", type=str, default="auto", help="Model cohort identifier (e.g. TCGA-PAN-RENAL, TCGA-PAN-CANCER-ALL, or 'auto')")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "metabric", "gse31210"], help="External dataset")
    args = parser.parse_args()

    artifact_path, project_id, available = find_model_artifact(args.model)

    if artifact_path is None or not artifact_path.exists():
        log.error("Artifact not found for model: %s", args.model)
        if available:
            log.info("Available trained models on disk:\n  %s", "\n  ".join(f"- {m}" for m in available))
            log.info("Run with: --model %s", available[0])
        else:
            log.error("No trained models found on disk or in Google Drive. Please run Cell 5 (retrain_fixed.py) first.")
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
