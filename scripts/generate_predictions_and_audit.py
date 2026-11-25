#!/usr/bin/env python3
"""Generate model outputs (predictions) for test split and run the biomedical audit.

Saves:
 - results/<project>/predictions_test.csv
 - results/<project>/biomedical_audit_<disease>.json

If `GROQ_API_KEY` is not set, writes a rule-based fallback audit report.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(project: str, disease_name: str | None = None):
    repo_root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo_root))

    from src.api.pipeline import _load_splits, run_biomedical_audit_pipeline
    from src.literature.pubmed_retrieval import get_disease_keyword
    from config.settings import MODELS_DIR, RESULTS_DIR

    if not disease_name or disease_name == "lung adenocarcinoma":
        disease_name = get_disease_keyword(project)

    # Search for clinical risk model first, then legacy primary models
    model_candidates = [
        MODELS_DIR / project / "xgboost_clinical_model.joblib",
        MODELS_DIR / project / "primary_model.joblib",
        MODELS_DIR / project / "ElasticNet.joblib",
        MODELS_DIR / project / "ShallowNN.joblib",
        MODELS_DIR / project / "xgboost_model.joblib",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if not model_path:
        raise FileNotFoundError(f"Model artifact not found in {MODELS_DIR / project}")

    logger.info("Loading model from %s", model_path)
    model_raw = joblib.load(model_path)
    model = model_raw.get("model", model_raw) if isinstance(model_raw, dict) else model_raw
    scaler = model_raw.get("scaler", None) if isinstance(model_raw, dict) else None

    splits = _load_splits(project)
    X_test, y_test = splits["test"]

    logger.info("Predicting probabilities on test set (%d samples)", X_test.shape[0])
    X_in = X_test
    if scaler is not None:
        try:
            X_in = scaler.transform(X_test.values)
        except Exception:
            pass

    # predict_proba may not exist for some models; handle gracefully
    try:
        y_prob = model.predict_proba(X_in)[:, 1]
    except Exception:
        # fallback to decision_function or predict
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_in)
            # map to [0,1]
            import numpy as np
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            y_prob = model.predict(X_in)

    import numpy as np
    y_pred = (y_prob >= 0.5).astype(int)

    out_dir = RESULTS_DIR / project
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({"sample_id": X_test.index.astype(str), "y_true": y_test.values, "y_prob": y_prob, "y_pred": y_pred})
    preds_path = out_dir / "predictions_test.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info("Saved predictions to %s", preds_path)

    # Prepare audit input
    audit_input = {
        "project_id": project,
        "disease_name": disease_name,
        "training_sample_size": int(splits["train"][0].shape[0]),
        "validation_sample_size": int(splits["val"][0].shape[0]),
        "model_metrics": {},
        "multi_omics_features": {"transcriptomics_features": list(X_test.columns)},
        "selected_biomarkers": [],
        "shap_importance_scores": {},
        "stability_scores": {},
    }

    # try to attach existing metrics and stability files
    metrics_path = out_dir / "eval_test.json"
    if metrics_path.exists():
        try:
            audit_input["model_metrics"] = json.loads(metrics_path.read_text())
        except Exception:
            pass

    stable_path = out_dir / "stable_biomarkers.csv"
    if stable_path.exists():
        try:
            df = pd.read_csv(stable_path)
            audit_input["selected_biomarkers"] = df["gene_id"].tolist()
            audit_input["stability_scores"] = dict(zip(df["gene_id"], df["stability_score"]))
        except Exception:
            pass

    # Run Groq audit (may raise if GROQ_API_KEY unset) — use pipeline function
    try:
        report = run_biomedical_audit_pipeline(audit_input)
        logger.info("Groq audit completed and saved")
    except Exception as exc:
        logger.warning("Groq audit failed: %s — generating fallback report", exc)
        # Basic rule-based fallback
        fs = audit_input
        total_features = len(fs["multi_omics_features"]["transcriptomics_features"]) if fs.get("multi_omics_features") else 0
        total_samples = max(int(fs.get("training_sample_size", 0)) + int(fs.get("validation_sample_size", 0)), 1)
        feature_sample_ratio = total_features / total_samples if total_samples else float("inf")
        report = {
            "data_audit": {
                "feature_sample_ratio": f"{feature_sample_ratio:.3f}",
                "overfitting_risk": "HIGH" if feature_sample_ratio > 10 else "MODERATE/LOW",
                "generalization_comment": "Small sample relative to feature count; treat results cautiously.",
            },
            "model_performance_review": {
                "performance_quality": fs.get("model_metrics", {}).get("auroc", None),
                "bias_risk": "UNKNOWN — no demographic metadata provided",
            },
            "biomarker_analysis": [],
            "flagged_unstable_features": [],
            "high_confidence_targets": [],
            "overall_system_verdict": "Insufficient GROQ config — returning rule-based summary."
        }
        out_path = out_dir / f"biomedical_audit_{disease_name.replace(' ','_')}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved fallback audit to %s", out_path)

    # print a short summary to stdout
    print("--- Prediction summary ---")
    print(preds_df.head(10).to_csv(index=False))
    print("Predictions saved to:", str(preds_path))
    print("Audit report saved to:", str(out_dir / f"biomedical_audit_{disease_name.replace(' ','_')}.json"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--disease', default=None, help="Disease name (default: auto-detected from project ID)")
    args = parser.parse_args()
    main(args.project, args.disease)
