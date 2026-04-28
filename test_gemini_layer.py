"""
Quick Gemini layer test using sample_positive_LUAD.csv
"""
import sys
import os
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import json
import pandas as pd

# ── Load sample CSV ───────────────────────────────────────────────────────────
sample_path = Path(__file__).resolve().parent / "ui" / "public" / "sample_positive_LUAD.csv"
df = pd.read_csv(sample_path)
sample = df.iloc[0]

# Pick top 10 features by value as mock biomarkers
top_features = sample.sort_values(ascending=False).head(10)
biomarkers = list(top_features.index)
shap_scores = {k: round(float(v), 4) for k, v in top_features.items()}
stability_scores = {k: 0.8 for k in biomarkers}

print("=" * 60)
print("GUIDO — Gemini Layer Test")
print("Sample: sample_positive_LUAD.csv")
print(f"Top biomarkers: {biomarkers[:5]} ...")
print("=" * 60)

# ── Build audit input ─────────────────────────────────────────────────────────
audit_input = {
    "disease_name": "Lung Adenocarcinoma",
    "project_id": "TCGA-LUAD",
    "training_sample_size": 403,
    "validation_sample_size": 87,
    "model_metrics": {"val_auroc": 1.0, "test_auroc": 1.0, "n_features": len(df.columns)},
    "multi_omics_features": {"transcriptomics_features": list(df.columns)},
    "selected_biomarkers": biomarkers,
    "shap_importance_scores": shap_scores,
    "stability_scores": stability_scores,
    "prediction_result": {"label": "positive", "prob": 0.92},
}

# ── Run through provider router ───────────────────────────────────────────────
from src.llm.provider_router import run_biomedical_audit

print("\nCalling provider router (Gemini primary)...")
result = run_biomedical_audit(audit_input)

# ── Print result ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"provider_used   : {result.get('provider_used')}")
print(f"fallback_reason : {result.get('fallback_reason')}")
print("=" * 60)

# Print key fields if present
for field in ["verdict", "confidence_score", "evidence_summary", "risk_explanation", "limitations", "cited_pubmed_ids"]:
    val = result.get(field)
    if val is not None:
        print(f"{field:20}: {val}")

# For GUIDO fallback format
for field in ["overall_system_verdict", "high_confidence_targets", "flagged_unstable_features"]:
    val = result.get(field)
    if val is not None:
        print(f"{field:20}: {val}")

print("\nFull result saved to: test_gemini_result.json")
out = Path(__file__).resolve().parent / "test_gemini_result.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print("Done.")
