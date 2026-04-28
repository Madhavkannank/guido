"""
Test GUIDO fallback (Groq) path end-to-end using sample_positive_LUAD.csv
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import json
import pandas as pd

sample_path = Path(__file__).resolve().parent / "ui" / "public" / "sample_positive_LUAD.csv"
df = pd.read_csv(sample_path)
sample = df.iloc[0]
top_features = sample.sort_values(ascending=False).head(10)
biomarkers = list(top_features.index)
shap_scores = {k: round(float(v), 4) for k, v in top_features.items()}
stability_scores = {k: 0.8 for k in biomarkers}

print("=" * 60)
print("GUIDO - Groq Fallback Test")
print("Top biomarkers: " + str(biomarkers[:5]) + " ...")
print("=" * 60)

audit_input = {
    "disease_name": "Lung Adenocarcinoma",
    "project_id": "TCGA-LUAD",
    "training_sample_size": 403,
    "validation_sample_size": 87,
    "model_metrics": {"val_auroc": 1.0, "test_auroc": 1.0, "n_features": len(df.columns)},
    "multi_omics_features": {"transcriptomics_features": list(df.columns)[:20]},
    "selected_biomarkers": biomarkers,
    "shap_importance_scores": shap_scores,
    "stability_scores": stability_scores,
    "prediction_result": {"label": "positive", "prob": 0.92},
}

# Force fallback by unsetting Gemini key
os.environ["GEMINI_API_KEY"] = ""

from src.llm.provider_router import run_biomedical_audit
print("\nCalling provider router (Gemini key unset -> GUIDO fallback)...")
result = run_biomedical_audit(audit_input)

print("\n" + "=" * 60)
print("provider_used   : " + str(result.get("provider_used")))
print("fallback_reason : " + str(result.get("fallback_reason")))
print("=" * 60)

for field in ["overall_system_verdict", "high_confidence_targets", "flagged_unstable_features"]:
    val = result.get(field)
    if val is not None:
        print(field + ": " + str(val)[:120])

out = Path(__file__).resolve().parent / "test_fallback_result.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nFull result saved to: " + out.name)
