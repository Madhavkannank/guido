from dotenv import load_dotenv
load_dotenv('.env')
import joblib, numpy as np
from pathlib import Path

artifact = joblib.load('models_artifacts/TCGA-LUAD/xgboost_clinical_model.joblib')
model = artifact['model']
scaler = artifact['scaler']
features = artifact['features']

print(f"Model type     : {type(model).__name__}")
print(f"N features     : {len(features)}")
print(f"Top 10 features: {features[:10]}")
print(f"XGB params     : max_depth={model.max_depth}, n_estimators={model.n_estimators}")
print(f"               : reg_lambda={model.reg_lambda}, min_child_weight={model.min_child_weight}")
print(f"               : scale_pos_weight={model.scale_pos_weight}")

import json
summary = json.load(open('results/TCGA-LUAD/xgboost_clinical_summary.json'))
print(f"\nTrain AUROC : {summary.get('train_auroc')}")
print(f"Val AUROC   : {summary.get('val_auroc')}")
print(f"Test AUROC  : {summary.get('test_auroc')}")
print(f"n_tumor     : {summary.get('n_tumor')}")
print(f"n_normal    : {summary.get('n_normal')}")
print(f"Imbalance   : {summary.get('n_tumor')/summary.get('n_normal'):.1f}:1")
