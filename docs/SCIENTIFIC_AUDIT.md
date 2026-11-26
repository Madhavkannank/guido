# GUIDO Clinical Risk Engine: Scientific & Survival-Analysis Audit Report

**Date:** 2026-11-26  
**Auditor:** Scientific Validation & Clinical Translation Pipeline  
**Target:** GUIDO Training Pipeline (`retrain_fixed.py`), External Validation (`src/validation/cross_cohort.py`), Model Architecture (`src/models/`), and Serving Layer (`src/api/app.py`).

---

## Executive Summary

Before implementing the new scientific development objectives, a systematic audit of the existing codebase was conducted across 10 critical bioinformatic and survival-analysis failure modes.

The audit identified **4 critical methodology issues** that must be resolved to ensure absolute scientific validity:
1. **External Validation Contamination:** When `include_external=True` was used in `extract_unified_cohort`, METABRIC and GSE31210 were concatenated into the training split pool. For genuine external hospital validation, training must be strictly restricted to TCGA, leaving external cohorts 100% blind.
2. **Cross-Cohort Preprocessing Leakage:** In `retrain_fixed.py`, z-score batch standardization was executed per-cohort *prior* to `train_test_split`, leaking test-set distribution parameters.
3. **Silent Endpoint Conflation:** If Progression-Free Interval (PFI) was missing, the pipeline fell back to Overall Survival (OS) without recording or separating the endpoint. PFI and OS are biologically distinct endpoints and must remain strictly isolated.
4. **Terminology Misalignment (DeepSurv):** `PyTorchDeepSurvClassifier` was minimizing binary cross-entropy on 3-year landmark labels rather than Cox proportional hazards partial likelihood. It is an MLP classifier, not a continuous survival model.

---

## Detailed Findings Across the 10 Scientific Audit Dimensions

### 1. Patient-Level Leakage
* **Status:** Mitigated in deduplication, but risk in multi-sample patients.
* **Finding:** TCGA barcodes contain sample-type codes (`-01A` for primary, `-06A` for metastatic). If multiple samples from the same 12-character patient barcode (`TCGA-XX-XXXX`) exist, standard random train/test split could assign one sample to training and another to testing.
* **Remediation:** Enforce `patient_id = barcode[:12]` as the grouping key using `StratifiedGroupKFold` or patient-level deduplication (keeping only primary `-01` biopsies).

### 2. Cohort Contamination
* **Status:** Addressed.
* **Finding:** Intra-cohort variance was not isolated during pan-cancer evaluation, allowing inter-organ baseline differences to dominate model predictions.
* **Remediation:** Implement organ-stratified Harrell's and Uno's C-index across all 33 cohorts.

### 3. External Validation Contamination
* **Status:** CRITICAL DEFECT IDENTIFIED.
* **Finding:** In `retrain_fixed.py`, `include_external=True` appended METABRIC and GSE31210 to `all_X` and `all_y` before the 80/20 train/test split. Consequently, 80% of METABRIC was trained on when generating the 7,781 patient artifact, invalidating external test claims on that artifact.
* **Remediation:** Set `include_external=False` as the immutable default for all TCGA training runs. External validation datasets must reside in an isolated test harness and never be seen during model fitting or feature selection.

### 4. Survival Event Handling & Endpoint Distinction
* **Status:** DEFECT IDENTIFIED.
* **Finding:** Code used:
  ```python
  if "PFI" in sub_cdr.columns and sub_cdr["PFI"].notna().sum() > 0:
      event = sub_cdr["PFI"]
  else:
      event = sub_cdr["OS"]
  ```
  This merged two clinically distinct outcomes across cohorts.
* **Remediation:** Explicitly declare the endpoint (`PFI` vs `OS`) in dataset schemas and model metadata. Never silently fall back between endpoints.

### 5. Missing or Invalid Survival Times
* **Status:** DEFECT IDENTIFIED.
* **Finding:** `time_series.fillna(0)` assigned follow-up $t=0$ to missing entries. Patients with $t \le 0$ or missing follow-up provide zero progression information and distort survival metrics.
* **Remediation:** Filter out patients with $t \le 0$ or NaN survival time before computing C-index or fitting survival models.

### 6. Censoring Treatment & Landmark vs Continuous Models
* **Status:** CLARIFIED.
* **Finding:** For 3-year landmark recurrence classification, patients censored at $t < 1095$ days without an event cannot be categorized as Low Risk ($Y=0$) without IPCW correction.
* **Remediation:** In landmark classification, use IPCW weighting or drop indeterminate early-censored samples. In continuous survival modeling (C-index, Cox regression), utilize all patients with proper right-censoring rules.

### 7. Preprocessing Leakage
* **Status:** DEFECT IDENTIFIED.
* **Finding:** Cohort batch normalization (`(X - mean) / std`) was applied before splitting into train/test partitions.
* **Remediation:** Preprocessing scalers (StandardScaler, MinMax, Imputer, RankTransformer) must fit strictly on `X_train` and transform `X_test`.

### 8. Feature Selection Leakage
* **Status:** PARTIALLY LEAK-FREE.
* **Finding:** ANOVA `f_classif` was correctly fit on `X_tr`, and hallmark statistics were fit on `X_tr`. However, because batch normalization ran before splitting, a subtle leak was present.
* **Remediation:** Wrap all feature selection and preprocessing inside a strict `scikit-learn` Pipeline or explicit two-stage train/test harness.

### 9. Misleading Survival-Model Terminology
* **Status:** CLARIFIED.
* **Finding:** `PyTorchDeepSurvClassifier` was an MLP predicting binary landmark recurrence using binary cross entropy, rather than DeepSurv (Cox partial likelihood).
* **Remediation:** Accurately name the component `PyTorchRiskMLP` or implement true negative log partial likelihood loss for continuous survival time analysis.

### 10. Evaluation Metrics & Organ Confounding
* **Status:** CORE OBJECTIVE.
* **Finding:** Reporting only a pooled Pan-Cancer C-Index (0.7268) does not prove intra-cancer discrimination.
* **Remediation:** Implement organ-stratified C-index, Uno's C-index with IPCW, bootstrap confidence intervals, and stage/age-adjusted Cox proportional hazards models.
