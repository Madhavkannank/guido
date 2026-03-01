// Static data harvested from results TCGA xgboost summary and evaluation JSON files.
// Retrained 2026-03-01 with three fixes:
//   (1) leak-free feature selection (inside train split only, ANOVA F-test)
//   (2) class-imbalance correction via scale_pos_weight = n_neg/n_pos
//   (3) regularised XGBoost (max_depth=4, reg_lambda=2, adaptive min_child_weight)
// Test-AUROC numbers come from held-out 15% test split, never touched during training.
// LGG / PAAD / SKCM have 0 normal samples in TCGA → binary classifier not possible.
// GBM has only 5 normal samples → insufficient for a reliable stratified split (excluded).

export const cancerTypes = [
  'BRCA','COAD','GBM','KICH','KIRC','KIRP','LGG','LIHC','LUAD','LUSC','PAAD','PRAD','SKCM'
]

// XGBoost synthetic-label models  (results/TCGA-*/xgboost_summary.json)
export const syntheticAUROC = {
  BRCA: 0.495, COAD: 0.483, GBM: 0.214, KICH: 0.510,
  KIRC: 0.677, KIRP: 0.535, LGG: 0.481, LIHC: 0.535,
  LUAD: 0.419, LUSC: 0.503, PAAD: 0.472, PRAD: 0.472, SKCM: 0.460,
}

// XGBoost clinical-label models — HONEST test-split AUROCs after fixes
// (results/TCGA-*/xgboost_clinical_summary.json → test_auroc)
// GBM excluded: only 5 normal samples total — insufficient for a reliable split.
export const clinicalAUROC = {
  BRCA: 0.999, COAD: 1.000, KICH: 1.000,
  KIRC: 0.999, KIRP: 1.000, LIHC: 0.998, LUAD: 0.996,
  LUSC: 1.000, PRAD: 0.937,
}

// Sample counts
export const sampleCounts = {
  BRCA: { total: 1218, tumor: 1104, normal: 114 },
  COAD: { total:  329, tumor:  288, normal:  41 },
  GBM:  { total:  172, tumor:  167, normal:   5 },
  KICH: { total:   91, tumor:   66, normal:  25 },
  KIRC: { total:  606, tumor:  534, normal:  72 },
  KIRP: { total:  323, tumor:  291, normal:  32 },
  LGG:  { total:  530, tumor:  530, normal:   0 },
  LIHC: { total:  423, tumor:  373, normal:  50 },
  LUAD: { total:  576, tumor:  517, normal:  59 },
  LUSC: { total:  553, tumor:  502, normal:  51 },
  PAAD: { total:  183, tumor:  183, normal:   0 },
  PRAD: { total:  550, tumor:  498, normal:  52 },
  SKCM: { total:  474, tumor:  474, normal:   0 },
}

// Detailed evaluation metrics — test split, from evaluation_clinical.json
// Re-computed after applying all three training fixes.
export const clinicalMetrics = {
  BRCA: { accuracy:0.9945, precision:1.0000, recall:0.9939, f1:0.9970, auroc:0.9993, specificity:1.0000, sensitivity:0.9939 },
  COAD: { accuracy:0.9800, precision:0.9778, recall:1.0000, f1:0.9888, auroc:1.0000, specificity:0.8333, sensitivity:1.0000 },
  KICH: { accuracy:0.9286, precision:0.9091, recall:1.0000, f1:0.9524, auroc:1.0000, specificity:0.7500, sensitivity:1.0000 },
  KIRC: { accuracy:0.9890, precision:1.0000, recall:0.9875, f1:0.9937, auroc:0.9989, specificity:1.0000, sensitivity:0.9875 },
  KIRP: { accuracy:1.0000, precision:1.0000, recall:1.0000, f1:1.0000, auroc:1.0000, specificity:1.0000, sensitivity:1.0000 },
  LIHC: { accuracy:0.9688, precision:1.0000, recall:0.9643, f1:0.9818, auroc:0.9978, specificity:1.0000, sensitivity:0.9643 },
  LUAD: { accuracy:0.9885, precision:1.0000, recall:0.9872, f1:0.9935, auroc:0.9957, specificity:1.0000, sensitivity:0.9872 },
  LUSC: { accuracy:0.9880, precision:1.0000, recall:0.9867, f1:0.9933, auroc:1.0000, specificity:1.0000, sensitivity:0.9867 },
  PRAD: { accuracy:0.9518, precision:0.9863, recall:0.9600, f1:0.9730, auroc:0.9367, specificity:0.8750, sensitivity:0.9600 },
}

// -- Derived chart-ready arrays --

/** AUROC comparison: clinical vs synthetic per cancer */
export const aurocComparison = cancerTypes
  .filter(c => clinicalAUROC[c] !== undefined)
  .map(c => ({
    cancer: c,
    Clinical: +(clinicalAUROC[c]).toFixed(3),
    Synthetic: +(syntheticAUROC[c]).toFixed(3),
  }))

/** Sample distribution per cancer (tumor vs normal) */
export const sampleDistribution = cancerTypes.map(c => ({
  cancer: c,
  Tumor:  sampleCounts[c]?.tumor  ?? 0,
  Normal: sampleCounts[c]?.normal ?? 0,
}))

/** Total patients across all cohorts */
export const totalPatients       = Object.values(sampleCounts).reduce((s,v) => s + v.total, 0)
export const totalCancerTypes    = cancerTypes.length
export const totalClinicalModels = Object.keys(clinicalMetrics).length
export const avgClinicalAUROC = +(
  Object.values(clinicalAUROC).reduce((s,v) => s+v, 0) / Object.values(clinicalAUROC).length
).toFixed(3)
