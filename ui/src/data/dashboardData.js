// Static data harvested from results TCGA xgboost summary and evaluation JSON files.
// These numbers come from the actual training runs.

export const cancerTypes = [
  'BRCA','COAD','GBM','KICH','KIRC','KIRP','LGG','LIHC','LUAD','LUSC','PAAD','PRAD','SKCM'
]

// XGBoost synthetic-label models  (results/TCGA-*/xgboost_summary.json)
export const syntheticAUROC = {
  BRCA: 0.495, COAD: 0.483, GBM: 0.214, KICH: 0.510,
  KIRC: 0.677, KIRP: 0.535, LGG: 0.481, LIHC: 0.535,
  LUAD: 0.419, LUSC: 0.503, PAAD: 0.472, PRAD: 0.472, SKCM: 0.460,
}

// XGBoost clinical-label models  (results/TCGA-*/xgboost_clinical_summary.json)
export const clinicalAUROC = {
  BRCA: 1.000, COAD: 1.000, GBM: 1.000, KICH: 1.000,
  KIRC: 1.000, KIRP: 1.000, LIHC: 0.993, LUAD: 1.000,
  LUSC: 1.000, PRAD: 0.972,
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

// Detailed evaluation metrics (clinical models — test split)
// Source: results/TCGA-*/evaluation_clinical.json → test_* fields
export const clinicalMetrics = {
  BRCA: { accuracy:1.000, precision:1.000, recall:1.000, f1:1.000, auroc:1.000, specificity:1.000, sensitivity:1.000 },
  KIRC: { accuracy:0.989, precision:0.988, recall:1.000, f1:0.994, auroc:1.000, specificity:0.909, sensitivity:1.000 },
  LUAD: { accuracy:1.000, precision:1.000, recall:1.000, f1:1.000, auroc:1.000, specificity:1.000, sensitivity:1.000 },
  LUSC: { accuracy:0.988, precision:0.987, recall:1.000, f1:0.993, auroc:1.000, specificity:0.875, sensitivity:1.000 },
  PRAD: { accuracy:0.940, precision:0.973, recall:0.960, f1:0.966, auroc:0.972, specificity:0.750, sensitivity:0.960 },
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
export const totalPatients = Object.values(sampleCounts).reduce((s,v) => s + v.total, 0)
export const totalCancerTypes = cancerTypes.length
export const avgClinicalAUROC = +(
  Object.values(clinicalAUROC).reduce((s,v) => s+v, 0) / Object.values(clinicalAUROC).length
).toFixed(3)
