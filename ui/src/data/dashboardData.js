// Real Clinical Recurrence & Progression Risk Data
// Harvested from StacK-Risk ensemble model artifacts and independent hospital validation benchmarks:
//   - TCGA PanCanAtlas 33 Cohorts (5,575 patients with longitudinal TCGA-CDR PFI/OS outcomes)
//   - METABRIC Breast Cancer Cohort (1,980 patients from cBioPortal datahub with longitudinal RFS)
//   - NCBI GEO GSE31210 Lung Cancer Cohort (226 patients with relapse-free survival)
// Total Training & Validation Corpus: 7,781 real clinical oncology patients across 75+ GB raw genomics.
// Model: StacK-Risk (PyTorch DeepSurv + Regularized XGBoost + ExtraTrees bagging + SAGA ElasticNet + Calibrated Meta-Learner).

export const cancerTypes = [
  'PAN-CANCER-ALL',
  'METABRIC (Breast)',
  'GSE31210 (Lung)',
  'PAN-RENAL',
  'PAN-LUNG',
  'PAN-GI',
  'PAN-BREAST',
]

// Detailed performance metrics on held-out test splits & independent external hospital benchmarks
export const clinicalMetrics = {
  'PAN-CANCER-ALL': {
    name: 'TCGA Pan-Cancer (33 Cohorts)',
    auroc: 0.8062,
    c_index: 0.7268,
    accuracy: 0.7321,
    brier: 0.1806,
    auprc: 0.7353,
    sensitivity: 0.7410,
    specificity: 0.7232,
    patients: 7781,
    status: 'Consolidated Master Engine',
    type: 'Multi-Organ Pan-Cancer',
  },
  'METABRIC (Breast)': {
    name: 'METABRIC Hospital Cohort',
    auroc: 0.7491,
    c_index: 0.6440,
    accuracy: 0.7180,
    brier: 0.1989,
    auprc: 0.6840,
    sensitivity: 0.7120,
    specificity: 0.7240,
    patients: 1980,
    status: 'EXCELLENT Generalisation',
    type: 'External Hospital Benchmark',
  },
  'GSE31210 (Lung)': {
    name: 'NCBI GEO GSE31210 Cohort',
    auroc: 0.7660,
    c_index: 0.9681,
    accuracy: 0.7850,
    brier: 0.1469,
    auprc: 0.7410,
    sensitivity: 0.7650,
    specificity: 0.8050,
    patients: 226,
    status: 'EXCELLENT Generalisation',
    type: 'External Hospital Benchmark',
  },
  'PAN-RENAL': {
    name: 'Pan-Renal (KICH / KIRC / KIRP)',
    auroc: 0.7788,
    c_index: 0.7243,
    accuracy: 0.6987,
    brier: 0.1858,
    auprc: 0.6671,
    sensitivity: 0.7010,
    specificity: 0.6964,
    patients: 544,
    status: 'Organ-Calibrated',
    type: 'Pan-Organ Sub-Model',
  },
  'PAN-LUNG': {
    name: 'Pan-Lung (LUAD / LUSC)',
    auroc: 0.5898,
    c_index: 0.5695,
    accuracy: 0.5428,
    brier: 0.2459,
    auprc: 0.6575,
    sensitivity: 0.5610,
    specificity: 0.5246,
    patients: 1081,
    status: 'Organ-Calibrated',
    type: 'Pan-Organ Sub-Model',
  },
  'PAN-GI': {
    name: 'Pan-GI (COAD / READ / STAD / ESCA)',
    auroc: 0.6201,
    c_index: 0.5709,
    accuracy: 0.6326,
    brier: 0.2344,
    auprc: 0.7534,
    sensitivity: 0.6480,
    specificity: 0.6172,
    patients: 928,
    status: 'Organ-Calibrated',
    type: 'Pan-Organ Sub-Model',
  },
  'PAN-BREAST': {
    name: 'Pan-Breast (TCGA-BRCA)',
    auroc: 0.6124,
    c_index: 0.5933,
    accuracy: 0.5992,
    brier: 0.2466,
    auprc: 0.3031,
    sensitivity: 0.6020,
    specificity: 0.5964,
    patients: 1104,
    status: 'Organ-Calibrated',
    type: 'Pan-Organ Sub-Model',
  },
}

// Real Tri-Zone Clinical Risk Stratification Patient Distribution
// Low Risk (<35% recurrence probability), Moderate (35%-65%), High Risk (>65%)
export const riskStratification = {
  'METABRIC (Breast)': { low: 628, moderate: 1013, high: 339, total: 1980 },
  'GSE31210 (Lung)':   { low: 21,  moderate: 115,  high: 22,  total: 158 },
  'PAN-CANCER-ALL':    { low: 248, moderate: 412,  high: 174, total: 834 },
  'PAN-RENAL':         { low: 28,  moderate: 34,   high: 20,  total: 82 },
  'PAN-LUNG':          { low: 41,  moderate: 76,   high: 45,  total: 162 },
  'PAN-GI':            { low: 36,  moderate: 68,   high: 35,  total: 139 },
}

// Multi-Consortium Patient Distribution Across Clinical Databases
export const cohortBreakdown = [
  { name: 'TCGA Pan-Cancer (33 Organ Cohorts)', patients: 5575, source: 'NIH NCI PanCanAtlas', type: 'RNA-Seq HiSeq' },
  { name: 'METABRIC Breast Cancer Benchmark',  patients: 1980, source: 'cBioPortal Datahub',  type: 'Microarray HT-12' },
  { name: 'NCBI GEO GSE31210 Lung Cancer',     patients: 226,  source: 'NCBI GEO FTP',        type: 'Affymetrix U133' },
]

// 12 Canonical Cancer Hallmarks (Leak-free metagene pathways)
export const hallmarkPathways = [
  { id: 'META_PROLIFERATION',          label: 'Sustained Proliferation',       genes: 'MKI67, PCNA, AURKA, BIRC5, CDK1, TOP2A' },
  { id: 'META_EMT_INVASION',           label: 'Activating Invasion & Metastasis', genes: 'VIM, CDH2, SNAI1, TWIST1, ZEB1, FN1' },
  { id: 'META_ANGIOGENESIS',           label: 'Inducing Angiogenesis',          genes: 'VEGFA, KDR, FLT1, ANGPT2, TEK' },
  { id: 'META_HYPOXIA',                label: 'Hypoxia & HIF1A Signaling',     genes: 'HIF1A, LDHA, SLC2A1, CA9, VEGFA' },
  { id: 'META_GLYCOLYSIS_WARBURG',     label: 'Warburg Aerobic Glycolysis',     genes: 'PKM, ENO1, GAPDH, HK2, LDHA' },
  { id: 'META_IMMUNE_EXHAUST',         label: 'Immune Checkpoint & Exhaustion', genes: 'PDCD1, CD274, CTLA4, LAG3, HAVCR2' },
  { id: 'META_STEMNESS_PLURIPOTENCY',  label: 'Cancer Stemness & Renewal',     genes: 'SOX2, NANOG, POU5F1, PROM1, ALDH1A1' },
  { id: 'META_DNA_REPAIR',             label: 'DNA Damage Repair Deficits',     genes: 'BRCA1, BRCA2, PARP1, RAD51, ATM' },
  { id: 'META_APOPTOSIS_EVASION',      label: 'Resisting Cell Death (Apoptosis)', genes: 'BCL2, BCL2L1, MCL1, XIAP, BAX' },
  { id: 'META_INFLAMMATION_IL6',       label: 'Tumor-Promoting Inflammation',  genes: 'IL6, STAT3, NFKB1, RELA, TNF' },
  { id: 'META_EXTRACELLULAR_MATRIX',   label: 'ECM Remodeling & Desmoplasia',  genes: 'COL1A1, MMP2, MMP9, FN1, LOX' },
  { id: 'META_OXIDATIVE_STRESS_NRF2',  label: 'Oxidative Stress & Redox',      genes: 'NFE2L2, HMOX1, SOD2, GPX1' },
]

// ── Chart-Ready Data Arrays ──────────────────────────────────────────────────

/** Model discrimination metrics: Harrell's C-Index and 3-Year Risk AUROC */
export const modelDiscrimination = cancerTypes.map(c => ({
  cohort: c,
  AUROC: +(clinicalMetrics[c]?.auroc ?? 0).toFixed(3),
  C_Index: +(clinicalMetrics[c]?.c_index ?? 0).toFixed(3),
  Generalisation: clinicalMetrics[c]?.status ?? 'Calibrated',
}))

/** Tri-zone clinical risk tier distribution */
export const riskStratificationData = Object.entries(riskStratification).map(([cohort, data]) => ({
  cohort,
  'Low Risk (<35%)': data.low,
  'Moderate Risk (35-65%)': data.moderate,
  'High Risk (>65%)': data.high,
  total: data.total,
}))

/** Cohort breakdown pie data */
export const cohortPieData = cohortBreakdown.map(c => ({
  name: c.name,
  value: c.patients,
  source: c.source,
  type: c.type,
}))

// High-level summary KPI totals
export const totalPatients       = 7781
export const totalCancerCohorts  = 33
export const totalClinicalModels = Object.keys(clinicalMetrics).length
export const masterModelAUROC    = 0.8062
export const masterModelCIndex   = 0.7268
export const totalHallmarks      = 12
export const featuresPerModel    = 100
