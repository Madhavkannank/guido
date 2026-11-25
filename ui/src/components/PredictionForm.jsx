import React, { useState } from 'react'
import axios from 'axios'

// Empty string = relative URLs (/api/…) — works in Docker where API and UI share the same origin.
// Vite's dev-server proxy handles /api/* → http://127.0.0.1:8000 during local development.
const API_BASE = import.meta.env.VITE_API_BASE || ''

const diseaseOptions = [
  // Master Pan-Cancer Clinical Model
  { value: 'pan_cancer_all', label: '★ TCGA-PAN-CANCER-ALL (Master 33-Cohort Clinical Model)' },
  // Pan-Organ Clinical Risk Models
  { value: 'pan_renal', label: '★ Pan-Renal Risk Model (KICH / KIRC / KIRP)' },
  { value: 'pan_lung', label: '★ Pan-Lung Risk Model (LUAD / LUSC)' },
  { value: 'pan_gi', label: '★ Pan-GI Colorectal/Gastric (COAD / READ / STAD / ESCA)' },
  { value: 'pan_hepatobiliary', label: '★ Pan-Hepatobiliary (LIHC / CHOL)' },
  { value: 'pan_breast', label: '★ Pan-Breast Oncology Model (BRCA)' },
  { value: 'pan_brain', label: '★ Pan-Brain Glioma / Glioblastoma (GBM / LGG)' },
  { value: 'pan_gyn', label: '★ Pan-Gynecologic (OV / UCEC / CESC / UCS)' },
  { value: 'pan_urological', label: '★ Pan-Urological (BLCA / PRAD)' },
  { value: 'pan_headneck', label: '★ Pan-Head & Neck (HNSC)' },
  { value: 'pan_endocrine', label: '★ Pan-Endocrine (THCA / ACC / PCPG)' },
  { value: 'pan_sarcoma', label: '★ Pan-Sarcoma (SARC)' },
  { value: 'pan_melanoma', label: '★ Pan-Melanoma (SKCM / UVM)' },
  { value: 'pan_hematologic', label: '★ Pan-Hematologic (LAML / DLBC / THYM)' },
  { value: 'pan_meso', label: '★ Pan-Mesothelioma (MESO)' },
  // All 33 TCGA Cohorts
  { value: 'adrenocortical_carcinoma', label: 'Adrenocortical Carcinoma (ACC)' },
  { value: 'bladder_urothelial_carcinoma', label: 'Bladder Urothelial Carcinoma (BLCA)' },
  { value: 'breast_invasive_carcinoma', label: 'Breast Invasive Carcinoma (BRCA)' },
  { value: 'cervical_squamous_cell_carcinoma', label: 'Cervical Squamous Cell Carcinoma (CESC)' },
  { value: 'cholangiocarcinoma', label: 'Cholangiocarcinoma (CHOL)' },
  { value: 'colon_adenocarcinoma', label: 'Colon Adenocarcinoma (COAD)' },
  { value: 'diffuse_large_b_cell_lymphoma', label: 'Diffuse Large B-Cell Lymphoma (DLBC)' },
  { value: 'esophageal_carcinoma', label: 'Esophageal Carcinoma (ESCA)' },
  { value: 'glioblastoma_multiforme', label: 'Glioblastoma Multiforme (GBM)' },
  { value: 'head_and_neck_squamous_cell_carcinoma', label: 'Head and Neck Squamous Cell Carcinoma (HNSC)' },
  { value: 'kidney_chromophobe', label: 'Kidney Chromophobe (KICH)' },
  { value: 'kidney_renal_clear_cell_carcinoma', label: 'Kidney Renal Clear Cell Carcinoma (KIRC)' },
  { value: 'kidney_renal_papillary_cell_carcinoma', label: 'Kidney Renal Papillary Cell Carcinoma (KIRP)' },
  { value: 'acute_myeloid_leukemia', label: 'Acute Myeloid Leukemia (LAML)' },
  { value: 'brain_lower_grade_glioma', label: 'Brain Lower Grade Glioma (LGG)' },
  { value: 'liver_hepatocellular_carcinoma', label: 'Liver Hepatocellular Carcinoma (LIHC)' },
  { value: 'lung_adenocarcinoma', label: 'Lung Adenocarcinoma (LUAD)' },
  { value: 'lung_squamous_cell_carcinoma', label: 'Lung Squamous Cell Carcinoma (LUSC)' },
  { value: 'mesothelioma', label: 'Mesothelioma (MESO)' },
  { value: 'ovarian_serous_cystadenocarcinoma', label: 'Ovarian Serous Cystadenocarcinoma (OV)' },
  { value: 'pancreatic_adenocarcinoma', label: 'Pancreatic Adenocarcinoma (PAAD)' },
  { value: 'pheochromocytoma_and_paraganglioma', label: 'Pheochromocytoma and Paraganglioma (PCPG)' },
  { value: 'prostate_adenocarcinoma', label: 'Prostate Adenocarcinoma (PRAD)' },
  { value: 'rectum_adenocarcinoma', label: 'Rectum Adenocarcinoma (READ)' },
  { value: 'sarcoma', label: 'Sarcoma (SARC)' },
  { value: 'skin_cutaneous_melanoma', label: 'Skin Cutaneous Melanoma (SKCM)' },
  { value: 'stomach_adenocarcinoma', label: 'Stomach Adenocarcinoma (STAD)' },
  { value: 'testicular_germ_cell_tumors', label: 'Testicular Germ Cell Tumors (TGCT)' },
  { value: 'thyroid_carcinoma', label: 'Thyroid Carcinoma (THCA)' },
  { value: 'thymoma', label: 'Thymoma (THYM)' },
  { value: 'uterine_corpus_endometrial_carcinoma', label: 'Uterine Corpus Endometrial Carcinoma (UCEC)' },
  { value: 'uterine_carcinosarcoma', label: 'Uterine Carcinosarcoma (UCS)' },
  { value: 'uveal_melanoma', label: 'Uveal Melanoma (UVM)' }
]

const SAMPLE_FILES = [
  {
    value: 'positive',
    label: 'Clinical Case — High Recurrence Risk Profile (LUAD)',
    url: '/sample_positive_LUAD.csv',
    filename: 'sample_positive_LUAD.csv',
    badge: 'HIGH RISK',
    badgeColor: 'text-[var(--crimson,#e06c75)]',
    note: 'Real clinical patient RNA-seq profile · High 3-Year Progression Risk',
  },
  {
    value: 'negative',
    label: 'Clinical Case — Low Recurrence Risk Profile (LUAD)',
    url: '/sample_negative_LUAD.csv',
    filename: 'sample_negative_LUAD.csv',
    badge: 'LOW RISK',
    badgeColor: 'text-[var(--green,#7A9A78)]',
    note: 'Real clinical patient RNA-seq profile · Favorable Progression-Free Prognosis',
  },
]

export default function PredictionForm({onResult}){
  const [csv, setCsv] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [projectId, setProjectId] = useState('TCGA-PAN-CANCER-ALL')
  const [disease, setDisease] = useState('pan_cancer_all')
  const [jsonPayload, setJsonPayload] = useState('')
  const [fileSource, setFileSource] = useState('own')   // 'own' | 'positive' | 'negative'

  const handleDiseaseChange = (e) => {
    const val = e.target.value;
    setDisease(val);
    const mapping = {
      'pan_cancer_all': 'TCGA-PAN-CANCER-ALL',
      // Pan-Organ
      'pan_renal': 'PAN-RENAL',
      'pan_lung': 'PAN-LUNG',
      'pan_gi': 'PAN-GI',
      'pan_hepatobiliary': 'PAN-HEPATOBILIARY',
      'pan_breast': 'PAN-BREAST',
      'pan_brain': 'PAN-BRAIN',
      'pan_gyn': 'PAN-GYN',
      'pan_urological': 'PAN-UROLOGICAL',
      'pan_headneck': 'PAN-HEADNECK',
      'pan_endocrine': 'PAN-ENDOCRINE',
      'pan_sarcoma': 'PAN-SARCOMA',
      'pan_melanoma': 'PAN-MELANOMA',
      'pan_hematologic': 'PAN-HEMATOLOGIC',
      'pan_meso': 'PAN-MESO',
      // TCGA 33 Cohorts
      'adrenocortical_carcinoma': 'TCGA-ACC',
      'bladder_urothelial_carcinoma': 'TCGA-BLCA',
      'breast_invasive_carcinoma': 'TCGA-BRCA',
      'cervical_squamous_cell_carcinoma': 'TCGA-CESC',
      'cholangiocarcinoma': 'TCGA-CHOL',
      'colon_adenocarcinoma': 'TCGA-COAD',
      'diffuse_large_b_cell_lymphoma': 'TCGA-DLBC',
      'esophageal_carcinoma': 'TCGA-ESCA',
      'glioblastoma_multiforme': 'TCGA-GBM',
      'head_and_neck_squamous_cell_carcinoma': 'TCGA-HNSC',
      'kidney_chromophobe': 'TCGA-KICH',
      'kidney_renal_clear_cell_carcinoma': 'TCGA-KIRC',
      'kidney_renal_papillary_cell_carcinoma': 'TCGA-KIRP',
      'acute_myeloid_leukemia': 'TCGA-LAML',
      'brain_lower_grade_glioma': 'TCGA-LGG',
      'liver_hepatocellular_carcinoma': 'TCGA-LIHC',
      'lung_adenocarcinoma': 'TCGA-LUAD',
      'lung_squamous_cell_carcinoma': 'TCGA-LUSC',
      'mesothelioma': 'TCGA-MESO',
      'ovarian_serous_cystadenocarcinoma': 'TCGA-OV',
      'pancreatic_adenocarcinoma': 'TCGA-PAAD',
      'pheochromocytoma_and_paraganglioma': 'TCGA-PCPG',
      'prostate_adenocarcinoma': 'TCGA-PRAD',
      'rectum_adenocarcinoma': 'TCGA-READ',
      'sarcoma': 'TCGA-SARC',
      'skin_cutaneous_melanoma': 'TCGA-SKCM',
      'stomach_adenocarcinoma': 'TCGA-STAD',
      'testicular_germ_cell_tumors': 'TCGA-TGCT',
      'thyroid_carcinoma': 'TCGA-THCA',
      'thymoma': 'TCGA-THYM',
      'uterine_corpus_endometrial_carcinoma': 'TCGA-UCEC',
      'uterine_carcinosarcoma': 'TCGA-UCS',
      'uveal_melanoma': 'TCGA-UVM'
    };
    if (mapping[val]) setProjectId(mapping[val]);
  };

  async function handleSubmit(e){
    e.preventDefault()
    setLoading(true); setError(null)
    try{
      const formData = new FormData()
      formData.append('project_id', projectId)
      formData.append('disease_name', disease)

      if (fileSource === 'own') {
        if (csv) formData.append('file', csv)
        if (!csv && jsonPayload) formData.append('json_payload', jsonPayload)
      } else {
        const sample = SAMPLE_FILES.find(s => s.value === fileSource)
        if (sample) {
          const resp = await fetch(sample.url)
          const blob = await resp.blob()
          const file = new File([blob], sample.filename, { type: 'text/csv' })
          formData.append('file', file)
        }
      }

      const resp = await axios.post(`${API_BASE}/api/ui/predict`, formData)
      onResult(resp.data)
    }catch(err){
      const detail = err?.response?.data?.detail || err?.response?.data?.message || err?.message || 'Unknown error'
      setError(`Prediction failed: ${detail}`)
    }finally{ setLoading(false) }
  }

  const selectedSample = SAMPLE_FILES.find(s => s.value === fileSource)

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="mb-6 p-4 rounded-lg bg-[rgba(90,106,83,0.05)] border border-[rgba(90,106,83,0.2)]">
        <h4 className="text-sm font-semibold text-[var(--green)] mb-3 uppercase tracking-wider">How to use your own clinical data</h4>
        <ul className="text-sm text-[var(--sub)] space-y-2 list-disc pl-4">
          <li><strong>1. Tissue Biopsy:</strong> Obtain a tumor tissue sample via biopsy or surgical resection.</li>
          <li><strong>2. RNA Sequencing:</strong> Send the sample to a clinical genomics lab (e.g., Illumina, Foundation Medicine) for bulk RNA-Sequencing.</li>
          <li><strong>3. Data Processing:</strong> Request raw gene expression counts and normalize using <code className="text-[var(--text)] bg-[rgba(255,255,255,0.05)] px-1 rounded">log2(count + 1)</code>.</li>
          <li><strong>4. Formatting:</strong> Format as a CSV where the header row contains gene names and values are normalized expression levels.</li>
        </ul>
      </div>

      <div className="grid grid-cols-1 gap-3">
        <div>
          <label className="text-sm subtle">Disease</label>
          <select className="w-full p-2 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text)]" value={disease} onChange={handleDiseaseChange}>
            {diseaseOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-sm subtle">File source</label>
          <select
            className="w-full p-2 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text)]"
            value={fileSource}
            onChange={e => { setFileSource(e.target.value); setCsv(null) }}
          >
            <option value="own">Upload my own file</option>
            <optgroup label="── Sample files ──">
              {SAMPLE_FILES.map(s => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </optgroup>
          </select>
        </div>

        {fileSource === 'own' ? (
          <>
            <div>
              <label className="text-sm subtle">Upload CSV (samples × features)</label>
              <div className="upload-zone mt-1">
                <input type="file" accept="text/csv" onChange={e => setCsv(e.target.files[0])} />
              </div>
            </div>
            <div>
              <label className="text-sm subtle">Or paste JSON sample (object or array)</label>
              <textarea rows={4} className="w-full p-2 mt-2 rounded" placeholder='{"GENE1":1.2,"GENE2":0.4}' value={jsonPayload} onChange={e=>setJsonPayload(e.target.value)} />
            </div>
          </>
        ) : (
          <div className="p-3 rounded-lg bg-[rgba(255,255,255,0.03)] border border-[var(--border)] flex items-start gap-3">
            <div className="mt-0.5 text-lg">🧬</div>
            <div>
              <p className="text-sm font-medium text-[var(--text)]">{selectedSample?.filename}</p>
              <p className="text-xs mt-0.5">
                Expected result: <span className={`font-semibold ${selectedSample?.badgeColor}`}>{selectedSample?.badge}</span>
              </p>
              <p className="text-[11px] text-[var(--sub)] mt-1">{selectedSample?.note}</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? 'Processing…' : 'Submit'}
        </button>
        {loading && <div className="subtle">This may take a moment…</div>}
      </div>
      {error && <div className="text-sm text-red-400">{error}</div>}
    </form>
  )
}
